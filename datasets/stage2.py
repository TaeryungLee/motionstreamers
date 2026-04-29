from __future__ import annotations

import json
import math
import random
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSED_ROOT = PROJECT_ROOT / "data" / "preprocessed"


def repo_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


class ArrayCache:
    def __init__(self, mmap: bool = True) -> None:
        self.mmap = bool(mmap)
        self._cache: dict[str, np.ndarray] = {}

    def load(self, path_str: str | None) -> np.ndarray | None:
        path = repo_path(path_str)
        if path is None:
            return None
        key = str(path)
        if key not in self._cache:
            self._cache[key] = np.load(path, mmap_mode="r" if self.mmap else None)
        return self._cache[key]


def rotvec_to_6d(rotvec: np.ndarray) -> np.ndarray:
    values = np.asarray(rotvec, dtype=np.float32)
    if values.size == 0:
        return np.zeros((*values.shape[:-1], 0), dtype=np.float32)
    if values.shape[-1] % 3 != 0:
        raise ValueError(f"rotvec last dim must be divisible by 3, got {values.shape}")
    prefix = values.shape[:-1]
    joints = values.shape[-1] // 3
    flat = values.reshape(-1, 3)
    mat = R.from_rotvec(flat).as_matrix().astype(np.float32)
    six = mat[:, :2, :].reshape(*prefix, joints * 6)
    return six.astype(np.float32)


def pose_dim(dataset: str) -> int:
    if dataset == "trumans":
        return 3 + 6 + 21 * 6 + 15 * 6 + 15 * 6
    if dataset == "lingo":
        return 3 + 6 + 21 * 6
    raise ValueError(f"unknown dataset: {dataset}")


def _safe_std(std: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(std, dtype=np.float32), 1e-6)


def _goal_xyz(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((3,), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 3:
        return np.zeros((3,), dtype=np.float32)
    return arr[:3].astype(np.float32)


def stable_hash_bucket(text: str, num_buckets: int = 4096) -> int:
    digest = hashlib.sha1(str(text).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % int(num_buckets)


def _sample_nearest_occupancy_crop(
    occ: np.ndarray,
    grid_meta: dict[str, Any],
    center_xyz: np.ndarray,
    nb_voxels: int = 32,
    xz_extent_m: float = 1.2,
    y_min_offset_m: float = 0.0,
    y_extent_m: float = 1.2,
) -> np.ndarray:
    """Return [1, nb, nb, nb] occupancy crop in world-axis local coordinates."""
    center = np.asarray(center_xyz, dtype=np.float32).reshape(3)
    n = int(nb_voxels)
    x_min = float(grid_meta["x_min"])
    x_max = float(grid_meta["x_max"])
    y_min = float(grid_meta["y_min"])
    y_max = float(grid_meta["y_max"])
    z_min = float(grid_meta["z_min"])
    z_max = float(grid_meta["z_max"])
    x_res = int(grid_meta["x_res"])
    y_res = int(grid_meta["y_res"])
    z_res = int(grid_meta["z_res"])

    xs = center[0] - xz_extent_m * 0.5 + (np.arange(n, dtype=np.float32) + 0.5) * (xz_extent_m / n)
    ys = y_min_offset_m + (np.arange(n, dtype=np.float32) + 0.5) * (y_extent_m / n)
    zs = center[2] - xz_extent_m * 0.5 + (np.arange(n, dtype=np.float32) + 0.5) * (xz_extent_m / n)

    xi = np.floor((xs - x_min) / max(x_max - x_min, 1e-6) * x_res).astype(np.int64)
    yi = np.floor((ys - y_min) / max(y_max - y_min, 1e-6) * y_res).astype(np.int64)
    zi = np.floor((zs - z_min) / max(z_max - z_min, 1e-6) * z_res).astype(np.int64)

    x_valid = (xi >= 0) & (xi < x_res)
    y_valid = (yi >= 0) & (yi < y_res)
    z_valid = (zi >= 0) & (zi < z_res)
    xi = np.clip(xi, 0, x_res - 1)
    yi = np.clip(yi, 0, y_res - 1)
    zi = np.clip(zi, 0, z_res - 1)

    crop = np.asarray(occ[np.ix_(xi, yi, zi)], dtype=np.float32)
    valid = x_valid[:, None, None] & y_valid[None, :, None] & z_valid[None, None, :]
    crop = np.where(valid, crop, 0.0).astype(np.float32)
    return crop[None]


class Stage2MotionDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        task: str,
        split: str = "train",
        root: Path | str = DEFAULT_PREPROCESSED_ROOT,
        stats_path: Path | str | None = None,
        normalize: bool = True,
        nb_voxels: int = 32,
        seed: int = 2026,
        max_records: int = 0,
        mmap: bool = True,
    ) -> None:
        if task not in {"move_wait", "action"}:
            raise ValueError(f"task must be move_wait or action, got {task}")
        self.dataset = str(dataset)
        self.task = str(task)
        self.split = str(split)
        self.root = Path(root)
        self.normalize = bool(normalize)
        self.nb_voxels = int(nb_voxels)
        self.seed = int(seed)
        self.array_cache = ArrayCache(mmap=mmap)
        self.stage2_root = self.root / self.dataset / "stage2"

        manifest_path = self.stage2_root / f"{self.task}_{self.split}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        payload = load_json(manifest_path)
        self.records: list[dict[str, Any]] = list(payload.get("records", []))
        if int(max_records) > 0:
            self.records = self.records[: int(max_records)]
        if not self.records:
            raise ValueError(f"empty Stage2 manifest: {manifest_path}")

        if stats_path is None:
            mean_file = self.stage2_root / "motion_mean.npy"
            std_file = self.stage2_root / "motion_std.npy"
            if mean_file.exists() and std_file.exists():
                self.motion_mean = np.asarray(np.load(mean_file), dtype=np.float32)
                self.motion_std = _safe_std(np.asarray(np.load(std_file), dtype=np.float32))
            else:
                stats_file = self.stage2_root / "normalization_stats_train_both.json"
                if not stats_file.exists():
                    raise FileNotFoundError(f"missing Stage2 normalization stats: {mean_file}, {std_file}, or {stats_file}")
                stats = load_json(stats_file)
                self.motion_mean = np.asarray(stats["mean"], dtype=np.float32)
                self.motion_std = _safe_std(np.asarray(stats["std"], dtype=np.float32))
        else:
            stats_file = Path(stats_path)
            if not stats_file.exists():
                raise FileNotFoundError(stats_file)
            stats = load_json(stats_file)
            self.motion_mean = np.asarray(stats["mean"], dtype=np.float32)
            self.motion_std = _safe_std(np.asarray(stats["std"], dtype=np.float32))
        expected_dim = pose_dim(self.dataset)
        if self.motion_mean.shape[0] != expected_dim:
            raise ValueError(f"normalization dim mismatch: got {self.motion_mean.shape[0]} expected {expected_dim}")

    @property
    def motion_dim(self) -> int:
        return int(self.motion_mean.shape[0])

    def __len__(self) -> int:
        return len(self.records)

    def _rng(self, index: int) -> random.Random:
        return random.Random((self.seed + 1000003 * int(index)) & 0xFFFFFFFF)

    def _read_motion_vector(
        self,
        record: dict[str, Any],
        frame_indices: np.ndarray,
        anchor_frame: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        smplx = record["smplx"]
        transl = self.array_cache.load(smplx.get("transl_path"))
        global_orient = self.array_cache.load(smplx.get("global_orient_path"))
        body_pose = self.array_cache.load(smplx.get("body_pose_path"))
        if transl is None or global_orient is None or body_pose is None:
            raise FileNotFoundError(f"missing SMPL-X arrays for {record.get('sequence_id')}")

        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        anchor = int(anchor_frame)
        root_world = np.asarray(transl[frame_indices], dtype=np.float32)
        anchor_root = np.asarray(transl[anchor], dtype=np.float32)
        root_local = root_world - anchor_root[None]

        parts = [
            root_local,
            rotvec_to_6d(np.asarray(global_orient[frame_indices], dtype=np.float32)),
            rotvec_to_6d(np.asarray(body_pose[frame_indices], dtype=np.float32)),
        ]
        if self.dataset == "trumans":
            left_hand = self.array_cache.load(smplx.get("left_hand_pose_path"))
            right_hand = self.array_cache.load(smplx.get("right_hand_pose_path"))
            if left_hand is None or right_hand is None:
                raise FileNotFoundError(f"missing TRUMANS hand arrays for {record.get('sequence_id')}")
            parts.append(rotvec_to_6d(np.asarray(left_hand[frame_indices], dtype=np.float32)))
            parts.append(rotvec_to_6d(np.asarray(right_hand[frame_indices], dtype=np.float32)))

        motion = np.concatenate(parts, axis=-1).astype(np.float32)
        if motion.shape[-1] != self.motion_dim:
            raise ValueError(f"motion dim mismatch: got {motion.shape[-1]} expected {self.motion_dim}")
        return motion, root_world.astype(np.float32), anchor_root.astype(np.float32)

    def _sample_move_wait(self, record: dict[str, Any], rng: random.Random) -> tuple[np.ndarray, int, int, int]:
        target_start = rng.randint(int(record["target_start_min"]), int(record["target_start_max"]))
        H = int(record["history_frames"])
        W = int(record["future_frames"])
        seq_start = int(record["sequence_global_start"])
        local = np.arange(target_start - H, target_start + W, dtype=np.int64)
        return seq_start + local, seq_start + target_start, H, W

    def _sample_action(self, record: dict[str, Any], rng: random.Random) -> tuple[np.ndarray, int, int, int]:
        H = int(record["history_frames"])
        seq_start = int(record["sequence_global_start"])
        if bool(record.get("first_segment_exception")):
            target_start = int(record["forced_target_start"])
        else:
            delta = rng.randint(int(record["aug_delta_min"]), int(record["aug_delta_max"]))
            target_start = int(record["action_start"]) - int(delta)
        action_end = int(record["action_end"])
        local = np.arange(target_start - H, action_end + 1, dtype=np.int64)
        return seq_start + local, seq_start + target_start, H, int(action_end - target_start + 1)

    def _scene_crops(self, record: dict[str, Any], anchor_root: np.ndarray, goal_world: np.ndarray, endpoint_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        occ_path = record.get("scene", {}).get("occupancy_grid_path")
        grid_meta = record.get("scene", {}).get("grid_meta")
        if occ_path is None or grid_meta is None:
            empty = np.zeros((1, self.nb_voxels, self.nb_voxels, self.nb_voxels), dtype=np.float32)
            return empty, empty
        occ = self.array_cache.load(occ_path)
        if occ is None:
            empty = np.zeros((1, self.nb_voxels, self.nb_voxels, self.nb_voxels), dtype=np.float32)
            return empty, empty
        current = _sample_nearest_occupancy_crop(occ, grid_meta, anchor_root, nb_voxels=self.nb_voxels)
        goal_center = goal_world if np.linalg.norm(goal_world) > 0.0 else endpoint_world
        goal = _sample_nearest_occupancy_crop(occ, grid_meta, goal_center, nb_voxels=self.nb_voxels)
        return current, goal

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        rng = self._rng(index)
        if self.task == "move_wait":
            frames, anchor, history_frames, target_frames = self._sample_move_wait(record, rng)
        else:
            frames, anchor, history_frames, target_frames = self._sample_action(record, rng)

        motion, root_world, anchor_root = self._read_motion_vector(record, frames, anchor)
        target_mask = np.zeros((motion.shape[0],), dtype=np.float32)
        target_mask[int(history_frames) :] = 1.0
        history_mask = 1.0 - target_mask
        action_time = np.zeros((motion.shape[0],), dtype=np.float32)
        if self.task == "action" and target_frames > 1:
            action_time[int(history_frames) :] = np.linspace(0.0, 1.0, int(target_frames), dtype=np.float32)

        endpoint_world = root_world[-1]
        body_goal_world = _goal_xyz(record.get("body_goal"))
        left_goal_world = _goal_xyz(record.get("left_hand_goal"))
        right_goal_world = _goal_xyz(record.get("right_hand_goal"))
        body_goal_valid = float(record.get("body_goal") is not None)
        left_goal_valid = float(record.get("left_hand_goal") is not None)
        right_goal_valid = float(record.get("right_hand_goal") is not None)

        scene_current, scene_goal = self._scene_crops(record, anchor_root, body_goal_world, endpoint_world)

        norm_motion = (motion - self.motion_mean[None]) / self.motion_std[None] if self.normalize else motion
        root_plan = motion[int(history_frames) :, :3].copy()
        if self.task == "move_wait":
            root_plan_mask = np.ones((root_plan.shape[0],), dtype=np.float32)
        else:
            root_plan = np.zeros((15, 3), dtype=np.float32)
            root_plan_mask = np.zeros((15,), dtype=np.float32)

        return {
            "motion": torch.from_numpy(norm_motion.astype(np.float32)),
            "motion_raw": torch.from_numpy(motion.astype(np.float32)),
            "target_mask": torch.from_numpy(target_mask),
            "history_mask": torch.from_numpy(history_mask.astype(np.float32)),
            "action_time": torch.from_numpy(action_time),
            "root_plan": torch.from_numpy(root_plan.astype(np.float32)),
            "root_plan_mask": torch.from_numpy(root_plan_mask.astype(np.float32)),
            "body_goal": torch.from_numpy((body_goal_world - anchor_root).astype(np.float32)),
            "left_hand_goal": torch.from_numpy((left_goal_world - anchor_root).astype(np.float32)),
            "right_hand_goal": torch.from_numpy((right_goal_world - anchor_root).astype(np.float32)),
            "goal_valid": torch.tensor([body_goal_valid, left_goal_valid, right_goal_valid], dtype=torch.float32),
            "scene_current": torch.from_numpy(scene_current.astype(np.float32)),
            "scene_goal": torch.from_numpy(scene_goal.astype(np.float32)),
            "length": torch.tensor(motion.shape[0], dtype=torch.long),
            "history_frames": torch.tensor(history_frames, dtype=torch.long),
            "target_frames": torch.tensor(target_frames, dtype=torch.long),
            "task_id": torch.tensor(0 if self.task == "move_wait" else 1, dtype=torch.long),
            "text_id": torch.tensor(stable_hash_bucket(str(record.get("text", ""))), dtype=torch.long),
            "goal_type_id": torch.tensor(stable_hash_bucket(str(record.get("goal_type", ""))), dtype=torch.long),
            "record_index": torch.tensor(index, dtype=torch.long),
            "scene_id": str(record.get("scene_id", "")),
            "sequence_id": str(record.get("sequence_id", "")),
            "segment_id": int(record.get("segment_id", -1)),
            "goal_type": str(record.get("goal_type", "")),
            "text": str(record.get("text", "")),
        }


def _pad_tensor(value: torch.Tensor, length: int) -> torch.Tensor:
    if value.shape[0] == length:
        return value
    pad_shape = (length - value.shape[0], *value.shape[1:])
    return torch.cat([value, value.new_zeros(pad_shape)], dim=0)


def stage2_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("empty Stage2 batch")
    max_len = max(int(item["motion"].shape[0]) for item in batch)
    out: dict[str, Any] = {}
    seq_keys = {"motion", "motion_raw", "target_mask", "history_mask", "action_time"}
    for key in seq_keys:
        out[key] = torch.stack([_pad_tensor(item[key], max_len) for item in batch], dim=0)
    valid = torch.zeros((len(batch), max_len), dtype=torch.float32)
    for idx, item in enumerate(batch):
        valid[idx, : int(item["motion"].shape[0])] = 1.0
    out["valid_mask"] = valid

    tensor_keys = [
        "root_plan",
        "root_plan_mask",
        "body_goal",
        "left_hand_goal",
        "right_hand_goal",
        "goal_valid",
        "scene_current",
        "scene_goal",
        "length",
        "history_frames",
        "target_frames",
        "task_id",
        "text_id",
        "goal_type_id",
        "record_index",
    ]
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    for key in ["scene_id", "sequence_id", "segment_id", "goal_type", "text"]:
        out[key] = [item[key] for item in batch]
    return out


def masked_motion_mse(pred: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = (target_mask * valid_mask).to(pred.dtype)[..., None]
    diff = (pred - target).square() * mask
    denom = mask.sum().clamp_min(1.0) * pred.shape[-1]
    return diff.sum() / denom
