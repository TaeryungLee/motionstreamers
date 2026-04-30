from __future__ import annotations

import json
import random
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSED_ROOT = Path("data") / "preprocessed"
JOINTS28_COUNT = 28
JOINTS28_DIM = JOINTS28_COUNT * 3
ROOT_PLAN_FRAMES = 21


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


def pose_dim(dataset: str) -> int:
    if dataset not in {"trumans", "lingo"}:
        raise ValueError(f"unknown dataset: {dataset}")
    return JOINTS28_DIM


def _safe_std(std: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(std, dtype=np.float32), 1e-6)


def _goal_xyz(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 3:
        return None
    return arr[:3].astype(np.float32)


def stable_hash_bucket(text: str, num_buckets: int = 4096) -> int:
    digest = hashlib.sha1(str(text).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % int(num_buckets)


def yaw_to_local_rotation(global_orient: np.ndarray) -> tuple[float, np.ndarray]:
    euler = R.from_rotvec(np.asarray(global_orient, dtype=np.float32)).as_euler("zxy")
    yaw = float(euler[2])
    world_to_local = R.from_euler("zxy", np.array([0.0, 0.0, -yaw], dtype=np.float32)).as_matrix().astype(np.float32)
    return yaw, world_to_local


def canonicalize_points(points: np.ndarray, anchor_xyz: np.ndarray, world_to_local: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    offset = np.asarray([anchor_xyz[0], 0.0, anchor_xyz[2]], dtype=np.float32)
    return ((values - offset) @ world_to_local.T).astype(np.float32)


def decanonicalize_points(points: np.ndarray, anchor_xyz: np.ndarray, world_to_local: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    offset = np.asarray([anchor_xyz[0], 0.0, anchor_xyz[2]], dtype=np.float32)
    return (values @ world_to_local + offset).astype(np.float32)


def _voxel_indices(points: np.ndarray, grid_meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    mins = np.array([grid_meta["x_min"], grid_meta["y_min"], grid_meta["z_min"]], dtype=np.float32)
    maxs = np.array([grid_meta["x_max"], grid_meta["y_max"], grid_meta["z_max"]], dtype=np.float32)
    res = np.array([grid_meta["x_res"], grid_meta["y_res"], grid_meta["z_res"]], dtype=np.int64)
    vox = np.floor((pts - mins[None]) / np.maximum(maxs - mins, 1e-6)[None] * res[None]).astype(np.int64)
    valid = np.all((vox >= 0) & (vox < res[None]), axis=-1)
    vox = np.clip(vox, 0, res - 1)
    return vox, valid


def _sample_yaw_aligned_occupancy_crop(
    occ: np.ndarray,
    grid_meta: dict[str, Any],
    center_xyz: np.ndarray,
    world_to_local: np.ndarray,
    nb_voxels: int = 32,
    xz_extent_m: float = 1.2,
    y_min_m: float = 0.1,
    y_max_m: float = 1.2,
) -> np.ndarray:
    """Return a local-yaw-aligned occupancy crop as [D, H, W]."""
    n = int(nb_voxels)
    xs = -xz_extent_m * 0.5 + (np.arange(n, dtype=np.float32) + 0.5) * (xz_extent_m / n)
    ys = y_min_m + (np.arange(n, dtype=np.float32) + 0.5) * ((y_max_m - y_min_m) / n)
    zs = -xz_extent_m * 0.5 + (np.arange(n, dtype=np.float32) + 0.5) * (xz_extent_m / n)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    local = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    world = decanonicalize_points(local, np.asarray(center_xyz, dtype=np.float32), world_to_local)
    vox, valid = _voxel_indices(world, grid_meta)
    values = np.asarray(occ[vox[:, 0], vox[:, 1], vox[:, 2]], dtype=np.float32)
    values = np.where(valid, values, 0.0).astype(np.float32)
    return values.reshape(n, n, n)


def _scene_vit_input(current: np.ndarray, goal: np.ndarray) -> np.ndarray:
    current_2d = np.asarray(current, dtype=np.float32).transpose(1, 0, 2)
    goal_2d = np.asarray(goal, dtype=np.float32).transpose(1, 0, 2)
    return np.concatenate([current_2d, goal_2d], axis=0).astype(np.float32)


def _infer_coord_norm_meta(dataset_root: Path) -> dict[str, float]:
    scenes_root = dataset_root / "scenes_v2"
    x_half = 3.0
    z_half = 4.0
    y_min = 0.0
    y_max = 2.0
    if scenes_root.exists():
        for path in scenes_root.glob("*/scene.json"):
            try:
                payload = load_json(path)
                meta = payload.get("grid_meta") or {}
                x_half = max(x_half, abs(float(meta.get("x_min", -x_half))), abs(float(meta.get("x_max", x_half))))
                z_half = max(z_half, abs(float(meta.get("z_min", -z_half))), abs(float(meta.get("z_max", z_half))))
                y_min = min(y_min, float(meta.get("y_min", y_min)))
                y_max = max(y_max, float(meta.get("y_max", y_max)))
            except Exception:
                continue
    return {
        "x_scale": float(max(x_half, 1e-6)),
        "z_scale": float(max(z_half, 1e-6)),
        "y_center": float((y_min + y_max) * 0.5),
        "y_scale": float(max((y_max - y_min) * 0.5, 1e-6)),
    }


def _build_motion_mean_std(meta: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    mean_joint = np.array([0.0, float(meta["y_center"]), 0.0], dtype=np.float32)
    std_joint = np.array([float(meta["x_scale"]), float(meta["y_scale"]), float(meta["z_scale"])], dtype=np.float32)
    mean = np.tile(mean_joint, JOINTS28_COUNT).astype(np.float32)
    std = np.tile(std_joint, JOINTS28_COUNT).astype(np.float32)
    return mean, std


def normalize_xyz(values: np.ndarray, meta: dict[str, float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).copy()
    arr[..., 0] = arr[..., 0] / float(meta["x_scale"])
    arr[..., 1] = (arr[..., 1] - float(meta["y_center"])) / float(meta["y_scale"])
    arr[..., 2] = arr[..., 2] / float(meta["z_scale"])
    return arr.astype(np.float32)


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
        randomize_offsets: bool | None = None,
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
        self.randomize_offsets = self.split == "train" if randomize_offsets is None else bool(randomize_offsets)
        self.array_cache = ArrayCache(mmap=mmap)
        self.stage2_root = self.root / self.dataset / "stage2"
        self.dataset_root = self.root / self.dataset

        manifest_path = self.stage2_root / f"{self.task}_{self.split}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        payload = load_json(manifest_path)
        self.records: list[dict[str, Any]] = list(payload.get("records", []))
        if int(max_records) > 0:
            self.records = self.records[: int(max_records)]
        if not self.records:
            raise ValueError(f"empty Stage2 manifest: {manifest_path}")
        self.expanded_index = self._build_expanded_index()

        coord_meta_path = Path(stats_path) if stats_path is not None else self.stage2_root / "joint_coord_norm_meta.json"
        if coord_meta_path.exists():
            self.coord_norm_meta = {key: float(value) for key, value in load_json(coord_meta_path).items() if key in {"x_scale", "z_scale", "y_center", "y_scale"}}
        else:
            self.coord_norm_meta = _infer_coord_norm_meta(self.dataset_root)
        self.motion_mean, self.motion_std = _build_motion_mean_std(self.coord_norm_meta)
        self.motion_std = _safe_std(self.motion_std)
        expected_dim = pose_dim(self.dataset)
        if self.motion_mean.shape[0] != expected_dim:
            raise ValueError(f"normalization dim mismatch: got {self.motion_mean.shape[0]} expected {expected_dim}")
        self.joints28_path = self.dataset_root / "joints28" / "joints28.npy"
        if not self.joints28_path.exists():
            raise FileNotFoundError(f"missing joints28 cache: {self.joints28_path}")

    @property
    def motion_dim(self) -> int:
        return int(self.motion_mean.shape[0])

    def __len__(self) -> int:
        return len(self.expanded_index)

    def _build_expanded_index(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for record_idx, record in enumerate(self.records):
            if self.task == "move_wait":
                start_min = int(record["target_start_min"])
                start_max = int(record["target_start_max"])
                for target_start in range(start_min, start_max + 1):
                    out.append((record_idx, int(target_start)))
            else:
                if bool(record.get("first_segment_exception")):
                    forced = record.get("forced_target_start")
                    if forced is not None:
                        out.append((record_idx, int(forced)))
                    continue
                action_start = int(record["action_start"])
                delta_min = int(record["aug_delta_min"])
                delta_max = int(record["aug_delta_max"])
                for delta in range(delta_min, delta_max + 1):
                    out.append((record_idx, int(action_start - delta)))
        if not out:
            raise ValueError(f"empty expanded Stage2 index for {self.dataset} {self.task} {self.split}")
        return out

    def _read_motion_vector(
        self,
        record: dict[str, Any],
        frame_indices: np.ndarray,
        anchor_frame: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        smplx = record["smplx"]
        global_orient = self.array_cache.load(smplx.get("global_orient_path"))
        joints28 = self.array_cache.load(str(self.joints28_path))
        if global_orient is None or joints28 is None:
            raise FileNotFoundError(f"missing Stage2 joint arrays for {record.get('sequence_id')}")

        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        anchor = int(anchor_frame)
        joints_world = np.asarray(joints28[frame_indices], dtype=np.float32)
        anchor_root = np.asarray(joints28[anchor, 0], dtype=np.float32)
        anchor_yaw, world_to_local = yaw_to_local_rotation(np.asarray(global_orient[anchor], dtype=np.float32))
        joints_local = canonicalize_points(joints_world.reshape(-1, 3), anchor_root, world_to_local).reshape(joints_world.shape)
        motion = joints_local.reshape(joints_local.shape[0], -1).astype(np.float32)
        if motion.shape[-1] != self.motion_dim:
            raise ValueError(f"motion dim mismatch: got {motion.shape[-1]} expected {self.motion_dim}")
        return motion, joints_world[:, 0].astype(np.float32), anchor_root.astype(np.float32), anchor_yaw, world_to_local

    def _sample_move_wait(self, record: dict[str, Any], target_start: int) -> tuple[np.ndarray, int, int, int]:
        H = int(record["history_frames"])
        W = int(record["future_frames"])
        seq_start = int(record["sequence_global_start"])
        local = np.arange(target_start - H, target_start + W, dtype=np.int64)
        return seq_start + local, seq_start + target_start - 1, H, W

    def _sample_action(self, record: dict[str, Any], target_start: int) -> tuple[np.ndarray, int, int, int]:
        H = int(record["history_frames"])
        seq_start = int(record["sequence_global_start"])
        action_end = int(record["action_end"])
        local = np.arange(target_start - H, action_end + 1, dtype=np.int64)
        return seq_start + local, seq_start + target_start - 1, H, int(action_end - target_start + 1)

    def _scene_tokens(
        self,
        record: dict[str, Any],
        anchor_root: np.ndarray,
        world_to_local: np.ndarray,
        goal_world: np.ndarray | None,
        endpoint_world: np.ndarray,
    ) -> np.ndarray:
        occ_path = record.get("scene", {}).get("occupancy_grid_path")
        grid_meta = record.get("scene", {}).get("grid_meta")
        if occ_path is None or grid_meta is None:
            return np.zeros((self.nb_voxels * 2, self.nb_voxels, self.nb_voxels), dtype=np.float32)
        occ = self.array_cache.load(occ_path)
        if occ is None:
            return np.zeros((self.nb_voxels * 2, self.nb_voxels, self.nb_voxels), dtype=np.float32)
        current = _sample_yaw_aligned_occupancy_crop(occ, grid_meta, anchor_root, world_to_local, nb_voxels=self.nb_voxels)
        goal_center = goal_world if goal_world is not None else endpoint_world
        goal = _sample_yaw_aligned_occupancy_crop(occ, grid_meta, goal_center, world_to_local, nb_voxels=self.nb_voxels)
        return _scene_vit_input(current, goal)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, target_start = self.expanded_index[int(index)]
        record = self.records[int(record_index)]
        if self.task == "move_wait":
            frames, anchor, history_frames, target_frames = self._sample_move_wait(record, target_start)
        else:
            frames, anchor, history_frames, target_frames = self._sample_action(record, target_start)

        motion_raw, root_world, anchor_root, anchor_yaw, world_to_local = self._read_motion_vector(record, frames, anchor)
        target_mask = np.zeros((motion_raw.shape[0],), dtype=np.float32)
        target_mask[int(history_frames) :] = 1.0
        history_mask = 1.0 - target_mask
        action_time = np.zeros((motion_raw.shape[0],), dtype=np.float32)
        if self.task == "action" and target_frames > 1:
            action_time[int(history_frames) :] = np.linspace(0.0, 1.0, int(target_frames), dtype=np.float32)

        endpoint_world = root_world[-1]
        body_goal_world = _goal_xyz(record.get("body_goal"))
        hand_goal_world = _goal_xyz(record.get("hand_goal"))
        if hand_goal_world is None:
            hand_goal_world = _goal_xyz(record.get("left_hand_goal"))
        if hand_goal_world is None:
            hand_goal_world = _goal_xyz(record.get("right_hand_goal"))
        body_goal_valid = float(body_goal_world is not None)
        hand_goal_valid = float(hand_goal_world is not None)
        body_goal_local = (
            canonicalize_points(body_goal_world[None], anchor_root, world_to_local)[0]
            if body_goal_world is not None
            else np.zeros((3,), dtype=np.float32)
        )
        hand_goal_local = (
            canonicalize_points(hand_goal_world[None], anchor_root, world_to_local)[0]
            if hand_goal_world is not None
            else np.zeros((3,), dtype=np.float32)
        )
        scene_goal_world = endpoint_world if self.task == "move_wait" else body_goal_world
        scene_occ = self._scene_tokens(record, anchor_root, world_to_local, scene_goal_world, endpoint_world)

        norm_motion = (motion_raw - self.motion_mean[None]) / self.motion_std[None] if self.normalize else motion_raw
        root_plan = motion_raw[int(history_frames) :, :3].copy()
        if self.task == "move_wait":
            root_plan_mask = np.ones((root_plan.shape[0],), dtype=np.float32)
        else:
            root_plan = np.zeros((ROOT_PLAN_FRAMES, 3), dtype=np.float32)
            root_plan_mask = np.zeros((ROOT_PLAN_FRAMES,), dtype=np.float32)
        if root_plan.shape[0] < ROOT_PLAN_FRAMES:
            pad = np.zeros((ROOT_PLAN_FRAMES - root_plan.shape[0], 3), dtype=np.float32)
            root_plan = np.concatenate([root_plan, pad], axis=0)
            root_plan_mask = np.concatenate([root_plan_mask, np.zeros((pad.shape[0],), dtype=np.float32)], axis=0)
        elif root_plan.shape[0] > ROOT_PLAN_FRAMES:
            root_plan = root_plan[:ROOT_PLAN_FRAMES]
            root_plan_mask = root_plan_mask[:ROOT_PLAN_FRAMES]
        root_plan_cond = normalize_xyz(root_plan, self.coord_norm_meta)
        body_goal_cond = normalize_xyz(body_goal_local[None], self.coord_norm_meta)[0]
        hand_goal_cond = normalize_xyz(hand_goal_local[None], self.coord_norm_meta)[0]

        return {
            "motion": torch.from_numpy(norm_motion.astype(np.float32)),
            "motion_raw": torch.from_numpy(motion_raw.astype(np.float32)),
            "target_mask": torch.from_numpy(target_mask),
            "history_mask": torch.from_numpy(history_mask.astype(np.float32)),
            "action_time": torch.from_numpy(action_time),
            "root_plan": torch.from_numpy(root_plan.astype(np.float32)),
            "root_plan_cond": torch.from_numpy(root_plan_cond.astype(np.float32)),
            "root_plan_mask": torch.from_numpy(root_plan_mask.astype(np.float32)),
            "body_goal": torch.from_numpy(body_goal_local.astype(np.float32)),
            "hand_goal": torch.from_numpy(hand_goal_local.astype(np.float32)),
            "body_goal_cond": torch.from_numpy(body_goal_cond.astype(np.float32)),
            "hand_goal_cond": torch.from_numpy(hand_goal_cond.astype(np.float32)),
            "goal_valid": torch.tensor([body_goal_valid, hand_goal_valid], dtype=torch.float32),
            "scene_occ": torch.from_numpy(scene_occ.astype(np.float32)),
            "anchor_root": torch.from_numpy(anchor_root.astype(np.float32)),
            "anchor_yaw": torch.tensor(float(anchor_yaw), dtype=torch.float32),
            "world_to_local": torch.from_numpy(world_to_local.astype(np.float32)),
            "length": torch.tensor(motion_raw.shape[0], dtype=torch.long),
            "history_frames": torch.tensor(history_frames, dtype=torch.long),
            "target_frames": torch.tensor(target_frames, dtype=torch.long),
            "task_id": torch.tensor(0 if self.task == "move_wait" else 1, dtype=torch.long),
            "text_id": torch.tensor(stable_hash_bucket(str(record.get("text", ""))), dtype=torch.long),
            "goal_type_id": torch.tensor(stable_hash_bucket(str(record.get("goal_type", ""))), dtype=torch.long),
            "record_index": torch.tensor(record_index, dtype=torch.long),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "target_start": torch.tensor(target_start, dtype=torch.long),
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
        "root_plan_cond",
        "root_plan_mask",
        "body_goal",
        "hand_goal",
        "body_goal_cond",
        "hand_goal_cond",
        "goal_valid",
        "scene_occ",
        "anchor_root",
        "anchor_yaw",
        "world_to_local",
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
