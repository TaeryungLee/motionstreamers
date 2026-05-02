from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.stage2 import (
    JOINTS28_DIM,
    DEFAULT_PREPROCESSED_ROOT,
    _build_motion_mean_std,
    _infer_coord_norm_meta,
    _goal_xyz,
    _sample_yaw_aligned_occupancy_crop,
    _scene_vit_input,
    canonicalize_points,
    normalize_xyz,
    repo_path,
    stable_hash_bucket,
    yaw_to_local_rotation,
)


MOVE_GOAL_TYPES = {"walk", "move", "stand_still"}


@dataclass(frozen=True)
class HSIMethodSpec:
    name: str
    history_frames: int
    window_frames: int

    @property
    def target_frames(self) -> int:
        return int(self.window_frames - self.history_frames)


HSI_METHOD_SPECS: dict[str, HSIMethodSpec] = {
    "lingo": HSIMethodSpec(name="lingo", history_frames=2, window_frames=16),
    "trumans": HSIMethodSpec(name="trumans", history_frames=2, window_frames=32),
    "dyn_hsi": HSIMethodSpec(name="dyn_hsi", history_frames=2, window_frames=48),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def scene_ref(scene_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "occupancy_grid_path": scene_payload.get("occupancy_grid_path"),
        "grid_meta": scene_payload.get("grid_meta"),
    }


def smplx_paths(motion_ref: dict[str, Any]) -> dict[str, str | None]:
    smpl = dict(motion_ref.get("smplx") or {})
    return {
        "global_orient_path": smpl.get("global_orient_path"),
        "joints_path": motion_ref.get("path"),
    }


def goal_payload(segment: dict[str, Any]) -> dict[str, Any]:
    goal_pose = segment.get("goal_pose") if isinstance(segment.get("goal_pose"), dict) else {}
    active_hand = str(segment.get("active_hand") or "").strip().lower()
    hand = _goal_xyz(goal_pose.get("hand"))
    left = _goal_xyz(goal_pose.get("left_hand"))
    right = _goal_xyz(goal_pose.get("right_hand"))
    if hand is not None and left is None and right is None:
        if active_hand == "left":
            left = hand
        elif active_hand == "right":
            right = hand
        elif active_hand in {"both", "none", ""}:
            left = hand
            right = hand
    out: dict[str, Any] = {
        "body_goal": _goal_xyz(goal_pose.get("pelvis")),
        "hand_goal": hand,
        "left_hand_goal": left,
        "right_hand_goal": right,
        "active_hand": active_hand,
    }
    return {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in out.items()}


def sequence_segments(sequence_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        list(sequence_payload.get("segment_list") or sequence_payload.get("segments") or []),
        key=lambda item: (int(item.get("start", 0)), int(item.get("end", 0)), int(item.get("segment_id", 0))),
    )


class HSIUnifiedDataset(Dataset):
    """Model-specific HSI windows built from full move/action segments.

    This dataset is intentionally separate from Stage2MotionDataset.  It scans
    scenes_v2 directly and emits fixed-window autoregressive HSI samples using
    each comparison method's native prefix/window convention.
    """

    def __init__(
        self,
        method: str,
        dataset: str,
        split: str = "train",
        root: Path | str = DEFAULT_PREPROCESSED_ROOT,
        max_target_frames: int = 300,
        nb_voxels: int = 32,
        max_records: int = 0,
        mmap: bool = True,
    ) -> None:
        method_key = str(method).lower()
        if method_key not in HSI_METHOD_SPECS:
            raise ValueError(f"unknown HSI method: {method}")
        if dataset not in {"trumans", "lingo"}:
            raise ValueError(f"HSI unified dataset supports trumans/lingo, got {dataset}")
        self.spec = HSI_METHOD_SPECS[method_key]
        self.method = method_key
        self.dataset = str(dataset)
        self.split = str(split)
        self.root = Path(root)
        self.dataset_root = self.root / self.dataset
        self.scenes_root = self.dataset_root / "scenes_v2"
        self.nb_voxels = int(nb_voxels)
        self.max_target_frames = int(max_target_frames)
        self.joints28_path = self.dataset_root / "joints28" / "joints28.npy"
        if not self.joints28_path.exists():
            raise FileNotFoundError(self.joints28_path)
        self.joints28 = np.load(repo_path(self.joints28_path), mmap_mode="r" if mmap else None)

        self.coord_norm_meta = self._load_coord_norm_meta()
        self.motion_mean, self.motion_std = _build_motion_mean_std(self.coord_norm_meta)
        self.motion_std = np.maximum(self.motion_std.astype(np.float32), 1e-6)
        self.records = self._scan_records()
        if int(max_records) > 0:
            self.records = self.records[: int(max_records)]
        if not self.records:
            raise ValueError(f"empty HSI unified dataset: method={method} dataset={dataset} split={split}")

    @property
    def motion_dim(self) -> int:
        return JOINTS28_DIM

    @property
    def window_frames(self) -> int:
        return self.spec.window_frames

    @property
    def history_frames(self) -> int:
        return self.spec.history_frames

    def _load_coord_norm_meta(self) -> dict[str, float]:
        path = self.dataset_root / "stage2" / "joint_coord_norm_meta.json"
        if path.exists():
            data = load_json(path)
            return {key: float(value) for key, value in data.items() if key in {"x_scale", "z_scale", "y_center", "y_scale"}}
        return _infer_coord_norm_meta(self.dataset_root)

    def _split_scene_ids(self) -> list[str]:
        split_path = self.dataset_root / f"{self.split}.txt"
        if split_path.exists():
            return [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        return sorted(p.name for p in self.scenes_root.iterdir() if p.is_dir())

    def _scan_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        H = self.history_frames
        target_len = self.spec.target_frames
        for scene_id in self._split_scene_ids():
            scene_dir = self.scenes_root / scene_id
            scene_path = scene_dir / "scene.json"
            if not scene_path.exists():
                continue
            scene_payload = load_json(scene_path)
            for seq_path in sorted((scene_dir / "sequences").glob("*.json")):
                sequence = load_json(seq_path)
                motion_ref = sequence["human_motion_ref"]
                seq_global_start = int(motion_ref["start"])
                seq_global_end = int(motion_ref["end"])
                seq_local_end = int(seq_global_end - seq_global_start)
                for ordinal, segment in enumerate(sequence_segments(sequence)):
                    seg_start = int(segment["start"])
                    seg_end = int(segment["end"])
                    if seg_end <= seg_start:
                        continue
                    goal_type = str(segment.get("goal_type") or "")
                    kind = "move" if goal_type in MOVE_GOAL_TYPES else "action"
                    first_segment_exception = int(ordinal) == 0 or seg_start - H < 0
                    first_target_start = seg_start + H if first_segment_exception else seg_start
                    if first_target_start > seg_end:
                        continue
                    target_cap_end = min(seg_end, first_target_start + self.max_target_frames - 1)
                    last_target_start = target_cap_end
                    target_start = first_target_start
                    while target_start <= last_target_start:
                        target_end = min(target_cap_end, target_start + target_len - 1)
                        if target_end < target_start:
                            break
                        rec = {
                            "dataset": self.dataset,
                            "split": self.split,
                            "method": self.method,
                            "scene_id": scene_id,
                            "sequence_id": str(sequence["sequence_id"]),
                            "sequence_global_start": seq_global_start,
                            "sequence_global_end": seq_global_end,
                            "sequence_local_end": seq_local_end,
                            "segment_id": int(segment.get("segment_id", -1)),
                            "segment_ordinal": int(ordinal),
                            "segment_start": seg_start,
                            "segment_end": seg_end,
                            "target_start": int(target_start),
                            "target_end": int(target_end),
                            "first_segment_exception": bool(first_segment_exception),
                            "kind": kind,
                            "goal_type": goal_type,
                            "text": str(segment.get("text") or goal_type or kind),
                            "smplx": smplx_paths(motion_ref),
                            "scene": scene_ref(scene_payload),
                        }
                        rec.update(goal_payload(segment))
                        records.append(rec)
                        target_start += target_len
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _read_global_orient(self, record: dict[str, Any], frame_indices: np.ndarray) -> np.ndarray:
        path = record["smplx"].get("global_orient_path")
        if path is None:
            return np.zeros((len(frame_indices), 3), dtype=np.float32)
        arr = np.load(repo_path(path), mmap_mode="r")
        return np.asarray(arr[frame_indices], dtype=np.float32)

    def _frame_indices(self, record: dict[str, Any]) -> tuple[np.ndarray, int, int]:
        H = self.history_frames
        target_len = self.spec.target_frames
        seq_start = int(record["sequence_global_start"])
        target_start = int(record["target_start"])
        target_end = int(record["target_end"])
        if bool(record.get("first_segment_exception")):
            hist_local = np.arange(int(record["segment_start"]), int(record["segment_start"]) + H, dtype=np.int64)
        else:
            hist_local = np.arange(target_start - H, target_start, dtype=np.int64)
        target_local = np.arange(target_start, target_end + 1, dtype=np.int64)
        valid_target = int(len(target_local))
        if valid_target < target_len:
            pad = np.repeat(target_local[-1:], target_len - valid_target, axis=0)
            target_local = np.concatenate([target_local, pad], axis=0)
        local = np.concatenate([hist_local, target_local[:target_len]], axis=0)
        local = np.clip(local, 0, int(record["sequence_local_end"]))
        return seq_start + local, H, valid_target

    def _scene_occ(self, record: dict[str, Any], anchor_root: np.ndarray, world_to_local: np.ndarray, goal_world: np.ndarray) -> np.ndarray:
        occ_path = record.get("scene", {}).get("occupancy_grid_path")
        grid_meta = record.get("scene", {}).get("grid_meta")
        if occ_path is None or grid_meta is None:
            return np.zeros((self.nb_voxels * 2, self.nb_voxels, self.nb_voxels), dtype=np.float32)
        occ = np.load(repo_path(occ_path), mmap_mode="r")
        current = _sample_yaw_aligned_occupancy_crop(occ, grid_meta, anchor_root, world_to_local, nb_voxels=self.nb_voxels)
        goal = _sample_yaw_aligned_occupancy_crop(occ, grid_meta, goal_world, world_to_local, nb_voxels=self.nb_voxels)
        return _scene_vit_input(current, goal)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        frames, H, valid_target = self._frame_indices(record)
        joints_world = np.asarray(self.joints28[frames], dtype=np.float32)
        global_orient = self._read_global_orient(record, frames)
        anchor_root = joints_world[0, 0].astype(np.float32)
        anchor_yaw, world_to_local = yaw_to_local_rotation(global_orient[0])
        joints_local = canonicalize_points(joints_world.reshape(-1, 3), anchor_root, world_to_local).reshape(joints_world.shape)
        motion_raw = joints_local.reshape(self.window_frames, -1).astype(np.float32)
        motion = ((motion_raw - self.motion_mean[None]) / self.motion_std[None]).astype(np.float32)

        target_mask = np.zeros((self.window_frames,), dtype=np.float32)
        target_mask[H : H + valid_target] = 1.0
        valid_mask = np.zeros((self.window_frames,), dtype=np.float32)
        valid_mask[: H + valid_target] = 1.0
        history_mask = np.zeros((self.window_frames,), dtype=np.float32)
        history_mask[:H] = 1.0

        body_goal_world = _goal_xyz(record.get("body_goal"))
        if body_goal_world is None:
            body_goal_world = joints_world[H + valid_target - 1, 0].astype(np.float32)
        hand_goal_world = _goal_xyz(record.get("hand_goal"))
        if hand_goal_world is None:
            hand_goal_world = _goal_xyz(record.get("left_hand_goal"))
        if hand_goal_world is None:
            hand_goal_world = _goal_xyz(record.get("right_hand_goal"))
        hand_valid = float(hand_goal_world is not None)
        if hand_goal_world is None:
            hand_goal_world = np.zeros((3,), dtype=np.float32)
        body_goal_local = canonicalize_points(body_goal_world[None], anchor_root, world_to_local)[0]
        hand_goal_local = canonicalize_points(hand_goal_world[None], anchor_root, world_to_local)[0]
        scene_occ = self._scene_occ(record, anchor_root, world_to_local, body_goal_world)

        is_loco = record["kind"] == "move"
        need_pelvis_dir = bool(is_loco or record.get("body_goal") is not None)
        need_scene = True
        is_pick = bool(hand_valid > 0.5)
        pi_value = max(0, min(self.window_frames - 1, H + valid_target - 1))

        return {
            "motion": torch.from_numpy(motion),
            "motion_raw": torch.from_numpy(motion_raw),
            "target_mask": torch.from_numpy(target_mask),
            "history_mask": torch.from_numpy(history_mask),
            "valid_mask": torch.from_numpy(valid_mask),
            "scene_occ": torch.from_numpy(scene_occ.astype(np.float32)),
            "body_goal": torch.from_numpy(body_goal_local.astype(np.float32)),
            "hand_goal": torch.from_numpy(hand_goal_local.astype(np.float32)),
            "body_goal_cond": torch.from_numpy(normalize_xyz(body_goal_local[None], self.coord_norm_meta)[0].astype(np.float32)),
            "hand_goal_cond": torch.from_numpy(normalize_xyz(hand_goal_local[None], self.coord_norm_meta)[0].astype(np.float32)),
            "goal_valid": torch.tensor([1.0, hand_valid], dtype=torch.float32),
            "task_id": torch.tensor(0 if is_loco else 1, dtype=torch.long),
            "text_id": torch.tensor(stable_hash_bucket(record.get("text", ""), 4096), dtype=torch.long),
            "goal_type_id": torch.tensor(stable_hash_bucket(record.get("goal_type", ""), 512), dtype=torch.long),
            "need_scene": torch.tensor(bool(need_scene), dtype=torch.bool),
            "need_pelvis_dir": torch.tensor(bool(need_pelvis_dir), dtype=torch.bool),
            "is_pick": torch.tensor(bool(is_pick), dtype=torch.bool),
            "need_pi": torch.tensor(True, dtype=torch.bool),
            "is_loco": torch.tensor(bool(is_loco), dtype=torch.bool),
            "pi": torch.tensor(int(pi_value), dtype=torch.long),
            "anchor_root": torch.from_numpy(anchor_root.astype(np.float32)),
            "anchor_yaw": torch.tensor(float(anchor_yaw), dtype=torch.float32),
            "world_to_local": torch.from_numpy(world_to_local.astype(np.float32)),
            "length": torch.tensor(self.window_frames, dtype=torch.long),
            "history_frames": torch.tensor(H, dtype=torch.long),
            "target_frames": torch.tensor(valid_target, dtype=torch.long),
            "dataset_name": self.dataset,
            "scene_id": str(record.get("scene_id", "")),
            "sequence_id": str(record.get("sequence_id", "")),
            "segment_id": str(record.get("segment_id", "")),
            "goal_type": str(record.get("goal_type", "")),
            "text": str(record.get("text", "")),
        }


def hsi_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [key for key, value in batch[0].items() if isinstance(value, torch.Tensor)]
    out: dict[str, Any] = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    for key in ["dataset_name", "scene_id", "sequence_id", "segment_id", "goal_type", "text"]:
        out[key] = [str(item.get(key, "")) for item in batch]
    return out
