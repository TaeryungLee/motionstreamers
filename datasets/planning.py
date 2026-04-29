from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSED_ROOT = PROJECT_ROOT / "data" / "preprocessed"
MODEL_MAP_SIZE = (60, 80)
LOCAL_X_RANGE = (-4.0, 4.0)
LOCAL_Z_RANGE = (-3.0, 3.0)
LOCAL_RESOLUTION = 0.1


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def world_to_grid(points_xy: np.ndarray, H: int, W: int, resolution: float, origin_xy: tuple[float, float]) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    points_xy = np.asarray(points_xy, dtype=np.float32)
    x0, y0 = float(origin_xy[0]), float(origin_xy[1])
    cols = np.floor((points_xy[:, 0] - x0) / resolution).astype(np.int64)
    rows = np.floor((points_xy[:, 1] - y0) / resolution).astype(np.int64)
    rows = (H - 1) - rows
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    return np.stack([rows[valid], cols[valid]], axis=-1)


def make_occupancy_map(
    points: np.ndarray,
    H: int,
    W: int,
    resolution: float,
    origin_xy: tuple[float, float] = (0.0, 0.0),
    radius: float = 0.2,
) -> np.ndarray:
    occ = np.zeros((H, W), dtype=np.float32)
    if points is None or len(points) == 0:
        return occ
    grid_points = world_to_grid(np.asarray(points, dtype=np.float32).reshape(-1, 2), H, W, resolution, origin_xy)
    rad_px = max(0, int(np.ceil(radius / resolution)))
    for row, col in grid_points:
        occ[max(0, row - rad_px) : min(H, row + rad_px + 1), max(0, col - rad_px) : min(W, col + rad_px + 1)] = 1.0
    return occ


def make_gaussian_map(
    points: np.ndarray,
    H: int,
    W: int,
    resolution: float,
    origin_xy: tuple[float, float],
    sigma: float = 0.2,
    truncate: float = 3.0,
) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.float32)
    if points is None or len(points) == 0:
        return out
    grid_points = world_to_grid(np.asarray(points, dtype=np.float32).reshape(-1, 2), H, W, resolution, origin_xy)
    sigma_px = max(1e-6, sigma / resolution)
    radius_px = max(1, int(np.ceil(truncate * sigma_px)))
    yy, xx = np.mgrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_px**2)).astype(np.float32)
    for row, col in grid_points:
        r0 = max(0, row - radius_px)
        r1 = min(H, row + radius_px + 1)
        c0 = max(0, col - radius_px)
        c1 = min(W, col + radius_px + 1)
        kr0 = r0 - (row - radius_px)
        kr1 = kernel.shape[0] - ((row + radius_px + 1) - r1)
        kc0 = c0 - (col - radius_px)
        kc1 = kernel.shape[1] - ((col + radius_px + 1) - c1)
        out[r0:r1, c0:c1] = np.maximum(out[r0:r1, c0:c1], kernel[kr0:kr1, kc0:kc1])
    return out


def goal_type_seq_to_phase_seq(goal_type_seq: np.ndarray, goal_type_vocab: dict[str, int]) -> np.ndarray:
    phase_seq = np.ones_like(goal_type_seq, dtype=np.int64)
    walk_idx = goal_type_vocab.get("walk")
    if walk_idx is not None:
        phase_seq[goal_type_seq == int(walk_idx)] = 0
    return phase_seq


def velocity_from_positions(pos: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(pos, dtype=np.float32)
    vel[:, 1:] = pos[:, 1:] - pos[:, :-1]
    return vel


def sample_scene_crop(
    scene_map: np.ndarray,
    center_xy: np.ndarray,
    origin_xy: tuple[float, float],
    resolution: float,
    out_size: tuple[int, int] = MODEL_MAP_SIZE,
) -> np.ndarray:
    """Sample a fixed metric ego-centered crop from scene channels."""
    H_out, W_out = out_size
    # Raw scene maps are stored as [C, x_res, z_res]. grid_sample expects
    # [C, H(z), W(x)], so transpose before sampling.
    scene_zx = np.transpose(scene_map, (0, 2, 1)).copy()
    C, H_raw, W_raw = scene_zx.shape
    x_values = float(center_xy[0]) + LOCAL_X_RANGE[0] + (np.arange(W_out, dtype=np.float32) + 0.5) * LOCAL_RESOLUTION
    z_values = float(center_xy[1]) + LOCAL_Z_RANGE[0] + (np.arange(H_out, dtype=np.float32) + 0.5) * LOCAL_RESOLUTION
    z_values = z_values[::-1].copy()
    xx, zz = np.meshgrid(x_values, z_values)
    col = (xx - float(origin_xy[0])) / float(resolution)
    row_from_bottom = (zz - float(origin_xy[1])) / float(resolution)
    row = (H_raw - 1) - row_from_bottom
    grid_x = 2.0 * col / max(W_raw - 1, 1) - 1.0
    grid_y = 2.0 * row / max(H_raw - 1, 1) - 1.0
    grid = torch.from_numpy(np.stack([grid_x, grid_y], axis=-1).astype(np.float32))[None]
    image = torch.from_numpy(scene_zx.astype(np.float32, copy=False))[None]
    crop = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    return crop.numpy().reshape(C, H_out, W_out).astype(np.float32)


class _BasePlanning(Dataset):
    dataset_name: str

    def __init__(
        self,
        root: Path | str = DEFAULT_PREPROCESSED_ROOT,
        split: str = "all",
        scene_id: Optional[str] = None,
        max_others: int = 3,
        include_distance_map: bool = True,
        model_ready: bool = True,
        sample_paths_file: Optional[Path | str] = None,
        model_past_frames: int = 30,
        model_future_frames: int = 72,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.scene_id = scene_id
        self.max_others = int(max_others)
        self.include_distance_map = include_distance_map
        self.model_ready = model_ready
        self.model_past_frames = int(model_past_frames)
        self.model_future_frames = int(model_future_frames)
        self.sample_root = self.root / self.dataset_name / "stage1_samples_v2"
        self.sample_paths_file = Path(sample_paths_file) if sample_paths_file is not None else None
        self.sample_paths = self._load_sample_paths_file() if self.sample_paths_file is not None else self._collect_sample_paths()
        self.scene_cache: dict[str, dict[str, Any]] = {}
        self.motion_cache: dict[Path, np.ndarray] = {}
        self.sequence_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self.goal_type_vocab = self._load_goal_type_vocab()

    def _goal_type_vocab_path(self) -> Path:
        return self.root / self.dataset_name / "scenes_v2" / "goal_type_vocab.json"

    def _load_goal_type_vocab(self) -> dict[str, int]:
        path = self._goal_type_vocab_path()
        if path.exists():
            return json.loads(path.read_text())
        return {"__background__": 0, "walk": 1}

    def _sample_index_path(self) -> Path:
        scene = self.scene_id or "__all__"
        return self.root / self.dataset_name / "stage1_sample_index_v2" / f"{self.split}_{scene}.json"

    def _load_sample_paths_file(self) -> list[str]:
        assert self.sample_paths_file is not None
        payload = json.loads(self.sample_paths_file.read_text())
        if payload.get("dataset") != self.dataset_name:
            raise ValueError(f"eval sample file dataset mismatch: {self.sample_paths_file}")
        sample_paths = payload.get("sample_paths")
        if not isinstance(sample_paths, list):
            raise ValueError(f"eval sample file must contain sample_paths list: {self.sample_paths_file}")
        return [str(path) for path in sample_paths]

    def _load_cached_sample_paths(self) -> list[str] | None:
        path = self._sample_index_path()
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        if payload.get("dataset_name") != self.dataset_name or payload.get("split") != self.split or payload.get("scene_id") != self.scene_id:
            return None
        paths = payload.get("sample_paths")
        return [str(item) for item in paths] if isinstance(paths, list) else None

    def _write_cached_sample_paths(self, paths: list[str]) -> None:
        path = self._sample_index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "dataset_name": self.dataset_name,
                    "split": self.split,
                    "scene_id": self.scene_id,
                    "num_samples": len(paths),
                    "sample_paths": paths,
                },
                separators=(",", ":"),
            )
        )

    def _split_scene_ids(self) -> list[str] | None:
        if self.split not in ("train", "test"):
            return None
        split_path = self.root / self.dataset_name / f"{self.split}.txt"
        if not split_path.exists():
            return None
        scene_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        if self.scene_id is not None:
            scene_ids = [scene_id for scene_id in scene_ids if scene_id == self.scene_id]
        return scene_ids

    def _collect_sample_paths(self) -> list[str]:
        if not self.sample_root.exists():
            raise FileNotFoundError(self.sample_root)
        cached = self._load_cached_sample_paths()
        if cached is not None:
            print(f"loaded cached {self.dataset_name} v2 sample index: split={self.split} samples={len(cached)}", file=sys.stderr, flush=True)
            return cached
        print(f"scanning {self.dataset_name} stage1 v2 samples: split={self.split} scene={self.scene_id or 'all'}", file=sys.stderr, flush=True)
        split_scene_ids = self._split_scene_ids()
        if split_scene_ids is not None:
            paths: list[str] = []
            for scene_id in tqdm(split_scene_ids, desc=f"{self.dataset_name} {self.split} scene glob", unit="scene", leave=False, file=sys.stderr):
                scene_dir = self.sample_root / scene_id
                if not scene_dir.exists():
                    continue
                paths.extend(path.relative_to(self.sample_root).as_posix() for path in scene_dir.glob("sample_*.json"))
            paths = sorted(paths)
        else:
            if self.scene_id is None:
                iterator = self.sample_root.glob("*/sample_*.json")
            else:
                iterator = (self.sample_root / self.scene_id).glob("sample_*.json")
            paths = sorted(path.relative_to(self.sample_root).as_posix() for path in tqdm(iterator, desc=f"{self.dataset_name} v2 glob", unit="sample", leave=False, file=sys.stderr))
        self._write_cached_sample_paths(paths)
        return paths

    def __len__(self) -> int:
        return len(self.sample_paths)

    def _load_scene_info(self, scene_id: str) -> dict[str, Any]:
        cached = self.scene_cache.get(scene_id)
        if cached is not None:
            return cached
        scene_json = self.root / self.dataset_name / "scenes_v2" / scene_id / "scene.json"
        scene_record = json.loads(scene_json.read_text())
        clearance = np.load(resolve_repo_path(scene_record["clearance_map_npy_path"])).astype(np.float32)
        obstacle_map = None
        if scene_record.get("occupancy_grid_path") is not None:
            occ = np.load(resolve_repo_path(scene_record["occupancy_grid_path"]), mmap_mode="r")
            obstacle_map = np.asarray(occ[:, 1:86, :].any(axis=1), dtype=np.float32)
        distance_map = None
        if self.include_distance_map and scene_record.get("distance_map_npy_path") is not None:
            distance_map = np.load(resolve_repo_path(scene_record["distance_map_npy_path"])).astype(np.float32)
        grid_meta = scene_record["grid_meta"]
        x_min = float(grid_meta["x_min"])
        z_min = float(grid_meta["z_min"])
        x_res = int(grid_meta["x_res"])
        resolution = float((float(grid_meta["x_max"]) - x_min) / x_res)
        cached = {
            "clearance": clearance,
            "distance_map": distance_map,
            "obstacle_map": obstacle_map,
            "origin_xy": (x_min, z_min),
            "resolution": resolution,
        }
        self.scene_cache[scene_id] = cached
        return cached

    def _load_transl(self, transl_path: str) -> np.ndarray:
        path = resolve_repo_path(transl_path)
        cached = self.motion_cache.get(path)
        if cached is None:
            cached = np.load(path)
            self.motion_cache[path] = cached
        return cached

    def _load_sequence_info(self, scene_id: str, sequence_id: str) -> dict[str, Any]:
        key = (scene_id, sequence_id)
        cached = self.sequence_cache.get(key)
        if cached is not None:
            return cached
        seq_json = self.root / self.dataset_name / "scenes_v2" / scene_id / "sequences" / f"{sequence_id}.json"
        cached = json.loads(seq_json.read_text())
        self.sequence_cache[key] = cached
        return cached

    def _goal_type_to_index(self, goal_type: str) -> int:
        return int(self.goal_type_vocab.get(str(goal_type), self.goal_type_vocab.get("__background__", 0)))

    def _make_frame_goal_type_seq(self, scene_id: str, sequence_id: str, window_start: int, window_end: int) -> np.ndarray:
        seq_info = self._load_sequence_info(scene_id, sequence_id)
        window_len = int(window_end) - int(window_start) + 1
        labels = np.zeros((window_len,), dtype=np.int64)
        for segment in seq_info.get("segment_list", []):
            overlap_start = max(int(window_start), int(segment["start"]))
            overlap_end = min(int(window_end), int(segment["end"]))
            if overlap_end < overlap_start:
                continue
            labels[overlap_start - int(window_start) : overlap_end - int(window_start) + 1] = self._goal_type_to_index(str(segment.get("goal_type") or "__background__"))
        return labels

    def _slice_root_xy(self, human_motion_ref: dict[str, Any], window_start: int, window_end: int) -> np.ndarray:
        transl = self._load_transl(human_motion_ref["smplx"]["transl_path"])
        global_start = int(human_motion_ref["start"]) + int(window_start)
        global_end = int(human_motion_ref["start"]) + int(window_end) + 1
        return np.asarray(transl[global_start:global_end, :], dtype=np.float32)[:, [0, 2]]

    def _load_sample_payload(self, index: int) -> dict[str, Any]:
        return json.loads((self.sample_root / self.sample_paths[index]).read_text())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        payload = self._load_sample_payload(index)
        scene_info = self._load_scene_info(payload["scene_id"])
        scene_channels = [scene_info["clearance"]]
        if self.include_distance_map and scene_info["distance_map"] is not None:
            scene_channels.append(scene_info["distance_map"])
        if scene_info["obstacle_map"] is not None:
            scene_channels.append(scene_info["obstacle_map"])
        scene_map = np.stack(scene_channels, axis=0).astype(np.float32)

        payload_len = int(payload["window"]["length"])
        past_len = self.model_past_frames
        future_len = self.model_future_frames
        total_len = past_len + future_len
        if payload_len != total_len:
            raise ValueError(
                f"model split past={past_len} future={future_len} requires window={total_len}, "
                f"but sample {payload.get('sample_id')} has window length {payload_len}"
            )
        slots = 1 + self.max_others

        ego_ref = payload["ego"]
        ego_abs = self._slice_root_xy(ego_ref["human_motion_ref"], ego_ref["window_start"], ego_ref["window_end"])
        if ego_abs.shape[0] != total_len:
            raise ValueError(f"sample {payload.get('sample_id')} has {ego_abs.shape[0]} frames, expected {total_len}")
        ego_current_abs = ego_abs[past_len - 1].astype(np.float32)

        abs_pos = np.zeros((slots, total_len, 2), dtype=np.float32)
        rel_pos = np.zeros((slots, total_len, 2), dtype=np.float32)
        entity_valid = np.zeros((slots,), dtype=np.bool_)
        abs_pos[0] = ego_abs
        rel_pos[0] = ego_abs - ego_current_abs[None]
        entity_valid[0] = True

        for idx_other, other_ref in enumerate(payload.get("others", [])[: self.max_others], start=1):
            other_abs = self._slice_root_xy(other_ref["human_motion_ref"], other_ref["window_start"], other_ref["window_end"])
            if other_abs.shape[0] != total_len:
                continue
            abs_pos[idx_other] = other_abs
            rel_pos[idx_other] = other_abs - ego_current_abs[None]
            entity_valid[idx_other] = True

        scene_crop = sample_scene_crop(scene_map, ego_current_abs, scene_info["origin_xy"], float(scene_info["resolution"]))
        local_origin = np.asarray([LOCAL_X_RANGE[0], LOCAL_Z_RANGE[0]], dtype=np.float32)
        goal_rel_xy = np.zeros((2,), dtype=np.float32)
        goal_valid = np.asarray(False)
        if ego_ref.get("body_goal") is not None:
            goal_xyz = np.asarray(ego_ref["body_goal"], dtype=np.float32)
            goal_rel_xy = goal_xyz[[0, 2]] - ego_current_abs
            goal_valid = np.asarray(True)
        goal_in_crop = np.asarray(
            bool(LOCAL_X_RANGE[0] <= goal_rel_xy[0] <= LOCAL_X_RANGE[1] and LOCAL_Z_RANGE[0] <= goal_rel_xy[1] <= LOCAL_Z_RANGE[1] and bool(goal_valid))
        )
        goal_map = make_gaussian_map(goal_rel_xy[None], MODEL_MAP_SIZE[0], MODEL_MAP_SIZE[1], LOCAL_RESOLUTION, tuple(local_origin), sigma=0.25)[None]

        goal_type_seq = self._make_frame_goal_type_seq(payload["scene_id"], ego_ref["sequence_id"], int(ego_ref["window_start"]), int(ego_ref["window_end"]))
        phase_seq = goal_type_seq_to_phase_seq(goal_type_seq, self.goal_type_vocab)

        past_rel = rel_pos[:, :past_len]
        future_rel = rel_pos[:, past_len:]
        past_abs = abs_pos[:, :past_len]
        future_abs = abs_pos[:, past_len:]
        past_vel = velocity_from_positions(past_abs)
        future_vel = velocity_from_positions(abs_pos[:, past_len - 1 :])[:, 1:]

        return {
            "scene_maps": torch.from_numpy(scene_crop).float(),
            "scene_crop": torch.from_numpy(scene_crop).float(),
            "goal": torch.from_numpy(goal_map).float(),
            "goal_map": torch.from_numpy(goal_map).float(),
            "goal_rel_xy": torch.from_numpy(goal_rel_xy).float(),
            "goal_valid": torch.tensor(bool(goal_valid), dtype=torch.bool),
            "goal_in_crop": torch.tensor(bool(goal_in_crop), dtype=torch.bool),
            "ego_current_xy": torch.from_numpy(ego_current_abs).float(),
            "map_origin": torch.from_numpy(local_origin).float(),
            "map_resolution": torch.tensor(LOCAL_RESOLUTION, dtype=torch.float32),
            "entity_valid": torch.from_numpy(entity_valid),
            "valid_slots": torch.from_numpy(entity_valid),
            "others_valid_mask": torch.from_numpy(entity_valid[1:]),
            "past_rel_pos": torch.from_numpy(past_rel).float(),
            "past_vel": torch.from_numpy(past_vel).float(),
            "future_rel_pos": torch.from_numpy(future_rel).float(),
            "future_vel": torch.from_numpy(future_vel).float(),
            "past_abs_pos": torch.from_numpy(past_abs).float(),
            "future_abs_pos": torch.from_numpy(future_abs).float(),
            "ego_past_root": torch.from_numpy(past_rel[0]).float(),
            "others_past_root": torch.from_numpy(past_rel[1:]).float(),
            "ego_future_root": torch.from_numpy(future_rel[0]).float(),
            "others_future_root": torch.from_numpy(future_rel[1:]).float(),
            "ego_goal_type_seq": torch.from_numpy(goal_type_seq).long(),
            "ego_phase_seq": torch.from_numpy(phase_seq).long(),
            "x0": torch.from_numpy(future_rel).float(),
            "target_x0": torch.from_numpy(future_rel).float(),
        }


class TrumansPlanning(_BasePlanning):
    dataset_name = "trumans"


class LingoPlanning(_BasePlanning):
    dataset_name = "lingo"


def planning_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(batch) == 0:
        raise ValueError("empty batch")
    out: dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out
