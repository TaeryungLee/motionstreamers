from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F


Mode = Literal["MOVE", "ACT", "WAIT"]


@dataclass
class PlannerState:
    mode: Mode = "MOVE"
    wait_point_xy: Optional[tuple[float, float]] = None
    interrupted: bool = False
    elapsed_wait_steps: int = 0


@dataclass
class ActiveAction:
    label: str
    goal_area_id: str
    goal_type_id: int
    interruptible: bool = True
    goal_tolerance: float = 0.35
    blocked_goal_policy: str = "WAIT_NEAR"


@dataclass
class PlanFields:
    ego_prior_60: np.ndarray
    others_prior_60: np.ndarray
    static_sdf: np.ndarray
    goal_map: np.ndarray
    ref_path_xy: np.ndarray
    goal_center_xy: tuple[float, float]
    map_origin_xy: tuple[float, float]
    map_resolution: float


@dataclass
class FinePlan:
    pos_xy: np.ndarray
    vel_xy: np.ndarray
    yaw: np.ndarray
    success: bool
    debug: dict[str, Any]


@dataclass
class Stage1PlanOutput:
    fine_root_xy: np.ndarray
    fine_root_vel_xy: np.ndarray
    fine_root_yaw: np.ndarray
    mode: Mode
    wait_point_xy: Optional[tuple[float, float]]
    interrupted: bool


@dataclass
class OptimizerConfig:
    horizon: int = 60
    dt: float = 1.0 / 30.0
    steps: int = 80
    lr: float = 0.08
    prior_weight: float = 1.0
    goal_weight: float = 2.0
    dynamic_weight: float = 2.0
    static_weight: float = 4.0
    smooth_weight: float = 0.2
    speed_weight: float = 0.05
    static_margin_m: float = 0.25
    speed_profile: Optional[dict[Mode, dict[str, float]]] = None


@dataclass
class Stage1OptimizerV2Config:
    horizon: int = 30
    dt: float = 1.0 / 30.0
    iters: int = 100
    lr: float = 0.01
    grad_clip: float = 1.0
    w_prior: float = 2.0
    w_goal: float = 3.0
    w_static: float = 10.0
    w_dyn: float = 10.0
    smooth_scale: float = 1.0
    static_margin: float = 0.25
    static_center: float = 0.20
    tau_static: float = 0.015
    dyn_margin: float = 0.50
    tau_dyn: float = 0.05
    goal_progress_weight: float = 1.0
    static_topk_weight: float = 3.0
    static_topk_frac: float = 0.2
    dijkstra_alpha: float = 10.0
    corridor_radius: float = 0.30
    start_open_radius: float = 0.25
    goal_open_radius: float = 0.30
    corridor_strength: float = 1.0
    terminal_weight_ratio: float = 2.0
    init_mode: str = "blend"
    goal_threshold: float = 0.2
    goal_unsafe_ratio_threshold: float = 0.25
    speed_bound_mps: float = 1.0
    acc_bound_mps2: float = 0.0
    start_vel_weight: float = 3.0


SPEED_PROFILE: dict[Mode, dict[str, float]] = {
    "MOVE": {"v_pref": 0.9, "v_max": 1.4},
    "ACT": {"v_pref": 0.0, "v_max": 0.15},
    "WAIT": {"v_pref": 0.0, "v_max": 0.05},
}


def load_speed_profile(path: str | Path) -> dict[Mode, dict[str, float]]:
    data = json.loads(Path(path).read_text())
    profile = data.get("planner_profile", data)
    out: dict[Mode, dict[str, float]] = {}
    for mode in ("MOVE", "ACT", "WAIT"):
        values = profile.get(mode, SPEED_PROFILE[mode])
        out[mode] = {
            "v_pref": float(values.get("v_pref", SPEED_PROFILE[mode]["v_pref"])),
            "v_max": float(values.get("v_max", SPEED_PROFILE[mode]["v_max"])),
        }
    return out


def _first_float(*values: Any) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def configure_stage1_motion_bounds(
    config: Stage1OptimizerV2Config,
    values: Mapping[str, Any],
    payload: Optional[Mapping[str, Any]] = None,
) -> Stage1OptimizerV2Config:
    stats = {}
    if payload is not None and isinstance(payload.get("speed_stats"), Mapping):
        stats = dict(payload["speed_stats"])

    speed_bound = _first_float(
        values.get("speed_mean_mps"),
        values.get("move_speed_mean"),
        stats.get("move_speed_mean"),
        values.get("speed_bound_mps"),
        stats.get("move_speed_p95"),
        config.speed_bound_mps,
    )
    acc_bound = _first_float(
        values.get("acc_bound_mps2"),
        values.get("acc_mean_mps2"),
        values.get("move_acc_mean"),
        stats.get("move_acc_mean"),
        stats.get("move_acc_p95"),
        config.acc_bound_mps2,
    )
    config.speed_bound_mps = float(speed_bound if speed_bound is not None else config.speed_bound_mps)
    config.acc_bound_mps2 = float(acc_bound if acc_bound is not None else config.acc_bound_mps2)
    return config


def as_numpy(x: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def interp_time(values: np.ndarray, out_len: int = 60) -> np.ndarray:
    values = as_numpy(values)
    if values.shape[-3] == out_len:
        return values.copy()
    flat_shape = values.shape[:-3]
    T, H, W = values.shape[-3:]
    x = torch.from_numpy(values.reshape(-1, 1, T, H, W))
    x = F.interpolate(x, size=(out_len, H, W), mode="trilinear", align_corners=False)
    return x[:, 0].numpy().reshape(*flat_shape, out_len, H, W).astype(np.float32)


def grid_to_world(row: np.ndarray, col: np.ndarray, origin_xy: tuple[float, float], resolution: float, height: int) -> np.ndarray:
    x = origin_xy[0] + (col.astype(np.float32) + 0.5) * float(resolution)
    y_from_bottom = (height - 1 - row.astype(np.float32)) + 0.5
    y = origin_xy[1] + y_from_bottom * float(resolution)
    return np.stack([x, y], axis=-1)


def world_to_grid_torch(
    pos_xy: torch.Tensor,
    origin_xy: tuple[float, float],
    resolution: float,
    height: int,
    width: int,
) -> torch.Tensor:
    x = (pos_xy[:, 0] - float(origin_xy[0])) / float(resolution) - 0.5
    y_from_bottom = (pos_xy[:, 1] - float(origin_xy[1])) / float(resolution) - 0.5
    row = (height - 1) - y_from_bottom
    col = x
    grid_x = (col / max(width - 1, 1)) * 2.0 - 1.0
    grid_y = (row / max(height - 1, 1)) * 2.0 - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


def sample_batched_static_maps(
    images: torch.Tensor,
    pos_xy: torch.Tensor,
    origin_xy: tuple[float, float],
    resolution: float,
) -> torch.Tensor:
    B, H, W = images.shape
    x = (pos_xy[..., 0] - float(origin_xy[0])) / float(resolution) - 0.5
    y_from_bottom = (pos_xy[..., 1] - float(origin_xy[1])) / float(resolution) - 0.5
    row = (H - 1) - y_from_bottom
    col = x
    grid_x = (col / max(W - 1, 1)) * 2.0 - 1.0
    grid_y = (row / max(H - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(B, -1, 1, 2)
    values = F.grid_sample(images[:, None], grid, mode="bilinear", padding_mode="border", align_corners=True)
    return values[:, 0, :, 0]


def sample_map_sequence(
    maps: torch.Tensor,
    pos_xy: torch.Tensor,
    origin_xy: tuple[float, float],
    resolution: float,
) -> torch.Tensor:
    T, H, W = maps.shape
    grid = world_to_grid_torch(pos_xy, origin_xy, resolution, H, W).view(T, 1, 1, 2)
    values = F.grid_sample(maps[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return values.view(T)


def sample_static_map(
    image: torch.Tensor,
    pos_xy: torch.Tensor,
    origin_xy: tuple[float, float],
    resolution: float,
) -> torch.Tensor:
    H, W = image.shape
    grid = world_to_grid_torch(pos_xy, origin_xy, resolution, H, W).view(1, -1, 1, 2)
    values = F.grid_sample(image.view(1, 1, H, W), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return values.view(-1)


def build_static_sdf(scene_maps: np.ndarray, resolution: float, walkable_threshold: float = 0.5) -> np.ndarray:
    scene_maps = as_numpy(scene_maps)
    clearance = scene_maps[0] if scene_maps.ndim == 3 else scene_maps
    walkable = clearance > walkable_threshold
    try:
        from scipy.ndimage import distance_transform_edt

        outside = distance_transform_edt(walkable) * float(resolution)
        inside = distance_transform_edt(~walkable) * float(resolution)
        return (outside - inside).astype(np.float32)
    except Exception:
        return np.where(walkable, 1.0, -1.0).astype(np.float32) * float(resolution)


def compute_goal_centroid(goal_map: np.ndarray, origin_xy: tuple[float, float], resolution: float) -> tuple[float, float]:
    goal_map = as_numpy(goal_map)
    if goal_map.ndim == 3:
        goal_map = goal_map[0]
    H, _ = goal_map.shape
    weights = np.maximum(goal_map, 0.0)
    if float(weights.sum()) <= 1e-8:
        return float(origin_xy[0]), float(origin_xy[1])
    rows, cols = np.indices(goal_map.shape)
    points = grid_to_world(rows.reshape(-1), cols.reshape(-1), origin_xy, resolution, H)
    w = weights.reshape(-1, 1)
    center = (points * w).sum(axis=0) / max(float(w.sum()), 1e-8)
    return float(center[0]), float(center[1])


def extract_ref_path_simple(
    ego_prior_60: np.ndarray,
    origin_xy: tuple[float, float],
    resolution: float,
    smooth_window: int = 5,
) -> np.ndarray:
    ego_prior_60 = as_numpy(ego_prior_60)
    T, H, W = ego_prior_60.shape
    flat_idx = ego_prior_60.reshape(T, -1).argmax(axis=1)
    rows = flat_idx // W
    cols = flat_idx % W
    path = grid_to_world(rows, cols, origin_xy, resolution, H)
    if smooth_window > 1:
        pad = smooth_window // 2
        padded = np.pad(path, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones((smooth_window,), dtype=np.float32) / float(smooth_window)
        path = np.stack([np.convolve(padded[:, dim], kernel, mode="valid") for dim in range(2)], axis=-1)
    return path.astype(np.float32)


def interp_path_time(path: np.ndarray, out_len: int = 60) -> np.ndarray:
    path = as_numpy(path)
    if path.shape[-2] == out_len:
        return path.astype(np.float32, copy=True)
    src = np.linspace(0.0, 1.0, num=path.shape[-2], dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=out_len, dtype=np.float32)
    flat = path.reshape(-1, path.shape[-2], 2)
    out = np.zeros((flat.shape[0], out_len, 2), dtype=np.float32)
    for idx in range(flat.shape[0]):
        out[idx, :, 0] = np.interp(dst, src, flat[idx, :, 0])
        out[idx, :, 1] = np.interp(dst, src, flat[idx, :, 1])
    return out.reshape(*path.shape[:-2], out_len, 2).astype(np.float32)


def trajectory_to_occ(
    path_xy: np.ndarray,
    height: int,
    width: int,
    origin_xy: tuple[float, float],
    resolution: float,
    sigma: float = 0.18,
) -> np.ndarray:
    path_xy = as_numpy(path_xy)
    squeeze = path_xy.ndim == 2
    if squeeze:
        path_xy = path_xy[None]
    K, T, _ = path_xy.shape
    rows, cols = np.indices((height, width), dtype=np.float32)
    x = float(origin_xy[0]) + (cols + 0.5) * float(resolution)
    y = float(origin_xy[1]) + ((height - 1 - rows) + 0.5) * float(resolution)
    occ = np.zeros((K, T, height, width), dtype=np.float32)
    for k in range(K):
        for t in range(T):
            dx = x - float(path_xy[k, t, 0])
            dy = y - float(path_xy[k, t, 1])
            occ[k, t] = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).astype(np.float32)
    return occ[0] if squeeze else occ


def rel_xy_to_grid_cell(
    xy: np.ndarray | tuple[float, float],
    height: int,
    width: int,
    origin_xy: tuple[float, float],
    resolution: float,
) -> tuple[int, int]:
    point = np.asarray(xy, dtype=np.float32)
    col = int(np.floor((float(point[0]) - float(origin_xy[0])) / float(resolution)))
    row_from_bottom = int(np.floor((float(point[1]) - float(origin_xy[1])) / float(resolution)))
    row = height - 1 - row_from_bottom
    return row, col


def grid_cell_to_rel_xy(
    row: int,
    col: int,
    height: int,
    origin_xy: tuple[float, float],
    resolution: float,
) -> tuple[float, float]:
    x = float(origin_xy[0]) + (float(col) + 0.5) * float(resolution)
    y = float(origin_xy[1]) + ((height - 1 - float(row)) + 0.5) * float(resolution)
    return x, y


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))).astype(np.float32)


def dijkstra_grid_path(cost: np.ndarray, start: tuple[int, int], target_mask: np.ndarray) -> list[tuple[int, int]]:
    H, W = cost.shape
    if not (0 <= start[0] < H and 0 <= start[1] < W) or not bool(target_mask.any()):
        return []
    dist = np.full((H, W), np.inf, dtype=np.float32)
    prev_r = np.full((H, W), -1, dtype=np.int16)
    prev_c = np.full((H, W), -1, dtype=np.int16)
    sr, sc = start
    dist[sr, sc] = 0.0
    queue: list[tuple[float, int, int]] = [(0.0, sr, sc)]
    neighbors = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.4142),
        (-1, 1, 1.4142),
        (1, -1, 1.4142),
        (1, 1, 1.4142),
    )
    end: tuple[int, int] | None = None
    while queue:
        d, r, c = heapq.heappop(queue)
        if d > float(dist[r, c]) + 1e-6:
            continue
        if bool(target_mask[r, c]):
            end = (r, c)
            break
        for dr, dc, step in neighbors:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            nd = d + step * 0.5 * (float(cost[r, c]) + float(cost[nr, nc]))
            if nd < float(dist[nr, nc]):
                dist[nr, nc] = nd
                prev_r[nr, nc] = r
                prev_c[nr, nc] = c
                heapq.heappush(queue, (nd, nr, nc))
    if end is None:
        return []
    path: list[tuple[int, int]] = []
    r, c = end
    while r >= 0 and c >= 0:
        path.append((r, c))
        if (r, c) == start:
            break
        r, c = int(prev_r[r, c]), int(prev_c[r, c])
    return path[::-1]


def dilate_binary_mask(mask: np.ndarray, radius_m: float, resolution: float) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    rad = max(0, int(np.ceil(float(radius_m) / float(resolution))))
    if rad <= 0:
        return mask.copy()
    yy, xx = np.ogrid[-rad : rad + 1, -rad : rad + 1]
    disk = (xx * xx + yy * yy) <= rad * rad
    try:
        from scipy.ndimage import binary_dilation

        return binary_dilation(mask, structure=disk)
    except Exception:
        out = mask.copy()
        H, W = mask.shape
        for r, c in np.argwhere(mask):
            r0, r1 = max(0, r - rad), min(H, r + rad + 1)
            c0, c1 = max(0, c - rad), min(W, c + rad + 1)
            dr0, dr1 = r0 - (r - rad), disk.shape[0] - ((r + rad + 1) - r1)
            dc0, dc1 = c0 - (c - rad), disk.shape[1] - ((c + rad + 1) - c1)
            out[r0:r1, c0:c1] |= disk[dr0:dr1, dc0:dc1]
        return out


def distance_to_goal_mask(goal_mask: np.ndarray, resolution: float) -> np.ndarray:
    goal_mask = np.asarray(goal_mask, dtype=bool)
    if not bool(goal_mask.any()):
        return np.zeros_like(goal_mask, dtype=np.float32)
    try:
        from scipy.ndimage import distance_transform_edt

        return (distance_transform_edt(~goal_mask) * float(resolution)).astype(np.float32)
    except Exception:
        coords = np.argwhere(goal_mask).astype(np.float32)
        rows, cols = np.indices(goal_mask.shape, dtype=np.float32)
        points = np.stack([rows.reshape(-1), cols.reshape(-1)], axis=-1)
        dist = np.sqrt(((points[:, None] - coords[None]) ** 2).sum(axis=-1)).min(axis=1)
        return (dist.reshape(goal_mask.shape) * float(resolution)).astype(np.float32)


def build_stage1_static_fields_v2(
    distance_field: np.ndarray,
    goal_map: np.ndarray,
    config: Stage1OptimizerV2Config,
    origin_xy: tuple[float, float] = (-4.0, -3.0),
    resolution: float = 0.1,
    start_xy: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    distance = as_numpy(distance_field)
    if distance.ndim == 3:
        distance = distance[0]
    goal = as_numpy(goal_map)
    if goal.ndim == 3:
        goal = goal[0]
    H, W = distance.shape
    raw_static = sigmoid_np((float(config.static_center) - distance) / float(config.tau_static))
    goal_mask = goal > float(config.goal_threshold)
    if not bool(goal_mask.any()) and float(goal.max(initial=0.0)) > 0.0:
        goal_mask = goal >= float(goal.max()) * 0.5
    safe_mask = distance >= float(config.static_margin)
    start_cell = rel_xy_to_grid_cell(start_xy, H, W, origin_xy, resolution)
    start_unsafe = True
    if 0 <= start_cell[0] < H and 0 <= start_cell[1] < W:
        start_unsafe = bool(distance[start_cell] < float(config.static_margin))
    goal_center = compute_goal_centroid(goal, origin_xy, resolution)
    goal_center_cell = rel_xy_to_grid_cell(goal_center, H, W, origin_xy, resolution)
    goal_center_unsafe = True
    if 0 <= goal_center_cell[0] < H and 0 <= goal_center_cell[1] < W:
        goal_center_unsafe = bool(distance[goal_center_cell] < float(config.static_margin))
    goal_unsafe_ratio = 0.0
    if bool(goal_mask.any()):
        goal_unsafe_ratio = float((distance[goal_mask] < float(config.static_margin)).mean())
    goal_unsafe = bool(goal_center_unsafe or goal_unsafe_ratio >= float(config.goal_unsafe_ratio_threshold))

    path: list[tuple[int, int]] = []
    corridor = np.zeros((H, W), dtype=bool)
    if start_unsafe or goal_unsafe:
        cost = 1.0 + float(config.dijkstra_alpha) * raw_static
        if start_unsafe and not goal_unsafe:
            target_mask = safe_mask.copy()
            if 0 <= start_cell[0] < H and 0 <= start_cell[1] < W:
                target_mask[start_cell] = False
        else:
            target_mask = goal_mask
        path = dijkstra_grid_path(cost, start_cell, target_mask)
        if path:
            path_mask = np.zeros((H, W), dtype=bool)
            for row, col in path:
                path_mask[row, col] = True
            corridor |= dilate_binary_mask(path_mask, float(config.corridor_radius), resolution)
        if start_unsafe:
            start_mask = np.zeros((H, W), dtype=bool)
            if 0 <= start_cell[0] < H and 0 <= start_cell[1] < W:
                start_mask[start_cell] = True
            corridor |= dilate_binary_mask(start_mask, float(config.start_open_radius), resolution)
        if goal_unsafe and bool(goal_mask.any()):
            corridor |= dilate_binary_mask(goal_mask, float(config.goal_open_radius), resolution)

    final_static = raw_static * (1.0 - float(config.corridor_strength) * corridor.astype(np.float32))
    final_static = np.clip(final_static, 0.0, 1.0).astype(np.float32)
    goal_distance = distance_to_goal_mask(goal_mask, resolution)
    return {
        "raw_static": raw_static.astype(np.float32),
        "final_static": final_static,
        "corridor": corridor,
        "goal_distance": goal_distance,
        "goal_mask": goal_mask,
        "start_unsafe": start_unsafe,
        "goal_unsafe": goal_unsafe,
        "goal_unsafe_ratio": goal_unsafe_ratio,
        "path": path,
    }


def limit_velocity_norm(vel: torch.Tensor, max_speed: float) -> torch.Tensor:
    if float(max_speed) <= 0.0:
        return vel
    speed = vel.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = torch.clamp(float(max_speed) / speed, max=1.0)
    return vel * scale


def limit_acceleration_norm(
    vel: torch.Tensor,
    current_vel_xy: torch.Tensor,
    max_acc: float,
    dt: float,
    max_speed: Optional[float] = None,
) -> torch.Tensor:
    if float(max_acc) <= 0.0 or vel.shape[1] == 0:
        return vel

    max_delta = float(max_acc) * float(dt)
    prev = current_vel_xy
    if max_speed is not None and float(max_speed) > 0.0:
        prev = limit_velocity_norm(prev[:, None], float(max_speed))[:, 0]

    out = []
    for t in range(vel.shape[1]):
        delta = vel[:, t] - prev
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(max_delta / delta_norm, max=1.0)
        clipped = prev + delta * scale
        out.append(clipped)
        prev = clipped
    return torch.stack(out, dim=1)


def constrain_stage1_velocity(
    vel: torch.Tensor,
    current_vel_xy: torch.Tensor,
    speed_bound: float,
    acc_bound: float,
    dt: float,
) -> torch.Tensor:
    vel = limit_velocity_norm(vel, speed_bound)
    vel = limit_acceleration_norm(vel, current_vel_xy, acc_bound, dt, max_speed=speed_bound)
    return vel


def integrate_positions_batch(current_pos_xy: torch.Tensor, vel_xy: torch.Tensor, dt: float) -> torch.Tensor:
    return current_pos_xy[:, None] + torch.cumsum(vel_xy * float(dt), dim=1)


def optimize_stage1_trajectory_batch_v2(
    pred_ego: torch.Tensor,
    pred_others: torch.Tensor,
    others_valid: torch.Tensor,
    distance_fields: torch.Tensor,
    static_fields: torch.Tensor,
    goal_distance_fields: torch.Tensor,
    gt_ego: torch.Tensor,
    gt_others: torch.Tensor,
    current_vel_xy: torch.Tensor,
    speed_bound: float,
    config: Stage1OptimizerV2Config,
    origin_xy: tuple[float, float] = (-4.0, -3.0),
    resolution: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = pred_ego.device
    B = pred_ego.shape[0]
    horizon = min(int(config.horizon), pred_ego.shape[1], gt_ego.shape[1])
    pred_ego = pred_ego[:, :horizon]
    gt_ego = gt_ego[:, :horizon]
    pred_others = pred_others[:, :, :horizon]
    gt_others = gt_others[:, :, :horizon]
    current_pos = torch.zeros((B, 2), device=device, dtype=pred_ego.dtype)
    speed_bound = float(speed_bound if float(speed_bound) > 0.0 else config.speed_bound_mps)
    acc_bound = float(config.acc_bound_mps2)
    current_vel_ref = limit_velocity_norm(current_vel_xy[:, None], speed_bound)[:, 0]
    pred_vel = torch.empty_like(pred_ego)
    pred_vel[:, 0] = (pred_ego[:, 0] - current_pos) / float(config.dt)
    if horizon > 1:
        pred_vel[:, 1:] = (pred_ego[:, 1:] - pred_ego[:, :-1]) / float(config.dt)
    current_vel_seq = current_vel_xy[:, None].expand_as(pred_vel)
    if config.init_mode == "prediction":
        init_vel = pred_vel
    elif config.init_mode == "current_velocity":
        init_vel = current_vel_seq
    elif config.init_mode == "blend":
        init_vel = 0.5 * (pred_vel + current_vel_seq)
    else:
        raise ValueError(f"unsupported init_mode: {config.init_mode}")
    init_vel = constrain_stage1_velocity(init_vel, current_vel_xy, speed_bound, acc_bound, float(config.dt))
    vel_param = torch.nn.Parameter(init_vel.detach().clone())
    optimizer = torch.optim.Adam([vel_param], lr=float(config.lr))
    w_init = float(config.start_vel_weight)
    w_acc = float(config.smooth_scale)
    w_jerk = 0.3 * float(config.smooth_scale)
    others_valid_f = others_valid[:, :, None].to(dtype=pred_ego.dtype, device=device)
    static_collision_threshold = float(sigmoid_np((float(config.static_center) - float(config.static_margin)) / float(config.tau_static)))

    debug: dict[str, float] = {}
    for _ in range(int(config.iters)):
        vel = constrain_stage1_velocity(vel_param, current_vel_xy, speed_bound, acc_bound, float(config.dt))
        pos = integrate_positions_batch(current_pos, vel, float(config.dt))
        prior_cost = (pos - pred_ego).square().sum(dim=-1).mean()
        goal_values = sample_batched_static_maps(goal_distance_fields, pos, origin_xy, resolution)
        if horizon > 1:
            progress_violation = F.relu(goal_values[:, 1:] - goal_values[:, :-1])
            progress_cost = progress_violation.square().mean()
        else:
            progress_cost = pred_ego.new_zeros(())
        goal_cost = (
            goal_values.mean()
            + goal_values.square().mean()
            + float(config.terminal_weight_ratio) * goal_values[:, -1].square().mean()
            + float(config.goal_progress_weight) * progress_cost
        )
        static_values = sample_batched_static_maps(static_fields, pos, origin_xy, resolution)
        if int(config.static_topk_frac * horizon) > 0:
            topk_count = max(1, min(horizon, int(math.ceil(float(config.static_topk_frac) * horizon))))
            static_topk = static_values.topk(topk_count, dim=1).values.mean()
        else:
            static_topk = pred_ego.new_zeros(())
        static_cost = static_values.mean() + float(config.static_topk_weight) * static_topk
        if pred_others.numel() > 0 and bool(others_valid.any()):
            dist_dyn = (pos[:, None] - pred_others).norm(dim=-1)
            dyn_values = torch.sigmoid((float(config.dyn_margin) - dist_dyn) / float(config.tau_dyn))
            dyn_cost = (dyn_values * others_valid_f).sum() / others_valid_f.sum().clamp_min(1.0) / float(horizon)
        else:
            dyn_cost = pred_ego.new_zeros(())
        init_cost = (vel[:, 0] - current_vel_ref).square().sum(dim=-1).mean()
        acc_cost = (vel[:, 1:] - vel[:, :-1]).square().sum(dim=-1).mean() if horizon > 1 else pred_ego.new_zeros(())
        jerk_cost = (
            (vel[:, 2:] - 2.0 * vel[:, 1:-1] + vel[:, :-2]).square().sum(dim=-1).mean()
            if horizon > 2
            else pred_ego.new_zeros(())
        )
        loss = (
            float(config.w_prior) * prior_cost
            + float(config.w_goal) * goal_cost
            + float(config.w_static) * static_cost
            + float(config.w_dyn) * dyn_cost
            + w_init * init_cost
            + w_acc * acc_cost
            + w_jerk * jerk_cost
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(config.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_([vel_param], float(config.grad_clip))
        optimizer.step()
        debug = {
            "loss": float(loss.detach().cpu()),
            "prior": float(prior_cost.detach().cpu()),
            "goal": float(goal_cost.detach().cpu()),
            "static": float(static_cost.detach().cpu()),
            "static_topk": float(static_topk.detach().cpu()),
            "goal_progress": float(progress_cost.detach().cpu()),
            "dynamic": float(dyn_cost.detach().cpu()),
            "init": float(init_cost.detach().cpu()),
            "acc": float(acc_cost.detach().cpu()),
            "jerk": float(jerk_cost.detach().cpu()),
        }

    with torch.no_grad():
        vel = constrain_stage1_velocity(vel_param, current_vel_xy, speed_bound, acc_bound, float(config.dt))
        pos = integrate_positions_batch(current_pos, vel, float(config.dt))
        gt_dist = (pos - gt_ego).norm(dim=-1)
        goal_values = sample_batched_static_maps(goal_distance_fields, pos, origin_xy, resolution)
        sampled_distance = sample_batched_static_maps(distance_fields, pos, origin_xy, resolution)
        sampled_static = sample_batched_static_maps(static_fields, pos, origin_xy, resolution)
        static_violation = sampled_static > static_collision_threshold
        raw_static_violation = sampled_distance < float(config.static_margin)
        if gt_others.numel() > 0 and bool(others_valid.any()):
            dyn_dist = (pos[:, None] - gt_others).norm(dim=-1)
            valid_dyn_dist = dyn_dist.masked_fill(~others_valid[:, :, None].to(device=device), float("inf"))
            min_dyn_dist = valid_dyn_dist.min(dim=1).values
            dyn_violation = min_dyn_dist < float(config.dyn_margin)
            finite_dyn = torch.isfinite(min_dyn_dist)
            dyn_collision_ratio = dyn_violation[finite_dyn].float().mean() if bool(finite_dyn.any()) else pred_ego.new_zeros(())
            min_dynamic_distance_m = min_dyn_dist[finite_dyn].min() if bool(finite_dyn.any()) else pred_ego.new_tensor(float("nan"))
        else:
            dyn_collision_ratio = pred_ego.new_zeros(())
            min_dynamic_distance_m = pred_ego.new_tensor(float("nan"))
        metrics = {
            "gt_ade_30_mm": float(gt_dist.mean().detach().cpu() * 1000.0),
            "gt_fde_30_mm": float(gt_dist[:, -1].mean().detach().cpu() * 1000.0),
            "goal_mean_dist_mm": float(goal_values.mean().detach().cpu() * 1000.0),
            "goal_final_dist_mm": float(goal_values[:, -1].mean().detach().cpu() * 1000.0),
            "static_collision_ratio": float(static_violation.float().mean().detach().cpu()),
            "raw_static_collision_ratio": float(raw_static_violation.float().mean().detach().cpu()),
            "mean_final_static": float(sampled_static.mean().detach().cpu()),
            "max_final_static": float(sampled_static.max().detach().cpu()),
            "static_collision_threshold": float(static_collision_threshold),
            "dynamic_collision_ratio": float(dyn_collision_ratio.detach().cpu()),
            "min_static_distance_m": float(sampled_distance.min().detach().cpu()),
            "min_dynamic_distance_m": float(min_dynamic_distance_m.detach().cpu()),
            "mean_speed_mps": float(vel.norm(dim=-1).mean().detach().cpu()),
            "max_speed_mps": float(vel.norm(dim=-1).max().detach().cpu()),
            "speed_bound_mps": float(speed_bound),
            "acc_bound_mps2": float(acc_bound),
        }
        if horizon > 0:
            delta_vel = torch.cat([vel[:, :1] - current_vel_ref[:, None], vel[:, 1:] - vel[:, :-1]], dim=1)
            acc_norm = delta_vel.norm(dim=-1) / float(config.dt)
            metrics.update(
                {
                    "mean_acc_mps2": float(acc_norm.mean().detach().cpu()),
                    "max_acc_mps2": float(acc_norm.max().detach().cpu()),
                    "start_vel_error_mps": float((vel[:, 0] - current_vel_ref).norm(dim=-1).mean().detach().cpu()),
                }
            )
        metrics.update({f"debug_{key}": value for key, value in debug.items()})
    return pos.detach(), metrics


def build_plan_fields(
    prediction: dict[str, Any],
    scene_maps: Any,
    goal_map: Any,
    map_origin_xy: tuple[float, float],
    map_resolution: float,
) -> PlanFields:
    scene_maps_np = as_numpy(scene_maps)
    goal_np = as_numpy(goal_map)
    H, W = scene_maps_np.shape[-2:]
    if "x0_hat" in prediction:
        x0 = as_numpy(prediction["x0_hat"])
        if x0.ndim == 4 and x0.shape[-1] == 2:
            x0 = x0[0]
        if x0.ndim == 3 and x0.shape[-1] == 2:
            path_60 = interp_path_time(x0, out_len=60)
            occ_60 = trajectory_to_occ(path_60, H, W, map_origin_xy, map_resolution)
            static_sdf = build_static_sdf(scene_maps_np, map_resolution)
            goal_center = compute_goal_centroid(goal_np, map_origin_xy, map_resolution)
            return PlanFields(
                ego_prior_60=occ_60[0],
                others_prior_60=occ_60[1:],
                static_sdf=static_sdf,
                goal_map=goal_np[0] if goal_np.ndim == 3 else goal_np,
                ref_path_xy=path_60[0],
                goal_center_xy=goal_center,
                map_origin_xy=map_origin_xy,
                map_resolution=float(map_resolution),
            )
        if x0.ndim == 5:
            x0 = x0[0]
        ego_occ = x0[0]
        others_occ = x0[1:]
    else:
        ego_occ = as_numpy(prediction["ego_future_occ"])
        others_occ = as_numpy(prediction["others_future_occ"])
        if ego_occ.ndim == 4:
            ego_occ = ego_occ[0]
        if others_occ.ndim == 5:
            others_occ = others_occ[0]
    ego_prior_60 = interp_time(ego_occ, out_len=60)
    others_prior_60 = interp_time(others_occ, out_len=60)
    static_sdf = build_static_sdf(scene_maps_np, map_resolution)
    ref_path = extract_ref_path_simple(ego_prior_60, map_origin_xy, map_resolution)
    goal_center = compute_goal_centroid(goal_np, map_origin_xy, map_resolution)
    return PlanFields(
        ego_prior_60=ego_prior_60,
        others_prior_60=others_prior_60,
        static_sdf=static_sdf,
        goal_map=goal_np[0] if goal_np.ndim == 3 else goal_np,
        ref_path_xy=ref_path,
        goal_center_xy=goal_center,
        map_origin_xy=map_origin_xy,
        map_resolution=float(map_resolution),
    )


def integrate_positions(current_pos_xy: torch.Tensor, vel_xy: torch.Tensor, dt: float) -> torch.Tensor:
    return current_pos_xy[None] + torch.cumsum(vel_xy * float(dt), dim=0)


def optimize_root(
    fields: PlanFields,
    planner_state: PlannerState,
    active_action: ActiveAction,
    current_pos_xy: tuple[float, float],
    current_vel_xy: tuple[float, float],
    config: OptimizerConfig = OptimizerConfig(),
) -> FinePlan:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon = int(config.horizon)
    speed_profile = config.speed_profile or SPEED_PROFILE
    profile = speed_profile[planner_state.mode]
    v_max = float(profile["v_max"])
    v_pref = float(profile["v_pref"])

    current_pos = torch.tensor(current_pos_xy, device=device, dtype=torch.float32)
    current_vel = torch.tensor(current_vel_xy, device=device, dtype=torch.float32)
    init_ref = torch.from_numpy(fields.ref_path_xy[:horizon]).to(device=device, dtype=torch.float32)
    init_vel = torch.empty(horizon, 2, device=device, dtype=torch.float32)
    init_vel[0] = current_vel
    init_vel[1:] = (init_ref[1:] - init_ref[:-1]) / float(config.dt)
    vel_param = nn_parameter = torch.nn.Parameter(init_vel.clamp(min=-v_max, max=v_max))

    ego_prior = torch.from_numpy(fields.ego_prior_60[:horizon]).to(device=device, dtype=torch.float32).clamp_min(0.0)
    others_prior = torch.from_numpy(fields.others_prior_60[:, :horizon]).to(device=device, dtype=torch.float32).clamp_min(0.0)
    static_sdf = torch.from_numpy(fields.static_sdf).to(device=device, dtype=torch.float32)
    goal_center = torch.tensor(fields.goal_center_xy, device=device, dtype=torch.float32)
    ref_path = init_ref
    optimizer = torch.optim.Adam([nn_parameter], lr=float(config.lr))

    debug: dict[str, Any] = {}
    for _ in range(int(config.steps)):
        vel = v_max * torch.tanh(vel_param / max(v_max, 1e-6))
        pos = integrate_positions(current_pos, vel, config.dt)

        prior_values = sample_map_sequence(ego_prior, pos, fields.map_origin_xy, fields.map_resolution)
        prior_cost = -torch.log(prior_values.clamp_min(1e-5)).mean()

        goal_cost = (pos[-1] - goal_center).square().sum()
        if planner_state.mode == "WAIT" and planner_state.wait_point_xy is not None:
            wait_point = torch.tensor(planner_state.wait_point_xy, device=device, dtype=torch.float32)
            goal_cost = (pos[-1] - wait_point).square().sum()
        elif planner_state.mode == "ACT":
            goal_cost = (pos - current_pos[None]).square().sum(dim=-1).mean()

        dyn_cost = torch.zeros((), device=device)
        if others_prior.numel() > 0:
            dyn_values = [sample_map_sequence(others_prior[j], pos, fields.map_origin_xy, fields.map_resolution) for j in range(others_prior.shape[0])]
            dyn_cost = torch.stack(dyn_values, dim=0).sum(dim=0).mean()

        sdf_values = sample_static_map(static_sdf, pos, fields.map_origin_xy, fields.map_resolution)
        static_cost = F.relu(float(config.static_margin_m) - sdf_values).square().mean()

        smooth_cost = (vel[1:] - vel[:-1]).square().sum(dim=-1).mean()
        speed_cost = (vel.norm(dim=-1) - v_pref).square().mean()
        loss = (
            config.prior_weight * prior_cost
            + config.goal_weight * goal_cost
            + config.dynamic_weight * dyn_cost
            + config.static_weight * static_cost
            + config.smooth_weight * smooth_cost
            + config.speed_weight * speed_cost
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        debug = {
            "loss": float(loss.detach().cpu()),
            "prior": float(prior_cost.detach().cpu()),
            "goal": float(goal_cost.detach().cpu()),
            "dynamic": float(dyn_cost.detach().cpu()),
            "static": float(static_cost.detach().cpu()),
            "smooth": float(smooth_cost.detach().cpu()),
            "speed": float(speed_cost.detach().cpu()),
        }

    with torch.no_grad():
        vel = v_max * torch.tanh(vel_param / max(v_max, 1e-6))
        pos = integrate_positions(current_pos, vel, config.dt)
        yaw = torch.atan2(vel[:, 1], vel[:, 0])
    return FinePlan(
        pos_xy=pos.detach().cpu().numpy().astype(np.float32),
        vel_xy=vel.detach().cpu().numpy().astype(np.float32),
        yaw=yaw.detach().cpu().numpy().astype(np.float32),
        success=True,
        debug=debug,
    )


def is_goal_blocked(fields: PlanFields, tau: float = 0.3) -> bool:
    goal_mask = as_numpy(fields.goal_map) > 0.1
    if not goal_mask.any() or fields.others_prior_60.size == 0:
        return False
    risk = fields.others_prior_60[:, :20, goal_mask].mean()
    return bool(risk > tau)


def is_act_collision_risky(fine_plan: FinePlan, fields: PlanFields, tau: float = 0.25, steps: int = 20) -> bool:
    if fields.others_prior_60.size == 0:
        return False
    device = torch.device("cpu")
    pos = torch.from_numpy(fine_plan.pos_xy[:steps]).to(device=device, dtype=torch.float32)
    risks = []
    for other in fields.others_prior_60:
        maps = torch.from_numpy(other[:steps]).to(device=device, dtype=torch.float32)
        risks.append(sample_map_sequence(maps, pos, fields.map_origin_xy, fields.map_resolution))
    risk = torch.stack(risks, dim=0).sum(dim=0).mean()
    return bool(float(risk) > tau)


def is_wait_resolved(fields: PlanFields, tau_block: float = 0.15) -> bool:
    return not is_goal_blocked(fields, tau=tau_block)


def choose_wait_point(fields: PlanFields, radius: float = 0.8, candidates: int = 32) -> tuple[float, float]:
    center = np.asarray(fields.goal_center_xy, dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * np.pi, num=candidates, endpoint=False, dtype=np.float32)
    pts = center[None] + radius * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    device = torch.device("cpu")
    pts_t = torch.from_numpy(pts).to(device=device, dtype=torch.float32)
    static = torch.from_numpy(fields.static_sdf).to(device=device, dtype=torch.float32)
    sdf = sample_static_map(static, pts_t, fields.map_origin_xy, fields.map_resolution).numpy()
    dyn = np.zeros((candidates,), dtype=np.float32)
    for other in fields.others_prior_60:
        maps = torch.from_numpy(other[:1]).repeat(candidates, 1, 1)
        dyn += sample_map_sequence(maps, pts_t, fields.map_origin_xy, fields.map_resolution).numpy()
    score = 2.0 * dyn + np.maximum(0.0, 0.25 - sdf) ** 2 + 0.2 * np.linalg.norm(pts - center[None], axis=-1)
    best = pts[int(np.argmin(score))]
    return float(best[0]), float(best[1])


def planner_step(
    fine_plan: FinePlan,
    fields: PlanFields,
    planner_state: PlannerState,
    active_action: ActiveAction,
) -> PlannerState:
    state = PlannerState(
        mode=planner_state.mode,
        wait_point_xy=planner_state.wait_point_xy,
        interrupted=planner_state.interrupted,
        elapsed_wait_steps=planner_state.elapsed_wait_steps,
    )
    if state.mode == "MOVE" and is_goal_blocked(fields):
        state.mode = "WAIT"
        state.wait_point_xy = choose_wait_point(fields)
        state.elapsed_wait_steps = 0
        return state
    if state.mode == "ACT" and active_action.interruptible and is_act_collision_risky(fine_plan, fields):
        state.mode = "WAIT"
        state.wait_point_xy = choose_wait_point(fields)
        state.interrupted = True
        state.elapsed_wait_steps = 0
        return state
    if state.mode == "WAIT":
        state.elapsed_wait_steps += 1
        if is_wait_resolved(fields):
            state.mode = "ACT" if state.interrupted else "MOVE"
            state.wait_point_xy = None
            state.interrupted = False
            state.elapsed_wait_steps = 0
    return state


def stage1_plan_step(
    prediction: dict[str, Any],
    scene_maps: Any,
    goal_map: Any,
    map_origin_xy: tuple[float, float],
    map_resolution: float,
    active_action: ActiveAction,
    planner_state: PlannerState,
    current_pos_xy: tuple[float, float],
    current_vel_xy: tuple[float, float],
    optimizer_config: OptimizerConfig = OptimizerConfig(),
) -> tuple[FinePlan, PlannerState, Stage1PlanOutput]:
    fields = build_plan_fields(
        prediction=prediction,
        scene_maps=scene_maps,
        goal_map=goal_map,
        map_origin_xy=map_origin_xy,
        map_resolution=map_resolution,
    )
    fine_plan = optimize_root(
        fields=fields,
        planner_state=planner_state,
        active_action=active_action,
        current_pos_xy=current_pos_xy,
        current_vel_xy=current_vel_xy,
        config=optimizer_config,
    )
    new_state = planner_step(
        fine_plan=fine_plan,
        fields=fields,
        planner_state=planner_state,
        active_action=active_action,
    )
    output = Stage1PlanOutput(
        fine_root_xy=fine_plan.pos_xy,
        fine_root_vel_xy=fine_plan.vel_xy,
        fine_root_yaw=fine_plan.yaw,
        mode=new_state.mode,
        wait_point_xy=new_state.wait_point_xy,
        interrupted=new_state.interrupted,
    )
    return fine_plan, new_state, output
