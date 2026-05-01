from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.planning import (
    LOCAL_RESOLUTION,
    LOCAL_X_RANGE,
    LOCAL_Z_RANGE,
    MODEL_MAP_SIZE,
    make_gaussian_map,
    sample_scene_crop,
)
from models.stage1_planner import (
    Stage1OptimizerV2Config,
    build_stage1_static_fields_v2,
    optimize_stage1_trajectory_batch_v2,
)
from models.stage1_predictor import Stage1Predictor


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_ROOT / "data" / "preprocessed"
MAX_SLOTS = 4
MAX_OTHERS = 3
LOCAL_ORIGIN = (LOCAL_X_RANGE[0], LOCAL_Z_RANGE[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-1 trajectory-only closed-loop episode simulation.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--episodes-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--optimizer-config", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--episodes-per-scene", type=int, default=100)
    parser.add_argument("--episode-seed", type=int, default=2026)
    parser.add_argument("--sample-set-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--past-frames", type=int, default=30)
    parser.add_argument("--future-frames", type=int, default=72)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--execute-frames", type=int, default=15)
    parser.add_argument("--arrival-radius-m", type=float, default=0.20)
    parser.add_argument("--action-direct-threshold-m", type=float, default=0.50)
    parser.add_argument("--max-frames-per-goal", type=int, default=300)
    parser.add_argument("--action-hold-frames", type=int, default=60)
    parser.add_argument("--interrupt-distance-m", type=float, default=0.25)
    parser.add_argument("--wait-release-distance-m", type=float, default=0.80)
    parser.add_argument("--collision-distance-m", type=float, default=0.50)
    parser.add_argument("--wait-ring-radius-m", type=float, default=0.80)
    parser.add_argument("--wait-hold-frames", type=int, default=30)
    parser.add_argument("--max-wait-frames", type=int, default=180)
    parser.add_argument("--w-goal", type=float, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--save-trajectories", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-window-plots-per-episode", type=int, default=24)
    return parser.parse_args()


def repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def normalize_goal_type(goal_type: Any) -> str:
    value = "" if goal_type is None else str(goal_type)
    return "move" if value in {"walk", "move"} else value


def body_goal_xy(goal: dict[str, Any]) -> Optional[np.ndarray]:
    value = goal.get("body_goal")
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 3 or not np.isfinite(arr[:3]).all():
        return None
    return arr[[0, 2]].astype(np.float32)


@dataclass
class SceneInfo:
    scene_id: str
    scene_maps: np.ndarray
    distance_map: np.ndarray
    origin_xy: tuple[float, float]
    resolution: float


@dataclass
class CharacterClip:
    character_id: str
    sequence_id: str
    local_start: int
    local_end: int
    root_xy: np.ndarray
    goals: list[dict[str, Any]]


def load_scene_info(dataset: str, scene_id: str, root: Path = DEFAULT_ROOT) -> SceneInfo:
    scene_json = root / dataset / "scenes_v2" / scene_id / "scene.json"
    scene = load_json(scene_json)
    clearance = np.load(repo_path(scene["clearance_map_npy_path"])).astype(np.float32)
    distance = np.load(repo_path(scene["distance_map_npy_path"])).astype(np.float32)
    occ = np.load(repo_path(scene["occupancy_grid_path"]), mmap_mode="r")
    obstacle = np.asarray(occ[:, 1:86, :].any(axis=1), dtype=np.float32)
    scene_maps = np.stack([clearance, distance, obstacle], axis=0).astype(np.float32)
    grid_meta = scene["grid_meta"]
    x_min = float(grid_meta["x_min"])
    z_min = float(grid_meta["z_min"])
    resolution = float((float(grid_meta["x_max"]) - x_min) / int(grid_meta["x_res"]))
    return SceneInfo(scene_id=scene_id, scene_maps=scene_maps, distance_map=distance, origin_xy=(x_min, z_min), resolution=resolution)


def resolve_character_clip(dataset: str, scene_id: str, char: dict[str, Any], root: Path = DEFAULT_ROOT) -> CharacterClip:
    if isinstance(char.get("source_window"), dict):
        window = char["source_window"]
        sequence_id = str(window["sequence_id"])
        local_start = int(window["local_start"])
        local_end = int(window["local_end"])
    else:
        goals = char.get("goal_sequence") or []
        if not goals:
            raise ValueError(f"character {char.get('character_id')} has no goal_sequence")
        first_source = goals[0].get("source_segment", {})
        last_source = goals[-1].get("source_segment", {})
        sequence_id = str(char.get("sequence_id") or first_source["sequence_id"])
        local_start = int(first_source["start"])
        local_end = int(last_source["end"])

    seq_path = root / dataset / "scenes_v2" / scene_id / "sequences" / f"{sequence_id}.json"
    seq = load_json(seq_path)
    motion = seq["human_motion_ref"]
    seq_global_start = int(motion["start"])
    transl = np.load(repo_path(motion["smplx"]["transl_path"]), mmap_mode="r")
    start = seq_global_start + local_start
    end = seq_global_start + local_end + 1
    root_xy = np.asarray(transl[start:end, :], dtype=np.float32)[:, [0, 2]]
    return CharacterClip(
        character_id=str(char.get("character_id", "unknown")),
        sequence_id=sequence_id,
        local_start=local_start,
        local_end=local_end,
        root_xy=root_xy,
        goals=list(char.get("goal_sequence") or []),
    )


def sample_episode_set(args: argparse.Namespace, episodes_dir: Path) -> list[str]:
    sample_set = args.sample_set_path
    if sample_set is None:
        sample_set = DEFAULT_ROOT / args.dataset / "stage1_loop_eval_episodes_v1" / f"{args.split}_scene{args.episodes_per_scene}_seed{args.episode_seed}.json"
    if sample_set.exists():
        payload = load_json(sample_set)
        return [str(path) for path in payload["episode_paths"]]

    rng = random.Random(int(args.episode_seed))
    episode_paths: list[str] = []
    per_scene: dict[str, int] = {}
    for scene_dir in sorted(path for path in episodes_dir.iterdir() if path.is_dir()):
        candidates = sorted(scene_dir.glob("episode_*.json"))
        if not candidates:
            continue
        selected = rng.sample(candidates, k=min(int(args.episodes_per_scene), len(candidates)))
        per_scene[scene_dir.name] = len(selected)
        episode_paths.extend(display_path(path.resolve()) for path in selected)
    write_json(
        sample_set,
        {
            "dataset": args.dataset,
            "split": args.split,
            "episodes_per_scene": int(args.episodes_per_scene),
            "seed": int(args.episode_seed),
            "per_scene": per_scene,
            "episode_paths": episode_paths,
        },
    )
    return episode_paths


def build_model(checkpoint: Path, past_frames: int, future_frames: int, device: torch.device) -> Stage1Predictor:
    ckpt = torch.load(checkpoint, map_location="cpu")
    ckpt_args = dict(ckpt.get("args") or {})
    model = Stage1Predictor(
        slots=MAX_SLOTS,
        past_frames=past_frames,
        future_frames=future_frames,
        scene_channels=3,
        hidden_dim=int(ckpt_args.get("model_hidden_dim", 256)),
        num_layers=int(ckpt_args.get("model_layers", 6)),
        num_heads=int(ckpt_args.get("model_heads", 8)),
        dropout=float(ckpt_args.get("model_dropout", 0.0)),
        num_timesteps=int(ckpt_args.get("num_diffusion_steps", 50)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def load_optimizer_config(path: Path, args: argparse.Namespace) -> Stage1OptimizerV2Config:
    payload = load_json(path)
    values = dict(payload.get("config") or payload)
    allowed = Stage1OptimizerV2Config.__dataclass_fields__.keys()
    config = Stage1OptimizerV2Config(**{key: value for key, value in values.items() if key in allowed})
    if args.w_goal is not None:
        config.w_goal = float(args.w_goal)
    config.horizon = int(args.horizon)
    setattr(config, "speed_bound_mps", float(values.get("speed_bound_mps", 1.0)))
    return config


def pad_history(points: np.ndarray, end_exclusive: int, length: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((length, 2), dtype=np.float32)
    indices = np.arange(end_exclusive - length, end_exclusive)
    indices = np.clip(indices, 0, len(points) - 1)
    return np.asarray(points[indices], dtype=np.float32)


def future_slice(points: np.ndarray, start: int, length: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((length, 2), dtype=np.float32)
    indices = np.arange(start, start + length)
    indices = np.clip(indices, 0, len(points) - 1)
    return np.asarray(points[indices], dtype=np.float32)


def velocity_from_history(abs_pos: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(abs_pos, dtype=np.float32)
    vel[:, 1:] = abs_pos[:, 1:] - abs_pos[:, :-1]
    return vel


def clipped_goal_for_map(goal_rel_xy: np.ndarray) -> tuple[np.ndarray, bool]:
    x, z = float(goal_rel_xy[0]), float(goal_rel_xy[1])
    in_crop = bool(LOCAL_X_RANGE[0] <= x <= LOCAL_X_RANGE[1] and LOCAL_Z_RANGE[0] <= z <= LOCAL_Z_RANGE[1])
    clipped = np.asarray(
        [
            np.clip(x, LOCAL_X_RANGE[0] + 0.2, LOCAL_X_RANGE[1] - 0.2),
            np.clip(z, LOCAL_Z_RANGE[0] + 0.2, LOCAL_Z_RANGE[1] - 0.2),
        ],
        dtype=np.float32,
    )
    return clipped, in_crop


def goal_distance_field_to_body_goal(goal_rel_xy: np.ndarray) -> np.ndarray:
    H, W = MODEL_MAP_SIZE
    z_values = LOCAL_Z_RANGE[1] - (np.arange(H, dtype=np.float32) + 0.5) * LOCAL_RESOLUTION
    x_values = LOCAL_X_RANGE[0] + (np.arange(W, dtype=np.float32) + 0.5) * LOCAL_RESOLUTION
    xx, zz = np.meshgrid(x_values, z_values)
    return np.sqrt((xx - float(goal_rel_xy[0])) ** 2 + (zz - float(goal_rel_xy[1])) ** 2).astype(np.float32)


def build_predictor_batch(
    scene: SceneInfo,
    ego_history: np.ndarray,
    others_clips: list[CharacterClip],
    sim_t: int,
    active_goal_xy: np.ndarray,
    past_frames: int,
    future_frames: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    ego_current = np.asarray(ego_history[-1], dtype=np.float32)
    scene_crop = sample_scene_crop(scene.scene_maps, ego_current, scene.origin_xy, scene.resolution)
    abs_hist = np.zeros((MAX_SLOTS, past_frames, 2), dtype=np.float32)
    abs_hist[0] = pad_history(ego_history, len(ego_history), past_frames)
    entity_valid = np.zeros((MAX_SLOTS,), dtype=np.bool_)
    entity_valid[0] = True
    others_future_abs = np.zeros((MAX_OTHERS, future_frames, 2), dtype=np.float32)
    for idx, clip in enumerate(others_clips[:MAX_OTHERS], start=1):
        abs_hist[idx] = pad_history(clip.root_xy, sim_t + 1, past_frames)
        others_future_abs[idx - 1] = future_slice(clip.root_xy, sim_t + 1, future_frames)
        entity_valid[idx] = True
    rel_hist = abs_hist - ego_current[None, None]
    past_vel = velocity_from_history(abs_hist)
    goal_rel_xy = np.asarray(active_goal_xy - ego_current, dtype=np.float32)
    map_goal, in_crop = clipped_goal_for_map(goal_rel_xy)
    goal_map = make_gaussian_map(map_goal[None], MODEL_MAP_SIZE[0], MODEL_MAP_SIZE[1], LOCAL_RESOLUTION, LOCAL_ORIGIN, sigma=0.25)[None]
    batch = {
        "scene_maps": torch.from_numpy(scene_crop[None]).float().to(device),
        "goal_map": torch.from_numpy(goal_map[None]).float().to(device),
        "goal_rel_xy": torch.from_numpy(goal_rel_xy[None]).float().to(device),
        "goal_valid": torch.ones((1,), dtype=torch.bool, device=device),
        "goal_in_crop": torch.tensor([in_crop], dtype=torch.bool, device=device),
        "entity_valid": torch.from_numpy(entity_valid[None]).to(device),
        "past_rel_pos": torch.from_numpy(rel_hist[None]).float().to(device),
        "past_vel": torch.from_numpy(past_vel[None]).float().to(device),
        "x0": torch.zeros((1, MAX_SLOTS, future_frames, 2), dtype=torch.float32, device=device),
    }
    return batch, scene_crop, others_future_abs


def action_intrusion_likely(goal_xy: np.ndarray, others_clips: list[CharacterClip], sim_t: int, horizon: int, threshold: float) -> bool:
    for clip in others_clips[:MAX_OTHERS]:
        fut = future_slice(clip.root_xy, sim_t, horizon)
        if len(fut) and float(np.min(np.linalg.norm(fut - goal_xy[None], axis=-1))) <= float(threshold):
            return True
    return False


def sample_distance_map(distance_map: np.ndarray, point_xy: np.ndarray, scene: SceneInfo) -> float:
    x, z = float(point_xy[0]), float(point_xy[1])
    col = int(math.floor((x - scene.origin_xy[0]) / scene.resolution))
    row_from_bottom = int(math.floor((z - scene.origin_xy[1]) / scene.resolution))
    row = distance_map.shape[1] - 1 - row_from_bottom
    # distance_map is [x,z], so index directly by x/z grid.
    ix = int(math.floor((x - scene.origin_xy[0]) / scene.resolution))
    iz = int(math.floor((z - scene.origin_xy[1]) / scene.resolution))
    if ix < 0 or ix >= distance_map.shape[0] or iz < 0 or iz >= distance_map.shape[1]:
        return 0.0
    return float(distance_map[ix, iz])


def choose_wait_point(scene: SceneInfo, current_xy: np.ndarray, goal_xy: np.ndarray, others_clips: list[CharacterClip], sim_t: int, args: argparse.Namespace) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, num=24, endpoint=False)
    candidates = [current_xy.copy()]
    for radius in (args.wait_ring_radius_m, args.wait_ring_radius_m * 1.5):
        for angle in angles:
            candidates.append(goal_xy + np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float32) * float(radius))
    best = current_xy.copy()
    best_score = float("inf")
    for point in candidates:
        static_distance = sample_distance_map(scene.distance_map, point, scene)
        static_penalty = max(0.0, 0.25 - static_distance) * 10.0
        dyn_penalty = 0.0
        for clip in others_clips[:MAX_OTHERS]:
            fut = future_slice(clip.root_xy, sim_t, int(args.horizon))
            if len(fut):
                min_dist = float(np.min(np.linalg.norm(fut - point[None], axis=-1)))
                dyn_penalty += max(0.0, 0.5 - min_dist) * 5.0
        score = static_penalty + dyn_penalty + 0.1 * float(np.linalg.norm(point - goal_xy))
        if score < best_score:
            best_score = score
            best = point.astype(np.float32)
    return best


def move_towards(current_xy: np.ndarray, target_xy: np.ndarray, frames: int, speed_mps: float, dt: float) -> np.ndarray:
    out = np.zeros((frames, 2), dtype=np.float32)
    pos = current_xy.astype(np.float32).copy()
    step_max = float(speed_mps) * float(dt)
    for idx in range(frames):
        delta = target_xy - pos
        dist = float(np.linalg.norm(delta))
        if dist > 1e-6:
            pos = pos + delta / dist * min(step_max, dist)
        out[idx] = pos
    return out


def run_optimized_move_window(
    scene: SceneInfo,
    model: Stage1Predictor,
    opt_config: Stage1OptimizerV2Config,
    device: torch.device,
    args: argparse.Namespace,
    ego_history: list[np.ndarray],
    others_clips: list[CharacterClip],
    sim_t: int,
    target_xy: np.ndarray,
    segment_type: str,
    goal_type: Any,
    segments: list[dict[str, Any]],
    opt_debug: list[dict[str, float]],
    action_goal_xy: np.ndarray | None = None,
) -> tuple[int, float, bool]:
    batch, scene_crop, others_future_abs = build_predictor_batch(
        scene,
        np.asarray(ego_history, dtype=np.float32),
        others_clips,
        sim_t,
        target_xy,
        int(args.past_frames),
        int(args.future_frames),
        device,
    )
    with torch.no_grad():
        sample = model.sample(batch, num_steps=int(args.num_sampling_steps), deterministic=True)
    pred = sample["x0_hat"]
    pred_ego = pred[:, 0, : int(args.horizon)]
    pred_others = pred[:, 1:, : int(args.horizon)]
    others_valid = batch["entity_valid"][:, 1:]
    current_xy = np.asarray(ego_history[-1], dtype=np.float32)
    target_rel_xy = np.asarray(target_xy - current_xy, dtype=np.float32)
    goal_map_point, _ = clipped_goal_for_map(target_rel_xy)
    goal_map = make_gaussian_map(
        goal_map_point[None],
        MODEL_MAP_SIZE[0],
        MODEL_MAP_SIZE[1],
        LOCAL_RESOLUTION,
        LOCAL_ORIGIN,
        sigma=0.25,
    )
    static_fields = build_stage1_static_fields_v2(
        scene_crop[1],
        goal_map,
        opt_config,
        start_xy=(0.0, 0.0),
    )
    goal_distance = goal_distance_field_to_body_goal(target_rel_xy)
    hist = np.asarray(ego_history, dtype=np.float32)
    if len(hist) >= 2:
        current_vel = (hist[-1] - hist[-2]) / float(opt_config.dt)
    else:
        current_vel = np.zeros((2,), dtype=np.float32)
    opt_rel, metrics = optimize_stage1_trajectory_batch_v2(
        pred_ego=pred_ego,
        pred_others=pred_others,
        others_valid=others_valid,
        distance_fields=torch.from_numpy(scene_crop[1][None]).float().to(device),
        static_fields=torch.from_numpy(static_fields["final_static"][None]).float().to(device),
        goal_distance_fields=torch.from_numpy(goal_distance[None]).float().to(device),
        gt_ego=torch.zeros((1, int(args.horizon), 2), dtype=torch.float32, device=device),
        gt_others=torch.from_numpy((others_future_abs[:, : int(args.horizon)] - current_xy[None, None])[None]).float().to(device),
        current_vel_xy=torch.from_numpy(current_vel[None]).float().to(device),
        speed_bound=float(opt_config.speed_bound_mps),
        config=opt_config,
    )
    opt_debug.append(metrics)
    opt_abs = opt_rel[0].detach().cpu().numpy() + current_xy[None]
    execute = min(int(args.execute_frames), len(opt_abs))
    window_start = len(ego_history) - 1
    for point in opt_abs[:execute]:
        ego_history.append(point.astype(np.float32))
        sim_t += 1
    final_dist = float(np.linalg.norm(ego_history[-1] - target_xy))
    window_reached = final_dist <= float(args.arrival_radius_m)
    window_end = len(ego_history) - 1
    segment: dict[str, Any] = {
        "type": str(segment_type),
        "goal_type": goal_type,
        "start": int(window_start),
        "end": int(window_end),
        "reached_in_window": bool(window_reached),
        "final_distance_m": float(final_dist),
        "goal_xy": [float(target_xy[0]), float(target_xy[1])],
    }
    if action_goal_xy is not None:
        segment["action_goal_xy"] = [float(action_goal_xy[0]), float(action_goal_xy[1])]
    segments.append(segment)
    return sim_t, final_dist, window_reached


def append_wait_hold_segment(
    ego_history: list[np.ndarray],
    sim_t: int,
    frames: int,
    wait_point: np.ndarray,
    action_goal_xy: np.ndarray,
    goal_type: Any,
    segments: list[dict[str, Any]],
) -> tuple[int, int]:
    start = len(ego_history) - 1
    for _ in range(int(frames)):
        ego_history.append(wait_point.astype(np.float32))
        sim_t += 1
    end = len(ego_history) - 1
    if end > start:
        segments.append(
            {
                "type": "WAIT",
                "goal_type": goal_type,
                "start": int(start),
                "end": int(end),
                "goal_xy": [float(wait_point[0]), float(wait_point[1])],
                "action_goal_xy": [float(action_goal_xy[0]), float(action_goal_xy[1])],
            }
        )
    return sim_t, max(0, end - start)


def compute_collisions(ego_xy: np.ndarray, others_clips: list[CharacterClip], collision_distance: float) -> dict[str, float]:
    if len(ego_xy) == 0:
        return {"dynamic_collision_ratio": 0.0, "min_dynamic_distance_m": float("nan")}
    min_dists = []
    for t, ego in enumerate(ego_xy):
        best = float("inf")
        for clip in others_clips[:MAX_OTHERS]:
            other = future_slice(clip.root_xy, t, 1)[0]
            best = min(best, float(np.linalg.norm(ego - other)))
        min_dists.append(best)
    arr = np.asarray(min_dists, dtype=np.float32)
    finite = np.isfinite(arr)
    if not bool(finite.any()):
        return {"dynamic_collision_ratio": 0.0, "min_dynamic_distance_m": float("nan")}
    return {
        "dynamic_collision_ratio": float((arr[finite] < float(collision_distance)).mean()),
        "min_dynamic_distance_m": float(arr[finite].min()),
    }


def compute_static_metrics(ego_xy: np.ndarray, scene: SceneInfo) -> dict[str, float]:
    values = np.asarray([sample_distance_map(scene.distance_map, p, scene) for p in ego_xy], dtype=np.float32)
    if len(values) == 0:
        return {"static_collision_ratio": 0.0, "min_static_distance_m": float("nan")}
    return {
        "static_collision_ratio": float((values < 0.25).mean()),
        "min_static_distance_m": float(values.min()),
    }


def plot_episode_overview(
    path: Path,
    scene: SceneInfo,
    ego_xy: np.ndarray,
    others_xy: np.ndarray,
    goals_xy: list[np.ndarray],
    segments: list[dict[str, Any]],
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    clearance = scene.scene_maps[0]
    extent = [
        scene.origin_xy[0],
        scene.origin_xy[0] + clearance.shape[0] * scene.resolution,
        scene.origin_xy[1],
        scene.origin_xy[1] + clearance.shape[1] * scene.resolution,
    ]
    ax.imshow(np.flipud(clearance.T), origin="lower", extent=extent, cmap="gray", alpha=0.55, vmin=0, vmax=1)
    for idx in range(others_xy.shape[0]):
        ax.plot(others_xy[idx, :, 0], others_xy[idx, :, 1], linewidth=1.0, alpha=0.65, label=f"other{idx + 1}")
    ax.plot(ego_xy[:, 0], ego_xy[:, 1], color="red", linewidth=2.0, label="ego")
    for idx, goal in enumerate(goals_xy):
        ax.scatter([goal[0]], [goal[1]], marker="*", s=90, edgecolors="black", linewidths=0.5, label="goal" if idx == 0 else None)
        ax.text(float(goal[0]), float(goal[1]), str(idx), fontsize=8)
    for seg in segments:
        if seg.get("type") == "MOVE":
            color = "#e41a1c"
        elif seg.get("type") == "MOVE_TO_WAIT":
            color = "#984ea3"
        elif seg.get("type") == "MOVE_BACK_TO_GOAL":
            color = "#ff7f00"
        elif seg.get("type") == "ACTION":
            color = "#377eb8"
        else:
            color = "#4daf4a"
        start = int(seg["start"])
        end = min(int(seg["end"]), len(ego_xy) - 1)
        if end > start:
            ax.plot(ego_xy[start : end + 1, 0], ego_xy[start : end + 1, 1], color=color, linewidth=3.0, alpha=0.45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_window(
    path: Path,
    scene: SceneInfo,
    ego_xy: np.ndarray,
    others_xy: np.ndarray,
    goal_xy: np.ndarray,
    segment: dict[str, Any],
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = int(segment["start"])
    end = min(int(segment["end"]), len(ego_xy) - 1)
    prev_start = max(0, start - 60)
    pad_end = min(len(ego_xy) - 1, end + 30)
    fig, ax = plt.subplots(figsize=(6, 5))
    clearance = scene.scene_maps[0]
    extent = [
        scene.origin_xy[0],
        scene.origin_xy[0] + clearance.shape[0] * scene.resolution,
        scene.origin_xy[1],
        scene.origin_xy[1] + clearance.shape[1] * scene.resolution,
    ]
    ax.imshow(np.flipud(clearance.T), origin="lower", extent=extent, cmap="gray", alpha=0.50, vmin=0, vmax=1)
    if start > prev_start:
        ax.plot(ego_xy[prev_start : start + 1, 0], ego_xy[prev_start : start + 1, 1], color="#ff8a8a", linewidth=1.1, alpha=0.75, label="ours previous")
    ax.plot(ego_xy[start : end + 1, 0], ego_xy[start : end + 1, 1], color="red", linewidth=2.4, label=f"ours current {segment.get('type', '')}")
    for idx in range(others_xy.shape[0]):
        color = f"C{idx}"
        if start > prev_start:
            ax.plot(
                others_xy[idx, prev_start : start + 1, 0],
                others_xy[idx, prev_start : start + 1, 1],
                color=color,
                linewidth=0.9,
                alpha=0.45,
                linestyle="-",
                label=f"other{idx + 1} previous",
            )
        ax.plot(
            others_xy[idx, start : end + 1, 0],
            others_xy[idx, start : end + 1, 1],
            color=color,
            linewidth=2.0,
            alpha=0.80,
            linestyle="-",
            label=f"other{idx + 1} current",
        )
        ax.scatter([others_xy[idx, start, 0]], [others_xy[idx, start, 1]], color=color, s=12, marker="o", zorder=5)
        ax.scatter([others_xy[idx, end, 0]], [others_xy[idx, end, 1]], color=color, s=18, marker="^", zorder=5)
    ax.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=70, c="#ffd400", edgecolors="black", linewidths=0.5, zorder=5, label="goal")
    if "action_goal_xy" in segment:
        action_goal = np.asarray(segment["action_goal_xy"], dtype=np.float32)
        if float(np.linalg.norm(action_goal - goal_xy)) > 1e-4:
            ax.scatter(
                [action_goal[0]],
                [action_goal[1]],
                marker="P",
                s=48,
                c="#00bcd4",
                edgecolors="black",
                linewidths=0.4,
                zorder=5,
                label="action goal",
            )
    if start < len(ego_xy):
        ax.scatter([ego_xy[start, 0]], [ego_xy[start, 1]], c="red", s=16, marker="o", zorder=6, label="ours start")
    if end < len(ego_xy):
        ax.scatter([ego_xy[end, 0]], [ego_xy[end, 1]], c="red", s=22, marker="x", zorder=6, label="ours end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}\nthin=previous, thick=current window")
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def simulate_episode(
    episode_path: Path,
    args: argparse.Namespace,
    model: Stage1Predictor,
    opt_config: Stage1OptimizerV2Config,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    episode = load_json(episode_path)
    scene_id = str(episode["scene_id"])
    scene = load_scene_info(args.dataset, scene_id)
    chars = episode["character_assignments"]
    ego_id = str(episode.get("ego_character_id") or "char_00")
    ego_char = next(ch for ch in chars if str(ch.get("character_id")) == ego_id)
    other_chars = [ch for ch in chars if str(ch.get("character_id")) != ego_id]
    ego_clip = resolve_character_clip(args.dataset, scene_id, ego_char)
    others_clips = [resolve_character_clip(args.dataset, scene_id, ch) for ch in other_chars[:MAX_OTHERS]]
    queue = list(ego_clip.goals)

    ego_history = [ego_clip.root_xy[0].astype(np.float32)]
    sim_t = 0
    completed = 0
    failed = 0
    timeouts = 0
    waits = 0
    actions_interrupted = 0
    per_goal: list[dict[str, Any]] = []
    opt_debug: list[dict[str, float]] = []
    segments: list[dict[str, Any]] = []

    while queue and sim_t < int(args.max_frames_per_goal) * max(1, len(ego_clip.goals)) + 500:
        goal = queue.pop(0)
        goal_xy = body_goal_xy(goal)
        if goal_xy is None:
            failed += 1
            per_goal.append({"goal_type": goal.get("goal_type"), "status": "failed_no_body_goal"})
            continue
        goal_type = normalize_goal_type(goal.get("goal_type"))
        goal_start_t = sim_t
        status = "unknown"
        should_move_to_goal = goal_type == "move" or float(np.linalg.norm(ego_history[-1] - goal_xy)) >= float(args.action_direct_threshold_m)
        if should_move_to_goal:
            while sim_t - goal_start_t < int(args.max_frames_per_goal):
                current = ego_history[-1]
                if float(np.linalg.norm(current - goal_xy)) <= float(args.arrival_radius_m):
                    status = "move_reached"
                    break
                sim_t, final_dist, window_reached = run_optimized_move_window(
                    scene,
                    model,
                    opt_config,
                    device,
                    args,
                    ego_history,
                    others_clips,
                    sim_t,
                    goal_xy,
                    "MOVE",
                    goal.get("goal_type"),
                    segments,
                    opt_debug,
                )
                if window_reached:
                    status = "move_reached"
                    break
            if status != "move_reached":
                failed += 1
                timeouts += 1
                status = "timeout"
        if goal_type == "move":
            if status == "move_reached":
                completed += 1
                status = "completed"
        elif status not in {"timeout"}:
            held = 0
            action_chunk_start = len(ego_history) - 1
            action_failed = False
            while held < int(args.action_hold_frames):
                current = ego_history[-1]
                if action_intrusion_likely(goal_xy, others_clips, sim_t, int(args.horizon), float(args.interrupt_distance_m)):
                    action_chunk_end = len(ego_history) - 1
                    if action_chunk_end > action_chunk_start:
                        segments.append(
                            {
                                "type": "ACTION",
                                "goal_type": goal.get("goal_type"),
                                "start": int(action_chunk_start),
                                "end": int(action_chunk_end),
                                "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
                            }
                        )
                    actions_interrupted += 1
                    wait_point = choose_wait_point(scene, current, goal_xy, others_clips, sim_t, args)
                    move_wait_start = sim_t
                    while (
                        float(np.linalg.norm(ego_history[-1] - wait_point)) > float(args.arrival_radius_m)
                        and sim_t - move_wait_start < int(args.max_frames_per_goal)
                    ):
                        phase_start_t = sim_t
                        sim_t, _, reached_wait = run_optimized_move_window(
                            scene,
                            model,
                            opt_config,
                            device,
                            args,
                            ego_history,
                            others_clips,
                            sim_t,
                            wait_point,
                            "MOVE_TO_WAIT",
                            goal.get("goal_type"),
                            segments,
                            opt_debug,
                            action_goal_xy=goal_xy,
                        )
                        waits += int(sim_t - phase_start_t)
                        if reached_wait:
                            break
                    if float(np.linalg.norm(ego_history[-1] - wait_point)) > float(args.arrival_radius_m):
                        failed += 1
                        timeouts += 1
                        status = "wait_move_timeout"
                        action_failed = True
                        break

                    waited = 0
                    while action_intrusion_likely(goal_xy, others_clips, sim_t, int(args.horizon), float(args.wait_release_distance_m)):
                        if waited >= int(args.max_wait_frames):
                            break
                        chunk = min(int(args.wait_hold_frames), int(args.max_wait_frames) - waited)
                        sim_t, advanced = append_wait_hold_segment(
                            ego_history,
                            sim_t,
                            chunk,
                            wait_point,
                            goal_xy,
                            goal.get("goal_type"),
                            segments,
                        )
                        waited += int(advanced)
                        waits += int(advanced)
                    if action_intrusion_likely(goal_xy, others_clips, sim_t, int(args.horizon), float(args.wait_release_distance_m)):
                        failed += 1
                        timeouts += 1
                        status = "wait_timeout"
                        action_failed = True
                        break

                    move_back_start = sim_t
                    while (
                        float(np.linalg.norm(ego_history[-1] - goal_xy)) > float(args.arrival_radius_m)
                        and sim_t - move_back_start < int(args.max_frames_per_goal)
                    ):
                        phase_start_t = sim_t
                        sim_t, _, reached_goal = run_optimized_move_window(
                            scene,
                            model,
                            opt_config,
                            device,
                            args,
                            ego_history,
                            others_clips,
                            sim_t,
                            goal_xy,
                            "MOVE_BACK_TO_GOAL",
                            goal.get("goal_type"),
                            segments,
                            opt_debug,
                            action_goal_xy=goal_xy,
                        )
                        waits += int(sim_t - phase_start_t)
                        if reached_goal:
                            break
                    if float(np.linalg.norm(ego_history[-1] - goal_xy)) > float(args.arrival_radius_m):
                        failed += 1
                        timeouts += 1
                        status = "wait_return_timeout"
                        action_failed = True
                        break
                    action_chunk_start = len(ego_history) - 1
                    continue
                ego_history.append(current.astype(np.float32))
                sim_t += 1
                held += 1
                if held % int(args.execute_frames) == 0 or held >= int(args.action_hold_frames):
                    segments.append(
                        {
                            "type": "ACTION",
                            "goal_type": goal.get("goal_type"),
                            "start": int(action_chunk_start),
                            "end": int(len(ego_history) - 1),
                            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
                        }
                    )
                    action_chunk_start = len(ego_history) - 1
            if not action_failed:
                completed += 1
                status = "completed"
        per_goal.append(
            {
                "goal_type": goal.get("goal_type"),
                "normalized_goal_type": goal_type,
                "status": status,
                "start_frame": int(goal_start_t),
                "end_frame": int(sim_t),
                "duration": int(sim_t - goal_start_t),
                "body_goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
                "final_distance_m": float(np.linalg.norm(ego_history[-1] - goal_xy)),
            }
        )

    ego_arr = np.asarray(ego_history, dtype=np.float32)
    dyn_metrics = compute_collisions(ego_arr, others_clips, float(args.collision_distance_m))
    static_metrics = compute_static_metrics(ego_arr, scene)
    total_goals = len(ego_clip.goals)
    result = {
        "episode_path": display_path(episode_path),
        "scene_id": scene_id,
        "episode_id": episode.get("episode_id", episode_path.stem),
        "total_goals": int(total_goals),
        "completed_goals": int(completed),
        "failed_goals": int(failed),
        "goal_success_rate": float(completed / max(total_goals, 1)),
        "timeout_count": int(timeouts),
        "wait_frames": int(waits),
        "action_interrupt_count": int(actions_interrupted),
        "simulated_frames": int(len(ego_arr)),
        "per_goal": per_goal,
        "segments": segments,
        **dyn_metrics,
        **static_metrics,
    }
    others_np = np.zeros((len(others_clips), len(ego_arr), 2), dtype=np.float32)
    for idx, clip in enumerate(others_clips):
        for t in range(len(ego_arr)):
            others_np[idx, t] = future_slice(clip.root_xy, t, 1)[0]
    if args.save_trajectories:
        traj_path = out_dir / "trajectories" / scene_id / f"{episode_path.stem}.npz"
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(traj_path, ego_xy=ego_arr, others_xy=others_np, result=json.dumps(result))
        result["trajectory_path"] = display_path(traj_path)
    if args.save_plots:
        plots_root = out_dir / "plots" / scene_id / episode_path.stem
        goals_xy = [body_goal_xy(goal) for goal in ego_clip.goals]
        goals_xy = [goal for goal in goals_xy if goal is not None]
        plot_episode_overview(
            plots_root / "overview.png",
            scene,
            ego_arr,
            others_np,
            goals_xy,
            segments,
            f"{episode.get('episode_id', episode_path.stem)}",
        )
        for idx, seg in enumerate(segments[: int(args.max_window_plots_per_episode)]):
            goal_np = np.asarray(seg["goal_xy"], dtype=np.float32)
            plot_window(
                plots_root / f"window_{idx:03d}_{seg.get('type', 'seg').lower()}.png",
                scene,
                ego_arr,
                others_np,
                goal_np,
                seg,
                f"{episode_path.stem} {idx:03d} {seg.get('type')} {seg.get('goal_type')}",
            )
        result["plot_dir"] = display_path(plots_root)
    return result


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    total_goals = sum(int(r["total_goals"]) for r in results)
    completed = sum(int(r["completed_goals"]) for r in results)
    return {
        "num_episodes": len(results),
        "total_goals": total_goals,
        "completed_goals": completed,
        "goal_success_rate": float(completed / max(total_goals, 1)),
        "episode_full_success_rate": float(np.mean([int(r["completed_goals"]) == int(r["total_goals"]) for r in results])),
        "mean_completed_goals": float(np.mean([r["completed_goals"] for r in results])),
        "timeout_count": int(sum(int(r["timeout_count"]) for r in results)),
        "action_interrupt_count": int(sum(int(r["action_interrupt_count"]) for r in results)),
        "mean_dynamic_collision_ratio": float(np.mean([r["dynamic_collision_ratio"] for r in results])),
        "mean_static_collision_ratio": float(np.mean([r["static_collision_ratio"] for r in results])),
        "min_dynamic_distance_m": float(np.nanmin([r["min_dynamic_distance_m"] for r in results])),
        "min_static_distance_m": float(np.nanmin([r["min_static_distance_m"] for r in results])),
    }


def main() -> None:
    args = parse_args()
    episodes_dir = args.episodes_dir or (DEFAULT_ROOT / args.dataset / "episodes_v3")
    run_name = args.run_name or f"stage1_loop_{args.dataset}_001"
    out_dir = args.output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    episode_paths = [repo_path(path) for path in sample_episode_set(args, episodes_dir)]
    if args.max_episodes is not None:
        episode_paths = episode_paths[: int(args.max_episodes)]

    model = build_model(args.checkpoint, int(args.past_frames), int(args.future_frames), device)
    opt_config = load_optimizer_config(args.optimizer_config, args)

    results: list[dict[str, Any]] = []
    jsonl_path = out_dir / "per_episode_metrics.jsonl"
    with jsonl_path.open("w") as handle:
        for episode_path in tqdm(episode_paths, desc="stage1 episode loop", unit="episode"):
            try:
                result = simulate_episode(episode_path, args, model, opt_config, device, out_dir)
            except Exception as exc:
                result = {
                    "episode_path": display_path(episode_path),
                    "error": repr(exc),
                    "total_goals": 0,
                    "completed_goals": 0,
                    "failed_goals": 0,
                    "goal_success_rate": 0.0,
                    "timeout_count": 0,
                    "wait_frames": 0,
                    "action_interrupt_count": 0,
                    "simulated_frames": 0,
                    "dynamic_collision_ratio": 0.0,
                    "static_collision_ratio": 0.0,
                    "min_dynamic_distance_m": float("nan"),
                    "min_static_distance_m": float("nan"),
                }
            results.append(result)
            handle.write(json.dumps(result) + "\n")
            handle.flush()
    summary = aggregate([r for r in results if "error" not in r])
    summary["num_errors"] = sum(1 for r in results if "error" in r)
    summary["run_name"] = run_name
    summary["dataset"] = args.dataset
    summary["checkpoint"] = str(args.checkpoint)
    summary["optimizer_config"] = str(args.optimizer_config)
    write_json(out_dir / "metrics.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
