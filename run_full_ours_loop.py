from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.full_ours_runtime import (
    EXECUTE_FRAMES,
    MOVEWAIT_FRAMES,
    OVERLAP_FRAMES,
    STAGE2_HISTORY_FRAMES,
    Stage1Runtime,
    Stage2Runtime,
    WorldState,
    blend_overlap,
    body_goal_xyz,
    clamp_plan_after_arrival,
    display_path,
    fit_joints28_episode_to_smplx,
    goal_duration_frames,
    hand_goal_xyz,
    joints_to_root_xz,
    normalize_goal_type,
    repo_path,
    xyz_from_xz,
)
from simulate_stage1_episode_loop import (
    MAX_OTHERS,
    action_intrusion_likely,
    body_goal_xy,
    choose_wait_point,
    compute_collisions,
    compute_static_metrics,
    future_slice,
    load_json,
    load_scene_info,
    plot_episode_overview,
    plot_window,
    resolve_character_clip,
    sample_episode_set,
)
from vis.episode_blender_common import (
    DEFAULT_BLENDER,
    probe_video_size,
    render_side_by_side_mp4,
    resolve_lingo_scene_obj,
    resolve_trumans_scene_blend,
    to_abs,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = Path("data") / "preprocessed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full ours Stage1+Stage1.5+Stage2 closed-loop generation.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--episodes-dir", type=Path, default=None)
    parser.add_argument("--sample-set-path", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--episodes-per-scene", type=int, default=5)
    parser.add_argument("--episode-seed", type=int, default=2026)
    parser.add_argument("--max-episodes", type=int, default=None)

    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-optimizer-config", type=Path, required=True)
    parser.add_argument("--stage2-movewait-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-action-checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--stage1-sampling-steps", type=int, default=50)
    parser.add_argument("--stage2-sampling-steps", type=int, default=100)
    parser.add_argument("--past-frames", type=int, default=30)
    parser.add_argument("--future-frames", type=int, default=72)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--execute-frames", type=int, default=EXECUTE_FRAMES)
    parser.add_argument("--arrival-radius-m", type=float, default=0.20)
    parser.add_argument("--action-direct-threshold-m", type=float, default=0.50)
    parser.add_argument("--max-frames-per-goal", type=int, default=300)
    parser.add_argument("--max-action-frames", type=int, default=300)
    parser.add_argument("--interrupt-distance-m", type=float, default=0.25)
    parser.add_argument("--wait-release-distance-m", type=float, default=0.80)
    parser.add_argument("--wait-ring-radius-m", type=float, default=0.80)
    parser.add_argument("--wait-hold-frames", type=int, default=30)
    parser.add_argument("--max-wait-frames", type=int, default=180)
    parser.add_argument("--collision-distance-m", type=float, default=0.50)
    parser.add_argument("--nb-voxels", type=int, default=32)
    parser.add_argument("--blend-overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-window-plots-per-episode", type=int, default=24)
    parser.add_argument("--save-root-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--root-video-fps", type=int, default=30)
    parser.add_argument("--root-video-history-frames", type=int, default=30)
    parser.add_argument("--render-blender", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-blender-videos", type=int, default=5, help="Maximum Blender videos per run. Use 0 for no limit.")
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--blender-render-mode", default="preview", choices=["preview", "final"])
    parser.add_argument("--blender-camera-mode", default=None)
    parser.add_argument("--blender-camera-scale", type=float, default=1.0)
    parser.add_argument("--blender-frame-limit", type=int, default=None)
    parser.add_argument("--show-targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fit-smplx", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fit-smooth-weight", type=float, default=0.0)
    return parser.parse_args()


def output_root(args: argparse.Namespace) -> Path:
    return args.output_root / str(args.run_name)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def character_global_start(dataset: str, scene_id: str, char: dict[str, Any]) -> tuple[int, int, int]:
    if isinstance(char.get("source_window"), dict):
        window = char["source_window"]
        sequence_id = str(window["sequence_id"])
        local_start = int(window["local_start"])
        local_end = int(window["local_end"])
    else:
        goals = char.get("goal_sequence") or []
        first_source = goals[0].get("source_segment", {})
        last_source = goals[-1].get("source_segment", {})
        sequence_id = str(char.get("sequence_id") or first_source["sequence_id"])
        local_start = int(first_source["start"])
        local_end = int(last_source["end"])
    seq = load_json(DEFAULT_ROOT / dataset / "scenes_v2" / scene_id / "sequences" / f"{sequence_id}.json")
    seq_global_start = int(seq["human_motion_ref"]["start"])
    return seq_global_start + local_start, seq_global_start + local_end, seq_global_start


def character_sequence_id(char: dict[str, Any]) -> str:
    if isinstance(char.get("source_window"), dict):
        return str(char["source_window"]["sequence_id"])
    goals = char.get("goal_sequence") or []
    if not goals:
        raise ValueError(f"Character {char.get('character_id')} has no goal_sequence.")
    source = goals[0].get("source_segment", {})
    if "sequence_id" not in source:
        raise ValueError(f"Character {char.get('character_id')} has no source sequence.")
    return str(source["sequence_id"])


def initial_gt_history(dataset: str, scene_id: str, ego_char: dict[str, Any], history_frames: int) -> tuple[np.ndarray, int]:
    joints_path = DEFAULT_ROOT / dataset / "joints28" / "joints28.npy"
    joints = np.load(repo_path(joints_path), mmap_mode="r")
    start, end, _ = character_global_start(dataset, scene_id, ego_char)
    indices = np.arange(start, start + int(history_frames), dtype=np.int64)
    indices = np.clip(indices, start, min(end, joints.shape[0] - 1))
    return np.asarray(joints[indices], dtype=np.float32), int(len(indices) - 1)


def gt_others_joints(dataset: str, scene_id: str, other_chars: list[dict[str, Any]], frames: int) -> np.ndarray:
    joints_path = DEFAULT_ROOT / dataset / "joints28" / "joints28.npy"
    joints = np.load(repo_path(joints_path), mmap_mode="r")
    out = []
    for char in other_chars[:MAX_OTHERS]:
        start, end, _ = character_global_start(dataset, scene_id, char)
        idx = np.arange(start, start + int(frames), dtype=np.int64)
        idx = np.clip(idx, start, min(end, joints.shape[0] - 1))
        out.append(np.asarray(joints[idx], dtype=np.float32))
    if not out:
        return np.zeros((0, int(frames), 28, 3), dtype=np.float32)
    return np.stack(out, axis=0).astype(np.float32)


def ensure_plan_len(plan_xy: np.ndarray, target_len: int = MOVEWAIT_FRAMES) -> np.ndarray:
    values = np.asarray(plan_xy, dtype=np.float32)
    if len(values) >= int(target_len):
        return values[: int(target_len)]
    if len(values) == 0:
        return np.zeros((int(target_len), 2), dtype=np.float32)
    pad = np.repeat(values[-1:], int(target_len) - len(values), axis=0)
    return np.concatenate([values, pad], axis=0)


def root_plan_xyz(plan_xy: np.ndarray, root_y: float) -> np.ndarray:
    plan = np.asarray(plan_xy, dtype=np.float32)
    y = np.full((len(plan), 1), float(root_y), dtype=np.float32)
    return np.concatenate([plan[:, :1], y, plan[:, 1:2]], axis=1).astype(np.float32)


def root_tracking_metrics(reference_xy: np.ndarray, generated_joints: np.ndarray) -> dict[str, Any]:
    generated_xy = joints_to_root_xz(generated_joints)
    reference = np.asarray(reference_xy, dtype=np.float32)[: len(generated_xy)]
    if len(generated_xy) == 0 or len(reference) == 0:
        return {
            "stage2_generated_root_15": generated_xy.tolist(),
            "stage2_vs_stage1_root_rmse_mm": None,
            "stage2_vs_stage1_root_fde_mm": None,
            "stage2_vs_stage1_root_max_mm": None,
        }
    n = min(len(reference), len(generated_xy))
    diff = generated_xy[:n] - reference[:n]
    dist = np.linalg.norm(diff, axis=-1)
    return {
        "stage2_generated_root_15": generated_xy.tolist(),
        "stage2_vs_stage1_root_rmse_mm": float(np.sqrt(np.mean(np.square(dist))) * 1000.0),
        "stage2_vs_stage1_root_fde_mm": float(dist[-1] * 1000.0),
        "stage2_vs_stage1_root_max_mm": float(np.max(dist) * 1000.0),
    }


def pad_motion_array(values: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) >= int(target_len):
        return arr[: int(target_len)].copy()
    if len(arr) == 0:
        raise ValueError("Cannot pad empty motion array.")
    pad = np.repeat(arr[-1:], int(target_len) - len(arr), axis=0)
    return np.concatenate([arr, pad], axis=0).astype(np.float32)


def export_gt_character_smplx_v2(
    dataset: str,
    scene_id: str,
    char: dict[str, Any],
    num_frames: int,
    out_path: Path,
) -> int:
    sequence_id = character_sequence_id(char)
    seq = load_json(DEFAULT_ROOT / dataset / "scenes_v2" / scene_id / "sequences" / f"{sequence_id}.json")
    smpl = seq["human_motion_ref"]["smplx"]
    global_start, global_end, _ = character_global_start(dataset, scene_id, char)
    end = min(int(global_end) + 1, int(global_start) + int(num_frames))

    payload: dict[str, np.ndarray] = {}
    for key in ("global_orient", "body_pose", "transl", "left_hand_pose", "right_hand_pose"):
        path_key = f"{key}_path"
        if path_key not in smpl:
            continue
        raw = np.load(repo_path(smpl[path_key]), mmap_mode="r")
        payload[key] = pad_motion_array(raw[int(global_start) : end], int(num_frames))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return int(num_frames)


def goal_overlay_position(goal: dict[str, Any]) -> tuple[list[float] | None, str | None]:
    body_goal = goal.get("body_goal")
    hand_goal = goal.get("hand_goal")
    goal_type = str(goal.get("goal_type") or "")
    prefer_hand = goal_type in {"open", "close", "pick_up", "put_down", "type", "wipe", "drink", "answer", "play"}
    if prefer_hand and isinstance(hand_goal, list) and len(hand_goal) == 3:
        return hand_goal, "hand"
    if isinstance(body_goal, list) and len(body_goal) == 3:
        return body_goal, "body"
    if isinstance(hand_goal, list) and len(hand_goal) == 3:
        return hand_goal, "hand"
    return None, None


def build_target_overlays_for_full_loop(episode: dict[str, Any], char_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    color_by_char = {str(item["character_id"]): str(item["color"]) for item in char_records}
    overlays: list[dict[str, Any]] = []
    for char_idx, char in enumerate(episode.get("character_assignments", [])):
        char_id = str(char.get("character_id", f"char_{char_idx:02d}"))
        for step_idx, goal in enumerate(char.get("goal_sequence", [])):
            position, position_kind = goal_overlay_position(goal)
            if position is None:
                continue
            text = str((goal.get("source_segment") or {}).get("text") or goal.get("goal_type") or "goal")
            overlays.append(
                {
                    "character_id": char_id,
                    "character_index": int(char_idx),
                    "step_index": int(step_idx),
                    "position": [float(position[0]), float(position[1]), float(position[2])],
                    "position_kind": position_kind,
                    "label": f"Char {char_idx}-{step_idx}: {text}",
                    "color": color_by_char.get(char_id),
                    "goal_type": goal.get("goal_type"),
                    "active_frame_start": 0,
                    "active_frame_end": 10**9,
                }
            )
    return overlays


def build_full_loop_blender_meta(
    args: argparse.Namespace,
    episode: dict[str, Any],
    scene_id: str,
    ego_id: str,
    chars: list[dict[str, Any]],
    ego_smplx_path: Path,
    num_frames: int,
    blender_dir: Path,
) -> Path:
    char_records: list[dict[str, Any]] = []
    for char_idx, char in enumerate(chars):
        char_id = str(char.get("character_id", f"char_{char_idx:02d}"))
        if char_id == ego_id:
            motion_pkl = ego_smplx_path
            color = "220,38,38"
            source_kind = "generated_full_loop"
        else:
            motion_pkl = blender_dir / f"{char_id}_gt_smplx_results.pkl"
            export_gt_character_smplx_v2(args.dataset, scene_id, char, num_frames, motion_pkl)
            color = "150,150,150"
            source_kind = "gt_episode_v2"
        char_records.append(
            {
                "character_id": char_id,
                "object_name": f"SMPLX-mesh-male_{char_id}",
                "sequence_id": character_sequence_id(char),
                "local_start": 0,
                "local_end": int(num_frames) - 1,
                "num_frames": int(num_frames),
                "motion_pkl": str(motion_pkl),
                "color": color,
                "load_hand": bool(args.dataset == "trumans"),
                "source_kind": source_kind,
            }
        )

    if args.dataset == "trumans":
        scene_asset = resolve_trumans_scene_blend(scene_id, None, Path("data/raw/trumans/Recordings_blend"))
        meta = {"scene_blend": str(scene_asset)}
    else:
        scene_asset, resolved_scene_mesh_id = resolve_lingo_scene_obj(scene_id, None, Path("data/raw/lingo/Scene_mesh"))
        meta = {"scene_obj": str(scene_asset), "resolved_scene_mesh_id": resolved_scene_mesh_id}
    meta.update(
        {
            "dataset": args.dataset,
            "scene_id": scene_id,
            "episode_id": str(episode.get("episode_id", "")),
            "smplx_gender": "male",
            "scenario_type": episode.get("scenario_type"),
            "num_characters": len(char_records),
            "characters": char_records,
            "target_overlays": build_target_overlays_for_full_loop(episode, char_records),
        }
    )
    meta_path = blender_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


def render_full_loop_blender_video(
    args: argparse.Namespace,
    scene_id: str,
    meta_path: Path,
    output_mp4: Path,
) -> None:
    script = PROJECT_ROOT / "vis" / ("blender_apply_trumans_episode.py" if args.dataset == "trumans" else "blender_apply_lingo_episode.py")
    meta = load_json(meta_path)
    camera_modes = [str(args.blender_camera_mode)] if args.blender_camera_mode is not None else ["oblique", "topdown"]
    render_paths: list[Path] = []
    work_dir = meta_path.parent
    for camera_mode in camera_modes:
        render_path = output_mp4 if len(camera_modes) == 1 else (work_dir / f"render_{camera_mode}.mp4")
        if args.dataset == "trumans":
            cmd = [
                str(to_abs(args.blender_bin)),
                "-b",
                str(Path(meta["scene_blend"])),
                "--python",
                str(script),
                "--",
            ]
        else:
            cmd = [
                str(to_abs(args.blender_bin)),
                "-b",
                "--factory-startup",
                "--python",
                str(script),
                "--",
            ]
        cmd.extend(
            [
                "--meta-json",
                str(meta_path),
                "--render-mode",
                str(args.blender_render_mode),
                "--camera-mode",
                str(camera_mode),
                "--camera-scale",
                str(float(args.blender_camera_scale)),
                "--fps",
                "30",
                "--render-mp4",
                str(render_path),
            ]
        )
        if args.blender_frame_limit is not None:
            cmd.extend(["--frame-limit", str(int(args.blender_frame_limit))])
        if bool(args.show_targets):
            cmd.append("--show-targets")
        if args.blender_render_mode == "preview":
            cmd.append("--bright-preview")
        if args.dataset == "trumans":
            cmd.append("--clear-existing-characters")
        subprocess.run(cmd, check=True)
        render_paths.append(render_path)

    if len(render_paths) > 1:
        render_side_by_side_mp4(render_paths, output_mp4, text_overlays=None, dry_run=False)


def render_full_loop_final_video(blender_video_path: Path, root_video_path: Path, output_path: Path) -> None:
    _, target_height = probe_video_size(blender_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_complex = (
        f"[0:v]setsar=1[blender];"
        f"[1:v]scale=-2:{int(target_height)},setsar=1[root];"
        f"[blender][root]hstack=inputs=2[out]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(blender_video_path),
        "-i",
        str(root_video_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def active_segment_at_frame(segments: list[dict[str, Any]], frame_idx: int) -> dict[str, Any] | None:
    for segment in segments:
        if int(segment.get("start", 0)) <= int(frame_idx) <= int(segment.get("end", 0)):
            return segment
    return None


def active_goal_xy_at_frame(segments: list[dict[str, Any]], frame_idx: int) -> np.ndarray | None:
    segment = active_segment_at_frame(segments, frame_idx)
    if segment is None or "goal_xy" not in segment:
        return None
    return np.asarray(segment["goal_xy"], dtype=np.float32)


def render_root_video_frame(
    scene: Any,
    ego_root: np.ndarray,
    others_root: np.ndarray,
    segments: list[dict[str, Any]],
    frame_idx: int,
    history_frames: int,
    title: str,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(7, 6))
    clearance = scene.scene_maps[0]
    extent = [
        scene.origin_xy[0],
        scene.origin_xy[0] + clearance.shape[0] * scene.resolution,
        scene.origin_xy[1],
        scene.origin_xy[1] + clearance.shape[1] * scene.resolution,
    ]
    ax.imshow(np.flipud(clearance.T), origin="lower", extent=extent, cmap="gray", alpha=0.55, vmin=0, vmax=1)

    start = max(0, int(frame_idx) - int(history_frames))
    end = int(frame_idx) + 1
    for idx in range(others_root.shape[0]):
        other = others_root[idx]
        color = "#8a8a8a"
        ax.plot(other[start:end, 0], other[start:end, 1], color=color, linewidth=1.1, alpha=0.55)
        ax.scatter([other[frame_idx, 0]], [other[frame_idx, 1]], color=color, s=18, marker="o", zorder=5)

    ax.plot(ego_root[start:end, 0], ego_root[start:end, 1], color="#ff8a8a", linewidth=1.2, alpha=0.75)
    ax.scatter([ego_root[frame_idx, 0]], [ego_root[frame_idx, 1]], color="red", s=28, marker="o", zorder=6)

    segment = active_segment_at_frame(segments, frame_idx)
    if segment is not None:
        seg_start = max(start, int(segment.get("start", frame_idx)))
        seg_end = min(end, int(segment.get("end", frame_idx)) + 1)
        if seg_end > seg_start:
            ax.plot(ego_root[seg_start:seg_end, 0], ego_root[seg_start:seg_end, 1], color="red", linewidth=3.0, alpha=0.95)
        goal = active_goal_xy_at_frame(segments, frame_idx)
        if goal is not None:
            ax.scatter([goal[0]], [goal[1]], marker="*", s=80, c="#ffd400", edgecolors="black", linewidths=0.5, zorder=7)
        phase = str(segment.get("type", ""))
        goal_type = str(segment.get("goal_type", ""))
    else:
        phase = "NONE"
        goal_type = ""

    ax.set_aspect("equal", adjustable="box")
    x0, x1 = extent[0], extent[1]
    z0, z1 = extent[2], extent[3]
    ax.set_xlim(x0, x1)
    ax.set_ylim(z1, z0)
    ax.set_title(f"{title}\nframe {frame_idx:04d} | {phase} {goal_type} | previous {history_frames} frames")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def save_root_debug_video(
    path: Path,
    scene: Any,
    ego_root: np.ndarray,
    others_root: np.ndarray,
    segments: list[dict[str, Any]],
    title: str,
    fps: int,
    history_frames: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        render_root_video_frame(scene, ego_root, others_root, segments, frame_idx, history_frames, title)
        for frame_idx in range(len(ego_root))
    ]
    imageio.mimsave(path, frames, fps=int(fps))


def append_segment(
    segments: list[dict[str, Any]],
    segment_type: str,
    goal: dict[str, Any],
    start: int,
    end: int,
    goal_xy: np.ndarray,
    final_distance: float | None = None,
    action_goal_xy: np.ndarray | None = None,
) -> int:
    record: dict[str, Any] = {
        "type": segment_type,
        "goal_type": goal.get("goal_type"),
        "start": int(start),
        "end": int(end),
        "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
    }
    if final_distance is not None:
        record["final_distance_m"] = float(final_distance)
        record["reached_in_window"] = bool(float(final_distance) <= 0.20)
    if action_goal_xy is not None:
        record["action_goal_xy"] = [float(action_goal_xy[0]), float(action_goal_xy[1])]
    segments.append(record)
    return len(segments) - 1


def execute_move_window(
    state: WorldState,
    stage1: Stage1Runtime,
    stage2: Stage2Runtime,
    scene: Any,
    scene_id: str,
    others_clips: list[Any],
    target_xy: np.ndarray,
    goal: dict[str, Any],
    reason: str,
    goal_index: int,
    segments: list[dict[str, Any]],
    stage1_records: list[dict[str, Any]],
    args: argparse.Namespace,
    action_goal_xy: np.ndarray | None = None,
) -> tuple[float, bool]:
    start_frame = len(state.ego_joints_world) - 1
    stage1_plan = stage1.plan(scene, state.ego_root_world, others_clips, state.sim_t, target_xy)
    plan21 = clamp_plan_after_arrival(
        ensure_plan_len(stage1_plan.planned_root_world_30, MOVEWAIT_FRAMES),
        target_xy,
        float(args.arrival_radius_m),
    )
    body_xyz = body_goal_xyz(goal, state.current_root_y)
    if body_xyz is None:
        body_xyz = xyz_from_xz(target_xy, state.current_root_y)
    gen = stage2.generate_move_wait(
        scene_id=scene_id,
        history_world=state.history_joints(),
        root_plan_world=root_plan_xyz(plan21, state.current_root_y),
        body_goal_world=body_xyz,
        goal_type=str(goal.get("goal_type", reason)),
    )
    blended = blend_overlap(state.previous_tail, gen.joints_world, OVERLAP_FRAMES) if bool(args.blend_overlap) else gen.joints_world.copy()
    execute = blended[: int(args.execute_frames)]
    segment_id = append_segment(
        segments,
        reason,
        goal,
        start_frame,
        start_frame + len(execute),
        target_xy,
        final_distance=float(np.linalg.norm(joints_to_root_xz(execute)[-1] - target_xy)),
        action_goal_xy=action_goal_xy,
    )
    state.append_window(execute, reason, goal_index, segment_id)
    state.previous_tail = blended[int(args.execute_frames) : int(args.execute_frames) + OVERLAP_FRAMES].copy()
    final_dist = float(np.linalg.norm(state.current_root_xy - target_xy))
    tracking = root_tracking_metrics(plan21, execute)
    stage1_records.append(
        {
            "frame": int(start_frame),
            "reason": reason,
            "target_xy": [float(target_xy[0]), float(target_xy[1])],
            "anchor_frame": "history_last",
            "yaw_source": "pelvis_hips_rigid",
            "yaw_hip_rad": float(gen.anchor_yaw),
            "stage2_anchor_root": gen.anchor_root.tolist(),
            "stage2_root_plan_world_21": plan21.tolist(),
            "stage1_plan_root_30": stage1_plan.planned_root_world_30.tolist(),
            "planned_root_world_30": stage1_plan.planned_root_world_30.tolist(),
            "pred_ego_world_30": stage1_plan.pred_ego_world_30.tolist(),
            "pred_others_world_30": stage1_plan.pred_others_world_30.tolist(),
            "debug": stage1_plan.debug,
            **tracking,
        }
    )
    return final_dist, final_dist <= float(args.arrival_radius_m)


def execute_wait_window(
    state: WorldState,
    stage2: Stage2Runtime,
    scene_id: str,
    wait_xy: np.ndarray,
    action_goal_xy: np.ndarray,
    goal: dict[str, Any],
    goal_index: int,
    segments: list[dict[str, Any]],
    args: argparse.Namespace,
) -> int:
    start_frame = len(state.ego_joints_world) - 1
    plan = np.repeat(state.current_root_xy[None], MOVEWAIT_FRAMES, axis=0)
    gen = stage2.generate_move_wait(
        scene_id=scene_id,
        history_world=state.history_joints(),
        root_plan_world=root_plan_xyz(plan, state.current_root_y),
        body_goal_world=xyz_from_xz(wait_xy, state.current_root_y),
        goal_type="wait",
    )
    blended = blend_overlap(state.previous_tail, gen.joints_world, OVERLAP_FRAMES) if bool(args.blend_overlap) else gen.joints_world.copy()
    execute = blended[: int(args.execute_frames)]
    segment_id = append_segment(
        segments,
        "WAIT",
        goal,
        start_frame,
        start_frame + len(execute),
        wait_xy,
        action_goal_xy=action_goal_xy,
    )
    state.append_window(execute, "WAIT", goal_index, segment_id)
    state.previous_tail = blended[int(args.execute_frames) : int(args.execute_frames) + OVERLAP_FRAMES].copy()
    return len(execute)


def generate_action_buffer(
    state: WorldState,
    stage2: Stage2Runtime,
    scene_id: str,
    goal: dict[str, Any],
    duration: int,
    blend_overlap_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    body_xyz = body_goal_xyz(goal, state.current_root_y)
    hand_xyz = hand_goal_xyz(goal)
    source = goal.get("source_segment") or {}
    gen = stage2.generate_action(
        scene_id=scene_id,
        history_world=state.history_joints(),
        duration=int(duration),
        body_goal_world=body_xyz,
        hand_goal_world=hand_xyz,
        text=str(source.get("text", goal.get("text", goal.get("goal_type", "")))),
        goal_type=str(goal.get("goal_type", "")),
    )
    blended = blend_overlap(state.previous_tail, gen.joints_world, OVERLAP_FRAMES) if bool(blend_overlap_enabled) else gen.joints_world.copy()
    return blended[: int(duration)].copy(), blended[int(duration) : int(duration) + OVERLAP_FRAMES].copy()


def simulate_full_episode(
    episode_path: Path,
    args: argparse.Namespace,
    stage1: Stage1Runtime,
    stage2: Stage2Runtime,
    device: torch.device,
    out_dir: Path,
    render_blender: bool = False,
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

    init_joints, init_sim_t = initial_gt_history(args.dataset, scene_id, ego_char, STAGE2_HISTORY_FRAMES)
    state = WorldState(
        sim_t=init_sim_t,
        ego_joints_world=[frame.astype(np.float32) for frame in init_joints],
        phase_per_frame=["GT_HISTORY"] * len(init_joints),
        goal_index_per_frame=[-1] * len(init_joints),
        segment_id_per_frame=[-1] * len(init_joints),
    )

    queue = list(ego_clip.goals)
    completed = 0
    failed = 0
    timeouts = 0
    waits = 0
    actions_interrupted = 0
    per_goal: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    stage1_records: list[dict[str, Any]] = []
    max_total_frames = int(args.max_frames_per_goal) * max(1, len(queue)) + 500

    for goal_index, goal in enumerate(queue):
        if state.sim_t >= max_total_frames:
            break
        goal_xy = body_goal_xy(goal)
        if goal_xy is None:
            failed += 1
            per_goal.append({"goal_type": goal.get("goal_type"), "status": "failed_no_body_goal"})
            continue
        goal_type = normalize_goal_type(goal.get("goal_type"))
        goal_start_frame = len(state.ego_joints_world) - 1
        status = "unknown"
        original_action_duration: int | None = None
        used_action_duration: int | None = None
        action_duration_clipped = False

        should_move = goal_type == "move" or float(np.linalg.norm(state.current_root_xy - goal_xy)) >= float(args.action_direct_threshold_m)
        if should_move:
            while len(state.ego_joints_world) - 1 - goal_start_frame < int(args.max_frames_per_goal):
                if float(np.linalg.norm(state.current_root_xy - goal_xy)) <= float(args.arrival_radius_m):
                    status = "move_reached"
                    break
                _, reached = execute_move_window(
                    state,
                    stage1,
                    stage2,
                    scene,
                    scene_id,
                    others_clips,
                    goal_xy,
                    goal,
                    "MOVE",
                    goal_index,
                    segments,
                    stage1_records,
                    args,
                )
                if reached:
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
            original_action_duration = goal_duration_frames(goal)
            used_action_duration = min(int(original_action_duration), int(args.max_action_frames))
            action_duration_clipped = int(used_action_duration) < int(original_action_duration)
            duration = int(used_action_duration)
            action_done = False
            while not action_done and state.sim_t < max_total_frames:
                action_buffer, action_tail = generate_action_buffer(state, stage2, scene_id, goal, duration, bool(args.blend_overlap))
                cursor = 0
                interrupted = False
                while cursor < duration:
                    if action_intrusion_likely(goal_xy, others_clips, state.sim_t, int(args.horizon), float(args.interrupt_distance_m)):
                        interrupted = True
                        actions_interrupted += 1
                        if len(state.ego_joints_world) >= OVERLAP_FRAMES:
                            state.previous_tail = np.stack(state.ego_joints_world[-OVERLAP_FRAMES:], axis=0).copy()
                        wait_point = choose_wait_point(scene, state.current_root_xy, goal_xy, others_clips, state.sim_t, args)
                        move_wait_start = state.sim_t
                        while (
                            float(np.linalg.norm(state.current_root_xy - wait_point)) > float(args.arrival_radius_m)
                            and state.sim_t - move_wait_start < int(args.max_frames_per_goal)
                        ):
                            _, reached_wait = execute_move_window(
                                state,
                                stage1,
                                stage2,
                                scene,
                                scene_id,
                                others_clips,
                                wait_point,
                                goal,
                                "MOVE_TO_WAIT",
                                goal_index,
                                segments,
                                stage1_records,
                                args,
                                action_goal_xy=goal_xy,
                            )
                            waits += int(args.execute_frames)
                            if reached_wait:
                                break
                        if float(np.linalg.norm(state.current_root_xy - wait_point)) > float(args.arrival_radius_m):
                            failed += 1
                            timeouts += 1
                            status = "wait_move_timeout"
                            action_done = True
                            break
                        waited = 0
                        while action_intrusion_likely(goal_xy, others_clips, state.sim_t, int(args.horizon), float(args.wait_release_distance_m)):
                            if waited >= int(args.max_wait_frames):
                                break
                            advanced = execute_wait_window(state, stage2, scene_id, wait_point, goal_xy, goal, goal_index, segments, args)
                            waits += int(advanced)
                            waited += int(advanced)
                        if action_intrusion_likely(goal_xy, others_clips, state.sim_t, int(args.horizon), float(args.wait_release_distance_m)):
                            failed += 1
                            timeouts += 1
                            status = "wait_timeout"
                            action_done = True
                            break
                        move_back_start = state.sim_t
                        while (
                            float(np.linalg.norm(state.current_root_xy - goal_xy)) > float(args.arrival_radius_m)
                            and state.sim_t - move_back_start < int(args.max_frames_per_goal)
                        ):
                            _, reached_goal = execute_move_window(
                                state,
                                stage1,
                                stage2,
                                scene,
                                scene_id,
                                others_clips,
                                goal_xy,
                                goal,
                                "MOVE_BACK_TO_GOAL",
                                goal_index,
                                segments,
                                stage1_records,
                                args,
                                action_goal_xy=goal_xy,
                            )
                            waits += int(args.execute_frames)
                            if reached_goal:
                                break
                        if float(np.linalg.norm(state.current_root_xy - goal_xy)) > float(args.arrival_radius_m):
                            failed += 1
                            timeouts += 1
                            status = "wait_return_timeout"
                            action_done = True
                            break
                        break

                    chunk = min(int(args.execute_frames), int(duration) - int(cursor))
                    start = len(state.ego_joints_world) - 1
                    segment_id = append_segment(segments, "ACTION", goal, start, start + chunk, goal_xy)
                    state.append_window(action_buffer[cursor : cursor + chunk], "ACTION", goal_index, segment_id)
                    cursor += chunk

                if interrupted:
                    if status.startswith("wait_"):
                        break
                    # Restart from the same capped action duration after wait/return.
                    continue
                state.previous_tail = action_tail.copy()
                completed += 1
                status = "completed"
                action_done = True

        goal_record = {
            "goal_type": goal.get("goal_type"),
            "normalized_goal_type": goal_type,
            "status": status,
            "start_frame": int(goal_start_frame),
            "end_frame": int(len(state.ego_joints_world) - 1),
            "duration": int(len(state.ego_joints_world) - 1 - goal_start_frame),
            "body_goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "final_distance_m": float(np.linalg.norm(state.current_root_xy - goal_xy)),
        }
        if original_action_duration is not None:
            goal_record.update(
                {
                    "original_action_duration": int(original_action_duration),
                    "used_action_duration": int(used_action_duration),
                    "action_duration_clipped": bool(action_duration_clipped),
                }
            )
        per_goal.append(goal_record)

    ego_joints = np.stack(state.ego_joints_world, axis=0).astype(np.float32)
    ego_root = joints_to_root_xz(ego_joints)
    others_root = np.zeros((len(others_clips), len(ego_root), 2), dtype=np.float32)
    for idx, clip in enumerate(others_clips):
        for t in range(len(ego_root)):
            others_root[idx, t] = future_slice(clip.root_xy, t, 1)[0]
    dyn_metrics = compute_collisions(ego_root, others_clips, float(args.collision_distance_m))
    static_metrics = compute_static_metrics(ego_root, scene)
    total_goals = len(ego_clip.goals)
    result: dict[str, Any] = {
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
        "simulated_frames": int(len(ego_root)),
        "per_goal": per_goal,
        "segments": segments,
        "stage1_records": stage1_records,
        **dyn_metrics,
        **static_metrics,
    }
    traj_path = out_dir / "trajectories" / scene_id / f"{episode_path.stem}.npz"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        traj_path,
        ego_joints28_world=ego_joints,
        ego_root_world=ego_root,
        others_root_world=others_root,
        phase_per_frame=np.asarray(state.phase_per_frame),
        goal_index_per_frame=np.asarray(state.goal_index_per_frame, dtype=np.int32),
        segment_id_per_frame=np.asarray(state.segment_id_per_frame, dtype=np.int32),
    )
    result["trajectory_path"] = display_path(traj_path)
    if bool(args.save_plots):
        plots_root = out_dir / "plots" / scene_id / episode_path.stem
        goals_xy = [body_goal_xy(goal) for goal in ego_clip.goals]
        goals_xy = [goal for goal in goals_xy if goal is not None]
        plot_episode_overview(
            plots_root / "overview.png",
            scene,
            ego_root,
            others_root,
            goals_xy,
            segments,
            f"{episode.get('episode_id', episode_path.stem)}",
        )
        for idx, seg in enumerate(segments[: int(args.max_window_plots_per_episode)]):
            goal_np = np.asarray(seg["goal_xy"], dtype=np.float32)
            plot_window(
                plots_root / f"window_{idx:03d}_{seg.get('type', 'seg').lower()}.png",
                scene,
                ego_root,
                others_root,
                goal_np,
                seg,
                f"{episode_path.stem} {idx:03d} {seg.get('type')} {seg.get('goal_type')}",
            )
        result["plot_dir"] = display_path(plots_root)

    root_video_path = out_dir / "root_videos" / scene_id / f"{episode_path.stem}.mp4"
    if bool(args.save_root_videos):
        save_root_debug_video(
            root_video_path,
            scene,
            ego_root,
            others_root,
            segments,
            f"{episode.get('episode_id', episode_path.stem)}",
            fps=int(args.root_video_fps),
            history_frames=int(args.root_video_history_frames),
        )
        result["root_video_path"] = display_path(root_video_path)

    ego_smplx_path: Path | None = None
    if bool(args.fit_smplx) or bool(render_blender):
        ego_smplx_path = out_dir / "smplx" / scene_id / f"{episode_path.stem}_ego_smplx_results.pkl"
        fit_joints28_episode_to_smplx(
            ego_joints,
            ego_smplx_path,
            device,
            smooth_weight=float(args.fit_smooth_weight),
        )
        result["ego_smplx_path"] = display_path(ego_smplx_path)

    if bool(render_blender):
        if ego_smplx_path is None:
            raise RuntimeError("ego_smplx_path was not created before Blender render.")
        blender_dir = out_dir / "blender" / scene_id / episode_path.stem
        blender_dir.mkdir(parents=True, exist_ok=True)
        meta_path = build_full_loop_blender_meta(
            args=args,
            episode=episode,
            scene_id=scene_id,
            ego_id=ego_id,
            chars=chars,
            ego_smplx_path=ego_smplx_path,
            num_frames=len(ego_joints),
            blender_dir=blender_dir,
        )
        video_path = out_dir / "videos" / scene_id / f"{episode_path.stem}.mp4"
        render_full_loop_blender_video(args, scene_id, meta_path, video_path)
        result["blender_meta_path"] = display_path(meta_path)
        result["blender_video_path"] = display_path(video_path)
        if bool(args.save_root_videos) and root_video_path.exists():
            final_video_path = out_dir / "final_videos" / scene_id / f"{episode_path.stem}.mp4"
            render_full_loop_final_video(video_path, root_video_path, final_video_path)
            result["final_video_path"] = display_path(final_video_path)

    result_path = out_dir / "results" / scene_id / f"{episode_path.stem}.json"
    write_json(result_path, result)
    result["result_path"] = display_path(result_path)
    return result


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    total = sum(int(item["total_goals"]) for item in results)
    completed = sum(int(item["completed_goals"]) for item in results)
    return {
        "num_episodes": len(results),
        "total_goals": int(total),
        "completed_goals": int(completed),
        "goal_success_rate": float(completed / max(total, 1)),
        "episode_full_success_rate": float(np.mean([float(item["goal_success_rate"]) >= 1.0 for item in results])),
        "timeout_count": int(sum(int(item["timeout_count"]) for item in results)),
        "action_interrupt_count": int(sum(int(item["action_interrupt_count"]) for item in results)),
        "mean_dynamic_collision_ratio": float(np.mean([float(item["dynamic_collision_ratio"]) for item in results])),
        "mean_static_collision_ratio": float(np.mean([float(item["static_collision_ratio"]) for item in results])),
    }


def main() -> None:
    args = parse_args()
    if int(args.execute_frames) != EXECUTE_FRAMES:
        raise ValueError(f"full loop currently expects execute_frames={EXECUTE_FRAMES}")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    episodes_dir = args.episodes_dir or (DEFAULT_ROOT / args.dataset / "episodes_v3")
    out_dir = output_root(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "args.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})

    stage1 = Stage1Runtime(
        checkpoint=args.stage1_checkpoint,
        optimizer_config=args.stage1_optimizer_config,
        device=device,
        horizon=int(args.horizon),
        past_frames=int(args.past_frames),
        future_frames=int(args.future_frames),
        num_sampling_steps=int(args.stage1_sampling_steps),
    )
    stage2 = Stage2Runtime(
        dataset=args.dataset,
        move_wait_checkpoint=args.stage2_movewait_checkpoint,
        action_checkpoint=args.stage2_action_checkpoint,
        device=device,
        num_sampling_steps=int(args.stage2_sampling_steps),
        nb_voxels=int(args.nb_voxels),
    )

    episode_paths = [repo_path(path) for path in sample_episode_set(args, repo_path(episodes_dir))]
    if args.max_episodes is not None:
        episode_paths = episode_paths[: int(args.max_episodes)]
    results = []
    blender_rendered = 0
    metrics_path = out_dir / "per_episode_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for episode_path in tqdm(episode_paths, desc="full ours loop", unit="episode"):
            render_blender = bool(args.render_blender) and (
                int(args.max_blender_videos) <= 0 or blender_rendered < int(args.max_blender_videos)
            )
            result = simulate_full_episode(episode_path, args, stage1, stage2, device, out_dir, render_blender=render_blender)
            if result.get("blender_video_path") is not None:
                blender_rendered += 1
            results.append(result)
            handle.write(json.dumps(result) + "\n")
            handle.flush()
    summary = aggregate(results)
    summary.update(
        {
            "run_name": args.run_name,
            "dataset": args.dataset,
            "stage1_checkpoint": display_path(repo_path(args.stage1_checkpoint)),
            "stage2_movewait_checkpoint": display_path(repo_path(args.stage2_movewait_checkpoint)),
            "stage2_action_checkpoint": display_path(repo_path(args.stage2_action_checkpoint)),
        }
    )
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
