from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

from build_multi_character_affordances import classify_goal_bucket, resolve_scene_ids
from preprocess_final import (
    PROJECT_ROOT,
    TRUMANS_ROOT_DEFAULT,
    build_lingo_plot_segments,
    build_trumans_plot_segments,
    compute_walkable_map,
    majority_filter,
    resolve_lingo_scene_split,
    save_mask,
    to_serializable,
    to_repo_relative,
    write_json,
)


LOCOMOTION_GOAL_TYPES = {"walk", "move"}
MIN_SPLIT_LEN = 16
BODY_GOAL_INITIAL_INSIDE_RADIUS_M = 0.50
BODY_GOAL_APPROACH_RADIUS_M = 0.025
CLEARANCE_FREE_THRESHOLD = 0.97
SEATED_OUTSIDE_STREAK = 8
STAND_SPLIT_REQUIRED_GOAL_DISPLACEMENT_M = 0.5
STAND_MOVE_START_DISPLACEMENT_M = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scenes_v2 with updated clearance and split locomotion/action segments.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--scene-list-file", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "preprocessed")
    parser.add_argument("--python-bin", type=Path, default=Path("python"))
    parser.add_argument("--trumans-root", type=Path, default=TRUMANS_ROOT_DEFAULT)
    parser.add_argument("--lingo-root", type=Path, default=None)
    return parser.parse_args()


def input_scene_dir(dataset: str, scene_id: str, output_root: Path) -> Path:
    return output_root / dataset / "scenes" / scene_id


def output_scene_dir(dataset: str, scene_id: str, output_root: Path) -> Path:
    return output_root / dataset / "scenes_v2" / scene_id


def plot_dir(scene_root_out: Path, dataset: str, sequence_id: str) -> Path:
    prefix = "trumans_sequence_actions" if dataset == "trumans" else "lingo_sequence_actions"
    return scene_root_out / "plots" / f"{prefix}_{sequence_id}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_clearance_v2(scene_root_out: Path, scene_occ: np.ndarray, dataset: str) -> tuple[Path, Path]:
    walkable = compute_walkable_map(scene_occ, use_floor=False, free_threshold=CLEARANCE_FREE_THRESHOLD)
    if dataset == "lingo":
        body_band = scene_occ[:, 1:86, :]
        free_ratio = 1.0 - body_band.mean(axis=1)
        walkable = free_ratio >= CLEARANCE_FREE_THRESHOLD
        walkable = majority_filter(walkable, kernel=5)
    npy_path = scene_root_out / "clearance_map.npy"
    vis_path = scene_root_out / "clearance_map.png"
    scene_root_out.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, walkable.astype(np.uint8))
    save_mask(walkable, vis_path, scale=4)
    return npy_path, vis_path


def save_distance_map_v2(scene_root_out: Path, clearance: np.ndarray, grid_meta: dict) -> Path:
    x_min = float(grid_meta["x_min"])
    x_max = float(grid_meta["x_max"])
    x_res = int(grid_meta["x_res"])
    resolution = float((x_max - x_min) / x_res)
    distance_map = distance_transform_edt(clearance.astype(bool)) * resolution
    npy_path = scene_root_out / "distance_map.npy"
    np.save(npy_path, distance_map.astype(np.float32))
    return npy_path


def world_to_grid(center_xz: tuple[float, float], grid_meta: dict, shape: tuple[int, int]) -> Optional[tuple[int, int]]:
    nx, nz = shape
    x_min, x_max = float(grid_meta["x_min"]), float(grid_meta["x_max"])
    z_min, z_max = float(grid_meta["z_min"]), float(grid_meta["z_max"])
    x, z = float(center_xz[0]), float(center_xz[1])
    if not (x_min <= x <= x_max and z_min <= z <= z_max):
        return None
    ix = int((x - x_min) / (x_max - x_min) * nx)
    iz = int((z - z_min) / (z_max - z_min) * nz)
    ix = max(0, min(nx - 1, ix))
    iz = max(0, min(nz - 1, iz))
    return ix, iz


def point_on_clearance(point_xz: np.ndarray, clearance: np.ndarray, grid_meta: dict) -> bool:
    grid = world_to_grid((float(point_xz[0]), float(point_xz[1])), grid_meta, clearance.shape)
    if grid is None:
        return False
    return bool(clearance[grid[0], grid[1]])


def find_body_goal_entry_frame(
    pelvis_track_xz: np.ndarray,
    body_goal_xz: np.ndarray,
    radius_m: float,
) -> Optional[int]:
    dists = np.linalg.norm(pelvis_track_xz - body_goal_xz[None, :], axis=-1)
    hit = np.nonzero(dists <= radius_m)[0]
    if len(hit) == 0:
        return None
    return int(hit[0])


def find_displacement_exit_frame(
    pelvis_track_xz: np.ndarray,
    threshold_m: float,
) -> Optional[int]:
    if len(pelvis_track_xz) == 0:
        return None
    dists = np.linalg.norm(pelvis_track_xz - pelvis_track_xz[0:1], axis=-1)
    hit = np.nonzero(dists >= threshold_m)[0]
    if len(hit) == 0:
        return None
    for idx in hit:
        if int(idx) > 0:
            return int(idx)
    return None


def find_seated_action_start_frame(
    pelvis_track_xz: np.ndarray,
    clearance: np.ndarray,
    grid_meta: dict,
) -> Optional[int]:
    inside = np.array([point_on_clearance(p, clearance, grid_meta) for p in pelvis_track_xz], dtype=bool)
    n = len(inside)
    for idx in range(n):
        if inside[idx]:
            continue
        if not inside[idx:].any():
            return int(idx)
        end = min(n, idx + SEATED_OUTSIDE_STREAK)
        if end - idx >= SEATED_OUTSIDE_STREAK and np.all(~inside[idx:end]):
            return int(idx)
    return None


def make_locomotion_goal_pose(
    dataset: str,
    orig_goal_pose: dict,
    joints_arr: np.ndarray,
    global_orient_arr: np.ndarray,
    global_frame: int,
) -> dict:
    goal_pose = copy.deepcopy(orig_goal_pose)
    pelvis = np.asarray(joints_arr[global_frame, 0, :], dtype=np.float32)
    goal_pose["pelvis"] = pelvis.tolist()
    goal_pose["yaw"] = None
    goal_pose["global_orient"] = np.asarray(global_orient_arr[global_frame], dtype=np.float32).tolist()
    goal_pose["hand"] = None
    if dataset == "trumans":
        goal_pose["left_hand"] = None
        goal_pose["right_hand"] = None
    return goal_pose


def remap_goal_type_v2(dataset: str, goal_type: str, text: Optional[str]) -> str:
    normalized_text = "" if text is None else str(text).strip().lower()
    if goal_type in LOCOMOTION_GOAL_TYPES:
        return "walk"
    if goal_type != "stand":
        return goal_type
    if dataset == "trumans":
        return "stand_up"
    if normalized_text in {"stand still", "maintains stand posture"}:
        return "stand_still"
    return "stand_up"


def split_segment_if_needed(
    dataset: str,
    segment: dict,
    seq_global_start: int,
    transl_arr: np.ndarray,
    joints_arr: np.ndarray,
    global_orient_arr: np.ndarray,
    clearance: np.ndarray,
    grid_meta: dict,
    previous_segment_ended_outside_clearance: bool,
) -> list[dict]:
    goal_type = str(segment.get("goal_type") or "")
    effective_goal_type = remap_goal_type_v2(dataset, goal_type, segment.get("text"))
    if goal_type in LOCOMOTION_GOAL_TYPES:
        out = copy.deepcopy(segment)
        out["goal_type"] = effective_goal_type
        out["orig_segment_id"] = int(segment["segment_id"])
        out["move_for_action"] = False
        return [out]

    local_start = int(segment["start"])
    local_end = int(segment["end"])
    if local_end < local_start:
        out = copy.deepcopy(segment)
        out["goal_type"] = effective_goal_type
        out["orig_segment_id"] = int(segment["segment_id"])
        out["move_for_action"] = False
        return [out]

    global_slice_start = seq_global_start + local_start
    global_slice_end = seq_global_start + local_end + 1
    transl_track = np.asarray(transl_arr[global_slice_start:global_slice_end], dtype=np.float32)
    pelvis_track_xz = transl_track[:, [0, 2]]
    if len(pelvis_track_xz) == 0:
        out = copy.deepcopy(segment)
        out["goal_type"] = effective_goal_type
        out["orig_segment_id"] = int(segment["segment_id"])
        out["move_for_action"] = False
        return [out]

    body_goal = np.asarray(segment["goal_pose"]["pelvis"], dtype=np.float32)
    body_goal_xz = body_goal[[0, 2]]
    # Split decision is based on first-frame pelvis vs body goal in 2D (x-z plane).
    initial_inside = float(np.linalg.norm(pelvis_track_xz[0] - body_goal_xz)) <= BODY_GOAL_INITIAL_INSIDE_RADIUS_M

    if effective_goal_type == "stand_up":
        stand_goal_displacement = float(np.linalg.norm(body_goal_xz - pelvis_track_xz[0]))
        if stand_goal_displacement < STAND_SPLIT_REQUIRED_GOAL_DISPLACEMENT_M:
            out = copy.deepcopy(segment)
            out["goal_type"] = effective_goal_type
            out["orig_segment_id"] = int(segment["segment_id"])
            out["move_for_action"] = False
            return [out]

        move_start_offset = find_displacement_exit_frame(
            pelvis_track_xz,
            STAND_MOVE_START_DISPLACEMENT_M,
        )
        if move_start_offset is None:
            out = copy.deepcopy(segment)
            out["goal_type"] = effective_goal_type
            out["orig_segment_id"] = int(segment["segment_id"])
            out["move_for_action"] = False
            return [out]

        stand_end = local_start + int(move_start_offset) - 1
        move_start = local_start + int(move_start_offset)
        stand_len = stand_end - local_start + 1
        move_len = local_end - move_start + 1
        if stand_len < MIN_SPLIT_LEN or move_len < MIN_SPLIT_LEN:
            out = copy.deepcopy(segment)
            out["goal_type"] = effective_goal_type
            out["orig_segment_id"] = int(segment["segment_id"])
            out["move_for_action"] = False
            return [out]

        move_global_frame = seq_global_start + local_end

        stand_segment = copy.deepcopy(segment)
        stand_segment["start"] = int(local_start)
        stand_segment["end"] = int(stand_end)
        stand_segment["interaction_frame"] = int(stand_end)
        stand_segment["goal_type"] = effective_goal_type
        stand_segment["move_for_action"] = False
        stand_segment["orig_segment_id"] = int(segment["segment_id"])

        move_segment = copy.deepcopy(segment)
        move_segment["start"] = int(move_start)
        move_segment["end"] = int(local_end)
        move_segment["interaction_frame"] = int(local_end)
        move_segment["goal_type"] = "walk"
        move_segment["text"] = "walk"
        move_segment["active_hand"] = "none"
        move_segment["goal_pose"] = make_locomotion_goal_pose(
            dataset=dataset,
            orig_goal_pose=segment["goal_pose"],
            joints_arr=joints_arr,
            global_orient_arr=global_orient_arr,
            global_frame=move_global_frame,
        )
        move_segment["move_for_action"] = True
        move_segment["orig_segment_id"] = int(segment["segment_id"])
        return [stand_segment, move_segment]

    if previous_segment_ended_outside_clearance:
        action_start_offset = 0
    else:
        if initial_inside:
            action_start_offset = 0
        else:
            action_start_offset = find_body_goal_entry_frame(
                pelvis_track_xz,
                body_goal_xz,
                BODY_GOAL_APPROACH_RADIUS_M,
            )

    if action_start_offset is None or int(action_start_offset) <= 0:
        out = copy.deepcopy(segment)
        out["goal_type"] = effective_goal_type
        out["orig_segment_id"] = int(segment["segment_id"])
        out["move_for_action"] = False
        return [out]

    action_start = local_start + int(action_start_offset)
    locomotion_end = action_start - 1

    locomotion_len = locomotion_end - local_start + 1
    action_len = local_end - action_start + 1
    if locomotion_len < MIN_SPLIT_LEN or action_len < MIN_SPLIT_LEN:
        out = copy.deepcopy(segment)
        out["goal_type"] = effective_goal_type
        out["orig_segment_id"] = int(segment["segment_id"])
        out["move_for_action"] = False
        return [out]

    locomotion_global_frame = seq_global_start + locomotion_end

    locomotion_segment = copy.deepcopy(segment)
    locomotion_segment["start"] = int(local_start)
    locomotion_segment["end"] = int(locomotion_end)
    locomotion_segment["interaction_frame"] = int(locomotion_end)
    locomotion_segment["goal_type"] = "walk"
    locomotion_segment["text"] = "walk"
    locomotion_segment["active_hand"] = "none"
    locomotion_segment["goal_pose"] = make_locomotion_goal_pose(
        dataset=dataset,
        orig_goal_pose=segment["goal_pose"],
        joints_arr=joints_arr,
        global_orient_arr=global_orient_arr,
        global_frame=locomotion_global_frame,
    )
    locomotion_segment["move_for_action"] = True
    locomotion_segment["orig_segment_id"] = int(segment["segment_id"])

    action_segment = copy.deepcopy(segment)
    action_segment["start"] = int(action_start)
    action_segment["goal_type"] = effective_goal_type
    action_segment["move_for_action"] = False
    action_segment["orig_segment_id"] = int(segment["segment_id"])

    return [locomotion_segment, action_segment]


def process_sequence(
    dataset: str,
    sequence_record: dict,
    clearance: np.ndarray,
    grid_meta: dict,
) -> dict:
    smpl_ref = sequence_record["human_motion_ref"]["smplx"]
    transl_arr = np.load(PROJECT_ROOT / smpl_ref["transl_path"], mmap_mode="r")
    global_orient_arr = np.load(PROJECT_ROOT / smpl_ref["global_orient_path"], mmap_mode="r")
    joints_arr = np.load(PROJECT_ROOT / sequence_record["human_motion_ref"]["path"], mmap_mode="r")
    seq_global_start = int(sequence_record["human_motion_ref"]["start"])

    new_segments: list[dict] = []
    previous_segment_ended_outside_clearance = False
    for segment in sequence_record.get("segment_list", []):
        split_segments = split_segment_if_needed(
            dataset=dataset,
            segment=segment,
            seq_global_start=seq_global_start,
            transl_arr=transl_arr,
            joints_arr=joints_arr,
            global_orient_arr=global_orient_arr,
            clearance=clearance,
            grid_meta=grid_meta,
            previous_segment_ended_outside_clearance=previous_segment_ended_outside_clearance,
        )
        new_segments.extend(split_segments)

        local_end = int(segment["end"])
        global_end = seq_global_start + local_end
        end_pelvis_xz = np.asarray(transl_arr[global_end], dtype=np.float32)[[0, 2]]
        previous_segment_ended_outside_clearance = not point_on_clearance(end_pelvis_xz, clearance, grid_meta)

    merged_segments: list[dict] = []
    for segment in new_segments:
        if (
            merged_segments
            and merged_segments[-1].get("goal_type") == "walk"
            and segment.get("goal_type") == "walk"
        ):
            prev = merged_segments[-1]
            prev["end"] = int(segment["end"])
            prev["interaction_frame"] = int(segment["interaction_frame"])
            prev["goal_pose"] = copy.deepcopy(segment["goal_pose"])
            prev["move_for_action"] = bool(prev.get("move_for_action", False) or segment.get("move_for_action", False))
            orig_ids = prev.get("orig_segment_id")
            if isinstance(orig_ids, list):
                prev_ids = orig_ids
            else:
                prev_ids = [orig_ids]
            curr_orig = segment.get("orig_segment_id")
            if isinstance(curr_orig, list):
                prev_ids.extend(curr_orig)
            else:
                prev_ids.append(curr_orig)
            deduped = []
            for item in prev_ids:
                if item not in deduped:
                    deduped.append(item)
            prev["orig_segment_id"] = deduped
            continue
        merged_segments.append(segment)

    for new_id, segment in enumerate(merged_segments):
        segment["segment_id"] = int(new_id)
        segment.pop("plot_path", None)

    updated = copy.deepcopy(sequence_record)
    updated["segment_list"] = merged_segments
    return updated


def generate_sequence_plots(
    dataset: str,
    scene_root_out: Path,
    sequence_record: dict,
    args: argparse.Namespace,
) -> None:
    scene_id = str(sequence_record["scene_id"])
    sequence_id = str(sequence_record["sequence_id"])
    if dataset == "trumans":
        payload = {
            "sequence_id": sequence_id,
            "scene_id": scene_id,
            "segment_list": build_trumans_plot_segments(sequence_record),
        }
        script = PROJECT_ROOT / "vis" / "visualize_trumans_sequence_actions.py"
        extra = ["--data-root", str(args.trumans_root)]
    else:
        payload = {
            "sequence_id": sequence_id,
            "scene_id": scene_id,
            "human_motion_ref": sequence_record["human_motion_ref"],
            "segment_list": build_lingo_plot_segments(sequence_record),
        }
        script = PROJECT_ROOT / "vis" / "visualize_lingo_sequence_actions.py"
        extra = ["--split", resolve_lingo_scene_split(scene_id, args)]

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"{sequence_id}.plot_segments.",
        dir="/tmp",
        delete=False,
    ) as handle:
        json.dump(to_serializable(payload), handle, indent=2)
        plot_input_path = Path(handle.name)
    try:
        cmd = [
            str(args.python_bin),
            str(script),
            "--sequence-id",
            sequence_id,
            "--scene-name",
            scene_id,
            "--segment-json",
            str(plot_input_path),
            "--output-dir",
            str(plot_dir(scene_root_out, dataset, sequence_id).parent),
        ] + extra
        subprocess.run(cmd, check=True)
    finally:
        plot_input_path.unlink(missing_ok=True)

    for segment in sequence_record["segment_list"]:
        segment["plot_path"] = to_repo_relative(
            plot_dir(scene_root_out, dataset, sequence_id)
            / f"{segment['segment_id']:02d}_{segment['start']:04d}_{segment['end']:04d}.png"
        )


def process_scene(dataset: str, scene_id: str, output_root: Path, args: argparse.Namespace) -> None:
    scene_root_in = input_scene_dir(dataset, scene_id, output_root)
    scene_root_out = output_scene_dir(dataset, scene_id, output_root)
    if scene_root_out.exists():
        shutil.rmtree(scene_root_out)
    scene_root_out.mkdir(parents=True, exist_ok=True)

    scene_record = load_json(scene_root_in / "scene.json")
    occ_path = PROJECT_ROOT / scene_record["occupancy_grid_path"]
    scene_occ = np.load(occ_path)
    clearance_npy, clearance_vis = save_clearance_v2(scene_root_out, scene_occ, dataset)
    clearance = np.load(clearance_npy).astype(bool)
    grid_meta = scene_record["grid_meta"]
    distance_npy = save_distance_map_v2(scene_root_out, clearance, grid_meta)

    updated_scene_record = copy.deepcopy(scene_record)
    updated_scene_record["clearance_map_npy_path"] = to_repo_relative(clearance_npy)
    updated_scene_record["clearance_map_vis_path"] = to_repo_relative(clearance_vis)
    updated_scene_record["distance_map_npy_path"] = to_repo_relative(distance_npy)
    write_json(scene_root_out / "scene.json", updated_scene_record)

    seq_out_dir = scene_root_out / "sequences"
    seq_out_dir.mkdir(parents=True, exist_ok=True)
    for seq_path in sorted((scene_root_in / "sequences").glob("*.json")):
        sequence_record = load_json(seq_path)
        updated_sequence = process_sequence(dataset, sequence_record, clearance, grid_meta)
        generate_sequence_plots(dataset, scene_root_out, updated_sequence, args)
        write_json(seq_out_dir / seq_path.name, updated_sequence)


def main() -> None:
    args = parse_args()
    scene_ids = resolve_scene_ids(args)
    for scene_id in scene_ids:
        process_scene(args.dataset, scene_id, args.output_root, args)
        print(f"processed {args.dataset} {scene_id}")


if __name__ == "__main__":
    main()
