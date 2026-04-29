from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from build_multi_character_affordances import PROJECT_ROOT, canonical_name, load_json, resolve_scene_ids


SUPPORT_PICKUP_TYPES = {"pick_up", "lift", "raise"}
SUPPORT_PUTDOWN_TYPES = {"put_down", "lower"}
DEFAULT_CHAR_CHOICES = [2, 3, 4]
DEFAULT_OBJECT_WINDOW_SIZES = [4, 5]
DEFAULT_LOCO_GOAL_NUMBERS = [4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-character episode seeds from episode bank v2.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--scene-list-file", type=Path, default=None)
    parser.add_argument("--episodes-per-scene", type=int, default=30)
    parser.add_argument("--episodes-per-character-count", type=int, default=10)
    parser.add_argument("--object-char-choices", type=str, default="2,3,4")
    parser.add_argument("--object-segment-sizes", type=str, default="4,5")
    parser.add_argument("--loco-char-choices", type=str, default="2,3,4")
    parser.add_argument("--loco-goal-numbers", type=str, default="4,5")
    parser.add_argument("--path-conflict-threshold-m", type=float, default=0.60)
    parser.add_argument("--goal-conflict-threshold-m", type=float, default=0.50)
    parser.add_argument("--start-overlap-diameter-m", type=float, default=None)
    parser.add_argument("--start-overlap-lateral-diameter-m", type=float, default=0.60)
    parser.add_argument("--start-overlap-fore-aft-diameter-m", type=float, default=0.60)
    parser.add_argument("--max-attempts-per-scene", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "preprocessed")
    return parser.parse_args()


def parse_int_list(value: str, default: Sequence[int], *, min_value: int = 1) -> List[int]:
    if value is None:
        return list(default)
    items = [int(v.strip()) for v in value.split(",") if v.strip()]
    values = [v for v in items if v >= min_value]
    if not values:
        return list(default)
    return values


def load_scene_bank(output_root: Path, dataset: str, scene_id: str) -> dict:
    path = output_root / dataset / "episode_bank_v2" / scene_id / "scene_static.json"
    if not path.exists():
        raise FileNotFoundError(f"scene bank not found: {path}")
    return load_json(path)


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_sequence_payload_cached(
    dataset: str,
    scene_id: str,
    sequence_id: str,
    sequence_cache: Dict[Tuple[str, str, str], dict],
) -> dict:
    key = (dataset, scene_id, sequence_id)
    cached = sequence_cache.get(key)
    if cached is not None:
        return cached

    path = PROJECT_ROOT / "data" / "preprocessed" / dataset / "scenes" / scene_id / "sequences" / f"{sequence_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"sequence json not found: {path}")
    payload = load_json(path)
    sequence_cache[key] = payload
    return payload


def load_npy_cached(path_str: str, array_cache: Dict[Path, np.ndarray]) -> np.ndarray:
    path = resolve_repo_path(path_str)
    cached = array_cache.get(path)
    if cached is not None:
        return cached
    arr = np.load(path, mmap_mode="r")
    array_cache[path] = arr
    return arr


def record_key(rec: dict) -> Tuple[str, int]:
    return str(rec["scene_id"]), int(rec["segment_id"])


def make_source_segment(rec: dict) -> dict:
    return {
        "scene_id": rec.get("scene_id"),
        "sequence_id": rec.get("sequence_id"),
        "segment_id": int(rec.get("segment_id", 0)),
        "start": int(rec.get("start", 0)),
        "end": int(rec.get("end", 0)),
        "interaction_frame": int(rec.get("interaction_frame", 0)),
        "global_frame_index": rec.get("global_frame_index"),
        "goal_type": rec.get("goal_type"),
        "goal_category": rec.get("goal_category"),
        "text": rec.get("text"),
    }


def make_goal_entry(rec: dict) -> dict:
    return {
        "goal_type": rec.get("goal_type"),
        "goal_category": rec.get("goal_category"),
        "mode": rec.get("mode"),
        "active_hand": rec.get("active_hand"),
        "target_name": rec.get("target_name"),
        "acted_on_object_name": rec.get("acted_on_object_name"),
        "acted_on_object_id": rec.get("acted_on_object_id"),
        "support_object_name": rec.get("support_object_name"),
        "support_object_id": rec.get("support_object_id"),
        "body_goal": rec.get("body_goal"),
        "hand_goal": rec.get("hand_goal"),
        "global_orient_rotvec": rec.get("global_orient_rotvec"),
        "body_mask_path": rec.get("body_mask_path"),
        "hand_mask_path": rec.get("hand_mask_path"),
        "seated_cluster_id": rec.get("seated_cluster_id"),
        "movable_seated_cluster_id": rec.get("movable_seated_cluster_id"),
        "fixed_cluster_id": rec.get("fixed_cluster_id"),
        "source_segment": make_source_segment(rec),
    }


def to_point3(value: Optional[list[float]]) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return None
    return arr


def infer_character_start_frame(char: dict) -> Tuple[str, int]:
    if char.get("source_window") is not None:
        window = char["source_window"]
        return str(window["sequence_id"]), int(window["local_start"])

    goals = char.get("goal_sequence", [])
    if not goals:
        raise ValueError(f"Character {char.get('character_id', 'unknown')} has no goal_sequence.")
    source = goals[0].get("source_segment", {})
    return str(source["sequence_id"]), int(source["start"])


def attach_start_states(
    dataset: str,
    scene_id: str,
    character_assignments: list[dict],
    sequence_cache: Dict[Tuple[str, str, str], dict],
    array_cache: Dict[Path, np.ndarray],
) -> None:
    for char in character_assignments:
        sequence_id, local_start = infer_character_start_frame(char)
        sequence_payload = load_sequence_payload_cached(
            dataset=dataset,
            scene_id=scene_id,
            sequence_id=sequence_id,
            sequence_cache=sequence_cache,
        )
        motion_ref = sequence_payload["human_motion_ref"]
        smpl = motion_ref["smplx"]
        seq_start = int(motion_ref["start"])
        global_frame = seq_start + int(local_start)

        transl_arr = load_npy_cached(str(smpl["transl_path"]), array_cache)
        global_orient_arr = load_npy_cached(str(smpl["global_orient_path"]), array_cache)
        body_translation = np.asarray(transl_arr[global_frame], dtype=np.float32)
        global_orient = np.asarray(global_orient_arr[global_frame], dtype=np.float32)

        char["start_state"] = {
            "sequence_id": sequence_id,
            "local_start": int(local_start),
            "global_frame_index": int(global_frame),
            "body_translation": body_translation.tolist(),
            "global_orient_rotvec": global_orient.tolist(),
        }


def attach_locomotion_window_terminal_goals(
    dataset: str,
    scene_id: str,
    character_assignments: list[dict],
    sequence_cache: Dict[Tuple[str, str, str], dict],
    array_cache: Dict[Path, np.ndarray],
) -> None:
    for char in character_assignments:
        window = char.get("source_window")
        if not isinstance(window, dict):
            continue

        sequence_id = str(window["sequence_id"])
        local_start = int(window["local_start"])
        local_end = int(window["local_end"])
        sequence_payload = load_sequence_payload_cached(
            dataset=dataset,
            scene_id=scene_id,
            sequence_id=sequence_id,
            sequence_cache=sequence_cache,
        )
        motion_ref = sequence_payload["human_motion_ref"]
        smpl = motion_ref["smplx"]
        seq_start = int(motion_ref["start"])
        transl_arr = load_npy_cached(str(smpl["transl_path"]), array_cache)
        global_orient_arr = load_npy_cached(str(smpl["global_orient_path"]), array_cache)
        goal_count = max(1, int(char.get("goal_count", 1)))
        if local_end <= local_start:
            goal_local_frames = [int(local_end)]
        else:
            goal_local_frames = (
                np.linspace(local_start, local_end, num=goal_count + 1, endpoint=True, dtype=np.int32)[1:].tolist()
            )

        goals: list[dict] = []
        previous_local = int(local_start)
        for goal_idx, local_frame in enumerate(goal_local_frames):
            local_frame = int(max(local_start, min(local_end, int(local_frame))))
            global_frame = seq_start + local_frame
            body_translation = np.asarray(transl_arr[global_frame], dtype=np.float32)
            global_orient = np.asarray(global_orient_arr[global_frame], dtype=np.float32)
            goals.append(
                {
                    "goal_type": "walk",
                    "goal_category": "locomotion",
                    "mode": None,
                    "active_hand": "none",
                    "target_name": None,
                    "acted_on_object_name": None,
                    "acted_on_object_id": None,
                    "support_object_name": None,
                    "support_object_id": None,
                    "body_goal": body_translation.tolist(),
                    "hand_goal": None,
                    "global_orient_rotvec": global_orient.tolist(),
                    "body_mask_path": None,
                    "hand_mask_path": None,
                    "seated_cluster_id": None,
                    "movable_seated_cluster_id": None,
                    "fixed_cluster_id": None,
                    "source_segment": {
                        "scene_id": scene_id,
                        "sequence_id": sequence_id,
                        "segment_id": int(goal_idx),
                        "start": int(previous_local),
                        "end": int(local_frame),
                        "interaction_frame": int(local_frame),
                        "global_frame_index": int(global_frame),
                        "goal_type": "walk",
                        "goal_category": "locomotion",
                        "text": f"Move to waypoint {goal_idx}",
                    },
                }
            )
            previous_local = int(local_frame)
        char["goal_sequence"] = goals


def detect_start_overlaps(
    character_assignments: list[dict],
    lateral_diameter_m: float,
    fore_aft_diameter_m: float,
) -> list[dict]:
    records: list[dict] = []
    lateral_diameter = float(lateral_diameter_m)
    fore_aft_diameter = float(fore_aft_diameter_m)
    if lateral_diameter <= 0.0 or fore_aft_diameter <= 0.0:
        return records

    start_shapes: list[tuple[str, np.ndarray, np.ndarray, dict]] = []
    for char in character_assignments:
        start_state = char.get("start_state")
        if not isinstance(start_state, dict):
            continue
        body_translation = to_point3(start_state.get("body_translation"))
        if body_translation is None:
            continue
        start_shapes.append(
            (
                str(char.get("character_id", "unknown")),
                np.asarray(body_translation[[0, 2]], dtype=np.float32),
                build_start_overlap_ellipse(
                    center_xz=np.asarray(body_translation[[0, 2]], dtype=np.float32),
                    global_orient_rotvec=start_state.get("global_orient_rotvec"),
                    lateral_diameter_m=lateral_diameter,
                    fore_aft_diameter_m=fore_aft_diameter,
                ),
                start_state,
            )
        )

    for i in range(len(start_shapes)):
        char_a, pos_a, poly_a, state_a = start_shapes[i]
        for j in range(i + 1, len(start_shapes)):
            char_b, pos_b, poly_b, state_b = start_shapes[j]
            distance = float(np.linalg.norm(pos_a - pos_b))
            if convex_polygons_overlap(poly_a, poly_b):
                records.append(
                    {
                        "character_a": char_a,
                        "character_b": char_b,
                        "distance_m": distance,
                        "lateral_diameter_m": lateral_diameter,
                        "fore_aft_diameter_m": fore_aft_diameter,
                        "sequence_a": state_a.get("sequence_id"),
                        "sequence_b": state_b.get("sequence_id"),
                        "location_xz": [
                            float((float(pos_a[0]) + float(pos_b[0])) * 0.5),
                            float((float(pos_a[1]) + float(pos_b[1])) * 0.5),
                        ],
                    }
                )
    return records


def detect_final_goal_overlaps(
    character_assignments: list[dict],
    lateral_diameter_m: float,
    fore_aft_diameter_m: float,
) -> list[dict]:
    records: list[dict] = []
    lateral_diameter = float(lateral_diameter_m)
    fore_aft_diameter = float(fore_aft_diameter_m)
    if lateral_diameter <= 0.0 or fore_aft_diameter <= 0.0:
        return records

    final_shapes: list[tuple[str, np.ndarray, np.ndarray, dict]] = []
    for char in character_assignments:
        goals = char.get("goal_sequence", [])
        if not isinstance(goals, list) or not goals:
            continue

        final_goal: Optional[dict] = None
        body_goal: Optional[np.ndarray] = None
        for goal in reversed(goals):
            body_goal = to_point3(goal.get("body_goal"))
            if body_goal is not None:
                final_goal = goal
                break
        if final_goal is None or body_goal is None:
            continue

        center_xz = np.asarray(body_goal[[0, 2]], dtype=np.float32)
        final_shapes.append(
            (
                str(char.get("character_id", "unknown")),
                center_xz,
                build_start_overlap_ellipse(
                    center_xz=center_xz,
                    global_orient_rotvec=final_goal.get("global_orient_rotvec"),
                    lateral_diameter_m=lateral_diameter,
                    fore_aft_diameter_m=fore_aft_diameter,
                ),
                final_goal,
            )
        )

    for i in range(len(final_shapes)):
        char_a, pos_a, poly_a, goal_a = final_shapes[i]
        for j in range(i + 1, len(final_shapes)):
            char_b, pos_b, poly_b, goal_b = final_shapes[j]
            distance = float(np.linalg.norm(pos_a - pos_b))
            if convex_polygons_overlap(poly_a, poly_b):
                source_a = goal_a.get("source_segment", {})
                source_b = goal_b.get("source_segment", {})
                records.append(
                    {
                        "character_a": char_a,
                        "character_b": char_b,
                        "distance_m": distance,
                        "lateral_diameter_m": lateral_diameter,
                        "fore_aft_diameter_m": fore_aft_diameter,
                        "sequence_a": source_a.get("sequence_id"),
                        "sequence_b": source_b.get("sequence_id"),
                        "segment_a": source_a.get("segment_id"),
                        "segment_b": source_b.get("segment_id"),
                        "goal_type_a": goal_a.get("goal_type"),
                        "goal_type_b": goal_b.get("goal_type"),
                        "location_xz": [
                            float((float(pos_a[0]) + float(pos_b[0])) * 0.5),
                            float((float(pos_a[1]) + float(pos_b[1])) * 0.5),
                        ],
                    }
                )
    return records


def point_in_convex_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
    for idx in range(int(poly.shape[0])):
        edge = poly[(idx + 1) % int(poly.shape[0])] - poly[idx]
        axis = np.asarray([-edge[1], edge[0]], dtype=np.float32)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-8:
            continue
        axis /= axis_norm
        proj_poly = poly @ axis
        proj_point = float(np.asarray(point, dtype=np.float32) @ axis)
        if proj_point < float(proj_poly.min()) or proj_point > float(proj_poly.max()):
            return False
    return True


def assignment_local_start_frame(char: dict) -> int:
    start_state = char.get("start_state")
    if isinstance(start_state, dict) and start_state.get("local_start") is not None:
        try:
            return int(start_state["local_start"])
        except (TypeError, ValueError):
            pass
    if isinstance(char.get("source_window"), dict):
        try:
            return int(char["source_window"]["local_start"])
        except (TypeError, ValueError, KeyError):
            pass
    goals = char.get("goal_sequence", [])
    if isinstance(goals, list) and goals:
        source = goals[0].get("source_segment", {})
        if isinstance(source, dict):
            raw = source.get("start")
            if raw is None:
                raw = source.get("interaction_frame")
            if raw is not None:
                try:
                    return int(raw)
                except (TypeError, ValueError):
                    pass
    return 0


def assignment_global_start_frame(char: dict) -> Optional[int]:
    start_state = char.get("start_state")
    if isinstance(start_state, dict) and start_state.get("global_frame_index") is not None:
        try:
            return int(start_state["global_frame_index"])
        except (TypeError, ValueError):
            pass
    goals = char.get("goal_sequence", [])
    if isinstance(goals, list) and goals:
        source = goals[0].get("source_segment", {})
        if isinstance(source, dict) and source.get("global_frame_index") is not None:
            try:
                return int(source["global_frame_index"])
            except (TypeError, ValueError):
                pass
    return None


def goal_relative_frame(
    char: dict,
    goal: dict,
    *,
    primary_key: str,
    fallback_keys: Sequence[str] = (),
    default_value: int = 0,
) -> int:
    source = goal.get("source_segment", {})
    raw = None
    if isinstance(source, dict):
        raw = source.get(primary_key)
        if raw is None:
            for key in fallback_keys:
                raw = source.get(key)
                if raw is not None:
                    break
    if raw is None:
        raw = default_value
    try:
        local_frame = int(raw)
    except (TypeError, ValueError):
        local_frame = int(default_value)
    return max(0, local_frame - assignment_local_start_frame(char))


def goal_global_frame(
    char: dict,
    goal: dict,
    *,
    primary_key: str,
    fallback_keys: Sequence[str] = (),
    default_value: int = 0,
) -> int:
    source = goal.get("source_segment", {})
    raw = None
    if isinstance(source, dict):
        if primary_key == "global_frame_index" and source.get("global_frame_index") is not None:
            try:
                return int(source["global_frame_index"])
            except (TypeError, ValueError):
                pass
        raw = source.get(primary_key)
        if raw is None:
            for key in fallback_keys:
                raw = source.get(key)
                if raw is not None:
                    break
    if raw is None:
        raw = default_value
    try:
        local_frame = int(raw)
    except (TypeError, ValueError):
        local_frame = int(default_value)

    global_start = assignment_global_start_frame(char)
    local_start = assignment_local_start_frame(char)
    if global_start is not None:
        return int(global_start + (local_frame - local_start))
    return local_frame


def detect_terminal_goal_blocking(
    character_assignments: list[dict],
    lateral_diameter_m: float,
    fore_aft_diameter_m: float,
) -> list[dict]:
    records: list[dict] = []
    lateral_diameter = float(lateral_diameter_m)
    fore_aft_diameter = float(fore_aft_diameter_m)
    if lateral_diameter <= 0.0 or fore_aft_diameter <= 0.0:
        return records

    terminal_occupancies: list[dict] = []
    for char in character_assignments:
        goals = char.get("goal_sequence", [])
        if not isinstance(goals, list) or not goals:
            continue
        final_goal: Optional[dict] = None
        final_body_goal: Optional[np.ndarray] = None
        final_idx = -1
        for idx in range(len(goals) - 1, -1, -1):
            body_goal = to_point3(goals[idx].get("body_goal"))
            if body_goal is not None:
                final_goal = goals[idx]
                final_body_goal = body_goal
                final_idx = idx
                break
        if final_goal is None or final_body_goal is None:
            continue

        center_xz = np.asarray(final_body_goal[[0, 2]], dtype=np.float32)
        terminal_occupancies.append(
            {
                "character_id": str(char.get("character_id", "unknown")),
                "goal_idx": int(final_idx),
                "goal": final_goal,
                "center_xz": center_xz,
                "poly": build_start_overlap_ellipse(
                    center_xz=center_xz,
                    global_orient_rotvec=final_goal.get("global_orient_rotvec"),
                    lateral_diameter_m=lateral_diameter,
                    fore_aft_diameter_m=fore_aft_diameter,
                ),
                "terminal_start_time": goal_global_frame(
                    char,
                    final_goal,
                    primary_key="start",
                    fallback_keys=("interaction_frame", "end"),
                    default_value=final_idx,
                ),
            }
        )

    for blocker in terminal_occupancies:
        for char in character_assignments:
            target_character_id = str(char.get("character_id", "unknown"))
            if target_character_id == blocker["character_id"]:
                continue
            goals = char.get("goal_sequence", [])
            if not isinstance(goals, list):
                continue
            for step_idx, goal in enumerate(goals):
                interaction_time = goal_global_frame(
                    char,
                    goal,
                    primary_key="interaction_frame",
                    fallback_keys=("end", "start"),
                    default_value=step_idx,
                )
                if blocker["terminal_start_time"] > interaction_time:
                    continue

                body_goal = to_point3(goal.get("body_goal"))
                hand_goal = to_point3(goal.get("hand_goal"))
                blocked_by_body = False
                blocked_by_hand = False
                distance = None
                if body_goal is not None:
                    goal_center_xz = np.asarray(body_goal[[0, 2]], dtype=np.float32)
                    distance = float(np.linalg.norm(goal_center_xz - blocker["center_xz"]))
                    goal_poly = build_start_overlap_ellipse(
                        center_xz=goal_center_xz,
                        global_orient_rotvec=goal.get("global_orient_rotvec"),
                        lateral_diameter_m=lateral_diameter,
                        fore_aft_diameter_m=fore_aft_diameter,
                    )
                    blocked_by_body = convex_polygons_overlap(blocker["poly"], goal_poly)

                if hand_goal is not None:
                    hand_xz = np.asarray(hand_goal[[0, 2]], dtype=np.float32)
                    blocked_by_hand = point_in_convex_polygon(hand_xz, blocker["poly"])

                if not blocked_by_body and not blocked_by_hand:
                    continue

                blocker_source = blocker["goal"].get("source_segment", {})
                target_source = goal.get("source_segment", {})
                records.append(
                    {
                        "blocking_character": blocker["character_id"],
                        "blocked_character": target_character_id,
                        "blocking_goal_idx": int(blocker["goal_idx"]),
                        "blocked_goal_idx": int(step_idx),
                        "blocking_sequence_id": blocker_source.get("sequence_id"),
                        "blocked_sequence_id": target_source.get("sequence_id"),
                        "blocking_segment_id": blocker_source.get("segment_id"),
                        "blocked_segment_id": target_source.get("segment_id"),
                        "blocking_goal_type": blocker["goal"].get("goal_type"),
                        "blocked_goal_type": goal.get("goal_type"),
                        "blocking_terminal_start_time": int(blocker["terminal_start_time"]),
                        "blocked_interaction_time": int(interaction_time),
                        "reason": "body_goal_overlap" if blocked_by_body else "hand_goal_inside_terminal_occupancy",
                        "distance_m": distance,
                        "location_xz": [
                            float(blocker["center_xz"][0]),
                            float(blocker["center_xz"][1]),
                        ],
                    }
                )
    return records


def start_forward_xz(global_orient_rotvec) -> np.ndarray:
    rotvec = to_point3(global_orient_rotvec)
    if rotvec is None:
        return np.array([0.0, 1.0], dtype=np.float32)
    try:
        forward = R.from_rotvec(np.asarray(rotvec, dtype=np.float32)).apply(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    except ValueError:
        return np.array([0.0, 1.0], dtype=np.float32)
    xz = np.asarray([forward[0], forward[2]], dtype=np.float32)
    norm = float(np.linalg.norm(xz))
    if norm < 1e-6 or not np.isfinite(norm):
        return np.array([0.0, 1.0], dtype=np.float32)
    return xz / norm


def build_start_overlap_ellipse(
    center_xz: np.ndarray,
    global_orient_rotvec,
    lateral_diameter_m: float,
    fore_aft_diameter_m: float,
    num_points: int = 32,
) -> np.ndarray:
    forward = start_forward_xz(global_orient_rotvec)
    side = np.asarray([-forward[1], forward[0]], dtype=np.float32)
    half_lr = 0.5 * float(lateral_diameter_m)
    half_fb = 0.5 * float(fore_aft_diameter_m)
    angles = np.linspace(0.0, 2.0 * np.pi, num=num_points, endpoint=False, dtype=np.float32)
    cos_t = np.cos(angles)[:, None]
    sin_t = np.sin(angles)[:, None]
    return (
        np.asarray(center_xz, dtype=np.float32)[None, :]
        + cos_t * (half_lr * side[None, :])
        + sin_t * (half_fb * forward[None, :])
    ).astype(np.float32)


def convex_polygons_overlap(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    for poly in (poly_a, poly_b):
        n = int(poly.shape[0])
        for idx in range(n):
            edge = poly[(idx + 1) % n] - poly[idx]
            axis = np.asarray([-edge[1], edge[0]], dtype=np.float32)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < 1e-8:
                continue
            axis /= axis_norm
            proj_a = poly_a @ axis
            proj_b = poly_b @ axis
            if float(proj_a.max()) < float(proj_b.min()) or float(proj_b.max()) < float(proj_a.min()):
                return False
    return True


def build_path_samples(character_assignments: list[dict]) -> list[dict]:
    samples: list[dict] = []
    for assignment in character_assignments:
        char_id = assignment.get("character_id", "unknown")
        for step_idx, goal in enumerate(assignment.get("goal_sequence", [])):
            body_goal = to_point3(goal.get("body_goal"))
            if body_goal is None:
                continue
            source = goal.get("source_segment", {})
            raw_time = source.get("global_frame_index")
            if raw_time is None:
                raw_time = source.get("interaction_frame")
            if raw_time is None:
                raw_time = step_idx
            try:
                time_idx = int(raw_time)
            except (TypeError, ValueError):
                time_idx = step_idx
            samples.append(
                {
                    "character_id": char_id,
                    "step_idx": int(step_idx),
                    "segment_id": int(source.get("segment_id", step_idx)),
                    "sequence_id": source.get("sequence_id"),
                    "time_idx": time_idx,
                    "body_goal_xz": np.array([body_goal[0], body_goal[2]], dtype=np.float32),
                    "goal": goal,
                }
            )
    return samples


def _pair_conflict_key(a: dict, b: dict, conflict_type: str) -> tuple:
    pair = tuple(sorted((str(a["character_id"]), str(b["character_id"]))))
    return (
        pair[0],
        pair[1],
        conflict_type,
        int(a.get("step_idx", 0)),
        int(b.get("step_idx", 0)),
    )


def classify_conflict(
    character_assignments: list[dict],
    path_threshold: float,
    goal_threshold: float,
) -> tuple[str, list[dict]]:
    path_samples = build_path_samples(character_assignments)
    if not path_samples:
        return "no_conflict", []

    path_conflict = False
    goal_conflict = False
    records: list[dict] = []
    seen: set[tuple] = set()

    by_char = {}
    for sample in path_samples:
        by_char.setdefault(sample["character_id"], []).append(sample)
    character_ids = list(by_char.keys())

    if len(character_ids) >= 2:
        for i in range(len(path_samples)):
            sample_a = path_samples[i]
            pos_a = sample_a["body_goal_xz"]
            for j in range(i + 1, len(path_samples)):
                sample_b = path_samples[j]
                if sample_a["character_id"] == sample_b["character_id"]:
                    continue
                pos_b = sample_b["body_goal_xz"]
                if not np.isfinite(pos_a).all() or not np.isfinite(pos_b).all():
                    continue
                distance = float(np.linalg.norm(pos_a - pos_b))
                if distance <= float(path_threshold):
                    key = _pair_conflict_key(sample_a, sample_b, "path")
                    if key not in seen:
                        records.append(
                            {
                                "conflict_type": "path",
                                "character_a": sample_a["character_id"],
                                "character_b": sample_b["character_id"],
                                "segment_a": int(sample_a["segment_id"]),
                                "segment_b": int(sample_b["segment_id"]),
                                "time_range": [int(min(sample_a["time_idx"], sample_b["time_idx"])), int(max(sample_a["time_idx"], sample_b["time_idx"]))],
                                "location": [
                                    float((float(pos_a[0]) + float(pos_b[0])) * 0.5),
                                    float((float(pos_a[1]) + float(pos_b[1])) * 0.5),
                                ],
                                "area_id": sample_a["goal"].get("fixed_cluster_id"),
                                "distance": distance,
                            }
                        )
                        seen.add(key)
                    path_conflict = True

    if len(character_ids) >= 2:
        goal_by_char: list[dict] = []
        for char_id, goals in by_char.items():
            if not goals:
                continue
            final_goal = goals[-1]
            body_goal = to_point3(final_goal["goal"].get("body_goal"))
            if body_goal is None:
                continue
            goal_by_char.append({"character_id": char_id, "sample": final_goal, "body_goal": body_goal})

        for i in range(len(goal_by_char)):
            a = goal_by_char[i]
            for j in range(i + 1, len(goal_by_char)):
                b = goal_by_char[j]
                area_a = a["sample"]["goal"].get("fixed_cluster_id")
                area_b = b["sample"]["goal"].get("fixed_cluster_id")
                if area_a is None:
                    area_a = a["sample"]["goal"].get("movable_seated_cluster_id")
                if area_b is None:
                    area_b = b["sample"]["goal"].get("movable_seated_cluster_id")
                if area_a is None:
                    area_a = a["sample"]["goal"].get("seated_cluster_id")
                if area_b is None:
                    area_b = b["sample"]["goal"].get("seated_cluster_id")

                same_goal_area = area_a is not None and area_b is not None and area_a == area_b
                distance = float(np.linalg.norm(a["body_goal"][[0, 2]] - b["body_goal"][[0, 2]]))
                if same_goal_area or distance <= float(goal_threshold):
                    key = _pair_conflict_key(a["sample"], b["sample"], "goal")
                    if key not in seen:
                        time_a = int(a["sample"].get("time_idx", 0))
                        time_b = int(b["sample"].get("time_idx", 0))
                        records.append(
                            {
                                "conflict_type": "goal",
                                "character_a": a["character_id"],
                                "character_b": b["character_id"],
                                "segment_a": int(a["sample"]["goal"]["source_segment"].get("segment_id", 0)),
                                "segment_b": int(b["sample"]["goal"]["source_segment"].get("segment_id", 0)),
                                "time_range": [min(time_a, time_b), max(time_a, time_b)],
                                "location": [
                                    float((float(a["body_goal"][0]) + float(b["body_goal"][0])) * 0.5),
                                    float((float(a["body_goal"][2]) + float(b["body_goal"][2])) * 0.5),
                                ],
                                "area_id": area_a or area_b,
                                "distance": distance,
                            }
                        )
                        seen.add(key)
                    goal_conflict = True

    if path_conflict and goal_conflict:
        conflict_type = "both"
    elif path_conflict:
        conflict_type = "path_conflict"
    elif goal_conflict:
        conflict_type = "goal_conflict"
    else:
        conflict_type = "no_conflict"

    return conflict_type, records


def segment_window_candidates(records: list[dict], size: int) -> List[Tuple[int, int]]:
    if size <= 0 or size > len(records):
        return []
    return [(i, i + size) for i in range(0, len(records) - size + 1)]


def select_indices(
    rng: random.Random,
    candidates: Sequence[int],
    k: int,
) -> list[int]:
    if len(candidates) < k:
        return list(candidates)
    return rng.sample(list(candidates), k=k)


def infer_support_key(rec: dict) -> Tuple[str, str]:
    obj_id = rec.get("acted_on_object_id") or rec.get("support_object_id")
    obj_name = canonical_name(rec.get("acted_on_object_name") or rec.get("support_object_name") or rec.get("target_name"))
    if obj_id is None:
        obj_id = obj_name
    if obj_id is None:
        obj_id = "unknown_object"
    return str(obj_id), str(obj_name or "unknown")


def support_event_kind(goal_type: str) -> str:
    if goal_type in SUPPORT_PICKUP_TYPES:
        return "pickup"
    if goal_type in SUPPORT_PUTDOWN_TYPES:
        return "putdown"
    return "support"


def build_support_needed_payload(all_goal_records: list[dict]) -> dict:
    support_entries: dict[Tuple[str, str], dict] = {}
    for rec in all_goal_records:
        goal_type = str(rec.get("goal_type", ""))
        if rec.get("goal_category") != "support-needed":
            continue
        obj_id, obj_name = infer_support_key(rec)
        entry_key = (obj_id, obj_name)
        if entry_key not in support_entries:
            support_entries[entry_key] = {
                "object_id": obj_id,
                "object_name": obj_name,
                "event_sequence": [],
                "first_pickup": None,
                "support_locations": [],
            }
        goal_entry = make_goal_entry(rec)
        event = support_event_kind(goal_type)
        payload = {"kind": event, "goal_type": goal_type, "goal": goal_entry}
        support_entries[entry_key]["event_sequence"].append(payload)

        if event == "pickup":
            first_pickup = support_entries[entry_key]["first_pickup"]
            if first_pickup is None:
                support_entries[entry_key]["first_pickup"] = payload
        elif event == "putdown":
            support_entries[entry_key]["support_locations"].append(goal_entry)

    for data in support_entries.values():
        data["event_sequence"] = sorted(
            data["event_sequence"],
            key=lambda x: int(x["goal"]["source_segment"].get("interaction_frame", 0)),
        )
        if data["event_sequence"]:
            data["first_pickup"] = (
                data["first_pickup"]
                or min(data["event_sequence"], key=lambda x: int(x["goal"]["source_segment"].get("interaction_frame", 0)))
            )

    return {
        "portable_object_goals": list(support_entries.values()),
    }


def sample_object_interaction_episode(
    rng: random.Random,
    scene_payload: dict,
    object_char_choices: List[int],
    segment_sizes: List[int],
    path_conflict_threshold: float,
    goal_conflict_threshold: float,
    num_char_override: Optional[int] = None,
) -> Optional[dict]:
    per_segment: list[dict] = scene_payload.get("per_segment_goal_list", [])
    if not per_segment:
        return None

    by_sequence: dict[str, list[dict]] = {}
    for rec in per_segment:
        if to_point3(rec.get("body_goal")) is None:
            continue
        by_sequence.setdefault(rec["sequence_id"], []).append(rec)
    for seq in by_sequence.values():
        seq.sort(key=lambda item: (int(item.get("start", 0)), int(item.get("segment_id", 0))))

    window_size = rng.choice(segment_sizes)
    window_size = max(1, int(window_size))
    sequence_candidates = [seq for seq, recs in by_sequence.items() if len(recs) >= window_size]
    if not sequence_candidates:
        return None

    valid_char_choices = [choice for choice in object_char_choices if int(choice) <= len(sequence_candidates)]
    if num_char_override is not None:
        if int(num_char_override) > len(sequence_candidates):
            return None
        num_char = int(num_char_override)
    elif valid_char_choices:
        num_char = rng.choice(valid_char_choices)
    else:
        num_char = len(sequence_candidates)
    selected_sequences = rng.sample(sequence_candidates, k=num_char)

    scene_common_fixed = scene_payload.get("fixed_cluster_list", [])
    scene_common_seated = scene_payload.get("seated_cluster_list", [])
    all_sequence_records: list[dict] = []
    used_movable_ids = set()

    for char_idx, sequence_id in enumerate(selected_sequences):
        records = by_sequence[sequence_id]
        windows = segment_window_candidates(records, window_size)
        if not windows:
            continue
        start, end = rng.choice(windows)
        window_records = records[start:end]
        goals = [make_goal_entry(rec) for rec in window_records]
        for rec in window_records:
            if rec.get("movable_seated_cluster_id") is not None:
                used_movable_ids.add(rec["movable_seated_cluster_id"])
        all_sequence_records.append(
            {
                "character_id": f"char_{char_idx:02d}",
                "sequence_id": sequence_id,
                "window": {
                    "start_segment_id": int(window_records[0]["segment_id"]),
                    "end_segment_id": int(window_records[-1]["segment_id"]),
                    "window_size": int(window_size),
                },
                "goal_sequence": goals,
            }
        )

    if not all_sequence_records:
        return None

    movable_seated = [
        cluster
        for cluster in scene_payload.get("movable_seated_cluster_list", [])
        if cluster.get("cluster_id") in used_movable_ids
    ]

    all_goals: list[dict] = []
    for char in all_sequence_records:
        all_goals.extend(char["goal_sequence"])
    support_payload = build_support_needed_payload(all_goals)
    conflict_type, conflict_records = classify_conflict(
        character_assignments=all_sequence_records,
        path_threshold=path_conflict_threshold,
        goal_threshold=goal_conflict_threshold,
    )
    return {
        "scenario_type": "object_interaction",
        "num_characters": len(all_sequence_records),
        "character_assignments": all_sequence_records,
        "source_sequences": sorted(seq["sequence_id"] for seq in all_sequence_records),
        "conflict_type": conflict_type,
        "conflict_records": conflict_records,
        "scene_common": {
            "seated_cluster_list": scene_common_seated,
            "movable_seated_cluster_list": movable_seated,
            "fixed_cluster_list": scene_common_fixed,
        },
        "support_needed": support_payload,
    }


def sample_locomotion_episode(
    rng: random.Random,
    scene_payload: dict,
    loco_char_choices: List[int],
    loco_goal_numbers: List[int],
    path_conflict_threshold: float,
    goal_conflict_threshold: float,
    num_char_override: Optional[int] = None,
) -> Optional[dict]:
    locomotion_sources = scene_payload.get("locomotion_source_list", [])
    if not locomotion_sources:
        return None

    window_frames = 600
    num_char = int(num_char_override) if num_char_override is not None else int(rng.choice(loco_char_choices))
    goal_count = max(1, int(rng.choice(loco_goal_numbers)))

    all_characters: list[dict] = []
    for _ in range(num_char):
        source = rng.choice(locomotion_sources)
        total_frames = max(1, int(source.get("total_frames", 0)))
        sequence_id = str(source["sequence_id"])
        if total_frames <= window_frames:
            local_start = 0
            local_end = total_frames - 1
        else:
            max_start = total_frames - window_frames
            local_start = rng.randint(0, max_start)
            local_end = local_start + window_frames - 1
        all_characters.append(
            {
                "character_id": f"char_{len(all_characters):02d}",
                "goal_count": int(goal_count),
                "source_window": {
                    "sequence_id": sequence_id,
                    "window_id": f"{sequence_id}__rand_{local_start:05d}_{local_end:05d}",
                    "local_start": int(local_start),
                    "local_end": int(local_end),
                    "length": int(local_end - local_start + 1),
                    "overlapping_segment_ids": [],
                },
                "goal_sequence": [],
            }
        )

    if not all_characters:
        return None

    all_goals: list[dict] = []
    for character in all_characters:
        all_goals.extend(character["goal_sequence"])
    support_payload = build_support_needed_payload(all_goals)
    conflict_type, conflict_records = classify_conflict(
        character_assignments=all_characters,
        path_threshold=path_conflict_threshold,
        goal_threshold=goal_conflict_threshold,
    )
    return {
        "scenario_type": "locomotion",
        "num_characters": len(all_characters),
        "character_assignments": all_characters,
        "source_sequences": sorted(set(ch["source_window"]["sequence_id"] for ch in all_characters)),
        "conflict_type": conflict_type,
        "conflict_records": conflict_records,
        "scene_common": {
            "seated_cluster_list": [],
            "movable_seated_cluster_list": [],
            "fixed_cluster_list": [],
        },
        "support_needed": support_payload,
    }


def sample_episode_for_scene(
    rng: random.Random,
    scene_payload: dict,
    object_char_choices: List[int],
    segment_sizes: List[int],
    loco_char_choices: List[int],
    loco_goal_numbers: List[int],
    path_conflict_threshold: float,
    goal_conflict_threshold: float,
    num_char_override: Optional[int] = None,
) -> Optional[dict]:
    scene_kind = scene_payload.get("scene_kind", "object_interaction")
    if scene_kind == "locomotion":
        return sample_locomotion_episode(
            rng,
            scene_payload,
            loco_char_choices=loco_char_choices,
            loco_goal_numbers=loco_goal_numbers,
            path_conflict_threshold=path_conflict_threshold,
            goal_conflict_threshold=goal_conflict_threshold,
            num_char_override=num_char_override,
        )
    return sample_object_interaction_episode(
        rng,
        scene_payload,
        object_char_choices=object_char_choices,
        segment_sizes=segment_sizes,
        path_conflict_threshold=path_conflict_threshold,
        goal_conflict_threshold=goal_conflict_threshold,
        num_char_override=num_char_override,
    )


def episode_signature(payload: dict) -> tuple:
    scenario_type = str(payload.get("scenario_type", "unknown"))
    assignments = payload.get("character_assignments", [])
    parts: list[tuple] = []
    for char in assignments:
        if scenario_type == "locomotion":
            window = char.get("source_window", {})
            parts.append(
                (
                    "loco",
                    str(window.get("sequence_id")),
                    int(window.get("local_start", -1)),
                    int(window.get("local_end", -1)),
                    int(char.get("goal_count", 0)),
                )
            )
        else:
            window = char.get("window", {})
            parts.append(
                (
                    "obj",
                    str(char.get("sequence_id")),
                    int(window.get("start_segment_id", -1)),
                    int(window.get("end_segment_id", -1)),
                    int(window.get("window_size", 0)),
                )
            )
    return (scenario_type, int(payload.get("num_characters", 0)), tuple(sorted(parts)))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    scene_ids = resolve_scene_ids(args)
    output_root = args.output_root / args.dataset / "episodes_v2"
    output_root.mkdir(parents=True, exist_ok=True)

    object_char_choices = parse_int_list(args.object_char_choices, DEFAULT_CHAR_CHOICES, min_value=1)
    object_segment_sizes = parse_int_list(args.object_segment_sizes, DEFAULT_OBJECT_WINDOW_SIZES, min_value=1)
    loco_char_choices = parse_int_list(args.loco_char_choices, DEFAULT_CHAR_CHOICES, min_value=1)
    loco_goal_numbers = parse_int_list(args.loco_goal_numbers, DEFAULT_LOCO_GOAL_NUMBERS, min_value=1)
    desired_char_counts = sorted(set(object_char_choices + loco_char_choices))
    episodes_per_character_count = max(0, int(args.episodes_per_character_count))
    target_counts_by_num = {int(num): episodes_per_character_count for num in desired_char_counts}
    target_episodes_per_scene = int(args.episodes_per_scene)
    if target_counts_by_num:
        target_episodes_per_scene = sum(target_counts_by_num.values())
    max_attempts_per_scene = args.max_attempts_per_scene or max(int(target_episodes_per_scene) * 200, 500)
    if args.start_overlap_diameter_m is not None:
        start_overlap_lateral_diameter = float(args.start_overlap_diameter_m)
        start_overlap_fore_aft_diameter = float(args.start_overlap_diameter_m)
    else:
        start_overlap_lateral_diameter = float(args.start_overlap_lateral_diameter_m)
        start_overlap_fore_aft_diameter = float(args.start_overlap_fore_aft_diameter_m)

    sequence_cache: Dict[Tuple[str, str, str], dict] = {}
    array_cache: Dict[Path, np.ndarray] = {}

    total_written = 0
    for scene_id in scene_ids:
        scene_root = output_root / scene_id
        scene_root.mkdir(parents=True, exist_ok=True)
        for stale_path in scene_root.glob("episode_*.json"):
            stale_path.unlink()
        bank = load_scene_bank(args.output_root, args.dataset, scene_id)
        written_for_scene = 0
        attempts = 0
        written_by_num: Dict[int, int] = {int(num): 0 for num in target_counts_by_num}
        seen_signatures: set[tuple] = set()
        start_overlap_rejections = 0
        final_goal_overlap_rejections = 0
        terminal_goal_block_rejections = 0
        duplicate_rejections = 0
        while written_for_scene < int(target_episodes_per_scene) and attempts < max_attempts_per_scene:
            attempts += 1
            underfilled_num_chars = [
                int(num)
                for num, target in target_counts_by_num.items()
                if int(written_by_num.get(int(num), 0)) < int(target)
            ]
            if not underfilled_num_chars:
                break
            target_num_char = int(rng.choice(underfilled_num_chars))
            payload = sample_episode_for_scene(
                rng=rng,
                scene_payload=bank,
                object_char_choices=object_char_choices,
                segment_sizes=object_segment_sizes,
                loco_char_choices=loco_char_choices,
                loco_goal_numbers=loco_goal_numbers,
                path_conflict_threshold=args.path_conflict_threshold_m,
                goal_conflict_threshold=args.goal_conflict_threshold_m,
                num_char_override=target_num_char,
            )
            if payload is None:
                continue
            if int(payload.get("num_characters", 0)) != target_num_char:
                continue
            signature = episode_signature(payload)
            if signature in seen_signatures:
                duplicate_rejections += 1
                continue

            attach_start_states(
                dataset=args.dataset,
                scene_id=scene_id,
                character_assignments=payload["character_assignments"],
                sequence_cache=sequence_cache,
                array_cache=array_cache,
            )
            attach_locomotion_window_terminal_goals(
                dataset=args.dataset,
                scene_id=scene_id,
                character_assignments=payload["character_assignments"],
                sequence_cache=sequence_cache,
                array_cache=array_cache,
            )
            payload["conflict_type"], payload["conflict_records"] = classify_conflict(
                character_assignments=payload["character_assignments"],
                path_threshold=args.path_conflict_threshold_m,
                goal_threshold=args.goal_conflict_threshold_m,
            )
            start_overlap_records = detect_start_overlaps(
                character_assignments=payload["character_assignments"],
                lateral_diameter_m=start_overlap_lateral_diameter,
                fore_aft_diameter_m=start_overlap_fore_aft_diameter,
            )
            if start_overlap_records:
                start_overlap_rejections += 1
                continue
            final_goal_overlap_records = detect_final_goal_overlaps(
                character_assignments=payload["character_assignments"],
                lateral_diameter_m=start_overlap_lateral_diameter,
                fore_aft_diameter_m=start_overlap_fore_aft_diameter,
            )
            if final_goal_overlap_records:
                final_goal_overlap_rejections += 1
                continue
            terminal_goal_block_records = detect_terminal_goal_blocking(
                character_assignments=payload["character_assignments"],
                lateral_diameter_m=start_overlap_lateral_diameter,
                fore_aft_diameter_m=start_overlap_fore_aft_diameter,
            )
            if terminal_goal_block_records:
                terminal_goal_block_rejections += 1
                continue

            payload["scene_id"] = scene_id
            payload["dataset"] = args.dataset
            payload["episode_id"] = f"{scene_id}_{payload['scenario_type']}_{written_for_scene:04d}"
            payload["split"] = args.split
            payload["seed"] = args.seed
            is_circle = abs(float(start_overlap_lateral_diameter) - float(start_overlap_fore_aft_diameter)) < 1e-6
            payload["start_overlap_shape"] = (
                {
                    "type": "circle",
                    "diameter_m": float(start_overlap_lateral_diameter),
                }
                if is_circle
                else {
                    "type": "ellipse",
                    "lateral_diameter_m": float(start_overlap_lateral_diameter),
                    "fore_aft_diameter_m": float(start_overlap_fore_aft_diameter),
                }
            )
            payload["final_goal_overlap_shape"] = (
                {
                    "type": "circle",
                    "diameter_m": float(start_overlap_lateral_diameter),
                }
                if is_circle
                else {
                    "type": "ellipse",
                    "lateral_diameter_m": float(start_overlap_lateral_diameter),
                    "fore_aft_diameter_m": float(start_overlap_fore_aft_diameter),
                }
            )
            payload["terminal_occupancy_shape"] = (
                {
                    "type": "circle",
                    "diameter_m": float(start_overlap_lateral_diameter),
                }
                if is_circle
                else {
                    "type": "ellipse",
                    "lateral_diameter_m": float(start_overlap_lateral_diameter),
                    "fore_aft_diameter_m": float(start_overlap_fore_aft_diameter),
                }
            )
            out_path = scene_root / f"episode_{written_for_scene:04d}.json"
            out_path.write_text(json.dumps(payload, indent=2))
            total_written += 1
            written_for_scene += 1
            written_by_num[target_num_char] = int(written_by_num.get(target_num_char, 0)) + 1
            seen_signatures.add(signature)
        print(
            f"{scene_id}: wrote {written_for_scene}/{target_episodes_per_scene} episodes "
            f"after {attempts} attempts "
            f"({start_overlap_rejections} rejected by start overlap, "
            f"{final_goal_overlap_rejections} rejected by final goal overlap, "
            f"{terminal_goal_block_rejections} rejected by terminal occupancy, "
            f"{duplicate_rejections} rejected as duplicate) "
            f"counts={{{', '.join(f'{num}:{written_by_num.get(num, 0)}' for num in sorted(written_by_num))}}}"
        )
    print(f"wrote {total_written} episodes under {output_root.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
