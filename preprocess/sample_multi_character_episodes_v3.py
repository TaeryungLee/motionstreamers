from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from build_multi_character_affordances import PROJECT_ROOT, load_json, resolve_scene_ids
from sample_multi_character_episodes import (
    DEFAULT_CHAR_CHOICES,
    DEFAULT_OBJECT_WINDOW_SIZES,
    build_support_needed_payload,
    classify_conflict,
    detect_start_overlaps,
    detect_terminal_goal_blocking,
    episode_signature,
    make_goal_entry,
    parse_int_list,
    segment_window_candidates,
    to_point3,
)


DEFAULT_LOCO_GOAL_NUMBERS = [1]
SOURCE_SCENES_NAME = "scenes_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v3 multi-character episodes with deterministic ours-first coverage.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--scene-list-file", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "preprocessed")
    parser.add_argument("--episode-bank-name", default="episode_bank_v3")
    parser.add_argument("--episodes-name", default="episodes_v3")
    parser.add_argument("--max-episodes-per-ours", type=int, default=4)
    parser.add_argument("--object-char-choices", type=str, default="2,3,4")
    parser.add_argument("--object-segment-sizes", type=str, default="4,5")
    parser.add_argument("--loco-char-choices", type=str, default="2,3,4")
    parser.add_argument("--loco-goal-numbers", type=str, default="1")
    parser.add_argument("--path-conflict-threshold-m", type=float, default=0.50)
    parser.add_argument("--goal-conflict-threshold-m", type=float, default=0.50)
    parser.add_argument("--overlap-diameter-m", type=float, default=0.50)
    parser.add_argument("--near-distance-m", type=float, default=1.2)
    parser.add_argument("--near-quota-per-ours", type=int, default=2)
    parser.add_argument("--max-other-combinations-per-ours", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_scene_bank(output_root: Path, dataset: str, scene_id: str, episode_bank_name: str) -> dict:
    path = output_root / dataset / episode_bank_name / scene_id / "scene_static.json"
    if not path.exists():
        raise FileNotFoundError(f"scene bank not found: {path}")
    return load_json(path)


def stable_scene_seed(seed: int, scene_id: str) -> int:
    value = int(seed)
    for ch in str(scene_id):
        value = (value * 131 + ord(ch)) % 1000000007
    return value


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_sequence_payload_cached_v3(
    dataset: str,
    scene_id: str,
    sequence_id: str,
    sequence_cache: Dict[Tuple[str, str, str], dict],
) -> dict:
    key = (dataset, scene_id, sequence_id)
    cached = sequence_cache.get(key)
    if cached is not None:
        return cached
    path = PROJECT_ROOT / "data" / "preprocessed" / dataset / SOURCE_SCENES_NAME / scene_id / "sequences" / f"{sequence_id}.json"
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


def infer_character_start_frame(char: dict) -> Tuple[str, int]:
    if char.get("source_window") is not None:
        window = char["source_window"]
        return str(window["sequence_id"]), int(window["local_start"])
    goals = char.get("goal_sequence", [])
    if not goals:
        raise ValueError(f"Character {char.get('character_id', 'unknown')} has no goal_sequence.")
    source = goals[0].get("source_segment", {})
    return str(source["sequence_id"]), int(source["start"])


def attach_start_states_v3(
    dataset: str,
    scene_id: str,
    character_assignments: list[dict],
    sequence_cache: Dict[Tuple[str, str, str], dict],
    array_cache: Dict[Path, np.ndarray],
) -> None:
    for char in character_assignments:
        sequence_id, local_start = infer_character_start_frame(char)
        sequence_payload = load_sequence_payload_cached_v3(dataset, scene_id, sequence_id, sequence_cache)
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


def attach_locomotion_window_terminal_goals_v3(
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
        sequence_payload = load_sequence_payload_cached_v3(dataset, scene_id, sequence_id, sequence_cache)
        motion_ref = sequence_payload["human_motion_ref"]
        smpl = motion_ref["smplx"]
        seq_start = int(motion_ref["start"])
        transl_arr = load_npy_cached(str(smpl["transl_path"]), array_cache)
        global_orient_arr = load_npy_cached(str(smpl["global_orient_path"]), array_cache)
        goal_count = max(1, int(char.get("goal_count", 1)))
        if local_end <= local_start:
            goal_local_frames = [int(local_end)]
        else:
            goal_local_frames = np.linspace(local_start, local_end, num=goal_count + 1, endpoint=True, dtype=np.int32)[1:].tolist()
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
                        "text": f"Move to goal {goal_idx}",
                    },
                }
            )
            previous_local = int(local_frame)
        char["goal_sequence"] = goals


def sort_key(value) -> tuple:
    if isinstance(value, dict):
        return tuple((str(k), sort_key(v)) for k, v in sorted(value.items(), key=lambda item: str(item[0])))
    if isinstance(value, list):
        return tuple(sort_key(v) for v in value)
    return (str(value),)


def build_object_candidates(scene_payload: dict, segment_sizes: Sequence[int]) -> list[dict]:
    by_sequence: dict[str, list[dict]] = {}
    for rec in scene_payload.get("per_segment_goal_list", []):
        if to_point3(rec.get("body_goal")) is None:
            continue
        by_sequence.setdefault(str(rec["sequence_id"]), []).append(rec)
    for records in by_sequence.values():
        records.sort(key=lambda item: (int(item.get("start", 0)), int(item.get("segment_id", 0))))

    candidates: list[dict] = []
    for window_size in sorted(set(int(v) for v in segment_sizes)):
        if window_size <= 0:
            continue
        for sequence_id in sorted(by_sequence):
            records = by_sequence[sequence_id]
            for start_idx, end_idx in segment_window_candidates(records, window_size):
                window_records = records[start_idx:end_idx]
                if len(window_records) != window_size:
                    continue
                key = (
                    "obj",
                    sequence_id,
                    int(window_records[0]["segment_id"]),
                    int(window_records[-1]["segment_id"]),
                    int(window_size),
                )
                candidates.append(
                    {
                        "kind": "object",
                        "key": key,
                        "sequence_id": sequence_id,
                        "window_size": int(window_size),
                        "window_records": window_records,
                    }
                )
    return sorted(candidates, key=lambda item: item["key"])


def build_locomotion_candidates(scene_payload: dict, goal_numbers: Sequence[int]) -> list[dict]:
    candidates: list[dict] = []
    for source in sorted(scene_payload.get("locomotion_source_list", []), key=lambda item: str(item.get("sequence_id"))):
        sequence_id = str(source["sequence_id"])
        windows = source.get("candidate_windows") or []
        if not windows:
            total_frames = max(1, int(source.get("total_frames", 0)))
            windows = [
                {
                    "window_id": f"{sequence_id}__full",
                    "local_start": 0,
                    "local_end": total_frames - 1,
                    "length": total_frames,
                    "overlapping_segment_ids": [],
                }
            ]
        for window in sorted(windows, key=lambda item: (int(item.get("local_start", 0)), str(item.get("window_id", "")))):
            for goal_count in sorted(set(int(v) for v in goal_numbers)):
                if goal_count <= 0:
                    continue
                key = (
                    "loco",
                    sequence_id,
                    str(window.get("window_id", "")),
                    int(window.get("local_start", 0)),
                    int(window.get("local_end", 0)),
                    int(goal_count),
                )
                candidates.append(
                    {
                        "kind": "locomotion",
                        "key": key,
                        "sequence_id": sequence_id,
                        "window": window,
                        "goal_count": int(goal_count),
                    }
                )
    return sorted(candidates, key=lambda item: item["key"])


def candidate_usage_key(candidate: dict) -> tuple:
    return tuple(candidate["key"])


def goal_body_xz(goal: dict) -> Optional[np.ndarray]:
    body_goal = to_point3(goal.get("body_goal"))
    if body_goal is None:
        return None
    return np.asarray(body_goal[[0, 2]], dtype=np.float32)


def character_sample_points_xz(char: dict) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    start_state = char.get("start_state")
    if isinstance(start_state, dict):
        start_body = to_point3(start_state.get("body_translation"))
        if start_body is not None:
            points.append(np.asarray(start_body[[0, 2]], dtype=np.float32))
    for goal in char.get("goal_sequence", []):
        point = goal_body_xz(goal)
        if point is not None:
            points.append(point)
    return points


def episode_ours_others_min_distance(payload: dict) -> float:
    assignments = payload.get("character_assignments", [])
    if not assignments:
        return float("inf")
    ego_id = str(payload.get("ego_character_id") or "char_00")
    ours_char = None
    other_chars: list[dict] = []
    for char in assignments:
        if str(char.get("character_id")) == ego_id:
            ours_char = char
        else:
            other_chars.append(char)
    if ours_char is None:
        ours_char = assignments[0]
        other_chars = assignments[1:]
    ours_points = character_sample_points_xz(ours_char)
    if not ours_points:
        return float("inf")
    best = float("inf")
    for other in other_chars:
        other_points = character_sample_points_xz(other)
        for ours_point in ours_points:
            for other_point in other_points:
                distance = float(np.linalg.norm(ours_point - other_point))
                if distance < best:
                    best = distance
    return best


def make_object_character(candidate: dict, character_id: str) -> dict:
    window_records = candidate["window_records"]
    goals = [make_goal_entry(rec) for rec in window_records]
    return {
        "character_id": character_id,
        "sequence_id": candidate["sequence_id"],
        "window": {
            "start_segment_id": int(window_records[0]["segment_id"]),
            "end_segment_id": int(window_records[-1]["segment_id"]),
            "window_size": int(candidate["window_size"]),
        },
        "goal_sequence": goals,
    }


def make_locomotion_character(candidate: dict, character_id: str) -> dict:
    window = candidate["window"]
    return {
        "character_id": character_id,
        "goal_count": int(candidate["goal_count"]),
        "source_window": {
            "sequence_id": str(candidate["sequence_id"]),
            "window_id": str(window.get("window_id") or f"{candidate['sequence_id']}__{int(window.get('local_start', 0)):05d}"),
            "local_start": int(window.get("local_start", 0)),
            "local_end": int(window.get("local_end", 0)),
            "length": int(window.get("length", int(window.get("local_end", 0)) - int(window.get("local_start", 0)) + 1)),
            "overlapping_segment_ids": list(window.get("overlapping_segment_ids", [])),
        },
        "goal_sequence": [],
    }


def make_character(candidate: dict, character_id: str) -> dict:
    if candidate["kind"] == "locomotion":
        return make_locomotion_character(candidate, character_id)
    return make_object_character(candidate, character_id)


def build_payload(scene_payload: dict, ours: dict, others: Sequence[dict], path_threshold: float, goal_threshold: float) -> dict:
    assignments = [make_character(ours, "char_00")]
    for idx, candidate in enumerate(others, start=1):
        assignments.append(make_character(candidate, f"char_{idx:02d}"))

    if ours["kind"] == "locomotion":
        scene_common = {
            "seated_cluster_list": [],
            "movable_seated_cluster_list": [],
            "fixed_cluster_list": [],
        }
        source_sequences = sorted(set(ch["source_window"]["sequence_id"] for ch in assignments))
        scenario_type = "locomotion"
    else:
        used_movable_ids = set()
        all_window_records = []
        for candidate in [ours, *others]:
            all_window_records.extend(candidate.get("window_records", []))
        for rec in all_window_records:
            if rec.get("movable_seated_cluster_id") is not None:
                used_movable_ids.add(rec["movable_seated_cluster_id"])
        scene_common = {
            "seated_cluster_list": scene_payload.get("seated_cluster_list", []),
            "movable_seated_cluster_list": [
                cluster
                for cluster in scene_payload.get("movable_seated_cluster_list", [])
                if cluster.get("cluster_id") in used_movable_ids
            ],
            "fixed_cluster_list": scene_payload.get("fixed_cluster_list", []),
        }
        source_sequences = sorted(set(str(ch["sequence_id"]) for ch in assignments))
        scenario_type = "object_interaction"

    all_goals: list[dict] = []
    for char in assignments:
        all_goals.extend(char.get("goal_sequence", []))
    conflict_type, conflict_records = classify_conflict(assignments, path_threshold, goal_threshold)
    return {
        "scenario_type": scenario_type,
        "num_characters": len(assignments),
        "ego_character_id": "char_00",
        "others_character_ids": [char["character_id"] for char in assignments[1:]],
        "ours_candidate_key": list(ours["key"]),
        "character_assignments": assignments,
        "source_sequences": source_sequences,
        "conflict_type": conflict_type,
        "conflict_records": conflict_records,
        "scene_common": scene_common,
        "support_needed": build_support_needed_payload(all_goals),
    }


def shuffled_prefix_combinations(
    candidates: Sequence[dict],
    k: int,
    *,
    rng: random.Random,
    usage_counts: Dict[tuple, int],
    limit: int,
) -> list[tuple[dict, ...]]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            int(usage_counts.get(candidate_usage_key(item), 0)),
            sort_key(item["key"]),
        ),
    )
    if len(ranked) > 48:
        head = ranked[:48]
        tail = ranked[48:]
        rng.shuffle(tail)
        ranked = head + tail[:48]
    combos = list(combinations(ranked, k))
    combos.sort(
        key=lambda combo: (
            sum(int(usage_counts.get(candidate_usage_key(item), 0)) for item in combo),
            tuple(sort_key(item["key"]) for item in combo),
        )
    )
    return combos[: max(1, int(limit))]


def is_valid_episode(
    payload: dict,
    *,
    dataset: str,
    scene_id: str,
    sequence_cache: Dict[Tuple[str, str, str], dict],
    array_cache: Dict[Path, np.ndarray],
    overlap_diameter_m: float,
) -> tuple[bool, dict[str, int]]:
    attach_start_states_v3(dataset, scene_id, payload["character_assignments"], sequence_cache, array_cache)
    attach_locomotion_window_terminal_goals_v3(dataset, scene_id, payload["character_assignments"], sequence_cache, array_cache)
    stats = {"start_overlap": 0, "terminal_occupancy": 0}
    start_records = detect_start_overlaps(
        payload["character_assignments"],
        lateral_diameter_m=overlap_diameter_m,
        fore_aft_diameter_m=overlap_diameter_m,
    )
    if start_records:
        payload["start_overlap_records"] = start_records
        stats["start_overlap"] = 1
        return False, stats
    terminal_records = detect_terminal_goal_blocking(
        payload["character_assignments"],
        lateral_diameter_m=overlap_diameter_m,
        fore_aft_diameter_m=overlap_diameter_m,
    )
    if terminal_records:
        payload["terminal_occupancy_records"] = terminal_records
        stats["terminal_occupancy"] = 1
        return False, stats
    return True, stats


def write_episode(
    payload: dict,
    *,
    out_dir: Path,
    scene_id: str,
    dataset: str,
    split: str,
    seed: int,
    episode_idx: int,
    overlap_diameter_m: float,
) -> Path:
    payload["scene_id"] = scene_id
    payload["dataset"] = dataset
    payload["episode_id"] = f"{scene_id}_{payload['scenario_type']}_{episode_idx:05d}"
    payload["split"] = split
    payload["seed"] = seed
    payload["start_overlap_shape"] = {"type": "circle", "diameter_m": float(overlap_diameter_m)}
    payload["terminal_occupancy_shape"] = {"type": "circle", "diameter_m": float(overlap_diameter_m)}
    out_path = out_dir / f"episode_{episode_idx:05d}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def process_scene(payload: tuple) -> dict:
    (
        dataset,
        scene_id,
        output_root,
        episode_bank_name,
        episodes_name,
        max_episodes_per_ours,
        object_char_choices,
        object_segment_sizes,
        loco_char_choices,
        loco_goal_numbers,
        path_conflict_threshold,
        goal_conflict_threshold,
        overlap_diameter_m,
        near_distance_m,
        near_quota_per_ours,
        max_other_combinations_per_ours,
        seed,
        split,
    ) = payload
    rng = random.Random(stable_scene_seed(int(seed), scene_id))
    bank = load_scene_bank(output_root, dataset, scene_id, episode_bank_name)
    scene_kind = str(bank.get("scene_kind", "object_interaction"))
    if scene_kind == "locomotion":
        all_candidates = build_locomotion_candidates(bank, loco_goal_numbers)
        char_choices = sorted(set(int(v) for v in loco_char_choices), reverse=True)
    else:
        all_candidates = build_object_candidates(bank, object_segment_sizes)
        char_choices = sorted(set(int(v) for v in object_char_choices), reverse=True)

    scene_root = output_root / dataset / episodes_name / scene_id
    scene_root.mkdir(parents=True, exist_ok=True)
    for stale_path in scene_root.glob("episode_*.json"):
        stale_path.unlink()

    sequence_cache: Dict[Tuple[str, str, str], dict] = {}
    array_cache: Dict[Path, np.ndarray] = {}
    usage_counts: Dict[tuple, int] = {}
    seen_signatures: set[tuple] = set()
    written = 0
    rejection_counts = {"start_overlap": 0, "terminal_occupancy": 0, "duplicate": 0, "insufficient_others": 0}
    episodes_by_num: Dict[int, int] = {}
    near_episode_count = 0
    any_episode_count = 0

    for ours in all_candidates:
        accepted_for_ours: list[tuple[dict, tuple[dict, ...], int, bool, float]] = []
        near_for_ours = 0
        for num_chars in char_choices:
            if len(accepted_for_ours) >= max_episodes_per_ours:
                break
            if num_chars < 1:
                continue
            num_others = int(num_chars) - 1
            others_pool = [
                candidate
                for candidate in all_candidates
                if candidate_usage_key(candidate) != candidate_usage_key(ours)
                and str(candidate.get("sequence_id")) != str(ours.get("sequence_id"))
            ]
            if len(others_pool) < num_others:
                rejection_counts["insufficient_others"] += 1
                continue
            for others in shuffled_prefix_combinations(
                others_pool,
                num_others,
                rng=rng,
                usage_counts=usage_counts,
                limit=max_other_combinations_per_ours,
            ):
                if len(accepted_for_ours) >= max_episodes_per_ours:
                    break
                if len({str(item.get("sequence_id")) for item in [ours, *others]}) != int(num_chars):
                    continue
                ep_payload = build_payload(
                    bank,
                    ours,
                    others,
                    path_threshold=path_conflict_threshold,
                    goal_threshold=goal_conflict_threshold,
                )
                signature = episode_signature(ep_payload)
                if signature in seen_signatures:
                    rejection_counts["duplicate"] += 1
                    continue
                valid, stats = is_valid_episode(
                    ep_payload,
                    dataset=dataset,
                    scene_id=scene_id,
                    sequence_cache=sequence_cache,
                    array_cache=array_cache,
                    overlap_diameter_m=overlap_diameter_m,
                )
                rejection_counts["start_overlap"] += int(stats.get("start_overlap", 0))
                rejection_counts["terminal_occupancy"] += int(stats.get("terminal_occupancy", 0))
                if not valid:
                    continue
                min_distance = episode_ours_others_min_distance(ep_payload)
                is_near = bool(np.isfinite(min_distance) and min_distance <= float(near_distance_m))
                near_quota = min(max_episodes_per_ours, max(0, int(near_quota_per_ours)))
                if not is_near and len(accepted_for_ours) < near_quota and near_for_ours < near_quota:
                    continue
                ep_payload["conflict_type"], ep_payload["conflict_records"] = classify_conflict(
                    ep_payload["character_assignments"],
                    path_conflict_threshold,
                    goal_conflict_threshold,
                )
                seen_signatures.add(signature)
                accepted_for_ours.append((ep_payload, tuple(others), int(num_chars), is_near, float(min_distance)))
                if is_near:
                    near_for_ours += 1
                for candidate in others:
                    key = candidate_usage_key(candidate)
                    usage_counts[key] = int(usage_counts.get(key, 0)) + 1
        if len(accepted_for_ours) < max_episodes_per_ours:
            for num_chars in char_choices:
                if len(accepted_for_ours) >= max_episodes_per_ours:
                    break
                num_others = int(num_chars) - 1
                others_pool = [
                    candidate
                    for candidate in all_candidates
                    if candidate_usage_key(candidate) != candidate_usage_key(ours)
                    and str(candidate.get("sequence_id")) != str(ours.get("sequence_id"))
                ]
                if len(others_pool) < num_others:
                    continue
                for others in shuffled_prefix_combinations(
                    others_pool,
                    num_others,
                    rng=rng,
                    usage_counts=usage_counts,
                    limit=max_other_combinations_per_ours,
                ):
                    if len(accepted_for_ours) >= max_episodes_per_ours:
                        break
                    if len({str(item.get("sequence_id")) for item in [ours, *others]}) != int(num_chars):
                        continue
                    ep_payload = build_payload(
                        bank,
                        ours,
                        others,
                        path_threshold=path_conflict_threshold,
                        goal_threshold=goal_conflict_threshold,
                    )
                    signature = episode_signature(ep_payload)
                    if signature in seen_signatures:
                        rejection_counts["duplicate"] += 1
                        continue
                    valid, stats = is_valid_episode(
                        ep_payload,
                        dataset=dataset,
                        scene_id=scene_id,
                        sequence_cache=sequence_cache,
                        array_cache=array_cache,
                        overlap_diameter_m=overlap_diameter_m,
                    )
                    rejection_counts["start_overlap"] += int(stats.get("start_overlap", 0))
                    rejection_counts["terminal_occupancy"] += int(stats.get("terminal_occupancy", 0))
                    if not valid:
                        continue
                    min_distance = episode_ours_others_min_distance(ep_payload)
                    is_near = bool(np.isfinite(min_distance) and min_distance <= float(near_distance_m))
                    ep_payload["conflict_type"], ep_payload["conflict_records"] = classify_conflict(
                        ep_payload["character_assignments"],
                        path_conflict_threshold,
                        goal_conflict_threshold,
                    )
                    seen_signatures.add(signature)
                    accepted_for_ours.append((ep_payload, tuple(others), int(num_chars), is_near, float(min_distance)))
                    if is_near:
                        near_for_ours += 1
                    for candidate in others:
                        key = candidate_usage_key(candidate)
                        usage_counts[key] = int(usage_counts.get(key, 0)) + 1
        for ep_payload, _others, num_chars, is_near, min_distance in accepted_for_ours:
            ep_payload["ours_others_min_distance_m"] = float(min_distance)
            ep_payload["near_ours_others"] = bool(is_near)
            ep_payload["near_distance_threshold_m"] = float(near_distance_m)
            write_episode(
                ep_payload,
                out_dir=scene_root,
                scene_id=scene_id,
                dataset=dataset,
                split=split,
                seed=seed,
                episode_idx=written,
                overlap_diameter_m=overlap_diameter_m,
            )
            written += 1
            episodes_by_num[int(num_chars)] = int(episodes_by_num.get(int(num_chars), 0)) + 1
            if is_near:
                near_episode_count += 1
            else:
                any_episode_count += 1
        if accepted_for_ours:
            usage_counts[candidate_usage_key(ours)] = int(usage_counts.get(candidate_usage_key(ours), 0)) + len(accepted_for_ours)

    summary = {
        "scene_id": scene_id,
        "candidate_count": len(all_candidates),
        "written": written,
        "episodes_by_num": episodes_by_num,
        "near_episodes": near_episode_count,
        "far_episodes": any_episode_count,
        "near_distance_m": float(near_distance_m),
        "near_quota_per_ours": int(near_quota_per_ours),
        "rejections": rejection_counts,
    }
    (scene_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    scene_ids = resolve_scene_ids(args)
    object_char_choices = parse_int_list(args.object_char_choices, DEFAULT_CHAR_CHOICES, min_value=1)
    object_segment_sizes = parse_int_list(args.object_segment_sizes, DEFAULT_OBJECT_WINDOW_SIZES, min_value=1)
    loco_char_choices = parse_int_list(args.loco_char_choices, DEFAULT_CHAR_CHOICES, min_value=1)
    loco_goal_numbers = parse_int_list(args.loco_goal_numbers, DEFAULT_LOCO_GOAL_NUMBERS, min_value=1)
    tasks = [
        (
            args.dataset,
            scene_id,
            args.output_root,
            str(args.episode_bank_name),
            str(args.episodes_name),
            max(0, int(args.max_episodes_per_ours)),
            object_char_choices,
            object_segment_sizes,
            loco_char_choices,
            loco_goal_numbers,
            float(args.path_conflict_threshold_m),
            float(args.goal_conflict_threshold_m),
            float(args.overlap_diameter_m),
            float(args.near_distance_m),
            int(args.near_quota_per_ours),
            int(args.max_other_combinations_per_ours),
            int(args.seed),
            str(args.split),
        )
        for scene_id in scene_ids
    ]
    total = 0
    workers = max(1, int(args.workers))
    if workers > 1 and len(tasks) > 1:
        with mp.Pool(processes=workers, maxtasksperchild=1) as pool:
            for summary in pool.imap_unordered(process_scene, tasks, chunksize=1):
                total += int(summary["written"])
                print(
                    f"{summary['scene_id']}: candidates={summary['candidate_count']} "
                    f"wrote={summary['written']} by_num={summary['episodes_by_num']} "
                    f"near={summary['near_episodes']} far={summary['far_episodes']} "
                    f"rejections={summary['rejections']}"
                )
    else:
        for task in tasks:
            summary = process_scene(task)
            total += int(summary["written"])
            print(
                f"{summary['scene_id']}: candidates={summary['candidate_count']} "
                f"wrote={summary['written']} by_num={summary['episodes_by_num']} "
                f"near={summary['near_episodes']} far={summary['far_episodes']} "
                f"rejections={summary['rejections']}"
            )
    out_root = args.output_root / args.dataset / str(args.episodes_name)
    print(f"wrote {total} episodes under {out_root.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
