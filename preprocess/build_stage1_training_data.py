from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from build_multi_character_affordances import PROJECT_ROOT
from preprocess_final import to_serializable, write_json


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "preprocessed"


@dataclass(frozen=True)
class WindowSpec:
    dataset: str
    split: str
    scene_id: str
    sequence_id: str
    segment_id: int
    orig_segment_id: int | list[int] | None
    segment_label: str
    goal_type: str
    text: str
    move_for_action: bool
    window_move_ratio: float
    window_all_move: bool
    body_goal: Optional[tuple[float, float, float]]
    goal_pose: Optional[dict]
    seq_global_start: int
    window_start: int
    window_end: int
    transl_path: str
    global_orient_path: str
    body_pose_path: str
    human_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage-1 planning samples from scenes_v2.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--scene-list-file", type=Path, default=None)
    parser.add_argument("--window", type=int, default=102)
    parser.add_argument("--past", type=int, default=30)
    parser.add_argument("--future", type=int, default=72)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--min-others", type=int, default=2)
    parser.add_argument("--max-others", type=int, default=4)
    parser.add_argument("--max-dist", type=float, default=0.6)
    parser.add_argument("--min-dist", type=float, default=0.4)
    parser.add_argument("--relaxed-4char-max-dist", type=float, default=0.8)
    parser.add_argument("--static-threshold", type=float, default=0.3)
    parser.add_argument("--static-ratio", type=float, default=0.2)
    parser.add_argument("--max-samples-per-ego", type=int, default=16)
    parser.add_argument("--target-all-window-move-ratio", type=float, default=0.75)
    parser.add_argument("--ego-move-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples-per-scene", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-scene-summaries", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def sequence_scene_root(output_root: Path, dataset: str, scene_id: str) -> Path:
    return output_root / dataset / "scenes_v2" / scene_id


def stage1_scene_root(output_root: Path, dataset: str, scene_id: str) -> Path:
    return output_root / dataset / "stage1_samples_v2" / scene_id


def is_valid_scene_dir(scene_dir: Path) -> bool:
    return scene_dir.is_dir() and (scene_dir / "scene.json").exists() and (scene_dir / "sequences").exists()


def infer_scene_split(output_root: Path, dataset: str, scene_id: str) -> Optional[str]:
    for split in ("train", "test"):
        split_file = output_root / dataset / f"{split}.txt"
        if not split_file.exists():
            continue
        scene_ids = {line.strip() for line in split_file.read_text().splitlines() if line.strip()}
        if scene_id in scene_ids:
            return split
    return None


def resolve_scene_ids(args: argparse.Namespace) -> list[str]:
    if args.scene_id is not None:
        return [args.scene_id]
    if args.scene_list_file is not None:
        return [line.strip() for line in args.scene_list_file.read_text().splitlines() if line.strip()]
    if args.split in ("train", "test"):
        split_file = args.output_root / args.dataset / f"{args.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    scenes_root = args.output_root / args.dataset / "scenes_v2"
    if not scenes_root.exists():
        raise FileNotFoundError(scenes_root)
    return sorted(path.name for path in scenes_root.iterdir() if is_valid_scene_dir(path))


def should_skip_scene(args: argparse.Namespace, scene_id: str) -> bool:
    if not args.skip_existing or args.dry_run:
        return False
    return (stage1_scene_root(args.output_root, args.dataset, scene_id) / "scene_index.json").exists()


def segment_label(segment: dict) -> str:
    goal_type = str(segment.get("goal_type") or "")
    if goal_type == "walk":
        return "move"
    return "action"


def body_goal_tuple(segment: dict) -> Optional[tuple[float, float, float]]:
    goal_pose = segment.get("goal_pose")
    if not isinstance(goal_pose, dict):
        return None
    pelvis = goal_pose.get("pelvis")
    if pelvis is None:
        return None
    arr = np.asarray(pelvis, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 3 or not np.isfinite(arr[:3]).all():
        return None
    return float(arr[0]), float(arr[1]), float(arr[2])


def enumerate_windows_for_sequence(
    dataset: str,
    split: str,
    scene_id: str,
    sequence_payload: dict,
    window: int,
    past: int,
    stride: int,
) -> list[WindowSpec]:
    motion_ref = sequence_payload["human_motion_ref"]
    smpl = motion_ref["smplx"]
    sequence_id = str(sequence_payload["sequence_id"])
    seq_global_start = int(motion_ref["start"])
    seq_global_end = int(motion_ref["end"])
    seq_len = seq_global_end - seq_global_start + 1
    if seq_len < window:
        return []

    segments = sorted(sequence_payload.get("segment_list", []), key=lambda item: (int(item["start"]), int(item["end"])))
    move_mask = np.zeros((seq_len,), dtype=np.bool_)
    for segment in segments:
        if segment_label(segment) != "move":
            continue
        start = max(0, int(segment["start"]))
        end = min(seq_len - 1, int(segment["end"]))
        if end >= start:
            move_mask[start : end + 1] = True

    def segment_for_present_frame(present_frame: int) -> Optional[dict]:
        for segment in segments:
            if int(segment["start"]) <= present_frame <= int(segment["end"]):
                return segment
        return None

    output: list[WindowSpec] = []
    last_window_start = seq_len - window
    for window_start in range(0, last_window_start + 1, stride):
        present_frame = window_start + past - 1
        segment = segment_for_present_frame(present_frame)
        if segment is None:
            continue
        window_end = window_start + window - 1
        window_move_count = int(move_mask[window_start : window_end + 1].sum())
        window_move_ratio = float(window_move_count / max(window, 1))
        output.append(
            WindowSpec(
                dataset=dataset,
                split=split,
                scene_id=scene_id,
                sequence_id=sequence_id,
                segment_id=int(segment["segment_id"]),
                orig_segment_id=segment.get("orig_segment_id"),
                segment_label=segment_label(segment),
                goal_type=str(segment.get("goal_type") or ""),
                text=str(segment.get("text") or ""),
                move_for_action=bool(segment.get("move_for_action", False)),
                window_move_ratio=window_move_ratio,
                window_all_move=bool(window_move_count == window),
                body_goal=body_goal_tuple(segment),
                goal_pose=segment.get("goal_pose"),
                seq_global_start=seq_global_start,
                window_start=int(window_start),
                window_end=int(window_end),
                transl_path=str(smpl["transl_path"]),
                global_orient_path=str(smpl["global_orient_path"]),
                body_pose_path=str(smpl["body_pose_path"]),
                human_path=str(motion_ref["path"]),
            )
        )
    return output


def count_sequence_windows(sequence_payload: dict, window: int, past: int, stride: int) -> int:
    return len(
        enumerate_windows_for_sequence(
            dataset="",
            split="",
            scene_id="",
            sequence_payload=sequence_payload,
            window=window,
            past=past,
            stride=stride,
        )
    )


def load_transl_cached(path_str: str, cache: Dict[Path, np.ndarray]) -> np.ndarray:
    path = resolve_repo_path(path_str)
    arr = cache.get(path)
    if arr is None:
        arr = np.load(path, mmap_mode="r")
        cache[path] = arr
    return arr


def window_track_xz(window: WindowSpec, transl_cache: Dict[Path, np.ndarray]) -> np.ndarray:
    transl = load_transl_cached(window.transl_path, transl_cache)
    g0 = window.seq_global_start + window.window_start
    g1 = window.seq_global_start + window.window_end + 1
    return np.asarray(transl[g0:g1, :], dtype=np.float32)[:, [0, 2]]


def displacement(track_xz: np.ndarray) -> float:
    if len(track_xz) == 0:
        return 0.0
    return float(np.linalg.norm(track_xz[-1] - track_xz[0]))


def build_window_motion_cache(
    windows: Sequence[WindowSpec],
    past: int,
) -> tuple[dict[tuple[str, int, int, int], np.ndarray], dict[tuple[str, int, int, int], np.ndarray], dict[tuple[str, int, int, int], float]]:
    transl_cache: Dict[Path, np.ndarray] = {}
    track_cache: dict[tuple[str, int, int, int], np.ndarray] = {}
    current_cache: dict[tuple[str, int, int, int], np.ndarray] = {}
    displacement_cache: dict[tuple[str, int, int, int], float] = {}
    current_index = past - 1
    for window in windows:
        key = window_key(window)
        track = window_track_xz(window, transl_cache)
        track_cache[key] = track
        current_cache[key] = track[current_index]
        displacement_cache[key] = displacement(track)
    return track_cache, current_cache, displacement_cache


def pair_distance_stats(ego_track_xz: np.ndarray, other_track_xz: np.ndarray, current_index: int) -> dict:
    dist = np.linalg.norm(ego_track_xz - other_track_xz, axis=-1)
    return {
        "min_dist": float(dist.min()),
        "max_dist": float(dist.max()),
        "mean_dist": float(dist.mean()),
        "current_dist": float(dist[current_index]),
        "frames_lt_1p5m": int(np.count_nonzero(dist < 1.5)),
    }


def candidate_is_valid(
    stats: dict,
    max_dist: float,
    min_dist: float,
    relaxed_4char_max_dist: float,
) -> tuple[bool, str]:
    if stats["min_dist"] < min_dist:
        return False, "overlap"
    if stats["current_dist"] <= max_dist:
        return True, "strict"
    if stats["current_dist"] <= relaxed_4char_max_dist:
        return True, "relaxed_4char_only"
    return False, "current_far"


def split_static_dynamic(
    candidates: list[dict],
    static_threshold: float,
) -> tuple[list[dict], list[dict]]:
    static_list: list[dict] = []
    dynamic_list: list[dict] = []
    for item in candidates:
        if float(item["total_displacement"]) <= static_threshold:
            item["motion_type"] = "static"
            static_list.append(item)
        else:
            item["motion_type"] = "dynamic"
            dynamic_list.append(item)
    return static_list, dynamic_list


def window_key(window: WindowSpec) -> tuple[str, int, int, int]:
    return (window.sequence_id, int(window.segment_id), int(window.window_start), int(window.window_end))


def candidate_key(candidate: dict) -> tuple[str, int, int, int]:
    return window_key(candidate["window"])


def target_static_count(num_others: int, static_ratio: float) -> int:
    return max(0, min(num_others, int(round(num_others * static_ratio))))


def pack_candidate_set(
    remaining_candidates: list[dict],
    rng: random.Random,
    num_others: int,
    static_threshold: float,
    static_ratio: float,
    min_dist: float,
    compatibility_cache: dict[tuple[tuple[str, int, int, int], tuple[str, int, int, int]], bool],
    track_cache: dict[tuple[str, int, int, int], np.ndarray],
    allow_one_relaxed: bool,
) -> list[dict]:
    if len(remaining_candidates) < num_others:
        return []

    copied_candidates = [dict(item) for item in remaining_candidates]
    strict_candidates = [item for item in copied_candidates if item["eligibility"] == "strict"]
    relaxed_candidates = [item for item in copied_candidates if item["eligibility"] == "relaxed_4char_only"]
    strict_static, strict_dynamic = split_static_dynamic(strict_candidates, static_threshold)
    relaxed_static, relaxed_dynamic = split_static_dynamic(relaxed_candidates, static_threshold)
    def prioritize_move(pool: list[dict]) -> None:
        rng.shuffle(pool)
        pool.sort(
            key=lambda item: (
                0 if item["window"].window_all_move else 1,
                -float(item["window"].window_move_ratio),
            )
        )

    prioritize_move(strict_static)
    prioritize_move(strict_dynamic)
    prioritize_move(relaxed_static)
    prioritize_move(relaxed_dynamic)

    required_static = target_static_count(num_others, static_ratio)
    required_dynamic = num_others - required_static
    selected: list[dict] = []
    relaxed_used = 0

    def compatible_with_selected(candidate: dict) -> bool:
        cand_key = candidate_key(candidate)
        cand_track = track_cache[cand_key]
        for chosen in selected:
            chosen_key = candidate_key(chosen)
            cache_key = tuple(sorted((cand_key, chosen_key)))
            cached = compatibility_cache.get(cache_key)
            if cached is None:
                chosen_track = track_cache[chosen_key]
                cached = bool(np.linalg.norm(cand_track - chosen_track, axis=-1).min() >= min_dist)
                compatibility_cache[cache_key] = cached
            if not cached:
                return False
        return True

    def add_from_pool(pool: list[dict], quota: int | None = None, relaxed: bool = False) -> None:
        nonlocal relaxed_used
        added = 0
        for candidate in pool:
            if candidate_key(candidate) in {candidate_key(item) for item in selected}:
                continue
            if relaxed and not allow_one_relaxed:
                continue
            if relaxed and relaxed_used >= 1:
                continue
            if compatible_with_selected(candidate):
                selected.append(candidate)
                if relaxed:
                    relaxed_used += 1
                added += 1
                if quota is not None and added >= quota:
                    break
                if len(selected) >= num_others:
                    break

    strict_slots = num_others - 1 if allow_one_relaxed else num_others
    strict_dynamic_target = min(required_dynamic, strict_slots)
    strict_static_target = min(required_static, strict_slots - strict_dynamic_target)

    add_from_pool(strict_dynamic, strict_dynamic_target)
    add_from_pool(strict_static, strict_static_target)

    if len(selected) < strict_slots:
        strict_fallback = strict_dynamic + strict_static
        rng.shuffle(strict_fallback)
        add_from_pool(strict_fallback, strict_slots - len(selected))

    if allow_one_relaxed and len(selected) < num_others:
        remaining_static_need = max(0, required_static - sum(1 for item in selected if item["motion_type"] == "static"))
        remaining_dynamic_need = max(0, required_dynamic - sum(1 for item in selected if item["motion_type"] == "dynamic"))
        if remaining_dynamic_need > 0:
            add_from_pool(relaxed_dynamic, 1, relaxed=True)
        if len(selected) < num_others and remaining_static_need > 0:
            add_from_pool(relaxed_static, 1, relaxed=True)
        if len(selected) < num_others:
            relaxed_fallback = relaxed_dynamic + relaxed_static
            prioritize_move(relaxed_fallback)
            add_from_pool(relaxed_fallback, 1, relaxed=True)

    if len(selected) < num_others:
        fallback_pool = strict_dynamic + strict_static
        prioritize_move(fallback_pool)
        add_from_pool(fallback_pool, None)

    if len(selected) < num_others:
        return []
    return selected[:num_others]


def pack_candidate_sets_for_ego(
    valid_candidates: list[dict],
    rng: random.Random,
    static_threshold: float,
    static_ratio: float,
    min_dist: float,
    requested_sets: Sequence[tuple[int, int]],
    track_cache: dict[tuple[str, int, int, int], np.ndarray],
) -> tuple[list[list[dict]], Counter]:
    remaining = [dict(item) for item in valid_candidates]
    rng.shuffle(remaining)
    remaining.sort(
        key=lambda item: (
            0 if item["window"].window_all_move else 1,
            -float(item["window"].window_move_ratio),
        )
    )
    packed_sets: list[list[dict]] = []
    rejection = Counter()
    compatibility_cache: dict[tuple[tuple[str, int, int, int], tuple[str, int, int, int]], bool] = {}

    for num_others, repeat in requested_sets:
        for _ in range(repeat):
            if len(remaining) < num_others:
                rejection[f"insufficient_pool_for_{num_others}"] += 1
                continue
            packed = pack_candidate_set(
                remaining_candidates=remaining,
                rng=rng,
                num_others=num_others,
                static_threshold=static_threshold,
                static_ratio=static_ratio,
                min_dist=min_dist,
                compatibility_cache=compatibility_cache,
                track_cache=track_cache,
                allow_one_relaxed=(num_others == 3),
            )
            if not packed:
                rejection[f"pack_failed_for_{num_others}"] += 1
                continue
            packed_sets.append(packed)
            used = {candidate_key(item) for item in packed}
            remaining = [item for item in remaining if candidate_key(item) not in used]
    return packed_sets, rejection


def make_human_motion_ref(window: WindowSpec) -> dict:
    return {
        "path": window.human_path,
        "start": int(window.seq_global_start),
        "smplx": {
            "transl_path": window.transl_path,
            "global_orient_path": window.global_orient_path,
            "body_pose_path": window.body_pose_path,
        },
    }


def make_sample_payload(
    sample_id: str,
    window_cfg: dict,
    ego: WindowSpec,
    others: list[dict],
    scene_split: Optional[str],
) -> dict:
    return {
        "sample_id": sample_id,
        "dataset": ego.dataset,
        "split": scene_split,
        "scene_id": ego.scene_id,
        "window": window_cfg,
        "ego": {
            "sequence_id": ego.sequence_id,
            "segment_id": int(ego.segment_id),
            "goal_type": ego.goal_type,
            "body_goal": ego.body_goal,
            "window_start": int(ego.window_start),
            "window_end": int(ego.window_end),
            "human_motion_ref": make_human_motion_ref(ego),
        },
        "others": [
            {
                "sequence_id": item["window"].sequence_id,
                "segment_id": int(item["window"].segment_id),
                "window_start": int(item["window"].window_start),
                "window_end": int(item["window"].window_end),
                "window_move_ratio": float(item["window"].window_move_ratio),
                "window_all_move": bool(item["window"].window_all_move),
                "human_motion_ref": make_human_motion_ref(item["window"]),
            }
            for item in others
        ],
        "stats": {
            "num_others": len(others),
            "num_static_others": sum(1 for item in others if item["motion_type"] == "static"),
            "num_dynamic_others": sum(1 for item in others if item["motion_type"] == "dynamic"),
            "num_all_window_move_others": sum(1 for item in others if item["window"].window_all_move),
            "all_window_move_other_ratio": float(
                sum(1 for item in others if item["window"].window_all_move) / max(len(others), 1)
            ),
            "other_window_move_ratios": [float(item["window"].window_move_ratio) for item in others],
        },
    }


def summarize_counts(counter: Counter, keys: Sequence[str]) -> dict:
    return {key: int(counter.get(key, 0)) for key in keys}


def run_scene(args: argparse.Namespace, scene_id: str, progress_bar: tqdm | None = None) -> dict:
    rng = random.Random(args.seed)
    scene_root = sequence_scene_root(args.output_root, args.dataset, scene_id)
    scene_json = scene_root / "scene.json"
    if not scene_json.exists():
        raise FileNotFoundError(scene_json)
    seq_paths = sorted((scene_root / "sequences").glob("*.json"))
    if not seq_paths:
        raise RuntimeError(f"no sequences found under {scene_root}")

    sequence_payloads = [load_json(path) for path in seq_paths]
    scene_split = infer_scene_split(args.output_root, args.dataset, scene_id)
    all_windows: list[WindowSpec] = []
    for sequence_payload in sequence_payloads:
        all_windows.extend(
            enumerate_windows_for_sequence(
                dataset=args.dataset,
                split=args.split,
                scene_id=scene_id,
                sequence_payload=sequence_payload,
                window=args.window,
                past=args.past,
                stride=args.stride,
            )
        )

    windows_by_sequence: dict[str, list[WindowSpec]] = defaultdict(list)
    for window in all_windows:
        windows_by_sequence[window.sequence_id].append(window)

    track_cache, current_cache, displacement_cache = build_window_motion_cache(all_windows, args.past)
    rejection = Counter()
    static_dynamic_counter = Counter()
    sample_count = 0
    selected_all_window_move_others = 0
    selected_total_others = 0
    num_others_hist = Counter()
    valid_candidate_counts: list[int] = []
    packed_set_counts: list[int] = []
    ego_windows_considered = 0
    ego_windows_skipped_non_move = 0
    total_candidate_pairs = 0
    per_ego_quota = max(0, int(args.max_samples_per_ego)) if args.max_samples_per_ego is not None else 0
    requested_sets = [
        (3, per_ego_quota),
        (2, per_ego_quota),
        (1, per_ego_quota),
    ]

    out_scene_root = stage1_scene_root(args.output_root, args.dataset, scene_id)
    if not args.dry_run:
        if out_scene_root.exists():
            shutil.rmtree(out_scene_root)
        out_scene_root.mkdir(parents=True, exist_ok=True)

    for ego in all_windows:
        if args.ego_move_only and ego.segment_label != "move":
            ego_windows_skipped_non_move += 1
            if progress_bar is not None:
                progress_bar.set_postfix_str(scene_id)
                progress_bar.update(1)
            continue
        ego_windows_considered += 1
        if progress_bar is not None:
            progress_bar.set_postfix_str(scene_id)
            progress_bar.update(1)
        ego_key = window_key(ego)
        ego_track = track_cache[ego_key]
        ego_current = current_cache[ego_key]
        valid_candidates: list[dict] = []
        for sequence_id, other_windows in windows_by_sequence.items():
            if sequence_id == ego.sequence_id:
                continue
            for other in other_windows:
                total_candidate_pairs += 1
                other_key = window_key(other)
                other_track = track_cache[other_key]
                if len(other_track) != len(ego_track):
                    rejection["length_mismatch"] += 1
                    continue
                current_dist = float(np.linalg.norm(ego_current - current_cache[other_key]))
                if current_dist > args.relaxed_4char_max_dist:
                    rejection["current_far"] += 1
                    continue
                stats = pair_distance_stats(ego_track, other_track, args.past - 1)
                ok, reason = candidate_is_valid(
                    stats,
                    args.max_dist,
                    args.min_dist,
                    args.relaxed_4char_max_dist,
                )
                if not ok:
                    rejection[reason] += 1
                    continue
                valid_candidates.append(
                    {
                        "window": other,
                        **stats,
                        "eligibility": reason,
                        "total_displacement": displacement_cache[other_key],
                        "window_move_ratio": other.window_move_ratio,
                        "window_all_move": other.window_all_move,
                    }
                )
        valid_candidate_counts.append(len(valid_candidates))
        packed_sets, set_rejections = pack_candidate_sets_for_ego(
            valid_candidates=valid_candidates,
            rng=rng,
            static_threshold=args.static_threshold,
            static_ratio=args.static_ratio,
            min_dist=args.min_dist,
            requested_sets=requested_sets,
            track_cache=track_cache,
        )
        rejection.update(set_rejections)
        if args.max_samples_per_ego is not None:
            packed_sets = packed_sets[: max(0, int(args.max_samples_per_ego))]
        packed_set_counts.append(len(packed_sets))
        if not packed_sets:
            rejection["no_valid_sets"] += 1
            continue

        for selected_others in packed_sets:
            add_total = len(selected_others)
            add_all_move = sum(1 for item in selected_others if item["window"].window_all_move)
            if add_total > 0 and float(args.target_all_window_move_ratio) > 0.0:
                next_all = selected_all_window_move_others + add_all_move
                next_total = selected_total_others + add_total
                next_ratio = float(next_all / max(next_total, 1))
                if next_ratio + 1e-12 < float(args.target_all_window_move_ratio):
                    rejection["target_all_window_move_ratio"] += 1
                    continue
            sample_id = f"{scene_id}__sample_{sample_count:06d}"
            payload = make_sample_payload(
                sample_id=sample_id,
                window_cfg={
                    "length": int(args.window),
                    "past": int(args.past),
                    "future": int(args.future),
                    "stride": int(args.stride),
                },
                ego=ego,
                others=selected_others,
                scene_split=scene_split,
            )
            sample_count += 1
            selected_all_window_move_others += add_all_move
            selected_total_others += add_total
            num_others_hist[len(selected_others) + 1] += 1
            for item in selected_others:
                static_dynamic_counter[item["motion_type"]] += 1

            if not args.dry_run:
                write_json(out_scene_root / f"sample_{sample_count - 1:06d}.json", payload)

            if args.max_samples_per_scene is not None and sample_count >= args.max_samples_per_scene:
                break
        if args.max_samples_per_scene is not None and sample_count >= args.max_samples_per_scene:
            break

    if not valid_candidate_counts:
        valid_candidate_counts = [0]
    if not packed_set_counts:
        packed_set_counts = [0]

    summary = {
        "dataset": args.dataset,
        "split": scene_split,
        "scene_id": scene_id,
        "num_sequences": len(sequence_payloads),
        "num_windows": len(all_windows),
        "ego_windows_considered": ego_windows_considered,
        "ego_windows_skipped_non_move": ego_windows_skipped_non_move,
        "total_candidate_pairs": total_candidate_pairs,
        "valid_samples": sample_count,
        "target_all_window_move_ratio": float(args.target_all_window_move_ratio),
        "selected_all_window_move_others": int(selected_all_window_move_others),
        "selected_total_others": int(selected_total_others),
        "selected_all_window_move_other_ratio": float(
            selected_all_window_move_others / max(selected_total_others, 1)
        ),
        "avg_valid_candidates_per_ego": float(np.mean(valid_candidate_counts)),
        "median_valid_candidates_per_ego": float(np.median(valid_candidate_counts)),
        "p90_valid_candidates_per_ego": float(np.percentile(valid_candidate_counts, 90)),
        "max_valid_candidates_per_ego": int(np.max(valid_candidate_counts)),
        "avg_sets_per_ego": float(np.mean(packed_set_counts)),
        "median_sets_per_ego": float(np.median(packed_set_counts)),
        "p90_sets_per_ego": float(np.percentile(packed_set_counts, 90)),
        "max_sets_per_ego": int(np.max(packed_set_counts)),
        "avg_num_characters_per_sample": float(
            np.mean([k for k, v in num_others_hist.items() for _ in range(v)]) if sample_count > 0 else 0.0
        ),
        "num_characters_hist": {str(k): int(v) for k, v in sorted(num_others_hist.items())},
        "others_motion_type_counts": summarize_counts(static_dynamic_counter, ["static", "dynamic"]),
        "rejections": dict(rejection),
    }

    if not args.dry_run:
        write_json(out_scene_root / "scene_index.json", summary)
    return summary


def scene_window_count(args: argparse.Namespace, scene_id: str) -> int:
    scene_root = sequence_scene_root(args.output_root, args.dataset, scene_id)
    scene_json = scene_root / "scene.json"
    if not scene_json.exists():
        return 0
    total = 0
    for path in sorted((scene_root / "sequences").glob("*.json")):
        total += count_sequence_windows(load_json(path), args.window, args.past, args.stride)
    return total


def main() -> None:
    args = parse_args()
    if args.window != args.past + args.future:
        raise ValueError("--window must equal --past + --future")
    raw_scene_ids = resolve_scene_ids(args)
    scene_ids = [scene_id for scene_id in raw_scene_ids if not should_skip_scene(args, scene_id)]
    skipped_existing = len(raw_scene_ids) - len(scene_ids)
    scene_windows = {scene_id: scene_window_count(args, scene_id) for scene_id in scene_ids}
    total_ego_windows = sum(scene_windows.values())
    summaries: list[dict] = []
    if args.workers <= 1:
        with tqdm(total=total_ego_windows, desc=f"{args.dataset}:all", unit="ego") as progress_bar:
            for scene_id in scene_ids:
                summary = run_scene(args, scene_id, progress_bar=progress_bar)
                summaries.append(summary)
                if args.print_scene_summaries:
                    print(json.dumps(summary, indent=2))
    else:
        with tqdm(total=total_ego_windows, desc=f"{args.dataset}:all", unit="ego") as progress_bar:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_scene = {executor.submit(run_scene, args, scene_id, None): scene_id for scene_id in scene_ids}
                for future in as_completed(future_to_scene):
                    scene_id = future_to_scene[future]
                    summary = future.result()
                    summaries.append(summary)
                    progress_bar.update(scene_windows.get(scene_id, 0))
                    progress_bar.set_postfix_str(scene_id)
                    if args.print_scene_summaries:
                        print(json.dumps(summary, indent=2))

    if len(summaries) > 1 and args.print_scene_summaries:
        total_samples = sum(item["valid_samples"] for item in summaries)
        total_ego = sum(item["ego_windows_considered"] for item in summaries)
        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "split": args.split,
                    "num_scenes": len(summaries),
                    "num_scenes_skipped_existing": skipped_existing,
                    "total_valid_samples": total_samples,
                    "total_ego_windows_considered": total_ego,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
