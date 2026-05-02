from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSED_ROOT = Path("data") / "preprocessed"
MOVE_WAIT_GOAL_TYPES = {"walk", "move", "stand_still"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage-2 MoveWait/Action manifests from scenes_v2.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--scene-list-file", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PREPROCESSED_ROOT)
    parser.add_argument("--stage2-dir-name", default="stage2")
    parser.add_argument("--history-frames", type=int, default=5)
    parser.add_argument("--move-future-frames", type=int, default=21)
    parser.add_argument("--action-min-target-frames", type=int, default=30)
    parser.add_argument("--action-max-target-frames", type=int, default=300)
    parser.add_argument("--action-max-aug-delta", type=int, default=25)
    parser.add_argument("--max-scenes", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def repo_rel(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def resolve_scene_ids(args: argparse.Namespace, split: str) -> list[str]:
    if args.scene_id:
        return [str(args.scene_id)]
    if args.scene_list_file is not None:
        return [line.strip() for line in args.scene_list_file.read_text().splitlines() if line.strip()]
    scenes_root = args.output_root / args.dataset / "scenes_v2"
    if split in {"train", "test"}:
        split_file = args.output_root / args.dataset / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        scene_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    else:
        scene_ids = sorted(
            p.name for p in scenes_root.iterdir() if p.is_dir() and (p / "scene.json").exists() and (p / "sequences").exists()
        )
    if int(args.max_scenes) > 0:
        scene_ids = scene_ids[: int(args.max_scenes)]
    return scene_ids


def splits_to_build(args: argparse.Namespace) -> list[str]:
    if args.split == "all":
        return ["train", "test"]
    return [str(args.split)]


def as_xyz(value: Any) -> list[float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 3:
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def sequence_segments(sequence_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        list(sequence_payload.get("segment_list") or sequence_payload.get("segments") or []),
        key=lambda item: (int(item.get("start", 0)), int(item.get("end", 0)), int(item.get("segment_id", 0))),
    )


def smplx_paths(motion_ref: dict[str, Any]) -> dict[str, str | None]:
    smpl = dict(motion_ref.get("smplx") or {})
    return {
        "transl_path": repo_rel(smpl.get("transl_path")),
        "global_orient_path": repo_rel(smpl.get("global_orient_path")),
        "body_pose_path": repo_rel(smpl.get("body_pose_path")),
        "left_hand_pose_path": repo_rel(smpl.get("left_hand_pose_path")),
        "right_hand_pose_path": repo_rel(smpl.get("right_hand_pose_path")),
        "joints_path": repo_rel(motion_ref.get("path")),
    }


def scene_ref(scene_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "occupancy_grid_path": repo_rel(scene_payload.get("occupancy_grid_path")),
        "grid_meta": scene_payload.get("grid_meta"),
    }


def goal_payload(segment: dict[str, Any]) -> dict[str, Any]:
    goal_pose = segment.get("goal_pose") if isinstance(segment.get("goal_pose"), dict) else {}
    active_hand = str(segment.get("active_hand") or "").strip().lower()
    hand = as_xyz(goal_pose.get("hand"))
    left = as_xyz(goal_pose.get("left_hand"))
    right = as_xyz(goal_pose.get("right_hand"))
    if hand is not None and left is None and right is None:
        if active_hand == "left":
            left = hand
        elif active_hand == "right":
            right = hand
        elif active_hand in {"both", "none", ""}:
            left = hand
            right = hand
    return {
        "body_goal": as_xyz(goal_pose.get("pelvis")),
        "hand_goal": hand,
        "left_hand_goal": left,
        "right_hand_goal": right,
        "active_hand": active_hand,
    }


def base_record(
    args: argparse.Namespace,
    split: str,
    scene_id: str,
    scene_payload: dict[str, Any],
    sequence_payload: dict[str, Any],
    segment: dict[str, Any],
) -> dict[str, Any]:
    motion_ref = sequence_payload["human_motion_ref"]
    seq_global_start = int(motion_ref["start"])
    seq_global_end = int(motion_ref["end"])
    seg_start = int(segment["start"])
    seg_end = int(segment["end"])
    return {
        "dataset": args.dataset,
        "split": split,
        "scene_id": scene_id,
        "sequence_id": str(sequence_payload["sequence_id"]),
        "sequence_global_start": seq_global_start,
        "sequence_global_end": seq_global_end,
        "sequence_local_start": 0,
        "sequence_local_end": int(seq_global_end - seq_global_start),
        "segment_id": int(segment.get("segment_id", -1)),
        "orig_segment_id": segment.get("orig_segment_id"),
        "segment_start": seg_start,
        "segment_end": seg_end,
        "segment_global_start": int(seq_global_start + seg_start),
        "segment_global_end": int(seq_global_start + seg_end),
        "goal_type": str(segment.get("goal_type") or ""),
        "text": str(segment.get("text") or ""),
        "move_for_action": bool(segment.get("move_for_action", False)),
        "smplx": smplx_paths(motion_ref),
        "scene": scene_ref(scene_payload),
    }


def build_move_wait_record(
    args: argparse.Namespace,
    split: str,
    scene_id: str,
    scene_payload: dict[str, Any],
    sequence_payload: dict[str, Any],
    segment: dict[str, Any],
    segment_ordinal: int,
) -> tuple[dict[str, Any] | None, str | None]:
    H = int(args.history_frames)
    W = int(args.move_future_frames)
    start = int(segment["start"])
    end = int(segment["end"])
    target_start_min = start + H
    target_start_max = end - W + 1
    if target_start_min > target_start_max:
        return None, "too_short"
    rec = base_record(args, split, scene_id, scene_payload, sequence_payload, segment)
    rec.update(
        {
            "kind": "move_wait",
            "segment_ordinal": int(segment_ordinal),
            "target_start_min": int(target_start_min),
            "target_start_max": int(target_start_max),
            "history_frames": H,
            "future_frames": W,
            "r_plan_frames": W,
        }
    )
    rec.update(goal_payload(segment))
    return rec, None


def build_action_record(
    args: argparse.Namespace,
    split: str,
    scene_id: str,
    scene_payload: dict[str, Any],
    sequence_payload: dict[str, Any],
    segment: dict[str, Any],
    segment_ordinal: int,
) -> tuple[dict[str, Any] | None, str | None]:
    H = int(args.history_frames)
    min_frames = int(args.action_min_target_frames)
    max_frames = int(args.action_max_target_frames)
    max_aug = int(args.action_max_aug_delta)
    action_start = int(segment["start"])
    action_end = int(segment["end"])
    base_len = action_end - action_start + 1
    first_segment_exception = int(segment_ordinal) == 0

    if first_segment_exception:
        forced_target_start = action_start + H
        target_len = action_end - forced_target_start + 1
        if target_len < min_frames:
            return None, "first_segment_too_short"
        if target_len > max_frames:
            return None, "first_segment_too_long"
        aug_delta_min = 0
        aug_delta_max = 0
    else:
        pre_context = action_start - H
        max_delta_allowed = min(max_aug, pre_context, max_frames - base_len)
        min_delta_required = max(0, min_frames - base_len)
        if max_delta_allowed < min_delta_required:
            if base_len > max_frames:
                return None, "too_long"
            if base_len + max(0, min(max_aug, pre_context)) < min_frames:
                return None, "too_short"
            return None, "no_valid_aug_delta"
        forced_target_start = None
        target_len = None
        aug_delta_min = int(min_delta_required)
        aug_delta_max = int(max_delta_allowed)

    rec = base_record(args, split, scene_id, scene_payload, sequence_payload, segment)
    rec.update(
        {
            "kind": "action",
            "segment_ordinal": int(segment_ordinal),
            "action_start": int(action_start),
            "action_end": int(action_end),
            "action_base_len": int(base_len),
            "history_frames": H,
            "target_max_frames": max_frames,
            "target_min_frames": min_frames,
            "aug_delta_min": int(aug_delta_min),
            "aug_delta_max": int(aug_delta_max),
            "first_segment_exception": bool(first_segment_exception),
            "forced_target_start": None if forced_target_start is None else int(forced_target_start),
            "forced_target_len": None if target_len is None else int(target_len),
        }
    )
    rec.update(goal_payload(segment))
    if rec.get("body_goal") is None:
        return None, "missing_body_goal"
    return rec, None


def build_for_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    scenes_root = args.output_root / args.dataset / "scenes_v2"
    move_wait_records: list[dict[str, Any]] = []
    action_records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "dataset": args.dataset,
        "split": split,
        "history_frames": int(args.history_frames),
        "move_future_frames": int(args.move_future_frames),
        "action_min_target_frames": int(args.action_min_target_frames),
        "action_max_target_frames": int(args.action_max_target_frames),
        "action_max_aug_delta": int(args.action_max_aug_delta),
        "num_scenes": 0,
        "num_sequences": 0,
        "move_wait_records": 0,
        "action_records": 0,
        "goal_type_counts": {},
        "action_goal_type_counts": {},
        "move_wait_goal_type_counts": {},
        "action_aug_delta_max_hist": {},
        "action_rejections": {},
        "move_wait_rejections": {},
    }
    goal_counter = Counter()
    action_goal_counter = Counter()
    move_goal_counter = Counter()
    action_aug_counter = Counter()
    action_rejections = Counter()
    move_rejections = Counter()

    scene_ids = resolve_scene_ids(args, split)
    stats["num_scenes"] = len(scene_ids)
    for scene_id in scene_ids:
        scene_dir = scenes_root / scene_id
        scene_json = scene_dir / "scene.json"
        if not scene_json.exists():
            continue
        scene_payload = load_json(scene_json)
        seq_paths = sorted((scene_dir / "sequences").glob("*.json"))
        for seq_path in seq_paths:
            sequence_payload = load_json(seq_path)
            stats["num_sequences"] += 1
            segments = sequence_segments(sequence_payload)
            for ordinal, segment in enumerate(segments):
                goal_type = str(segment.get("goal_type") or "")
                goal_counter[goal_type] += 1
                if goal_type in MOVE_WAIT_GOAL_TYPES:
                    rec, reason = build_move_wait_record(args, split, scene_id, scene_payload, sequence_payload, segment, ordinal)
                    if rec is None:
                        move_rejections[reason or "unknown"] += 1
                        continue
                    move_wait_records.append(rec)
                    move_goal_counter[goal_type] += 1
                else:
                    rec, reason = build_action_record(args, split, scene_id, scene_payload, sequence_payload, segment, ordinal)
                    if rec is None:
                        action_rejections[reason or "unknown"] += 1
                        continue
                    action_records.append(rec)
                    action_goal_counter[goal_type] += 1
                    action_aug_counter[int(rec["aug_delta_max"])] += 1

    stats["move_wait_records"] = len(move_wait_records)
    stats["action_records"] = len(action_records)
    stats["goal_type_counts"] = dict(sorted(goal_counter.items()))
    stats["action_goal_type_counts"] = dict(sorted(action_goal_counter.items()))
    stats["move_wait_goal_type_counts"] = dict(sorted(move_goal_counter.items()))
    stats["action_aug_delta_max_hist"] = {str(k): int(v) for k, v in sorted(action_aug_counter.items())}
    stats["action_rejections"] = dict(sorted(action_rejections.items()))
    stats["move_wait_rejections"] = dict(sorted(move_rejections.items()))
    return {"move_wait": move_wait_records, "action": action_records, "stats": stats}


def manifest_payload(args: argparse.Namespace, split: str, kind: str, records: list[dict[str, Any]], stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "split": split,
        "kind": kind,
        "records": records,
        "config": {
            "history_frames": int(args.history_frames),
            "move_future_frames": int(args.move_future_frames),
            "action_min_target_frames": int(args.action_min_target_frames),
            "action_max_target_frames": int(args.action_max_target_frames),
            "action_max_aug_delta": int(args.action_max_aug_delta),
            "move_wait_goal_types": sorted(MOVE_WAIT_GOAL_TYPES),
        },
        "stats": stats,
    }


def main() -> None:
    args = parse_args()
    out_root = args.output_root / args.dataset / str(args.stage2_dir_name)
    for split in splits_to_build(args):
        built = build_for_split(args, split)
        stats = built["stats"]
        write_json(out_root / f"move_wait_{split}.json", manifest_payload(args, split, "move_wait", built["move_wait"], stats))
        write_json(out_root / f"action_{split}.json", manifest_payload(args, split, "action", built["action"], stats))
        write_json(out_root / f"stats_{split}.json", stats)
        print(
            f"{args.dataset} {split}: move_wait={stats['move_wait_records']} "
            f"action={stats['action_records']} sequences={stats['num_sequences']}"
        )


if __name__ == "__main__":
    main()
