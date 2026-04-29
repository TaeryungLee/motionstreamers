from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSED_ROOT = PROJECT_ROOT / "data" / "preprocessed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh stage1_samples_v1 metadata from current scenes_v2.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PREPROCESSED_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def sequence_json_path(output_root: Path, dataset: str, scene_id: str, sequence_id: str) -> Path:
    return output_root / dataset / "scenes_v2" / scene_id / "sequences" / f"{sequence_id}.json"


def find_segment_for_frame(segment_list: list[dict], frame_idx: int) -> dict | None:
    for segment in segment_list:
        if int(segment["start"]) <= frame_idx <= int(segment["end"]):
            return segment
    return None


def refresh_actor(actor: dict, scene_id: str, output_root: Path, dataset: str, past: int, update_goal: bool) -> tuple[bool, str | None, str | None]:
    seq_path = sequence_json_path(output_root, dataset, scene_id, actor["sequence_id"])
    if not seq_path.exists():
        return False, "missing_sequence", None
    seq_record = load_json(seq_path)
    present_frame = int(actor["window_start"]) + int(past) - 1
    segment = find_segment_for_frame(seq_record.get("segment_list", []), present_frame)
    if segment is None:
        return False, "no_segment_for_present", None

    changed = False
    old_goal_type = actor.get("goal_type")
    new_goal_type = str(segment.get("goal_type") or "")
    if int(actor.get("segment_id", -1)) != int(segment["segment_id"]):
        actor["segment_id"] = int(segment["segment_id"])
        changed = True
    if update_goal:
        if old_goal_type != new_goal_type:
            actor["goal_type"] = new_goal_type
            changed = True
        goal_pose = segment.get("goal_pose")
        new_body_goal = None
        if isinstance(goal_pose, dict) and goal_pose.get("pelvis") is not None:
            new_body_goal = goal_pose["pelvis"]
        if actor.get("body_goal") != new_body_goal:
            actor["body_goal"] = new_body_goal
            changed = True
    return changed, None, new_goal_type if update_goal else None


def refresh_scene(output_root: Path, dataset: str, scene_id: str, dry_run: bool) -> dict:
    scene_root = output_root / dataset / "stage1_samples_v1" / scene_id
    sample_paths = sorted(scene_root.glob("sample_*.json"))
    stats = Counter()

    for sample_path in sample_paths:
        payload = load_json(sample_path)
        past = int(payload["window"]["past"])
        sample_changed = False

        changed, error, new_goal_type = refresh_actor(
            actor=payload["ego"],
            scene_id=scene_id,
            output_root=output_root,
            dataset=dataset,
            past=past,
            update_goal=True,
        )
        if error is not None:
            stats[error] += 1
            continue
        if changed:
            sample_changed = True
            stats["ego_updated"] += 1
        if new_goal_type is not None:
            stats[f"ego_goal_type::{new_goal_type}"] += 1

        for other in payload.get("others", []):
            changed, error, _ = refresh_actor(
                actor=other,
                scene_id=scene_id,
                output_root=output_root,
                dataset=dataset,
                past=past,
                update_goal=False,
            )
            if error is not None:
                stats[error] += 1
                continue
            if changed:
                sample_changed = True
                stats["other_segment_updated"] += 1

        if sample_changed:
            stats["samples_updated"] += 1
            if not dry_run:
                write_json(sample_path, payload)

    return {
        "dataset": dataset,
        "scene_id": scene_id,
        "num_samples": len(sample_paths),
        "stats": dict(stats),
    }


def main() -> None:
    args = parse_args()
    sample_root = args.output_root / args.dataset / "stage1_samples_v1"
    if args.scene_id is not None:
        scene_ids = [args.scene_id]
    else:
        scene_ids = sorted(path.name for path in sample_root.iterdir() if path.is_dir())

    summaries = []
    for scene_id in scene_ids:
        summary = refresh_scene(args.output_root, args.dataset, scene_id, args.dry_run)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    if len(summaries) > 1:
        total_samples = sum(item["num_samples"] for item in summaries)
        total_updated = sum(int(item["stats"].get("samples_updated", 0)) for item in summaries)
        print(json.dumps({"dataset": args.dataset, "num_scenes": len(summaries), "num_samples": total_samples, "samples_updated": total_updated}, indent=2))


if __name__ == "__main__":
    main()
