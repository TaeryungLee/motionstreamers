from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrity checks for preprocessed TRUMANS/LINGO outputs.")
    parser.add_argument("--dataset", choices=["trumans", "lingo", "both"], default="both")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "preprocessed")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-errors", type=int, default=2000)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def record_error(errors: list[dict], max_errors: int, **payload):
    if len(errors) >= max_errors:
        return
    errors.append(payload)


def check_scene(scene_path: Path, dataset: str, errors: list[dict], max_errors: int) -> dict:
    scene = load_json(scene_path)
    scene_id = scene.get("scene_id")
    if not scene_id:
        record_error(errors, max_errors, type="scene", path=str(scene_path), msg="missing scene_id")
        return {"scene_id": None, "sequence_ids": []}

    required_fields = ["occupancy_grid_path", "clearance_map_npy_path", "clearance_map_vis_path", "sequence_ids"]
    for field in required_fields:
        if field not in scene:
            record_error(errors, max_errors, type="scene", scene_id=scene_id, path=str(scene_path), msg=f"missing {field}")

    for field in ["occupancy_grid_path", "clearance_map_npy_path", "clearance_map_vis_path"]:
        resolved = resolve_path(scene.get(field))
        if resolved is None or not resolved.exists():
            record_error(errors, max_errors, type="scene", scene_id=scene_id, path=str(scene_path), msg=f"path missing: {field}={scene.get(field)}")

    sequence_ids = scene.get("sequence_ids") or []
    if not isinstance(sequence_ids, list):
        record_error(errors, max_errors, type="scene", scene_id=scene_id, path=str(scene_path), msg="sequence_ids not list")
        sequence_ids = []

    return {"scene_id": scene_id, "sequence_ids": sequence_ids}


def joints_length(human_motion_ref: dict, errors: list[dict], max_errors: int, ctx: dict) -> int | None:
    path_str = human_motion_ref.get("path")
    start = human_motion_ref.get("start")
    end = human_motion_ref.get("end")
    if path_str is None or start is None or end is None:
        record_error(errors, max_errors, **ctx, msg="human_motion_ref missing path/start/end")
        return None
    path = resolve_path(path_str)
    if path is None or not path.exists():
        record_error(errors, max_errors, **ctx, msg=f"human_motion_ref path missing: {path_str}")
        return None
    try:
        joints = np.load(path, mmap_mode="r")
    except Exception as exc:  # noqa: BLE001
        record_error(errors, max_errors, **ctx, msg=f"failed to load joints: {path_str} ({exc})")
        return None
    total = joints.shape[0]
    if not (0 <= int(start) < int(end) <= total):
        record_error(errors, max_errors, **ctx, msg=f"human_motion_ref range invalid: start={start} end={end} total={total}")
    return total


def check_sequence(sequence_path: Path, dataset: str, scene_id: str, errors: list[dict], max_errors: int) -> str | None:
    seq = load_json(sequence_path)
    seq_id = seq.get("sequence_id")
    if not seq_id:
        record_error(errors, max_errors, type="sequence", scene_id=scene_id, path=str(sequence_path), msg="missing sequence_id")
        return None
    if seq.get("scene_id") != scene_id:
        if dataset == "trumans":
            record_error(errors, max_errors, type="sequence", scene_id=scene_id, sequence_id=seq_id, path=str(sequence_path), msg=f"scene_id mismatch: {seq.get('scene_id')}")
        else:
            # LINGO scene ids can be aliases; do not mark as error.
            pass

    human_motion_ref = seq.get("human_motion_ref", {})
    total_frames = joints_length(human_motion_ref, errors, max_errors, {"type": "sequence", "scene_id": scene_id, "sequence_id": seq_id, "path": str(sequence_path)})
    seq_len = None
    if total_frames is not None:
        seq_len = int(human_motion_ref.get("end")) - int(human_motion_ref.get("start"))
        if seq_len <= 0:
            record_error(errors, max_errors, type="sequence", scene_id=scene_id, sequence_id=seq_id, path=str(sequence_path), msg="sequence length <= 0")

    object_list = seq.get("object_list") or []
    object_ids = set()
    for obj in object_list:
        oid = obj.get("object_id")
        if not oid:
            record_error(errors, max_errors, type="sequence", scene_id=scene_id, sequence_id=seq_id, path=str(sequence_path), msg="object_list entry missing object_id")
            continue
        if oid in object_ids:
            record_error(errors, max_errors, type="sequence", scene_id=scene_id, sequence_id=seq_id, path=str(sequence_path), msg=f"duplicate object_id: {oid}")
        object_ids.add(oid)

    segments = seq.get("segment_list") or []
    if not isinstance(segments, list) or len(segments) == 0:
        record_error(errors, max_errors, type="sequence", scene_id=scene_id, sequence_id=seq_id, path=str(sequence_path), msg="segment_list missing or empty")
        return seq_id

    for seg in segments:
        seg_id = seg.get("segment_id")
        start = seg.get("start")
        end = seg.get("end")
        interaction = seg.get("interaction_frame")
        if start is None or end is None or interaction is None:
            record_error(errors, max_errors, type="segment", scene_id=scene_id, sequence_id=seq_id, msg=f"missing start/end/interaction for segment {seg_id}")
            continue
        if seq_len is not None:
            if dataset == "lingo":
                # LINGO end is inclusive in our export.
                end_ok = int(end) <= seq_len
            else:
                end_ok = int(end) < seq_len
            if not (0 <= int(start) <= int(end) and end_ok):
                record_error(errors, max_errors, type="segment", scene_id=scene_id, sequence_id=seq_id, msg=f"segment range invalid: {seg_id} start={start} end={end} len={seq_len}")
            if not (int(start) <= int(interaction) <= int(end)):
                record_error(errors, max_errors, type="segment", scene_id=scene_id, sequence_id=seq_id, msg=f"interaction outside segment: {seg_id} interaction={interaction}")

        goal_pose = seg.get("goal_pose")
        if not isinstance(goal_pose, dict):
            record_error(errors, max_errors, type="segment", scene_id=scene_id, sequence_id=seq_id, msg=f"goal_pose missing: {seg_id}")

        for ref_key in ["acted_on_object_id", "support_object_id"]:
            ref_val = seg.get(ref_key)
            if ref_val is not None and ref_val not in object_ids:
                record_error(errors, max_errors, type="segment", scene_id=scene_id, sequence_id=seq_id, msg=f"{ref_key} not in object_list: {ref_val}")

        plot_path = seg.get("plot_path")
        if plot_path:
            resolved = resolve_path(plot_path)
            if resolved is None or not resolved.exists():
                record_error(errors, max_errors, type="segment", scene_id=scene_id, sequence_id=seq_id, msg=f"plot_path missing: {plot_path}")
            else:
                if scene_id not in plot_path:
                    record_error(errors, max_errors, type="segment", scene_id=scene_id, sequence_id=seq_id, msg=f"plot_path scene mismatch: {plot_path}")

    return seq_id


def run_checks(dataset: str, root: Path, max_errors: int) -> dict:
    errors: list[dict[str, Any]] = []
    stats = defaultdict(int)
    scenes_dir = root / dataset / "scenes"
    if not scenes_dir.exists():
        return {"dataset": dataset, "errors": [{"type": "root", "msg": f"missing scenes dir: {scenes_dir}"}], "stats": {}}

    for scene_path in scenes_dir.glob("*/scene.json"):
        stats["scenes"] += 1
        scene_info = check_scene(scene_path, dataset, errors, max_errors)
        scene_id = scene_info["scene_id"]
        if scene_id is None:
            continue
        sequences_dir = scene_path.parent / "sequences"
        seq_files = list(sequences_dir.glob("*.json"))
        stats["sequences"] += len(seq_files)

        declared_ids = set(scene_info["sequence_ids"])
        found_ids = set()
        for seq_path in seq_files:
            seq_id = check_sequence(seq_path, dataset, scene_id, errors, max_errors)
            if seq_id:
                found_ids.add(seq_id)

        missing = declared_ids - found_ids
        extra = found_ids - declared_ids
        if missing:
            record_error(errors, max_errors, type="scene", scene_id=scene_id, msg=f"missing sequences declared in scene.json: {sorted(list(missing))[:5]} (+{max(0, len(missing)-5)} more)")
        if extra:
            record_error(errors, max_errors, type="scene", scene_id=scene_id, msg=f"extra sequence files not in scene.json: {sorted(list(extra))[:5]} (+{max(0, len(extra)-5)} more)")

    return {"dataset": dataset, "errors": errors, "stats": dict(stats)}


def main() -> None:
    args = parse_args()
    datasets = ["trumans", "lingo"] if args.dataset == "both" else [args.dataset]
    report = {"root": str(args.root), "datasets": []}
    for dataset in datasets:
        report["datasets"].append(run_checks(dataset, args.root, args.max_errors))

    if args.report is None:
        report_path = args.root / "_integrity_report.json"
    else:
        report_path = args.report
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    total_errors = sum(len(d.get("errors", [])) for d in report["datasets"])
    print(f"Integrity report written to: {report_path}")
    print(f"Total errors: {total_errors}")
    if total_errors:
        for dataset_report in report["datasets"]:
            if not dataset_report.get("errors"):
                continue
            sample = dataset_report["errors"][:5]
            print(f"{dataset_report['dataset']} sample errors:")
            for item in sample:
                print("  -", item)


if __name__ == "__main__":
    main()
