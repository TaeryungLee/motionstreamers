from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check motion continuity across LINGO sequences per scene.")
    parser.add_argument("--preprocessed-root", type=Path, default=PROJECT_ROOT / "data" / "preprocessed" / "lingo")
    parser.add_argument("--lingo-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "preprocessed" / "lingo" / "_continuity_report.json")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "data" / "preprocessed" / "lingo" / "_continuity_summary.json")
    parser.add_argument("--dist-threshold", type=float, default=0.2, help="XZ distance threshold for continuity (meters).")
    parser.add_argument("--gap-threshold", type=int, default=0, help="Max allowed gap in frames (<=).")
    return parser.parse_args()


def resolve_lingo_root(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    candidates = [
        PROJECT_ROOT / "data" / "raw" / "lingo" / "dataset",
        PROJECT_ROOT / "lingo-release" / "dataset",
        Path("/mnt/hdd1/data/lingo_dataset/dataset"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not resolve LINGO dataset root.")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    lingo_root = resolve_lingo_root(args.lingo_root)
    joints = np.load(lingo_root / "human_joints_aligned.npy", mmap_mode="r")

    scenes_dir = args.preprocessed_root / "scenes"
    report: dict[str, Any] = {"root": str(args.preprocessed_root), "scenes": []}
    summary: dict[str, Any] = {"root": str(args.preprocessed_root), "scenes": []}

    for scene_dir in sorted(scenes_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        seq_dir = scene_dir / "sequences"
        if not seq_dir.exists():
            continue
        seq_files = sorted(seq_dir.glob("lingo_*.json"))
        sequences = []
        for p in seq_files:
            d = load_json(p)
            ref = d.get("human_motion_ref", {})
            start = ref.get("start")
            end = ref.get("end")
            if start is None or end is None:
                continue
            sequences.append(
                {
                    "sequence_id": d.get("sequence_id"),
                    "start": int(start),
                    "end": int(end),
                    "path": str(p),
                    "text": d.get("segment_list", [{}])[0].get("text"),
                }
            )
        if len(sequences) < 2:
            continue
        sequences.sort(key=lambda x: (x["start"], x["end"], x["sequence_id"]))

        pairs = []
        summary_pairs = []
        for prev, cur in zip(sequences[:-1], sequences[1:]):
            prev_end = prev["end"] - 1
            cur_start = cur["start"]
            if prev_end < 0 or cur_start < 0 or prev_end >= joints.shape[0] or cur_start >= joints.shape[0]:
                continue
            p_prev = joints[prev_end, 0, :]
            p_cur = joints[cur_start, 0, :]
            delta = p_cur - p_prev
            dist = float(np.linalg.norm(delta[[0, 2]]))
            overlap = cur["start"] <= prev["end"]
            gap = cur["start"] - prev["end"]
            is_continuous = (gap <= args.gap_threshold) and (dist <= args.dist_threshold)
            pairs.append(
                {
                    "prev_sequence_id": prev["sequence_id"],
                    "next_sequence_id": cur["sequence_id"],
                    "prev_start": prev["start"],
                    "prev_end": prev["end"],
                    "next_start": cur["start"],
                    "next_end": cur["end"],
                    "gap": int(gap),
                    "overlap": bool(overlap),
                    "pelvis_delta_xz": [float(delta[0]), float(delta[2])],
                    "pelvis_dist_xz": dist,
                    "prev_text": prev.get("text"),
                    "next_text": cur.get("text"),
                }
            )
            summary_pairs.append(
                {
                    "prev_sequence_id": prev["sequence_id"],
                    "next_sequence_id": cur["sequence_id"],
                    "gap": int(gap),
                    "pelvis_dist_xz": dist,
                    "continuous": bool(is_continuous),
                }
            )

        report["scenes"].append(
            {
                "scene_id": scene_dir.name,
                "num_sequences": len(sequences),
                "num_pairs": len(pairs),
                "pairs": pairs,
            }
        )
        if summary_pairs:
            cont = sum(1 for p in summary_pairs if p["continuous"])
            summary["scenes"].append(
                {
                    "scene_id": scene_dir.name,
                    "num_sequences": len(sequences),
                    "num_pairs": len(summary_pairs),
                    "continuous_pairs": cont,
                    "continuity_ratio": cont / len(summary_pairs),
                    "pairs": summary_pairs,
                }
            )

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote continuity report: {args.out}")
    print(f"Wrote continuity summary: {args.summary}")


if __name__ == "__main__":
    main()
