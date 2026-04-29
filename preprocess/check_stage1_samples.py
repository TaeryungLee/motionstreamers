from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_multi_character_affordances import PROJECT_ROOT


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "preprocessed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Stage-1 sample coverage per ego window.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def stage1_root(output_root: Path, dataset: str) -> Path:
    return output_root / dataset / "stage1_samples_v1"


def resolve_scene_dirs(args: argparse.Namespace) -> list[Path]:
    root = stage1_root(args.output_root, args.dataset)
    if args.scene_id is not None:
        scene_dir = root / args.scene_id
        if not scene_dir.exists():
            raise FileNotFoundError(scene_dir)
        return [scene_dir]
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(path for path in root.iterdir() if path.is_dir())


def ego_key(sample: dict) -> tuple:
    ego = sample["ego"]
    return (
        sample["scene_id"],
        ego["sequence_id"],
        int(ego["segment_id"]),
        int(ego["window_start"]),
        int(ego["window_end"]),
    )


def summarize_counter(values: list[int]) -> dict:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "min": 0,
            "max": 0,
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": int(arr.min()),
        "max": int(arr.max()),
    }


def summarize_scene(scene_dir: Path) -> dict:
    sample_paths = sorted(scene_dir.glob("sample_*.json"))
    per_ego_counts: dict[tuple, Counter] = defaultdict(Counter)
    for path in sample_paths:
        sample = load_json(path)
        nchar = 1 + len(sample.get("others", []))
        per_ego_counts[ego_key(sample)][nchar] += 1

    two_counts = [counter.get(2, 0) for counter in per_ego_counts.values()]
    three_counts = [counter.get(3, 0) for counter in per_ego_counts.values()]
    four_counts = [counter.get(4, 0) for counter in per_ego_counts.values()]

    exact_pattern_hist = Counter(
        (counter.get(2, 0), counter.get(3, 0), counter.get(4, 0))
        for counter in per_ego_counts.values()
    )
    top_patterns = [
        {"counts_2_3_4": list(pattern), "num_egos": int(freq)}
        for pattern, freq in exact_pattern_hist.most_common(10)
    ]

    return {
        "scene_id": scene_dir.name,
        "num_samples": len(sample_paths),
        "num_unique_egos": len(per_ego_counts),
        "two_char_per_ego": summarize_counter(two_counts),
        "three_char_per_ego": summarize_counter(three_counts),
        "four_char_per_ego": summarize_counter(four_counts),
        "top_ego_count_patterns": top_patterns,
        "egos_missing_2char": int(sum(1 for v in two_counts if v == 0)),
        "egos_missing_3char": int(sum(1 for v in three_counts if v == 0)),
        "egos_missing_4char": int(sum(1 for v in four_counts if v == 0)),
    }


def main() -> None:
    args = parse_args()
    scene_dirs = resolve_scene_dirs(args)
    summaries = [summarize_scene(scene_dir) for scene_dir in scene_dirs]
    for summary in summaries:
        print(json.dumps(summary, indent=2))

    if len(summaries) > 1:
        aggregate = {
            "dataset": args.dataset,
            "num_scenes": len(summaries),
            "num_samples": int(sum(item["num_samples"] for item in summaries)),
            "num_unique_egos": int(sum(item["num_unique_egos"] for item in summaries)),
            "scenes_with_missing_2char": int(sum(1 for item in summaries if item["egos_missing_2char"] > 0)),
            "scenes_with_missing_3char": int(sum(1 for item in summaries if item["egos_missing_3char"] > 0)),
            "scenes_with_missing_4char": int(sum(1 for item in summaries if item["egos_missing_4char"] > 0)),
        }
        print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
