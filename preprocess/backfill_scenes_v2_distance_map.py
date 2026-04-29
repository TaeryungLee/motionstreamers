from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_scenes_v2 import save_distance_map_v2
from preprocess_final import PROJECT_ROOT, to_repo_relative, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill distance_map.npy for existing scenes_v2 scenes.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--scene-list-file", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "preprocessed")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_scene_ids_local(args: argparse.Namespace) -> list[str]:
    if args.scene_id is not None:
        return [args.scene_id]
    if args.scene_list_file is not None:
        return [line.strip() for line in args.scene_list_file.read_text().splitlines() if line.strip()]
    if args.split is not None:
        split_file = args.output_root / args.dataset / f"{args.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    scenes_root = args.output_root / args.dataset / "scenes_v2"
    return sorted(path.name for path in scenes_root.iterdir() if (path / "scene.json").exists())


def main() -> None:
    args = parse_args()
    scene_ids = resolve_scene_ids_local(args)
    for scene_id in scene_ids:
        scene_root = args.output_root / args.dataset / "scenes_v2" / scene_id
        scene_json = scene_root / "scene.json"
        if not scene_json.exists():
            continue
        scene_record = load_json(scene_json)
        clearance_path = PROJECT_ROOT / scene_record["clearance_map_npy_path"]
        clearance = np.load(clearance_path).astype(bool)
        distance_npy = save_distance_map_v2(scene_root, clearance, scene_record["grid_meta"])
        scene_record["distance_map_npy_path"] = to_repo_relative(distance_npy)
        write_json(scene_json, scene_record)
        print(f"backfilled {args.dataset} {scene_id}")


if __name__ == "__main__":
    main()
