from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from preprocess_final import (
    compute_walkable_map,
    majority_filter,
    resolve_lingo_root,
    save_mask,
    TRUMANS_ROOT_DEFAULT,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute clearance maps for TRUMANS/LINGO.")
    parser.add_argument("--dataset", choices=["trumans", "lingo", "both"], default="both")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "data" / "preprocessed")
    parser.add_argument("--trumans-root", type=Path, default=TRUMANS_ROOT_DEFAULT)
    return parser.parse_args()


def recompute_trumans(args: argparse.Namespace) -> None:
    scenes_dir = args.out_root / "trumans" / "scenes"
    for scene_dir in sorted(scenes_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        occ_path = args.trumans_root / "Scene" / f"{scene_id}.npy"
        if not occ_path.exists():
            continue
        occ = np.load(occ_path)
        walkable = compute_walkable_map(occ, use_floor=False)
        np.save(scene_dir / "clearance_map.npy", walkable.astype(np.uint8))
        save_mask(walkable, scene_dir / "clearance_map.png", scale=4)


def recompute_lingo(args: argparse.Namespace) -> None:
    lingo_root = resolve_lingo_root(argparse.Namespace(lingo_root=None))
    scenes_dir = args.out_root / "lingo" / "scenes"
    folder = "Scene" if args.split == "train" else "Scene_vis"
    for scene_dir in sorted(scenes_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        occ_path = lingo_root / folder / f"{scene_id}.npy"
        if not occ_path.exists():
            continue
        occ = np.load(occ_path)
        body_band = occ[:, 1:86, :]
        free_ratio = 1.0 - body_band.mean(axis=1)
        walkable = free_ratio >= 0.95
        walkable = majority_filter(walkable, kernel=5)
        np.save(scene_dir / "clearance_map.npy", walkable.astype(np.uint8))
        save_mask(walkable, scene_dir / "clearance_map.png", scale=4)


def main() -> None:
    args = parse_args()
    if args.dataset in {"trumans", "both"}:
        recompute_trumans(args)
    if args.dataset in {"lingo", "both"}:
        recompute_lingo(args)


if __name__ == "__main__":
    main()
