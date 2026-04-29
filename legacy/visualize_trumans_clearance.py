import argparse
from pathlib import Path

import numpy as np
from PIL import Image


SCENE_GRID = (-3.0, 0.0, -4.0, 3.0, 2.0, 4.0, 300, 100, 400)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize TRUMANS 2D clearance / walkable maps from scene occupancy."
    )
    parser.add_argument("scene_id", help="Scene id, e.g. 4ad6d9bd-5131-4926-a8e5-801140c7127d")
    parser.add_argument(
        "--trumans-root",
        type=Path,
        default=Path("data/raw/trumans"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )
    parser.add_argument(
        "--height-cm",
        type=float,
        default=170.0,
        help="Human height in cm used to define the clearance band.",
    )
    parser.add_argument(
        "--free-threshold",
        type=float,
        default=0.95,
        help="Required free ratio inside the clearance band.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Output image upscaling factor.",
    )
    return parser.parse_args()


def save_mask(mask: np.ndarray, path: Path, scale: int) -> None:
    img = Image.fromarray((mask.astype(np.uint8) * 255).T, mode="L")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), resample=Image.NEAREST)
    img.save(path)


def main():
    args = parse_args()
    scene_path = args.trumans_root / "Scene" / f"{args.scene_id}.npy"
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene occupancy not found: {scene_path}")

    occ = np.load(scene_path)
    if occ.shape != (SCENE_GRID[6], SCENE_GRID[7], SCENE_GRID[8]):
        raise ValueError(f"Unexpected scene occupancy shape: {occ.shape}")

    y_bins = int(SCENE_GRID[7])
    y_extent_m = SCENE_GRID[4] - SCENE_GRID[1]
    meters_per_voxel = y_extent_m / y_bins
    body_top_idx = min(y_bins - 1, int(round((args.height_cm / 100.0) / meters_per_voxel)))

    y0_occupied = occ[:, 0, :]
    free_ratio = (~occ[:, 1 : body_top_idx + 1, :]).mean(axis=1)
    walkable = y0_occupied & (free_ratio >= args.free_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"trumans_{args.scene_id}"
    height_tag = int(round(args.height_cm))
    thr_tag = int(round(args.free_threshold * 100))

    save_mask(y0_occupied, args.output_dir / f"{prefix}_y0_walkable.png", args.scale)
    save_mask(walkable, args.output_dir / f"{prefix}_walkable_h{height_tag}_thr{thr_tag:03d}.png", args.scale)

    y0_pgm = args.output_dir / f"{prefix}_y0_walkable.pgm"
    Image.fromarray((y0_occupied.astype(np.uint8) * 255).T, mode="L").save(y0_pgm)

    y_slices_dir = args.output_dir / f"{prefix}_y_slices"
    y_slices_dir.mkdir(parents=True, exist_ok=True)
    for y in range(0, y_bins, 10):
        save_mask(occ[:, y, :], y_slices_dir / f"y_{y:03d}.png", args.scale)

    y_0_to_10_dir = args.output_dir / f"{prefix}_y_0_to_10"
    y_0_to_10_dir.mkdir(parents=True, exist_ok=True)
    for y in range(0, 11):
        save_mask(occ[:, y, :], y_0_to_10_dir / f"y_{y:03d}.png", args.scale)

    print(f"scene_path={scene_path}")
    print(f"shape={occ.shape}")
    print(f"meters_per_voxel={meters_per_voxel:.6f}")
    print(f"body_band=1..{body_top_idx}")
    print(f"y0_occupied_ratio={y0_occupied.mean():.6f}")
    print(f"walkable_ratio={walkable.mean():.6f}")
    print(args.output_dir / f"{prefix}_walkable_h{height_tag}_thr{thr_tag:03d}.png")


if __name__ == "__main__":
    main()
