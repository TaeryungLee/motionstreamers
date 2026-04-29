from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump LINGO occupancy Y-slices as images.")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "preprocessed" / "lingo" / "debug_slices")
    return parser.parse_args()


def resolve_lingo_root() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "raw" / "lingo" / "dataset",
        PROJECT_ROOT / "lingo-release" / "dataset",
        Path("/mnt/hdd1/data/lingo_dataset/dataset"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not resolve LINGO dataset root.")


def save_slice(mask: np.ndarray, path: Path, scale: int = 2) -> None:
    img = Image.fromarray((mask.astype(np.uint8) * 255).T, mode="L")
    if scale > 1:
        resampling = getattr(Image, "Resampling", Image)
        img = img.resize((img.width * scale, img.height * scale), resample=resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    args = parse_args()
    root = resolve_lingo_root()
    folder = "Scene" if args.split == "train" else "Scene_vis"
    occ = np.load(root / folder / f"{args.scene_id}.npy")
    out_dir = args.out_dir / args.scene_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for y in range(0, occ.shape[1], args.step):
        mask = occ[:, y, :]
        save_slice(mask, out_dir / f"y_{y:03d}.png")
    print(out_dir)


if __name__ == "__main__":
    main()
