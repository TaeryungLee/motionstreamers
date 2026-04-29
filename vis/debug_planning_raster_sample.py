from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.planning import LingoPlanning, TrumansPlanning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a raster debug image for one planning sample.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-distance-map", action="store_true")
    return parser.parse_args()


def to_rgb(map_2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(map_2d, dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros((16, 16), dtype=np.float32)
    maxv = float(arr.max())
    if maxv > 0:
        arr = arr / maxv
    arr = np.clip(arr, 0.0, 1.0)
    img = (arr * 255.0).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def make_tile(title: str, map_2d: np.ndarray) -> Image.Image:
    rgb = to_rgb(map_2d)
    img = Image.fromarray(rgb, mode="RGB")
    canvas = Image.new("RGB", (img.width, img.height + 24), color=(20, 20, 20))
    canvas.paste(img, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title, fill=(255, 255, 255))
    return canvas


def main() -> None:
    args = parse_args()
    dataset_cls = TrumansPlanning if args.dataset == "trumans" else LingoPlanning
    ds = dataset_cls(scene_id=args.scene_id, include_distance_map=args.include_distance_map)
    item = ds[args.index]

    tiles = [
        make_tile("scene_map_0", item["scene_maps"][0].numpy()),
        make_tile("goal", item["goal"][0].numpy()),
        make_tile("ego_current_occ", item["ego_current_occ"][0].numpy()),
        make_tile("others_current_occ", item["others_current_occ"][0].numpy()),
        make_tile("ego_past_map_sum", item["ego_past_map"].sum(dim=0).numpy()),
        make_tile("others_past_map_sum", item["others_past_map"].sum(dim=0).numpy()),
        make_tile("others_future_occ_t0", item["others_future_occupancy"][0].numpy()),
    ]
    if item["scene_maps"].shape[0] > 1:
        tiles.insert(1, make_tile("scene_map_1_distance", item["scene_maps"][1].numpy()))

    total_width = sum(tile.width for tile in tiles)
    max_height = max(tile.height for tile in tiles)
    canvas = Image.new("RGB", (total_width, max_height), color=(0, 0, 0))
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
