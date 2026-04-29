from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize TRUMANS 2D interaction heatmaps from scene.json.")
    parser.add_argument("--scene-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sigma", type=float, default=8.0, help="Gaussian sigma in grid cells.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def to_abs(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root() / path


def compute_walkable(scene_occ: np.ndarray, body_height_voxels: int = 85, free_threshold: float = 0.95) -> np.ndarray:
    floor = scene_occ[:, 0, :]
    body_band = scene_occ[:, 1 : body_height_voxels + 1, :]
    free_ratio = 1.0 - body_band.mean(axis=1)
    return floor & (free_ratio >= free_threshold)


def world_to_grid(point_xyz: list[float] | np.ndarray, x_res: int, z_res: int, x_min: float = -3.0, x_max: float = 3.0, z_min: float = -4.0, z_max: float = 4.0) -> tuple[int, int]:
    point = np.asarray(point_xyz, dtype=np.float32)
    gx = int(round((float(point[0]) - x_min) / (x_max - x_min) * (x_res - 1)))
    gz = int(round((float(point[2]) - z_min) / (z_max - z_min) * (z_res - 1)))
    gx = max(0, min(x_res - 1, gx))
    gz = max(0, min(z_res - 1, gz))
    return gx, gz


def add_gaussian(heatmap: np.ndarray, center_xy: tuple[int, int], sigma: float) -> None:
    x0, y0 = center_xy
    radius = int(max(3, round(3 * sigma)))
    x_min = max(0, x0 - radius)
    x_max = min(heatmap.shape[0], x0 + radius + 1)
    y_min = max(0, y0 - radius)
    y_max = min(heatmap.shape[1], y0 + radius + 1)
    xs = np.arange(x_min, x_max)[:, None]
    ys = np.arange(y_min, y_max)[None, :]
    blob = np.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * sigma * sigma))
    heatmap[x_min:x_max, y_min:y_max] += blob


def normalize_map(heatmap: np.ndarray) -> np.ndarray:
    if float(heatmap.max()) <= 0:
        return heatmap
    return heatmap / float(heatmap.max())


def plot_overlay(base: np.ndarray, heatmap: np.ndarray, objects: list[dict], title: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 9))
    plt.imshow(base.T, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
    if float(heatmap.max()) > 0:
        plt.imshow(heatmap.T, origin="lower", cmap="hot", alpha=0.65, vmin=0.0, vmax=max(1e-6, float(heatmap.max())))
    for obj in objects:
        x, y = obj["grid_xy"]
        plt.scatter([x], [y], c="cyan", s=20)
        plt.text(x + 2, y + 2, obj["object_id"], color="cyan", fontsize=7)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    scene_json = json.loads(args.scene_json.read_text())
    output_dir = args.output_dir or args.scene_json.parent / "interaction_maps"
    scene_occ = np.load(to_abs(scene_json["scene_occ_path"]))
    walkable = compute_walkable(scene_occ)
    base = walkable.astype(np.float32)

    x_res, z_res = scene_occ.shape[0], scene_occ.shape[2]
    object_entries = []
    pelvis_map = np.zeros((x_res, z_res), dtype=np.float32)
    hand_map = np.zeros((x_res, z_res), dtype=np.float32)
    combined_map = np.zeros((x_res, z_res), dtype=np.float32)

    for obj in scene_json["object_list"]:
        grid_xy = world_to_grid(obj["initial_location"], x_res, z_res)
        object_entries.append({"object_id": obj["object_id"], "grid_xy": grid_xy})
        for interaction in obj.get("interactions", []):
            pelvis_xy = world_to_grid(interaction["pelvis_position"], x_res, z_res)
            add_gaussian(pelvis_map, pelvis_xy, args.sigma)
            add_gaussian(combined_map, pelvis_xy, args.sigma)
            if interaction.get("hand_position") is not None:
                hand_xy = world_to_grid(interaction["hand_position"], x_res, z_res)
                add_gaussian(hand_map, hand_xy, args.sigma)
                add_gaussian(combined_map, hand_xy, args.sigma)

            object_xy = world_to_grid(interaction["object_position"], x_res, z_res)
            add_gaussian(combined_map, object_xy, args.sigma * 0.7)

    plot_overlay(base, normalize_map(combined_map), object_entries, "Combined Interaction Map", output_dir / "combined_interaction_map.png")
    plot_overlay(base, normalize_map(pelvis_map), object_entries, "Pelvis Interaction Map", output_dir / "pelvis_interaction_map.png")
    plot_overlay(base, normalize_map(hand_map), object_entries, "Hand Interaction Map", output_dir / "hand_interaction_map.png")

    for obj in scene_json["object_list"]:
        obj_map = np.zeros((x_res, z_res), dtype=np.float32)
        obj_entry = {"object_id": obj["object_id"], "grid_xy": world_to_grid(obj["initial_location"], x_res, z_res)}
        for interaction in obj.get("interactions", []):
            add_gaussian(obj_map, world_to_grid(interaction["pelvis_position"], x_res, z_res), args.sigma)
        plot_overlay(base, normalize_map(obj_map), [obj_entry], f"{obj['object_id']} Pelvis Interaction Map", output_dir / f"{obj['object_id']}.png")

    print(output_dir)


if __name__ == "__main__":
    main()
