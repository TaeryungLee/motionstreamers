import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def world_to_px(x, z, x_min, x_max, z_min, z_max, x_res, z_res):
    px = int(round((x - x_min) / (x_max - x_min) * (x_res - 1)))
    pz = int(round((z - z_min) / (z_max - z_min) * (z_res - 1)))
    return max(0, min(x_res - 1, px)), max(0, min(z_res - 1, pz))


def draw_polyline(draw, pts, color, width=2):
    if len(pts) < 2:
        return
    for p0, p1 in zip(pts[:-1], pts[1:]):
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=width)


def build_walkable_background(scene_occ, body_height_voxels=85, free_threshold=0.95):
    floor = scene_occ[:, 0, :]
    body_band = scene_occ[:, 1 : body_height_voxels + 1, :]
    free_ratio = 1.0 - body_band.mean(axis=1)
    walkable = floor & (free_ratio >= free_threshold)
    bg = np.where(walkable, 245, 25).astype(np.uint8).T
    return np.stack([bg, bg, bg], axis=-1)


def blender_world_to_motion(loc):
    return np.array([loc[0], loc[2], -loc[1]], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize nearby Blender object candidates for a TRUMANS action slice."
    )
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--event-start", type=int, required=True)
    parser.add_argument("--event-end", type=int, required=True)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--body-height-voxels", type=int, default=85)
    parser.add_argument("--free-threshold", type=float, default=0.95)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/trumans"))
    parser.add_argument("--blender-json", type=Path, default=Path("outputs/4ad6d9bd-5131-4926-a8e5-801140c7127d.transforms.json"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    seg = np.load(args.data_root / "seg_name.npy", allow_pickle=True)
    joints = np.load(args.data_root / "human_joints.npy", mmap_mode="r")
    scene = np.load(args.data_root / "Scene" / f"{args.scene_name}.npy")

    idx = np.where(seg == args.sequence_id)[0]
    if len(idx) == 0:
        raise ValueError(f"Sequence not found: {args.sequence_id}")

    frame_ids = idx[args.event_start : args.event_end + 1]
    pelvis = np.asarray(joints[frame_ids, 0, :], dtype=np.float32)
    pelvis_mean = pelvis.mean(axis=0)
    pelvis_end = pelvis[-1]

    blender_dump = json.loads(args.blender_json.read_text())
    candidates = []
    for obj in blender_dump["objects"]:
        if obj["type"] not in {"MESH", "EMPTY"}:
            continue
        motion_loc = blender_world_to_motion(obj["world_location"])
        dist = float(np.linalg.norm(motion_loc - pelvis_mean))
        candidates.append(
            {
                "name": obj["name"],
                "type": obj["type"],
                "parent": obj["parent"],
                "data_name": obj.get("data_name"),
                "motion_location": motion_loc,
                "dist": dist,
            }
        )
    candidates.sort(key=lambda row: row["dist"])
    candidates = candidates[: args.topk]

    bg = build_walkable_background(scene, args.body_height_voxels, args.free_threshold)
    canvas = Image.fromarray(bg, mode="RGB")
    draw = ImageDraw.Draw(canvas)

    x_min, x_max = -3.0, 3.0
    z_min, z_max = -4.0, 4.0
    x_res, z_res = scene.shape[0], scene.shape[2]

    pelvis_px = [
        world_to_px(float(p[0]), float(p[2]), x_min, x_max, z_min, z_max, x_res, z_res)
        for p in pelvis
    ]
    draw_polyline(draw, pelvis_px, (255, 140, 0), width=4)

    for pt, color, label in [
        (pelvis[0], (255, 196, 120), "start"),
        (pelvis_end, (220, 30, 30), "end"),
        (pelvis_mean, (255, 80, 80), "mean"),
    ]:
        px, pz = world_to_px(float(pt[0]), float(pt[2]), x_min, x_max, z_min, z_max, x_res, z_res)
        draw.ellipse((px - 5, pz - 5, px + 5, pz + 5), fill=color, outline=(255, 255, 255))
        draw.text((px + 6, pz - 8), label, fill=color)

    palette = [
        (180, 0, 255),
        (0, 200, 200),
        (255, 220, 0),
        (255, 0, 120),
        (0, 120, 255),
        (120, 180, 0),
        (255, 128, 0),
        (120, 120, 255),
        (255, 80, 80),
        (80, 80, 80),
        (0, 160, 120),
        (180, 120, 0),
    ]

    text_y = 10
    draw.text(
        (10, text_y),
        f"{args.sequence_id} {args.event_start}-{args.event_end} sit candidates",
        fill=(0, 0, 0),
    )
    text_y += 18
    for rank, row in enumerate(candidates, start=1):
        color = palette[(rank - 1) % len(palette)]
        loc = row["motion_location"]
        px, pz = world_to_px(float(loc[0]), float(loc[2]), x_min, x_max, z_min, z_max, x_res, z_res)
        draw.rectangle((px - 4, pz - 4, px + 4, pz + 4), fill=color, outline=(255, 255, 255))
        draw.text((px + 6, pz - 8), str(rank), fill=color)
        label = f"{rank:02d} {row['name']} d={row['dist']:.3f} data={row['data_name']}"
        draw.text((10, text_y), label[:120], fill=color)
        text_y += 16

    if args.output is None:
        output = Path("outputs") / f"trumans_blender_candidates_{args.sequence_id}_{args.event_start:04d}_{args.event_end:04d}.png"
    else:
        output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)


if __name__ == "__main__":
    main()
