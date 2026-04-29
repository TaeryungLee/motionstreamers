import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LINGO_LEFT_HAND_IDX = 24
LINGO_RIGHT_HAND_IDX = 26


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize LINGO sequence-level action segments.")
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--segment-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--body-height-voxels", type=int, default=85)
    parser.add_argument("--free-threshold", type=float, default=0.95)
    return parser.parse_args()


def resolve_lingo_root() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "raw" / "lingo" / "dataset",
        PROJECT_ROOT / "lingo-release" / "dataset",
        Path("/mnt/hdd1/data/lingo_dataset/dataset"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not resolve LINGO dataset root.")


def world_to_px(x, z, x_min, x_max, z_min, z_max, x_res, z_res):
    px = int(round((x - x_min) / (x_max - x_min) * (x_res - 1)))
    pz = int(round((z - z_min) / (z_max - z_min) * (z_res - 1)))
    return max(0, min(x_res - 1, px)), max(0, min(z_res - 1, pz))


def draw_polyline(draw, pts, color, width=2):
    if len(pts) < 2:
        return
    for p0, p1 in zip(pts[:-1], pts[1:]):
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=width)


def draw_text_block(draw, xy, lines, fill, bg_fill=(255, 255, 255), padding=6, line_gap=4):
    x, y = xy
    widths = []
    heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    box_w = max(widths) + padding * 2
    box_h = sum(heights) + line_gap * (len(lines) - 1) + padding * 2
    draw.rectangle((x, y, x + box_w, y + box_h), fill=bg_fill)
    cursor_y = y + padding
    for line, height in zip(lines, heights):
        draw.text((x + padding, cursor_y), line, fill=fill)
        cursor_y += height + line_gap


def wrap_text_to_width(draw, text, max_width):
    words = text.split()
    if not words:
        return [text]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


from typing import Optional


def majority_filter(mask: np.ndarray, kernel: int = 5, threshold: Optional[int] = None) -> np.ndarray:
    if kernel % 2 == 0:
        raise ValueError("kernel must be odd")
    if threshold is None:
        threshold = (kernel * kernel) // 2 + 1
    pad = kernel // 2
    padded = np.pad(mask.astype(np.uint8), pad, mode="constant")
    acc = np.zeros_like(mask, dtype=np.uint8)
    for dx in range(kernel):
        for dz in range(kernel):
            acc += padded[dx : dx + mask.shape[0], dz : dz + mask.shape[1]]
    return acc >= threshold


def build_walkable_background(scene_occ, body_height_voxels, free_threshold):
    body_band = scene_occ[:, 1 : body_height_voxels + 1, :]
    free_ratio = 1.0 - body_band.mean(axis=1)
    walkable = free_ratio >= free_threshold
    walkable = majority_filter(walkable, kernel=5)
    bg = np.where(walkable, 245, 25).astype(np.uint8).T
    return np.stack([bg, bg, bg], axis=-1)


def main():
    args = parse_args()
    lingo_root = resolve_lingo_root()
    payload = json.loads(args.segment_json.read_text())
    segments = payload["segment_list"]
    human_motion_ref = payload.get("human_motion_ref", {})
    global_start = int(human_motion_ref.get("start", 0))

    joints = np.load(lingo_root / "human_joints_aligned.npy", mmap_mode="r")
    scene_folder = "Scene" if args.split == "train" else "Scene_vis"
    scene_occ = np.load(lingo_root / scene_folder / f"{args.scene_name}.npy")
    bg_rgb = build_walkable_background(scene_occ, args.body_height_voxels, args.free_threshold)

    if args.split == "train":
        x_min, x_max, z_min, z_max = -3.0, 3.0, -4.0, 4.0
    else:
        x_min, x_max, z_min, z_max = -4.0, 4.0, -6.0, 6.0
    x_res, z_res = scene_occ.shape[0], scene_occ.shape[2]

    def to_px(arr):
        return [
            world_to_px(float(p[0]), float(p[2]), x_min, x_max, z_min, z_max, x_res, z_res)
            for p in arr
        ]

    out_dir = args.output_dir / f"lingo_sequence_actions_{args.sequence_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for event in segments:
        start = int(event["start"])
        end = int(event["end"])
        interaction = int(event["interaction_frame"])
        g_start = global_start + start
        g_end = global_start + end
        g_interaction = global_start + interaction

        pelvis_seg = np.asarray(joints[g_start:g_end, 0, :], dtype=np.float32)
        left_seg = np.asarray(joints[g_start:g_end, LINGO_LEFT_HAND_IDX, :], dtype=np.float32)
        right_seg = np.asarray(joints[g_start:g_end, LINGO_RIGHT_HAND_IDX, :], dtype=np.float32)

        pelvis_interact = np.asarray(joints[g_interaction, 0, :], dtype=np.float32)
        left_interact = np.asarray(joints[g_interaction, LINGO_LEFT_HAND_IDX, :], dtype=np.float32)
        right_interact = np.asarray(joints[g_interaction, LINGO_RIGHT_HAND_IDX, :], dtype=np.float32)

        object_loc = None
        if event.get("object_pose") and event["object_pose"].get("location") is not None:
            object_loc = np.asarray(event["object_pose"]["location"], dtype=np.float32)

        header_h = 92
        canvas = Image.new("RGB", (bg_rgb.shape[1], bg_rgb.shape[0] + header_h), (25, 25, 25))
        canvas.paste(Image.fromarray(bg_rgb.copy(), mode="RGB"), (0, header_h))
        draw = ImageDraw.Draw(canvas)

        def shift_pts(points):
            return [(x, y + header_h) for x, y in points]

        draw_polyline(draw, shift_pts(to_px(pelvis_seg)), (255, 140, 0), width=4)
        if event.get("hand") in {"left", "both"}:
            draw_polyline(draw, shift_pts(to_px(left_seg)), (60, 180, 75), width=3)
        if event.get("hand") in {"right", "both"}:
            draw_polyline(draw, shift_pts(to_px(right_seg)), (70, 130, 255), width=3)

        points_to_draw = [
            (pelvis_interact, (220, 30, 30), "pelvis*"),
        ]
        if event.get("hand") in {"left", "both"}:
            points_to_draw.append((left_interact, (60, 180, 75), "L*"))
        if event.get("hand") in {"right", "both"}:
            points_to_draw.append((right_interact, (70, 130, 255), "R*"))
        if object_loc is not None:
            points_to_draw.append((object_loc, (180, 0, 255), event.get("goal_object") or event.get("goal_object_name") or "object"))

        for pt, color, label in points_to_draw:
            px, pz = world_to_px(float(pt[0]), float(pt[2]), x_min, x_max, z_min, z_max, x_res, z_res)
            radius = 5 if label.startswith("pelvis") else 4
            draw.ellipse(
                (px - radius, pz + header_h - radius, px + radius, pz + header_h + radius),
                fill=color,
                outline=(255, 255, 255),
            )
            draw.text((px + 6, pz + header_h - 8), label, fill=color)

        header = f"{event['segment_id']:02d} {event['text']}"
        detail = f"interaction={interaction} obj={event.get('goal_object')}"
        wrapped = []
        for line in [header, detail]:
            wrapped.extend(wrap_text_to_width(draw, line, canvas.width - 24))
        draw_text_block(draw, (10, 10), wrapped, fill=(220, 20, 20), bg_fill=(255, 255, 255))

        out_path = out_dir / f"{event['segment_id']:02d}_{start:04d}_{end:04d}.png"
        canvas.save(out_path)




if __name__ == "__main__":
    main()
