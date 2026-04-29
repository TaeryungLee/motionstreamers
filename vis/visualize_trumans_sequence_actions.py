import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ACTION_LABELS = [
    "Lie down",
    "Squat",
    "Mouse",
    "Keyboard",
    "Laptop",
    "Phone",
    "Book",
    "Bottle",
    "Pen",
    "Vase",
]

OBJECT_INTERACTION_KEYWORDS = (
    "pick up",
    "put down",
    "open",
    "close",
    "move",
    "drag",
    "slide",
    "wipe",
    "type",
    "write",
    "drink",
    "pour",
    "water",
    "hold",
    "tap",
    "click",
    "dial",
    "answer",
    "call",
    "use",
)

BODY_ACTION_KEYWORDS = (
    "sit down",
    "stand up",
    "lie down",
    "squat",
    "kneel",
    "crouch",
)

BODY_ACTION_NEGATIVE_KEYWORDS = (
    "bottle",
    "cup",
    "phone",
    "pen",
    "mouse",
    "handbag",
    "vase",
    "book",
    "keyboard",
    "laptop",
    "microwave",
    "oven",
    "drawer",
    "cabinet",
    "door",
    "fridge",
    "refrigerator",
)

BODY_ACTION_POSITIVE_KEYWORDS = (
    "chair",
    "seat",
    "stool",
    "bench",
    "sofa",
    "couch",
    "bed",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize TRUMANS sequence-level action segments with pelvis/LH/RH trajectories."
    )
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/trumans"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )
    parser.add_argument("--segment-json", type=Path, default=None)
    parser.add_argument(
        "--body-height-voxels",
        type=int,
        default=85,
        help="170cm ~= 85 voxels when y spans 0..2m over 100 voxels.",
    )
    parser.add_argument(
        "--free-threshold",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--hand-distance-threshold",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--body-distance-threshold",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="Write summary.json alongside the rendered images.",
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=0.05,
    )
    return parser.parse_args()


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


def load_event_lines(action_file: Path):
    lines = action_file.read_text().strip().splitlines()
    events = []
    for event_i, line in enumerate(lines):
        start, end, text = line.split("\t")
        events.append(
            {
                "event_id": event_i,
                "start": int(start),
                "end": int(end),
                "text": text,
            }
        )
    return events


def build_walkable_background(scene_occ, body_height_voxels, free_threshold):
    body_band = scene_occ[:, 1 : body_height_voxels + 1, :]
    free_ratio = 1.0 - body_band.mean(axis=1)
    walkable = free_ratio >= free_threshold
    bg = np.where(walkable, 245, 25).astype(np.uint8).T
    return np.stack([bg, bg, bg], axis=-1)


def nearest_objects(obj_all, frame_idx_local, query_xyz, topk=5):
    records = []
    for name, entry in obj_all.items():
        loc = np.asarray(entry["location"][frame_idx_local], dtype=np.float32)
        dist = float(np.linalg.norm(query_xyz - loc))
        records.append((dist, name, loc))
    records.sort(key=lambda x: x[0])
    return records[:topk]


def blender_world_to_motion(location):
    loc = np.asarray(location, dtype=np.float32)
    return np.array([loc[0], loc[2], -loc[1]], dtype=np.float32)


def nearest_blender_objects(blender_objects, query_xyz, topk=5):
    records = []
    for obj in blender_objects:
        loc = obj["motion_location"]
        dist = float(np.linalg.norm(query_xyz - loc))
        records.append((dist, obj["name"], loc))
    records.sort(key=lambda x: x[0])
    return records[:topk]


def filter_body_action_candidates(blender_objects):
    filtered = []
    for obj in blender_objects:
        name = obj["name"].lower()
        if any(keyword in name for keyword in BODY_ACTION_NEGATIVE_KEYWORDS):
            continue
        filtered.append(obj)
    return filtered


def select_query_track(text, pelvis_seg, left_seg, right_seg):
    if text.startswith("Left hand"):
        return "left", left_seg
    if text.startswith("Right hand"):
        return "right", right_seg
    return "pelvis", pelvis_seg


def query_name_from_segment(event):
    goal_type = event.get("goal_type")
    hand = event.get("hand")
    if goal_type == "body":
        return "pelvis"
    if hand in {"left", "right", "both"}:
        return hand
    if goal_type == "hand":
        return "hand"
    return "pelvis"


def text_implies_object_interaction(text):
    lower = text.lower()
    return any(keyword in lower for keyword in OBJECT_INTERACTION_KEYWORDS)


def summarize_action_label(label_slice, threshold):
    colmax = label_slice.max(axis=0)
    active = [
        {"label": ACTION_LABELS[i], "score": float(colmax[i])}
        for i in range(len(ACTION_LABELS))
        if colmax[i] > threshold
    ]
    active.sort(key=lambda row: row["score"], reverse=True)
    return active


def trumans_base_sequence_id(sequence_id):
    return sequence_id.split("_augment", 1)[0]


def main():
    args = parse_args()
    root = args.data_root
    seq = args.sequence_id
    source_seq = trumans_base_sequence_id(seq)
    scene_name = args.scene_name

    seg = np.load(root / "seg_name.npy", allow_pickle=True)
    joints = np.load(root / "human_joints.npy", mmap_mode="r")
    action_label = np.load(root / "action_label.npy")
    object_flag = np.load(root / "object_flag.npy")
    object_list = np.load(root / "object_list.npy", allow_pickle=True)
    scene = np.load(root / "Scene" / f"{scene_name}.npy")
    obj_all_path = root / "Object_all" / "Object_pose" / f"{source_seq}.npy"
    try:
        obj_all = np.load(obj_all_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"[warn] Object_all not found: {obj_all_path}")
        obj_all = {}
    blend_dump_path = args.output_dir / f"{scene_name}.transforms.json"
    blender_objects = []
    blender_by_name = {}
    if blend_dump_path.exists():
        dump = json.loads(blend_dump_path.read_text())
        for obj in dump["objects"]:
            if obj["type"] not in {"MESH", "EMPTY"}:
                continue
            record = {
                "name": obj["name"],
                "motion_location": blender_world_to_motion(obj["world_location"]),
            }
            blender_objects.append(record)
            blender_by_name[obj["name"]] = record

    idx = np.where(seg == seq)[0]
    if len(idx) == 0:
        raise ValueError(f"Sequence not found in seg_name.npy: {seq}")

    if args.segment_json is not None:
        events = json.loads(args.segment_json.read_text())["segment_list"]
    else:
        action_file = root / "Actions" / f"{source_seq}.txt"
        events = load_event_lines(action_file)

    pelvis_full = np.asarray(joints[idx, 0, :], dtype=np.float32)
    left_full = np.asarray(joints[idx, 22, :], dtype=np.float32)
    right_full = np.asarray(joints[idx, 23, :], dtype=np.float32)

    bg_rgb = build_walkable_background(
        scene_occ=scene,
        body_height_voxels=args.body_height_voxels,
        free_threshold=args.free_threshold,
    )

    x_min, x_max = -3.0, 3.0
    z_min, z_max = -4.0, 4.0
    x_res, z_res = scene.shape[0], scene.shape[2]

    def to_px(arr):
        return [
            world_to_px(
                float(p[0]),
                float(p[2]),
                x_min,
                x_max,
                z_min,
                z_max,
                x_res,
                z_res,
            )
            for p in arr
        ]

    out_dir = args.output_dir / f"trumans_sequence_actions_{seq}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    prev_end = None
    for event in events:
        event_id = int(event.get("event_id", event.get("segment_id", 0)))
        start = event["start"]
        end = event["end"]
        text = event["text"]

        render_start = start if prev_end is None else prev_end
        render_frame_ids = idx[render_start : end + 1]
        pelvis_seg = np.asarray(joints[render_frame_ids, 0, :], dtype=np.float32)
        left_seg = np.asarray(joints[render_frame_ids, 22, :], dtype=np.float32)
        right_seg = np.asarray(joints[render_frame_ids, 23, :], dtype=np.float32)

        action_frame_ids = idx[start : end + 1]
        pelvis_action = np.asarray(joints[action_frame_ids, 0, :], dtype=np.float32)
        left_action = np.asarray(joints[action_frame_ids, 22, :], dtype=np.float32)
        right_action = np.asarray(joints[action_frame_ids, 23, :], dtype=np.float32)
        label_info = summarize_action_label(action_label[action_frame_ids], args.label_threshold)
        active_slot_ids = sorted({int(slot_id) for row in object_flag[action_frame_ids] for slot_id in np.where(row >= 0)[0]})
        active_object_names = [str(object_list[slot_id]) for slot_id in active_slot_ids]
        query_name = query_name_from_segment(event)

        object_name = event.get("goal_object")
        object_source = event.get("goal_object_source")
        interaction_local_frame = int(event.get("interaction_frame", (start + end) // 2))
        interaction_global = int(idx[interaction_local_frame])
        pelvis_interact = np.asarray(joints[interaction_global, 0, :], dtype=np.float32)
        left_interact = np.asarray(joints[interaction_global, 22, :], dtype=np.float32)
        right_interact = np.asarray(joints[interaction_global, 23, :], dtype=np.float32)
        object_seg = None
        object_loc = None
        if object_name is not None and object_source == "Object_all" and object_name in obj_all:
            object_seg = np.asarray(
                obj_all[object_name]["location"][render_start : end + 1],
                dtype=np.float32,
            )
            frame_offset = max(0, min(interaction_local_frame - render_start, len(object_seg) - 1))
            object_loc = object_seg[frame_offset]
        elif event.get("goal_object_location") is not None:
            object_loc = np.asarray(event["goal_object_location"], dtype=np.float32)
        elif object_name is not None and object_source == "blend_dump" and object_name in blender_by_name:
            object_loc = np.asarray(blender_by_name[object_name]["motion_location"], dtype=np.float32)

        header_h = 92
        canvas = Image.new("RGB", (bg_rgb.shape[1], bg_rgb.shape[0] + header_h), (25, 25, 25))
        canvas.paste(Image.fromarray(bg_rgb.copy(), mode="RGB"), (0, header_h))
        draw = ImageDraw.Draw(canvas)

        # Current event segment trajectories.
        def shift_pts(points):
            return [(x, y + header_h) for x, y in points]

        draw_polyline(draw, shift_pts(to_px(pelvis_seg)), (255, 140, 0), width=4)
        draw_polyline(draw, shift_pts(to_px(left_seg)), (60, 180, 75), width=3)
        draw_polyline(draw, shift_pts(to_px(right_seg)), (70, 130, 255), width=3)
        if object_seg is not None:
            draw_polyline(draw, shift_pts(to_px(object_seg)), (180, 0, 255), width=3)

        points_to_draw = [
            (pelvis_interact, (220, 30, 30), "pelvis*"),
            (left_interact, (60, 180, 75), "L*"),
            (right_interact, (70, 130, 255), "R*"),
        ]
        if object_loc is not None:
            points_to_draw.append((object_loc, (180, 0, 255), object_name))
        dataset_active_points = []
        if active_object_names:
            for name in active_object_names:
                dataset_name = name.split("(", 1)[0].strip()
                if not dataset_name or dataset_name not in obj_all:
                    continue
                locations = obj_all[dataset_name].get("location", [])
                if not locations:
                    continue
                frame_idx = max(0, min(interaction_local_frame, len(locations) - 1))
                dataset_active_points.append(np.asarray(locations[frame_idx], dtype=np.float32))
        for active_pt in dataset_active_points:
            points_to_draw.append((active_pt, (0, 200, 200), "given"))
        for pt, color, label in points_to_draw:
            px, pz = world_to_px(
                float(pt[0]),
                float(pt[2]),
                x_min,
                x_max,
                z_min,
                z_max,
                x_res,
                z_res,
            )
            radius = 5 if label.startswith("pelvis") else 4
            draw.ellipse(
                (px - radius, pz + header_h - radius, px + radius, pz + header_h + radius),
                fill=color,
                outline=(255, 255, 255),
            )
            draw.text((px + 6, pz + header_h - 8), label, fill=color)

        header = f"{event_id:02d} action={start}-{end} render={render_start}-{end} {text}"
        if object_name is None:
            detail = f"interaction={interaction_local_frame} query={query_name} obj=none"
        else:
            detail = (
                f"interaction={interaction_local_frame} "
                f"query={query_name} obj={object_name} src={object_source}"
            )
        if label_info:
            label_text = "labels=" + ", ".join(f"{row['label']}:{row['score']:.2f}" for row in label_info[:3])
        else:
            label_text = "labels=none"
        if active_object_names:
            given_text = "given=" + ", ".join(name.split("(")[0] for name in active_object_names[:3])
        else:
            given_text = "given=none"
        wrapped_lines = []
        text_max_width = canvas.width - 24
        for line in [header, detail, label_text, given_text]:
            wrapped_lines.extend(wrap_text_to_width(draw, line, text_max_width))

        draw_text_block(
            draw,
            (10, 10),
            wrapped_lines,
            fill=(220, 20, 20),
            bg_fill=(255, 255, 255),
        )

        png_path = out_dir / f"{event_id:02d}_{start:04d}_{end:04d}.png"
        png_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(str(png_path), format="PNG")

        summary_lines.append(
            {
                "event_id": event_id,
                "render_range": [render_start, end],
                "range": [start, end],
                "text": text,
                "interaction_frame": interaction_local_frame,
                "query_track": query_name,
                "selected_object": object_name,
                "selected_object_source": object_source,
                "active_labels": label_info,
                "given_active_objects": active_object_names,
            }
        )
        prev_end = end

    if args.write_summary:
        summary_path = out_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary_lines, indent=2),
            encoding="utf-8",
        )



if __name__ == "__main__":
    main()
