import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "preprocess") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "preprocess"))

try:
    from build_multi_character_affordances import PROJECT_ROOT, load_json  # type: ignore
except ModuleNotFoundError:  # when running from repo root without PYTHONPATH tweaks
    from preprocess.build_multi_character_affordances import PROJECT_ROOT, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize multi-character episodes in top-view with per-character colors.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--scene-id", default=None, help="Optional single scene id.")
    parser.add_argument("--scene-list-file", type=Path, default=None, help="Optional newline separated scene ids.")
    parser.add_argument("--scene-ids", nargs="*", default=None, help="Optional scene list on CLI.")
    parser.add_argument("--episodes-dir", type=Path, default=None, help="Directory containing <scene_id>/episode_0000.json")
    parser.add_argument("--scene-bank-dir", type=Path, default=None, help="Directory containing <scene_id>/scene_static.json")
    parser.add_argument("--output-root", type=Path, default=None, help="Optional explicit output root for vis. default: per-scene/episode folder")
    parser.add_argument("--episode-id", default=None, help="Visualize a single episode file name (e.g. episode_0001.json)")
    parser.add_argument("--draw-conflicts", action="store_true", help="Draw conflict points from conflict_records.")
    parser.add_argument("--scale", type=int, default=4, help="Pixel upscaling factor for top-view map.")
    parser.add_argument("--line-width", type=int, default=4, help="Body path line width.")
    parser.add_argument("--point-radius", type=int, default=6, help="Body keypoint radius.")
    parser.add_argument("--hand-point-radius", type=int, default=4, help="Hand keypoint radius.")
    parser.add_argument("--top-margin", type=int, default=120, help="Header margin for text overlay.")
    parser.add_argument("--font-size", type=int, default=24, help="Font size for episode labels.")
    parser.add_argument("--label-overlap-avoidance", action="store_true", default=True, help="Skip/relocate overlapping goal/conflict labels.")
    return parser.parse_args()


COLOR_PALETTE = [
    (220, 38, 38),
    (16, 185, 129),
    (59, 130, 246),
    (245, 158, 11),
    (168, 85, 247),
    (14, 165, 233),
    (234, 88, 12),
    (236, 72, 153),
    (79, 70, 229),
    (217, 119, 6),
    (6, 182, 212),
    (249, 115, 22),
]


def collect_scene_ids(args: argparse.Namespace) -> list[str]:
    if args.scene_id is not None:
        return [args.scene_id]
    if args.scene_ids:
        return [s.strip() for s in args.scene_ids if s.strip()]
    if args.scene_list_file is not None:
        return [line.strip() for line in args.scene_list_file.read_text().splitlines() if line.strip()]
    if args.episodes_dir is None:
        return []
    return sorted([p.name for p in args.episodes_dir.iterdir() if p.is_dir()])


def to_point3(value: object) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return None
    return arr


def world_to_pixel(x: float, z: float, x_min: float, x_max: float, z_min: float, z_max: float, x_res: int, z_res: int) -> tuple[int, int]:
    px = int(round((x - x_min) / (x_max - x_min) * (x_res - 1)))
    pz = int(round((z - z_min) / (z_max - z_min) * (z_res - 1)))
    return max(0, min(x_res - 1, px)), max(0, min(z_res - 1, pz))


def make_color(idx: int) -> tuple[int, int, int]:
    if idx < len(COLOR_PALETTE):
        return COLOR_PALETTE[idx]
    h = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.96)
    return int(r * 255), int(g * 255), int(b * 255)


def fit_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    fill: tuple[int, int, int],
    font: ImageFont.ImageFont,
    max_x: int,
    left_margin: int = 8,
) -> tuple[int, int]:
    if not text:
        return x, y

    ellipsis = "..."
    tx, ty = x, y
    full_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = full_bbox[2] - full_bbox[0]
    available = max(0, max_x - left_margin - x)

    if text_w <= available and text_w > 0:
        draw.text((tx, ty), text, fill=fill, font=font)
        return tx, ty

    # Try shifting left if near right edge first.
    shifted_x = max(left_margin, max_x - left_margin - text_w)
    tx = min(tx, shifted_x)
    if tx < x:
        tx = max(left_margin, tx)
        available = max(0, max_x - left_margin - tx)

    if text_w <= available:
        draw.text((tx, ty), text, fill=fill, font=font)
        return tx, ty

    # Fallback: truncate with ellipsis.
    truncated = text
    ell_bbox = draw.textbbox((0, 0), ellipsis, font=font)
    ell_w = ell_bbox[2] - ell_bbox[0]
    if available <= ell_w:
        truncated = ellipsis
    else:
        while truncated and (draw.textbbox((0, 0), truncated + ellipsis, font=font)[2] - draw.textbbox((0, 0), truncated + ellipsis, font=font)[0]) > available:
            truncated = truncated[:-1]
        if truncated != text:
            truncated = f"{truncated}{ellipsis}" if truncated else ellipsis
    draw.text((tx, ty), truncated, fill=fill, font=font)
    return tx, ty


def measure_text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    if not text:
        return 0
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def split_token_to_width(
    draw: ImageDraw.ImageDraw,
    token: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    if not token:
        return [""]
    if max_width <= 0:
        return [token]

    chunks: list[str] = []
    current = ""
    for ch in token:
        candidate = ch if not current else current + ch
        if current and measure_text_width(draw, candidate, font) > max_width:
            chunks.append(current)
            current = ch
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [token]


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    if not text:
        return [""]
    if max_width <= 0:
        return [text]

    wrapped: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split(" ")
        current = ""
        for word in words:
            if not word:
                continue
            candidate = word if not current else f"{current} {word}"
            if measure_text_width(draw, candidate, font) <= max_width:
                current = candidate
                continue
            if current:
                wrapped.append(current)
                current = ""
            if measure_text_width(draw, word, font) <= max_width:
                current = word
                continue
            pieces = split_token_to_width(draw, word, font, max_width)
            wrapped.extend(pieces[:-1])
            current = pieces[-1]
        if current:
            wrapped.append(current)
        if not paragraph:
            wrapped.append("")
    return wrapped or [text]


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    fill: tuple[int, int, int],
    font: ImageFont.ImageFont,
    max_width: int,
    line_h: int,
) -> int:
    for line in wrap_text_to_width(draw, text, font, max_width):
        draw.text((x, y), line, fill=fill, font=font)
        y += line_h
    return y


def truncate_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if not text or max_width <= 0:
        return ""
    if measure_text_width(draw, text, font) <= max_width:
        return text
    ellipsis = "..."
    ell_w = measure_text_width(draw, ellipsis, font)
    if ell_w >= max_width:
        return ""
    truncated = text
    while truncated:
        candidate = f"{truncated}{ellipsis}"
        if measure_text_width(draw, candidate, font) <= max_width:
            return candidate
        truncated = truncated[:-1]
    return ""


def rect_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    left1, top1, right1, bottom1 = a
    left2, top2, right2, bottom2 = b
    return not (right1 < left2 or right2 < left1 or bottom1 < top2 or bottom2 < top1)


def place_text_no_overlap(
    draw: ImageDraw.ImageDraw,
    text: str,
    anchor: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    canvas_w: int,
    canvas_h: int,
    occupied: list[tuple[int, int, int, int]],
    top_margin: int,
    left_margin: int = 8,
) -> bool:
    if not text:
        return False

    ax, ay = anchor
    max_width = max(40, canvas_w - left_margin - 4)
    label = truncate_text_to_width(draw, text, font, max_width)
    if not label:
        return False

    l, t, r, b = draw.textbbox((0, 0), label, font=font)
    text_w = max(1, r - l)
    text_h = max(1, b - t)
    offsets = [
        (6, 6),
        (6, -text_h - 2),
        (6, text_h + 2),
        (-text_w - 6, 6),
        (-text_w - 6, -text_h - 2),
        (-text_w - 6, text_h + 2),
        (6, -text_h * 2 - 6),
        (-text_w - 6, -text_h * 2 - 6),
    ]

    for dx, dy in offsets:
        x = ax + dx
        y = ay + dy
        if x < left_margin:
            x = left_margin
        if y < top_margin:
            y = top_margin + 1
        if x + text_w > canvas_w - 2:
            x = max(left_margin, canvas_w - text_w - 2)
        if y + text_h > canvas_h - 2:
            y = canvas_h - text_h - 2
        if y < top_margin + 1:
            continue
        box = (x - 1, y - 1, x + text_w + 1, y + text_h + 1)
        if any(rect_intersects(box, occluded) for occluded in occupied):
            continue
        draw.text((x, y), label, fill=fill, font=font)
        occupied.append(box)
        return True
    return False


def load_font(font_size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, font_size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except (OSError, TypeError):
        return ImageFont.load_default()


def load_scene_meta(scene_bank_dir: Path, dataset: str, scene_id: str) -> dict:
    path = scene_bank_dir / scene_id / "scene_static.json"
    if not path.exists():
        raise FileNotFoundError(f"Scene static not found: {path}")
    return load_json(path)


def resolve_clearance_path(scene_payload: dict) -> Path:
    candidates = [
        scene_payload.get("clearance_map_npy_path"),
        scene_payload.get("clearance_map_path"),
        scene_payload.get("clearance_map"),
    ]
    for value in candidates:
        if not value:
            continue
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No clearance map found in scene_static.")


def load_goal_samples(episode: dict) -> list[dict]:
    chars = episode.get("character_assignments", [])
    outputs = []
    for cidx, char in enumerate(chars):
        color = make_color(cidx)
        goal_sequence = char.get("goal_sequence", [])
        parsed = []
        for step_idx, goal in enumerate(goal_sequence):
            body = to_point3(goal.get("body_goal"))
            hand = to_point3(goal.get("hand_goal"))
            source_segment = goal.get("source_segment", {})
            interaction_frame = source_segment.get("interaction_frame")
            segment_id = source_segment.get("segment_id", step_idx)
            try:
                frame_token = int(interaction_frame)
            except (TypeError, ValueError):
                frame_token = step_idx
            parsed.append(
                {
                    "step_idx": step_idx,
                    "frame_token": frame_token,
                    "segment_id": int(segment_id),
                    "body": body,
                    "hand": hand,
                    "goal_type": goal.get("goal_type"),
                    "goal_category": goal.get("goal_category"),
                    "text": source_segment.get("text"),
                    "source": goal,
                }
            )
        parsed.sort(key=lambda row: (row["frame_token"], row["segment_id"], row["step_idx"]))
        outputs.append(
            {
                "character_id": char.get("character_id", f"char_{cidx:02d}"),
                "color": color,
                "steps": parsed,
            }
        )
    return outputs


def draw_polyline(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]], color: tuple[int, int, int], width: int = 2):
    if len(pts) < 2:
        return
    for p0, p1 in zip(pts[:-1], pts[1:]):
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=width)


def draw_episode(
    bg_rgb: np.ndarray,
    meta: dict,
    episode: dict,
    args: argparse.Namespace,
    out_path: Path,
):
    x_min = float(meta["x_min"])
    x_max = float(meta["x_max"])
    z_min = float(meta["z_min"])
    z_max = float(meta["z_max"])
    x_res = int(meta.get("x_res", bg_rgb.shape[1]))
    z_res = int(meta.get("z_res", bg_rgb.shape[0]))

    map_h, map_w = bg_rgb.shape[:2]
    assert map_w == x_res and map_h == z_res, f"background map shape mismatch: {(map_w, map_h)} vs {(x_res, z_res)}"

    scale = max(1, int(args.scale))
    canvas_w = map_w * scale
    font = load_font(max(10, int(args.font_size)))
    bbox = font.getbbox("Ag")
    line_h = max(18, (bbox[3] - bbox[1]) + 6)

    char_tracks = load_goal_samples(episode)
    body_width = max(1, int(args.line_width))
    legend_items = []
    for char in char_tracks:
        steps = char["steps"]
        if steps:
            first_step = steps[0]
            if first_step["body"] is not None:
                last_step = steps[-1]
                legend_items.append(
                    (
                        char["character_id"],
                        char["color"],
                        first_step["goal_type"],
                        last_step["goal_type"],
                    )
                )

    header_lines = [
        f"episode={episode.get('episode_id', 'episode')}",
        f"scene={episode.get('scene_id')}",
        f"scenario={episode.get('scenario_type', 'unknown')} | conflict={episode.get('conflict_type', 'n/a')}",
        f"characters={episode.get('num_characters', len(char_tracks))} | sequence_count={len(episode.get('source_sequences', []))}",
    ]
    legend_x = 10
    legend_label_x = legend_x + 22
    tmp_draw = ImageDraw.Draw(Image.new("RGB", (canvas_w, 4), (0, 0, 0)))
    text_max_w = max(40, canvas_w - legend_x - 10)
    legend_text_max_w = max(40, canvas_w - legend_label_x - 10)
    required_header_h = 10
    for line in header_lines:
        required_header_h += len(wrap_text_to_width(tmp_draw, line, font, text_max_w)) * line_h
    if legend_items:
        required_header_h += 4
        required_header_h += line_h
        for name, _color, first_goal, last_goal in legend_items:
            label = f"{name}: {first_goal or 'n/a'} -> {last_goal or 'n/a'}"
            required_header_h += len(wrap_text_to_width(tmp_draw, label, font, legend_text_max_w)) * line_h
    required_header_h += 10

    top_margin = max(int(args.top_margin), required_header_h)
    canvas_h = map_h * scale + top_margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (12, 12, 12))
    bg = Image.fromarray(bg_rgb, mode="RGB").resize((canvas_w, map_h * scale), resample=Image.Resampling.NEAREST)
    canvas.paste(bg, (0, top_margin))
    draw = ImageDraw.Draw(canvas)
    occupied_labels: list[tuple[int, int, int, int]] = []

    for char in char_tracks:
        c = char["color"]
        body_pts = []
        for step in char["steps"]:
            if step["body"] is not None:
                px, pz = world_to_pixel(float(step["body"][0]), float(step["body"][2]), x_min, x_max, z_min, z_max, x_res, z_res)
                body_pts.append((px * scale, pz * scale + top_margin))

        draw_polyline(draw, body_pts, c, width=max(1, body_width * scale))
        for step in char["steps"]:
            if step["body"] is not None:
                px, pz = world_to_pixel(float(step["body"][0]), float(step["body"][2]), x_min, x_max, z_min, z_max, x_res, z_res)
                px = px * scale
                pz = pz * scale + top_margin
                r = max(1, args.point_radius * scale // 2)
                draw.ellipse((px - r, pz - r, px + r, pz + r), outline=(255, 255, 255), fill=c, width=max(1, scale))
                if args.label_overlap_avoidance:
                    occupied_labels.append((px - r - 2, pz - r - 2, px + r + 2, pz + r + 2))
                goal_name = step.get("goal_type") or step.get("goal_category") or "goal"
                txt = f"{char['character_id']}:{goal_name}"
                if args.label_overlap_avoidance:
                    place_text_no_overlap(
                        draw=draw,
                        text=txt,
                        anchor=(px, pz),
                        font=font,
                        fill=c,
                        canvas_w=canvas_w,
                        canvas_h=canvas_h,
                        occupied=occupied_labels,
                        top_margin=top_margin,
                        left_margin=10,
                    )
                else:
                    fit_text_to_width(draw, txt, px + 5, pz + 5, fill=c, font=font, max_x=canvas_w, left_margin=10)
            if step["hand"] is not None and args.hand_point_radius > 0:
                px, pz = world_to_pixel(float(step["hand"][0]), float(step["hand"][2]), x_min, x_max, z_min, z_max, x_res, z_res)
                px = px * scale
                pz = pz * scale + top_margin
                hr = max(1, args.hand_point_radius * scale // 2)
                draw.rectangle((px - hr, pz - hr, px + hr, pz + hr), fill=(245, 245, 245), outline=c, width=max(1, scale))
                if args.label_overlap_avoidance:
                    occupied_labels.append((px - hr - 1, pz - hr - 1, px + hr + 1, pz + hr + 1))

    # conflict points
    if args.draw_conflicts:
        for idx, cr in enumerate(episode.get("conflict_records", [])):
            loc = cr.get("location")
            if not isinstance(loc, list) or len(loc) < 2:
                continue
            px, pz = world_to_pixel(float(loc[0]), float(loc[1]), x_min, x_max, z_min, z_max, x_res, z_res)
            px = px * scale
            pz = pz * scale + top_margin
            r = max(4, args.point_radius * scale // 2)
            color = (255, 70, 70)
            draw.ellipse((px - r, pz - r, px + r, pz + r), fill=None, outline=color, width=max(1, max(2, scale)))
            conflict_text = f"{idx+1}:{cr.get('conflict_type', 'conflict')}"
            if args.label_overlap_avoidance:
                place_text_no_overlap(
                    draw=draw,
                    text=conflict_text,
                    anchor=(px, pz),
                    font=font,
                    fill=color,
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                    occupied=occupied_labels,
                    top_margin=top_margin,
                    left_margin=10,
                )
            else:
                fit_text_to_width(
                    draw,
                    conflict_text,
                    px + r + 1,
                    pz - r,
                    fill=color,
                    font=font,
                    max_x=canvas_w,
                    left_margin=10,
                )

    legend_y = 10
    for line in header_lines:
        legend_y = draw_wrapped_text(
            draw,
            line,
            legend_x,
            legend_y,
            fill=(235, 235, 235),
            font=font,
            max_width=text_max_w,
            line_h=line_h,
        )
    if legend_items:
        legend_y += 4
        draw.text((legend_x, legend_y), "Legend:", fill=(235, 235, 235), font=font)
        legend_y += line_h
        for name, c, first_goal, last_goal in legend_items:
            draw.rectangle((legend_x, legend_y, legend_x + 16, legend_y + 12), fill=c)
            label = f"{name}: {first_goal or 'n/a'} -> {last_goal or 'n/a'}"
            label_y = legend_y - 1
            legend_y = draw_wrapped_text(
                draw,
                label,
                legend_label_x,
                label_y,
                fill=c,
                font=font,
                max_width=legend_text_max_w,
                line_h=line_h,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")


def iter_episode_files(episodes_dir: Path, episode_id: Optional[str], scene_id: str) -> Iterable[Path]:
    scene_dir = episodes_dir / scene_id
    if not scene_dir.exists():
        return []
    if episode_id is not None:
        p = scene_dir / episode_id
        return [p] if p.exists() else []
    return sorted(scene_dir.glob("episode_*.json"))


def iter_episodes_to_render(
    episodes_dir: Path,
    scene_ids: list[str],
    episode_id: Optional[str],
) -> list[tuple[str, Path]]:
    outputs = []
    for scene_id in scene_ids:
        for ep in iter_episode_files(episodes_dir, episode_id, scene_id):
            if ep.suffix.lower() != ".json":
                continue
            outputs.append((scene_id, ep))
    return outputs


def main() -> None:
    args = parse_args()
    episodes_root = args.episodes_dir or (PROJECT_ROOT / "data" / "preprocessed" / args.dataset / "episodes_v3")
    scene_bank_root = args.scene_bank_dir or (PROJECT_ROOT / "data" / "preprocessed" / args.dataset / "episode_bank_v3")
    scene_ids = collect_scene_ids(args)
    if not scene_ids:
        raise ValueError("No scene ids provided and no auto-discoverable scenes found.")

    total = 0
    for scene_id, episode_path in iter_episodes_to_render(episodes_root, scene_ids, args.episode_id):
        episode = load_json(episode_path)
        scene_payload = load_scene_meta(scene_bank_root, args.dataset, episode.get("scene_id", scene_id))

        clearance_path = resolve_clearance_path(scene_payload)
        clearance = np.load(clearance_path, allow_pickle=True)
        base = np.where(clearance, 245, 20).astype(np.uint8)
        bg_rgb = np.stack([base, base, base], axis=-1).transpose((1, 0, 2))

        if args.output_root is not None:
            out_dir = args.output_root
            out_name = f"{episode.get('episode_id', episode_path.stem)}_top_view.png"
            out_path = out_dir / scene_id / out_name
        else:
            out_dir = episode_path.parent / "vis"
            out_name = f"{episode_path.stem}_top_view.png"
            out_path = out_dir / out_name

        draw_episode(bg_rgb, scene_payload["grid_meta"], episode, args, out_path)
        total += 1

    print(f"wrote {total} episode visualizations")


if __name__ == "__main__":
    main()
