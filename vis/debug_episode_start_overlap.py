from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "preprocess") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "preprocess"))

from preprocess.build_multi_character_affordances import PROJECT_ROOT, load_json
from preprocess.sample_multi_character_episodes import build_start_overlap_ellipse, start_forward_xz


COLOR_PALETTE = [
    (220, 38, 38),
    (16, 185, 129),
    (59, 130, 246),
    (245, 158, 11),
    (168, 85, 247),
    (14, 165, 233),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw start-frame overlap ellipses for a multi-character episode.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--episode-json", type=Path, required=True)
    parser.add_argument("--scene-bank-dir", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--top-margin", type=int, default=90)
    parser.add_argument("--font-size", type=int, default=20)
    parser.add_argument("--arrow-scale", type=float, default=0.24, help="Arrow length in meters.")
    return parser.parse_args()


def load_font(font_size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def resolve_clearance_path(scene_payload: dict) -> Path:
    for key in ("clearance_map_npy_path", "clearance_map_path", "clearance_map"):
        value = scene_payload.get(key)
        if not value:
            continue
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No clearance map found in scene_static.")


def world_to_pixel(
    x: float,
    z: float,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    x_res: int,
    z_res: int,
) -> tuple[int, int]:
    px = int(round((x - x_min) / (x_max - x_min) * (x_res - 1)))
    pz = int(round((z - z_min) / (z_max - z_min) * (z_res - 1)))
    return max(0, min(x_res - 1, px)), max(0, min(z_res - 1, pz))


def color_with_alpha(rgb: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
    return int(rgb[0]), int(rgb[1]), int(rgb[2]), int(alpha)


def main() -> None:
    args = parse_args()
    episode_path = args.episode_json
    if not episode_path.is_absolute():
        episode_path = ROOT_DIR / episode_path
    if not episode_path.exists():
        raise FileNotFoundError(episode_path)

    episode = load_json(episode_path)
    scene_id = str(episode.get("scene_id"))
    scene_bank_dir = args.scene_bank_dir or (PROJECT_ROOT / "data" / "preprocessed" / args.dataset / "episode_bank_v2")
    scene_static_path = scene_bank_dir / scene_id / "scene_static.json"
    if not scene_static_path.exists():
        raise FileNotFoundError(scene_static_path)
    scene_payload = load_json(scene_static_path)

    clearance = np.load(resolve_clearance_path(scene_payload), allow_pickle=True).astype(bool)
    base = np.where(clearance, 242, 40).astype(np.uint8)
    bg_rgb = np.stack([base, base, base], axis=-1).transpose((1, 0, 2))

    meta = scene_payload["grid_meta"]
    x_min = float(meta["x_min"])
    x_max = float(meta["x_max"])
    z_min = float(meta["z_min"])
    z_max = float(meta["z_max"])
    x_res = int(meta.get("x_res", bg_rgb.shape[1]))
    z_res = int(meta.get("z_res", bg_rgb.shape[0]))

    scale = max(1, int(args.scale))
    top_margin = max(0, int(args.top_margin))
    canvas_w = x_res * scale
    canvas_h = z_res * scale + top_margin

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (18, 18, 18, 255))
    bg = Image.fromarray(bg_rgb, mode="RGB").resize((canvas_w, z_res * scale), resample=Image.Resampling.NEAREST).convert("RGBA")
    canvas.paste(bg, (0, top_margin))
    draw = ImageDraw.Draw(canvas, mode="RGBA")
    font = load_font(max(10, int(args.font_size)))

    shape = episode.get("start_overlap_shape", {})
    lateral_diameter = float(shape.get("lateral_diameter_m", 0.50))
    fore_aft_diameter = float(shape.get("fore_aft_diameter_m", 0.30))

    header_lines = [
        f"episode={episode.get('episode_id', episode_path.stem)}",
        f"scene={scene_id}",
        f"start overlap ellipse: lateral={lateral_diameter:.2f}m, fore-aft={fore_aft_diameter:.2f}m",
    ]
    y = 10
    for line in header_lines:
        draw.text((10, y), line, fill=(236, 236, 236, 255), font=font)
        y += max(20, int(args.font_size) + 6)

    for idx, assignment in enumerate(episode.get("character_assignments", [])):
        start_state = assignment.get("start_state", {})
        body_translation = np.asarray(start_state.get("body_translation"), dtype=np.float32)
        if body_translation.shape != (3,) or not np.isfinite(body_translation).all():
            continue

        center_xz = np.asarray([float(body_translation[0]), float(body_translation[2])], dtype=np.float32)
        poly = build_start_overlap_ellipse(
            center_xz=center_xz,
            global_orient_rotvec=start_state.get("global_orient_rotvec"),
            lateral_diameter_m=lateral_diameter,
            fore_aft_diameter_m=fore_aft_diameter,
        )
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        poly_px: list[tuple[int, int]] = []
        for x, z in poly:
            px, pz = world_to_pixel(float(x), float(z), x_min, x_max, z_min, z_max, x_res, z_res)
            poly_px.append((px * scale, pz * scale + top_margin))

        draw.polygon(poly_px, fill=color_with_alpha(color, 56), outline=color_with_alpha(color, 255))

        px, pz = world_to_pixel(float(center_xz[0]), float(center_xz[1]), x_min, x_max, z_min, z_max, x_res, z_res)
        px = px * scale
        pz = pz * scale + top_margin
        center_r = max(3, scale * 2)
        draw.ellipse((px - center_r, pz - center_r, px + center_r, pz + center_r), fill=color_with_alpha(color, 255))

        forward = start_forward_xz(start_state.get("global_orient_rotvec"))
        arrow_tip = center_xz + forward * float(args.arrow_scale)
        ax, az = world_to_pixel(float(arrow_tip[0]), float(arrow_tip[1]), x_min, x_max, z_min, z_max, x_res, z_res)
        ax = ax * scale
        az = az * scale + top_margin
        draw.line((px, pz, ax, az), fill=color_with_alpha(color, 255), width=max(2, scale))
        head = np.asarray([ax - px, az - pz], dtype=np.float32)
        head_norm = float(np.linalg.norm(head))
        if head_norm > 1e-6:
            head /= head_norm
            left = np.asarray([-head[1], head[0]], dtype=np.float32)
            head_len = float(max(6, scale * 3))
            head_width = float(max(4, scale * 2))
            tip = np.asarray([ax, az], dtype=np.float32)
            p1 = tip - head * head_len + left * head_width
            p2 = tip - head * head_len - left * head_width
            draw.polygon(
                [(int(tip[0]), int(tip[1])), (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))],
                fill=color_with_alpha(color, 255),
            )

        label = str(assignment.get("character_id", f"char_{idx:02d}"))
        draw.text((px + 8, pz - 20), label, fill=color_with_alpha(color, 255), font=font)

    output_path = args.output_path
    if output_path is None:
        output_path = episode_path.parent / "vis" / f"{episode_path.stem}_start_overlap_debug.png"
    elif not output_path.is_absolute():
        output_path = ROOT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
