from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

AREA_COLORS = {
    "seat": np.array([230, 60, 60], dtype=np.uint8),
    "support": np.array([60, 120, 230], dtype=np.uint8),
    "object": np.array([40, 170, 170], dtype=np.uint8),
    "interactable": np.array([240, 170, 40], dtype=np.uint8),
}

AREA_LABELS = {
    "seat": "seat",
    "support": "support",
    "object": "fixed object(hand)",
    "interactable": "interactable(body)",
    "origin": "object origin",
}

AREA_SIZE_BY_TYPE = {
    "seat": (0.4, 0.4),
    "support": (0.2, 0.2),
    "object": (0.2, 0.2),
    "interactable": (0.4, 0.4),
}

LEGEND_SCALE = 2.5

TRUMANS_SEATED_GOAL_TYPES = {"sit", "lie", "slide"}
TRUMANS_SUPPORT_NEEDED_GOAL_TYPES = {"pick_up", "put_down", "lift", "lower", "raise", "rotate"}
TRUMANS_FIXED_INTERACTION_GOAL_TYPES = {
    "open",
    "close",
    "turn",
    "type",
    "wipe",
    "water",
    "shut_down",
    "create",
    "hit",
    "place_hand",
}

LINGO_SEATED_GOAL_TYPES = {"sit", "lie"}
LINGO_SUPPORT_NEEDED_GOAL_TYPES = {"pick_up", "put_down"}
LINGO_FIXED_INTERACTION_GOAL_TYPES = {"type", "wash", "blow_out", "brush", "punch", "kick", "crawl", "rotate"}

SEAT_OBJECT_HINTS = {
    "chair",
    "sofa",
    "couch",
    "bed",
    "bench",
    "stool",
    "toilet",
    "seat",
}

PORTABLE_OBJECT_HINTS = {
    "cup",
    "bottle",
    "book",
    "phone",
    "pen",
    "pencil",
    "remote",
    "plate",
    "bowl",
    "fork",
    "knife",
    "spoon",
    "vase",
    "glass",
    "mug",
    "can",
    "bag",
    "box",
    "laptop",
    "tablet",
    "mouse",
}

SCENELESS_HANDHELD_HINTS = {
    "guitar",
    "violin",
    "cello",
    "ukulele",
    "bass",
    "banjo",
    "flute",
    "saxophone",
    "trumpet",
    "trombone",
    "clarinet",
}

SMPLX_JOINT_INDEX = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_shoulder": 16,
    "right_shoulder": 17,
}

TRUMANS_LEFT_HAND_IDX = 22
TRUMANS_RIGHT_HAND_IDX = 23
LINGO_LEFT_HAND_IDX = 24
LINGO_RIGHT_HAND_IDX = 26

SMPLX_MODEL_CANDIDATES = (
    PROJECT_ROOT / "human_models",
    PROJECT_ROOT / "smpl_models",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build simplified scene affordances for multi-character generation.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--scene-list-file", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "preprocessed")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def to_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def get_scene_paths(dataset: str, scene_id: str, output_root: Path) -> Tuple[Path, Path]:
    scene_root = output_root / dataset / "scenes" / scene_id
    mc_root = output_root / dataset / "multi_character" / scene_id
    return scene_root, mc_root


def resolve_scene_ids(args: argparse.Namespace) -> List[str]:
    if args.scene_id is not None:
        return [args.scene_id]
    if args.scene_list_file is not None:
        return [line.strip() for line in args.scene_list_file.read_text().splitlines() if line.strip()]
    if args.split is not None:
        split_file = args.output_root / args.dataset / f"{args.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    raise ValueError("Provide one of --scene-id, --scene-list-file, or --split.")


def normalize_label(value: Optional[str]) -> str:
    if value is None:
        return ""
    normalized = str(value).lower().replace("_", " ")
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


CANONICAL_NAME_ALIASES = {
    "fridge": "refrigerator",
    "cupboard": "cabinet",
    "couch": "sofa",
    "office chair": "chair",
    "seat": "chair",
    "stool": "chair",
    "water bottle": "bottle",
}


def canonical_name(value: Optional[str], *, seat_mode: bool = False) -> str:
    name = normalize_label(value)
    if not name:
        return "unknown"
    name = CANONICAL_NAME_ALIASES.get(name, name)
    if seat_mode and any(token in name for token in SEAT_OBJECT_HINTS):
        return "seat"
    return name.replace(" ", "_")


def is_seat_like_name(value: Optional[str]) -> bool:
    name = normalize_label(value)
    return any(token in name for token in SEAT_OBJECT_HINTS)


def is_portable_name(value: Optional[str]) -> bool:
    name = normalize_label(value)
    return any(token in name for token in PORTABLE_OBJECT_HINTS)


def is_sceneless_handheld_text(text: Optional[str]) -> bool:
    normalized = normalize_label(text)
    return any(token in normalized for token in SCENELESS_HANDHELD_HINTS)


def classify_goal_bucket(dataset: str, goal_type: str) -> Optional[str]:
    if dataset == "trumans":
        if goal_type in TRUMANS_SEATED_GOAL_TYPES:
            return "seated"
        if goal_type in TRUMANS_SUPPORT_NEEDED_GOAL_TYPES:
            return "support-needed"
        if goal_type in TRUMANS_FIXED_INTERACTION_GOAL_TYPES:
            return "fixed"
        return None
    if goal_type in LINGO_SEATED_GOAL_TYPES:
        return "seated"
    if goal_type in LINGO_SUPPORT_NEEDED_GOAL_TYPES:
        return "support-needed"
    if goal_type in LINGO_FIXED_INTERACTION_GOAL_TYPES:
        return "fixed"
    return None


def infer_target_name_from_text(text: Optional[str], goal_type: str) -> Optional[str]:
    normalized = normalize_label(text)
    if not normalized:
        return None

    patterns = []
    if goal_type == "pick_up":
        patterns.extend(
            [
                r"pick up ([a-z0-9 ]+?) with",
                r"picks up (?:the )?([a-z0-9 ]+?)(?: with|$)",
                r"pick(?:ing)? up (?:the )?([a-z0-9 ]+?)(?: with|$)",
            ]
        )
    elif goal_type == "put_down":
        patterns.extend(
            [
                r"put down ([a-z0-9 ]+?) (?:in|with|on|$)",
                r"puts down (?:the )?([a-z0-9 ]+?)(?: with|$)",
                r"put(?:ting)? down (?:the )?([a-z0-9 ]+?)(?: with|$)",
                r"set down (?:the )?([a-z0-9 ]+?)(?: with|$)",
                r"set (?:the )?([a-z0-9 ]+?) down(?: with|$)",
                r"use .* to set down (?:the )?([a-z0-9 ]+?)(?: with|$)",
            ]
        )
    elif goal_type in {"lift", "lower", "raise", "rotate"}:
        patterns.append(rf"{goal_type} (?:the )?([a-z0-9 ]+?)(?: with|$)")
    elif goal_type in {"open", "close"}:
        patterns.append(rf"{goal_type}(?:s)? (?:the )?([a-z0-9 ]+?)(?: with|$)")
    elif goal_type == "turn":
        patterns.extend(
            [
                r"turn(?:s)? (?:on|off) (?:the )?([a-z0-9 ]+?)(?: with|$)",
                r"turn(?:s)? (?:the )?([a-z0-9 ]+?)(?: with|$)",
            ]
        )
    elif goal_type == "type":
        patterns.extend(
            [
                r"type on ([a-z0-9 ]+?)(?: with| while|$)",
                r"typing on (?:the )?([a-z0-9 ]+?)(?: with|$)",
                r"use (?:the )?([a-z0-9 ]+?)(?: with|$)",
            ]
        )
    elif goal_type == "wash":
        patterns.extend(
            [
                r"wash hands at ([a-z0-9 ]+?)(?: with| while|$)",
                r"take shower in ([a-z0-9 ]+?)(?: with| while|$)",
            ]
        )
    elif goal_type == "blow_out":
        patterns.append(r"blow out ([a-z0-9 ]+?)(?: with|$)")
    elif goal_type == "brush":
        patterns.append(r"brush (?:the )?([a-z0-9 ]+?)(?: with|$)")
    elif goal_type in {"punch", "kick"}:
        patterns.append(rf"{goal_type} ([a-z0-9 ]+?)(?: with|$)")
    elif goal_type == "crawl":
        patterns.append(r"crawl on to ([a-z0-9 ]+?)(?: with|$)")
    elif goal_type == "wipe":
        patterns.append(r"wipe (?:the )?([a-z0-9 ]+?)(?: with|$)")
    elif goal_type == "water":
        patterns.append(r"water (?:the )?([a-z0-9 ]+?)(?: with|$)")
    elif goal_type == "shut_down":
        patterns.append(r"shut down (?:the )?([a-z0-9 ]+?)(?: with|$)")
    elif goal_type == "hit":
        patterns.append(r"hit (?:the )?([a-z0-9 ]+?)(?: with|$)")
    elif goal_type == "place_hand":
        patterns.extend(
            [
                r"place (?:both |the |left |right )?hands? on (?:the )?([a-z0-9 ]+?)(?: with|$)",
                r"put (?:the |left |right )?hand on (?:the )?([a-z0-9 ]+?)(?: with|$)",
            ]
        )

    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.I)
        if match:
            return canonical_name(match.group(1), seat_mode=False)
    return None


def coerce_location3(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1 or arr.shape[0] < 3:
        return None
    arr = arr[:3]
    if not np.isfinite(arr).all():
        return None
    return arr


def resolve_repo_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@lru_cache(maxsize=32)
def load_numpy_mmap(path_str: str) -> np.ndarray:
    return np.load(path_str, mmap_mode="r")


@lru_cache(maxsize=1)
def load_smplx_runtime():
    try:
        import smplx  # type: ignore
        import torch  # type: ignore
    except ImportError:
        return None

    last_error = None
    for model_root in SMPLX_MODEL_CANDIDATES:
        if not model_root.exists():
            continue
        try:
            model = smplx.create(
                str(model_root),
                model_type="smplx",
                gender="NEUTRAL",
                use_pca=False,
                ext="npz",
            )
            model = model.eval()
            return torch, model
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            last_error = exc
            continue
    if last_error is not None:
        raise RuntimeError(f"Failed to load SMPL-X model from {SMPLX_MODEL_CANDIDATES}: {last_error}") from last_error
    return None


def yaw_from_global_orient(global_orient: Optional[np.ndarray]) -> Optional[float]:
    if global_orient is None:
        return None
    try:
        forward = R.from_rotvec(np.asarray(global_orient, dtype=np.float32)).apply(
            np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )
    except ValueError:
        return None
    planar = np.asarray([float(forward[0]), float(forward[2])], dtype=np.float32)
    if float(np.linalg.norm(planar)) <= 1e-8:
        return None
    return float(math.atan2(float(planar[0]), float(planar[1])))


def smplx_facing_yaw(sequence: dict, segment: dict) -> Optional[float]:
    runtime = load_smplx_runtime()
    if runtime is None:
        return None
    torch, smpl_model = runtime

    motion_ref = sequence.get("human_motion_ref", {})
    smpl_ref = motion_ref.get("smplx", {})
    global_orient_path = resolve_repo_path(smpl_ref.get("global_orient_path"))
    body_pose_path = resolve_repo_path(smpl_ref.get("body_pose_path"))
    transl_path = resolve_repo_path(smpl_ref.get("transl_path"))
    if global_orient_path is None or body_pose_path is None or transl_path is None:
        return None

    seq_start = int(motion_ref.get("start", 0))
    interaction_frame = int(segment.get("interaction_frame", 0))
    frame_idx = seq_start + interaction_frame

    global_orient_arr = load_numpy_mmap(str(global_orient_path))
    body_pose_arr = load_numpy_mmap(str(body_pose_path))
    transl_arr = load_numpy_mmap(str(transl_path))
    if not (0 <= frame_idx < len(global_orient_arr) and 0 <= frame_idx < len(body_pose_arr) and 0 <= frame_idx < len(transl_arr)):
        return None

    global_orient = np.array(global_orient_arr[frame_idx], dtype=np.float32, copy=True).reshape(1, -1)
    body_pose = np.array(body_pose_arr[frame_idx], dtype=np.float32, copy=True).reshape(1, -1)
    transl = np.array(transl_arr[frame_idx], dtype=np.float32, copy=True).reshape(1, -1)

    kwargs = {
        "global_orient": torch.from_numpy(global_orient),
        "body_pose": torch.from_numpy(body_pose),
        "transl": torch.from_numpy(transl),
        "return_verts": False,
    }

    left_hand_pose_path = resolve_repo_path(smpl_ref.get("left_hand_pose_path"))
    right_hand_pose_path = resolve_repo_path(smpl_ref.get("right_hand_pose_path"))
    if left_hand_pose_path is not None and left_hand_pose_path.exists():
        left_hand_pose_arr = load_numpy_mmap(str(left_hand_pose_path))
        if 0 <= frame_idx < len(left_hand_pose_arr):
            kwargs["left_hand_pose"] = torch.from_numpy(
                np.array(left_hand_pose_arr[frame_idx], dtype=np.float32, copy=True).reshape(1, -1)
            )
    if right_hand_pose_path is not None and right_hand_pose_path.exists():
        right_hand_pose_arr = load_numpy_mmap(str(right_hand_pose_path))
        if 0 <= frame_idx < len(right_hand_pose_arr):
            kwargs["right_hand_pose"] = torch.from_numpy(
                np.array(right_hand_pose_arr[frame_idx], dtype=np.float32, copy=True).reshape(1, -1)
            )

    with torch.no_grad():
        joints = smpl_model(**kwargs).joints[0].detach().cpu().numpy()

    pelvis = joints[SMPLX_JOINT_INDEX["pelvis"], [0, 2]]
    left_hip = joints[SMPLX_JOINT_INDEX["left_hip"], [0, 2]]
    right_hip = joints[SMPLX_JOINT_INDEX["right_hip"], [0, 2]]
    spine1 = joints[SMPLX_JOINT_INDEX["spine1"], [0, 2]]
    left_shoulder = joints[SMPLX_JOINT_INDEX["left_shoulder"], [0, 2]]
    right_shoulder = joints[SMPLX_JOINT_INDEX["right_shoulder"], [0, 2]]

    lateral = right_hip - left_hip
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm <= 1e-8:
        return yaw_from_global_orient(np.asarray(global_orient[0], dtype=np.float32))
    lateral = lateral / lateral_norm

    candidate_a = np.asarray([-lateral[1], lateral[0]], dtype=np.float32)
    candidate_b = -candidate_a

    torso_hint = (spine1 - pelvis) + (0.5 * (left_shoulder + right_shoulder) - pelvis)
    torso_hint_norm = float(np.linalg.norm(torso_hint))
    if torso_hint_norm <= 1e-6:
        fallback_yaw = yaw_from_global_orient(np.asarray(global_orient[0], dtype=np.float32))
        if fallback_yaw is None:
            return None
        torso_hint = np.asarray([math.sin(fallback_yaw), math.cos(fallback_yaw)], dtype=np.float32)
        torso_hint_norm = float(np.linalg.norm(torso_hint))
    torso_hint = torso_hint / max(torso_hint_norm, 1e-8)

    facing = candidate_a if float(np.dot(candidate_a, torso_hint)) >= float(np.dot(candidate_b, torso_hint)) else candidate_b
    return float(math.atan2(float(facing[0]), float(facing[1])))


def world_to_grid(center_xz: Tuple[float, float], grid_meta: dict, shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    nx, nz = shape
    x_min, x_max = float(grid_meta["x_min"]), float(grid_meta["x_max"])
    z_min, z_max = float(grid_meta["z_min"]), float(grid_meta["z_max"])
    x, z = float(center_xz[0]), float(center_xz[1])
    if not (x_min <= x <= x_max and z_min <= z <= z_max):
        return None
    ix = int((x - x_min) / (x_max - x_min) * nx)
    iz = int((z - z_min) / (z_max - z_min) * nz)
    ix = max(0, min(nx - 1, ix))
    iz = max(0, min(nz - 1, iz))
    return ix, iz


def grid_to_world(ix: int, iz: int, grid_meta: dict, shape: Tuple[int, int]) -> Tuple[float, float]:
    nx, nz = shape
    x_min, x_max = float(grid_meta["x_min"]), float(grid_meta["x_max"])
    z_min, z_max = float(grid_meta["z_min"]), float(grid_meta["z_max"])
    x = x_min + (ix + 0.5) * ((x_max - x_min) / nx)
    z = z_min + (iz + 0.5) * ((z_max - z_min) / nz)
    return float(x), float(z)


def cell_centers(grid_meta: dict, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    nx, nz = shape
    x_min, x_max = float(grid_meta["x_min"]), float(grid_meta["x_max"])
    z_min, z_max = float(grid_meta["z_min"]), float(grid_meta["z_max"])
    xs = x_min + (np.arange(nx) + 0.5) * ((x_max - x_min) / nx)
    zs = z_min + (np.arange(nz) + 0.5) * ((z_max - z_min) / nz)
    return np.meshgrid(xs, zs, indexing="ij")


def rect_mask(
    grid_meta: dict,
    shape: Tuple[int, int],
    center_xz: Tuple[float, float],
    size_xy: Tuple[float, float],
    yaw: float,
) -> np.ndarray:
    xx, zz = cell_centers(grid_meta, shape)
    dx = xx - float(center_xz[0])
    dz = zz - float(center_xz[1])
    cos_y = float(np.cos(yaw))
    sin_y = float(np.sin(yaw))
    local_x = cos_y * dx + sin_y * dz
    local_z = -sin_y * dx + cos_y * dz
    half_x = float(size_xy[0]) * 0.5
    half_z = float(size_xy[1]) * 0.5
    return (np.abs(local_x) <= half_x) & (np.abs(local_z) <= half_z)


def project_mask_to_clearance(mask: np.ndarray, clearance: np.ndarray, min_ratio: float = 0.25) -> Optional[np.ndarray]:
    if not mask.any():
        return None
    projected = mask & clearance.astype(bool)
    if not projected.any():
        return None
    ratio = float(projected.sum()) / float(mask.sum())
    if ratio < min_ratio:
        return None
    return projected


def nearest_valid_center(
    center_xz: Tuple[float, float],
    size_xy: Tuple[float, float],
    yaw: float,
    clearance: np.ndarray,
    grid_meta: dict,
    min_ratio: float = 0.25,
) -> Tuple[Tuple[float, float], np.ndarray]:
    directions = [
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.707, 0.707),
        (0.707, -0.707),
        (-0.707, 0.707),
        (-0.707, -0.707),
    ]
    best_ratio = -1.0
    best_center = center_xz
    best_mask = None
    for radius in (0.0, 0.15, 0.30, 0.45, 0.60):
        for dx, dz in directions:
            candidate_center = (center_xz[0] + radius * dx, center_xz[1] + radius * dz)
            mask = rect_mask(grid_meta, clearance.shape, candidate_center, size_xy, yaw)
            projected = mask & clearance.astype(bool)
            if not mask.any():
                continue
            ratio = float(projected.sum()) / float(mask.sum())
            if ratio > best_ratio:
                best_ratio = ratio
                best_center = candidate_center
                best_mask = projected if projected.any() else None
            if ratio >= min_ratio and projected.any():
                return candidate_center, projected

    if best_mask is not None:
        return best_center, best_mask

    grid = world_to_grid(center_xz, grid_meta, clearance.shape)
    if grid is not None:
        ix, iz = grid
        coords = np.argwhere(clearance)
        if coords.size > 0:
            dists = np.sum((coords - np.asarray([ix, iz])) ** 2, axis=1)
            best_idx = coords[int(np.argmin(dists))]
            fallback_center = grid_to_world(int(best_idx[0]), int(best_idx[1]), grid_meta, clearance.shape)
            fallback_mask = rect_mask(grid_meta, clearance.shape, fallback_center, size_xy, yaw) & clearance
            return fallback_center, fallback_mask

    raise RuntimeError("Failed to project area to clearance.")


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask.astype(np.uint8))


def add_legend(image: Image.Image, *, include_origin: bool) -> Image.Image:
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    entries = [(name, tuple(int(v) for v in AREA_COLORS[name])) for name in AREA_COLORS]
    if include_origin:
        entries.append(("origin", (255, 0, 255)))

    padding = int(10 * LEGEND_SCALE)
    swatch = int(14 * LEGEND_SCALE)
    line_h = int(18 * LEGEND_SCALE)
    text_pad = int(8 * LEGEND_SCALE)
    legend_w = 0
    for name, _ in entries:
        label = AREA_LABELS[name]
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = int((bbox[2] - bbox[0]) * LEGEND_SCALE)
        legend_w = max(legend_w, swatch + text_pad + text_w)
    legend_w += padding * 2
    legend_h = padding * 2 + len(entries) * line_h

    x0, y0 = 12, 12
    x1, y1 = x0 + legend_w, y0 + legend_h
    draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255), outline=(0, 0, 0), width=1)

    cursor_y = y0 + padding
    for name, color in entries:
        draw.rectangle((x0 + padding, cursor_y + 2, x0 + padding + swatch, cursor_y + 2 + swatch), fill=color, outline=(0, 0, 0))
        text_img = Image.new("RGBA", (legend_w, line_h), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((0, 0), AREA_LABELS[name], fill=(0, 0, 0), font=font)
        text_bbox = text_img.getbbox()
        if text_bbox is not None:
            text_img = text_img.crop(text_bbox)
            scaled = text_img.resize(
                (max(1, int(text_img.width * LEGEND_SCALE)), max(1, int(text_img.height * LEGEND_SCALE))),
                resample=getattr(Image, "Resampling", Image).NEAREST,
            )
            image.alpha_composite(scaled, (x0 + padding + swatch + text_pad, cursor_y))
        cursor_y += line_h
    return image.convert("RGB")


def draw_yaw_arrow(
    image: Image.Image,
    *,
    center_xz: Tuple[float, float],
    yaw: float,
    grid_meta: dict,
    shape: Tuple[int, int],
    scale: int,
    mask: Optional[np.ndarray] = None,
) -> None:
    if mask is not None and mask.any():
        coords = np.argwhere(mask)
        row_min, col_min = coords.min(axis=0)
        row_max, col_max = coords.max(axis=0)
        cx = ((row_min + row_max + 1) * scale) / 2.0
        cy = ((col_min + col_max + 1) * scale) / 2.0
    else:
        grid = world_to_grid(center_xz, grid_meta, shape)
        if grid is None:
            return
        ix, iz = grid
        cx = ix * scale + scale / 2.0
        cy = iz * scale + scale / 2.0
    length = max(12, int(5 * scale))
    head = max(4, int(2 * scale))

    dx = math.cos(float(yaw)) * length
    dy = math.sin(float(yaw)) * length
    ex = cx + dx
    ey = cy + dy

    angle = math.atan2(dy, dx)
    left = (ex - head * math.cos(angle - math.pi / 6), ey - head * math.sin(angle - math.pi / 6))
    right = (ex - head * math.cos(angle + math.pi / 6), ey - head * math.sin(angle + math.pi / 6))

    draw = ImageDraw.Draw(image)
    draw.line((cx, cy, ex, ey), fill=(255, 255, 255), width=4)
    draw.line((ex, ey, left[0], left[1]), fill=(255, 255, 255), width=4)
    draw.line((ex, ey, right[0], right[1]), fill=(255, 255, 255), width=4)
    draw.line((cx, cy, ex, ey), fill=(0, 0, 0), width=2)
    draw.line((ex, ey, left[0], left[1]), fill=(0, 0, 0), width=2)
    draw.line((ex, ey, right[0], right[1]), fill=(0, 0, 0), width=2)


def save_vis(base_clearance: np.ndarray, accum: Dict[str, np.ndarray], out_path: Path) -> None:
    base = np.where(base_clearance, 245, 20).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1)
    for area_type, mask in accum.items():
        color = AREA_COLORS[area_type]
        active = mask > 0
        if active.any():
            rgb[active] = (0.6 * rgb[active] + 0.4 * color).astype(np.uint8)
    image = Image.fromarray(np.transpose(rgb, (1, 0, 2)), mode="RGB")
    image = image.resize((image.width * 4, image.height * 4), resample=getattr(Image, "Resampling", Image).NEAREST)
    image = add_legend(image, include_origin=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def save_object_visualizations(
    scene_objects: List[dict],
    area_records: List[dict],
    clearance: np.ndarray,
    grid_meta: dict,
    out_root: Path,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    area_masks: Dict[str, np.ndarray] = {}
    for area in area_records:
        area_masks[area["area_id"]] = np.load(PROJECT_ROOT / area["mask_path"]).astype(bool)

    vis_root = out_root / "objects_vis"
    vis_root.mkdir(parents=True, exist_ok=True)
    base = np.where(clearance, 245, 20).astype(np.uint8)
    scale = 4

    for obj in scene_objects:
        rgb = np.stack([base, base, base], axis=-1)
        object_areas = []
        for area in area_records:
            if area.get("scene_object_id") != obj["scene_object_id"]:
                continue
            object_areas.append(area)
            mask = area_masks[area["area_id"]]
            color = AREA_COLORS[area["area_type"]]
            active = mask > 0
            if active.any():
                rgb[active] = (0.6 * rgb[active] + 0.4 * color).astype(np.uint8)

        loc = coerce_location3(obj.get("initial_location"))
        if loc is not None:
            grid = world_to_grid((float(loc[0]), float(loc[2])), grid_meta, clearance.shape)
            if grid is not None:
                ix, iz = grid
                x0, x1 = max(0, ix - 2), min(clearance.shape[0], ix + 3)
                z0, z1 = max(0, iz - 2), min(clearance.shape[1], iz + 3)
                rgb[x0:x1, z0:z1] = np.array([255, 0, 255], dtype=np.uint8)

        image = Image.fromarray(np.transpose(rgb, (1, 0, 2)), mode="RGB")
        image = image.resize((image.width * scale, image.height * scale), resample=getattr(Image, "Resampling", Image).NEAREST)
        for area in object_areas:
            if area["area_type"] not in {"seat", "interactable"}:
                continue
            draw_yaw_arrow(
                image,
                center_xz=(float(area["center"][0]), float(area["center"][1])),
                yaw=float(area["yaw"]),
                grid_meta=grid_meta,
                shape=clearance.shape,
                scale=scale,
                mask=area_masks[area["area_id"]],
            )
        image = add_legend(image, include_origin=True)
        image.save(vis_root / f"{obj['scene_object_id']}.png")


def save_affordance_visualizations(
    area_records: List[dict],
    clearance: np.ndarray,
    grid_meta: dict,
    out_root: Path,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    vis_root = out_root / "affordance_vis"
    vis_root.mkdir(parents=True, exist_ok=True)
    base = np.where(clearance, 245, 20).astype(np.uint8)
    scale = 4

    for area in area_records:
        rgb = np.stack([base, base, base], axis=-1)
        mask = np.load(PROJECT_ROOT / area["mask_path"]).astype(bool)
        color = AREA_COLORS[area["area_type"]]
        active = mask > 0
        if active.any():
            rgb[active] = (0.6 * rgb[active] + 0.4 * color).astype(np.uint8)

        center = area.get("center")
        if center is not None and len(center) >= 2:
            grid = world_to_grid((float(center[0]), float(center[1])), grid_meta, clearance.shape)
            if grid is not None:
                ix, iz = grid
                x0, x1 = max(0, ix - 2), min(clearance.shape[0], ix + 3)
                z0, z1 = max(0, iz - 2), min(clearance.shape[1], iz + 3)
                rgb[x0:x1, z0:z1] = np.array([255, 0, 255], dtype=np.uint8)

        image = Image.fromarray(np.transpose(rgb, (1, 0, 2)), mode="RGB")
        image = image.resize((image.width * scale, image.height * scale), resample=getattr(Image, "Resampling", Image).NEAREST)
        if area["area_type"] in {"seat", "interactable"}:
            draw_yaw_arrow(
                image,
                center_xz=(float(area["center"][0]), float(area["center"][1])),
                yaw=float(area["yaw"]),
                grid_meta=grid_meta,
                shape=clearance.shape,
                scale=scale,
                mask=mask,
            )
        image = add_legend(image, include_origin=True)
        if area["area_type"] in {"seat", "interactable"}:
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.load_default()
            except OSError:
                font = None
            draw.text((12, image.height - 24), f"yaw={float(area['yaw']):.3f}", fill=(255, 255, 255), stroke_fill=(0, 0, 0), stroke_width=2, font=font)
        seq = area.get("source_sequence_id", "seq")
        seg = int(area.get("source_segment_id", -1))
        file_name = f"{area['area_id']}__{seq}__seg{seg:02d}.png"
        image.save(vis_root / file_name)


def save_grouped_affordance_lists(
    scene_objects: List[dict],
    area_records: List[dict],
    out_root: Path,
) -> None:
    grouped_root = out_root / "grouped_areas"
    grouped_root.mkdir(parents=True, exist_ok=True)

    by_object_and_type: Dict[tuple[str, str], List[List[float]]] = {}
    for area in area_records:
        key = (area["scene_object_id"], area["area_type"])
        by_object_and_type.setdefault(key, []).append([float(area["center"][0]), float(area["center"][1])])

    for obj in scene_objects:
        area_list_paths: Dict[str, str] = {}
        for area_type in AREA_COLORS:
            key = (obj["scene_object_id"], area_type)
            centers = by_object_and_type.get(key)
            if not centers:
                continue
            out_path = grouped_root / f"{obj['scene_object_id']}_{area_type}.npy"
            np.save(out_path, np.asarray(centers, dtype=np.float32))
            area_list_paths[area_type] = to_repo_relative(out_path)
        if area_list_paths:
            obj["area_list_paths"] = area_list_paths


def normalize_goal_type(segment: dict) -> str:
    value = segment.get("goal_type")
    if value is None:
        return ""
    return str(value).strip()


def infer_segment_mode(dataset: str, segment: dict) -> Optional[str]:
    goal_type = normalize_goal_type(segment)
    if not goal_type:
        return None
    bucket = classify_goal_bucket(dataset, goal_type)
    if bucket == "seated":
        return "seat"
    if bucket == "support-needed":
        return "support"
    if bucket == "fixed":
        text_target_name = infer_target_name_from_text(segment.get("text"), goal_type)
        target_name = text_target_name or segment.get("support_object_name") or segment.get("acted_on_object_name")
        if not target_name:
            return None
        if is_portable_name(target_name):
            return None
        if is_sceneless_handheld_text(segment.get("text")):
            return None
        return "object"
    if goal_type == "stand":
        return None
    return None


def get_pelvis_and_hand(dataset: str, sequence: dict, segment: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    goal_pose = segment.get("goal_pose", {})
    motion_ref = sequence.get("human_motion_ref", {})
    joints_path = resolve_repo_path(motion_ref.get("path"))
    pelvis = None
    hand = None
    if joints_path is not None and joints_path.exists():
        frame_idx = int(motion_ref.get("start", 0)) + int(segment.get("interaction_frame", 0))
        joints_arr = load_numpy_mmap(str(joints_path))
        if 0 <= frame_idx < len(joints_arr):
            frame_joints = np.asarray(joints_arr[frame_idx], dtype=np.float32)
            if frame_joints.ndim == 2 and frame_joints.shape[1] >= 3 and frame_joints.shape[0] > 0:
                pelvis = np.asarray(frame_joints[0, :3], dtype=np.float32)
                left_idx = TRUMANS_LEFT_HAND_IDX if dataset == "trumans" else LINGO_LEFT_HAND_IDX
                right_idx = TRUMANS_RIGHT_HAND_IDX if dataset == "trumans" else LINGO_RIGHT_HAND_IDX
                left_hand = (
                    np.asarray(frame_joints[left_idx, :3], dtype=np.float32)
                    if 0 <= left_idx < frame_joints.shape[0]
                    else None
                )
                right_hand = (
                    np.asarray(frame_joints[right_idx, :3], dtype=np.float32)
                    if 0 <= right_idx < frame_joints.shape[0]
                    else None
                )
                active_hand = str(segment.get("active_hand") or "").strip().lower()
                if active_hand == "left":
                    hand = left_hand
                elif active_hand == "right":
                    hand = right_hand
                elif active_hand == "both":
                    if left_hand is not None and right_hand is not None:
                        hand = ((left_hand + right_hand) * 0.5).astype(np.float32)
                    else:
                        hand = left_hand if left_hand is not None else right_hand
                else:
                    if left_hand is not None and right_hand is not None:
                        hand = ((left_hand + right_hand) * 0.5).astype(np.float32)
                    else:
                        hand = left_hand if left_hand is not None else right_hand
    global_orient = coerce_location3(goal_pose.get("global_orient"))
    yaw = smplx_facing_yaw(sequence, segment)
    if yaw is None:
        yaw = yaw_from_global_orient(global_orient)
    if yaw is None and goal_pose.get("yaw") is not None:
        yaw = float(goal_pose.get("yaw") or 0.0)
    if yaw is None:
        yaw = 0.0
    return pelvis, hand, yaw


def target_name_for_segment(segment: dict, mode: str) -> str:
    support_name = segment.get("support_object_name")
    acted_name = segment.get("acted_on_object_name")
    goal_type = normalize_goal_type(segment)
    text_target_name = infer_target_name_from_text(segment.get("text"), goal_type)
    if mode == "seat":
        return canonical_name(support_name or acted_name or text_target_name or "seat", seat_mode=True)
    if mode == "support":
        return canonical_name(acted_name or support_name or text_target_name or "support", seat_mode=False)
    if mode == "object":
        return canonical_name(text_target_name or support_name or acted_name or "object", seat_mode=False)
    raise ValueError(mode)


def segment_seed(dataset: str, sequence: dict, segment: dict) -> Optional[dict]:
    mode = infer_segment_mode(dataset, segment)
    if mode is None:
        return None
    pelvis, hand, yaw = get_pelvis_and_hand(dataset, sequence, segment)
    if mode == "seat":
        if pelvis is None:
            return None
        return {
            "mode": "seat",
            "object_name": "seat",
            "center_xz": (float(pelvis[0]), float(pelvis[2])),
            "yaw": yaw,
        }
    if mode in {"support", "object"}:
        if hand is None or pelvis is None:
            return None
        return {
            "mode": mode,
            "object_name": target_name_for_segment(segment, mode),
            "center_xz": (float(hand[0]), float(hand[2])),
            "body_center_xz": (float(pelvis[0]), float(pelvis[2])),
            "yaw": yaw,
        }
    return None


def cluster_scene_objects(dataset: str, sequence_records: List[dict]) -> Tuple[List[dict], Dict[Tuple[str, str, str, int], str]]:
    seeds: List[Tuple[Tuple[str, str, str, int], dict]] = []
    for sequence in sequence_records:
        for segment in sequence.get("segment_list", []):
            seed = segment_seed(dataset, sequence, segment)
            if seed is None:
                continue
            key = (sequence["sequence_id"], seed["mode"], seed["object_name"], int(segment["segment_id"]))
            seeds.append((key, seed))

    scene_objects: List[dict] = []
    seed_to_scene: Dict[Tuple[str, str, str, int], str] = {}
    counters: Dict[str, int] = {}
    member_anchor_centers_by_idx: List[List[np.ndarray]] = []
    member_compare_centers_by_idx: List[List[np.ndarray]] = []

    for key, seed in seeds:
        base_name = seed["object_name"]
        mode = seed["mode"]
        center = np.asarray(seed["center_xz"], dtype=np.float32)
        compare_center = np.asarray(seed["center_xz"], dtype=np.float32)
        if mode == "seat":
            compare_center = np.asarray(seed.get("body_center_xz", seed["center_xz"]), dtype=np.float32)
        if mode == "seat":
            threshold = 0.75
        elif mode == "support":
            threshold = 0.60
        else:
            threshold = 0.45
        chosen_idx = None
        best_dist = None
        for idx, existing in enumerate(scene_objects):
            if existing["object_name"] != base_name or existing["anchor_kind"] != mode:
                continue
            member_compare_centers = member_compare_centers_by_idx[idx]
            dist = min(float(np.linalg.norm(member_center - compare_center)) for member_center in member_compare_centers)
            if dist <= threshold and (best_dist is None or dist < best_dist):
                chosen_idx = idx
                best_dist = dist

        if chosen_idx is None:
            counters[base_name] = counters.get(base_name, 0) + 1
            scene_object_id = f"{base_name}_{counters[base_name]:02d}"
            scene_objects.append(
                {
                    "scene_object_id": scene_object_id,
                    "object_name": base_name,
                    "anchor_kind": mode,
                    "movable": mode == "support",
                    "initial_location": [float(center[0]), 0.0, float(center[1])],
                    "center_xz": [float(center[0]), float(center[1])],
                    "source_keys": [list(key)],
                }
            )
            member_anchor_centers_by_idx.append([center.copy()])
            member_compare_centers_by_idx.append([compare_center.copy()])
            chosen_idx = len(scene_objects) - 1
        else:
            scene_objects[chosen_idx]["source_keys"].append(list(key))
            member_anchor_centers_by_idx[chosen_idx].append(center.copy())
            member_compare_centers_by_idx[chosen_idx].append(compare_center.copy())
            mean_center = np.mean(np.stack(member_anchor_centers_by_idx[chosen_idx], axis=0), axis=0)
            scene_objects[chosen_idx]["center_xz"] = [float(mean_center[0]), float(mean_center[1])]
            scene_objects[chosen_idx]["initial_location"] = [float(mean_center[0]), 0.0, float(mean_center[1])]

        seed_to_scene[key] = scene_objects[chosen_idx]["scene_object_id"]

    return scene_objects, seed_to_scene


def build_area_records(
    dataset: str,
    sequence_records: List[dict],
    scene_objects: List[dict],
    seed_to_scene: Dict[Tuple[str, str, str, int], str],
    clearance: np.ndarray,
    grid_meta: dict,
    out_root: Path,
) -> Tuple[List[dict], Dict[str, np.ndarray]]:
    object_by_id = {obj["scene_object_id"]: obj for obj in scene_objects}
    area_counts_by_object: Dict[str, int] = {}
    accum = {name: np.zeros_like(clearance, dtype=np.uint16) for name in AREA_COLORS}
    area_records: List[dict] = []

    for sequence in sequence_records:
        for segment in sequence.get("segment_list", []):
            seed = segment_seed(dataset, sequence, segment)
            if seed is None:
                continue

            key = (sequence["sequence_id"], seed["mode"], seed["object_name"], int(segment["segment_id"]))
            scene_object_id = seed_to_scene[key]
            object_name = object_by_id[scene_object_id]["object_name"]
            yaw = float(seed["yaw"])

            candidates = []
            if seed["mode"] == "seat":
                candidates.append(("seat", seed["center_xz"], AREA_SIZE_BY_TYPE["seat"], yaw))
            elif seed["mode"] == "support":
                candidates.append(("interactable", seed["body_center_xz"], AREA_SIZE_BY_TYPE["interactable"], yaw))
                candidates.append(("support", seed["center_xz"], AREA_SIZE_BY_TYPE["support"], yaw))
            elif seed["mode"] == "object":
                candidates.append(("interactable", seed["body_center_xz"], AREA_SIZE_BY_TYPE["interactable"], yaw))
                candidates.append(("object", seed["center_xz"], AREA_SIZE_BY_TYPE["object"], yaw))

            for area_type, base_center, size_xy, area_yaw in candidates:
                center_xz = (float(base_center[0]), float(base_center[1]))
                mask = rect_mask(grid_meta, clearance.shape, center_xz, size_xy, area_yaw)
                if not mask.any():
                    continue
                area_counts_by_object[object_name] = area_counts_by_object.get(object_name, 0) + 1
                area_id = f"{object_name}_{area_counts_by_object[object_name]:04d}"
                mask_path = out_root / "areas" / f"{area_id}.npy"
                save_mask(mask, mask_path)
                accum[area_type][mask] += 1
                area_records.append(
                    {
                        "area_id": area_id,
                        "area_type": area_type,
                        "scene_object_id": scene_object_id,
                        "object_name": object_name,
                        "anchor_kind": object_by_id[scene_object_id]["anchor_kind"],
                        "center": [float(center_xz[0]), float(center_xz[1])],
                        "yaw": area_yaw,
                        "size_xy": [float(size_xy[0]), float(size_xy[1])],
                        "mask_path": to_repo_relative(mask_path),
                        "source_sequence_id": sequence["sequence_id"],
                        "source_segment_id": segment["segment_id"],
                        "source_goal_type": segment.get("goal_type"),
                        "source_text": segment.get("text"),
                    }
                )

    return area_records, accum


def build_scene_affordances(dataset: str, scene_id: str, output_root: Path) -> Path:
    scene_root, mc_root = get_scene_paths(dataset, scene_id, output_root)
    if mc_root.exists():
        shutil.rmtree(mc_root)
    mc_root.mkdir(parents=True, exist_ok=True)
    scene_record = load_json(scene_root / "scene.json")
    sequence_records = []
    for sequence_id in scene_record.get("sequence_ids", []):
        if dataset == "trumans" and "_augment" in sequence_id:
            continue
        seq_path = scene_root / "sequences" / f"{sequence_id}.json"
        if seq_path.exists():
            sequence_records.append(load_json(seq_path))

    clearance = np.load(PROJECT_ROOT / scene_record["clearance_map_npy_path"]).astype(bool)
    scene_objects, seed_to_scene = cluster_scene_objects(dataset, sequence_records)
    area_records, accum = build_area_records(dataset, sequence_records, scene_objects, seed_to_scene, clearance, scene_record["grid_meta"], mc_root)

    npz_path = mc_root / "affordance_maps.npz"
    vis_path = mc_root / "affordance_map.png"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        seat=accum["seat"].astype(np.uint16),
        support=accum["support"].astype(np.uint16),
        object=accum["object"].astype(np.uint16),
        interactable=accum["interactable"].astype(np.uint16),
        combined=(accum["seat"] + accum["support"] + accum["object"] + accum["interactable"]).astype(np.uint16),
    )
    save_vis(clearance, accum, vis_path)
    save_object_visualizations(scene_objects, area_records, clearance, scene_record["grid_meta"], mc_root)
    save_affordance_visualizations(area_records, clearance, scene_record["grid_meta"], mc_root)
    save_grouped_affordance_lists(scene_objects, area_records, mc_root)

    payload = {
        "scene_id": scene_id,
        "dataset": dataset,
        "occupancy_grid_path": scene_record["occupancy_grid_path"],
        "clearance_map_npy_path": scene_record["clearance_map_npy_path"],
        "clearance_map_vis_path": scene_record["clearance_map_vis_path"],
        "grid_meta": scene_record["grid_meta"],
        "object_state_list": scene_objects,
        "affordance_area_list": area_records,
        "affordance_map_npy_path": to_repo_relative(npz_path),
        "affordance_map_vis_path": to_repo_relative(vis_path),
    }
    out_json = mc_root / "scene_affordances.json"
    write_json(out_json, payload)
    return out_json


def _process_scene(payload: Tuple[str, str, Path]) -> str:
    dataset, scene_id, output_root = payload
    out_json = build_scene_affordances(dataset, scene_id, output_root)
    return to_repo_relative(out_json)


def main() -> None:
    args = parse_args()
    scene_ids = resolve_scene_ids(args)
    if args.workers > 1 and len(scene_ids) > 1:
        with mp.Pool(processes=args.workers, maxtasksperchild=1) as pool:
            for out_json in pool.imap_unordered(
                _process_scene,
                [(args.dataset, scene_id, args.output_root) for scene_id in scene_ids],
                chunksize=1,
            ):
                print(out_json)
    else:
        for scene_id in scene_ids:
            out_json = build_scene_affordances(args.dataset, scene_id, args.output_root)
            print(to_repo_relative(out_json))


if __name__ == "__main__":
    main()
