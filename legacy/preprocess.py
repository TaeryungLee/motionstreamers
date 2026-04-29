from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
TRUMANS_SCENE_GRID = np.array([-3.0, 0.0, -4.0, 3.0, 2.0, 4.0, 300, 100, 400], dtype=np.float32)
LINGO_TRAIN_SCENE_GRID = np.array([-3.0, 0.0, -4.0, 3.0, 2.0, 4.0, 300, 100, 400], dtype=np.float32)
LINGO_TEST_SCENE_GRID = np.array([-4.0, 0.0, -6.0, 4.0, 2.0, 6.0, 400, 100, 600], dtype=np.float32)
TRUMANS_LEFT_HAND_IDX = 22
TRUMANS_RIGHT_HAND_IDX = 23
LINGO_LEFT_HAND_IDX = 24
LINGO_RIGHT_HAND_IDX = 26


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_batch(batch: Any) -> str:
    import torch

    if torch.is_tensor(batch):
        return f"Tensor(shape={tuple(batch.shape)}, dtype={batch.dtype})"
    if isinstance(batch, (list, tuple)):
        parts = [summarize_batch(item) for item in batch]
        return f"{type(batch).__name__}({', '.join(parts)})"
    if isinstance(batch, dict):
        parts = [f"{key}={summarize_batch(value)}" for key, value in batch.items()]
        return f"dict({', '.join(parts)})"
    return repr(type(batch).__name__)


def iterate_loader(name: str, loader: Iterable, max_batches: int) -> None:
    print(f"[{name}] iterating loader", flush=True)
    for batch_idx, batch in enumerate(loader):
        print(f"[{name}] batch {batch_idx}: {summarize_batch(batch)}", flush=True)
        if batch_idx + 1 >= max_batches:
            break


def to_serializable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")


def to_repo_relative(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        try:
            return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            return str(path)


def save_mask(mask: np.ndarray, path: Path, scale: int = 4) -> None:
    img = Image.fromarray((mask.astype(np.uint8) * 255).T, mode="L")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), resample=Image.NEAREST)
    img.save(path)


def scene_output_dir(scene_id: str, args) -> Path:
    return args.output_dir / "preprocess" / args.dataset / "scenes" / scene_id


def scene_sequence_dir(scene_id: str, args) -> Path:
    return scene_output_dir(scene_id, args) / "sequences"


def scene_plot_dir(scene_id: str, sequence_id: str, args) -> Path:
    return scene_output_dir(scene_id, args) / "plots" / f"trumans_sequence_actions_{sequence_id}"


def cache_scene_dir(scene_id: str, args) -> Path:
    return args.output_dir / "cache" / args.dataset / scene_id


def compute_walkable_map(scene_occ: np.ndarray, body_height_voxels: int, free_threshold: float) -> dict:
    floor = scene_occ[:, 0, :]
    body_band = scene_occ[:, 1 : body_height_voxels + 1, :]
    free_ratio = 1.0 - body_band.mean(axis=1)
    walkable = floor & (free_ratio >= free_threshold)
    return {
        "walkable": walkable,
        "floor_support": floor,
        "free_ratio": free_ratio,
        "body_height_voxels": body_height_voxels,
        "free_threshold": free_threshold,
        "walkable_ratio": float(walkable.mean()),
        "floor_support_ratio": float(floor.mean()),
    }


def save_walkable_visualizations(
    output_dir: Path,
    prefix: str,
    scene_occ: np.ndarray,
    walkable_info: dict,
    scale: int = 4,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    walkable = walkable_info["walkable"]
    body_height_voxels = walkable_info["body_height_voxels"]
    free_threshold = walkable_info["free_threshold"]
    height_tag = int(round(body_height_voxels * 2))
    thr_tag = int(round(free_threshold * 100))

    walkable_png = output_dir / f"{prefix}_walkable_h{height_tag}_thr{thr_tag:03d}.png"

    save_mask(walkable, walkable_png, scale=scale)

    return {
        "walkable_png": walkable_png,
    }


def parse_hand_hint(text: str) -> str | None:
    lower = text.lower()
    if lower.startswith("left hand") or "with the left hand" in lower or "using the left hand" in lower:
        return "left"
    if lower.startswith("right hand") or "with the right hand" in lower or "using the right hand" in lower:
        return "right"
    if "both hands" in lower:
        return "both"
    return None


def trumans_base_sequence_id(sequence_id: str) -> str:
    return sequence_id.split("_augment", 1)[0]


def blender_world_to_motion(location: list[float] | np.ndarray) -> np.ndarray:
    loc = np.asarray(location, dtype=np.float32)
    return np.array([loc[0], loc[2], -loc[1]], dtype=np.float32)


def nearest_named_objects(object_pose_dict: dict, frame_idx_local: int, query_xyz: np.ndarray, topk: int = 5) -> list[dict]:
    rows = []
    for name, entry in object_pose_dict.items():
        loc = np.asarray(entry["location"][frame_idx_local], dtype=np.float32)
        rot = None
        if "rotation" in entry and frame_idx_local < len(entry["rotation"]):
            rot = np.asarray(entry["rotation"][frame_idx_local], dtype=np.float32)
        dist = float(np.linalg.norm(query_xyz - loc))
        rows.append({"name": name, "distance": dist, "location": loc, "rotation": rot})
    rows.sort(key=lambda row: row["distance"])
    return rows[:topk]


def point_to_aabb_distance(point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    lo = np.minimum(bbox_min, bbox_max)
    hi = np.maximum(bbox_min, bbox_max)
    clamped = np.minimum(np.maximum(point, lo), hi)
    return float(np.linalg.norm(point - clamped))


def resolve_lingo_dataset_root(args) -> Path:
    candidates = []
    if args.lingo_root:
        candidates.append(Path(args.lingo_root))
    candidates.extend(
        [
            PROJECT_ROOT / "data" / "raw" / "lingo" / "dataset",
            PROJECT_ROOT / "lingo-release" / "dataset",
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find an extracted LINGO dataset directory. "
        "Expected one of: "
        + ", ".join(str(path) for path in candidates)
        + ". Current data/raw/lingo only contains zip files, so extract dataset.zip first."
    )


def build_trumans_loader(args):
    from torch.utils.data import DataLoader

    module = load_module_from_path(
        "official_trumans_dataset",
        PROJECT_ROOT / "trumans" / "datasets" / "trumans.py",
    )
    is_train = args.split == "train"
    dataset = module.TrumansDataset(
        folder=str(args.trumans_root),
        device=args.device,
        mesh_grid=args.trumans_mesh_grid,
        batch_size=args.batch_size,
        seq_len=args.trumans_seq_len,
        step=args.trumans_step,
        nb_voxels=args.trumans_nb_voxels,
        train=is_train,
        load_scene=args.trumans_load_scene,
        load_action=args.trumans_load_action,
        no_objects=args.trumans_no_objects,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=args.pin_memory,
    )
    return dataset, loader


def build_lingo_loader(args):
    from torch.utils.data import DataLoader

    module = load_module_from_path(
        "official_lingo_dataset",
        PROJECT_ROOT / "lingo-release" / "code" / "datasets" / "lingo.py",
    )
    dataset_root = resolve_lingo_dataset_root(args)
    is_train = args.split == "train"
    dataset = module.LingoDataset(
        folder=str(dataset_root),
        device=args.device,
        mesh_grid=args.lingo_mesh_grid,
        batch_size=args.batch_size,
        step=args.lingo_step,
        nb_voxels=args.lingo_nb_voxels,
        train=is_train,
        load_scene=args.lingo_load_scene,
        load_language=args.lingo_load_language,
        load_pelvis_goal=args.lingo_load_pelvis_goal,
        load_hand_goal=args.lingo_load_hand_goal,
        max_window_size=args.lingo_max_window_size,
        use_pi=args.lingo_use_pi,
        vis=args.lingo_vis,
        start_type=args.lingo_start_type,
        test_scene_name=args.lingo_test_scene_name,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=args.pin_memory,
    )
    return dataset, loader


def run_blender_dump(
    blend_file: Path,
    output_json: Path,
    args,
    include_names: list[str] | None = None,
    no_bbox: bool | None = None,
) -> Path:
    cmd = [
        args.python_bin,
        str(PROJECT_ROOT / "extract_blend_objects.py"),
        str(blend_file),
        "--output",
        str(output_json),
        "--blender-bin",
        args.blender_bin,
        "--noaudio",
        "--factory-startup",
    ]
    if no_bbox is None:
        no_bbox = args.blender_no_bbox
    if no_bbox:
        cmd.append("--no-bbox")
    for name in sorted(set(include_names or [])):
        cmd.extend(["--include-name", name])
    subprocess.run(cmd, check=True)
    return output_json


def load_trumans_scene_occ(scene_id: str, trumans_root: Path) -> np.ndarray:
    return np.load(trumans_root / "Scene" / f"{scene_id}.npy")


def load_lingo_scene_occ(scene_id: str, lingo_root: Path, split: str) -> np.ndarray:
    folder = "Scene" if split == "train" else "Scene_vis"
    return np.load(lingo_root / folder / f"{scene_id}.npy")


def load_lingo_language_motion_dict(lingo_root: Path) -> dict:
    with open(lingo_root / "language_motion_dict" / "language_motion_dict__inter_and_loco__16.pkl", "rb") as handle:
        return pickle.load(handle)


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

BODY_OBJECT_NAME_KEYWORDS = (
    ("chair", "chair"),
    ("seat", "chair"),
    ("stool", "chair"),
    ("bench", "chair"),
    ("sofa", "sofa"),
    ("couch", "sofa"),
    ("bed", "bed"),
)

FIXED_OBJECT_NAME_KEYWORDS = (
    "chair",
    "seat",
    "stool",
    "bench",
    "sofa",
    "couch",
    "bed",
    "refrigerator",
    "fridge",
    "cabinet",
    "drawer",
    "door",
    "oven",
    "microwave",
    "table",
    "desk",
    "counter",
    "shelf",
    "rack",
    "sink",
    "toilet",
    "bathtub",
    "wardrobe",
    "closet",
    "window",
    "monitor",
    "tv",
    "television",
    "washer",
    "dryer",
    "lamp",
    "nightstand",
    "dresser",
)

MOVABLE_OBJECT_NAME_KEYWORDS = (
    "cup",
    "bottle",
    "handbag",
    "phone",
    "pen",
    "book",
    "vase",
    "laptop",
    "mouse",
    "keyboard",
    "backpack",
    "briefcase",
    "remote",
    "plate",
    "bowl",
    "can",
    "box",
    "basket",
    "toy",
    "pillow",
)

HAND_OBJECT_TEXT_NAMES = (
    ("water bottle", "water bottle"),
    ("bottle", "water bottle"),
    ("refrigerator", "refrigerator"),
    ("fridge", "refrigerator"),
    ("handbag", "handbag"),
    ("bag", "bag"),
    ("backpack", "backpack"),
    ("briefcase", "briefcase"),
    ("cup", "cup"),
    ("phone", "phone"),
    ("mouse", "mouse"),
    ("keyboard", "keyboard"),
    ("laptop", "laptop"),
    ("book", "book"),
    ("pen", "pen"),
    ("vase", "vase"),
    ("plate", "plate"),
    ("bowl", "bowl"),
    ("box", "box"),
    ("basket", "basket"),
    ("remote", "remote"),
    ("pillow", "pillow"),
)

HEURISTIC_FURNITURE_CLASSES = {"chair", "sofa", "bed"}

CANONICAL_NAME_RULES = (
    ("static chair", "chair"),
    ("movable chair", "chair"),
    ("chair base", "chair"),
    ("chair seat", "chair"),
    ("stool", "chair"),
    ("bench", "chair"),
    ("sofa", "sofa"),
    ("couch", "sofa"),
    ("bed", "bed"),
    ("drawer base", "drawer"),
    ("drawer drawer", "drawer"),
    ("cabinet base", "cabinet"),
    ("cabinet door", "cabinet"),
    ("cupboard", "cabinet"),
    ("refrigerator base", "refrigerator"),
    ("refrigerator door", "refrigerator"),
    ("fridge base", "refrigerator"),
    ("fridge door", "refrigerator"),
    ("oven base", "oven"),
    ("oven door", "oven"),
    ("microwave base", "microwave"),
    ("microwave door", "microwave"),
    ("laptop base", "laptop"),
    ("laptop screen", "laptop"),
    ("book right", "book"),
    ("book left", "book"),
    ("door door", "door"),
    ("water bottle", "water bottle"),
    ("bottle", "water bottle"),
    ("cup", "cup"),
    ("phone", "phone"),
    ("handbag", "handbag"),
    ("bag", "bag"),
    ("backpack", "backpack"),
    ("briefcase", "briefcase"),
    ("mouse", "mouse"),
    ("keyboard", "keyboard"),
    ("book", "book"),
    ("pen", "pen"),
    ("vase", "vase"),
    ("table", "table"),
    ("desk", "table"),
    ("monitor", "monitor"),
    ("whiteboard", "whiteboard"),
    ("door", "door"),
)


def is_body_action(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in ("sit down", "stand up", "lie down", "squat", "kneel", "crouch"))


def infer_goal_type(text: str, hand: str | None) -> str:
    if is_body_action(text):
        return "body"
    if hand is not None:
        return "hand"
    return "unknown"


def infer_hand(text: str) -> str | None:
    return parse_hand_hint(text)


def is_explicit_object_interaction(text: str) -> bool:
    lower = text.lower()
    return any(
        keyword in lower
        for keyword in (
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
    )


def infer_goal_object_name_from_text(text: str) -> str | None:
    lower = text.lower()
    for phrase, name in HAND_OBJECT_TEXT_NAMES:
        if phrase in lower:
            return name
    return None


def canonicalize_goal_name(name: str | None) -> str | None:
    if name is None:
        return None
    lower = name.lower().strip().replace("_", " ").replace("-", " ")
    for pattern, canonical in CANONICAL_NAME_RULES:
        if pattern in lower:
            return canonical
    return " ".join(lower.split())


def infer_object_class_from_id(object_id: str | None) -> str | None:
    if object_id is None:
        return None
    lower = object_id.lower().strip()
    if "/model" in lower and not any(pattern in lower for pattern, _ in CANONICAL_NAME_RULES):
        return None
    return canonicalize_goal_name(lower)


def infer_body_goal_name(object_id: str | None, scene_objects: list[dict] | None = None) -> str | None:
    if object_id is None:
        return None
    lower = object_id.lower()
    for keyword, name in BODY_OBJECT_NAME_KEYWORDS:
        if keyword in lower:
            return name

    if scene_objects is not None:
        by_id = {obj["object_id"]: obj for obj in scene_objects}
        obj = by_id.get(object_id)
        if obj is not None and "motion_bbox_min" in obj and "motion_bbox_max" in obj:
            bbox_min = np.asarray(obj["motion_bbox_min"], dtype=np.float32)
            bbox_max = np.asarray(obj["motion_bbox_max"], dtype=np.float32)
            footprint = np.abs(bbox_max[[0, 2]] - bbox_min[[0, 2]])
            width = float(min(footprint))
            depth = float(max(footprint))
            if depth >= 1.7 or width >= 1.3:
                return "bed"
            if depth >= 1.1 or width >= 0.9:
                return "sofa"
            return "chair"

    return "chair"


def infer_goal_object_name(text: str, goal_object_id: str | None, goal_type: str, scene_objects: list[dict] | None = None) -> str | None:
    object_name = infer_object_class_from_id(goal_object_id)
    if object_name is not None:
        return object_name
    if goal_type == "hand":
        return canonicalize_goal_name(infer_goal_object_name_from_text(text))
    if goal_type == "body":
        return canonicalize_goal_name(infer_body_goal_name(goal_object_id, scene_objects))
    return canonicalize_goal_name(infer_goal_object_name_from_text(text))


def infer_movable(goal_object_name: str | None, object_id: str | None, goal_object_source: str | None) -> bool:
    text = " ".join(part for part in (goal_object_name, object_id, goal_object_source) if part).lower()
    if any(keyword in text for keyword in FIXED_OBJECT_NAME_KEYWORDS):
        return False
    if any(keyword in text for keyword in MOVABLE_OBJECT_NAME_KEYWORDS):
        return True
    if goal_object_source == "Object_all":
        return True
    return False


def lingo_extract_text(text_entry: Any) -> str:
    if isinstance(text_entry, (list, tuple)):
        return str(text_entry[0])
    return str(text_entry)


def build_lingo_scene_sequence_index(args) -> dict[str, list[int]]:
    lingo_root = resolve_lingo_dataset_root(args)
    language_motion_dict = load_lingo_language_motion_dict(lingo_root)
    with open(lingo_root / "scene_name.pkl", "rb") as handle:
        scene_name = pickle.load(handle)

    scene_to_sequences: dict[str, list[int]] = {}
    start_idx = language_motion_dict["start_idx"]
    for seq_idx, start in enumerate(start_idx):
        scene_id = str(scene_name[int(start)])
        if args.split == "test" and args.lingo_test_scene_name and scene_id != args.lingo_test_scene_name:
            continue
        scene_to_sequences.setdefault(scene_id, []).append(int(seq_idx))
    return scene_to_sequences


def lingo_infer_hand(text: str, left_inter: int, right_inter: int) -> str | None:
    hinted = infer_hand(text)
    if hinted is not None:
        return hinted
    if left_inter != -1 and right_inter != -1:
        return "both"
    if left_inter != -1:
        return "left"
    if right_inter != -1:
        return "right"
    return None


def lingo_interaction_frame_and_targets(
    text: str,
    start_idx: int,
    end_idx: int,
    end_range: int,
    left_inter: int,
    right_inter: int,
    joints: np.ndarray,
) -> dict:
    hand = lingo_infer_hand(text, left_inter, right_inter)
    goal_type = infer_goal_type(text, hand)

    if hand == "left":
        interaction_frame = int(left_inter)
        hand_goal = np.asarray(joints[interaction_frame, LINGO_LEFT_HAND_IDX, :], dtype=np.float32)
    elif hand == "right":
        interaction_frame = int(right_inter)
        hand_goal = np.asarray(joints[interaction_frame, LINGO_RIGHT_HAND_IDX, :], dtype=np.float32)
    elif hand == "both":
        interaction_frame = int(left_inter if left_inter != -1 else right_inter)
        hand_goal = np.asarray(joints[interaction_frame, LINGO_LEFT_HAND_IDX, :], dtype=np.float32)
    elif "sit down" in text.lower() or "lie down" in text.lower():
        interaction_frame = int(end_range)
        hand_goal = None
    else:
        interaction_frame = int(end_idx - 3)
        hand_goal = None

    pelvis_goal = np.asarray(joints[interaction_frame, 0, :], dtype=np.float32)
    yaw_goal = None
    if interaction_frame < len(joints) - 1:
        delta = np.asarray(joints[min(interaction_frame + 1, len(joints) - 1), 0, :], dtype=np.float32) - pelvis_goal
        if float(np.linalg.norm(delta[[0, 2]])) > 1e-6:
            yaw_goal = float(np.arctan2(delta[2], delta[0]))

    return {
        "interaction_frame": interaction_frame,
        "goal_type": goal_type,
        "hand": hand,
        "pelvis_goal": pelvis_goal,
        "yaw_goal": yaw_goal,
        "hand_goal": hand_goal,
    }


def lingo_estimate_object_pose(
    text: str,
    goal_type: str,
    hand: str | None,
    pelvis_goal: np.ndarray,
    hand_goal: np.ndarray | None,
    start_idx: int,
) -> dict | None:
    object_class = canonicalize_goal_name(infer_goal_object_name_from_text(text))
    if object_class is None and goal_type == "body":
        lower = text.lower()
        if "lie down" in lower:
            object_class = "bed"
        elif "sit down" in lower or "stand up" in lower:
            object_class = "chair"
        else:
            object_class = "chair"
    if object_class is None:
        return None

    if goal_type == "hand" and hand_goal is not None:
        location = np.asarray(hand_goal, dtype=np.float32)
    else:
        location = np.asarray(pelvis_goal, dtype=np.float32)
    return {
        "frame": int(start_idx),
        "location": location,
        "rotation_euler": None,
        "object_class": object_class,
    }


def cluster_lingo_object_slots(segment_rows: list[dict], distance_threshold: float = 0.75) -> tuple[list[dict], dict[tuple[str, int], str]]:
    clusters: list[dict] = []
    segment_to_object_id: dict[tuple[str, int], str] = {}
    counters: dict[str, int] = {}

    for row in segment_rows:
        object_pose = row.get("object_pose")
        object_class = row.get("goal_object_name")
        if object_pose is None or object_class is None:
            continue
        location = np.asarray(object_pose["location"], dtype=np.float32)
        matched = None
        for cluster in clusters:
            if cluster["object_class"] != object_class:
                continue
            if float(np.linalg.norm(location - cluster["initial_location"])) <= distance_threshold:
                matched = cluster
                break

        if matched is None:
            counters[object_class] = counters.get(object_class, 0) + 1
            object_id = f"{object_class.replace(' ', '_')}_{counters[object_class]:02d}"
            matched = {
                "object_id": object_id,
                "object_class": object_class,
                "movable": infer_movable(object_class, object_id, "lingo_proxy"),
                "initial_location": location,
                "motion_location": location,
                "motion_rotation_euler": object_pose.get("rotation_euler"),
                "source": "lingo_proxy",
                "interactions": [],
            }
            clusters.append(matched)

        segment_to_object_id[(row["sequence_id"], row["segment_id"])] = matched["object_id"]
        matched["interactions"].append(
            {
                "sequence_id": row["sequence_id"],
                "segment_id": row["segment_id"],
                "interaction_frame": row["interaction_frame"],
                "hand": row["hand"],
                "object_position": object_pose["location"],
                "pelvis_position": row["pelvis_goal"],
                "hand_position": row["hand_goal"],
            }
        )

    clusters.sort(key=lambda item: item["object_id"])
    return clusters, segment_to_object_id


def generate_sequence_plot(sequence_id: str, scene_id: str, segment_list: list[dict], args) -> Path:
    plot_dir = scene_plot_dir(scene_id, sequence_id, args)
    plot_input_path = scene_sequence_dir(scene_id, args) / f"{sequence_id}.plot_segments.json"
    write_json(
        plot_input_path,
        {
            "sequence_id": sequence_id,
            "scene_id": scene_id,
            "segment_list": segment_list,
        },
    )
    cmd = [
        args.python_bin,
        str(PROJECT_ROOT / "visualize_trumans_sequence_actions.py"),
        "--sequence-id",
        sequence_id,
        "--scene-name",
        scene_id,
        "--segment-json",
        str(plot_input_path),
        "--output-dir",
        str(plot_dir.parent),
    ]
    subprocess.run(cmd, check=True)
    return plot_dir


def generate_lingo_sequence_plot(sequence_id: str, scene_id: str, segment_list: list[dict], args) -> Path:
    plot_dir = scene_output_dir(scene_id, args) / "plots" / f"lingo_sequence_actions_{sequence_id}"
    plot_input_path = scene_sequence_dir(scene_id, args) / f"{sequence_id}.plot_segments.json"
    write_json(
        plot_input_path,
        {
            "sequence_id": sequence_id,
            "scene_id": scene_id,
            "segment_list": segment_list,
        },
    )
    cmd = [
        args.python_bin,
        str(PROJECT_ROOT / "visualize_lingo_sequence_actions.py"),
        "--sequence-id",
        sequence_id,
        "--scene-name",
        scene_id,
        "--segment-json",
        str(plot_input_path),
        "--output-dir",
        str(plot_dir.parent),
        "--split",
        args.split,
    ]
    subprocess.run(cmd, check=True)
    return plot_dir


def generate_interaction_maps(scene_json_path: Path, args) -> Path:
    output_dir = scene_json_path.parent / "interaction_maps"
    cmd = [
        args.python_bin,
        str(PROJECT_ROOT / "visualize_trumans_interaction_map.py"),
        "--scene-json",
        str(scene_json_path),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)
    return output_dir


def refresh_sequence_plot(sequence_record: dict, args) -> None:
    plot_dir = generate_sequence_plot(
        sequence_id=sequence_record["sequence_id"],
        scene_id=sequence_record["scene_id"],
        segment_list=sequence_record["segment_list"],
        args=args,
    )
    for segment in sequence_record["segment_list"]:
        segment["plot_path"] = to_repo_relative(
            plot_dir / f"{segment['segment_id']:02d}_{segment['start']:04d}_{segment['end']:04d}.png"
        )


def refresh_lingo_sequence_plot(sequence_record: dict, args) -> None:
    plot_dir = generate_lingo_sequence_plot(
        sequence_id=sequence_record["sequence_id"],
        scene_id=sequence_record["scene_id"],
        segment_list=sequence_record["segment_list"],
        args=args,
    )
    for segment in sequence_record["segment_list"]:
        segment["plot_path"] = to_repo_relative(
            plot_dir / f"{segment['segment_id']:02d}_{segment['start']:04d}_{segment['end']:04d}.png"
        )


def strip_plot_only_segment_fields(sequence_record: dict) -> dict:
    for segment in sequence_record.get("segment_list", []):
        segment.pop("goal_object_location", None)
    return sequence_record


def blender_record_to_scene_object(obj: dict) -> dict:
    record = {
        "object_id": obj["name"],
        "object_class": obj.get("data_name") or obj["type"],
        "motion_location": blender_world_to_motion(obj["world_location"]),
        "motion_rotation_euler": [
            obj["world_rotation_euler"][0],
            obj["world_rotation_euler"][2],
            -obj["world_rotation_euler"][1],
        ],
        "source": "blend_dump",
    }
    if "bbox_min" in obj and "bbox_max" in obj:
        record["motion_bbox_min"] = blender_world_to_motion(obj["bbox_min"])
        record["motion_bbox_max"] = blender_world_to_motion(obj["bbox_max"])
        record["dimensions"] = obj.get("dimensions")
    return record


def load_blender_scene_objects(scene_id: str, args, object_ids: list[str] | None = None) -> tuple[Path | None, list[dict]]:
    recording_dir = args.trumans_root / "Recordings_blend" / scene_id
    blend_file = recording_dir / f"{scene_id}.blend"
    cache_dir = cache_scene_dir(scene_id, args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if object_ids:
        blend_dump = cache_dir / "candidate_bbox.json"
        if blend_file.exists() and (args.force_blender_dump or not blend_dump.exists()):
            run_blender_dump(blend_file, blend_dump, args, include_names=object_ids, no_bbox=False)
    else:
        blend_dump = cache_dir / "transforms.json"
        if blend_file.exists() and (args.force_blender_dump or not blend_dump.exists()):
            run_blender_dump(blend_file, blend_dump, args, no_bbox=True)

    objects = []
    if blend_dump.exists():
        dump = json.loads(blend_dump.read_text())
        for obj in dump["objects"]:
            if obj["type"] not in {"MESH", "EMPTY"}:
                continue
            objects.append(blender_record_to_scene_object(obj))
    return (blend_dump if blend_dump.exists() else None), objects


def filter_body_candidates(scene_objects: list[dict]) -> list[dict]:
    filtered = []
    for obj in scene_objects:
        name = obj["object_id"].lower()
        cls = str(obj["object_class"]).lower()
        joined = f"{name} {cls}"
        if any(keyword in joined for keyword in BODY_ACTION_NEGATIVE_KEYWORDS):
            continue
        filtered.append(obj)
    return filtered


def lookup_scene_object(scene_objects: list[dict], object_id: str | None) -> dict | None:
    if object_id is None:
        return None
    for obj in scene_objects:
        if obj["object_id"] == object_id:
            return obj
    return None


def resolve_segment_start_object_pose(
    segment_start: int,
    goal_object: str | None,
    goal_object_source: str | None,
    object_pose_dict: dict,
    scene_objects: list[dict],
) -> dict | None:
    if goal_object is None:
        return None
    if goal_object_source == "Object_all":
        entry = object_pose_dict.get(goal_object)
        if entry is None:
            return None
        if segment_start >= len(entry.get("location", [])):
            return None
        pose = {
            "frame": segment_start,
            "location": np.asarray(entry["location"][segment_start], dtype=np.float32),
            "rotation_euler": None,
        }
        if "rotation" in entry and segment_start < len(entry["rotation"]):
            pose["rotation_euler"] = np.asarray(entry["rotation"][segment_start], dtype=np.float32)
        return pose

    obj = lookup_scene_object(scene_objects, goal_object)
    if obj is None:
        return None
    return {
        "frame": segment_start,
        "location": np.asarray(obj["motion_location"], dtype=np.float32) if obj.get("motion_location") is not None else None,
        "rotation_euler": (
            np.asarray(obj["motion_rotation_euler"], dtype=np.float32)
            if obj.get("motion_rotation_euler") is not None
            else None
        ),
    }


def preprocess_trumans_scene(
    scene_id: str,
    args,
    sequence_ids: list[str],
    object_list: list[dict],
    blend_dump_path: Path | None = None,
) -> dict:
    scene_occ = load_trumans_scene_occ(scene_id, args.trumans_root)
    scene_dir = scene_output_dir(scene_id, args)
    walkable_info = compute_walkable_map(
        scene_occ,
        body_height_voxels=args.body_height_voxels,
        free_threshold=args.free_threshold,
    )
    prefix = f"trumans_{scene_id}"
    vis_paths = save_walkable_visualizations(scene_dir, prefix, scene_occ, walkable_info)

    result = {
        "scene_id": scene_id,
        "scene_occ_path": to_repo_relative(args.trumans_root / "Scene" / f"{scene_id}.npy"),
        "walkable_map_path": to_repo_relative(vis_paths["walkable_png"]),
        "blend_dump_path": to_repo_relative(blend_dump_path) if blend_dump_path is not None else None,
        "sequence_ids": sequence_ids,
        "object_list": object_list,
    }
    return result


def preprocess_lingo_scene(scene_id: str, args) -> dict:
    lingo_root = resolve_lingo_dataset_root(args)
    scene_occ = load_lingo_scene_occ(scene_id, lingo_root, args.split)
    scene_dir = scene_output_dir(scene_id, args)
    walkable_info = compute_walkable_map(
        scene_occ,
        body_height_voxels=args.body_height_voxels,
        free_threshold=args.free_threshold,
    )
    prefix = f"lingo_{scene_id}"
    vis_paths = save_walkable_visualizations(scene_dir, prefix, scene_occ, walkable_info)
    return {
        "scene_id": scene_id,
        "scene_occ_path": to_repo_relative(
            lingo_root / ("Scene" if args.split == "train" else "Scene_vis") / f"{scene_id}.npy"
        ),
        "walkable_map_path": to_repo_relative(vis_paths["walkable_png"]),
        "object_list": [],
    }


def preprocess_lingo_sequence(sequence_index: int, args, scene_id: str | None = None) -> dict:
    lingo_root = resolve_lingo_dataset_root(args)
    language_motion_dict = load_lingo_language_motion_dict(lingo_root)
    with open(lingo_root / "scene_name.pkl", "rb") as handle:
        scene_name = pickle.load(handle)
    joints = np.load(lingo_root / "human_joints_aligned.npy", mmap_mode="r")

    start_idx = int(language_motion_dict["start_idx"][sequence_index])
    end_idx = int(language_motion_dict["end_idx"][sequence_index])
    end_range = int(language_motion_dict["end_range"][sequence_index])
    text = lingo_extract_text(language_motion_dict["text"][sequence_index])
    current_scene_id = str(scene_name[start_idx])
    if scene_id is not None and current_scene_id != scene_id:
        raise ValueError(f"Sequence {sequence_index} belongs to scene {current_scene_id}, not {scene_id}")

    left_inter = int(language_motion_dict["left_hand_inter_frame"][sequence_index])
    right_inter = int(language_motion_dict["right_hand_inter_frame"][sequence_index])
    interaction = lingo_interaction_frame_and_targets(
        text=text,
        start_idx=start_idx,
        end_idx=end_idx,
        end_range=end_range,
        left_inter=left_inter,
        right_inter=right_inter,
        joints=joints,
    )
    object_pose = lingo_estimate_object_pose(
        text=text,
        goal_type=interaction["goal_type"],
        hand=interaction["hand"],
        pelvis_goal=interaction["pelvis_goal"],
        hand_goal=interaction["hand_goal"],
        start_idx=start_idx,
    )

    sequence_id = f"lingo_{sequence_index:06d}"
    segment = {
        "segment_id": 0,
        "start": start_idx,
        "end": end_idx,
        "interaction_frame": interaction["interaction_frame"],
        "text": text,
        "goal_type": interaction["goal_type"],
        "hand": interaction["hand"],
        "goal_object": None,
        "goal_object_name": None if object_pose is None else object_pose["object_class"],
        "goal_object_source": None if object_pose is None else "lingo_proxy",
        "object_pose": None if object_pose is None else {
            "frame": object_pose["frame"],
            "location": object_pose["location"],
            "rotation_euler": object_pose["rotation_euler"],
        },
        "pelvis_goal": interaction["pelvis_goal"],
        "yaw_goal": interaction["yaw_goal"],
        "hand_goal": interaction["hand_goal"],
        "plot_path": to_repo_relative(
            scene_output_dir(current_scene_id, args) / "plots" / f"lingo_sequence_actions_{sequence_id}" / f"00_{start_idx:04d}_{end_idx:04d}.png"
        ),
    }
    sequence_record = {
        "sequence_id": sequence_id,
        "scene_id": current_scene_id,
        "segment_list": [segment],
        "human_pose_ref": {
            "source_path": to_repo_relative(lingo_root / "human_joints_aligned.npy"),
            "global_start": start_idx,
            "global_end_exclusive": end_idx,
            "step": args.lingo_step,
            "num_frames": int((end_idx - start_idx) // args.lingo_step),
        },
        "plot_dir": to_repo_relative(scene_output_dir(current_scene_id, args) / "plots" / f"lingo_sequence_actions_{sequence_id}"),
    }
    return sequence_record


def load_trumans_events(action_file: Path) -> list[dict]:
    if not action_file.exists():
        stem = action_file.stem
        if "_augment" in stem:
            base_stem = stem.split("_augment", 1)[0]
            fallback = action_file.with_name(f"{base_stem}{action_file.suffix}")
            if fallback.exists():
                action_file = fallback
    events = []
    for event_id, line in enumerate(action_file.read_text().splitlines()):
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        start, end, text = parts
        events.append(
            {
                "event_id": event_id,
                "start": int(start),
                "end": int(end),
                "text": text.strip(),
            }
        )
    return events


def build_trumans_scene_sequence_index(trumans_root: Path) -> dict[str, list[str]]:
    seg_name = np.load(trumans_root / "seg_name.npy", allow_pickle=True)
    scene_flag = np.load(trumans_root / "scene_flag.npy", allow_pickle=True)
    scene_list = np.load(trumans_root / "scene_list.npy", allow_pickle=True)
    scene_to_sequences: dict[str, list[str]] = {}
    seen = set()
    for global_idx, seq in enumerate(seg_name.tolist()):
        if seq in seen:
            continue
        seen.add(seq)
        scene_id = str(scene_list[int(scene_flag[global_idx])])
        scene_to_sequences.setdefault(scene_id, []).append(str(seq))
    return scene_to_sequences


def select_trumans_event_object(
    event: dict,
    joints: np.ndarray,
    sequence_indices: np.ndarray,
    object_pose_dict: dict,
    scene_objects: list[dict],
) -> dict:
    start = event["start"]
    end = event["end"]
    text = event["text"]
    hand = infer_hand(event["text"])
    frame_ids = sequence_indices[start : end + 1]
    pelvis_track = np.asarray(joints[frame_ids, 0, :], dtype=np.float32)
    left_track = np.asarray(joints[frame_ids, TRUMANS_LEFT_HAND_IDX, :], dtype=np.float32)
    right_track = np.asarray(joints[frame_ids, TRUMANS_RIGHT_HAND_IDX, :], dtype=np.float32)

    if is_body_action(text):
        object_pool = filter_body_candidates(scene_objects)
        best = None
        for offset, local_frame in enumerate(range(start, end + 1)):
            query_xyz = pelvis_track[offset]
            for obj in object_pool:
                obj_loc = np.asarray(obj["motion_location"], dtype=np.float32)
                if "motion_bbox_min" in obj and "motion_bbox_max" in obj:
                    dist = point_to_aabb_distance(
                        query_xyz,
                        np.asarray(obj["motion_bbox_min"], dtype=np.float32),
                        np.asarray(obj["motion_bbox_max"], dtype=np.float32),
                    )
                else:
                    dist = float(np.linalg.norm(query_xyz - obj_loc))
                if best is None or dist < best["distance"]:
                    best = {
                        "interaction_frame": local_frame,
                        "object_id": obj["object_id"],
                        "object_location": obj_loc,
                        "distance": dist,
                        "source": "blend_dump",
                    }
        return {
            "interaction_frame": (start + end) // 2 if best is None else best["interaction_frame"],
            "goal_object": None if best is None else best["object_id"],
            "goal_object_location": None if best is None else best["object_location"],
            "goal_object_source": None if best is None else best["source"],
        }

    if hand == "left":
        query_tracks = [("left", left_track)]
    elif hand == "right":
        query_tracks = [("right", right_track)]
    elif is_explicit_object_interaction(text):
        query_tracks = [("left", left_track), ("right", right_track)]
    else:
        query_tracks = [("pelvis", pelvis_track)]

    best = None
    for _query_name, query_track in query_tracks:
        for offset, local_frame in enumerate(range(start, end + 1)):
            nearest = nearest_named_objects(
                object_pose_dict,
                local_frame,
                query_track[offset],
                topk=1,
            )
            if not nearest:
                continue
            row = nearest[0]
            if best is None or row["distance"] < best["distance"]:
                best = {
                    "interaction_frame": local_frame,
                    "goal_object": row["name"],
                    "goal_object_location": row["location"],
                    "goal_object_source": "Object_all",
                    "distance": row["distance"],
                }
    if best is None:
        return {
            "interaction_frame": (start + end) // 2,
            "goal_object": None,
            "goal_object_location": None,
            "goal_object_source": None,
        }
    return best


def preprocess_trumans_sequence(sequence_id: str, scene_id: str, scene_objects: list[dict], args) -> dict:
    trumans_root = args.trumans_root
    source_sequence_id = trumans_base_sequence_id(sequence_id)
    seg_name = np.load(trumans_root / "seg_name.npy", allow_pickle=True)
    joints = np.load(trumans_root / "human_joints.npy", mmap_mode="r")

    idx = np.where(seg_name == sequence_id)[0]
    if len(idx) == 0:
        raise ValueError(f"Sequence not found in seg_name.npy: {sequence_id}")

    obj_pose_path = trumans_root / "Object_all" / "Object_pose" / f"{source_sequence_id}.npy"
    object_pose_dict = np.load(obj_pose_path, allow_pickle=True).item() if obj_pose_path.exists() else {}
    events = load_trumans_events(trumans_root / "Actions" / f"{source_sequence_id}.txt")
    if any(is_body_action(event["text"]) for event in events):
        scene_objects = attach_body_candidate_bboxes(scene_id, scene_objects, args)

    segment_list = []
    interacted_object_ids = set()
    goals = []
    prev_end = None
    for event in events:
        selection = select_trumans_event_object(event, joints, idx, object_pose_dict, scene_objects)
        if selection["goal_object"] is not None:
            interacted_object_ids.add(selection["goal_object"])
        segment_start = event["start"] if prev_end is None else prev_end
        goal_type = infer_goal_type(event["text"], infer_hand(event["text"]))
        goal_object_name = infer_goal_object_name(event["text"], selection["goal_object"], goal_type)
        object_pose = resolve_segment_start_object_pose(
            segment_start=segment_start,
            goal_object=selection["goal_object"],
            goal_object_source=selection["goal_object_source"],
            object_pose_dict=object_pose_dict,
            scene_objects=scene_objects,
        )
        interaction_global = int(idx[selection["interaction_frame"]])
        interaction_pose = {
            "pelvis": np.asarray(joints[interaction_global, 0, :], dtype=np.float32),
            "left_hand": np.asarray(joints[interaction_global, TRUMANS_LEFT_HAND_IDX, :], dtype=np.float32),
            "right_hand": np.asarray(joints[interaction_global, TRUMANS_RIGHT_HAND_IDX, :], dtype=np.float32),
        }
        segment_list.append(
            {
                "segment_id": event["event_id"],
                "start": segment_start,
                "end": event["end"],
                "interaction_frame": selection["interaction_frame"],
                "text": event["text"],
                "goal_type": goal_type,
                "hand": infer_hand(event["text"]),
                "goal_object": selection["goal_object"],
                "goal_object_name": goal_object_name,
                "goal_object_source": selection["goal_object_source"],
                "object_pose": object_pose,
                "goal_object_location": selection["goal_object_location"],
                "plot_path": to_repo_relative(
                    scene_plot_dir(scene_id, sequence_id, args) / f"{event['event_id']:02d}_{event['start']:04d}_{event['end']:04d}.png"
                ),
            }
        )
        if selection["goal_object"] is not None:
            hand_name = infer_hand(event["text"])
            hand_position = None
            if hand_name == "left":
                hand_position = interaction_pose["left_hand"]
            elif hand_name == "right":
                hand_position = interaction_pose["right_hand"]
            goals.append(
                {
                    "sequence_id": sequence_id,
                    "segment_id": event["event_id"],
                    "text": event["text"],
                    "goal_type": goal_type,
                    "hand": hand_name,
                    "interaction_frame": selection["interaction_frame"],
                    "object_id": selection["goal_object"],
                    "goal_object_name": goal_object_name,
                    "goal_object_source": selection["goal_object_source"],
                    "object_location": selection["goal_object_location"],
                    "pelvis_position": interaction_pose["pelvis"],
                    "hand_position": hand_position,
                }
            )
        prev_end = event["end"]

    plot_dir = generate_sequence_plot(sequence_id, scene_id, segment_list, args)

    return {
        "sequence_id": sequence_id,
        "scene_id": scene_id,
        "segment_list": segment_list,
        "human_pose_ref": {
            "source_path": to_repo_relative(trumans_root / "human_joints.npy"),
            "global_start": int(idx[0]),
            "global_end_exclusive": int(idx[-1]) + 1,
            "num_frames": int(len(idx)),
        },
        "plot_dir": to_repo_relative(plot_dir),
        "_interacted_object_ids": sorted(interacted_object_ids),
        "_goals": goals,
    }


def finalize_scene_objects_with_bbox(scene_id: str, scene_objects: list[dict], object_ids: list[str], args) -> tuple[Path | None, list[dict]]:
    object_id_set = set(object_ids)
    if not object_id_set:
        return None, []
    bbox_dump_path, bbox_objects = load_blender_scene_objects(scene_id, args, object_ids=sorted(object_id_set))
    if bbox_objects:
        by_id = {obj["object_id"]: obj for obj in bbox_objects}
        return bbox_dump_path, [by_id[obj_id] for obj_id in object_ids if obj_id in by_id]
    return None, [obj for obj in scene_objects if obj["object_id"] in object_id_set]


def attach_body_candidate_bboxes(scene_id: str, scene_objects: list[dict], args) -> list[dict]:
    body_candidates = filter_body_candidates(scene_objects)
    candidate_ids = [obj["object_id"] for obj in body_candidates]
    if not candidate_ids:
        return scene_objects
    _, bbox_objects = finalize_scene_objects_with_bbox(scene_id, scene_objects, candidate_ids, args)
    if not bbox_objects:
        return scene_objects
    bbox_by_id = {obj["object_id"]: obj for obj in bbox_objects}
    merged = []
    for obj in scene_objects:
        bbox_obj = bbox_by_id.get(obj["object_id"])
        if bbox_obj is None:
            merged.append(obj)
            continue
        enriched = dict(obj)
        if "motion_bbox_min" in bbox_obj and "motion_bbox_max" in bbox_obj:
            enriched["motion_bbox_min"] = bbox_obj["motion_bbox_min"]
            enriched["motion_bbox_max"] = bbox_obj["motion_bbox_max"]
        if "dimensions" in bbox_obj:
            enriched["dimensions"] = bbox_obj["dimensions"]
        merged.append(enriched)
    return merged


def validate_object_id_bijection(object_list: list[dict]) -> None:
    id_to_data: dict[str, str | None] = {}
    data_to_id: dict[str, str] = {}
    for obj in object_list:
        object_id = obj["object_id"]
        data_name = obj.get("blender_data_name")
        prev_data = id_to_data.get(object_id)
        if prev_data is not None and prev_data != data_name:
            raise ValueError(f"object_id {object_id} maps to multiple blender_data_name values: {prev_data}, {data_name}")
        id_to_data[object_id] = data_name
        if data_name is None:
            continue
        prev_id = data_to_id.get(data_name)
        if prev_id is not None and prev_id != object_id:
            raise ValueError(f"blender_data_name {data_name} maps to multiple object_id values: {prev_id}, {object_id}")
        data_to_id[data_name] = object_id


def canonicalize_scene_object_ids(object_list: list[dict]) -> tuple[list[dict], dict[str, str]]:
    counters: dict[str, int] = {}
    data_name_to_object_id: dict[str, str] = {}
    id_map: dict[str, str] = {}
    canonical_objects: list[dict] = []

    for obj in sorted(object_list, key=lambda row: (str(row.get("object_class")), str(row.get("blender_data_name")), str(row["object_id"]))):
        row = dict(obj)
        original_id = row["object_id"]
        object_class = row.get("object_class")
        data_name = row.get("blender_data_name")

        if object_class in HEURISTIC_FURNITURE_CLASSES and data_name:
            canonical_id = data_name_to_object_id.get(data_name)
            if canonical_id is None:
                counters[object_class] = counters.get(object_class, 0) + 1
                canonical_id = f"{object_class}_{counters[object_class]:02d}"
                data_name_to_object_id[data_name] = canonical_id
        else:
            canonical_id = original_id

        row["object_id"] = canonical_id
        canonical_objects.append(row)
        id_map[original_id] = canonical_id

    validate_object_id_bijection(canonical_objects)
    return canonical_objects, id_map


def build_scene_object_list(
    scene_objects: list[dict],
    sequence_goal_rows: list[dict],
) -> tuple[list[dict], dict[str, str]]:
    by_id = {obj["object_id"]: dict(obj) for obj in scene_objects}
    merged: dict[str, dict] = {}
    for row in sequence_goal_rows:
        object_id = row["object_id"]
        if object_id in by_id:
            base = dict(by_id[object_id])
        else:
            base = {
                "object_id": object_id,
                "motion_location": row.get("object_location"),
                "motion_rotation_euler": None,
                "source": row.get("goal_object_source"),
            }
        goal_name = infer_object_class_from_id(object_id) or row["goal_object_name"] or infer_body_goal_name(object_id, scene_objects)
        base["blender_data_name"] = base.pop("object_class", None)
        base["object_class"] = goal_name
        base["movable"] = infer_movable(goal_name, object_id, row.get("goal_object_source"))
        if base.get("motion_location") is None:
            base["motion_location"] = row.get("object_location")
        base["initial_location"] = base.get("motion_location")
        interaction_entry = {
            "sequence_id": row["sequence_id"],
            "segment_id": row["segment_id"],
            "interaction_frame": row["interaction_frame"],
            "hand": row["hand"],
            "object_position": row["object_location"],
            "pelvis_position": row["pelvis_position"],
            "hand_position": row["hand_position"],
        }
        if object_id not in merged:
            base["interactions"] = [interaction_entry]
            merged[object_id] = base
            continue
        prev = merged[object_id]
        prev["movable"] = bool(prev["movable"] or base["movable"])
        if prev.get("object_class") is None and base.get("object_class") is not None:
            prev["object_class"] = base["object_class"]
        if prev.get("motion_location") is None and base.get("motion_location") is not None:
            prev["motion_location"] = base["motion_location"]
            prev["initial_location"] = base["motion_location"]
        prev.setdefault("interactions", []).append(interaction_entry)
    return canonicalize_scene_object_ids(list(merged.values()))


def remap_sequence_record_object_ids(sequence_record: dict, object_id_map: dict[str, str]) -> dict:
    for segment in sequence_record.get("segment_list", []):
        goal_object = segment.get("goal_object")
        if goal_object in object_id_map:
            segment["goal_object"] = object_id_map[goal_object]
    return sequence_record


def remap_goal_rows(sequence_goal_rows: list[dict], object_id_map: dict[str, str]) -> list[dict]:
    remapped = []
    for row in sequence_goal_rows:
        item = dict(row)
        if item["object_id"] in object_id_map:
            item["object_id"] = object_id_map[item["object_id"]]
        remapped.append(item)
    return remapped


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scene/sequence preprocessing for TRUMANS and LINGO."
    )
    parser.add_argument("--dataset", choices=["trumans", "lingo"], default="trumans")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-batches", type=int, default=1)

    parser.add_argument(
        "--trumans-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "trumans",
    )
    parser.add_argument("--trumans-seq-len", type=int, default=16)
    parser.add_argument("--trumans-step", type=int, default=3)
    parser.add_argument("--trumans-nb-voxels", type=int, default=32)
    parser.add_argument(
        "--trumans-mesh-grid",
        type=float,
        nargs=6,
        default=[-0.6, 0.6, 0.0, 1.2, -0.6, 0.6],
    )
    parser.add_argument("--trumans-load-scene", action="store_true")
    parser.add_argument("--trumans-load-action", action="store_true")
    parser.add_argument("--trumans-no-objects", action="store_true")

    parser.add_argument("--lingo-root", type=Path, default=None)
    parser.add_argument("--lingo-step", type=int, default=3)
    parser.add_argument(
        "--lingo-nb-voxels",
        type=int,
        nargs=3,
        default=[32, 32, 32],
    )
    parser.add_argument(
        "--lingo-mesh-grid",
        type=float,
        nargs=6,
        default=[-0.6, 0.6, 0.1, 1.2, -0.6, 0.6],
    )
    parser.add_argument("--lingo-max-window-size", type=int, default=16)
    parser.add_argument("--lingo-vis", action="store_true")
    parser.add_argument("--lingo-load-scene", action="store_true")
    parser.add_argument("--lingo-load-language", action="store_true")
    parser.add_argument("--lingo-load-pelvis-goal", action="store_true")
    parser.add_argument("--lingo-load-hand-goal", action="store_true")
    parser.add_argument("--lingo-use-pi", action="store_true")
    parser.add_argument("--lingo-start-type", default="stand")
    parser.add_argument("--lingo-test-scene-name", default=None)

    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--sequence-id", default=None)
    parser.add_argument("--sequence-index", type=int, default=None)
    parser.add_argument("--body-height-voxels", type=int, default=85)
    parser.add_argument("--free-threshold", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--summary-json", type=Path, default=None)

    parser.add_argument("--run-blender", action="store_true")
    parser.add_argument("--force-blender-dump", action="store_true")
    parser.add_argument("--blender-no-bbox", action="store_true")
    parser.add_argument("--blender-bin", default="blender")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--break-after-first", action="store_true")
    parser.add_argument(
        "--break-mode",
        choices=["none", "sequence", "scene"],
        default="none",
        help="Breakpoint granularity for dataset preprocessing. "
        "'sequence' breaks after each processed sequence, "
        "'scene' breaks after each finalized scene.",
    )
    return parser.parse_args()


def maybe_break(args, mode: str, first_processed: dict[str, bool]) -> None:
    if args.break_mode != mode:
        return
    if args.break_after_first:
        if first_processed[mode]:
            return
        first_processed[mode] = True
    breakpoint()


def resolve_run_mode(args) -> str:
    if args.sequence_id is not None or args.sequence_index is not None:
        return "sequence"
    if args.scene_id is not None:
        return "scene"
    return "dataset"


def run_iterate_task(args) -> None:
    if args.dataset == "trumans":
        dataset, loader = build_trumans_loader(args)
    else:
        dataset, loader = build_lingo_loader(args)
    print(f"[{args.dataset}] dataset size: {len(dataset)}", flush=True)
    iterate_loader(args.dataset, loader, args.max_batches)


def resolve_trumans_scene_id_for_sequence(sequence_id: str, args) -> str:
    scene_to_sequences = build_trumans_scene_sequence_index(args.trumans_root)
    for scene_id, sequence_ids in scene_to_sequences.items():
        if sequence_id in sequence_ids:
            return scene_id
    raise ValueError(f"Could not resolve scene for sequence {sequence_id}")


def build_trumans_scene_artifacts(scene_id: str, sequence_ids: list[str], sequence_goal_rows: list[dict], scene_objects: list[dict], args) -> tuple[dict, dict[str, str]]:
    interacted_object_ids = sorted({row["object_id"] for row in sequence_goal_rows})
    candidate_dump_path, candidate_objects = finalize_scene_objects_with_bbox(
        scene_id,
        scene_objects,
        interacted_object_ids,
        args,
    )
    object_list, object_id_map = build_scene_object_list(candidate_objects, sequence_goal_rows)
    remapped_goal_rows = remap_goal_rows(sequence_goal_rows, object_id_map)
    scene_record = preprocess_trumans_scene(
        scene_id,
        args,
        sequence_ids=sequence_ids,
        object_list=object_list,
        blend_dump_path=candidate_dump_path,
    )
    return scene_record, object_id_map


def build_lingo_scene_artifacts(scene_id: str, sequence_records: list[dict], args) -> tuple[dict, dict[str, str]]:
    segment_rows = []
    for sequence_record in sequence_records:
        for segment in sequence_record["segment_list"]:
            segment_rows.append(
                {
                    "sequence_id": sequence_record["sequence_id"],
                    "segment_id": segment["segment_id"],
                    "interaction_frame": segment["interaction_frame"],
                    "hand": segment["hand"],
                    "goal_object_name": segment["goal_object_name"],
                    "object_pose": segment["object_pose"],
                    "pelvis_goal": segment["pelvis_goal"],
                    "hand_goal": segment["hand_goal"],
                }
            )
    object_list, segment_to_object_id = cluster_lingo_object_slots(segment_rows)
    for sequence_record in sequence_records:
        for segment in sequence_record["segment_list"]:
            segment["goal_object"] = segment_to_object_id.get((sequence_record["sequence_id"], segment["segment_id"]))
    scene_record = preprocess_lingo_scene(scene_id, args)
    scene_record["sequence_ids"] = [record["sequence_id"] for record in sequence_records]
    scene_record["object_list"] = object_list
    return scene_record, segment_to_object_id


def run_trumans_dataset_task(args) -> None:
    scene_to_sequences = build_trumans_scene_sequence_index(args.trumans_root)
    first_processed = {"sequence": False, "scene": False}
    for scene_id, sequence_ids in scene_to_sequences.items():
        _, scene_objects = load_blender_scene_objects(scene_id, args)
        scene_goal_rows = []
        pending_sequence_records = []
        for sequence_id in sequence_ids:
            sequence_record = preprocess_trumans_sequence(sequence_id, scene_id, scene_objects, args)
            sequence_record.pop("_interacted_object_ids", None)
            scene_goal_rows.extend(sequence_record.pop("_goals", []))
            pending_sequence_records.append(sequence_record)
            maybe_break(args, "sequence", first_processed)

        scene_record, object_id_map = build_trumans_scene_artifacts(
            scene_id,
            sequence_ids,
            scene_goal_rows,
            scene_objects,
            args,
        )
        for sequence_record in pending_sequence_records:
            remap_sequence_record_object_ids(sequence_record, object_id_map)
            refresh_sequence_plot(sequence_record, args)
            strip_plot_only_segment_fields(sequence_record)
            sequence_out_path = scene_sequence_dir(scene_id, args) / f"{sequence_record['sequence_id']}.json"
            write_json(sequence_out_path, sequence_record)

        scene_out_path = scene_output_dir(scene_id, args) / "scene.json"
        write_json(scene_out_path, scene_record)
        maybe_break(args, "scene", first_processed)

    print(args.output_dir / "preprocess" / "trumans" / "scenes")


def run_lingo_dataset_task(args) -> None:
    scene_to_sequences = build_lingo_scene_sequence_index(args)
    first_processed = {"sequence": False, "scene": False}
    for scene_id, sequence_indices in scene_to_sequences.items():
        pending_sequence_records = []
        for sequence_index in sequence_indices:
            sequence_record = preprocess_lingo_sequence(sequence_index, args, scene_id=scene_id)
            pending_sequence_records.append(sequence_record)
            maybe_break(args, "sequence", first_processed)

        scene_record, _ = build_lingo_scene_artifacts(scene_id, pending_sequence_records, args)
        for sequence_record in pending_sequence_records:
            refresh_lingo_sequence_plot(sequence_record, args)
            sequence_out_path = scene_sequence_dir(scene_id, args) / f"{sequence_record['sequence_id']}.json"
            write_json(sequence_out_path, sequence_record)

        scene_out_path = scene_output_dir(scene_id, args) / "scene.json"
        write_json(scene_out_path, scene_record)
        generate_interaction_maps(scene_out_path, args)
        maybe_break(args, "scene", first_processed)

    print(args.output_dir / "preprocess" / "lingo" / "scenes")


def run_scene_task(args) -> dict:
    if not args.scene_id:
        raise ValueError("--scene-id is required")
    if args.dataset == "trumans":
        _, scene_objects = load_blender_scene_objects(args.scene_id, args)
        return preprocess_trumans_scene(
            args.scene_id,
            args,
            sequence_ids=[],
            object_list=scene_objects,
            blend_dump_path=cache_scene_dir(args.scene_id, args) / "transforms.json",
        )
    scene_to_sequences = build_lingo_scene_sequence_index(args)
    sequence_indices = scene_to_sequences.get(args.scene_id, [])
    sequence_records = [preprocess_lingo_sequence(index, args, scene_id=args.scene_id) for index in sequence_indices]
    scene_record, _ = build_lingo_scene_artifacts(args.scene_id, sequence_records, args)
    for sequence_record in sequence_records:
        refresh_lingo_sequence_plot(sequence_record, args)
        sequence_out_path = scene_sequence_dir(args.scene_id, args) / f"{sequence_record['sequence_id']}.json"
        write_json(sequence_out_path, sequence_record)
    scene_out_path = scene_output_dir(args.scene_id, args) / "scene.json"
    write_json(scene_out_path, scene_record)
    generate_interaction_maps(scene_out_path, args)
    if args.break_mode == "scene":
        breakpoint()
    return scene_record


def run_sequence_task(args) -> dict:
    if args.dataset == "trumans":
        if not args.sequence_id:
            raise ValueError("--sequence-id is required for TRUMANS")
        scene_id = resolve_trumans_scene_id_for_sequence(args.sequence_id, args)
        _, scene_objects = load_blender_scene_objects(scene_id, args)
        return preprocess_trumans_sequence(args.sequence_id, scene_id, scene_objects, args)
    if args.sequence_index is None:
        raise ValueError("--sequence-index is required for LINGO")
    return preprocess_lingo_sequence(args.sequence_index, args)


def attach_scene_to_sequence_result(result: dict, args) -> dict:
    if args.dataset == "trumans":
        sequence_result = result
        sequence_result.pop("_interacted_object_ids", None)
        sequence_goal_rows = sequence_result.pop("_goals", [])
        scene_id = sequence_result["scene_id"]
        scene_objects = load_blender_scene_objects(scene_id, args)[1]
        scene_record, object_id_map = build_trumans_scene_artifacts(
            scene_id,
            [sequence_result["sequence_id"]],
            sequence_goal_rows,
            scene_objects,
            args,
        )
        remap_sequence_record_object_ids(sequence_result, object_id_map)
        refresh_sequence_plot(sequence_result, args)
        strip_plot_only_segment_fields(sequence_result)
        sequence_out_path = scene_sequence_dir(scene_id, args) / f"{sequence_result['sequence_id']}.json"
        write_json(sequence_out_path, sequence_result)
        scene_out_path = scene_output_dir(scene_id, args) / "scene.json"
        write_json(scene_out_path, scene_record)
        return {"sequence": sequence_result, "scene_path": to_repo_relative(scene_out_path)}

    sequence_result = result
    scene_id = sequence_result["scene_id"]
    scene_record, _ = build_lingo_scene_artifacts(scene_id, [sequence_result], args)
    refresh_lingo_sequence_plot(sequence_result, args)
    sequence_out_path = scene_sequence_dir(scene_id, args) / f"{sequence_result['sequence_id']}.json"
    write_json(sequence_out_path, sequence_result)
    scene_out_path = scene_output_dir(scene_id, args) / "scene.json"
    write_json(scene_out_path, scene_record)
    generate_interaction_maps(scene_out_path, args)
    return {"sequence": sequence_result, "scene_path": to_repo_relative(scene_out_path)}


def resolve_output_path(args, mode: str, result: dict | None = None) -> Path:
    if args.summary_json is not None:
        return args.summary_json
    if mode == "scene":
        return scene_output_dir(args.scene_id, args) / "_scene_preview.json"
    if args.dataset == "trumans":
        scene_id = None
        if result is not None and "sequence" in result:
            scene_id = result["sequence"]["scene_id"]
        elif result is not None and "scene_id" in result:
            scene_id = result["scene_id"]
        else:
            scene_id = resolve_trumans_scene_id_for_sequence(args.sequence_id, args)
        return scene_sequence_dir(scene_id, args) / f"{args.sequence_id}.bundle.json"
    scene_id = None
    if result is not None and "sequence" in result:
        scene_id = result["sequence"]["scene_id"]
    elif result is not None and "scene_id" in result:
        scene_id = result["scene_id"]
    if scene_id is None and args.scene_id is not None:
        scene_id = args.scene_id
    if scene_id is None:
        raise ValueError("Could not resolve LINGO scene output path.")
    sequence_id = None
    if result is not None and "sequence" in result:
        sequence_id = result["sequence"]["sequence_id"]
    elif result is not None and "sequence_id" in result:
        sequence_id = result["sequence_id"]
    else:
        sequence_id = f"lingo_{args.sequence_index:06d}"
    return scene_sequence_dir(scene_id, args) / f"{sequence_id}.bundle.json"


def main():
    args = parse_args()
    mode = resolve_run_mode(args)

    if mode == "dataset":
        if args.dataset == "trumans":
            run_trumans_dataset_task(args)
        else:
            run_lingo_dataset_task(args)
        return

    result = run_scene_task(args) if mode == "scene" else run_sequence_task(args)
    if mode == "sequence":
        result = attach_scene_to_sequence_result(result, args)

    out_path = resolve_output_path(args, mode, result)
    write_json(out_path, result)
    print(out_path)


if __name__ == "__main__":
    main()
