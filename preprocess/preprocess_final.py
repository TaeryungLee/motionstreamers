from __future__ import annotations

import argparse
import json
import shutil
import pickle
import re
import subprocess
import tempfile
import multiprocessing as mp
import signal
import random
from contextlib import nullcontext
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BLENDER_BIN_DEFAULT = PROJECT_ROOT / "tools" / "blender" / "blender"
BLENDER_TIMEOUT_SEC = 15 * 60
PLOT_TIMEOUT_SEC = 5 * 60
TRUMANS_ROOT_DEFAULT = PROJECT_ROOT / "data" / "raw" / "trumans"
LINGO_ROOT_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw" / "lingo" / "dataset",
    PROJECT_ROOT / "lingo-release" / "dataset",
    Path("/mnt/hdd1/data/lingo_dataset/dataset"),
]

TRUMANS_GRID_META = {
    "x_min": -3.0,
    "x_max": 3.0,
    "y_min": 0.0,
    "y_max": 2.0,
    "z_min": -4.0,
    "z_max": 4.0,
    "x_res": 300,
    "y_res": 100,
    "z_res": 400,
}
LINGO_TRAIN_GRID_META = {
    "x_min": -3.0,
    "x_max": 3.0,
    "y_min": 0.0,
    "y_max": 2.0,
    "z_min": -4.0,
    "z_max": 4.0,
    "x_res": 300,
    "y_res": 100,
    "z_res": 400,
}
LINGO_TEST_GRID_META = {
    "x_min": -4.0,
    "x_max": 4.0,
    "y_min": 0.0,
    "y_max": 2.0,
    "z_min": -6.0,
    "z_max": 6.0,
    "x_res": 400,
    "y_res": 100,
    "z_res": 600,
}

TRUMANS_LEFT_HAND_IDX = 22
TRUMANS_RIGHT_HAND_IDX = 23
LINGO_LEFT_HAND_IDX = 24
LINGO_RIGHT_HAND_IDX = 26

BODY_OBJECT_NEGATIVE_KEYWORDS = (
    "bottle",
    "cup",
    "phone",
    "pen",
    "mouse",
    "handbag",
    "bag",
    "vase",
    "book",
    "keyboard",
    "laptop",
    "camera",
    "fork",
    "apple",
    "guitar",
    "gamepad",
    "toothbrush",
)

OBJECT_NAME_RULES = (
    ("office chair", "chair"),
    ("cupboard", "cabinet"),
    ("chair", "chair"),
    ("seat", "chair"),
    ("stool", "chair"),
    ("bench", "chair"),
    ("sofa", "sofa"),
    ("couch", "sofa"),
    ("bed", "bed"),
    ("cup", "cup"),
    ("water bottle", "bottle"),
    ("bottle", "bottle"),
    ("book", "book"),
    ("phone", "phone"),
    ("keyboard", "keyboard"),
    ("mouse", "mouse"),
    ("laptop", "laptop"),
    ("pen", "pen"),
    ("vase", "vase"),
    ("cabinet", "cabinet"),
    ("drawer", "drawer"),
    ("refrigerator", "refrigerator"),
    ("fridge", "refrigerator"),
    ("microwave", "microwave"),
    ("oven", "oven"),
    ("door", "door"),
    ("table", "table"),
    ("desk", "desk"),
    ("toilet", "toilet"),
    ("bathtub", "bathtub"),
    ("washbasin", "washbasin"),
    ("sink", "sink"),
    ("blackboard", "blackboard"),
    ("whiteboard", "whiteboard"),
    ("globe", "globe"),
    ("camera", "camera"),
    ("guitar", "guitar"),
    ("drum kit", "drum_kit"),
    ("gamepad", "gamepad"),
    ("baseball bat", "baseball_bat"),
    ("punching bag", "punching_bag"),
    ("toothbrush", "toothbrush"),
    ("desk lamp", "desk_lamp"),
    ("candle", "candle"),
    ("fork", "fork"),
    ("wok", "wok"),
    ("apple", "apple"),
    ("earphone", "earphone"),
    ("handbag", "handbag"),
    ("bag", "bag"),
    ("backpack", "backpack"),
)

MOVABLE_OBJECTS = {
    "cup",
    "bottle",
    "book",
    "phone",
    "mouse",
    "laptop",
    "pen",
    "vase",
    "camera",
    "guitar",
    "gamepad",
    "baseball_bat",
    "toothbrush",
    "candle",
    "fork",
    "wok",
    "apple",
    "earphone",
    "handbag",
    "bag",
    "backpack",
}

GOAL_TYPE_RULES = (
    ("sit down", "sit"),
    ("stand up", "stand"),
    ("lie down", "lie"),
    ("pick up", "pick_up"),
    ("put down", "put_down"),
    ("open", "open"),
    ("close", "close"),
    ("drink from", "drink"),
    ("type on", "type"),
    ("write on", "write"),
    ("read", "read"),
    ("talk on", "talk"),
    ("wash", "wash"),
    ("play", "play"),
    ("walk", "walk"),
    ("turn ", "turn"),
    ("kneel down", "kneel"),
    ("get up from kneeling", "stand"),
    ("squat down", "squat"),
    ("bend forward", "bend"),
    ("straighten up", "straighten"),
    ("maintains sit posture", "sit"),
    ("maintains stand posture", "stand"),
    ("stand still", "stand"),
)

GOAL_TYPE_FALLBACK_RULES = (
    (re.compile(r"\bsit cross-legged\b", re.I), "sit"),
    (re.compile(r"\blie face (?:down|up)\b", re.I), "lie"),
    (re.compile(r"\btake shower\b", re.I), "wash"),
    (re.compile(r"\bget up from lying to sitting\b", re.I), "sit"),
    (re.compile(r"\bmaintains lie\b", re.I), "lie"),
    (re.compile(r"\bmaintains bend posture\b", re.I), "bend"),
    (re.compile(r"\bmaintains kneel posture\b", re.I), "kneel"),
    (re.compile(r"\bmaintains squat posture\b", re.I), "squat"),
    (re.compile(r"\bdrink(?:s)? water\b", re.I), "drink"),
    (re.compile(r"\b(?:left|right) hand closing\b", re.I), "close"),
    (re.compile(r"\b(?:left|right) hand turns? (?:on|off)\b", re.I), "turn"),
    (re.compile(r"\b(?:left|right) hand picks? up\b", re.I), "pick_up"),
    (re.compile(r"\b(?:left|right) hand pick(?:ing)? up\b", re.I), "pick_up"),
    (re.compile(r"\b(?:left|right) hand puts? down\b", re.I), "put_down"),
    (re.compile(r"\b(?:left|right) hand put(?:ting)? down\b", re.I), "put_down"),
    (re.compile(r"\bboth hands typing on\b", re.I), "type"),
    (re.compile(r"\buse .*keyboard\b", re.I), "type"),
    (re.compile(r"\buse .*phone\b", re.I), "operate"),
    (re.compile(r"\buse .*mouse\b", re.I), "operate"),
    (re.compile(r"\buse .*laptop\b", re.I), "operate"),
    (re.compile(r"\buses? the (?:mouse|phone)\b", re.I), "operate"),
    (re.compile(r"\boperate the mouse\b", re.I), "operate"),
    (re.compile(r"\btake the call\b", re.I), "answer"),
    (re.compile(r"\btake the mouse\b", re.I), "operate"),
    (re.compile(r"\bmake a phone call\b", re.I), "call"),
    (re.compile(r"\bput .* hand on\b", re.I), "place_hand"),
    (re.compile(r"\bput .* down\b", re.I), "put_down"),
    (re.compile(r"\bset down\b", re.I), "put_down"),
    (re.compile(r"\bset .* down\b", re.I), "put_down"),
    (re.compile(r"\bwrite\b", re.I), "write"),
    (re.compile(r"\btyp(?:e|ing)\b", re.I), "type"),
    (re.compile(r"\bplace the (?:left|right|both) hand on\b", re.I), "place_hand"),
    (re.compile(r"\bleave the (?:left|right) hand on\b", re.I), "rest"),
    (re.compile(r"\bplace the (?:left|right) leg on\b", re.I), "step"),
    (re.compile(r"\bput both legs on\b", re.I), "step"),
    (re.compile(r"\bswipe\b", re.I), "swipe"),
    (re.compile(r"\bright hand writes\b", re.I), "write"),
    (re.compile(r"\bfiddle with\b", re.I), "fiddle"),
    (re.compile(r"^drawer$", re.I), "open"),
    (re.compile(r"\blisten to\b", re.I), "listen"),
    (re.compile(r"\btoss\b", re.I), "toss"),
    (re.compile(r"\bwave\b", re.I), "wave"),
    (re.compile(r"\bpunch\b", re.I), "punch"),
    (re.compile(r"\bkick\b", re.I), "kick"),
    (re.compile(r"\bcrawl\b", re.I), "crawl"),
    (re.compile(r"\bcrouch(?: down)?\b", re.I), "squat"),
    (re.compile(r"\bstroll around\b", re.I), "walk"),
    (re.compile(r"\bpicks? up\b", re.I), "pick_up"),
    (re.compile(r"\bputs? down\b", re.I), "put_down"),
    (re.compile(r"\bplace(?:s)? .* down\b", re.I), "put_down"),
    (re.compile(r"\bset(?:s)? .* down\b", re.I), "put_down"),
    (re.compile(r"\blower(?:s)?\b", re.I), "lower"),
    (re.compile(r"\blift(?:s)?\b", re.I), "lift"),
    (re.compile(r"\braise(?:s)?\b", re.I), "raise"),
    (re.compile(r"\bmove(?:s)?\b", re.I), "move"),
    (re.compile(r"\bdrag(?:s|ging)?\b", re.I), "drag"),
    (re.compile(r"\bpush(?:es|ing)?\b", re.I), "push"),
    (re.compile(r"\bslide(?:s|ing)?\b", re.I), "slide"),
    (re.compile(r"\bwipe(?:s|ing)?\b", re.I), "wipe"),
    (re.compile(r"\btap(?:s|ping)?\b", re.I), "tap"),
    (re.compile(r"\bclick(?:s|ing)?\b", re.I), "click"),
    (re.compile(r"\bhit(?:s|ting)?\b", re.I), "hit"),
    (re.compile(r"\bhold(?:s|ing)?\b", re.I), "hold"),
    (re.compile(r"\bmake(?:s)? a call\b", re.I), "call"),
    (re.compile(r"\banswer(?:s|ing)?\b", re.I), "answer"),
    (re.compile(r"\bdial(?:s|ing)?\b", re.I), "dial"),
    (re.compile(r"\bpour(?:s|ing)?\b", re.I), "pour"),
    (re.compile(r"\bwater(?:s|ing)?\b", re.I), "water"),
    (re.compile(r"\brest(?:s|ing)?\b", re.I), "rest"),
    (re.compile(r"\bpat(?:s|ting)?\b", re.I), "pat"),
    (re.compile(r"\bstep(?:s|ping)? on\b", re.I), "step"),
    (re.compile(r"\brotate(?:s|ing)?\b", re.I), "rotate"),
    (re.compile(r"\bswing(?:s|ing)?\b", re.I), "swing"),
    (re.compile(r"\bbrush(?:es|ing)?\b", re.I), "brush"),
    (re.compile(r"\bblow(?:s|ing)? out\b", re.I), "blow_out"),
    (re.compile(r"\beat(?:s|ing)?\b", re.I), "eat"),
    (re.compile(r"\btake(?:s)? photo\b", re.I), "take_photo"),
    (re.compile(r"\bshut(?:s)? down\b", re.I), "shut_down"),
    (re.compile(r"\bcreate(?:s|ing)?\b", re.I), "create"),
    (re.compile(r"\bplace(?:s)? (?:both |the |left |right )?hands? on\b", re.I), "place_hand"),
)

TRUMANS_ACTED_PATTERNS = [
    re.compile(r"pick(?:s)? up the ([a-z ]+?)(?= with(?: the)? (?:left|right) hand|$)", re.I),
    re.compile(r"put(?:s)? down the ([a-z ]+?)(?= with(?: the)? (?:left|right) hand|$)", re.I),
    re.compile(r"open(?:s)? the ([a-z ]+?)(?= with(?: the)? (?:left|right) hand|$)", re.I),
    re.compile(r"close(?:s)? the ([a-z ]+?)(?= with(?: the)? (?:left|right) hand|$)", re.I),
    re.compile(r"turn(?:s)? (?:on|off) the ([a-z ]+?)(?= with(?: the)? (?:left|right) hand|$)", re.I),
    re.compile(r"type(?:s)? on the ([a-z ]+?)(?= with(?: both hands|(?: the)? (?:left|right) hand)| while |$)", re.I),
    re.compile(r"wipe(?:s)? the ([a-z ]+?)(?= with(?: the)? (?:left|right) hand|$)", re.I),
    re.compile(r"water(?:s)? the ([a-z ]+?)(?= with(?: the)? (?:left|right) hand|$)", re.I),
    re.compile(r"shut down the ([a-z ]+?)(?= with(?: the)? (?:left|right) hand|$)", re.I),
    re.compile(r"hit(?:s)? the ([a-z ]+?)(?= with(?: the)? (?:left|right) hand|$)", re.I),
    re.compile(r"put (?:the |left |right )?hand on the ([a-z ]+?)(?= with|$)", re.I),
    re.compile(r"place (?:both |the |left |right )?hands? on the ([a-z ]+?)(?= with|$)", re.I),
    re.compile(r"drink(?:s)? from the ([a-z ]+?)(?= with(?: the)? (?:left|right) hand|$)", re.I),
    re.compile(r"write(?:s)? on the ([a-z ]+?)(?= with(?: the)? (?:left|right) hand|$)", re.I),
]

LINGO_ROLE_PATTERNS = [
    (re.compile(r"pick up ([a-z ]+?) with", re.I), "acted"),
    (re.compile(r"put down ([a-z ]+?) in .* on ([a-z ]+?)(?= with(?: the)? (?:left|right) hand| while |$)", re.I), "acted_support"),
    (re.compile(r"put down ([a-z ]+?) on ([a-z ]+?)(?= with(?: the)? (?:left|right) hand| while |$)", re.I), "acted_support"),
    (re.compile(r"drink from ([a-z ]+?)(?= with(?: the)? (?:left|right) hand|$)", re.I), "acted"),
    (re.compile(r"read ([a-z ]+?) with", re.I), "acted"),
    (re.compile(r"talk on ([a-z ]+?) with", re.I), "acted"),
    (re.compile(r"type on ([a-z ]+?)(?= with(?: both hands|(?: the)? (?:left|right) hand)| while |$)", re.I), "support"),
    (re.compile(r"write on ([a-z ]+?)(?= with(?: both hands|(?: the)? (?:left|right) hand)| while |$)", re.I), "support"),
    (re.compile(r"wash hands at ([a-z ]+?)(?= while |$)", re.I), "support"),
    (re.compile(r"sit down on ([a-z ]+?)(?= while |$)", re.I), "support"),
    (re.compile(r"stand up from ([a-z ]+?)(?= while |$)", re.I), "support"),
    (re.compile(r"lie down on ([a-z ]+?)(?= while |$)", re.I), "support"),
    (re.compile(r"sit down in front of ([a-z ]+?)(?= while |$)", re.I), "support"),
]

ROLE_NAME_SUFFIX_PATTERNS = [
    re.compile(r"\bwith(?: the)? (?:left|right) hand\b", re.I),
    re.compile(r"\bwith both hands\b", re.I),
    re.compile(r"\bwhile in sitting position\b", re.I),
    re.compile(r"\bwhile seated\b", re.I),
    re.compile(r"\bin sitting position\b", re.I),
]

ROLE_NAME_INVALID_VALUES = {
    "kneeling",
    "squatting",
    "standing",
    "standing position",
    "sitting position",
}


def to_repo_relative(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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


def save_mask(mask: np.ndarray, path: Path, scale: int = 4) -> None:
    image = Image.fromarray((mask.astype(np.uint8) * 255).T, mode="L")
    if scale > 1:
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize((image.width * scale, image.height * scale), resample=resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def preprocess_root(args) -> Path:
    return args.output_dir / "preprocessed" / args.dataset


def scene_dir(scene_id: str, args) -> Path:
    return preprocess_root(args) / "scenes" / scene_id


def sequence_dir(scene_id: str, args) -> Path:
    return scene_dir(scene_id, args) / "sequences"


def plot_dir(scene_id: str, sequence_id: str, dataset: str, args) -> Path:
    prefix = "trumans_sequence_actions" if dataset == "trumans" else "lingo_sequence_actions"
    return scene_dir(scene_id, args) / "plots" / f"{prefix}_{sequence_id}"


def cache_dir(scene_id: str, args) -> Path:
    return args.output_dir / "preprocessed_cache" / args.dataset / scene_id


def resolve_lingo_scene_split(scene_id: str, args) -> str:
    root = resolve_lingo_root(args)
    if (root / "Scene" / f"{scene_id}.npy").exists():
        return "train"
    if (root / "Scene_vis" / f"{scene_id}.npy").exists():
        return "test"
    return "train"


def reset_scene_outputs(scene_id: str, args) -> None:
    for path in [sequence_dir(scene_id, args), scene_dir(scene_id, args) / "plots"]:
        if path.exists():
            shutil.rmtree(path)


def build_scene_split_map(args) -> dict[str, str]:
    scenes_root = preprocess_root(args) / "scenes"
    scene_ids = []
    if scenes_root.exists():
        for scene_json in sorted(scenes_root.glob("*/scene.json")):
            scene_record = json.loads(scene_json.read_text())
            if args.dataset == "lingo" and not scene_record.get("sequence_ids"):
                continue
            scene_ids.append(scene_record["scene_id"])

    if not scene_ids:
        if args.dataset == "trumans":
            root = args.trumans_root
            train_dir = root / "Scene"
            scene_ids = sorted(path.stem for path in train_dir.glob("*.npy")) if train_dir.exists() else []
        else:
            root = resolve_lingo_root(args)
            train_dir = root / "Scene"
            scene_ids = sorted(path.stem for path in train_dir.glob("*.npy")) if train_dir.exists() else []

    if args.dataset == "lingo":
        mirror_scene_ids = sorted(scene_id for scene_id in scene_ids if scene_id.endswith("_mirror"))
        non_mirror_scene_ids = sorted(scene_id for scene_id in scene_ids if not scene_id.endswith("_mirror"))

        rng = random.Random(42)
        rng.shuffle(non_mirror_scene_ids)
        test_count = max(1, round(len(non_mirror_scene_ids) * 0.2)) if non_mirror_scene_ids else 0
        test_ids = set(non_mirror_scene_ids[:test_count])

        split_map = {scene_id: "train" for scene_id in mirror_scene_ids}
        for scene_id in sorted(non_mirror_scene_ids):
            split_map[scene_id] = "test" if scene_id in test_ids else "train"
        return split_map

    rng = random.Random(42)
    rng.shuffle(scene_ids)
    test_count = max(1, round(len(scene_ids) * 0.2)) if scene_ids else 0
    test_ids = set(scene_ids[:test_count])
    return {scene_id: ("test" if scene_id in test_ids else "train") for scene_id in sorted(scene_ids)}


def write_split_file(args) -> None:
    split_map = build_scene_split_map(args)
    if not split_map:
        return
    out_root = preprocess_root(args)
    out_root.mkdir(parents=True, exist_ok=True)
    legacy_path = out_root / "split.txt"
    if legacy_path.exists():
        legacy_path.unlink()
    split_to_scenes: dict[str, list[str]] = {"train": [], "test": []}
    for split in ("train", "test"):
        split_to_scenes[split] = sorted(k for k, v in split_map.items() if v == split)
        out_path = out_root / f"{split}.txt"
        out_path.write_text("\n".join(split_to_scenes[split]) + "\n")


def progress(iterable, desc: str, leave: bool = True):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=leave)


class SceneTimeoutError(RuntimeError):
    pass


def _timeout_handler(signum, frame):
    raise SceneTimeoutError("scene processing timed out")


def with_scene_timeout(seconds: int):
    if seconds <= 0:
        return nullcontext()
    return _alarm_context(seconds)


class _alarm_context:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self._prev_handler = None

    def __enter__(self):
        self._prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        signal.alarm(0)
        if self._prev_handler is not None:
            signal.signal(signal.SIGALRM, self._prev_handler)
        return False


def resolve_lingo_root(args) -> Path:
    if args.lingo_root is not None:
        return args.lingo_root
    for candidate in LINGO_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not resolve LINGO dataset root.")


def compute_walkable_map(
    scene_occ: np.ndarray,
    body_height_voxels: int = 85,
    free_threshold: float = 0.95,
    use_floor: bool = True,
) -> np.ndarray:
    body_band = scene_occ[:, 1 : body_height_voxels + 1, :]
    free_ratio = 1.0 - body_band.mean(axis=1)
    if not use_floor:
        return free_ratio >= free_threshold
    floor = scene_occ[:, 0, :]
    return floor & (free_ratio >= free_threshold)


def majority_filter(mask: np.ndarray, kernel: int = 5, threshold: int | None = None) -> np.ndarray:
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


def save_clearance(scene_id: str, scene_occ: np.ndarray, dataset: str, args) -> tuple[Path, Path]:
    walkable = compute_walkable_map(scene_occ, use_floor=False)
    if dataset == "lingo":
        # LINGO occupancy is noisy; floor mask is unreliable. Use free space only.
        body_band = scene_occ[:, 1:86, :]
        free_ratio = 1.0 - body_band.mean(axis=1)
        walkable = free_ratio >= 0.95
        walkable = majority_filter(walkable, kernel=5)
    out_dir = scene_dir(scene_id, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / "clearance_map.npy"
    vis_path = out_dir / "clearance_map.png"
    np.save(npy_path, walkable.astype(np.uint8))
    save_mask(walkable, vis_path, scale=4)
    return npy_path, vis_path


def canonicalize_name(name: str | None) -> str | None:
    if name is None:
        return None
    lower = name.lower().strip().replace("_", " ").replace("-", " ")
    for pattern, canonical in sorted(OBJECT_NAME_RULES, key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])", lower):
            return canonical
    return " ".join(lower.split()) if lower else None


def normalize_extracted_role_name(name: str | None) -> str | None:
    if name is None:
        return None
    lower = name.lower().strip().replace("_", " ").replace("-", " ")
    lower = re.sub(r"^(?:the|a|an)\s+", "", lower)
    for pattern in ROLE_NAME_SUFFIX_PATTERNS:
        lower = pattern.sub("", lower)
    lower = " ".join(lower.split())
    if not lower or lower in ROLE_NAME_INVALID_VALUES:
        return None
    return canonicalize_name(lower)


def infer_goal_type(text: str) -> str:
    lower = text.lower()
    for pattern, goal_type in GOAL_TYPE_RULES:
        if pattern in lower:
            return goal_type
    for pattern, goal_type in GOAL_TYPE_FALLBACK_RULES:
        if pattern.search(lower):
            return goal_type
    return "interact"


def infer_active_hand(text: str, left_inter: int | None = None, right_inter: int | None = None) -> str:
    lower = text.lower()
    if "both hands" in lower:
        return "both"
    if "left hand" in lower:
        return "left"
    if "right hand" in lower:
        return "right"
    if left_inter is not None and right_inter is not None:
        if left_inter != -1 and right_inter != -1:
            return "both"
        if left_inter != -1:
            return "left"
        if right_inter != -1:
            return "right"
    return "none"


def infer_movable(object_name: str | None) -> bool:
    return object_name in MOVABLE_OBJECTS


def compute_yaw_from_track(track: np.ndarray, frame_idx: int) -> float | None:
    if len(track) == 0:
        return None
    i0 = max(0, min(len(track) - 1, frame_idx))
    i1 = max(0, min(len(track) - 1, frame_idx + 1))
    delta = np.asarray(track[i1], dtype=np.float32) - np.asarray(track[i0], dtype=np.float32)
    if float(np.linalg.norm(delta[[0, 2]])) < 1e-6:
        return None
    return float(np.arctan2(delta[2], delta[0]))


def world_to_motion(location: list[float] | np.ndarray) -> np.ndarray:
    loc = np.asarray(location, dtype=np.float32)
    return np.array([loc[0], loc[2], -loc[1]], dtype=np.float32)


def point_to_aabb_distance(point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    lo = np.minimum(bbox_min, bbox_max)
    hi = np.maximum(bbox_min, bbox_max)
    clamped = np.minimum(np.maximum(point, lo), hi)
    return float(np.linalg.norm(point - clamped))


def parse_trumans_events(action_file: Path) -> list[dict]:
    events = []
    if not action_file.exists():
        return events
    for event_id, line in enumerate(action_file.read_text().splitlines()):
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        start, end, text = parts
        events.append({"segment_id": event_id, "start": int(start), "end": int(end), "text": text.strip()})
    return events


def trumans_base_sequence_id(sequence_id: str) -> str:
    return sequence_id.split("_augment", 1)[0]


def build_trumans_scene_index(root: Path) -> dict[str, list[str]]:
    seg_name = np.load(root / "seg_name.npy", allow_pickle=True)
    scene_flag = np.load(root / "scene_flag.npy", allow_pickle=True)
    scene_list = np.load(root / "scene_list.npy", allow_pickle=True)
    scene_to_sequences: dict[str, list[str]] = {}
    seen = set()
    for idx, seq in enumerate(seg_name.tolist()):
        if seq in seen:
            continue
        seen.add(seq)
        scene_id = str(scene_list[int(scene_flag[idx])])
        scene_to_sequences.setdefault(scene_id, []).append(str(seq))
    return scene_to_sequences


def resolve_trumans_scene_for_sequence(sequence_id: str, root: Path) -> str:
    scene_index = build_trumans_scene_index(root)
    for scene_id, sequence_ids in scene_index.items():
        if sequence_id in sequence_ids:
            return scene_id
    raise ValueError(f"Could not resolve scene for {sequence_id}")


def run_blender_dump(blend_file: Path, output_json: Path, args, include_names: list[str] | None = None, no_bbox: bool = True) -> None:
    cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "extract_blend_objects.py"),
        str(blend_file),
        "--output",
        str(output_json),
        "--blender-bin",
        args.blender_bin,
        "--noaudio",
        "--factory-startup",
    ]
    if no_bbox:
        cmd.append("--no-bbox")
    for name in sorted(set(include_names or [])):
        cmd.extend(["--include-name", name])
    subprocess.run(cmd, check=True, timeout=BLENDER_TIMEOUT_SEC)


def blender_record_to_object(record: dict) -> dict:
    obj = {
        "object_id": record["name"],
        "object_name": canonicalize_name(record.get("data_name") or record["name"]),
        "initial_location": world_to_motion(record["world_location"]),
        "initial_rotation": [
            record["world_rotation_euler"][0],
            record["world_rotation_euler"][2],
            -record["world_rotation_euler"][1],
        ],
        "source": "blend_dump",
    }
    if "bbox_min" in record and "bbox_max" in record:
        obj["bbox"] = {
            "min": world_to_motion(record["bbox_min"]),
            "max": world_to_motion(record["bbox_max"]),
        }
    return obj


def load_blender_scene_objects(scene_id: str, args, include_names: list[str] | None = None, no_bbox: bool = True) -> list[dict]:
    recording_dir = args.trumans_root / "Recordings_blend" / scene_id
    blend_file = recording_dir / f"{scene_id}.blend"
    if not blend_file.exists():
        return []
    cdir = cache_dir(scene_id, args)
    cdir.mkdir(parents=True, exist_ok=True)
    cache_name = "transforms.json" if no_bbox else "bbox.json"
    out_json = cdir / cache_name
    if not out_json.exists() or args.force_blender_dump:
        run_blender_dump(blend_file, out_json, args, include_names=include_names, no_bbox=no_bbox)
    payload = json.loads(out_json.read_text())
    objects = []
    for record in payload.get("objects", []):
        if record.get("type") not in {"MESH", "EMPTY"}:
            continue
        objects.append(blender_record_to_object(record))
    return objects


def load_object_all_pose(root: Path, sequence_id: str) -> dict:
    source_id = trumans_base_sequence_id(sequence_id)
    path = root / "Object_all" / "Object_pose" / f"{source_id}.npy"
    if not path.exists():
        return {}
    return np.load(path, allow_pickle=True).item()


def nearest_object_all(object_pose_dict: dict, local_frame: int, query_xyz: np.ndarray) -> dict | None:
    best = None
    for name, entry in object_pose_dict.items():
        if local_frame >= len(entry.get("location", [])):
            continue
        location = np.asarray(entry["location"][local_frame], dtype=np.float32)
        distance = float(np.linalg.norm(query_xyz - location))
        candidate = {
            "object_id": name,
            "object_name": canonicalize_name(name),
            "location": location,
            "rotation": (
                np.asarray(entry["rotation"][local_frame], dtype=np.float32)
                if "rotation" in entry and local_frame < len(entry["rotation"])
                else None
            ),
            "distance": distance,
            "source": "Object_all",
        }
        if best is None or distance < best["distance"]:
            best = candidate
    return best


TRUMANS_SUPPORT_TRANSFER_GOAL_TYPES = {"pick_up", "put_down", "lift", "lower", "raise", "rotate"}
TRUMANS_BODY_ONLY_GOAL_TYPES = {"stand", "walk", "bend", "straighten", "kneel", "squat"}
TRUMANS_TEXT_GROUNDED_FIXED_GOAL_TYPES = {"open", "close", "turn", "type", "wipe", "water", "shut_down", "hit", "place_hand"}


def trumans_event_hand_tracks(active_hand: str, left_track: np.ndarray, right_track: np.ndarray) -> list[tuple[str, np.ndarray]]:
    if active_hand == "left":
        return [("left", left_track)]
    if active_hand == "right":
        return [("right", right_track)]
    if active_hand == "both":
        return [("left", left_track), ("right", right_track)]
    return [("left", left_track), ("right", right_track)]


def summarize_object_hand_relation(
    object_pose_dict: dict,
    object_name: str,
    start: int,
    end: int,
    query_tracks: list[tuple[str, np.ndarray]],
) -> dict | None:
    entry = object_pose_dict.get(object_name)
    if entry is None:
        return None
    frame_ids = list(range(start, end + 1))
    if not frame_ids:
        return None

    n = len(frame_ids)
    dist = np.full(n, np.inf, dtype=np.float32)
    loc_valid = np.zeros(n, dtype=bool)
    locs = np.zeros((n, 3), dtype=np.float32)
    rot_valid = np.zeros(n, dtype=bool)
    rots = np.zeros((n, 3), dtype=np.float32)

    locations = entry.get("location", [])
    rotations = entry.get("rotation", [])

    for i, local_frame in enumerate(frame_ids):
        if local_frame < len(locations):
            loc = np.asarray(locations[local_frame], dtype=np.float32)
            locs[i] = loc
            loc_valid[i] = np.isfinite(loc).all()
            if loc_valid[i]:
                dist[i] = min(float(np.linalg.norm(track[i] - loc)) for _, track in query_tracks)
        if local_frame < len(rotations):
            rot = np.asarray(rotations[local_frame], dtype=np.float32)
            rots[i] = rot
            rot_valid[i] = np.isfinite(rot).all()

    if not np.isfinite(dist).any():
        return None

    speed = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        if loc_valid[i - 1] and loc_valid[i]:
            speed[i] = float(np.linalg.norm(locs[i] - locs[i - 1]))

    rot_speed = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        if rot_valid[i - 1] and rot_valid[i]:
            rot_speed[i] = float(np.linalg.norm(rots[i] - rots[i - 1]))

    best_idx = int(np.argmin(dist))
    d_min = float(dist[best_idx])
    contact_threshold = min(1.5 * d_min, 0.10)
    if not np.isfinite(contact_threshold):
        contact_threshold = 0.10

    return {
        "object_id": object_name,
        "object_name": canonicalize_name(object_name),
        "frame_ids": frame_ids,
        "distance": dist,
        "speed": speed,
        "rot_speed": rot_speed,
        "locations": locs,
        "rotations": rots,
        "best_idx": best_idx,
        "d_min": d_min,
        "contact_threshold": contact_threshold,
        "source": "Object_all",
    }


def trumans_is_body_only_turn(text: str, active_hand: str) -> bool:
    lower = text.lower()
    if active_hand in {"left", "right", "both"}:
        return False
    if re.search(r"\bturn(?:s)? (?:on|off)\b", lower):
        return False
    return True


def summarize_scene_object_hand_relation(
    scene_objects: list[dict],
    start: int,
    end: int,
    query_tracks: list[tuple[str, np.ndarray]],
    preferred_object_name: str | None = None,
) -> list[dict]:
    frame_ids = list(range(start, end + 1))
    if not frame_ids:
        return []

    preferred_canonical = canonicalize_name(preferred_object_name) if preferred_object_name else None
    preferred: list[dict] = []
    fallback: list[dict] = []

    for obj in scene_objects:
        joined = " ".join(str(obj.get(k, "")) for k in ("object_id", "object_name")).lower()
        if any(keyword in joined for keyword in BODY_OBJECT_NEGATIVE_KEYWORDS):
            continue

        bbox_min = bbox_max = None
        if obj.get("bbox") is not None:
            bbox_min = np.asarray(obj["bbox"]["min"], dtype=np.float32)
            bbox_max = np.asarray(obj["bbox"]["max"], dtype=np.float32)
        obj_loc = np.asarray(obj["initial_location"], dtype=np.float32)

        n = len(frame_ids)
        dist = np.full(n, np.inf, dtype=np.float32)
        locs = np.repeat(obj_loc[None, :], n, axis=0)
        rots = np.zeros((n, 3), dtype=np.float32)
        if obj.get("initial_rotation") is not None:
            rots[:] = np.asarray(obj["initial_rotation"], dtype=np.float32)

        for i, _local_frame in enumerate(frame_ids):
            if bbox_min is not None and bbox_max is not None:
                dist[i] = min(point_to_aabb_distance(track[i], bbox_min, bbox_max) for _, track in query_tracks)
            else:
                dist[i] = min(float(np.linalg.norm(track[i] - obj_loc)) for _, track in query_tracks)

        if not np.isfinite(dist).any():
            continue

        best_idx = int(np.argmin(dist))
        d_min = float(dist[best_idx])
        contact_threshold = min(1.5 * d_min, 0.10)
        if not np.isfinite(contact_threshold):
            contact_threshold = 0.10

        summary = {
            "object_id": obj["object_id"],
            "object_name": canonicalize_name(obj["object_name"]),
            "frame_ids": frame_ids,
            "distance": dist,
            "locations": locs,
            "rotations": rots,
            "best_idx": best_idx,
            "d_min": d_min,
            "contact_threshold": contact_threshold,
            "source": "blend_dump",
        }
        if preferred_canonical is not None and summary["object_name"] == preferred_canonical:
            preferred.append(summary)
        else:
            fallback.append(summary)

    return preferred if preferred else fallback


def first_contact_index(distance: np.ndarray, threshold: float) -> int | None:
    hits = np.where(distance <= threshold)[0]
    if hits.size == 0:
        return None
    return int(hits[0])


def first_release_index(
    distance: np.ndarray,
    threshold: float,
) -> int | None:
    contact_idx = first_contact_index(distance, threshold)
    if contact_idx is None:
        return None
    for i in range(contact_idx + 1, len(distance)):
        if distance[i] > threshold and distance[i - 1] <= threshold:
            return int(i)
    if distance[-1] <= threshold:
        return len(distance) - 1
    return None


def select_hand_interaction_trumans(
    goal_type: str,
    start: int,
    end: int,
    query_tracks: list[tuple[str, np.ndarray]],
    object_pose_dict: dict,
    preferred_object_name: str | None = None,
) -> dict | None:
    candidates: list[dict] = []
    for object_name in object_pose_dict.keys():
        summary = summarize_object_hand_relation(object_pose_dict, object_name, start, end, query_tracks)
        if summary is not None:
            candidates.append(summary)
    if not candidates:
        return None

    if preferred_object_name is not None:
        preferred_canonical = canonicalize_name(preferred_object_name)
        preferred_candidates = [item for item in candidates if item["object_name"] == preferred_canonical]
        if preferred_candidates:
            candidates = preferred_candidates

    chosen = min(candidates, key=lambda item: item["d_min"])
    dist = chosen["distance"]
    threshold = chosen["contact_threshold"]

    if goal_type in {"pick_up", "lift", "raise", "rotate"}:
        idx = first_contact_index(dist, threshold)
    elif goal_type in {"put_down", "lower"}:
        idx = first_release_index(dist, threshold)
    else:
        idx = first_contact_index(dist, threshold)

    if idx is None:
        idx = int(chosen["best_idx"])

    local_frame = int(chosen["frame_ids"][idx])
    return {
        "interaction_frame": local_frame,
        "object_id": chosen["object_id"],
        "object_name": chosen["object_name"],
        "object_location": np.asarray(chosen["locations"][idx], dtype=np.float32),
        "object_rotation": np.asarray(chosen["rotations"][idx], dtype=np.float32),
        "distance": float(chosen["distance"][idx]),
        "source": chosen["source"],
    }


def select_fixed_interaction_trumans(
    goal_type: str,
    start: int,
    end: int,
    query_tracks: list[tuple[str, np.ndarray]],
    scene_objects: list[dict],
    preferred_object_name: str | None = None,
) -> dict | None:
    candidates = summarize_scene_object_hand_relation(
        scene_objects,
        start,
        end,
        query_tracks,
        preferred_object_name=preferred_object_name,
    )
    if not candidates:
        return None

    chosen = min(candidates, key=lambda item: item["d_min"])
    dist = chosen["distance"]
    threshold = chosen["contact_threshold"]
    idx = first_contact_index(dist, threshold)
    if idx is None:
        idx = int(chosen["best_idx"])

    local_frame = int(chosen["frame_ids"][idx])
    return {
        "interaction_frame": local_frame,
        "object_id": chosen["object_id"],
        "object_name": chosen["object_name"],
        "object_location": np.asarray(chosen["locations"][idx], dtype=np.float32),
        "object_rotation": np.asarray(chosen["rotations"][idx], dtype=np.float32),
        "distance": float(chosen["distance"][idx]),
        "source": chosen["source"],
    }


def select_body_object_trumans(text: str, start: int, end: int, pelvis_track: np.ndarray, scene_objects: list[dict]) -> dict | None:
    del text
    best = None
    for frame_offset, local_frame in enumerate(range(start, end + 1)):
        query = pelvis_track[frame_offset]
        for obj in scene_objects:
            joined = " ".join(str(obj.get(k, "")) for k in ("object_id", "object_name")).lower()
            if any(keyword in joined for keyword in BODY_OBJECT_NEGATIVE_KEYWORDS):
                continue
            if obj.get("bbox") is not None:
                bbox_min = np.asarray(obj["bbox"]["min"], dtype=np.float32)
                bbox_max = np.asarray(obj["bbox"]["max"], dtype=np.float32)
                distance = point_to_aabb_distance(query, bbox_min, bbox_max)
            else:
                distance = float(np.linalg.norm(query - np.asarray(obj["initial_location"], dtype=np.float32)))
            candidate = {
                "interaction_frame": local_frame,
                "object_id": obj["object_id"],
                "object_name": obj["object_name"],
                "object_location": np.asarray(obj["initial_location"], dtype=np.float32),
                "object_rotation": (
                    np.asarray(obj["initial_rotation"], dtype=np.float32)
                    if obj.get("initial_rotation") is not None
                    else None
                ),
                "distance": distance,
                "source": "blend_dump",
            }
            if best is None or distance < best["distance"]:
                best = candidate
    return best


def parse_trumans_object_roles(text: str, goal_type: str, grounded_name: str | None) -> tuple[str | None, str | None]:
    lower = text.lower()
    acted_on = None
    for pattern in TRUMANS_ACTED_PATTERNS:
        match = pattern.search(lower)
        if match:
            acted_on = normalize_extracted_role_name(match.group(1))
            break
    if acted_on is None and goal_type in TRUMANS_TEXT_GROUNDED_FIXED_GOAL_TYPES:
        acted_on = grounded_name
    support = None
    if goal_type in {"sit", "lie"}:
        support = "chair"
    return acted_on, support


def sequence_object_entry(
    object_id: str,
    object_name: str | None,
    movable: bool,
    initial_location: np.ndarray | None,
    source: str | None = None,
) -> dict:
    return {
        "object_id": object_id,
        "object_name": object_name,
        "movable": movable,
        "initial_location": initial_location,
        "source": source,
    }


def build_trumans_sequence(sequence_id: str, scene_id: str, args) -> dict:
    root = args.trumans_root
    seg_name = np.load(root / "seg_name.npy", allow_pickle=True)
    joints = np.load(root / "human_joints.npy", mmap_mode="r")
    global_orient = np.load(root / "human_orient.npy", mmap_mode="r")
    idx = np.where(seg_name == sequence_id)[0]
    if len(idx) == 0:
        raise ValueError(f"Sequence not found: {sequence_id}")

    object_pose_dict = load_object_all_pose(root, sequence_id)
    source_id = trumans_base_sequence_id(sequence_id)
    events = parse_trumans_events(root / "Actions" / f"{source_id}.txt")

    scene_object_bbox_needed = any(
        infer_goal_type(event["text"]) in {"sit", "lie"} | TRUMANS_TEXT_GROUNDED_FIXED_GOAL_TYPES
        for event in events
    )
    scene_objects = load_blender_scene_objects(scene_id, args, no_bbox=not scene_object_bbox_needed)

    segment_list = []
    object_map: dict[str, dict] = {}
    prev_end = None
    seq_last_local = len(idx) - 1

    for event in events:
        event_start = max(0, min(seq_last_local, int(event["start"])))
        event_end = max(0, min(seq_last_local, int(event["end"])))
        if event_end < event_start:
            continue

        local_start = event_start if prev_end is None else max(0, min(seq_last_local, prev_end))
        local_end = event_end
        global_ids = idx[event_start : event_end + 1]
        pelvis_track = np.asarray(joints[global_ids, 0, :], dtype=np.float32)
        left_track = np.asarray(joints[global_ids, TRUMANS_LEFT_HAND_IDX, :], dtype=np.float32)
        right_track = np.asarray(joints[global_ids, TRUMANS_RIGHT_HAND_IDX, :], dtype=np.float32)

        goal_type = infer_goal_type(event["text"])
        active_hand = infer_active_hand(event["text"])
        text_acted_on_name, _ = parse_trumans_object_roles(event["text"], goal_type, None)

        if goal_type in {"sit", "lie"}:
            selected = select_body_object_trumans(event["text"], event_start, event_end, pelvis_track, scene_objects)
            interaction_frame = (event_start + event_end) // 2 if selected is None else selected["interaction_frame"]
        elif goal_type == "turn" and trumans_is_body_only_turn(event["text"], active_hand):
            selected = None
            interaction_frame = event_end
        elif goal_type in TRUMANS_BODY_ONLY_GOAL_TYPES:
            selected = None
            interaction_frame = event_end
        elif text_acted_on_name is not None and goal_type in TRUMANS_TEXT_GROUNDED_FIXED_GOAL_TYPES and not infer_movable(text_acted_on_name):
            query_tracks = trumans_event_hand_tracks(active_hand, left_track, right_track)
            selected = select_fixed_interaction_trumans(
                goal_type,
                event_start,
                event_end,
                query_tracks,
                scene_objects,
                preferred_object_name=text_acted_on_name,
            )
            interaction_frame = (event_start + event_end) // 2 if selected is None else selected["interaction_frame"]
        else:
            query_tracks = trumans_event_hand_tracks(active_hand, left_track, right_track)
            selected = select_hand_interaction_trumans(
                goal_type,
                event_start,
                event_end,
                query_tracks,
                object_pose_dict,
                preferred_object_name=text_acted_on_name,
            )
            interaction_frame = (event_start + event_end) // 2 if selected is None else selected["interaction_frame"]

        interaction_global = int(idx[interaction_frame])
        interaction_local = int(interaction_frame)
        pelvis_goal = np.asarray(joints[interaction_global, 0, :], dtype=np.float32)
        global_orient_goal = None
        if 0 <= interaction_global < len(global_orient):
            global_orient_goal = np.asarray(global_orient[interaction_global], dtype=np.float32)
        left_hand_goal = None
        right_hand_goal = None
        if active_hand in {"left", "both"}:
            left_hand_goal = np.asarray(joints[interaction_global, TRUMANS_LEFT_HAND_IDX, :], dtype=np.float32)
        if active_hand in {"right", "both"}:
            right_hand_goal = np.asarray(joints[interaction_global, TRUMANS_RIGHT_HAND_IDX, :], dtype=np.float32)

        if active_hand == "left":
            hand_goal = left_hand_goal
        elif active_hand == "right":
            hand_goal = right_hand_goal
        elif active_hand == "both":
            if left_hand_goal is not None and right_hand_goal is not None:
                hand_goal = ((left_hand_goal + right_hand_goal) * 0.5).astype(np.float32)
            else:
                hand_goal = left_hand_goal if left_hand_goal is not None else right_hand_goal
        else:
            hand_goal = None

        grounded_name = None if selected is None else selected["object_name"]
        acted_on_name, support_name = parse_trumans_object_roles(event["text"], goal_type, grounded_name)

        acted_on_id = None
        support_id = None
        acted_on_source = None
        support_source = None

        if selected is not None:
            object_id = selected["object_id"]
            object_name = selected["object_name"]
            if goal_type in {"sit", "stand", "lie"}:
                support_id = object_id
                support_source = selected.get("source")
            else:
                acted_on_id = object_id
                acted_on_source = selected.get("source")
            if object_id not in object_map:
                object_map[object_id] = sequence_object_entry(
                    object_id=object_id,
                    object_name=object_name,
                    movable=infer_movable(object_name),
                    initial_location=np.asarray(selected.get("object_location", selected.get("location")), dtype=np.float32),
                    source=selected.get("source"),
                )

        segment_list.append(
            {
                "segment_id": event["segment_id"],
                "start": local_start,
                "end": local_end,
                "interaction_frame": interaction_local,
                "text": event["text"],
                "goal_type": goal_type,
                "active_hand": active_hand,
                "goal_pose": {
                    "pelvis": pelvis_goal,
                    "yaw": None,
                    "global_orient": global_orient_goal,
                    "hand": hand_goal,
                    "left_hand": left_hand_goal,
                    "right_hand": right_hand_goal,
                },
                "acted_on_object_name": acted_on_name,
                "support_object_name": support_name,
                "acted_on_object_id": acted_on_id,
                "support_object_id": support_id,
                "acted_on_object_source": acted_on_source,
                "support_object_source": support_source,
            }
        )
        prev_end = event_end

    global_start = int(idx[0])
    global_end = int(idx[-1]) + 1
    return {
        "sequence_id": sequence_id,
        "scene_id": scene_id,
        "human_motion_ref": {
            "path": to_repo_relative(root / "human_joints.npy"),
            "start": global_start,
            "end": global_end,
            "smplx": {
                "global_orient_path": to_repo_relative(root / "human_orient.npy"),
                "body_pose_path": to_repo_relative(root / "human_pose.npy"),
                "transl_path": to_repo_relative(root / "human_transl.npy"),
                "left_hand_pose_path": to_repo_relative(root / "left_hand_pose.npy"),
                "right_hand_pose_path": to_repo_relative(root / "right_hand_pose.npy"),
            },
        },
        "segment_list": segment_list,
        "object_list": list(object_map.values()),
    }


def load_lingo_ranges_long(root: Path) -> tuple[np.ndarray, np.ndarray]:
    start_idx = np.load(root / "start_idx.npy")
    end_idx = np.load(root / "end_idx.npy")
    return start_idx, end_idx


def load_lingo_texts(root: Path) -> list[str]:
    with open(root / "text_aug.pkl", "rb") as handle:
        texts = pickle.load(handle)
    return [t[0] if isinstance(t, (list, tuple)) else str(t) for t in texts]


@lru_cache(maxsize=4)
def load_lingo_body_goal_meta(root_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(root_str)
    seq_start, seq_end = load_lingo_ranges_long(root)
    with open(root / "language_motion_dict" / "language_motion_dict__inter_and_loco__16.pkl", "rb") as handle:
        motion_dict = pickle.load(handle)

    win_start = np.asarray(motion_dict["start_idx"], dtype=np.int64)
    win_end = np.asarray(motion_dict["end_idx"], dtype=np.int64)
    win_need = np.asarray(motion_dict["need_pelvis_dir"], dtype=bool)
    win_end_range = np.asarray(motion_dict["end_range"], dtype=np.int64)

    num_seq = int(seq_start.shape[0])
    counts = np.zeros(num_seq, dtype=np.int32)
    need_vals = np.zeros(num_seq, dtype=bool)
    need_consistent = np.ones(num_seq, dtype=bool)
    end_range_vals = np.full(num_seq, -1, dtype=np.int64)
    end_range_consistent = np.ones(num_seq, dtype=bool)

    seq_ids = np.searchsorted(seq_end, win_start, side="right")
    valid = seq_ids < num_seq
    valid_seq_ids = seq_ids[valid]
    valid_win_start = win_start[valid]
    valid_win_end = win_end[valid]
    valid_need = win_need[valid]
    valid_end_range = win_end_range[valid]

    valid &= valid_win_start >= seq_start[valid_seq_ids]
    valid &= valid_win_end <= seq_end[valid_seq_ids]

    mapped_seq_ids = valid_seq_ids[valid]
    mapped_need = valid_need[valid]
    mapped_end_range = valid_end_range[valid]

    for seq_idx, need_value, end_range_value in zip(mapped_seq_ids, mapped_need, mapped_end_range):
        if counts[seq_idx] == 0:
            need_vals[seq_idx] = bool(need_value)
            end_range_vals[seq_idx] = int(end_range_value)
        else:
            if need_vals[seq_idx] != bool(need_value):
                need_consistent[seq_idx] = False
            if end_range_vals[seq_idx] != int(end_range_value):
                end_range_consistent[seq_idx] = False
        counts[seq_idx] += 1

    resolved_end_range = end_range_vals.copy()
    resolved_end_range[~end_range_consistent] = -1
    return need_vals, need_consistent, resolved_end_range, counts


def build_lingo_scene_index(root: Path, allowed_scene_ids: set[str] | None = None) -> dict[str, list[int]]:
    with open(root / "scene_name.pkl", "rb") as handle:
        scene_name = pickle.load(handle)
    start_idx = np.load(root / "start_idx.npy")
    scene_to_sequences: dict[str, list[int]] = {}
    for seq_idx, start in enumerate(start_idx):
        scene_id = str(scene_name[int(start)])
        if allowed_scene_ids is not None and scene_id not in allowed_scene_ids:
            continue
        scene_to_sequences.setdefault(scene_id, []).append(int(seq_idx))
    return scene_to_sequences


def parse_lingo_object_roles(text: str) -> tuple[str | None, str | None]:
    lower = text.lower()
    for pattern, role in LINGO_ROLE_PATTERNS:
        match = pattern.search(lower)
        if not match:
            continue
        if role == "acted":
            return normalize_extracted_role_name(match.group(1)), None
        if role == "support":
            return None, normalize_extracted_role_name(match.group(1))
        if role == "acted_support":
            return normalize_extracted_role_name(match.group(1)), normalize_extracted_role_name(match.group(2))
    return None, None


def make_lingo_object_id(name: str, role: str, counters: dict[tuple[str, str], int]) -> str:
    key = (role, name)
    counters[key] += 1
    return f"{role}_{name}_{counters[key]:02d}"


def build_lingo_sequence(sequence_index: int, args, scene_id_override: str | None = None) -> dict:
    root = resolve_lingo_root(args)
    texts = load_lingo_texts(root)
    with open(root / "scene_name.pkl", "rb") as handle:
        scene_name = pickle.load(handle)
    joints = np.load(root / "human_joints_aligned.npy", mmap_mode="r")
    global_orient = np.load(root / "human_orient.npy", mmap_mode="r")
    left_hand_inter_arr = np.load(root / "left_hand_inter_frame.npy", mmap_mode="r")
    right_hand_inter_arr = np.load(root / "right_hand_inter_frame.npy", mmap_mode="r")
    need_pelvis_dir_arr, need_pelvis_consistent_arr, end_range_arr, body_meta_counts = load_lingo_body_goal_meta(str(root))

    start_idx_arr, end_idx_arr = load_lingo_ranges_long(root)
    start_idx = int(start_idx_arr[sequence_index])
    end_idx = int(end_idx_arr[sequence_index])
    left_inter = int(left_hand_inter_arr[sequence_index])
    right_inter = int(right_hand_inter_arr[sequence_index])

    text = texts[sequence_index]
    scene_id = scene_id_override if scene_id_override is not None else str(scene_name[start_idx])
    goal_type = infer_goal_type(text)
    active_hand = infer_active_hand(text, left_inter=left_inter, right_inter=right_inter)

    if left_inter != -1:
        interaction_frame = left_inter
        hand_goal = np.asarray(joints[left_inter, LINGO_LEFT_HAND_IDX, :], dtype=np.float32)
    elif right_inter != -1:
        interaction_frame = right_inter
        hand_goal = np.asarray(joints[right_inter, LINGO_RIGHT_HAND_IDX, :], dtype=np.float32)
    else:
        hand_goal = None
        body_goal_frame = None
        if body_meta_counts[sequence_index] > 0 and bool(need_pelvis_consistent_arr[sequence_index]) and bool(need_pelvis_dir_arr[sequence_index]):
            if goal_type in {"sit", "lie"}:
                candidate = int(end_range_arr[sequence_index])
                if start_idx <= candidate < end_idx:
                    body_goal_frame = candidate
            else:
                candidate = max(start_idx, end_idx - 3)
                if start_idx <= candidate < end_idx:
                    body_goal_frame = candidate
        if body_goal_frame is None:
            body_goal_frame = max(start_idx, end_idx - 1)
        interaction_frame = body_goal_frame

    pelvis_goal = np.asarray(joints[interaction_frame, 0, :], dtype=np.float32)
    global_orient_goal = None
    if 0 <= interaction_frame < len(global_orient):
        global_orient_goal = np.asarray(global_orient[interaction_frame], dtype=np.float32)

    acted_on_name, support_name = parse_lingo_object_roles(text)
    object_counters: dict[tuple[str, str], int] = defaultdict(int)
    object_list = []
    acted_on_id = None
    support_id = None

    if acted_on_name is not None:
        acted_on_id = make_lingo_object_id(acted_on_name, "acted_on", object_counters)
        object_list.append(
            sequence_object_entry(
                object_id=acted_on_id,
                object_name=acted_on_name,
                movable=infer_movable(acted_on_name),
                initial_location=None if hand_goal is None else np.asarray(hand_goal, dtype=np.float32),
            )
        )

    if support_name is not None:
        support_id = make_lingo_object_id(support_name, "support", object_counters)
        object_list.append(
            sequence_object_entry(
                object_id=support_id,
                object_name=support_name,
                movable=infer_movable(support_name),
                initial_location=np.asarray(pelvis_goal, dtype=np.float32),
            )
        )

    sequence_id = f"lingo_{sequence_index:06d}"
    segment = {
        "segment_id": 0,
        "start": 0,
        "end": end_idx - start_idx,
        "interaction_frame": interaction_frame - start_idx,
        "text": text,
        "goal_type": goal_type,
        "active_hand": active_hand,
        "goal_pose": {
            "pelvis": pelvis_goal,
            "yaw": None,
            "global_orient": global_orient_goal,
            "hand": hand_goal,
        },
        "acted_on_object_name": acted_on_name,
        "support_object_name": support_name,
        "acted_on_object_id": acted_on_id,
        "support_object_id": support_id,
    }
    return {
        "sequence_id": sequence_id,
        "scene_id": scene_id,
        "human_motion_ref": {
            "path": to_repo_relative(root / "human_joints_aligned.npy"),
            "start": start_idx,
            "end": end_idx,
            "smplx": {
                "global_orient_path": to_repo_relative(root / "human_orient.npy"),
                "body_pose_path": to_repo_relative(root / "human_pose.npy"),
                "transl_path": to_repo_relative(root / "transl_aligned.npy"),
            },
        },
        "segment_list": [segment],
        "object_list": object_list,
    }


def merge_lingo_sequences(sequence_records: list[dict], args) -> list[dict]:
    if not sequence_records:
        return []
    root = resolve_lingo_root(args)
    joints = np.load(root / "human_joints_aligned.npy", mmap_mode="r")

    def hand_pelvis_at(frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pelvis = np.asarray(joints[frame, 0, :], dtype=np.float32)
        left = np.asarray(joints[frame, LINGO_LEFT_HAND_IDX, :], dtype=np.float32)
        right = np.asarray(joints[frame, LINGO_RIGHT_HAND_IDX, :], dtype=np.float32)
        return pelvis, left, right

    recs = sorted(sequence_records, key=lambda r: r["human_motion_ref"]["start"])
    groups: list[list[dict]] = []
    current: list[dict] = [recs[0]]
    for nxt in recs[1:]:
        prev = current[-1]
        prev_end = prev["human_motion_ref"]["end"] - 1
        next_start = nxt["human_motion_ref"]["start"]
        if prev_end < 0 or next_start < 0 or prev_end >= joints.shape[0] or next_start >= joints.shape[0]:
            groups.append(current)
            current = [nxt]
            continue
        p0, l0, r0 = hand_pelvis_at(prev_end)
        p1, l1, r1 = hand_pelvis_at(next_start)
        pelvis_dist = float(np.linalg.norm((p1 - p0)[[0, 2]]))
        left_dist = float(np.linalg.norm((l1 - l0)[[0, 2]]))
        right_dist = float(np.linalg.norm((r1 - r0)[[0, 2]]))
        if pelvis_dist <= 0.1 and left_dist <= 0.1 and right_dist <= 0.1:
            current.append(nxt)
        else:
            groups.append(current)
            current = [nxt]
    groups.append(current)

    merged_records: list[dict] = []
    for group in groups:
        if len(group) == 1:
            merged_records.append(group[0])
            continue
        scene_id = group[0]["scene_id"]
        start = min(r["human_motion_ref"]["start"] for r in group)
        end = max(r["human_motion_ref"]["end"] for r in group)
        merged_id = f"{group[0]['sequence_id']}_to_{group[-1]['sequence_id']}"

        segment_list = []
        object_list = []
        obj_id_map: dict[str, str] = {}
        seg_id = 0
        for r in group:
            seq_id = r["sequence_id"]
            offset = r["human_motion_ref"]["start"] - start
            for obj in r.get("object_list", []):
                old_id = obj["object_id"]
                new_id = f"{seq_id}:{old_id}"
                obj_id_map[(seq_id, old_id)] = new_id
                new_obj = dict(obj)
                new_obj["object_id"] = new_id
                object_list.append(new_obj)

            for seg in r["segment_list"]:
                new_seg = dict(seg)
                new_seg["segment_id"] = seg_id
                seg_id += 1
                new_seg["start"] = int(seg["start"]) + offset
                new_seg["end"] = int(seg["end"]) + offset
                new_seg["interaction_frame"] = int(seg["interaction_frame"]) + offset
                acted = seg.get("acted_on_object_id")
                support = seg.get("support_object_id")
                if acted is not None:
                    new_seg["acted_on_object_id"] = obj_id_map.get((seq_id, acted), f"{seq_id}:{acted}")
                if support is not None:
                    new_seg["support_object_id"] = obj_id_map.get((seq_id, support), f"{seq_id}:{support}")
                segment_list.append(new_seg)

        merged_records.append(
            {
                "sequence_id": merged_id,
                "scene_id": scene_id,
                "human_motion_ref": {
                    **{k: v for k, v in group[0]["human_motion_ref"].items() if k not in {"start", "end"}},
                    "start": start,
                    "end": end,
                },
                "segment_list": segment_list,
                "object_list": object_list,
            }
        )
    return merged_records


def build_scene_record(dataset: str, scene_id: str, sequence_ids: list[str], args) -> dict:
    if dataset == "trumans":
        occ_path = args.trumans_root / "Scene" / f"{scene_id}.npy"
        grid_meta = TRUMANS_GRID_META
    else:
        root = resolve_lingo_root(args)
        train_occ = root / "Scene" / f"{scene_id}.npy"
        test_occ = root / "Scene_vis" / f"{scene_id}.npy"
        if train_occ.exists():
            occ_path = train_occ
            grid_meta = LINGO_TRAIN_GRID_META
        else:
            occ_path = test_occ
            grid_meta = LINGO_TEST_GRID_META

    scene_occ = np.load(occ_path)
    clearance_npy, clearance_vis = save_clearance(scene_id, scene_occ, dataset, args)
    return {
        "scene_id": scene_id,
        "occupancy_grid_path": to_repo_relative(occ_path),
        "clearance_map_npy_path": to_repo_relative(clearance_npy),
        "clearance_map_vis_path": to_repo_relative(clearance_vis),
        "grid_meta": grid_meta,
        "sequence_ids": sequence_ids,
    }


def build_trumans_plot_segments(sequence_record: dict) -> list[dict]:
    object_by_id = {obj["object_id"]: obj for obj in sequence_record.get("object_list", [])}
    plot_segments = []
    for segment in sequence_record["segment_list"]:
        object_id = segment.get("support_object_id") or segment.get("acted_on_object_id")
        object_source = segment.get("support_object_source") or segment.get("acted_on_object_source")
        object_loc = None
        if object_id is not None and object_id in object_by_id:
            object_loc = object_by_id[object_id].get("initial_location")
        plot_segments.append(
            {
                "segment_id": segment["segment_id"],
                "start": segment["start"],
                "end": segment["end"],
                "interaction_frame": segment["interaction_frame"],
                "text": segment["text"],
                "goal_type": "body" if segment["goal_type"] in {"sit", "stand", "lie"} else "hand",
                "hand": segment["active_hand"],
                "goal_object": object_id,
                "goal_object_source": object_source,
                "goal_object_location": object_loc,
            }
        )
    return plot_segments


def build_lingo_plot_segments(sequence_record: dict) -> list[dict]:
    object_by_id = {obj["object_id"]: obj for obj in sequence_record.get("object_list", [])}
    plot_segments = []
    for segment in sequence_record["segment_list"]:
        object_id = segment.get("support_object_id") or segment.get("acted_on_object_id")
        object_pose = None
        if object_id is not None and object_id in object_by_id:
            object_pose = {
                "frame": segment["start"],
                "location": object_by_id[object_id].get("initial_location"),
                "rotation_euler": None,
            }
        plot_segments.append(
            {
                "segment_id": segment["segment_id"],
                "start": segment["start"],
                "end": segment["end"],
                "interaction_frame": segment["interaction_frame"],
                "text": segment["text"],
                "hand": segment["active_hand"],
                "goal_object": object_id,
                "goal_object_name": object_by_id.get(object_id, {}).get("object_name") if object_id is not None else None,
                "object_pose": object_pose,
            }
        )
    return plot_segments


def generate_sequence_plots(sequence_record: dict, dataset: str, args) -> None:
    scene_id = sequence_record["scene_id"]
    sequence_id = sequence_record["sequence_id"]
    if dataset == "trumans":
        payload = {
            "sequence_id": sequence_id,
            "scene_id": scene_id,
            "segment_list": build_trumans_plot_segments(sequence_record),
        }
        script = PROJECT_ROOT / "vis" / "visualize_trumans_sequence_actions.py"
        extra = ["--data-root", str(args.trumans_root)]
    else:
        payload = {
            "sequence_id": sequence_id,
            "scene_id": scene_id,
            "human_motion_ref": sequence_record["human_motion_ref"],
            "segment_list": build_lingo_plot_segments(sequence_record),
        }
        script = PROJECT_ROOT / "vis" / "visualize_lingo_sequence_actions.py"
        extra = ["--split", resolve_lingo_scene_split(scene_id, args)]
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"{sequence_id}.plot_segments.",
        dir="/tmp",
        delete=False,
    ) as handle:
        json.dump(to_serializable(payload), handle, indent=2)
        plot_input_path = Path(handle.name)
    try:
        cmd = [
            args.python_bin,
            str(script),
            "--sequence-id",
            sequence_id,
            "--scene-name",
            scene_id,
            "--segment-json",
            str(plot_input_path),
            "--output-dir",
            str(plot_dir(scene_id, sequence_id, dataset, args).parent),
        ] + extra
        subprocess.run(cmd, check=True, timeout=PLOT_TIMEOUT_SEC)
    finally:
        plot_input_path.unlink(missing_ok=True)
    for segment in sequence_record["segment_list"]:
        segment["plot_path"] = to_repo_relative(
            plot_dir(scene_id, sequence_id, dataset, args) / f"{segment['segment_id']:02d}_{segment['start']:04d}_{segment['end']:04d}.png"
        )


def run_dataset(args) -> None:
    if args.dataset == "trumans":
        scene_index = build_trumans_scene_index(args.trumans_root)
        if args.workers > 1:
            scene_ids = list(scene_index.keys())
            with mp.Pool(processes=args.workers, maxtasksperchild=1) as pool:
                for _ in progress(
                    pool.imap_unordered(
                        _process_trumans_scene,
                        [(scene_id, scene_index[scene_id], args) for scene_id in scene_ids],
                        chunksize=1,
                    ),
                    desc="TRUMANS scenes",
                ):
                    pass
        else:
            for scene_id, sequence_ids in progress(scene_index.items(), desc="TRUMANS scenes"):
                _process_trumans_scene((scene_id, sequence_ids, args))
    else:
        root = resolve_lingo_root(args)
        allowed_scene_ids = set()
        train_scene_dir = root / "Scene"
        test_scene_dir = root / "Scene_vis"
        if train_scene_dir.exists():
            allowed_scene_ids.update(p.stem for p in train_scene_dir.glob("*.npy"))
        if test_scene_dir.exists():
            allowed_scene_ids.update(p.stem for p in test_scene_dir.glob("*.npy"))
        if not allowed_scene_ids:
            allowed_scene_ids = None
        scene_index = build_lingo_scene_index(root, allowed_scene_ids=allowed_scene_ids)
        if allowed_scene_ids is not None:
            for scene_id in sorted(allowed_scene_ids):
                scene_index.setdefault(scene_id, [])
        if args.workers > 1:
            scene_ids = list(scene_index.keys())
            with mp.Pool(processes=args.workers, maxtasksperchild=1) as pool:
                for _ in progress(
                    pool.imap_unordered(
                        _process_lingo_scene,
                        [(scene_id, scene_index[scene_id], args) for scene_id in scene_ids],
                        chunksize=1,
                    ),
                    desc="LINGO scenes",
                ):
                    pass
        else:
            for scene_id, sequence_indices in progress(scene_index.items(), desc="LINGO scenes"):
                _process_lingo_scene((scene_id, sequence_indices, args))
    write_split_file(args)


def run_scene(args) -> dict:
    if args.dataset == "trumans":
        if args.scene_id is None:
            raise ValueError("--scene-id is required")
        scene_index = build_trumans_scene_index(args.trumans_root)
        sequence_ids = scene_index.get(args.scene_id, [])
        reset_scene_outputs(args.scene_id, args)
        scene_record = build_scene_record("trumans", args.scene_id, sequence_ids, args)
        for sequence_id in sequence_ids:
            sequence_record = build_trumans_sequence(sequence_id, args.scene_id, args)
            generate_sequence_plots(sequence_record, "trumans", args)
            write_json(sequence_dir(args.scene_id, args) / f"{sequence_id}.json", sequence_record)
        write_json(scene_dir(args.scene_id, args) / "scene.json", scene_record)
        return scene_record

    if args.scene_id is None:
        raise ValueError("--scene-id is required")
    root = resolve_lingo_root(args)
    scene_folder = "Scene" if args.split == "train" else "Scene_vis"
    scene_root = root / scene_folder
    allowed_scene_ids = None
    if scene_root.exists():
        allowed_scene_ids = {p.stem for p in scene_root.glob("*.npy")}
    scene_index = build_lingo_scene_index(root, allowed_scene_ids=allowed_scene_ids)
    if allowed_scene_ids is not None:
        for scene_id in sorted(allowed_scene_ids):
            scene_index.setdefault(scene_id, [])
    sequence_indices = scene_index.get(args.scene_id, [])
    reset_scene_outputs(args.scene_id, args)
    raw_records = [build_lingo_sequence(idx, args, scene_id_override=args.scene_id) for idx in sequence_indices]
    merged_records = merge_lingo_sequences(raw_records, args)
    scene_record = build_scene_record("lingo", args.scene_id, [r["sequence_id"] for r in merged_records], args)
    for sequence_record in merged_records:
        generate_sequence_plots(sequence_record, "lingo", args)
        write_json(sequence_dir(args.scene_id, args) / f"{sequence_record['sequence_id']}.json", sequence_record)
    write_json(scene_dir(args.scene_id, args) / "scene.json", scene_record)
    write_split_file(args)
    return scene_record


def _process_trumans_scene(payload: tuple[str, list[str], argparse.Namespace]) -> str:
    scene_id, sequence_ids, args = payload
    print(f"[scene start] trumans {scene_id}", flush=True)
    try:
        with with_scene_timeout(args.scene_timeout_sec):
            reset_scene_outputs(scene_id, args)
            scene_record = build_scene_record("trumans", scene_id, sequence_ids, args)
            for sequence_id in sequence_ids:
                sequence_record = build_trumans_sequence(sequence_id, scene_id, args)
                generate_sequence_plots(sequence_record, "trumans", args)
                write_json(sequence_dir(scene_id, args) / f"{sequence_id}.json", sequence_record)
            write_json(scene_dir(scene_id, args) / "scene.json", scene_record)
        print(f"[scene done] trumans {scene_id}", flush=True)
    except SceneTimeoutError:
        print(f"[scene timeout] trumans {scene_id}", flush=True)
    return scene_id


def _process_lingo_scene(payload: tuple[str, list[int], argparse.Namespace]) -> str:
    scene_id, sequence_indices, args = payload
    print(f"[scene start] lingo {scene_id}", flush=True)
    try:
        with with_scene_timeout(args.scene_timeout_sec):
            reset_scene_outputs(scene_id, args)
            raw_records = [build_lingo_sequence(idx, args, scene_id_override=scene_id) for idx in sequence_indices]
            merged_records = merge_lingo_sequences(raw_records, args)
            sequence_ids = [r["sequence_id"] for r in merged_records]
            scene_record = build_scene_record("lingo", scene_id, sequence_ids, args)
            for sequence_record in merged_records:
                generate_sequence_plots(sequence_record, "lingo", args)
                write_json(sequence_dir(scene_id, args) / f"{sequence_record['sequence_id']}.json", sequence_record)
            write_json(scene_dir(scene_id, args) / "scene.json", scene_record)
        print(f"[scene done] lingo {scene_id}", flush=True)
    except SceneTimeoutError:
        print(f"[scene timeout] lingo {scene_id}", flush=True)
    return scene_id


def run_sequence(args) -> dict:
    if args.dataset == "trumans":
        if args.sequence_id is None:
            raise ValueError("--sequence-id is required for TRUMANS")
        scene_id = resolve_trumans_scene_for_sequence(args.sequence_id, args.trumans_root)
        sequence_record = build_trumans_sequence(args.sequence_id, scene_id, args)
        scene_record = build_scene_record("trumans", scene_id, [args.sequence_id], args)
        generate_sequence_plots(sequence_record, "trumans", args)
        write_json(sequence_dir(scene_id, args) / f"{args.sequence_id}.json", sequence_record)
        write_json(scene_dir(scene_id, args) / "scene.json", scene_record)
        return {"scene": scene_record, "sequence": sequence_record}

    if args.sequence_index is None:
        raise ValueError("--sequence-index is required for LINGO")
    root = resolve_lingo_root(args)
    scene_folder = "Scene" if args.split == "train" else "Scene_vis"
    scene_root = root / scene_folder
    allowed_scene_ids = None
    if scene_root.exists():
        allowed_scene_ids = {p.stem for p in scene_root.glob("*.npy")}
    scene_index = build_lingo_scene_index(root, allowed_scene_ids=allowed_scene_ids)
    scene_id = None
    for key, indices in scene_index.items():
        if args.sequence_index in indices:
            scene_id = key
            break
    sequence_record = build_lingo_sequence(args.sequence_index, args, scene_id_override=scene_id)
    scene_id = sequence_record["scene_id"]
    scene_record = build_scene_record("lingo", scene_id, [sequence_record["sequence_id"]], args)
    generate_sequence_plots(sequence_record, "lingo", args)
    write_json(sequence_dir(scene_id, args) / f"{sequence_record['sequence_id']}.json", sequence_record)
    write_json(scene_dir(scene_id, args) / "scene.json", scene_record)
    write_split_file(args)
    return {"scene": scene_record, "sequence": sequence_record}


def parse_args():
    parser = argparse.ArgumentParser(description="Final unified preprocessing for TRUMANS and LINGO.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--sequence-id", default=None)
    parser.add_argument("--sequence-index", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--trumans-root", type=Path, default=TRUMANS_ROOT_DEFAULT)
    parser.add_argument("--lingo-root", type=Path, default=None)
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--blender-bin", default=str(BLENDER_BIN_DEFAULT))
    parser.add_argument("--force-blender-dump", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--scene-timeout-sec", type=int, default=0)
    return parser.parse_args()


def resolve_mode(args) -> str:
    if args.sequence_id is not None or args.sequence_index is not None:
        return "sequence"
    if args.scene_id is not None:
        return "scene"
    return "dataset"


def main():
    args = parse_args()
    mode = resolve_mode(args)
    if mode == "dataset":
        run_dataset(args)
        print(preprocess_root(args) / "scenes")
        return
    if mode == "scene":
        result = run_scene(args)
        out_path = scene_dir(args.scene_id, args) / "scene.json"
        print(out_path)
        return
    result = run_sequence(args)
    scene_id = result["sequence"]["scene_id"]
    out_path = sequence_dir(scene_id, args) / f"{result['sequence']['sequence_id']}.json"
    print(out_path)


if __name__ == "__main__":
    main()
