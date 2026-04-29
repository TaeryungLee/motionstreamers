from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from build_multi_character_affordances import (
    AREA_COLORS,
    AREA_SIZE_BY_TYPE,
    PROJECT_ROOT,
    add_legend,
    canonical_name,
    classify_goal_bucket,
    get_pelvis_and_hand,
    infer_target_name_from_text,
    infer_segment_mode,
    is_seat_like_name,
    load_json,
    rect_mask,
    resolve_scene_ids,
    save_mask,
    to_repo_relative,
    write_json,
)


TRUMANS_EXCLUDED_SCENE_SUPPORT_TOKENS = {
    "slim_jeans",
    "jeans",
    "pants",
    "shirt",
    "sweater",
    "dress",
    "skirt",
    "shoe",
    "shoes",
    "boot",
    "boots",
    "sock",
    "socks",
    "hair",
    "beard",
    "mustache",
    "body",
    "eye",
    "teeth",
    "tongue",
    "tearline",
    "eyeocclusion",
    "bushy",
    "blowback",
    "stubble",
    "curtain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-scene source banks for multi-character episode sampling.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--scene-list-file", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "preprocessed")
    parser.add_argument("--source-scenes-name", default="scenes_v2")
    parser.add_argument("--episode-bank-name", default="episode_bank_v3")
    parser.add_argument("--loco-window-frames", type=int, default=600)
    parser.add_argument("--loco-min-window-frames", type=int, default=300)
    return parser.parse_args()


def is_loco_scene(dataset: str, scene_id: str) -> bool:
    return dataset == "lingo" and "loco" in scene_id.lower()


def get_scene_paths(dataset: str, scene_id: str, output_root: Path, source_scenes_name: str, episode_bank_name: str) -> Tuple[Path, Path]:
    scene_root = output_root / dataset / source_scenes_name / scene_id
    bank_root = output_root / dataset / episode_bank_name / scene_id
    return scene_root, bank_root


def load_scene_sequences(dataset: str, scene_root: Path, scene_record: dict) -> List[dict]:
    sequence_records: List[dict] = []
    for sequence_id in scene_record.get("sequence_ids", []):
        if dataset == "trumans" and "_augment" in sequence_id:
            continue
        seq_path = scene_root / "sequences" / f"{sequence_id}.json"
        if seq_path.exists():
            sequence_records.append(load_json(seq_path))
    return sequence_records


def normalize_goal_type(segment: dict) -> str:
    value = segment.get("goal_type")
    return "" if value is None else str(value).strip()


def goal_category(dataset: str, segment: dict) -> str:
    bucket = classify_goal_bucket(dataset, normalize_goal_type(segment))
    if bucket is None:
        return "other"
    return bucket


def segment_global_frame_index(sequence: dict, segment: dict) -> int:
    motion_ref = sequence.get("human_motion_ref", {})
    return int(motion_ref.get("start", 0)) + int(segment.get("interaction_frame", 0))


def coerce_xyz(value) -> Optional[List[float]]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.shape[0] < 3 or not np.isfinite(arr[:3]).all():
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def mask_to_vis(clearance: np.ndarray, layers: List[Tuple[np.ndarray, str]], out_path: Path) -> None:
    base = np.where(clearance, 245, 20).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1)
    for mask, area_type in layers:
        if mask is None or not mask.any():
            continue
        color = AREA_COLORS[area_type]
        rgb[mask] = (0.6 * rgb[mask] + 0.4 * color).astype(np.uint8)
    image = Image.fromarray(np.transpose(rgb, (1, 0, 2)), mode="RGB")
    image = image.resize((image.width * 4, image.height * 4), resample=getattr(Image, "Resampling", Image).NEAREST)
    image = add_legend(image, include_origin=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


@lru_cache(maxsize=1)
def load_trumans_given_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = PROJECT_ROOT / "data" / "raw" / "trumans"
    seg_name = np.load(root / "seg_name.npy", allow_pickle=True)
    object_flag = np.load(root / "object_flag.npy", mmap_mode="r")
    object_list = np.load(root / "object_list.npy", allow_pickle=True)
    return seg_name, object_flag, object_list


@lru_cache(maxsize=1)
def load_trumans_joint_arrays() -> Tuple[np.ndarray, np.ndarray]:
    root = PROJECT_ROOT / "data" / "raw" / "trumans"
    seg_name = np.load(root / "seg_name.npy", allow_pickle=True)
    joints = np.load(root / "human_joints.npy", mmap_mode="r")
    return seg_name, joints


@lru_cache(maxsize=256)
def load_trumans_object_pose(sequence_id: str) -> dict:
    root = PROJECT_ROOT / "data" / "raw" / "trumans" / "Object_all" / "Object_pose"
    base_sequence_id = sequence_id.split("_augment", 1)[0]
    path = root / f"{base_sequence_id}.npy"
    if not path.exists():
        return {}
    return np.load(path, allow_pickle=True).item()


def world_to_motion(location: list[float] | np.ndarray) -> np.ndarray:
    loc = np.asarray(location, dtype=np.float32)
    return np.array([loc[0], loc[2], -loc[1]], dtype=np.float32)


def point_to_aabb_distance(point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    lo = np.minimum(bbox_min, bbox_max)
    hi = np.maximum(bbox_min, bbox_max)
    clamped = np.minimum(np.maximum(point, lo), hi)
    return float(np.linalg.norm(point - clamped))


def is_excluded_trumans_scene_support_name(name: Optional[str]) -> bool:
    normalized = canonical_name(name)
    if not normalized or normalized == "unknown":
        return True
    if normalized.startswith("cc_base_"):
        return True
    return any(token in normalized for token in TRUMANS_EXCLUDED_SCENE_SUPPORT_TOKENS)


@lru_cache(maxsize=128)
def load_trumans_scene_support_bboxes(scene_id: str) -> List[dict]:
    cache_path = PROJECT_ROOT / "data" / "preprocessed_cache" / "trumans" / scene_id / "bbox.json"
    if not cache_path.exists():
        return []
    payload = json.loads(cache_path.read_text())
    outputs: List[dict] = []
    for record in payload.get("objects", []):
        if record.get("type") not in {"MESH", "EMPTY"}:
            continue
        object_name = canonical_name(record.get("data_name") or record.get("name"))
        if is_excluded_trumans_scene_support_name(object_name):
            continue
        bbox_min = record.get("bbox_min")
        bbox_max = record.get("bbox_max")
        if bbox_min is None or bbox_max is None:
            continue
        outputs.append(
            {
                "object_id": record["name"],
                "object_name": object_name,
                "bbox_min": world_to_motion(bbox_min),
                "bbox_max": world_to_motion(bbox_max),
            }
        )
    return outputs


def trumans_given_active_objects(sequence_id: str, start: int, end: int) -> List[str]:
    seg_name, object_flag, object_list = load_trumans_given_arrays()
    idx = np.where(seg_name == sequence_id)[0]
    if len(idx) == 0:
        return []
    local_start = max(0, min(int(start), len(idx) - 1))
    local_end = max(local_start, min(int(end), len(idx) - 1))
    frame_ids = idx[local_start : local_end + 1]
    if len(frame_ids) == 0:
        return []
    slot_ids = sorted({int(slot_id) for row in object_flag[frame_ids] for slot_id in np.where(row >= 0)[0]})
    names = []
    for slot_id in slot_ids:
        raw_name = str(object_list[slot_id]).split("(")[0].strip()
        if raw_name:
            names.append(raw_name)
    return names


def trumans_pelvis_track(sequence_id: str, start: int, end: int) -> Optional[np.ndarray]:
    seg_name, joints = load_trumans_joint_arrays()
    idx = np.where(seg_name == sequence_id)[0]
    if len(idx) == 0:
        return None
    local_start = max(0, min(int(start), len(idx) - 1))
    local_end = max(local_start, min(int(end), len(idx) - 1))
    frame_ids = idx[local_start : local_end + 1]
    if len(frame_ids) == 0:
        return None
    return np.asarray(joints[frame_ids, 0, :], dtype=np.float32)


def trumans_given_object_min_distance(sequence_id: str, start: int, end: int, object_names: List[str]) -> Tuple[Optional[str], Optional[float]]:
    pelvis_track = trumans_pelvis_track(sequence_id, start, end)
    if pelvis_track is None or len(pelvis_track) == 0:
        return None, None
    object_pose = load_trumans_object_pose(sequence_id)
    best_name = None
    best_dist = None
    for name in object_names:
        entry = object_pose.get(name)
        if entry is None:
            continue
        locations = entry.get("location", [])
        if start >= len(locations):
            continue
        n = min(len(pelvis_track), len(locations) - start)
        if n <= 0:
            continue
        for i in range(n):
            loc = np.asarray(locations[start + i], dtype=np.float32)
            if not np.isfinite(loc).all():
                continue
            dist = float(np.linalg.norm(pelvis_track[i][[0, 2]] - loc[[0, 2]]))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_name = name
    return best_name, best_dist


def trumans_scene_chair_min_distance(scene_id: str, sequence_id: str, start: int, end: int) -> Optional[float]:
    pelvis_track = trumans_pelvis_track(sequence_id, start, end)
    if pelvis_track is None or len(pelvis_track) == 0:
        return None
    scene_objects = load_trumans_scene_support_bboxes(scene_id)
    if not scene_objects:
        return None
    best = None
    for pelvis in pelvis_track:
        for obj in scene_objects:
            dist = point_to_aabb_distance(
                np.asarray(pelvis, dtype=np.float32),
                np.asarray(obj["bbox_min"], dtype=np.float32),
                np.asarray(obj["bbox_max"], dtype=np.float32),
            )
            if best is None or dist < best:
                best = dist
    return best


def normalize_given_chair_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    normalized = str(name).strip()
    if normalized.startswith("movable_chair_base_"):
        return normalized.replace("movable_chair_base_", "movable_chair_seat_", 1)
    return normalized


def infer_given_seated_name(
    dataset: str,
    scene_id: str,
    sequence_id: str,
    segment: dict,
) -> Tuple[Optional[str], Optional[str], List[str], Optional[float], Optional[float]]:
    if dataset != "trumans":
        return None, None, [], None, None
    goal_type = normalize_goal_type(segment)
    if goal_type not in {"sit", "lie", "slide"}:
        return None, None, [], None, None
    active_names = trumans_given_active_objects(sequence_id, int(segment.get("start", 0)), int(segment.get("end", 0)))
    chair_names = [name for name in active_names if is_seat_like_name(name)]
    if not chair_names:
        return None, None, active_names, None, None
    chosen_name, given_min = trumans_given_object_min_distance(
        sequence_id,
        int(segment.get("start", 0)),
        int(segment.get("end", 0)),
        chair_names,
    )
    chosen_name = normalize_given_chair_name(chosen_name)
    scene_chair_min = trumans_scene_chair_min_distance(scene_id, sequence_id, int(segment.get("start", 0)), int(segment.get("end", 0)))
    given_valid = bool(given_min is not None and given_min <= 0.20)
    if not given_valid:
        return None, None, active_names, given_min, scene_chair_min
    return chosen_name, chosen_name, active_names, given_min, scene_chair_min


def build_segment_goal_bank(
    dataset: str,
    scene_id: str,
    sequence_records: List[dict],
    clearance: np.ndarray,
    grid_meta: dict,
    out_root: Path,
) -> Tuple[List[dict], Dict[Tuple[str, int], dict]]:
    per_segment_root = out_root / "per_segment"
    vis_root = out_root / "vis" / "per_segment"
    segment_records: List[dict] = []
    record_by_key: Dict[Tuple[str, int], dict] = {}

    for sequence in sequence_records:
        sequence_id = sequence["sequence_id"]
        for segment in sequence.get("segment_list", []):
            segment_id = int(segment["segment_id"])
            category = goal_category(dataset, segment)
            mode = infer_segment_mode(dataset, segment)
            pelvis, hand, yaw = get_pelvis_and_hand(dataset, sequence, segment)
            goal_pose = segment.get("goal_pose", {})
            body_goal = None if pelvis is None else [float(pelvis[0]), float(pelvis[1]), float(pelvis[2])]
            hand_goal = None if hand is None else [float(hand[0]), float(hand[1]), float(hand[2])]
            global_orient = coerce_xyz(goal_pose.get("global_orient"))
            given_seated_name, movable_seated_name, given_active_objects, given_min_dist, scene_chair_min_dist = infer_given_seated_name(
                dataset, scene_id, sequence_id, segment
            )

            support_object_name = segment.get("support_object_name")
            support_object_id = segment.get("support_object_id")
            if mode == "seat" and given_seated_name is not None:
                support_object_name = "chair"
                support_object_id = given_seated_name

            body_mask = None
            hand_mask = None
            body_mask_path = None
            hand_mask_path = None
            target_name = None

            if mode == "seat" and pelvis is not None:
                target_name = "seat"
                body_mask = rect_mask(
                    grid_meta,
                    clearance.shape,
                    (float(pelvis[0]), float(pelvis[2])),
                    AREA_SIZE_BY_TYPE["seat"],
                    float(yaw),
                )
            elif mode in {"support", "object"} and pelvis is not None and hand is not None:
                text_target_name = infer_target_name_from_text(segment.get("text"), normalize_goal_type(segment))
                target_name = canonical_name(
                    (
                        text_target_name
                        if mode == "object" and text_target_name
                        else segment.get("support_object_name")
                    )
                    or segment.get("acted_on_object_name")
                    or text_target_name
                    or "unknown"
                )
                body_mask = rect_mask(
                    grid_meta,
                    clearance.shape,
                    (float(pelvis[0]), float(pelvis[2])),
                    AREA_SIZE_BY_TYPE["interactable"],
                    float(yaw),
                )
                hand_mask = rect_mask(
                    grid_meta,
                    clearance.shape,
                    (float(hand[0]), float(hand[2])),
                    AREA_SIZE_BY_TYPE["support" if mode == "support" else "object"],
                    float(yaw),
                )

            stem = f"{sequence_id}__seg{segment_id:02d}"
            if body_mask is not None and body_mask.any():
                body_mask_path = per_segment_root / f"{stem}__body.npy"
                save_mask(body_mask, body_mask_path)
            if hand_mask is not None and hand_mask.any():
                hand_mask_path = per_segment_root / f"{stem}__hand.npy"
                save_mask(hand_mask, hand_mask_path)

            vis_layers: List[Tuple[np.ndarray, str]] = []
            if body_mask is not None and body_mask.any():
                vis_layers.append((body_mask, "seat" if mode == "seat" else "interactable"))
            if hand_mask is not None and hand_mask.any():
                vis_layers.append((hand_mask, "support" if mode == "support" else "object"))
            if vis_layers:
                mask_to_vis(clearance, vis_layers, vis_root / f"{stem}.png")

            record = {
                "scene_id": scene_id,
                "sequence_id": sequence_id,
                "segment_id": segment_id,
                "start": int(segment.get("start", 0)),
                "end": int(segment.get("end", 0)),
                "interaction_frame": int(segment.get("interaction_frame", 0)),
                "global_frame_index": segment_global_frame_index(sequence, segment),
                "text": segment.get("text"),
                "goal_type": segment.get("goal_type"),
                "goal_category": category,
                "mode": mode,
                "active_hand": segment.get("active_hand"),
                "acted_on_object_name": segment.get("acted_on_object_name"),
                "support_object_name": support_object_name,
                "acted_on_object_id": segment.get("acted_on_object_id"),
                "support_object_id": support_object_id,
                "body_goal": body_goal,
                "hand_goal": hand_goal,
                "global_orient_rotvec": global_orient,
                "body_mask_path": to_repo_relative(body_mask_path) if body_mask_path is not None else None,
                "hand_mask_path": to_repo_relative(hand_mask_path) if hand_mask_path is not None else None,
                "target_name": target_name,
                "seated_cluster_id": None,
                "movable_seated_cluster_id": None,
                "fixed_cluster_id": None,
                "given_active_objects": given_active_objects,
                "given_seated_name": given_seated_name,
                "movable_seated_name": movable_seated_name,
                "given_movable_min_dist": given_min_dist,
                "scene_chair_min_dist": scene_chair_min_dist,
            }
            segment_records.append(record)
            record_by_key[(sequence_id, segment_id)] = record

    return segment_records, record_by_key


def cluster_records(segment_records: List[dict], mode: str) -> List[dict]:
    assert mode in {"seat", "movable_seat", "object"}
    clusters: List[dict] = []
    threshold = 0.75 if mode in {"seat", "movable_seat"} else 0.45

    for record in segment_records:
        if mode == "object":
            if record.get("mode") != "object":
                continue
        elif mode == "movable_seat":
            if record.get("mode") != "seat" or not record.get("movable_seated_name"):
                continue
        else:
            if record.get("mode") != "seat" or record.get("movable_seated_name"):
                continue

        if mode in {"seat", "movable_seat"}:
            if record.get("body_goal") is None:
                continue
            compare_center = np.asarray([record["body_goal"][0], record["body_goal"][2]], dtype=np.float32)
            anchor_center = compare_center
        else:
            hand_goal = record.get("hand_goal")
            if hand_goal is None:
                continue
            compare_center = np.asarray([hand_goal[0], hand_goal[2]], dtype=np.float32)
            anchor_center = compare_center

        if mode == "seat":
            object_name = "seat"
        elif mode == "movable_seat":
            object_name = str(record.get("movable_seated_name") or "movable_seat")
        else:
            object_name = str(record.get("target_name") or "object")
        chosen_idx = None
        best_dist = None
        for idx, cluster in enumerate(clusters):
            if cluster["object_name"] != object_name:
                continue
            member_compare = cluster["compare_centers"]
            dist = min(float(np.linalg.norm(center - compare_center)) for center in member_compare)
            if dist <= threshold and (best_dist is None or dist < best_dist):
                chosen_idx = idx
                best_dist = dist

        if chosen_idx is None:
            clusters.append(
                {
                    "mode": mode,
                    "object_name": object_name,
                    "members": [(record["sequence_id"], int(record["segment_id"]))],
                    "anchor_centers": [anchor_center.copy()],
                    "compare_centers": [compare_center.copy()],
                }
            )
        else:
            clusters[chosen_idx]["members"].append((record["sequence_id"], int(record["segment_id"])))
            clusters[chosen_idx]["anchor_centers"].append(anchor_center.copy())
            clusters[chosen_idx]["compare_centers"].append(compare_center.copy())

    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(clusters):
            j = i + 1
            while j < len(clusters):
                a = clusters[i]
                b = clusters[j]
                if a["object_name"] != b["object_name"]:
                    j += 1
                    continue
                min_dist = min(
                    float(np.linalg.norm(center_a - center_b))
                    for center_a in a["anchor_centers"]
                    for center_b in b["anchor_centers"]
                )
                if min_dist <= threshold:
                    a["members"].extend(b["members"])
                    a["anchor_centers"].extend(b["anchor_centers"])
                    a["compare_centers"].extend(b["compare_centers"])
                    clusters.pop(j)
                    merged = True
                else:
                    j += 1
            i += 1

    if mode == "seat":
        for idx, cluster in enumerate(clusters, start=1):
            cluster["cluster_id"] = f"seated_{idx:04d}"
    elif mode == "movable_seat":
        for idx, cluster in enumerate(clusters, start=1):
            cluster["cluster_id"] = f"movable_seated_{idx:04d}"
    else:
        counters: Dict[str, int] = {}
        for cluster in clusters:
            object_name = str(cluster["object_name"])
            counters[object_name] = counters.get(object_name, 0) + 1
            cluster["cluster_id"] = f"fixed_{object_name}_{counters[object_name]:04d}"
    return clusters


def build_cluster_outputs(
    cluster_mode: str,
    clusters: List[dict],
    record_by_key: Dict[Tuple[str, int], dict],
    clearance: np.ndarray,
    out_root: Path,
) -> List[dict]:
    assert cluster_mode in {"seat", "movable_seat", "object"}
    if cluster_mode == "seat":
        cluster_subdir = "seated"
        vis_subdir = "seated"
    elif cluster_mode == "movable_seat":
        cluster_subdir = "movable_seated"
        vis_subdir = "movable_seated"
    else:
        cluster_subdir = "fixed"
        vis_subdir = "fixed"
    cluster_dir = out_root / cluster_subdir
    vis_dir = out_root / "vis" / vis_subdir
    cluster_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[dict] = []
    for cluster in clusters:
        body_union = np.zeros_like(clearance, dtype=bool)
        hand_union = np.zeros_like(clearance, dtype=bool)
        member_refs = []
        body_goals = []
        hand_goals = []

        for key in cluster["members"]:
            record = record_by_key[key]
            member_refs.append({"sequence_id": record["sequence_id"], "segment_id": int(record["segment_id"])})
            if record.get("body_goal") is not None:
                body_goals.append(record["body_goal"])
            if record.get("hand_goal") is not None:
                hand_goals.append(record["hand_goal"])
            body_mask_path = record.get("body_mask_path")
            hand_mask_path = record.get("hand_mask_path")
            if body_mask_path:
                body_union |= np.load(PROJECT_ROOT / body_mask_path).astype(bool)
            if hand_mask_path:
                hand_union |= np.load(PROJECT_ROOT / hand_mask_path).astype(bool)

        body_path = cluster_dir / f"{cluster['cluster_id']}_body.npy"
        save_mask(body_union, body_path)
        layers = [(body_union, "seat" if cluster_mode in {"seat", "movable_seat"} else "interactable")]
        payload = {
            "cluster_id": cluster["cluster_id"],
            "object_name": cluster["object_name"],
            "member_segment_refs": member_refs,
            "body_goal_list": body_goals,
            "body_union_mask_path": to_repo_relative(body_path),
        }

        if cluster_mode == "object":
            hand_path = cluster_dir / f"{cluster['cluster_id']}_hand.npy"
            save_mask(hand_union, hand_path)
            layers.append((hand_union, "object"))
            payload["hand_goal_list"] = hand_goals
            payload["hand_union_mask_path"] = to_repo_relative(hand_path)

        mask_to_vis(clearance, layers, vis_dir / f"{cluster['cluster_id']}.png")
        outputs.append(payload)

    return outputs


def assign_cluster_refs(
    clusters: List[dict],
    record_by_key: Dict[Tuple[str, int], dict],
    *,
    cluster_mode: str,
) -> None:
    if cluster_mode == "seat":
        field = "seated_cluster_id"
    elif cluster_mode == "movable_seat":
        field = "movable_seated_cluster_id"
    else:
        field = "fixed_cluster_id"
    for cluster in clusters:
        for key in cluster["members"]:
            record_by_key[key][field] = cluster["cluster_id"]


def build_locomotion_source_list(
    sequence_records: List[dict],
    *,
    window_frames: int,
    min_window_frames: int,
) -> List[dict]:
    outputs: List[dict] = []
    for sequence in sequence_records:
        motion_ref = sequence.get("human_motion_ref", {})
        total_frames = int(motion_ref.get("end", 0)) - int(motion_ref.get("start", 0)) + 1
        if total_frames <= 0:
            continue

        windows = []
        cursor = 0
        window_index = 0
        while cursor < total_frames:
            remaining = total_frames - cursor
            length = min(window_frames, remaining)
            if length < min_window_frames:
                break
            local_start = cursor
            local_end = cursor + length - 1
            overlapping_segment_ids = [
                int(segment["segment_id"])
                for segment in sequence.get("segment_list", [])
                if not (int(segment.get("end", 0)) < local_start or int(segment.get("start", 0)) > local_end)
            ]
            windows.append(
                {
                    "window_id": f"{sequence['sequence_id']}__clip{window_index:02d}",
                    "local_start": local_start,
                    "local_end": local_end,
                    "length": length,
                    "overlapping_segment_ids": overlapping_segment_ids,
                }
            )
            cursor += window_frames
            window_index += 1

        outputs.append(
            {
                "sequence_id": sequence["sequence_id"],
                "total_frames": total_frames,
                "candidate_windows": windows,
            }
        )
    return outputs


def build_scene_source_bank(
    dataset: str,
    scene_id: str,
    output_root: Path,
    source_scenes_name: str,
    episode_bank_name: str,
    loco_window_frames: int,
    loco_min_window_frames: int,
) -> Path:
    scene_root, bank_root = get_scene_paths(dataset, scene_id, output_root, source_scenes_name, episode_bank_name)
    if bank_root.exists():
        shutil.rmtree(bank_root)
    bank_root.mkdir(parents=True, exist_ok=True)

    scene_record = load_json(scene_root / "scene.json")
    sequence_records = load_scene_sequences(dataset, scene_root, scene_record)
    clearance = np.load(PROJECT_ROOT / scene_record["clearance_map_npy_path"]).astype(bool)

    segment_records, record_by_key = build_segment_goal_bank(
        dataset=dataset,
        scene_id=scene_id,
        sequence_records=sequence_records,
        clearance=clearance,
        grid_meta=scene_record["grid_meta"],
        out_root=bank_root,
    )

    seated_clusters = cluster_records(segment_records, "seat")
    movable_seated_clusters = cluster_records(segment_records, "movable_seat")
    fixed_clusters = cluster_records(segment_records, "object")
    assign_cluster_refs(seated_clusters, record_by_key, cluster_mode="seat")
    assign_cluster_refs(movable_seated_clusters, record_by_key, cluster_mode="movable_seat")
    assign_cluster_refs(fixed_clusters, record_by_key, cluster_mode="object")

    seated_cluster_list = build_cluster_outputs("seat", seated_clusters, record_by_key, clearance, bank_root)
    movable_seated_cluster_list = build_cluster_outputs("movable_seat", movable_seated_clusters, record_by_key, clearance, bank_root)
    fixed_cluster_list = build_cluster_outputs("object", fixed_clusters, record_by_key, clearance, bank_root)

    locomotion_source_list = []
    if is_loco_scene(dataset, scene_id):
        locomotion_source_list = build_locomotion_source_list(
            sequence_records,
            window_frames=loco_window_frames,
            min_window_frames=loco_min_window_frames,
        )

    grid_meta = dict(scene_record["grid_meta"])
    grid_resolution = float(grid_meta["x_max"] - grid_meta["x_min"]) / float(clearance.shape[0])
    payload = {
        "scene_id": scene_id,
        "dataset": dataset,
        "scene_kind": "locomotion" if is_loco_scene(dataset, scene_id) else "object_interaction",
        "grid_meta": grid_meta,
        "grid_resolution_m": grid_resolution,
        "occupancy_grid_path": scene_record["occupancy_grid_path"],
        "clearance_map_npy_path": scene_record["clearance_map_npy_path"],
        "clearance_map_vis_path": scene_record["clearance_map_vis_path"],
        "sequence_ids": [sequence["sequence_id"] for sequence in sequence_records],
        "seated_cluster_list": seated_cluster_list,
        "movable_seated_cluster_list": movable_seated_cluster_list,
        "fixed_cluster_list": fixed_cluster_list,
        "per_segment_goal_list": segment_records,
        "locomotion_source_list": locomotion_source_list,
    }
    out_json = bank_root / "scene_static.json"
    write_json(out_json, payload)
    return out_json


def _process_scene(payload: Tuple[str, str, Path, str, str, int, int]) -> str:
    dataset, scene_id, output_root, source_scenes_name, episode_bank_name, loco_window_frames, loco_min_window_frames = payload
    out_json = build_scene_source_bank(
        dataset,
        scene_id,
        output_root,
        source_scenes_name,
        episode_bank_name,
        loco_window_frames,
        loco_min_window_frames,
    )
    return to_repo_relative(out_json)


def main() -> None:
    args = parse_args()
    scene_ids = resolve_scene_ids(args)
    tasks = [
        (
            args.dataset,
            scene_id,
            args.output_root,
            str(args.source_scenes_name),
            str(args.episode_bank_name),
            int(args.loco_window_frames),
            int(args.loco_min_window_frames),
        )
        for scene_id in scene_ids
    ]
    if args.workers > 1 and len(tasks) > 1:
        with mp.Pool(processes=args.workers, maxtasksperchild=1) as pool:
            for out_json in pool.imap_unordered(_process_scene, tasks, chunksize=1):
                print(out_json)
    else:
        for payload in tasks:
            print(_process_scene(payload))


if __name__ == "__main__":
    main()
