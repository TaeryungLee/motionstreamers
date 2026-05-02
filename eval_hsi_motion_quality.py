from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import math
import os
import pickle
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PREPROCESSED_ROOT = Path("data") / "preprocessed"
DEFAULT_PRIOR_MDM_ROOT = Path("externals") / "priorMDM"
FOOT_LEFT_INDICES = (7, 10)
FOOT_RIGHT_INDICES = (8, 11)


def repo_path(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def display_path(path: Path | str) -> str:
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def load_json(path: Path | str) -> Any:
    return json.loads(repo_path(path).read_text())


def write_json(path: Path | str, payload: Any) -> None:
    out = repo_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))


@contextlib.contextmanager
def working_directory(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@contextlib.contextmanager
def prior_mdm_context(prior_mdm_root: Path) -> Iterator[None]:
    root = repo_path(prior_mdm_root)
    added = False
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
        added = True
    with working_directory(root):
        try:
            yield
        finally:
            if added:
                try:
                    sys.path.remove(str(root))
                except ValueError:
                    pass


def result_paths(args: argparse.Namespace) -> list[Path]:
    if args.results_glob:
        paths = [repo_path(path) for pattern in args.results_glob for path in sorted(glob.glob(str(pattern)))]
    else:
        root = repo_path(args.generated_root)
        paths = sorted((root / "results").glob("**/*.json"))
        if not paths:
            paths = sorted(root.glob("**/*.json"))
    skip_names = {"args.json", "summary.json", "metrics.json", "metrics_by_goal_type.json"}
    return [path for path in paths if path.name not in skip_names]


def resolve_trajectory_path(result: dict[str, Any], result_path: Path) -> Path | None:
    candidates = []
    for key in ("trajectory_path", "traj_path", "motion_path", "npz_path"):
        value = result.get(key)
        if value:
            candidates.append(repo_path(value))
            candidates.append(result_path.parent / str(value))
    stem = result_path.stem
    generated_root = result_path.parents[2] if len(result_path.parents) >= 3 and result_path.parent.parent.name == "results" else None
    if generated_root is not None:
        scene_id = result.get("scene_id", result_path.parent.name)
        candidates.append(generated_root / "trajectories" / str(scene_id) / f"{stem}.npz")
    for path in candidates:
        if path.exists():
            return path
    return None


def load_generated_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def pick_first_array(payload: dict[str, np.ndarray], keys: list[str]) -> np.ndarray | None:
    for key in keys:
        if key in payload:
            return np.asarray(payload[key])
    return None


def load_generated_joints28(payload: dict[str, np.ndarray]) -> np.ndarray | None:
    arr = pick_first_array(
        payload,
        [
            "ego_joints28_world",
            "joints28_world",
            "ego_joints_world",
            "joints_world",
            "pred_joints28_world",
            "generated_joints28_world",
        ],
    )
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[-1] == 84:
        arr = arr.reshape(arr.shape[0], 28, 3)
    if arr.ndim != 3 or arr.shape[1:] != (28, 3):
        raise ValueError(f"generated joints shape must be [T,28,3] or [T,84], got {arr.shape}")
    return arr


def character_global_start(dataset: str, scene_id: str, char: dict[str, Any], root: Path) -> tuple[int, int, str]:
    if isinstance(char.get("source_window"), dict):
        window = char["source_window"]
        sequence_id = str(window["sequence_id"])
        local_start = int(window["local_start"])
        local_end = int(window["local_end"])
    else:
        goals = char.get("goal_sequence") or []
        if not goals:
            raise ValueError(f"character {char.get('character_id')} has no goal_sequence")
        first_source = goals[0].get("source_segment", {})
        last_source = goals[-1].get("source_segment", {})
        sequence_id = str(char.get("sequence_id") or first_source["sequence_id"])
        local_start = int(first_source["start"])
        local_end = int(last_source["end"])
    seq_path = root / dataset / "scenes_v2" / scene_id / "sequences" / f"{sequence_id}.json"
    seq = load_json(seq_path)
    seq_global_start = int(seq["human_motion_ref"]["start"])
    return seq_global_start + local_start, seq_global_start + local_end, sequence_id


def load_gt_joints28(dataset: str, episode_path: Path, generated_len: int, root: Path) -> np.ndarray:
    episode = load_json(episode_path)
    scene_id = str(episode["scene_id"])
    ego_id = str(episode.get("ego_character_id") or "char_00")
    ego_char = next(ch for ch in episode["character_assignments"] if str(ch.get("character_id")) == ego_id)
    start, end, _ = character_global_start(dataset, scene_id, ego_char, root)
    cache_path = root / dataset / "joints28" / "joints28.npy"
    joints = np.load(repo_path(cache_path), mmap_mode="r")
    indices = np.arange(start, start + int(generated_len), dtype=np.int64)
    indices = np.clip(indices, start, min(end, joints.shape[0] - 1))
    return np.asarray(joints[indices], dtype=np.float32)


def sequence_motion_ref(dataset: str, scene_id: str, sequence_id: str, root: Path) -> dict[str, Any]:
    seq_path = root / dataset / "scenes_v2" / scene_id / "sequences" / f"{sequence_id}.json"
    seq = load_json(seq_path)
    return dict(seq["human_motion_ref"])


def load_gt_smplx_params(dataset: str, episode_path: Path, generated_len: int, root: Path) -> dict[str, np.ndarray] | None:
    episode = load_json(episode_path)
    scene_id = str(episode["scene_id"])
    ego_id = str(episode.get("ego_character_id") or "char_00")
    ego_char = next(ch for ch in episode["character_assignments"] if str(ch.get("character_id")) == ego_id)
    start, end, sequence_id = character_global_start(dataset, scene_id, ego_char, root)
    motion = sequence_motion_ref(dataset, scene_id, sequence_id, root)
    smplx_ref = dict(motion.get("smplx") or {})
    required = {
        "global_orient": smplx_ref.get("global_orient_path"),
        "body_pose": smplx_ref.get("body_pose_path"),
        "transl": smplx_ref.get("transl_path"),
    }
    if any(value is None for value in required.values()):
        return None
    indices = np.arange(start, start + int(generated_len), dtype=np.int64)
    indices = np.clip(indices, start, end)
    out: dict[str, np.ndarray] = {}
    for name, path in required.items():
        arr = np.load(repo_path(path), mmap_mode="r")
        out[name] = np.asarray(arr[indices], dtype=np.float32)
    return out


def load_smplx_fit(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    required = ("global_orient", "body_pose", "transl")
    if not all(key in loaded for key in required):
        raise ValueError(f"SMPL-X fit cache is missing required keys: {display_path(path)}")
    return {key: np.asarray(loaded[key], dtype=np.float32) for key in required}


def fit_joints28_to_smplx_cached(
    joints28_world: np.ndarray,
    cache_path: Path,
    device: torch.device,
    smooth_weight: float,
) -> dict[str, np.ndarray]:
    cache_path = repo_path(cache_path)
    if not cache_path.exists():
        from models.full_ours_runtime import fit_joints28_episode_to_smplx

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fit_joints28_episode_to_smplx(
            np.asarray(joints28_world, dtype=np.float32),
            cache_path,
            device,
            smooth_weight=float(smooth_weight),
        )
    return load_smplx_fit(cache_path)


def generated_fit_cache_path(args: argparse.Namespace, result: dict[str, Any], result_path: Path, traj_path: Path, num_frames: int) -> Path:
    scene_id = str(result.get("scene_id") or result_path.parent.name)
    stat = traj_path.stat()
    key_text = f"{traj_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{int(num_frames)}"
    key = hashlib.sha1(key_text.encode("utf-8")).hexdigest()[:12]
    return repo_path(Path("outputs") / str(args.run_name) / "smplx_fit_cache" / "generated" / scene_id / f"{result_path.stem}_{key}.pkl")


def convert_y_up_rotations_to_prior_coords(rot_mats: torch.Tensor) -> torch.Tensor:
    # Our world is x/z-horizontal and y-up. PriorMDM BABEL rfeats use z-up.
    # Rotate coordinates about +x so (x, z, y) becomes (x, -z, y) in a right-handed z-up frame.
    p = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=rot_mats.dtype, device=rot_mats.device)
    return p @ rot_mats @ p.transpose(0, 1)


def smplx_params_to_babel135(params: dict[str, np.ndarray], prior_mdm_root: Path, device: torch.device) -> np.ndarray:
    global_orient = np.asarray(params["global_orient"], dtype=np.float32)
    body_pose = np.asarray(params["body_pose"], dtype=np.float32)
    transl = np.asarray(params["transl"], dtype=np.float32)
    T = int(min(len(global_orient), len(body_pose), len(transl)))
    if T <= 0:
        return np.zeros((0, 135), dtype=np.float32)
    body_pose = body_pose[:T, :63]
    poses = np.concatenate([global_orient[:T, :3], body_pose], axis=1).reshape(T, 22, 3)
    trans_prior = np.stack([transl[:T, 0], -transl[:T, 2], transl[:T, 1]], axis=-1).astype(np.float32)
    with prior_mdm_context(prior_mdm_root):
        from data_loaders.amass.transforms.rots2rfeats.globvelandy import Globalvelandy
        from data_loaders.amass.transforms.smpl import RotTransDatastruct
        from data_loaders.amass.tools_teach.geometry import axis_angle_to_matrix

        pose_t = torch.from_numpy(poses).to(device=device, dtype=torch.float32)
        trans_t = torch.from_numpy(trans_prior).to(device=device, dtype=torch.float32)
        mats = axis_angle_to_matrix(pose_t.reshape(-1, 3)).reshape(T, 22, 3, 3)
        mats = convert_y_up_rotations_to_prior_coords(mats)
        norm_path = "./data_loaders/amass/deps/transforms/rots2rfeats/globalvelandy/rot6d/babel-amass"
        transform = Globalvelandy(path=norm_path, normalization=True, canonicalize=True, offset=True).to(device)
        feats = transform(RotTransDatastruct(rots=mats, trans=trans_t))
    return feats.detach().cpu().numpy().astype(np.float32)


def segment_bounds(goal: dict[str, Any], total_len: int) -> tuple[int, int]:
    start = int(goal.get("start_frame", 0))
    end = int(goal.get("end_frame", total_len - 1))
    start = max(0, min(start, total_len - 1))
    end = max(start, min(end, total_len - 1))
    return start, end + 1


def is_completed(goal: dict[str, Any]) -> bool:
    return str(goal.get("status", "")).lower() == "completed"


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(seq, dtype=np.float32)
    if len(values) == int(target_len):
        return values
    if len(values) <= 1:
        return np.repeat(values[:1], int(target_len), axis=0)
    src = np.linspace(0.0, 1.0, len(values))
    dst = np.linspace(0.0, 1.0, int(target_len))
    flat = values.reshape(len(values), -1)
    out = np.stack([np.interp(dst, src, flat[:, dim]) for dim in range(flat.shape[1])], axis=-1)
    return out.reshape((int(target_len),) + values.shape[1:]).astype(np.float32)


def joints28_embedding(seq: np.ndarray, sample_frames: int) -> np.ndarray:
    joints = resample_sequence(np.asarray(seq, dtype=np.float32), sample_frames)
    root = joints[:, :1]
    local = joints - root
    vel = np.diff(joints[:, [0, 7, 8, 10, 11]], axis=0, prepend=joints[:1, [0, 7, 8, 10, 11]])
    summary = np.concatenate(
        [
            local.mean(axis=0).reshape(-1),
            local.std(axis=0).reshape(-1),
            vel.mean(axis=0).reshape(-1),
            vel.std(axis=0).reshape(-1),
            (joints[:, 0].max(axis=0) - joints[:, 0].min(axis=0)).reshape(-1),
        ],
        axis=0,
    )
    return summary.astype(np.float32)


def root_jerk(seq: np.ndarray) -> float:
    joints = np.asarray(seq, dtype=np.float32)
    if len(joints) < 4:
        return 0.0
    root = joints[:, 0]
    jerk = np.diff(root, n=3, axis=0)
    return float(np.linalg.norm(jerk, axis=-1).mean())


def foot_sliding(seq: np.ndarray, contact_height_margin_m: float, contact_vel_threshold_mpf: float) -> dict[str, float]:
    joints = np.asarray(seq, dtype=np.float32)
    if len(joints) < 2:
        return {"foot_sliding_mpf": 0.0, "foot_contact_ratio": 0.0}
    feet = joints[:, list(FOOT_LEFT_INDICES + FOOT_RIGHT_INDICES)]
    heights = feet[:, :, 1]
    min_h = heights.min(axis=1, keepdims=True)
    contact = heights <= (min_h + float(contact_height_margin_m))
    vel = np.linalg.norm(np.diff(feet[:, :, [0, 2]], axis=0), axis=-1)
    contact_next = contact[1:]
    sliding = vel[contact_next]
    if sliding.size == 0:
        return {"foot_sliding_mpf": 0.0, "foot_contact_ratio": float(contact.mean())}
    over = np.maximum(sliding - float(contact_vel_threshold_mpf), 0.0)
    return {"foot_sliding_mpf": float(over.mean()), "foot_contact_ratio": float(contact.mean())}


def activation_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(values, axis=0), np.cov(values, rowvar=False)


def frechet_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    from scipy import linalg

    mu1, sigma1 = activation_statistics(a)
    mu2, sigma2 = activation_statistics(b)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def diversity(values: np.ndarray, times: int, rng: np.random.Generator) -> float:
    if len(values) < 2:
        return 0.0
    pairs = min(int(times), len(values) * (len(values) - 1))
    i = rng.integers(0, len(values), size=pairs)
    j = rng.integers(0, len(values), size=pairs)
    same = i == j
    if same.any():
        j[same] = (j[same] + 1) % len(values)
    return float(np.linalg.norm(values[i] - values[j], axis=-1).mean())


def prior_mdm_embeddings(motions: list[np.ndarray], prior_mdm_root: Path, device: torch.device, batch_size: int) -> np.ndarray:
    if not motions:
        return np.zeros((0, 512), dtype=np.float32)
    with prior_mdm_context(prior_mdm_root):
        from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper

        wrapper = EvaluatorMDMWrapper("babel", device)
        chunks = []
        for start in tqdm(range(0, len(motions), int(batch_size)), desc="PriorMDM embeddings", unit="batch"):
            batch = motions[start : start + int(batch_size)]
            max_len = max(len(item) for item in batch)
            padded = np.zeros((len(batch), max_len, 135), dtype=np.float32)
            lengths = np.asarray([len(item) for item in batch], dtype=np.int64)
            for idx, item in enumerate(batch):
                padded[idx, : len(item)] = item
            with torch.no_grad():
                emb = wrapper.get_motion_embeddings(
                    torch.from_numpy(padded).to(device=device),
                    torch.from_numpy(lengths).to(device=device),
                )
            chunks.append(emb.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


@dataclass
class EvalSegments:
    gen_joints: list[np.ndarray]
    gt_joints: list[np.ndarray]
    gen_motion135: list[np.ndarray]
    gt_motion135: list[np.ndarray]
    records: list[dict[str, Any]]


def append_segment_metrics(
    segment: np.ndarray,
    prefix: str,
    sums: dict[str, float],
    counts: dict[str, int],
    args: argparse.Namespace,
) -> None:
    jerk = root_jerk(segment)
    foot = foot_sliding(segment, float(args.contact_height_margin_m), float(args.contact_velocity_threshold_mpf))
    values = {f"{prefix}_root_jerk_mpf3": jerk, f"{prefix}_foot_sliding_mpf": foot["foot_sliding_mpf"]}
    for key, value in values.items():
        sums[key] = sums.get(key, 0.0) + float(value)
        counts[key] = counts.get(key, 0) + 1


def collect_segments(args: argparse.Namespace) -> tuple[EvalSegments, dict[str, Any], dict[str, dict[str, Any]]]:
    root = repo_path(args.preprocessed_root)
    paths = result_paths(args)
    if args.max_results > 0:
        paths = paths[: int(args.max_results)]
    segments = EvalSegments([], [], [], [], [])
    aggregate_counts: dict[str, Any] = {
        "num_results": 0,
        "num_results_with_error": 0,
        "total_goals": 0,
        "completed_goals": 0,
        "failed_goals": 0,
        "timeout_count": 0,
        "action_interrupt_count": 0,
        "wait_frames": 0,
        "dynamic_collision_ratios": [],
        "static_collision_ratios": [],
        "missing_gt": 0,
        "missing_generated_joints": 0,
        "missing_prior_mdm_features": 0,
    }
    by_goal_type: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "completed": 0, "final_distance_m": []})
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")

    for result_path in tqdm(paths, desc="loading results", unit="episode"):
        try:
            result = load_json(result_path)
            aggregate_counts["num_results"] += 1
            traj_path = resolve_trajectory_path(result, result_path)
            if traj_path is None:
                raise FileNotFoundError(f"missing trajectory npz for {display_path(result_path)}")
            payload = load_generated_npz(traj_path)
            gen_joints = load_generated_joints28(payload)
            if gen_joints is None:
                aggregate_counts["missing_generated_joints"] += 1
                continue
            episode_ref = result.get("episode_path")
            if not episode_ref:
                aggregate_counts["missing_gt"] += 1
                continue
            episode_path = repo_path(episode_ref)
            gt_joints = load_gt_joints28(str(args.dataset), episode_path, len(gen_joints), root)
            gt_motion135_all = None
            gen_motion135_all = None
            if bool(args.compute_prior_mdm):
                gen_fit_path = generated_fit_cache_path(args, result, result_path, traj_path, len(gen_joints))
                gen_smplx = fit_joints28_to_smplx_cached(
                    gen_joints,
                    gen_fit_path,
                    device,
                    smooth_weight=float(args.fit_smooth_weight),
                )
                gen_motion135_all = smplx_params_to_babel135(gen_smplx, args.prior_mdm_root, device)
                gt_smplx = load_gt_smplx_params(str(args.dataset), episode_path, len(gen_joints), root)
                if gt_smplx is not None:
                    gt_motion135_all = smplx_params_to_babel135(gt_smplx, args.prior_mdm_root, device)

            per_goal = list(result.get("per_goal") or [])
            aggregate_counts["total_goals"] += int(result.get("total_goals", len(per_goal)))
            aggregate_counts["completed_goals"] += int(result.get("completed_goals", sum(is_completed(goal) for goal in per_goal)))
            aggregate_counts["failed_goals"] += int(result.get("failed_goals", max(0, len(per_goal) - sum(is_completed(goal) for goal in per_goal))))
            aggregate_counts["timeout_count"] += int(result.get("timeout_count", 0))
            aggregate_counts["action_interrupt_count"] += int(result.get("action_interrupt_count", 0))
            aggregate_counts["wait_frames"] += int(result.get("wait_frames", 0))
            if "dynamic_collision_ratio" in result:
                aggregate_counts["dynamic_collision_ratios"].append(float(result["dynamic_collision_ratio"]))
            if "static_collision_ratio" in result:
                aggregate_counts["static_collision_ratios"].append(float(result["static_collision_ratio"]))

            for goal_index, goal in enumerate(per_goal):
                goal_type = str(goal.get("normalized_goal_type") or goal.get("goal_type") or "unknown")
                by_goal_type[goal_type]["total"] += 1
                by_goal_type[goal_type]["completed"] += int(is_completed(goal))
                if goal.get("final_distance_m") is not None:
                    by_goal_type[goal_type]["final_distance_m"].append(float(goal["final_distance_m"]))
                if not is_completed(goal):
                    continue
                start, end = segment_bounds(goal, len(gen_joints))
                if end - start < int(args.min_segment_frames):
                    continue
                if end - start > int(args.max_segment_frames):
                    end = start + int(args.max_segment_frames)
                gen_seg = gen_joints[start:end]
                gt_seg = gt_joints[start:end]
                segments.gen_joints.append(gen_seg)
                segments.gt_joints.append(gt_seg)
                append_segment_metrics(gen_seg, "gen", metric_sums, metric_counts, args)
                append_segment_metrics(gt_seg, "gt", metric_sums, metric_counts, args)
                rec = {
                    "result_path": display_path(result_path),
                    "trajectory_path": display_path(traj_path),
                    "episode_path": display_path(episode_path),
                    "scene_id": result.get("scene_id"),
                    "episode_id": result.get("episode_id"),
                    "goal_index": int(goal_index),
                    "goal_type": goal_type,
                    "start": int(start),
                    "end": int(end),
                    "frames": int(end - start),
                }
                if bool(args.compute_prior_mdm) and gen_motion135_all is not None and gt_motion135_all is not None:
                    max_motion_end = min(end, len(gen_motion135_all), len(gt_motion135_all))
                    if max_motion_end - start >= int(args.min_segment_frames):
                        segments.gen_motion135.append(np.asarray(gen_motion135_all[start:max_motion_end], dtype=np.float32))
                        segments.gt_motion135.append(np.asarray(gt_motion135_all[start:max_motion_end], dtype=np.float32))
                        rec["prior_mdm_usable"] = True
                    else:
                        aggregate_counts["missing_prior_mdm_features"] += 1
                        rec["prior_mdm_usable"] = False
                else:
                    aggregate_counts["missing_prior_mdm_features"] += 1
                    rec["prior_mdm_usable"] = False
                segments.records.append(rec)
        except Exception as exc:
            aggregate_counts["num_results_with_error"] += 1
            segments.records.append({"result_path": display_path(result_path), "error": repr(exc)})

    for key, total in metric_sums.items():
        aggregate_counts[key] = float(total / max(metric_counts.get(key, 0), 1))
    return segments, aggregate_counts, dict(by_goal_type)


def compute_metrics(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    segments, counts, by_goal_type = collect_segments(args)
    metrics: dict[str, Any] = {
        "dataset": str(args.dataset),
        "generated_root": display_path(args.generated_root),
        "num_results": int(counts["num_results"]),
        "num_results_with_error": int(counts["num_results_with_error"]),
        "total_goals": int(counts["total_goals"]),
        "completed_goals": int(counts["completed_goals"]),
        "failed_goals": int(counts["failed_goals"]),
        "goal_success_rate": float(counts["completed_goals"] / max(counts["total_goals"], 1)),
        "timeout_count": int(counts["timeout_count"]),
        "action_interrupt_count": int(counts["action_interrupt_count"]),
        "wait_frames": int(counts["wait_frames"]),
        "goal_segments_used": int(len(segments.gen_joints)),
        "prior_mdm_segments_used": int(len(segments.gen_motion135)),
        "missing_gt": int(counts["missing_gt"]),
        "missing_generated_joints": int(counts["missing_generated_joints"]),
        "missing_prior_mdm_features": int(counts["missing_prior_mdm_features"]),
    }
    for key in ("dynamic_collision_ratios", "static_collision_ratios"):
        values = np.asarray(counts[key], dtype=np.float32)
        if len(values):
            metrics[f"mean_{key[:-1]}"] = float(values.mean())
    for key, value in counts.items():
        if key.startswith(("gen_", "gt_")):
            metrics[key] = float(value)

    if segments.gen_joints and len(segments.gen_joints) == len(segments.gt_joints):
        gen_emb = np.stack([joints28_embedding(item, int(args.joints_embedding_frames)) for item in segments.gen_joints], axis=0)
        gt_emb = np.stack([joints28_embedding(item, int(args.joints_embedding_frames)) for item in segments.gt_joints], axis=0)
        metrics["joints28_fid"] = frechet_distance(gt_emb, gen_emb) if len(gen_emb) >= 2 and len(gt_emb) >= 2 else None
        metrics["joints28_diversity"] = diversity(gen_emb, int(args.diversity_times), rng)
        metrics["gt_joints28_diversity"] = diversity(gt_emb, int(args.diversity_times), rng)

    if bool(args.compute_prior_mdm) and segments.gen_motion135 and len(segments.gen_motion135) == len(segments.gt_motion135):
        device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
        gen_act = prior_mdm_embeddings(segments.gen_motion135, args.prior_mdm_root, device, int(args.prior_mdm_batch_size))
        gt_act = prior_mdm_embeddings(segments.gt_motion135, args.prior_mdm_root, device, int(args.prior_mdm_batch_size))
        if len(gen_act) >= 2 and len(gt_act) >= 2:
            metrics["prior_mdm_babel_fid"] = frechet_distance(gt_act, gen_act)
        else:
            metrics["prior_mdm_babel_fid"] = None
        metrics["prior_mdm_babel_diversity"] = diversity(gen_act, int(args.diversity_times), rng)
        metrics["gt_prior_mdm_babel_diversity"] = diversity(gt_act, int(args.diversity_times), rng)

    out_dir = repo_path(Path("outputs") / str(args.run_name))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)

    by_goal_payload: dict[str, Any] = {}
    for goal_type, values in by_goal_type.items():
        final_dist = np.asarray(values.get("final_distance_m") or [], dtype=np.float32)
        by_goal_payload[goal_type] = {
            "total": int(values["total"]),
            "completed": int(values["completed"]),
            "success_rate": float(values["completed"] / max(values["total"], 1)),
            "mean_final_distance_m": float(final_dist.mean()) if len(final_dist) else None,
        }
    write_json(out_dir / "metrics_by_goal_type.json", by_goal_payload)
    seg_path = out_dir / "segments_used.jsonl"
    with seg_path.open("w", encoding="utf-8") as handle:
        for record in segments.records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated HSI/baseline episodes with goal-level motion quality metrics.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--results-glob", nargs="*", default=None)
    parser.add_argument("--preprocessed-root", type=Path, default=DEFAULT_PREPROCESSED_ROOT)
    parser.add_argument("--prior-mdm-root", type=Path, default=DEFAULT_PRIOR_MDM_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-prior-mdm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-prior-mdm", action="store_true")
    parser.add_argument("--prior-mdm-batch-size", type=int, default=32)
    parser.add_argument("--fit-smooth-weight", type=float, default=0.0)
    parser.add_argument("--min-segment-frames", type=int, default=20)
    parser.add_argument("--max-segment-frames", type=int, default=250)
    parser.add_argument("--joints-embedding-frames", type=int, default=64)
    parser.add_argument("--diversity-times", type=int, default=10000)
    parser.add_argument("--contact-height-margin-m", type=float, default=0.03)
    parser.add_argument("--contact-velocity-threshold-mpf", type=float, default=0.015)
    parser.add_argument("--max-results", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if bool(args.require_prior_mdm):
        args.compute_prior_mdm = True
    return args


def main() -> None:
    args = parse_args()
    metrics = compute_metrics(args)
    if bool(args.require_prior_mdm) and int(metrics.get("prior_mdm_segments_used", 0)) <= 0:
        raise RuntimeError("PriorMDM was required, but no usable motion_135/SMPL-X segments were found.")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
