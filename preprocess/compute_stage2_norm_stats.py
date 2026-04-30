from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSED_ROOT = Path("data") / "preprocessed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Stage-2 motion vector normalization statistics.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--kind", choices=["move_wait", "action", "both", "all_motion"], default="both")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PREPROCESSED_ROOT)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--samples-per-record", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def repo_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


class ArrayCache:
    def __init__(self) -> None:
        self._cache: dict[str, np.ndarray] = {}

    def load(self, path_str: str | None) -> np.ndarray | None:
        path = repo_path(path_str)
        if path is None:
            return None
        key = str(path)
        if key not in self._cache:
            self._cache[key] = np.load(path, mmap_mode="r")
        return self._cache[key]


def rotvec_to_6d(rotvec: np.ndarray) -> np.ndarray:
    values = np.asarray(rotvec, dtype=np.float32)
    if values.size == 0:
        return np.zeros((*values.shape[:-1], 0), dtype=np.float32)
    if values.shape[-1] % 3 != 0:
        raise ValueError(f"rotvec last dim must be divisible by 3, got {values.shape}")
    prefix = values.shape[:-1]
    joints = values.shape[-1] // 3
    flat = values.reshape(-1, 3)
    mat = R.from_rotvec(flat).as_matrix().astype(np.float32)
    six = mat[:, :2, :].reshape(*prefix, joints * 6)
    return six.astype(np.float32)


def pose_dim(dataset: str) -> int:
    if dataset == "trumans":
        return 3 + 6 + 21 * 6 + 15 * 6 + 15 * 6
    return 3 + 6 + 21 * 6


def read_motion_vector(record: dict[str, Any], frame_indices: np.ndarray, anchor_frame: int, cache: ArrayCache) -> np.ndarray:
    smplx = record["smplx"]
    transl = cache.load(smplx.get("transl_path"))
    global_orient = cache.load(smplx.get("global_orient_path"))
    body_pose = cache.load(smplx.get("body_pose_path"))
    if transl is None or global_orient is None or body_pose is None:
        raise FileNotFoundError(f"missing SMPL-X arrays for record {record.get('sequence_id')}")
    root = np.asarray(transl[frame_indices], dtype=np.float32) - np.asarray(transl[int(anchor_frame)], dtype=np.float32)[None]
    global_6d = rotvec_to_6d(np.asarray(global_orient[frame_indices], dtype=np.float32))
    body_6d = rotvec_to_6d(np.asarray(body_pose[frame_indices], dtype=np.float32))
    parts = [root, global_6d, body_6d]
    if str(record.get("dataset")) == "trumans":
        left_hand = cache.load(smplx.get("left_hand_pose_path"))
        right_hand = cache.load(smplx.get("right_hand_pose_path"))
        if left_hand is None or right_hand is None:
            raise FileNotFoundError(f"missing TRUMANS hand arrays for record {record.get('sequence_id')}")
        parts.append(rotvec_to_6d(np.asarray(left_hand[frame_indices], dtype=np.float32)))
        parts.append(rotvec_to_6d(np.asarray(right_hand[frame_indices], dtype=np.float32)))
    out = np.concatenate(parts, axis=-1).astype(np.float32)
    expected = pose_dim(str(record.get("dataset")))
    if out.shape[-1] != expected:
        raise ValueError(f"pose dim mismatch: got {out.shape[-1]} expected {expected}")
    return out


def read_sequence_motion_vector(record: dict[str, Any], cache: ArrayCache) -> np.ndarray:
    smplx = record["smplx"]
    transl = cache.load(smplx.get("transl_path"))
    global_orient = cache.load(smplx.get("global_orient_path"))
    body_pose = cache.load(smplx.get("body_pose_path"))
    if transl is None or global_orient is None or body_pose is None:
        raise FileNotFoundError(f"missing SMPL-X arrays for record {record.get('sequence_id')}")

    start = int(record["sequence_global_start"])
    end = int(record["sequence_global_end"])
    frame_indices = np.arange(start, end + 1, dtype=np.int64)
    root = np.asarray(transl[frame_indices], dtype=np.float32) - np.asarray(transl[start], dtype=np.float32)[None]
    global_6d = rotvec_to_6d(np.asarray(global_orient[frame_indices], dtype=np.float32))
    body_6d = rotvec_to_6d(np.asarray(body_pose[frame_indices], dtype=np.float32))
    parts = [root, global_6d, body_6d]
    if str(record.get("dataset")) == "trumans":
        left_hand = cache.load(smplx.get("left_hand_pose_path"))
        right_hand = cache.load(smplx.get("right_hand_pose_path"))
        if left_hand is None or right_hand is None:
            raise FileNotFoundError(f"missing TRUMANS hand arrays for record {record.get('sequence_id')}")
        parts.append(rotvec_to_6d(np.asarray(left_hand[frame_indices], dtype=np.float32)))
        parts.append(rotvec_to_6d(np.asarray(right_hand[frame_indices], dtype=np.float32)))
    out = np.concatenate(parts, axis=-1).astype(np.float32)
    expected = pose_dim(str(record.get("dataset")))
    if out.shape[-1] != expected:
        raise ValueError(f"pose dim mismatch: got {out.shape[-1]} expected {expected}")
    return out


def sample_move_wait_frames(record: dict[str, Any], rng: random.Random) -> tuple[np.ndarray, int]:
    target_start = rng.randint(int(record["target_start_min"]), int(record["target_start_max"]))
    H = int(record["history_frames"])
    W = int(record["future_frames"])
    seq_start = int(record["sequence_global_start"])
    local_indices = np.arange(target_start - H, target_start + W, dtype=np.int64)
    return seq_start + local_indices, seq_start + target_start


def sample_action_frames(record: dict[str, Any], rng: random.Random) -> tuple[np.ndarray, int]:
    H = int(record["history_frames"])
    seq_start = int(record["sequence_global_start"])
    if bool(record.get("first_segment_exception")):
        target_start = int(record["forced_target_start"])
    else:
        delta = rng.randint(int(record["aug_delta_min"]), int(record["aug_delta_max"]))
        target_start = int(record["action_start"]) - int(delta)
    action_end = int(record["action_end"])
    local_indices = np.arange(target_start - H, action_end + 1, dtype=np.int64)
    return seq_start + local_indices, seq_start + target_start


class RunningStats:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.count = 0
        self.sum = np.zeros((dim,), dtype=np.float64)
        self.sumsq = np.zeros((dim,), dtype=np.float64)
        self.min = np.full((dim,), np.inf, dtype=np.float64)
        self.max = np.full((dim,), -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1, self.dim)
        if arr.shape[0] == 0:
            return
        self.count += int(arr.shape[0])
        self.sum += arr.sum(axis=0)
        self.sumsq += np.square(arr).sum(axis=0)
        self.min = np.minimum(self.min, arr.min(axis=0))
        self.max = np.maximum(self.max, arr.max(axis=0))

    def finish(self) -> dict[str, Any]:
        denom = max(self.count, 1)
        mean = self.sum / float(denom)
        var = np.maximum(self.sumsq / float(denom) - np.square(mean), 1e-12)
        std = np.sqrt(var)
        return {
            "count": int(self.count),
            "dim": int(self.dim),
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "min": self.min.astype(float).tolist(),
            "max": self.max.astype(float).tolist(),
        }


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    stage2_root = args.output_root / args.dataset / "stage2"
    if args.kind == "all_motion":
        kinds = ["move_wait", "action"]
    else:
        kinds = ["move_wait", "action"] if args.kind == "both" else [args.kind]
    records: list[dict[str, Any]] = []
    for kind in kinds:
        path = stage2_root / f"{kind}_{args.split}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = load_json(path)
        for rec in payload.get("records", []):
            rec = dict(rec)
            rec["_manifest_kind"] = kind
            records.append(rec)
    return records


def unique_sequence_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in records:
        smplx = record.get("smplx", {})
        key = (
            record.get("sequence_id"),
            int(record.get("sequence_global_start", -1)),
            int(record.get("sequence_global_end", -1)),
            smplx.get("transl_path"),
            smplx.get("global_orient_path"),
            smplx.get("body_pose_path"),
            smplx.get("left_hand_pose_path"),
            smplx.get("right_hand_pose_path"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def maybe_limit_records(records: list[dict[str, Any]], max_records: int, rng: random.Random) -> list[dict[str, Any]]:
    if int(max_records) <= 0 or int(max_records) >= len(records):
        return records
    indices = list(range(len(records)))
    rng.shuffle(indices)
    selected = sorted(indices[: int(max_records)])
    return [records[idx] for idx in selected]


def progress_iter(values: list[dict[str, Any]], desc: str):
    if tqdm is None:
        return values
    return tqdm(values, desc=desc, unit="record")


def compute_stats(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(int(args.seed))
    records = load_records(args)
    if args.kind == "all_motion":
        records = unique_sequence_records(records)
    records = maybe_limit_records(records, int(args.max_records), rng)
    stats = RunningStats(pose_dim(args.dataset))
    cache = ArrayCache()
    skipped: dict[str, int] = {}
    for record in progress_iter(records, "stage2 norm stats"):
        try:
            if args.kind == "all_motion":
                values = read_sequence_motion_vector(record, cache)
                stats.update(values)
            else:
                for _ in range(max(1, int(args.samples_per_record))):
                    if record["_manifest_kind"] == "move_wait":
                        frames, anchor = sample_move_wait_frames(record, rng)
                    else:
                        frames, anchor = sample_action_frames(record, rng)
                    values = read_motion_vector(record, frames, anchor, cache)
                    stats.update(values)
        except Exception as exc:  # keep stats job robust over isolated bad records
            key = type(exc).__name__
            skipped[key] = skipped.get(key, 0) + 1
            continue
    result = stats.finish()
    if args.kind == "all_motion":
        mean = np.asarray(result["mean"], dtype=np.float32)
        std = np.asarray(result["std"], dtype=np.float32)
        mean[:3] = 0.0
        std[:3] = 1.0
        result["mean"] = mean.astype(float).tolist()
        result["std"] = std.astype(float).tolist()
    result.update(
        {
            "dataset": args.dataset,
            "split": args.split,
            "kind": args.kind,
            "num_records": len(records),
            "samples_per_record": int(args.samples_per_record),
            "seed": int(args.seed),
            "skipped": skipped,
            "representation": {
                "translation": "local root translation in the same layout used by Stage2MotionDataset",
                "all_motion_translation_stats": "root translation mean/std are fixed to 0/1",
                "global_orient": "axis-angle to 6D",
                "body_pose": "21 axis-angle joints to 6D",
                "hand_pose": "TRUMANS only, 15 axis-angle joints per hand to 6D",
                "betas": "unused",
            },
        }
    )
    return result


def compute_scene_scale_meta(args: argparse.Namespace) -> dict[str, Any]:
    scenes_root = args.output_root / args.dataset / "scenes_v2"
    x_half = 3.0
    z_half = 4.0
    y_min = 0.0
    y_max = 2.0
    num_scenes = 0
    if scenes_root.exists():
        for path in scenes_root.glob("*/scene.json"):
            try:
                payload = load_json(path)
                grid = payload.get("grid_meta") or {}
                x_half = max(x_half, abs(float(grid.get("x_min", -x_half))), abs(float(grid.get("x_max", x_half))))
                z_half = max(z_half, abs(float(grid.get("z_min", -z_half))), abs(float(grid.get("z_max", z_half))))
                y_min = min(y_min, float(grid.get("y_min", y_min)))
                y_max = max(y_max, float(grid.get("y_max", y_max)))
                num_scenes += 1
            except Exception:
                continue
    return {
        "dataset": args.dataset,
        "representation": "canonical_local_smplx_joints28_xyz",
        "normalization": "scene_extent_affine",
        "x_scale": float(max(x_half, 1e-6)),
        "z_scale": float(max(z_half, 1e-6)),
        "y_center": float((y_min + y_max) * 0.5),
        "y_scale": float(max((y_max - y_min) * 0.5, 1e-6)),
        "num_scenes": int(num_scenes),
    }


def mean_std_from_scene_scale(meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    joint_mean = np.array([0.0, float(meta["y_center"]), 0.0], dtype=np.float32)
    joint_std = np.array([float(meta["x_scale"]), float(meta["y_scale"]), float(meta["z_scale"])], dtype=np.float32)
    return np.tile(joint_mean, 28).astype(np.float32), np.tile(joint_std, 28).astype(np.float32)


def main() -> None:
    args = parse_args()
    stage2_root = args.output_root / args.dataset / "stage2"
    stage2_root.mkdir(parents=True, exist_ok=True)
    meta = compute_scene_scale_meta(args)
    mean, std = mean_std_from_scene_scale(meta)
    mean_path = stage2_root / "motion_mean.npy"
    std_path = stage2_root / "motion_std.npy"
    coord_path = stage2_root / "joint_coord_norm_meta.json"
    np.save(mean_path, mean)
    np.save(std_path, std)
    write_json(coord_path, meta)
    stats_meta = dict(meta)
    stats_meta.update(
        {
            "dim": int(mean.shape[0]),
            "mean_path": str(mean_path),
            "std_path": str(std_path),
            "coord_meta_path": str(coord_path),
            "joint_count": 28,
            "joint_dim": 3,
        }
    )
    out_path = stage2_root / "motion_stats_meta.json"
    write_json(out_path, stats_meta)
    print(f"wrote {coord_path}")
    print(f"wrote {mean_path}")
    print(f"wrote {std_path}")
    print(f"dim={mean.shape[0]} x_scale={meta['x_scale']} y_center={meta['y_center']} y_scale={meta['y_scale']} z_scale={meta['z_scale']}")


if __name__ == "__main__":
    main()
