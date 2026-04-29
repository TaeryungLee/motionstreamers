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
DEFAULT_PREPROCESSED_ROOT = PROJECT_ROOT / "data" / "preprocessed"


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


def main() -> None:
    args = parse_args()
    result = compute_stats(args)
    stage2_root = args.output_root / args.dataset / "stage2"
    if args.kind == "all_motion":
        mean = np.asarray(result["mean"], dtype=np.float32)
        std = np.asarray(result["std"], dtype=np.float32)
        stage2_root.mkdir(parents=True, exist_ok=True)
        mean_path = stage2_root / "motion_mean.npy"
        std_path = stage2_root / "motion_std.npy"
        np.save(mean_path, mean)
        np.save(std_path, std)
        meta = dict(result)
        meta.pop("mean", None)
        meta.pop("std", None)
        meta.update(
            {
                "mean_path": str(mean_path),
                "std_path": str(std_path),
                "root_dims": [0, 1, 2],
                "root_mean": [0.0, 0.0, 0.0],
                "root_std": [1.0, 1.0, 1.0],
            }
        )
        out_path = stage2_root / "motion_stats_meta.json"
        write_json(out_path, meta)
    else:
        out_path = stage2_root / f"normalization_stats_{args.split}_{args.kind}.json"
        write_json(out_path, result)
    print(f"wrote {out_path}")
    print(f"count={result['count']} dim={result['dim']} records={result['num_records']} skipped={result['skipped']}")


if __name__ == "__main__":
    main()
