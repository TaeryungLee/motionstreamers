from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.build_stage2_joints28_cache import SMPLX_SELECTED_JOINTS_28

DEFAULT_RAW_ROOT = Path("data") / "raw" / "babel"
DEFAULT_OUTPUT_ROOT = Path("data") / "preprocessed"
DEFAULT_SMPLX_MODEL_DIR = Path("smpl_models")

BABEL_TO_OURS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)

MOVE_LABELS = {
    "walk",
    "walk forward",
    "walk backwards",
    "walk backward",
    "walk back",
    "run",
    "run forward",
    "jog",
    "step forward",
    "step back",
    "step backward",
    "step backwards",
    "turn",
    "turn left",
    "turn right",
    "turn around",
    "turn back",
    "turn around left",
    "turn around right",
    "stand",
    "stand still",
    "stand in place",
    "stand with arms down",
    "stop",
}

MOVE_CATEGORIES = {
    "walk",
    "run",
    "jog",
    "turn",
    "stand",
    "step",
    "forward movement",
    "backwards movement",
    "sideways movement",
}

EXCLUDE_LABEL_PARTS = (
    "transition",
    "tpose",
    "t pose",
    "apose",
    "a pose",
    "sit",
    "stand up",
    "jump",
    "dance",
    "pick",
    "place",
    "take",
    "throw",
    "catch",
    "crawl",
    "kneel",
    "squat",
    "climb",
    "hop",
    "kick",
    "punch",
    "wave",
    "gesture",
    "stretch",
    "exercise",
)

EXCLUDE_CATEGORIES = {
    "transition",
    "t pose",
    "a pose",
    "sit",
    "stand up",
    "jump",
    "dance",
    "take/pick something up",
    "place something",
    "throw",
    "catch",
    "interact with/use object",
    "touch object",
    "gesture",
    "exercise/training",
}


def repo_path(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def repo_rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BABEL Stage-2 MoveWait joints28 cache and manifests.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smplx-model-dir", type=Path, default=DEFAULT_SMPLX_MODEL_DIR)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--history-frames", type=int, default=5)
    parser.add_argument("--move-future-frames", type=int, default=30)
    parser.add_argument("--min-segment-frames", type=int, default=0)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--gender", choices=["male", "female", "neutral"], default="neutral")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug-vis-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def output_splits(split: str) -> list[tuple[str, str]]:
    if split == "train":
        return [("train", "train")]
    if split == "test":
        return [("val", "test")]
    return [("train", "train"), ("val", "test")]


def normalize_label(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().replace("_", " ").split())


def is_move_wait_label(label: dict[str, Any]) -> bool:
    names = {
        normalize_label(label.get("proc_label")),
        normalize_label(label.get("raw_label")),
    }
    cats = {normalize_label(cat) for cat in (label.get("act_cat") or [])}
    joined = " ".join(sorted(names | cats))
    if any(part in joined for part in EXCLUDE_LABEL_PARTS):
        return False
    if cats & EXCLUDE_CATEGORIES:
        return False
    if names & MOVE_LABELS:
        return True
    if cats & MOVE_CATEGORIES:
        return True
    return False


def goal_type_for_label(label: dict[str, Any]) -> str:
    names = {normalize_label(label.get("proc_label")), normalize_label(label.get("raw_label"))}
    cats = {normalize_label(cat) for cat in (label.get("act_cat") or [])}
    joined = " ".join(sorted(names | cats))
    if "stand" in joined or "stop" in joined:
        return "stand_still"
    return "walk"


def segment_bounds(label: dict[str, Any], fps: float, num_frames: int) -> tuple[int, int] | None:
    if "start_t" not in label or "end_t" not in label:
        return None
    start = int(round(float(label["start_t"]) * float(fps)))
    end_exclusive = int(round(float(label["end_t"]) * float(fps)))
    start = max(0, min(start, int(num_frames) - 1))
    end = max(0, min(end_exclusive - 1, int(num_frames) - 1))
    if end < start:
        return None
    return start, end


def convert_points_babel_to_ours(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    return (values @ BABEL_TO_OURS.T).astype(np.float32)


def convert_rotvec_babel_to_ours(rotvec: np.ndarray) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=np.float32).reshape(-1, 3)
    mats = R.from_rotvec(rv).as_matrix().astype(np.float32)
    converted = BABEL_TO_OURS[None] @ mats @ BABEL_TO_OURS.T[None]
    return R.from_matrix(converted).as_rotvec().astype(np.float32).reshape(rotvec.shape)


def make_smplx_model(args: argparse.Namespace, device: torch.device):
    import smplx

    return smplx.create(
        str(repo_path(args.smplx_model_dir)),
        model_type="smplx",
        gender=str(args.gender),
        ext="npz",
        num_betas=10,
        use_pca=False,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
        batch_size=int(args.chunk_size),
    ).to(device).eval()


def forward_joints28(
    model: torch.nn.Module,
    poses: np.ndarray,
    trans: np.ndarray,
    selected: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)
    total = int(poses.shape[0])
    out = np.empty((total, 28, 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, total, int(chunk_size)):
            end = min(start + int(chunk_size), total)
            batch = int(end - start)
            run_batch = int(chunk_size)
            p_np = np.zeros((run_batch, poses.shape[1]), dtype=np.float32)
            t_np = np.zeros((run_batch, 3), dtype=np.float32)
            p_np[:batch] = poses[start:end]
            t_np[:batch] = trans[start:end]
            p = torch.from_numpy(p_np).to(device=device)
            t = torch.from_numpy(t_np).to(device=device)
            zeros3 = torch.zeros((run_batch, 3), dtype=torch.float32, device=device)
            smpl_out = model(
                global_orient=p[:, 0:3],
                body_pose=p[:, 3:66],
                left_hand_pose=p[:, 66:111],
                right_hand_pose=p[:, 111:156],
                transl=t,
                betas=torch.zeros((run_batch, 10), dtype=torch.float32, device=device),
                expression=torch.zeros((run_batch, 10), dtype=torch.float32, device=device),
                jaw_pose=zeros3,
                leye_pose=zeros3,
                reye_pose=zeros3,
                return_verts=False,
            )
            joints = smpl_out.joints.index_select(1, selected)[:batch].detach().cpu().numpy().astype(np.float32)
            out[start:end] = convert_points_babel_to_ours(joints)
    return out


def build_record(
    args: argparse.Namespace,
    split_name: str,
    sequence_index: int,
    sequence_id: str,
    sequence_global_start: int,
    sequence_frames: int,
    label: dict[str, Any],
    segment_id: int,
    start: int,
    end: int,
    body_goal: np.ndarray,
    global_orient_path: Path,
) -> dict[str, Any] | None:
    H = int(args.history_frames)
    W = int(args.move_future_frames)
    target_start_min = int(start + H)
    target_start_max = int(end - W + 1)
    if target_start_min > target_start_max:
        return None
    goal_type = goal_type_for_label(label)
    text = normalize_label(label.get("proc_label") or label.get("raw_label") or goal_type)
    seq_global_end = int(sequence_global_start + sequence_frames - 1)
    return {
        "dataset": "babel",
        "split": split_name,
        "scene_id": "babel",
        "sequence_id": str(sequence_id),
        "sequence_index": int(sequence_index),
        "sequence_global_start": int(sequence_global_start),
        "sequence_global_end": seq_global_end,
        "sequence_local_start": 0,
        "sequence_local_end": int(sequence_frames - 1),
        "segment_id": int(segment_id),
        "orig_segment_id": label.get("seg_id"),
        "segment_start": int(start),
        "segment_end": int(end),
        "segment_global_start": int(sequence_global_start + start),
        "segment_global_end": int(sequence_global_start + end),
        "target_start_min": target_start_min,
        "target_start_max": target_start_max,
        "history_frames": H,
        "future_frames": W,
        "r_plan_frames": W,
        "kind": "move_wait",
        "goal_type": goal_type,
        "text": text,
        "move_for_action": False,
        "body_goal": [float(x) for x in np.asarray(body_goal, dtype=np.float32).reshape(3)],
        "hand_goal": None,
        "left_hand_goal": None,
        "right_hand_goal": None,
        "active_hand": "",
        "smplx": {
            "transl_path": None,
            "global_orient_path": repo_rel(global_orient_path),
            "body_pose_path": None,
            "left_hand_pose_path": None,
            "right_hand_pose_path": None,
            "joints_path": None,
        },
        "scene": {},
    }


def split_manifest_payload(args: argparse.Namespace, split_name: str, records: list[dict[str, Any]], stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": "babel",
        "split": split_name,
        "kind": "move_wait",
        "history_frames": int(args.history_frames),
        "move_future_frames": int(args.move_future_frames),
        "records": records,
        "stats": stats,
        "source": {
            "raw_root": repo_rel(args.raw_root),
            "motion": f"{'train' if split_name == 'train' else 'val'}.pth.tar",
            "annotation": f"babel_v2.1/{'train' if split_name == 'train' else 'val'}.json",
        },
    }


def make_debug_vis(
    path: Path,
    joints28: np.ndarray,
    global_orient: np.ndarray,
    record: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from datasets.stage2 import canonicalize_points, yaw_to_local_rotation

    seq_start = int(record["sequence_global_start"])
    H = int(record["history_frames"])
    W = int(record["future_frames"])
    target_start = int(record["target_start_min"])
    frames = seq_start + np.arange(target_start - H, target_start + W, dtype=np.int64)
    frames = np.clip(frames, 0, len(joints28) - 1)
    anchor = int(seq_start + target_start - 1)
    world = np.asarray(joints28[frames], dtype=np.float32)
    anchor_root = np.asarray(joints28[anchor, 0], dtype=np.float32)
    _, world_to_local = yaw_to_local_rotation(np.asarray(global_orient[anchor], dtype=np.float32))
    local = canonicalize_points(world.reshape(-1, 3), anchor_root, world_to_local).reshape(world.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    names = [("World after axis conversion", world), ("Canonical local", local)]
    for ax, (title, arr) in zip(axes, names):
        root = arr[:, 0]
        ax.plot(root[:H, 0], root[:H, 2], color="tab:red", linewidth=3, label="history")
        ax.plot(root[H:, 0], root[H:, 2], color="tab:blue", linewidth=2, label="future")
        for idx in [0, H - 1, len(root) - 1]:
            sk = arr[idx]
            ax.scatter(sk[:, 0], sk[:, 2], s=8, alpha=0.45)
            for a, b in [(0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 12), (1, 4), (4, 7), (2, 5), (5, 8)]:
                if a < sk.shape[0] and b < sk.shape[0]:
                    ax.plot([sk[a, 0], sk[b, 0]], [sk[a, 2], sk[b, 2]], color="0.2", linewidth=0.8, alpha=0.5)
        ax.scatter(root[H - 1, 0], root[H - 1, 2], marker="o", color="black", s=28, label="anchor")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.axis("equal")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"{record['sequence_id']} / {record['text']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build(args: argparse.Namespace) -> dict[str, Any]:
    raw_root = repo_path(args.raw_root)
    out_root = repo_path(args.output_root) / "babel"
    joints_dir = out_root / "joints28"
    stage2_dir = out_root / "stage2"
    joints_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)

    joints_path = joints_dir / "joints28.npy"
    orient_path = joints_dir / "global_orient.npy"
    meta_path = joints_dir / "joints28_meta.json"
    if (joints_path.exists() or orient_path.exists()) and not bool(args.overwrite):
        raise FileExistsError(f"{repo_rel(joints_path)} exists; pass --overwrite to rebuild")

    splits = output_splits(str(args.split))
    loaded: list[tuple[str, str, list[dict[str, Any]], dict[str, Any]]] = []
    total_frames = 0
    for raw_split, out_split in splits:
        motion_path = raw_root / f"{raw_split}.pth.tar"
        ann_path = raw_root / "babel_v2.1" / f"{raw_split}.json"
        if not motion_path.exists():
            raise FileNotFoundError(motion_path)
        if not ann_path.exists():
            raise FileNotFoundError(ann_path)
        motions = joblib.load(motion_path)
        if int(args.max_sequences) > 0:
            motions = motions[: int(args.max_sequences)]
        annotations = load_json(ann_path)
        loaded.append((raw_split, out_split, motions, annotations))
        total_frames += sum(int(item["poses"].shape[0]) for item in motions)

    joints_out = np.lib.format.open_memmap(joints_path, mode="w+", dtype=np.float32, shape=(total_frames, 28, 3))
    orient_out = np.lib.format.open_memmap(orient_path, mode="w+", dtype=np.float32, shape=(total_frames, 3))

    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    model = make_smplx_model(args, device)
    selected = torch.tensor(SMPLX_SELECTED_JOINTS_28, dtype=torch.long, device=device)

    records_by_split: dict[str, list[dict[str, Any]]] = {out_split: [] for _, out_split in splits}
    label_counts: Counter[str] = Counter()
    reject_counts: Counter[str] = Counter()
    sequence_counts: dict[str, int] = {}
    frame_cursor = 0
    debug_record: dict[str, Any] | None = None

    for raw_split, out_split, motions, annotations in loaded:
        sequence_counts[out_split] = len(motions)
        progress = tqdm(motions, desc=f"babel {out_split} smplx forward", unit="seq")
        for seq_idx, item in enumerate(progress):
            poses = np.asarray(item["poses"], dtype=np.float32)
            trans = np.asarray(item["trans"], dtype=np.float32)
            fps = float(item.get("fps", 30))
            sequence_frames = int(poses.shape[0])
            seq_start = int(frame_cursor)
            seq_end = seq_start + sequence_frames
            joints_seq = forward_joints28(model, poses, trans, selected, device, int(args.chunk_size))
            orient_seq = convert_rotvec_babel_to_ours(poses[:, 0:3])
            joints_out[seq_start:seq_end] = joints_seq
            orient_out[seq_start:seq_end] = orient_seq

            babel_id = str(item.get("babel_id"))
            ann = annotations.get(babel_id)
            if ann is None:
                reject_counts["missing_annotation"] += 1
                frame_cursor = seq_end
                continue
            labels = ((ann.get("frame_ann") or {}).get("labels") or [])
            for label_idx, label in enumerate(labels):
                if not is_move_wait_label(label):
                    reject_counts["label_filtered"] += 1
                    continue
                bounds = segment_bounds(label, fps, sequence_frames)
                if bounds is None:
                    reject_counts["bad_bounds"] += 1
                    continue
                start, end = bounds
                if int(args.min_segment_frames) > 0 and (end - start + 1) < int(args.min_segment_frames):
                    reject_counts["too_short_min_segment"] += 1
                    continue
                body_goal = joints_seq[end, 0]
                record = build_record(
                    args,
                    out_split,
                    sequence_index=seq_idx,
                    sequence_id=f"{raw_split}_{babel_id}",
                    sequence_global_start=seq_start,
                    sequence_frames=sequence_frames,
                    label=label,
                    segment_id=label_idx,
                    start=start,
                    end=end,
                    body_goal=body_goal,
                    global_orient_path=orient_path,
                )
                if record is None:
                    reject_counts["too_short_window"] += 1
                    continue
                records_by_split[out_split].append(record)
                label_counts[record["text"]] += 1
                if debug_record is None:
                    debug_record = record
            frame_cursor = seq_end

    joints_out.flush()
    orient_out.flush()

    coord_meta = {
        "dataset": "babel",
        "normalization": "fixed_stage2_joint_xyz",
        "representation": "canonical_local_smplx_joints28_xyz",
        "x_scale": 3.0,
        "y_center": 1.0,
        "y_scale": 1.0,
        "z_scale": 4.0,
    }
    write_json(stage2_dir / "joint_coord_norm_meta.json", coord_meta)

    stats = {
        "dataset": "babel",
        "history_frames": int(args.history_frames),
        "move_future_frames": int(args.move_future_frames),
        "total_frames": int(total_frames),
        "sequence_counts": sequence_counts,
        "record_counts": {key: len(value) for key, value in records_by_split.items()},
        "label_counts": dict(label_counts.most_common(100)),
        "reject_counts": dict(sorted(reject_counts.items())),
        "axis_conversion": "[x,y,z]_babel_amass -> [x,z,-y]_ours",
        "joint_source": "smplx_forward_from_babel_poses_trans",
    }

    for out_split, records in records_by_split.items():
        write_json(stage2_dir / f"move_wait_{out_split}.json", split_manifest_payload(args, out_split, records, stats))
    write_json(stage2_dir / "babel_stage2_stats.json", stats)
    write_json(
        meta_path,
        {
            "dataset": "babel",
            "source": "smplx_forward_from_babel_poses_trans",
            "shape": [int(total_frames), 28, 3],
            "dtype": "float32",
            "path": repo_rel(joints_path),
            "global_orient_path": repo_rel(orient_path),
            "joint_indices": SMPLX_SELECTED_JOINTS_28,
            "joint_index_basis": "smplx_model_output_joints",
            "smplx_model_dir": repo_rel(args.smplx_model_dir),
            "gender": str(args.gender),
            "axis_conversion_matrix": BABEL_TO_OURS.tolist(),
            "raw_root": repo_rel(raw_root),
        },
    )

    if args.debug_vis_path is not None and debug_record is not None:
        make_debug_vis(repo_path(args.debug_vis_path), np.asarray(joints_out), np.asarray(orient_out), debug_record)

    return stats


def main() -> None:
    args = parse_args()
    stats = build(args)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
