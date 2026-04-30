from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSED_ROOT = Path("data") / "preprocessed"
DEFAULT_SMPLX_MODEL_DIR = Path("smpl_models")

SMPLX_SELECTED_JOINTS_28 = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    23,
    24,
    25,
    34,
    40,
    49,
]


def repo_path(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def repo_rel(path: Path | str) -> str:
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build raw-frame-aligned SMPL-X joints28 cache for Stage-2.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PREPROCESSED_ROOT)
    parser.add_argument("--trumans-root", type=Path, default=Path("data") / "raw" / "trumans")
    parser.add_argument("--lingo-root", type=Path, default=Path("data") / "raw" / "lingo" / "dataset")
    parser.add_argument("--smplx-model-dir", type=Path, default=DEFAULT_SMPLX_MODEL_DIR)
    parser.add_argument("--gender", choices=["male", "female", "neutral"], default="male")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--max-frames", type=int, default=0, help="Debug only. 0 builds the full raw-frame cache.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dataset_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    if args.dataset == "trumans":
        root = repo_path(args.trumans_root)
        return {
            "root": root,
            "transl": root / "human_transl.npy",
            "global_orient": root / "human_orient.npy",
            "body_pose": root / "human_pose.npy",
            "left_hand_pose": root / "left_hand_pose.npy",
            "right_hand_pose": root / "right_hand_pose.npy",
        }
    root = repo_path(args.lingo_root)
    return {
        "root": root,
        "transl": root / "transl_aligned.npy",
        "global_orient": root / "human_orient.npy",
        "body_pose": root / "human_pose.npy",
        "left_hand_pose": None,
        "right_hand_pose": None,
    }


def load_required(path: Path | None, name: str) -> np.ndarray | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"missing {name}: {repo_rel(path)}")
    return np.load(path, mmap_mode="r")


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


def zeros_like_pose(batch: int, dims: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((batch, dims), dtype=torch.float32, device=device)


def tensor_slice(values: np.ndarray, start: int, end: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values[start:end], dtype=np.float32)).to(device=device)


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    paths = dataset_paths(args)
    transl = load_required(paths["transl"], "transl")
    global_orient = load_required(paths["global_orient"], "global_orient")
    body_pose = load_required(paths["body_pose"], "body_pose")
    left_hand = load_required(paths["left_hand_pose"], "left_hand_pose")
    right_hand = load_required(paths["right_hand_pose"], "right_hand_pose")
    assert transl is not None and global_orient is not None and body_pose is not None

    source_num_frames = int(transl.shape[0])
    num_frames = source_num_frames if int(args.max_frames) <= 0 else min(source_num_frames, int(args.max_frames))
    if global_orient.shape[0] != source_num_frames or body_pose.shape[0] != source_num_frames:
        raise ValueError("SMPL-X arrays have inconsistent frame counts")
    if left_hand is not None and left_hand.shape[0] != source_num_frames:
        raise ValueError("left hand pose frame count mismatch")
    if right_hand is not None and right_hand.shape[0] != source_num_frames:
        raise ValueError("right hand pose frame count mismatch")

    out_dir = repo_path(args.output_root) / args.dataset / "joints28"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "joints28.npy"
    meta_path = out_dir / "joints28_meta.json"
    if out_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"{repo_rel(out_path)} exists; pass --overwrite to rebuild")

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    model = make_smplx_model(args, device)
    selected = torch.tensor(SMPLX_SELECTED_JOINTS_28, dtype=torch.long, device=device)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(num_frames, 28, 3))
    chunk = int(args.chunk_size)

    with torch.no_grad():
        for start in tqdm(range(0, num_frames, chunk), desc=f"{args.dataset} joints28", unit="frame"):
            end = min(start + chunk, num_frames)
            batch = end - start
            kwargs = {
                "transl": tensor_slice(transl, start, end, device),
                "global_orient": tensor_slice(global_orient, start, end, device),
                "body_pose": tensor_slice(body_pose, start, end, device),
                "left_hand_pose": tensor_slice(left_hand, start, end, device) if left_hand is not None else zeros_like_pose(batch, 45, device),
                "right_hand_pose": tensor_slice(right_hand, start, end, device) if right_hand is not None else zeros_like_pose(batch, 45, device),
                "return_verts": False,
            }
            smpl_out = model(**kwargs)
            joints = smpl_out.joints.index_select(1, selected).detach().cpu().numpy().astype(np.float32)
            out[start:end] = joints
    out.flush()

    meta: dict[str, Any] = {
        "dataset": args.dataset,
        "source": "smplx_forward",
        "shape": [num_frames, 28, 3],
        "source_num_frames": source_num_frames,
        "max_frames": int(args.max_frames),
        "dtype": "float32",
        "path": repo_rel(out_path),
        "joint_indices": SMPLX_SELECTED_JOINTS_28,
        "joint_index_basis": "smplx_model_output_joints",
        "smplx_model_dir": repo_rel(args.smplx_model_dir),
        "gender": args.gender,
        "chunk_size": int(args.chunk_size),
        "raw_paths": {key: None if value is None else repo_rel(value) for key, value in paths.items()},
        "missing_hand_pose_policy": "zeros",
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def main() -> None:
    args = parse_args()
    meta = build_cache(args)
    print(f"wrote {meta['path']} shape={meta['shape']}")


if __name__ == "__main__":
    main()
