from __future__ import annotations

import argparse
import json
import pickle
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BLENDER = PROJECT_ROOT / "tools" / "blender-3.6.17-linux-x64" / "blender"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Stage2 MoveWait/Action manifest samples in Blender.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--kinds", choices=["both", "move_wait", "action"], default="both")
    parser.add_argument("--num-per-kind", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2_data_blender_vis"))
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--render-mode", choices=["preview", "final"], default="preview")
    parser.add_argument("--camera-mode", choices=["scene", "topdown", "oblique"], default="oblique")
    parser.add_argument("--camera-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render-mp4", action="store_true")
    parser.add_argument("--save-blend", action="store_true")
    parser.add_argument("--bright-preview", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def repo_path(path_str: str | Path | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_records(dataset: str, split: str, kind: str) -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "data" / "preprocessed" / dataset / "stage2" / f"{kind}_{split}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return list(load_json(path).get("records", []))


def sample_record_frames(record: dict[str, Any], kind: str, rng: random.Random) -> tuple[np.ndarray, int, int]:
    H = int(record["history_frames"])
    seq_start = int(record["sequence_global_start"])
    if kind == "move_wait":
        target_start = rng.randint(int(record["target_start_min"]), int(record["target_start_max"]))
        end_local = target_start + int(record["future_frames"]) - 1
    else:
        if bool(record.get("first_segment_exception")):
            target_start = int(record["forced_target_start"])
        else:
            delta = rng.randint(int(record["aug_delta_min"]), int(record["aug_delta_max"]))
            target_start = int(record["action_start"]) - int(delta)
        end_local = int(record["action_end"])
    start_local = target_start - H
    return np.arange(seq_start + start_local, seq_start + end_local + 1, dtype=np.int64), start_local, end_local


def export_motion_pkl(record: dict[str, Any], frames: np.ndarray, out_path: Path) -> None:
    smplx = record["smplx"]

    def read(key: str) -> np.ndarray | None:
        path = repo_path(smplx.get(key))
        if path is None:
            return None
        return np.asarray(np.load(path, mmap_mode="r")[frames], dtype=np.float32)

    payload = {
        "global_orient": read("global_orient_path"),
        "body_pose": read("body_pose_path"),
        "transl": read("transl_path"),
    }
    left = read("left_hand_pose_path")
    right = read("right_hand_pose_path")
    if left is not None:
        payload["left_hand_pose"] = left
    if right is not None:
        payload["right_hand_pose"] = right

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def resolve_trumans_scene_blend(scene_id: str) -> Path:
    root = PROJECT_ROOT / "data" / "raw" / "trumans" / "Recordings_blend"
    candidates = [root / f"{scene_id}.blend", root / scene_id / f"{scene_id}.blend"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"TRUMANS scene blend not found for {scene_id}; checked {candidates}")


def candidate_lingo_scene_ids(scene_id: str) -> list[str]:
    out: list[str] = []

    def add(value: str) -> None:
        if value and value not in out:
            out.append(value)

    add(scene_id)
    add(scene_id.replace("_", "-"))
    add(scene_id.replace("-", "_"))
    if scene_id.endswith("_mirror"):
        base = scene_id[:-7]
        add(base)
        add(base.replace("_", "-"))
        add(base.replace("-", "_"))
    return out


def resolve_lingo_scene_obj(scene_id: str) -> Path:
    root = PROJECT_ROOT / "data" / "raw" / "lingo" / "Scene_mesh"
    for candidate in candidate_lingo_scene_ids(scene_id):
        path = root / candidate / "mesh_low.obj"
        if path.exists():
            return path
    raise FileNotFoundError(f"LINGO scene mesh not found for {scene_id}")


def build_blender_cmd(args: argparse.Namespace, record: dict[str, Any], motion_pkl: Path, out_dir: Path) -> list[str]:
    scene_id = str(record["scene_id"])
    blender = str(repo_path(args.blender_bin))
    if args.dataset == "trumans":
        scene_blend = resolve_trumans_scene_blend(scene_id)
        cmd = [
            blender,
            "-b",
            str(scene_blend),
            "--python",
            str(SCRIPT_DIR / "blender_apply_smplx_segment.py"),
            "--",
            "--motion-pkl",
            str(motion_pkl),
            "--smplx-gender",
            "male",
            "--render-mode",
            args.render_mode,
            "--camera-mode",
            args.camera_mode if args.camera_mode in {"scene", "topdown", "oblique"} else "oblique",
            "--camera-scale",
            str(float(args.camera_scale)),
            "--fps",
            str(int(args.fps)),
            "--character-color",
            "#e53935",
            "--clear-existing-characters",
        ]
    else:
        scene_obj = resolve_lingo_scene_obj(scene_id)
        cmd = [
            blender,
            "-b",
            "--factory-startup",
            "--python",
            str(SCRIPT_DIR / "blender_apply_lingo_segment.py"),
            "--",
            "--motion-pkl",
            str(motion_pkl),
            "--scene-obj",
            str(scene_obj),
            "--smplx-gender",
            "male",
            "--render-mode",
            args.render_mode,
            "--camera-mode",
            "topdown" if args.camera_mode == "topdown" else "oblique",
            "--camera-scale",
            str(float(args.camera_scale)),
            "--fps",
            str(int(args.fps)),
            "--character-color",
            "#e53935",
        ]
    if args.bright_preview:
        cmd.append("--bright-preview")
    if args.render_mp4:
        cmd.extend(["--render-mp4", str(out_dir / "render.mp4")])
    if args.save_blend:
        cmd.extend(["--save-blend", str(out_dir / "animated_segment.blend")])
    return cmd


def render_one(args: argparse.Namespace, record: dict[str, Any], kind: str, idx: int, rng: random.Random) -> None:
    frames, start_local, end_local = sample_record_frames(record, kind, rng)
    out_dir = repo_path(args.output_dir) / args.dataset / kind / f"sample_{idx:03d}"
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    motion_pkl = out_dir / "smplx_results.pkl"
    export_motion_pkl(record, frames, motion_pkl)
    meta = {
        "dataset": args.dataset,
        "kind": kind,
        "scene_id": record.get("scene_id"),
        "sequence_id": record.get("sequence_id"),
        "segment_id": record.get("segment_id"),
        "goal_type": record.get("goal_type"),
        "text": record.get("text"),
        "start_local": int(start_local),
        "end_local": int(end_local),
        "num_frames": int(frames.shape[0]),
        "motion_pkl": str(motion_pkl),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    cmd = build_blender_cmd(args, record, motion_pkl, out_dir)
    print("render:", json.dumps(meta, sort_keys=True))
    print("cmd:", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    kinds = ["move_wait", "action"] if args.kinds == "both" else [args.kinds]
    for kind in kinds:
        records = load_records(args.dataset, args.split, kind)
        if len(records) < int(args.num_per_kind):
            selected = list(range(len(records)))
        else:
            selected = sorted(rng.sample(range(len(records)), int(args.num_per_kind)))
        for out_idx, rec_idx in enumerate(selected):
            render_one(args, records[rec_idx], kind, out_idx, rng)


if __name__ == "__main__":
    main()
