from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_BLENDER = PROJECT_ROOT / "tools" / "blender-3.6.17-linux-x64" / "blender"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a TRUMANS segment clip and load it into the original Blender SMPL-X pipeline.")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--segment-id", type=int, required=True)
    parser.add_argument("--scene-blend", type=Path, default=None)
    parser.add_argument("--blend-root", type=Path, default=Path("data/raw/trumans/Recordings_blend"))
    parser.add_argument("--object-name", default="SMPLX-mesh-male")
    parser.add_argument("--smplx-gender", default="male", choices=["female", "male", "neutral"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "blender_segments")
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render-mode", default="preview", choices=["preview", "final"])
    parser.add_argument("--camera-mode", default="scene", choices=["scene", "topdown", "oblique"])
    parser.add_argument("--camera-scale", type=float, default=1.0)
    parser.add_argument("--bright-preview", action="store_true")
    parser.add_argument("--character-color", default=None)
    parser.add_argument("--render-mp4", action="store_true")
    parser.add_argument("--save-blend", action="store_true")
    parser.add_argument("--clear-existing-characters", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def to_abs_or_none(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def export_segment_clip(sequence: dict, segment: dict, out_path: Path) -> None:
    motion_ref = sequence["human_motion_ref"]
    smpl = motion_ref["smplx"]
    seq_start = int(motion_ref["start"])
    clip_start = seq_start + int(segment["start"])
    clip_end = seq_start + int(segment["end"]) + 1

    global_orient = np.asarray(np.load(resolve_repo_path(smpl["global_orient_path"]), mmap_mode="r")[clip_start:clip_end], dtype=np.float32)
    body_pose = np.asarray(np.load(resolve_repo_path(smpl["body_pose_path"]), mmap_mode="r")[clip_start:clip_end], dtype=np.float32)
    transl = np.asarray(np.load(resolve_repo_path(smpl["transl_path"]), mmap_mode="r")[clip_start:clip_end], dtype=np.float32)

    payload = {
        "global_orient": global_orient,
        "body_pose": body_pose,
        "transl": transl,
    }
    if "left_hand_pose_path" in smpl:
        payload["left_hand_pose"] = np.asarray(
            np.load(resolve_repo_path(smpl["left_hand_pose_path"]), mmap_mode="r")[clip_start:clip_end],
            dtype=np.float32,
        )
    if "right_hand_pose_path" in smpl:
        payload["right_hand_pose"] = np.asarray(
            np.load(resolve_repo_path(smpl["right_hand_pose_path"]), mmap_mode="r")[clip_start:clip_end],
            dtype=np.float32,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    args = parse_args()
    scene_root = PROJECT_ROOT / "data" / "preprocessed" / "trumans" / "scenes" / args.scene_id
    seq_path = scene_root / "sequences" / f"{args.sequence_id}.json"
    if not seq_path.exists():
        raise FileNotFoundError(seq_path)

    sequence = load_json(seq_path)
    segment = next((seg for seg in sequence["segment_list"] if int(seg["segment_id"]) == int(args.segment_id)), None)
    if segment is None:
        raise ValueError(f"segment_id={args.segment_id} not found in {seq_path}")

    if args.scene_blend is not None:
        scene_blend = to_abs_or_none(args.scene_blend)
    else:
        candidate_root = to_abs_or_none(args.blend_root)
        direct = candidate_root / f"{args.scene_id}.blend"
        nested = candidate_root / args.scene_id / f"{args.scene_id}.blend"
        if direct.exists():
            scene_blend = direct
        else:
            scene_blend = nested
    if scene_blend is None or not scene_blend.exists():
        raise FileNotFoundError(
            f"Scene blend not found: {scene_blend}. "
            f"Provide --scene-blend explicitly or place {args.scene_id}.blend under {args.blend_root}."
        )

    out_dir = to_abs_or_none(args.output_dir / args.scene_id / args.sequence_id / f"seg{int(args.segment_id):02d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    motion_pkl = out_dir / "smplx_results.pkl"
    export_segment_clip(sequence, segment, motion_pkl)

    meta = {
        "scene_id": args.scene_id,
        "sequence_id": args.sequence_id,
        "segment_id": int(args.segment_id),
        "text": segment["text"],
        "goal_type": segment["goal_type"],
        "segment_start": int(segment["start"]),
        "segment_end": int(segment["end"]),
        "interaction_frame": int(segment["interaction_frame"]),
        "scene_blend": str(scene_blend),
        "motion_pkl": str(motion_pkl),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    cmd = [
        str(to_abs_or_none(args.blender_bin)),
        "-b",
        str(scene_blend),
        "--python",
        str(SCRIPT_DIR / "blender_apply_smplx_segment.py"),
        "--",
        "--motion-pkl",
        str(motion_pkl),
        "--object-name",
        args.object_name,
        "--smplx-gender",
        args.smplx_gender,
        "--render-mode",
        args.render_mode,
        "--camera-mode",
        args.camera_mode,
        "--camera-scale",
        str(float(args.camera_scale)),
        "--character-color",
        str(args.character_color) if args.character_color is not None else "",
        "--bright-preview" if args.bright_preview else "",
        "--fps",
        str(int(args.fps)),
    ]
    cmd = [part for part in cmd if part != ""]
    if args.save_blend:
        cmd.extend(["--save-blend", str(out_dir / "animated_segment.blend")])
    if args.render_mp4:
        cmd.extend(["--render-mp4", str(out_dir / "render.mp4")])
    if args.clear_existing_characters:
        cmd.append("--clear-existing-characters")

    print("motion_pkl:", motion_pkl)
    print("meta_json:", out_dir / "meta.json")
    print("blender_cmd:", " ".join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
