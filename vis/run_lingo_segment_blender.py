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
    parser = argparse.ArgumentParser(description="Export a LINGO segment clip and render it in Blender with scene mesh.")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--segment-id", type=int, required=True)
    parser.add_argument("--scene-obj", type=Path, default=None)
    parser.add_argument("--mesh-root", type=Path, default=Path("data/raw/lingo/Scene_mesh"))
    parser.add_argument("--object-name", default="SMPLX-mesh-male")
    parser.add_argument("--smplx-gender", default="male", choices=["female", "male", "neutral"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "blender_segments_lingo")
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render-mode", default="preview", choices=["preview", "final"])
    parser.add_argument("--camera-mode", default="oblique", choices=["oblique", "topdown"])
    parser.add_argument("--camera-scale", type=float, default=1.0)
    parser.add_argument("--bright-preview", action="store_true")
    parser.add_argument("--character-color", default=None)
    parser.add_argument("--render-mp4", action="store_true")
    parser.add_argument("--save-blend", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def to_abs(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def candidate_scene_ids(scene_id: str) -> list[str]:
    candidates: list[str] = []

    def add(name: str) -> None:
        if name and name not in candidates:
            candidates.append(name)

    add(scene_id)
    add(scene_id.replace("_", "-"))
    add(scene_id.replace("-", "_"))
    if scene_id.endswith("_mirror"):
        base = scene_id[:-7]
        add(base)
        add(base.replace("_", "-"))
        add(base.replace("-", "_"))
    return candidates


def resolve_scene_obj(scene_id: str, scene_obj: Path | None, mesh_root: Path) -> tuple[Path, str]:
    if scene_obj is not None:
        resolved = to_abs(scene_obj)
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved, resolved.parent.name

    mesh_root = to_abs(mesh_root)
    for candidate in candidate_scene_ids(scene_id):
        scene_mesh = mesh_root / candidate / "mesh_low.obj"
        if scene_mesh.exists():
            return scene_mesh, candidate

    raise FileNotFoundError(
        f"No scene mesh found for scene_id={scene_id}. "
        f"Checked under {mesh_root} using candidates {candidate_scene_ids(scene_id)}."
    )


def export_segment_clip(sequence: dict, segment: dict, out_path: Path) -> None:
    motion_ref = sequence["human_motion_ref"]
    smpl = motion_ref["smplx"]
    seq_start = int(motion_ref["start"])
    clip_start = seq_start + int(segment["start"])
    clip_end = seq_start + int(segment["end"]) + 1

    global_orient = np.asarray(
        np.load(resolve_repo_path(smpl["global_orient_path"]), mmap_mode="r")[clip_start:clip_end],
        dtype=np.float32,
    )
    body_pose = np.asarray(
        np.load(resolve_repo_path(smpl["body_pose_path"]), mmap_mode="r")[clip_start:clip_end],
        dtype=np.float32,
    )
    transl = np.asarray(
        np.load(resolve_repo_path(smpl["transl_path"]), mmap_mode="r")[clip_start:clip_end],
        dtype=np.float32,
    )

    payload = {
        "global_orient": global_orient,
        "body_pose": body_pose,
        "transl": transl,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    args = parse_args()
    scene_root = PROJECT_ROOT / "data" / "preprocessed" / "lingo" / "scenes" / args.scene_id
    seq_path = scene_root / "sequences" / f"{args.sequence_id}.json"
    if not seq_path.exists():
        raise FileNotFoundError(seq_path)

    sequence = load_json(seq_path)
    segment = next((seg for seg in sequence["segment_list"] if int(seg["segment_id"]) == int(args.segment_id)), None)
    if segment is None:
        raise ValueError(f"segment_id={args.segment_id} not found in {seq_path}")

    scene_obj, resolved_scene_mesh_id = resolve_scene_obj(args.scene_id, args.scene_obj, args.mesh_root)

    out_dir = to_abs(args.output_dir / args.scene_id / args.sequence_id / f"seg{int(args.segment_id):02d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    motion_pkl = out_dir / "smplx_results.pkl"
    export_segment_clip(sequence, segment, motion_pkl)

    meta = {
        "scene_id": args.scene_id,
        "resolved_scene_mesh_id": resolved_scene_mesh_id,
        "sequence_id": args.sequence_id,
        "segment_id": int(args.segment_id),
        "text": segment["text"],
        "goal_type": segment["goal_type"],
        "segment_start": int(segment["start"]),
        "segment_end": int(segment["end"]),
        "interaction_frame": int(segment["interaction_frame"]),
        "scene_obj": str(scene_obj),
        "motion_pkl": str(motion_pkl),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    cmd = [
        str(to_abs(args.blender_bin)),
        "-b",
        "--factory-startup",
        "--python",
        str(SCRIPT_DIR / "blender_apply_lingo_segment.py"),
        "--",
        "--motion-pkl",
        str(motion_pkl),
        "--scene-obj",
        str(scene_obj),
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
        "--fps",
        str(int(args.fps)),
    ]
    cmd = [part for part in cmd if part != ""]
    if args.bright_preview:
        cmd.append("--bright-preview")
    if args.save_blend:
        cmd.extend(["--save-blend", str(out_dir / "animated_segment.blend")])
    if args.render_mp4:
        cmd.extend(["--render-mp4", str(out_dir / "render.mp4")])

    print("scene_obj:", scene_obj)
    print("motion_pkl:", motion_pkl)
    print("meta_json:", out_dir / "meta.json")
    print("blender_cmd:", " ".join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
