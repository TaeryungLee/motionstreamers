from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SMPLX_JOINT_INDEX = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_shoulder": 16,
    "right_shoulder": 17,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump SMPL-X interaction-frame meshes and facing diagnostics.")
    parser.add_argument("--dataset", choices=["lingo", "trumans"], required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scene-object-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "interaction_facing",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@lru_cache(maxsize=128)
def load_numpy(path_str: str) -> np.ndarray:
    return np.load(path_str, mmap_mode="r")


@lru_cache(maxsize=1)
def load_smplx_runtime():
    import smplx  # type: ignore
    import torch  # type: ignore

    for candidate in (PROJECT_ROOT / "human_models", PROJECT_ROOT / "smpl_models"):
        if not candidate.exists():
            continue
        try:
            model = smplx.create(
                str(candidate),
                model_type="smplx",
                gender="NEUTRAL",
                use_pca=False,
                ext="npz",
            )
            return torch, model.eval()
        except Exception:
            continue
    raise RuntimeError("Failed to load SMPL-X model from human_models or smpl_models.")


def yaw_from_forward(forward_xz: np.ndarray) -> Optional[float]:
    norm = float(np.linalg.norm(forward_xz))
    if norm <= 1e-8:
        return None
    return float(math.atan2(float(forward_xz[0]), float(forward_xz[1])))


def yaw_from_global_orient(rotvec: np.ndarray) -> Optional[float]:
    from scipy.spatial.transform import Rotation as R

    forward = R.from_rotvec(np.asarray(rotvec, dtype=np.float32)).apply(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    return yaw_from_forward(np.asarray([forward[0], forward[2]], dtype=np.float32))


def hip_facing_yaw(joints: np.ndarray, fallback_yaw: Optional[float]) -> Optional[float]:
    pelvis = joints[SMPLX_JOINT_INDEX["pelvis"], [0, 2]]
    left_hip = joints[SMPLX_JOINT_INDEX["left_hip"], [0, 2]]
    right_hip = joints[SMPLX_JOINT_INDEX["right_hip"], [0, 2]]
    spine1 = joints[SMPLX_JOINT_INDEX["spine1"], [0, 2]]
    left_shoulder = joints[SMPLX_JOINT_INDEX["left_shoulder"], [0, 2]]
    right_shoulder = joints[SMPLX_JOINT_INDEX["right_shoulder"], [0, 2]]

    lateral = right_hip - left_hip
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm <= 1e-8:
        return fallback_yaw
    lateral = lateral / lateral_norm

    candidate_a = np.asarray([-lateral[1], lateral[0]], dtype=np.float32)
    candidate_b = -candidate_a

    torso_hint = (spine1 - pelvis) + (0.5 * (left_shoulder + right_shoulder) - pelvis)
    torso_norm = float(np.linalg.norm(torso_hint))
    if torso_norm <= 1e-8:
        if fallback_yaw is None:
            return None
        torso_hint = np.asarray([math.sin(fallback_yaw), math.cos(fallback_yaw)], dtype=np.float32)
        torso_norm = float(np.linalg.norm(torso_hint))
    torso_hint = torso_hint / max(torso_norm, 1e-8)

    facing = candidate_a if float(np.dot(candidate_a, torso_hint)) >= float(np.dot(candidate_b, torso_hint)) else candidate_b
    return yaw_from_forward(facing)


def shoulder_facing_yaw(joints: np.ndarray, fallback_yaw: Optional[float]) -> Optional[float]:
    left_shoulder = joints[SMPLX_JOINT_INDEX["left_shoulder"], [0, 2]]
    right_shoulder = joints[SMPLX_JOINT_INDEX["right_shoulder"], [0, 2]]
    pelvis = joints[SMPLX_JOINT_INDEX["pelvis"], [0, 2]]
    spine1 = joints[SMPLX_JOINT_INDEX["spine1"], [0, 2]]

    lateral = right_shoulder - left_shoulder
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm <= 1e-8:
        return fallback_yaw
    lateral = lateral / lateral_norm

    candidate_a = np.asarray([-lateral[1], lateral[0]], dtype=np.float32)
    candidate_b = -candidate_a
    torso_hint = spine1 - pelvis
    torso_norm = float(np.linalg.norm(torso_hint))
    if torso_norm <= 1e-8:
        if fallback_yaw is None:
            return None
        torso_hint = np.asarray([math.sin(fallback_yaw), math.cos(fallback_yaw)], dtype=np.float32)
        torso_norm = float(np.linalg.norm(torso_hint))
    torso_hint = torso_hint / max(torso_norm, 1e-8)

    facing = candidate_a if float(np.dot(candidate_a, torso_hint)) >= float(np.dot(candidate_b, torso_hint)) else candidate_b
    return yaw_from_forward(facing)


def hand_direction_yaw(segment: dict) -> Optional[float]:
    goal_pose = segment.get("goal_pose", {})
    pelvis = np.asarray(goal_pose.get("pelvis"), dtype=np.float32) if goal_pose.get("pelvis") is not None else None
    hand = np.asarray(goal_pose.get("hand"), dtype=np.float32) if goal_pose.get("hand") is not None else None
    if pelvis is None or hand is None:
        return None
    vec = hand[[0, 2]] - pelvis[[0, 2]]
    return yaw_from_forward(vec)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for v in vertices:
            handle.write(f"v {float(v[0]):.8f} {float(v[1]):.8f} {float(v[2]):.8f}\n")
        for f in faces:
            handle.write(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}\n")


def forward_frame_mesh(sequence: dict, frame_idx: int):
    torch, model = load_smplx_runtime()
    smpl_ref = sequence["human_motion_ref"]["smplx"]

    global_orient = np.array(load_numpy(str(resolve_repo_path(smpl_ref["global_orient_path"])))[frame_idx], dtype=np.float32, copy=True).reshape(1, -1)
    body_pose = np.array(load_numpy(str(resolve_repo_path(smpl_ref["body_pose_path"])))[frame_idx], dtype=np.float32, copy=True).reshape(1, -1)
    transl = np.array(load_numpy(str(resolve_repo_path(smpl_ref["transl_path"])))[frame_idx], dtype=np.float32, copy=True).reshape(1, -1)

    kwargs = {
        "global_orient": torch.from_numpy(global_orient),
        "body_pose": torch.from_numpy(body_pose),
        "transl": torch.from_numpy(transl),
        "return_verts": True,
    }
    if "left_hand_pose_path" in smpl_ref:
        kwargs["left_hand_pose"] = torch.from_numpy(
            np.array(load_numpy(str(resolve_repo_path(smpl_ref["left_hand_pose_path"])))[frame_idx], dtype=np.float32, copy=True).reshape(1, -1)
        )
    if "right_hand_pose_path" in smpl_ref:
        kwargs["right_hand_pose"] = torch.from_numpy(
            np.array(load_numpy(str(resolve_repo_path(smpl_ref["right_hand_pose_path"])))[frame_idx], dtype=np.float32, copy=True).reshape(1, -1)
        )

    with torch.no_grad():
        out = model(**kwargs)

    vertices = out.vertices[0].detach().cpu().numpy()
    joints = out.joints[0].detach().cpu().numpy()
    faces = model.faces.astype(np.int32)
    return vertices, joints, faces, global_orient[0]


def main() -> None:
    args = parse_args()
    scene_root = PROJECT_ROOT / "data" / "preprocessed" / args.dataset / "scenes" / args.scene_id
    affordance_path = PROJECT_ROOT / "data" / "preprocessed" / args.dataset / "multi_character" / args.scene_id / "scene_affordances.json"
    affordance = load_json(affordance_path)
    scene_object = next(obj for obj in affordance["object_state_list"] if obj["scene_object_id"] == args.scene_object_id)

    out_root = args.output_dir / args.dataset / args.scene_id / args.scene_object_id
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for seq_id, _mode, _name, seg_id in scene_object["source_keys"]:
        seq = load_json(scene_root / "sequences" / f"{seq_id}.json")
        segment = next(seg for seg in seq["segment_list"] if int(seg["segment_id"]) == int(seg_id))
        frame_idx = int(seq["human_motion_ref"]["start"]) + int(segment["interaction_frame"])
        vertices, joints, faces, global_orient = forward_frame_mesh(seq, frame_idx)

        mesh_path = out_root / f"{seq_id}__seg{int(seg_id):02d}.obj"
        write_obj(mesh_path, vertices, faces)

        global_yaw = yaw_from_global_orient(global_orient)
        hip_yaw = hip_facing_yaw(joints, global_yaw)
        shoulder_yaw = shoulder_facing_yaw(joints, global_yaw)
        hand_yaw = hand_direction_yaw(segment)

        rows.append(
            {
                "sequence_id": seq_id,
                "segment_id": int(seg_id),
                "text": segment["text"],
                "goal_type": segment["goal_type"],
                "interaction_frame_local": int(segment["interaction_frame"]),
                "interaction_frame_global": frame_idx,
                "mesh_path": str(mesh_path.relative_to(PROJECT_ROOT)),
                "global_orient": [float(x) for x in global_orient.tolist()],
                "global_orient_forward_yaw": global_yaw,
                "hip_facing_yaw": hip_yaw,
                "shoulder_facing_yaw": shoulder_yaw,
                "body_to_hand_yaw": hand_yaw,
                "pelvis": [float(x) for x in np.asarray(segment["goal_pose"].get("pelvis"), dtype=np.float32).tolist()],
                "hand": None if segment["goal_pose"].get("hand") is None else [float(x) for x in np.asarray(segment["goal_pose"]["hand"], dtype=np.float32).tolist()],
                "left_hand": None if segment["goal_pose"].get("left_hand") is None else [float(x) for x in np.asarray(segment["goal_pose"]["left_hand"], dtype=np.float32).tolist()],
                "right_hand": None if segment["goal_pose"].get("right_hand") is None else [float(x) for x in np.asarray(segment["goal_pose"]["right_hand"], dtype=np.float32).tolist()],
            }
        )

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(summary_path)
    for row in rows:
        print(
            row["sequence_id"],
            row["segment_id"],
            row["goal_type"],
            f"global={row['global_orient_forward_yaw']:.4f}" if row["global_orient_forward_yaw"] is not None else "global=None",
            f"hip={row['hip_facing_yaw']:.4f}" if row["hip_facing_yaw"] is not None else "hip=None",
            f"shoulder={row['shoulder_facing_yaw']:.4f}" if row["shoulder_facing_yaw"] is not None else "shoulder=None",
            f"hand={row['body_to_hand_yaw']:.4f}" if row["body_to_hand_yaw"] is not None else "hand=None",
            row["text"],
        )


if __name__ == "__main__":
    main()
