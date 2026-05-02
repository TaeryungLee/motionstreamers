from __future__ import annotations

from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from datasets.stage2 import (
    ROOT_PLAN_FRAMES,
    _sample_yaw_aligned_occupancy_crop,
    _scene_vit_input,
    canonicalize_points,
    decanonicalize_points,
    normalize_xyz,
    pose_dim,
)
from models.stage1_planner import (
    Stage1OptimizerV2Config,
    build_stage1_static_fields_v2,
    configure_stage1_motion_bounds,
    optimize_stage1_trajectory_batch_v2,
)
from models.stage2_generator import Stage2Generator
from train_stage2 import diffusion_schedule, sample_stage2_motion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSED_ROOT = Path("data") / "preprocessed"
PELVIS_INDEX = 0
LEFT_HIP_INDEX = 1
RIGHT_HIP_INDEX = 2
STAGE2_HISTORY_FRAMES = 5
EXECUTE_FRAMES = 15
OVERLAP_FRAMES = 6
MOVEWAIT_FRAMES = EXECUTE_FRAMES + OVERLAP_FRAMES
LINGO_REST_PELVIS_HIPS = np.asarray(
    [
        [0.0000e00, 0.0000e00, 0.0000e00],
        [5.6144e-02, -9.4542e-02, -2.3475e-02],
        [-5.7870e-02, -1.0517e-01, -1.6559e-02],
    ],
    dtype=np.float64,
)


Phase = Literal["MOVE", "WAIT", "ACTION"]


def repo_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def joints_to_root_xz(joints_world: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints_world, dtype=np.float32)
    return joints[..., PELVIS_INDEX, :][..., [0, 2]].astype(np.float32)


def rigid_transform_3d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_arr = np.asarray(src, dtype=np.float64)
    dst_arr = np.asarray(dst, dtype=np.float64)
    centroid_src = src_arr.mean(axis=0)
    centroid_dst = dst_arr.mean(axis=0)
    src_centered = src_arr - centroid_src
    dst_centered = dst_arr - centroid_dst
    h = dst_centered.T @ src_centered
    u, _, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[2, :] *= -1
        rot = vt.T @ u.T
    trans = -rot @ centroid_dst + centroid_src
    return rot.astype(np.float32), trans.astype(np.float32)


def yaw_from_pelvis_hips(joints28_frame: np.ndarray) -> float:
    frame = np.asarray(joints28_frame, dtype=np.float64)
    pelvis_hips = frame[[PELVIS_INDEX, LEFT_HIP_INDEX, RIGHT_HIP_INDEX]]
    if not np.isfinite(pelvis_hips).all():
        return 0.0
    if float(np.linalg.norm(pelvis_hips[LEFT_HIP_INDEX] - pelvis_hips[RIGHT_HIP_INDEX])) <= 1e-8:
        return 0.0
    rot, _ = rigid_transform_3d(pelvis_hips, LINGO_REST_PELVIS_HIPS)
    return float(R.from_matrix(rot).as_euler("zxy")[2])


def world_to_local_from_yaw(yaw: float) -> np.ndarray:
    return R.from_euler("zxy", np.asarray([0.0, 0.0, -float(yaw)], dtype=np.float32)).as_matrix().astype(np.float32)


def xyz_from_xz(xz: np.ndarray, y: float) -> np.ndarray:
    value = np.asarray(xz, dtype=np.float32).reshape(2)
    return np.asarray([value[0], float(y), value[1]], dtype=np.float32)


def body_goal_xyz(goal: dict[str, Any], fallback_y: float) -> np.ndarray | None:
    value = goal.get("body_goal")
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 3 or not np.isfinite(arr[:3]).all():
        return None
    return arr[:3].astype(np.float32)


def hand_goal_xyz(goal: dict[str, Any]) -> np.ndarray | None:
    for key in ("hand_goal", "left_hand_goal", "right_hand_goal"):
        value = goal.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] >= 3 and np.isfinite(arr[:3]).all():
            return arr[:3].astype(np.float32)
    return None


def goal_duration_frames(goal: dict[str, Any], fallback: int = 60) -> int:
    source = goal.get("source_segment") or {}
    if "start" in source and "end" in source:
        return max(1, int(source["end"]) - int(source["start"]) + 1)
    if "duration" in goal:
        return max(1, int(goal["duration"]))
    return int(fallback)


def normalize_goal_type(goal_type: Any) -> str:
    value = "" if goal_type is None else str(goal_type)
    return "move" if value in {"walk", "move"} else value


def smoothstep_weights(length: int = OVERLAP_FRAMES) -> np.ndarray:
    if int(length) <= 1:
        return np.ones((max(1, int(length)), 1, 1), dtype=np.float32)
    t = np.linspace(0.0, 1.0, int(length), dtype=np.float32)
    alpha = t * t * (3.0 - 2.0 * t)
    return alpha[:, None, None].astype(np.float32)


def blend_overlap(previous_tail: np.ndarray | None, generated_world: np.ndarray, overlap: int = OVERLAP_FRAMES) -> np.ndarray:
    out = np.asarray(generated_world, dtype=np.float32).copy()
    if previous_tail is None or len(previous_tail) == 0 or len(out) == 0:
        return out
    n = min(int(overlap), len(previous_tail), len(out))
    alpha = smoothstep_weights(n)
    out[:n] = (1.0 - alpha) * np.asarray(previous_tail[-n:], dtype=np.float32) + alpha * out[:n]
    return out


def clamp_plan_after_arrival(root_plan: np.ndarray, target_xy: np.ndarray, arrival_radius: float) -> np.ndarray:
    plan = np.asarray(root_plan, dtype=np.float32).copy()
    if len(plan) == 0:
        return plan
    dist = np.linalg.norm(plan - np.asarray(target_xy, dtype=np.float32)[None], axis=-1)
    hit = np.flatnonzero(dist <= float(arrival_radius))
    if len(hit) > 0:
        first = int(hit[0])
        plan[first + 1 :] = plan[first]
    return plan


@dataclass
class Stage1Plan:
    planned_root_world_30: np.ndarray
    pred_ego_world_30: np.ndarray
    pred_others_world_30: np.ndarray
    debug: dict[str, float]
    reached_in_plan: bool


@dataclass
class Stage2Generation:
    joints_world: np.ndarray
    root_world: np.ndarray
    anchor_root: np.ndarray
    anchor_yaw: float
    world_to_local: np.ndarray
    task: str
    target_frames: int


@dataclass
class PhaseCommand:
    phase: Phase
    reason: str
    target_xy: np.ndarray
    body_goal: np.ndarray | None = None
    hand_goal: np.ndarray | None = None
    goal: dict[str, Any] | None = None


@dataclass
class WorldState:
    sim_t: int
    ego_joints_world: list[np.ndarray]
    previous_tail: np.ndarray | None = None
    phase_per_frame: list[str] = field(default_factory=list)
    goal_index_per_frame: list[int] = field(default_factory=list)
    segment_id_per_frame: list[int] = field(default_factory=list)

    @property
    def ego_root_world(self) -> np.ndarray:
        if not self.ego_joints_world:
            return np.zeros((0, 2), dtype=np.float32)
        return joints_to_root_xz(np.stack(self.ego_joints_world, axis=0))

    @property
    def current_joints(self) -> np.ndarray:
        return np.asarray(self.ego_joints_world[-1], dtype=np.float32)

    @property
    def current_root_xy(self) -> np.ndarray:
        return np.asarray(self.current_joints[PELVIS_INDEX, [0, 2]], dtype=np.float32)

    @property
    def current_root_y(self) -> float:
        return float(self.current_joints[PELVIS_INDEX, 1])

    def history_joints(self, frames: int = STAGE2_HISTORY_FRAMES) -> np.ndarray:
        values = np.stack(self.ego_joints_world, axis=0).astype(np.float32)
        if len(values) >= int(frames):
            return values[-int(frames) :]
        pad = np.repeat(values[:1], int(frames) - len(values), axis=0)
        return np.concatenate([pad, values], axis=0)

    def history_root_xz(self, frames: int) -> np.ndarray:
        roots = self.ego_root_world
        if len(roots) >= int(frames):
            return roots[-int(frames) :]
        pad = np.repeat(roots[:1], int(frames) - len(roots), axis=0)
        return np.concatenate([pad, roots], axis=0)

    def append_window(self, joints_world: np.ndarray, phase: str, goal_index: int, segment_id: int) -> None:
        for frame in np.asarray(joints_world, dtype=np.float32):
            self.ego_joints_world.append(frame.astype(np.float32))
            self.phase_per_frame.append(str(phase))
            self.goal_index_per_frame.append(int(goal_index))
            self.segment_id_per_frame.append(int(segment_id))
            self.sim_t += 1


class Stage1Runtime:
    def __init__(
        self,
        checkpoint: Path,
        optimizer_config: Path,
        device: torch.device,
        horizon: int = 30,
        past_frames: int = 30,
        future_frames: int = 72,
        num_sampling_steps: int = 50,
    ) -> None:
        from simulate_stage1_episode_loop import build_model

        self.device = device
        self.horizon = int(horizon)
        self.past_frames = int(past_frames)
        self.future_frames = int(future_frames)
        self.num_sampling_steps = int(num_sampling_steps)
        self.model = build_model(repo_path(checkpoint), self.past_frames, self.future_frames, device)
        self.config = self._load_optimizer_config(repo_path(optimizer_config), self.horizon)

    @staticmethod
    def _load_optimizer_config(path: Path, horizon: int) -> Stage1OptimizerV2Config:
        payload = load_json(path)
        values = dict(payload.get("config") or payload)
        allowed = Stage1OptimizerV2Config.__dataclass_fields__.keys()
        config = Stage1OptimizerV2Config(**{key: value for key, value in values.items() if key in allowed})
        config.horizon = int(horizon)
        return configure_stage1_motion_bounds(config, values, payload)

    def plan(
        self,
        scene: Any,
        ego_root_history: np.ndarray,
        others_clips: list[Any],
        sim_t: int,
        target_xy: np.ndarray,
    ) -> Stage1Plan:
        from simulate_stage1_episode_loop import (
            LOCAL_ORIGIN,
            LOCAL_RESOLUTION,
            MODEL_MAP_SIZE,
            build_predictor_batch,
            clipped_goal_for_map,
            future_slice,
            goal_distance_field_to_body_goal,
            make_gaussian_map,
        )

        batch, scene_crop, others_future_abs = build_predictor_batch(
            scene,
            np.asarray(ego_root_history, dtype=np.float32),
            others_clips,
            int(sim_t),
            np.asarray(target_xy, dtype=np.float32),
            self.past_frames,
            self.future_frames,
            self.device,
        )
        with torch.no_grad():
            sample = self.model.sample(batch, num_steps=self.num_sampling_steps, deterministic=True)
        pred = sample["x0_hat"]
        pred_ego = pred[:, 0, : self.horizon]
        pred_others = pred[:, 1:, : self.horizon]
        others_valid = batch["entity_valid"][:, 1:]
        current_xy = np.asarray(ego_root_history[-1], dtype=np.float32)
        target_rel_xy = np.asarray(target_xy - current_xy, dtype=np.float32)
        goal_map_point, _ = clipped_goal_for_map(target_rel_xy)
        goal_map = make_gaussian_map(
            goal_map_point[None],
            MODEL_MAP_SIZE[0],
            MODEL_MAP_SIZE[1],
            LOCAL_RESOLUTION,
            LOCAL_ORIGIN,
            sigma=0.25,
        )
        static_fields = build_stage1_static_fields_v2(scene_crop[1], goal_map, self.config, start_xy=(0.0, 0.0))
        goal_distance = goal_distance_field_to_body_goal(target_rel_xy)
        if len(ego_root_history) >= 2:
            current_vel = (ego_root_history[-1] - ego_root_history[-2]) / float(self.config.dt)
        else:
            current_vel = np.zeros((2,), dtype=np.float32)
        opt_rel, metrics = optimize_stage1_trajectory_batch_v2(
            pred_ego=pred_ego,
            pred_others=pred_others,
            others_valid=others_valid,
            distance_fields=torch.from_numpy(scene_crop[1][None]).float().to(self.device),
            static_fields=torch.from_numpy(static_fields["final_static"][None]).float().to(self.device),
            goal_distance_fields=torch.from_numpy(goal_distance[None]).float().to(self.device),
            gt_ego=torch.zeros((1, self.horizon, 2), dtype=torch.float32, device=self.device),
            gt_others=torch.from_numpy((others_future_abs[:, : self.horizon] - current_xy[None, None])[None]).float().to(self.device),
            current_vel_xy=torch.from_numpy(current_vel[None]).float().to(self.device),
            speed_bound=float(self.config.speed_bound_mps),
            config=self.config,
        )
        opt_world = opt_rel[0].detach().cpu().numpy().astype(np.float32) + current_xy[None]
        pred_ego_world = pred_ego[0].detach().cpu().numpy().astype(np.float32) + current_xy[None]
        pred_others_world = pred_others[0].detach().cpu().numpy().astype(np.float32) + current_xy[None, None]
        reached = bool((np.linalg.norm(opt_world - target_xy[None], axis=-1) <= float(self.config.goal_threshold)).any())
        return Stage1Plan(
            planned_root_world_30=opt_world,
            pred_ego_world_30=pred_ego_world,
            pred_others_world_30=pred_others_world,
            debug=metrics,
            reached_in_plan=reached,
        )


class Stage2Runtime:
    def __init__(
        self,
        dataset: str,
        move_wait_checkpoint: Path,
        action_checkpoint: Path,
        device: torch.device,
        num_sampling_steps: int = 25,
        preprocessed_root: Path = DEFAULT_PREPROCESSED_ROOT,
        nb_voxels: int = 32,
    ) -> None:
        self.dataset = str(dataset)
        self.device = device
        self.num_sampling_steps = int(num_sampling_steps)
        self.preprocessed_root = repo_path(preprocessed_root)
        self.dataset_root = self.preprocessed_root / self.dataset
        self.nb_voxels = int(nb_voxels)
        self.coord_norm_meta = self._load_coord_norm_meta()
        self.motion_mean, self.motion_std = self._motion_mean_std()
        self.motion_mean_t = torch.from_numpy(self.motion_mean).float().to(device)
        self.motion_std_t = torch.from_numpy(self.motion_std).float().to(device)
        self.move_wait = self._load_model(repo_path(move_wait_checkpoint), "move_wait")
        self.action = self._load_model(repo_path(action_checkpoint), "action")
        steps = int(max(self.move_wait.num_timesteps, self.action.num_timesteps))
        self.sqrt_alpha_bar, self.sqrt_one_minus_alpha_bar = diffusion_schedule(steps, device)

    def _load_coord_norm_meta(self) -> dict[str, float]:
        path = self.dataset_root / "stage2" / "joint_coord_norm_meta.json"
        if path.exists():
            data = load_json(path)
            return {key: float(data[key]) for key in ("x_scale", "z_scale", "y_center", "y_scale")}
        return {"x_scale": 3.0, "z_scale": 4.0, "y_center": 1.0, "y_scale": 1.0}

    def _motion_mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        mean_joint = np.array([0.0, self.coord_norm_meta["y_center"], 0.0], dtype=np.float32)
        std_joint = np.array([self.coord_norm_meta["x_scale"], self.coord_norm_meta["y_scale"], self.coord_norm_meta["z_scale"]], dtype=np.float32)
        mean = np.tile(mean_joint, 28).astype(np.float32)
        std = np.maximum(np.tile(std_joint, 28).astype(np.float32), 1e-6)
        return mean, std

    def _load_model(self, checkpoint: Path, task: str) -> Stage2Generator:
        ckpt = torch.load(checkpoint, map_location="cpu")
        ckpt_args = dict(ckpt.get("args") or {})
        model = Stage2Generator(
            motion_dim=pose_dim(self.dataset),
            task=task,
            hidden_dim=int(ckpt_args.get("hidden_dim", 512)),
            context_dim=int(ckpt_args.get("context_dim", 256)),
            num_heads=int(ckpt_args.get("num_heads", 8)),
            num_layers=int(ckpt_args.get("num_layers", 8)),
            num_timesteps=int(ckpt_args.get("num_diffusion_steps", 100)),
            dropout=float(ckpt_args.get("dropout", 0.1)),
            text_model_name=str(ckpt_args.get("text_model_name", "google/mt5-small")),
            text_max_length=int(ckpt_args.get("text_max_length", 64)),
            text_local_files_only=bool(ckpt_args.get("text_local_files_only", True)),
        )
        model.load_state_dict(ckpt["model"])
        model.to(self.device).eval()
        return model

    def _scene_occ(self, scene_id: str, anchor_root: np.ndarray, world_to_local: np.ndarray, goal_world: np.ndarray) -> np.ndarray:
        scene_path = self.dataset_root / "scenes_v2" / scene_id / "scene.json"
        scene = load_json(scene_path)
        occ_path = repo_path(scene["occupancy_grid_path"])
        occ = np.load(occ_path, mmap_mode="r")
        grid_meta = scene["grid_meta"]
        current = _sample_yaw_aligned_occupancy_crop(
            occ,
            grid_meta,
            anchor_root,
            world_to_local,
            nb_voxels=self.nb_voxels,
        )
        goal = _sample_yaw_aligned_occupancy_crop(
            occ,
            grid_meta,
            goal_world,
            world_to_local,
            nb_voxels=self.nb_voxels,
        )
        return _scene_vit_input(current, goal)

    def _batch(
        self,
        scene_id: str,
        history_world: np.ndarray,
        target_frames: int,
        task: str,
        body_goal_world: np.ndarray | None,
        hand_goal_world: np.ndarray | None,
        root_plan_world: np.ndarray | None,
        text: str = "",
        goal_type: str = "",
    ) -> dict[str, Any]:
        history = np.asarray(history_world, dtype=np.float32)
        if history.shape[0] != STAGE2_HISTORY_FRAMES:
            raise ValueError(f"history must have {STAGE2_HISTORY_FRAMES} frames, got {history.shape}")
        target_frames = int(target_frames)
        total_frames = STAGE2_HISTORY_FRAMES + target_frames
        anchor_frame = history[-1]
        anchor_root = anchor_frame[PELVIS_INDEX].astype(np.float32)
        yaw = yaw_from_pelvis_hips(anchor_frame)
        world_to_local = world_to_local_from_yaw(yaw)
        history_local = canonicalize_points(history.reshape(-1, 3), anchor_root, world_to_local).reshape(history.shape)
        motion_raw = np.zeros((total_frames, pose_dim(self.dataset)), dtype=np.float32)
        motion_raw[:STAGE2_HISTORY_FRAMES] = history_local.reshape(STAGE2_HISTORY_FRAMES, -1)
        target_mask = np.zeros((total_frames,), dtype=np.float32)
        target_mask[STAGE2_HISTORY_FRAMES:] = 1.0
        history_mask = 1.0 - target_mask
        action_time = np.zeros((total_frames,), dtype=np.float32)
        if task == "action" and target_frames > 1:
            action_time[STAGE2_HISTORY_FRAMES:] = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)

        if body_goal_world is None:
            body_goal_world = history[-1, PELVIS_INDEX].astype(np.float32)
            body_valid = 0.0
        else:
            body_goal_world = np.asarray(body_goal_world, dtype=np.float32)
            body_valid = 1.0
        if hand_goal_world is None:
            hand_goal_world = np.zeros((3,), dtype=np.float32)
            hand_valid = 0.0
        else:
            hand_goal_world = np.asarray(hand_goal_world, dtype=np.float32)
            hand_valid = 1.0
        body_goal_local = canonicalize_points(body_goal_world[None], anchor_root, world_to_local)[0]
        hand_goal_local = canonicalize_points(hand_goal_world[None], anchor_root, world_to_local)[0]

        root_plan = np.zeros((ROOT_PLAN_FRAMES, 3), dtype=np.float32)
        root_plan_mask = np.zeros((ROOT_PLAN_FRAMES,), dtype=np.float32)
        scene_goal = body_goal_world
        if task == "move_wait" and root_plan_world is not None:
            plan_world = np.asarray(root_plan_world, dtype=np.float32)
            plan_len = min(ROOT_PLAN_FRAMES, len(plan_world))
            root_plan[:plan_len] = canonicalize_points(plan_world[:plan_len], anchor_root, world_to_local)
            root_plan_mask[:plan_len] = 1.0
            if plan_len > 0:
                scene_goal = plan_world[plan_len - 1]

        scene_occ = self._scene_occ(scene_id, anchor_root, world_to_local, scene_goal)
        motion = (motion_raw - self.motion_mean[None]) / self.motion_std[None]
        root_plan_cond = normalize_xyz(root_plan, self.coord_norm_meta)
        body_goal_cond = normalize_xyz(body_goal_local[None], self.coord_norm_meta)[0]
        hand_goal_cond = normalize_xyz(hand_goal_local[None], self.coord_norm_meta)[0]

        batch: dict[str, Any] = {
            "motion": torch.from_numpy(motion[None]).float().to(self.device),
            "motion_raw": torch.from_numpy(motion_raw[None]).float().to(self.device),
            "target_mask": torch.from_numpy(target_mask[None]).float().to(self.device),
            "history_mask": torch.from_numpy(history_mask[None]).float().to(self.device),
            "action_time": torch.from_numpy(action_time[None]).float().to(self.device),
            "valid_mask": torch.ones((1, total_frames), dtype=torch.float32, device=self.device),
            "root_plan": torch.from_numpy(root_plan[None]).float().to(self.device),
            "root_plan_cond": torch.from_numpy(root_plan_cond[None]).float().to(self.device),
            "root_plan_mask": torch.from_numpy(root_plan_mask[None]).float().to(self.device),
            "body_goal": torch.from_numpy(body_goal_local[None]).float().to(self.device),
            "hand_goal": torch.from_numpy(hand_goal_local[None]).float().to(self.device),
            "body_goal_cond": torch.from_numpy(body_goal_cond[None]).float().to(self.device),
            "hand_goal_cond": torch.from_numpy(hand_goal_cond[None]).float().to(self.device),
            "goal_valid": torch.tensor([[body_valid, hand_valid]], dtype=torch.float32, device=self.device),
            "scene_occ": torch.from_numpy(scene_occ[None]).float().to(self.device),
            "anchor_root": torch.from_numpy(anchor_root[None]).float().to(self.device),
            "anchor_yaw": torch.tensor([yaw], dtype=torch.float32, device=self.device),
            "world_to_local": torch.from_numpy(world_to_local[None]).float().to(self.device),
            "length": torch.tensor([total_frames], dtype=torch.long, device=self.device),
            "history_frames": torch.tensor([STAGE2_HISTORY_FRAMES], dtype=torch.long, device=self.device),
            "target_frames": torch.tensor([target_frames], dtype=torch.long, device=self.device),
            "task_id": torch.tensor([0 if task == "move_wait" else 1], dtype=torch.long, device=self.device),
            "text_id": torch.tensor([0], dtype=torch.long, device=self.device),
            "goal_type_id": torch.tensor([0], dtype=torch.long, device=self.device),
            "record_index": torch.tensor([0], dtype=torch.long, device=self.device),
            "sample_index": torch.tensor([0], dtype=torch.long, device=self.device),
            "target_start": torch.tensor([0], dtype=torch.long, device=self.device),
            "scene_id": [scene_id],
            "sequence_id": [""],
            "segment_id": [-1],
            "goal_type": [goal_type],
            "text": [text],
        }
        batch["_anchor_root_np"] = anchor_root
        batch["_anchor_yaw_np"] = float(yaw)
        batch["_world_to_local_np"] = world_to_local
        return batch

    @torch.no_grad()
    def generate_move_wait(
        self,
        scene_id: str,
        history_world: np.ndarray,
        root_plan_world: np.ndarray,
        body_goal_world: np.ndarray | None,
        goal_type: str = "move",
    ) -> Stage2Generation:
        batch = self._batch(
            scene_id=scene_id,
            history_world=history_world,
            target_frames=MOVEWAIT_FRAMES,
            task="move_wait",
            body_goal_world=body_goal_world,
            hand_goal_world=None,
            root_plan_world=root_plan_world,
            goal_type=goal_type,
        )
        pred = sample_stage2_motion(
            self.move_wait,
            batch,
            self.sqrt_alpha_bar[: self.move_wait.num_timesteps],
            self.sqrt_one_minus_alpha_bar[: self.move_wait.num_timesteps],
            num_steps=self.num_sampling_steps,
        )
        raw = (pred * self.motion_std_t.view(1, 1, -1) + self.motion_mean_t.view(1, 1, -1))[0].detach().cpu().numpy()
        local = raw[STAGE2_HISTORY_FRAMES:].reshape(MOVEWAIT_FRAMES, 28, 3)
        world = decanonicalize_points(local.reshape(-1, 3), batch["_anchor_root_np"], batch["_world_to_local_np"]).reshape(local.shape)
        return Stage2Generation(
            joints_world=world.astype(np.float32),
            root_world=joints_to_root_xz(world),
            anchor_root=batch["_anchor_root_np"],
            anchor_yaw=float(batch["_anchor_yaw_np"]),
            world_to_local=batch["_world_to_local_np"],
            task="move_wait",
            target_frames=MOVEWAIT_FRAMES,
        )

    @torch.no_grad()
    def generate_action(
        self,
        scene_id: str,
        history_world: np.ndarray,
        duration: int,
        body_goal_world: np.ndarray | None,
        hand_goal_world: np.ndarray | None,
        text: str,
        goal_type: str,
    ) -> Stage2Generation:
        target_frames = int(duration) + OVERLAP_FRAMES
        batch = self._batch(
            scene_id=scene_id,
            history_world=history_world,
            target_frames=target_frames,
            task="action",
            body_goal_world=body_goal_world,
            hand_goal_world=hand_goal_world,
            root_plan_world=None,
            text=text,
            goal_type=goal_type,
        )
        pred = sample_stage2_motion(
            self.action,
            batch,
            self.sqrt_alpha_bar[: self.action.num_timesteps],
            self.sqrt_one_minus_alpha_bar[: self.action.num_timesteps],
            num_steps=self.num_sampling_steps,
        )
        raw = (pred * self.motion_std_t.view(1, 1, -1) + self.motion_mean_t.view(1, 1, -1))[0].detach().cpu().numpy()
        local = raw[STAGE2_HISTORY_FRAMES:].reshape(target_frames, 28, 3)
        world = decanonicalize_points(local.reshape(-1, 3), batch["_anchor_root_np"], batch["_world_to_local_np"]).reshape(local.shape)
        return Stage2Generation(
            joints_world=world.astype(np.float32),
            root_world=joints_to_root_xz(world),
            anchor_root=batch["_anchor_root_np"],
            anchor_yaw=float(batch["_anchor_yaw_np"]),
            world_to_local=batch["_world_to_local_np"],
            task="action",
            target_frames=target_frames,
        )


def fit_joints28_episode_to_smplx(
    joints_world: np.ndarray,
    out_path: Path,
    device: torch.device,
    smooth_weight: float = 0.0,
) -> None:
    from models.joints_to_smplx import JointsToSMPLX, joints_to_smpl

    joints_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 34, 40, 49]
    model = JointsToSMPLX(input_dim=84, output_dim=132, hidden_dim=128).to(device)
    ckpt_path = repo_path(Path("ckpts") / "joints2smplx" / "train_joint2smpl__input_84__hidden_128__all__last.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    joints = torch.from_numpy(np.asarray(joints_world, dtype=np.float32).reshape(len(joints_world), -1)).float().to(device)
    with torch.enable_grad():
        pose, transl, left_hand, right_hand, _ = joints_to_smpl(model, joints, joints_ind, 1, smooth_weight=float(smooth_weight))
    payload = {
        "global_orient": np.asarray(pose[:, :3], dtype=np.float32),
        "body_pose": np.asarray(pose[:, 3:], dtype=np.float32),
        "transl": np.asarray(transl, dtype=np.float32),
        "left_hand_pose": np.asarray(left_hand, dtype=np.float32),
        "right_hand_pose": np.asarray(right_hand, dtype=np.float32),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
