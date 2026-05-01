from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.stage2 import Stage2MotionDataset, masked_motion_mse, stage2_collate_fn
from models.stage2_generator import Stage2Generator


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SMPLX_MODEL_DIR = Path("smpl_models")
FOOT_JOINT_GROUPS = ((7, 10), (8, 11))


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def output_root(args: argparse.Namespace) -> Path:
    return Path("outputs") / str(args.run_name)


def checkpoint_dir(args: argparse.Namespace) -> Path:
    return output_root(args) / "checkpoints"


class TextLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, file=sys.stderr, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def build_summary_writer(args: argparse.Namespace):
    if args.no_tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    log_dir = output_root(args) / "tensorboard"
    print(f"tensorboard log dir: {log_dir}", file=sys.stderr, flush=True)
    return SummaryWriter(log_dir=str(log_dir))


def write_scalars(writer: Any, prefix: str, metrics: dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", float(value), int(step))


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return out


def resolve_amp_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported AMP dtype: {name}")


def autocast_context(device: torch.device, dtype: torch.dtype, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def diffusion_schedule(num_steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    betas = torch.linspace(1e-4, 2e-2, int(num_steps), device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alpha_bar.sqrt(), (1.0 - alpha_bar).sqrt()


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
) -> torch.Tensor:
    a = sqrt_alpha_bar[t].view(-1, 1, 1)
    b = sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
    return a * x0 + b * noise


@torch.no_grad()
def sample_stage2_motion(
    model: torch.nn.Module,
    batch: dict[str, Any],
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
    num_steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    x0 = batch["motion"]
    x_t = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    history = batch["history_mask"].to(x0.dtype)[..., None]
    x_t = x_t * (1.0 - history) + x0 * history
    total_steps = int(sqrt_alpha_bar.shape[0])
    steps = max(1, min(int(num_steps), total_steps))
    schedule = torch.linspace(total_steps - 1, 0, steps, device=x0.device).round().long()
    schedule = torch.unique_consecutive(schedule)
    if schedule[-1] != 0:
        schedule = torch.cat([schedule, torch.zeros(1, device=x0.device, dtype=torch.long)])
    last_x0_hat = x_t
    for idx, step in enumerate(schedule):
        diffusion_t = torch.full((x0.shape[0],), int(step.item()), device=x0.device, dtype=torch.long)
        batch_step = dict(batch)
        batch_step["x_t"] = x_t
        last_x0_hat = model(batch_step, diffusion_t=diffusion_t)["x0_hat"]
        if int(step.item()) == 0:
            x_t = last_x0_hat
        else:
            next_step = schedule[idx + 1] if idx + 1 < len(schedule) else torch.zeros((), device=x0.device, dtype=torch.long)
            alpha_t = sqrt_alpha_bar[diffusion_t].view(-1, 1, 1).square()
            alpha_next = sqrt_alpha_bar[int(next_step.item())].view(1, 1, 1).square()
            eps_hat = (x_t - alpha_t.sqrt() * last_x0_hat) / (1.0 - alpha_t).sqrt().clamp_min(1e-8)
            x_t = alpha_next.sqrt() * last_x0_hat + (1.0 - alpha_next).sqrt() * eps_hat
        x_t = x_t * (1.0 - history) + x0 * history
    return x_t


def masked_scalar_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def denormalize_motion(motion: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return motion * std.view(1, 1, -1) + mean.view(1, 1, -1)


def root_velocity_loss(pred_raw: torch.Tensor, target_raw: torch.Tensor, target_mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if pred_raw.shape[1] < 2:
        return pred_raw.new_zeros(())
    pred_vel = pred_raw[:, 1:, :3] - pred_raw[:, :-1, :3]
    target_vel = target_raw[:, 1:, :3] - target_raw[:, :-1, :3]
    mask = target_mask[:, 1:] * valid_mask[:, 1:]
    values = (pred_vel - target_vel).square().mean(dim=-1)
    return masked_scalar_mean(values, mask)


def root_smooth_loss(pred_raw: torch.Tensor, target_mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if pred_raw.shape[1] < 3:
        return pred_raw.new_zeros(())
    acc = pred_raw[:, 2:, :3] - 2.0 * pred_raw[:, 1:-1, :3] + pred_raw[:, :-2, :3]
    mask = target_mask[:, 2:] * valid_mask[:, 2:]
    values = acc.square().mean(dim=-1)
    return masked_scalar_mean(values, mask)


def foot_sliding_loss(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    target_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mode = str(args.foot_sliding_task_mode)
    if mode == "off" or float(args.foot_sliding_loss_weight) <= 0.0:
        zero = pred_raw.new_zeros(())
        return zero, zero, zero
    if mode == "move_wait" and str(args.task) != "move_wait":
        zero = pred_raw.new_zeros(())
        return zero, zero, zero
    if pred_raw.shape[-1] < 84 or pred_raw.shape[1] < 2:
        zero = pred_raw.new_zeros(())
        return zero, zero, zero

    B, T = pred_raw.shape[:2]
    pred_joints = pred_raw[..., :84].reshape(B, T, 28, 3)
    target_joints = target_raw[..., :84].reshape(B, T, 28, 3)
    foot_idx = torch.tensor(FOOT_JOINT_GROUPS, dtype=torch.long, device=pred_raw.device)

    pred_feet = pred_joints.index_select(2, foot_idx.reshape(-1)).reshape(B, T, 2, 2, 3)
    target_feet = target_joints.index_select(2, foot_idx.reshape(-1)).reshape(B, T, 2, 2, 3)
    pred_foot_pos = pred_feet.mean(dim=3)
    target_foot_pos = target_feet.mean(dim=3)

    target_height = target_feet[..., 1].amin(dim=3)
    lowest_height = target_height.amin(dim=-1, keepdim=True)
    height_contact = target_height <= (lowest_height + float(args.foot_contact_height_margin_m))

    target_step = target_foot_pos[:, 1:, :, [0, 2]] - target_foot_pos[:, :-1, :, [0, 2]]
    target_speed = target_step.norm(dim=-1)
    contact = (
        height_contact[:, 1:]
        & height_contact[:, :-1]
        & (target_speed <= float(args.foot_contact_vel_threshold_m_per_frame))
    )
    frame_mask = (target_mask[:, 1:] * target_mask[:, :-1] * valid_mask[:, 1:] * valid_mask[:, :-1]).to(pred_raw.dtype)
    mask = frame_mask[:, :, None] * contact.to(pred_raw.dtype)
    if mask.sum() <= 0:
        zero = pred_raw.new_zeros(())
        return zero, zero, zero

    pred_step = pred_foot_pos[:, 1:, :, [0, 2]] - pred_foot_pos[:, :-1, :, [0, 2]]
    values = pred_step.square().sum(dim=-1)
    loss = masked_scalar_mean(values, mask)
    slide_mm = loss.sqrt() * 1000.0
    contact_ratio = mask.sum() / (frame_mask.sum().clamp_min(1.0) * 2.0)
    return loss, slide_mm, contact_ratio


def transition_loss(pred: torch.Tensor, target: torch.Tensor, pred_raw: torch.Tensor, target_raw: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    B, T = pred.shape[:2]
    h = batch["history_frames"].long().clamp(min=1, max=max(T - 1, 1))
    rows = torch.arange(B, device=pred.device)
    cur = h
    prev = (h - 1).clamp_min(0)
    valid = (batch["target_mask"][rows, cur] * batch["valid_mask"][rows, cur]).to(pred.dtype)
    pose = (pred[rows, cur] - target[rows, cur]).square().mean(dim=-1)
    pred_step = pred_raw[rows, cur, :3] - target_raw[rows, prev, :3]
    target_step = target_raw[rows, cur, :3] - target_raw[rows, prev, :3]
    root_step = (pred_step - target_step).square().mean(dim=-1)
    return masked_scalar_mean(pose + root_step, valid)


def root_plan_tracking_loss(pred_raw: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    B, T = pred_raw.shape[:2]
    h_len = max(1, int(batch["history_frames"].max().item()))
    plan = batch["root_plan"].to(pred_raw.dtype)
    plan_mask = batch["root_plan_mask"].to(pred_raw.dtype)
    length = min(plan.shape[1], max(T - h_len, 0))
    if length <= 0:
        return pred_raw.new_zeros(())
    pred = pred_raw[:, h_len : h_len + length, :3]
    target = plan[:, :length]
    mask = plan_mask[:, :length] * batch["valid_mask"][:, h_len : h_len + length].to(pred_raw.dtype)
    values = (pred - target).square().mean(dim=-1)
    return masked_scalar_mean(values, mask)


def body_goal_proxy_loss(pred_raw: torch.Tensor, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    root = pred_raw[:, :, :3]
    goal = batch["body_goal"].to(root.dtype)
    valid = batch["goal_valid"][:, 0].to(root.dtype) * batch["target_mask"].amax(dim=1).to(root.dtype)
    is_action = (batch["task_id"].long() == 1).to(root.dtype)
    sample_valid = valid * is_action
    if sample_valid.sum() <= 0:
        zero = pred_raw.new_zeros(())
        return zero, zero
    frame_mask = batch["target_mask"].to(root.dtype) * batch["valid_mask"].to(root.dtype)
    dist2 = (root - goal[:, None]).square().sum(dim=-1)
    dist2 = dist2 + (1.0 - frame_mask) * 1.0e6
    min_dist2 = dist2.min(dim=1).values
    loss = masked_scalar_mean(min_dist2, sample_valid)
    metric = masked_scalar_mean(min_dist2.sqrt(), sample_valid)
    return loss, metric


def stage2_component_loss(
    pred: torch.Tensor,
    x0: torch.Tensor,
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    param_loss = masked_motion_mse(pred, x0, batch["target_mask"], batch["valid_mask"])
    root_loss = masked_motion_mse(pred_raw[..., :3], target_raw[..., :3], batch["target_mask"], batch["valid_mask"])
    root_plan_loss = root_plan_tracking_loss(pred_raw, batch)
    vel_loss = root_velocity_loss(pred_raw, target_raw, batch["target_mask"], batch["valid_mask"])
    trans_loss = transition_loss(pred, x0, pred_raw, target_raw, batch)
    smooth_loss = root_smooth_loss(pred_raw, batch["target_mask"], batch["valid_mask"])
    foot_loss, foot_slide_mm, foot_contact_ratio = foot_sliding_loss(
        pred_raw,
        target_raw,
        batch["target_mask"],
        batch["valid_mask"],
        args,
    )
    body_goal_loss, body_goal_min_dist = body_goal_proxy_loss(pred_raw, batch)
    loss = (
        float(args.param_loss_weight) * param_loss
        + float(args.root_loss_weight) * root_loss
        + float(args.root_plan_loss_weight) * root_plan_loss
        + float(args.root_vel_loss_weight) * vel_loss
        + float(args.transition_loss_weight) * trans_loss
        + float(args.smooth_loss_weight) * smooth_loss
        + float(args.foot_sliding_loss_weight) * foot_loss
        + float(args.body_goal_loss_weight) * body_goal_loss
    )

    with torch.no_grad():
        metrics = {
            "loss": float(loss.detach().cpu()),
            "param_mse": float(param_loss.detach().cpu()),
            "root_mse_m2": float(root_loss.detach().cpu()),
            "root_plan_mse_m2": float(root_plan_loss.detach().cpu()),
            "root_vel_mse_m2_per_frame": float(vel_loss.detach().cpu()),
            "transition_mse": float(trans_loss.detach().cpu()),
            "smooth_mse_m2_per_frame2": float(smooth_loss.detach().cpu()),
            "foot_sliding_mse_m2_per_frame": float(foot_loss.detach().cpu()),
            "body_goal_proxy_mse_m2": float(body_goal_loss.detach().cpu()),
            "root_rmse_mm": float(root_loss.detach().sqrt().cpu() * 1000.0),
            "root_plan_rmse_mm": float(root_plan_loss.detach().sqrt().cpu() * 1000.0),
            "root_vel_rmse_mm_per_frame": float(vel_loss.detach().sqrt().cpu() * 1000.0),
            "foot_sliding_mm_per_frame": float(foot_slide_mm.detach().cpu()),
            "foot_contact_ratio": float(foot_contact_ratio.detach().cpu()),
            "body_goal_proxy_min_dist_mm": float(body_goal_min_dist.detach().cpu() * 1000.0),
            "param_rmse": float(param_loss.detach().sqrt().cpu()),
        }
    return loss, metrics


def rollout_forcing_weight(args: argparse.Namespace, global_step: int) -> float:
    if not bool(args.rollout_forcing):
        return 0.0
    target = float(args.rollout_forcing_weight)
    warmup = max(0, int(args.rollout_forcing_warmup_steps))
    if warmup <= 0:
        return target
    return target * min(1.0, max(0.0, float(global_step) / float(warmup)))


def _offset_from_anchor(anchor_root: torch.Tensor) -> torch.Tensor:
    offset = anchor_root.new_zeros(anchor_root.shape)
    offset[..., 0] = anchor_root[..., 0]
    offset[..., 2] = anchor_root[..., 2]
    return offset


def _local_to_world(points: torch.Tensor, anchor_root: torch.Tensor, world_to_local: torch.Tensor) -> torch.Tensor:
    return points @ world_to_local[:, None, :, :] + _offset_from_anchor(anchor_root)[:, None, None, :]


def _world_to_local(points: torch.Tensor, anchor_root: torch.Tensor, world_to_local: torch.Tensor) -> torch.Tensor:
    centered = points - _offset_from_anchor(anchor_root)[:, None, None, :]
    return centered @ world_to_local.transpose(1, 2)[:, None, :, :]


def build_rollout_forcing_batch(
    batch: dict[str, Any],
    pred_raw: torch.Tensor,
    motion_mean: torch.Tensor,
    motion_std: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, float]:
    if str(args.task) != "move_wait" or "rollout_valid" not in batch:
        return None, 0.0
    valid = batch["rollout_valid"].to(device=pred_raw.device) > 0.5
    if valid.sum() <= 0:
        return None, 0.0
    idx = valid.nonzero(as_tuple=False).flatten()
    out: dict[str, Any] = {
        "motion": batch["rollout_motion"][idx].clone(),
        "motion_raw": batch["rollout_motion_raw"][idx].clone(),
        "target_mask": batch["rollout_target_mask"][idx],
        "history_mask": batch["rollout_history_mask"][idx],
        "action_time": batch["rollout_action_time"][idx],
        "valid_mask": batch["rollout_valid_mask"][idx],
        "root_plan": batch["rollout_root_plan"][idx],
        "root_plan_cond": batch["rollout_root_plan_cond"][idx],
        "root_plan_mask": batch["rollout_root_plan_mask"][idx],
        "body_goal": batch["rollout_body_goal"][idx],
        "hand_goal": batch["rollout_hand_goal"][idx],
        "body_goal_cond": batch["rollout_body_goal_cond"][idx],
        "hand_goal_cond": batch["rollout_hand_goal_cond"][idx],
        "goal_valid": batch["rollout_goal_valid"][idx],
        "scene_occ": batch["rollout_scene_occ"][idx],
        "anchor_root": batch["rollout_anchor_root"][idx],
        "anchor_yaw": batch["rollout_anchor_yaw"][idx],
        "world_to_local": batch["rollout_world_to_local"][idx],
        "length": batch["rollout_length"][idx],
        "history_frames": batch["rollout_history_frames"][idx],
        "target_frames": batch["rollout_target_frames"][idx],
        "task_id": batch["task_id"][idx],
    }

    pred_valid = pred_raw.detach()[idx]
    history_frames = out["history_frames"].long()
    stride = int(args.rollout_forcing_stride_frames)
    pred_local_segments = []
    for row in range(pred_valid.shape[0]):
        h = int(history_frames[row].item())
        start = int(stride)
        end = min(start + h, pred_valid.shape[1])
        segment = pred_valid[row, start:end, :84].reshape(end - start, 28, 3)
        if segment.shape[0] < h:
            pad = segment.new_zeros((h - segment.shape[0], 28, 3))
            segment = torch.cat([segment, pad], dim=0)
        pred_local_segments.append(segment)
    pred_a_local = torch.stack(pred_local_segments, dim=0)
    pred_world = _local_to_world(pred_a_local, batch["anchor_root"][idx], batch["world_to_local"][idx])
    pred_b_local = _world_to_local(pred_world, out["anchor_root"], out["world_to_local"]).reshape(pred_world.shape[0], pred_world.shape[1], -1)

    for row in range(out["motion_raw"].shape[0]):
        h = int(history_frames[row].item())
        out["motion_raw"][row, :h, :84] = pred_b_local[row, :h]
    out["motion"] = (out["motion_raw"] - motion_mean.view(1, 1, -1)) / motion_std.view(1, 1, -1)
    return out, float(valid.float().mean().detach().cpu())


def forward_diffusion_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
    motion_mean: torch.Tensor,
    motion_std: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    x0 = batch["motion"]
    B = x0.shape[0]
    t = torch.randint(0, sqrt_alpha_bar.shape[0], (B,), device=x0.device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, sqrt_alpha_bar, sqrt_one_minus_alpha_bar)
    history = batch["history_mask"].to(x0.dtype)[..., None]
    x_t = x_t * (1.0 - history) + x0 * history
    batch = dict(batch)
    batch["x_t"] = x_t
    pred = model(batch, diffusion_t=t)["x0_hat"]
    pred_raw = denormalize_motion(pred, motion_mean, motion_std)
    loss, metrics = stage2_component_loss(pred, x0, pred_raw, batch["motion_raw"], batch, args)
    return loss, metrics, pred_raw


def training_step(
    model: torch.nn.Module,
    batch: dict[str, Any],
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
    motion_mean: torch.Tensor,
    motion_std: torch.Tensor,
    args: argparse.Namespace,
    global_step: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss, metrics, pred_raw = forward_diffusion_loss(
        model,
        batch,
        sqrt_alpha_bar,
        sqrt_one_minus_alpha_bar,
        motion_mean,
        motion_std,
        args,
    )
    metrics["base_loss"] = metrics["loss"]
    weight = rollout_forcing_weight(args, global_step)
    metrics["rollout_weight"] = float(weight)
    metrics["rollout_valid_ratio"] = 0.0
    metrics["rollout_loss"] = 0.0
    if weight > 0.0:
        rollout_batch, valid_ratio = build_rollout_forcing_batch(batch, pred_raw, motion_mean, motion_std, args)
        metrics["rollout_valid_ratio"] = float(valid_ratio)
        if rollout_batch is not None:
            rollout_loss, rollout_metrics, _ = forward_diffusion_loss(
                model,
                rollout_batch,
                sqrt_alpha_bar,
                sqrt_one_minus_alpha_bar,
                motion_mean,
                motion_std,
                args,
            )
            loss = loss + float(weight) * rollout_loss
            metrics["rollout_loss"] = float(rollout_loss.detach().cpu())
            for key in ["root_rmse_mm", "root_plan_rmse_mm", "foot_sliding_mm_per_frame", "foot_contact_ratio"]:
                if key in rollout_metrics:
                    metrics[f"rollout_{key}"] = float(rollout_metrics[key])
    metrics["loss"] = float(loss.detach().cpu())
    return loss, metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
    motion_mean: torch.Tensor,
    motion_std: torch.Tensor,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    global_step: int = 0,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="stage2 eval", unit="batch", leave=False)):
        if int(args.max_eval_batches) > 0 and batch_idx >= int(args.max_eval_batches):
            break
        batch = move_to_device(batch, device)
        with autocast_context(device, amp_dtype, amp_enabled):
            _, metrics = training_step(model, batch, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, motion_mean, motion_std, args, global_step=global_step)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
    denom = max(count, 1)
    return {key: value / denom for key, value in sums.items()}


@torch.no_grad()
def render_eval_visualizations(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    motion_mean: torch.Tensor,
    motion_std: torch.Tensor,
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
    logger: TextLogger,
) -> None:
    if bool(args.no_eval_vis) or int(args.eval_vis_samples) <= 0:
        return
    try:
        from vis.stage2_eval_render import render_stage2_pair
    except Exception as exc:
        logger.write(f"eval vis import failed step={step}: {type(exc).__name__}: {exc}")
        return

    model.eval()
    out_root = output_root(args) / "eval_vis" / f"step_{int(step):08d}"
    saved = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(step))
    dataset = loader.dataset
    num_items = len(dataset)
    rng = random.Random(int(args.seed) + int(step))
    num_vis_candidates = min(num_items, max(int(args.eval_vis_samples), int(args.batch_size)))
    indices = rng.sample(range(num_items), num_vis_candidates)
    vis_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(args.eval_vis_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=stage2_collate_fn,
        drop_last=False,
        **worker_kwargs(args),
    )
    for batch in vis_loader:
        if saved >= int(args.eval_vis_samples):
            break
        batch = move_to_device(batch, device)
        pred = sample_stage2_motion(
            model,
            batch,
            sqrt_alpha_bar,
            sqrt_one_minus_alpha_bar,
            num_steps=int(args.eval_vis_sampling_steps),
            generator=generator,
        )
        pred_raw = denormalize_motion(pred, motion_mean, motion_std).detach().cpu()
        gt_raw = batch["motion_raw"].detach().cpu()
        lengths = batch["length"].detach().cpu().long()
        for local_idx in range(pred_raw.shape[0]):
            if saved >= int(args.eval_vis_samples):
                break
            length = int(lengths[local_idx].item())
            meta = {
                "dataset": args.dataset,
                "task": args.task,
                "step": int(step),
                "batch_index": int(local_idx),
                "sample_index": int(batch["sample_index"][local_idx].detach().cpu().item()),
                "record_index": int(batch["record_index"][local_idx].detach().cpu().item()),
                "target_start": int(batch["target_start"][local_idx].detach().cpu().item()),
                "scene_id": batch["scene_id"][local_idx],
                "sequence_id": batch["sequence_id"][local_idx],
                "segment_id": int(batch["segment_id"][local_idx]),
                "goal_type": batch["goal_type"][local_idx],
                "text": batch["text"][local_idx],
                "length": length,
                "fit_smooth_weight": float(args.eval_vis_fit_smooth_weight),
            }
            sample_dir = out_root / f"sample_{saved:03d}"
            try:
                render_stage2_pair(
                    gt_raw[local_idx, :length],
                    pred_raw[local_idx, :length],
                    dataset=args.dataset,
                    out_dir=sample_dir,
                    smplx_model_dir=resolve_project_path(Path(args.smplx_model_dir)),
                    render_device=str(args.eval_vis_device),
                    image_size=int(args.eval_vis_image_size),
                    fps=int(args.eval_vis_fps),
                    frame_stride=int(args.eval_vis_frame_stride),
                    meta=meta,
                    anchor_root=batch["anchor_root"][local_idx].detach().cpu(),
                    world_to_local=batch["world_to_local"][local_idx].detach().cpu(),
                    fit_smooth_weight=float(args.eval_vis_fit_smooth_weight),
                )
                saved += 1
            except Exception as exc:
                logger.write(f"eval vis sample failed step={step} sample={saved}: {type(exc).__name__}: {exc}")
                saved += 1
    logger.write(f"eval vis step={step} saved={saved} dir={out_root}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    scheduler: Any,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    filename: str,
) -> Path:
    path = checkpoint_dir(args) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
            "scaler": scaler.state_dict() if hasattr(scaler, "state_dict") else None,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_metric": best_metric,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    device: torch.device,
) -> tuple[int, int, float | None]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0)), ckpt.get("best_metric")


def build_lr_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer, total_steps: int):
    if str(args.lr_scheduler) == "none":
        return None
    warmup = max(0, int(args.lr_warmup_steps))
    total = max(1, int(total_steps))
    min_ratio = float(args.min_lr_ratio)

    def lr_lambda(step: int) -> float:
        current = int(step)
        if warmup > 0 and current < warmup:
            return max(float(current + 1) / float(warmup), 1.0e-8)
        denom = max(1, total - warmup)
        progress = min(max(float(current - warmup) / float(denom), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def worker_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.num_workers) <= 0:
        return {}
    return {
        "persistent_workers": not args.no_persistent_workers,
        "prefetch_factor": int(args.prefetch_factor),
    }


def build_loader(args: argparse.Namespace, split: str, shuffle: bool, max_records: int = 0) -> tuple[Stage2MotionDataset, DataLoader]:
    dataset = Stage2MotionDataset(
        dataset=args.dataset,
        task=args.task,
        split=split,
        max_records=max_records,
        nb_voxels=int(args.nb_voxels),
        seed=int(args.seed),
        randomize_offsets=bool(shuffle),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=stage2_collate_fn,
        drop_last=shuffle,
        **worker_kwargs(args),
    )
    return dataset, loader


def dry_run(args: argparse.Namespace) -> None:
    dataset, loader = build_loader(args, args.split, shuffle=False, max_records=max(int(args.batch_size), 4))
    batch = next(iter(loader))
    print(f"dataset={args.dataset} task={args.task} split={args.split} records={len(dataset)}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"{key}: list len={len(value)} first={value[0] if value else None}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 full-body motion training entrypoint.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--task", choices=["move_wait", "action"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-split", default="test")

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=8, help="Kept for StableMoFusion option compatibility; U-Net uses two blocks per stage.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-diffusion-steps", type=int, default=100)
    parser.add_argument("--text-model-name", default="google/mt5-small")
    parser.add_argument("--text-max-length", type=int, default=64)
    parser.add_argument("--text-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nb-voxels", type=int, default=32)
    parser.add_argument("--param-loss-weight", type=float, default=1.0)
    parser.add_argument("--root-loss-weight", type=float, default=1.0)
    parser.add_argument("--root-plan-loss-weight", type=float, default=5.0)
    parser.add_argument("--root-vel-loss-weight", type=float, default=0.5)
    parser.add_argument("--transition-loss-weight", type=float, default=0.5)
    parser.add_argument("--smooth-loss-weight", type=float, default=0.05)
    parser.add_argument("--foot-sliding-loss-weight", type=float, default=1.0)
    parser.add_argument("--foot-sliding-task-mode", choices=["off", "move_wait", "all"], default="move_wait")
    parser.add_argument("--foot-contact-height-margin-m", type=float, default=0.03)
    parser.add_argument("--foot-contact-vel-threshold-m-per-frame", type=float, default=0.015)
    parser.add_argument("--rollout-forcing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rollout-forcing-weight", type=float, default=0.5)
    parser.add_argument("--rollout-forcing-warmup-steps", type=int, default=4000)
    parser.add_argument("--rollout-forcing-stride-frames", type=int, default=15)
    parser.add_argument("--body-goal-loss-weight", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=["none", "warmup_cosine"], default="warmup_cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=2000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")

    parser.add_argument("--log-every-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=5000)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--eval-vis-samples", type=int, default=10)
    parser.add_argument("--eval-vis-batch-size", type=int, default=10)
    parser.add_argument("--eval-vis-every-steps", type=int, default=1000)
    parser.add_argument("--eval-vis-sampling-steps", type=int, default=25)
    parser.add_argument("--eval-vis-image-size", type=int, default=384)
    parser.add_argument("--eval-vis-fps", type=int, default=30)
    parser.add_argument("--eval-vis-frame-stride", type=int, default=4)
    parser.add_argument("--eval-vis-fit-smooth-weight", type=float, default=0.1)
    parser.add_argument("--eval-vis-device", default="cuda")
    parser.add_argument("--smplx-model-dir", type=Path, default=DEFAULT_SMPLX_MODEL_DIR)
    parser.add_argument("--no-eval-vis", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    if args.dry_run:
        dry_run(args)
        return

    out_dir = output_root(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = TextLogger(out_dir / "train.log")
    logger.write(f"args {json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, sort_keys=True)}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    amp_enabled = bool(args.amp) and device.type == "cuda"

    train_dataset, train_loader = build_loader(args, args.split, shuffle=True)
    eval_dataset, eval_loader = build_loader(args, args.eval_split, shuffle=False)
    logger.write(f"train samples {len(train_dataset)} eval samples {len(eval_dataset)} motion_dim {train_dataset.motion_dim}")

    model = Stage2Generator(
        motion_dim=train_dataset.motion_dim,
        task=args.task,
        hidden_dim=int(args.hidden_dim),
        context_dim=int(args.context_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        num_timesteps=int(args.num_diffusion_steps),
        dropout=float(args.dropout),
        text_model_name=str(args.text_model_name),
        text_max_length=int(args.text_max_length),
        text_local_files_only=bool(args.text_local_files_only),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    total_steps = max(1, len(train_loader) * int(args.epochs))
    scheduler = build_lr_scheduler(args, optimizer, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    writer = build_summary_writer(args)
    sqrt_alpha_bar, sqrt_one_minus_alpha_bar = diffusion_schedule(int(args.num_diffusion_steps), device)
    motion_mean = torch.from_numpy(train_dataset.motion_mean).float().to(device)
    motion_std = torch.from_numpy(train_dataset.motion_std).float().to(device)

    start_epoch = 0
    global_step = 0
    best_metric: float | None = None
    best_filename = "best.pt"
    if args.resume is not None:
        start_epoch, global_step, loaded_best_metric = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)
        best_filename = "best_aftter_resume.pt"
        logger.write(
            f"resumed from {args.resume}: epoch={start_epoch} step={global_step} "
            f"previous_best={loaded_best_metric}; reset resume best -> {best_filename}"
        )

    precision = f"amp_{args.amp_dtype}" if amp_enabled else "fp32"
    logger.write(
        f"training dataset={args.dataset} task={args.task} batch={args.batch_size} workers={args.num_workers} "
        f"precision={precision} lr_scheduler={args.lr_scheduler} total_steps={total_steps}"
    )

    for epoch in range(start_epoch, int(args.epochs)):
        model.train()
        pbar = tqdm(train_loader, desc=f"stage2 {args.task} epoch {epoch}", unit="batch")
        for batch in pbar:
            global_step += 1
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_dtype, amp_enabled):
                loss, metrics = training_step(
                    model,
                    batch,
                    sqrt_alpha_bar,
                    sqrt_one_minus_alpha_bar,
                    motion_mean,
                    motion_std,
                    args,
                    global_step=global_step,
                )
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            metrics["lr"] = float(optimizer.param_groups[0]["lr"])
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}", root_mm=f"{metrics['root_rmse_mm']:.1f}", foot_mm=f"{metrics['foot_sliding_mm_per_frame']:.1f}")

            if global_step % int(args.log_every_steps) == 0:
                write_scalars(writer, "train", metrics, global_step)
                logger.write(f"train step={global_step} {json.dumps(metrics, sort_keys=True)}")
                eval_metrics = evaluate(
                    model,
                    eval_loader,
                    device,
                    args,
                    sqrt_alpha_bar,
                    sqrt_one_minus_alpha_bar,
                    motion_mean,
                    motion_std,
                    amp_dtype,
                    amp_enabled,
                    global_step=global_step,
                )
                write_scalars(writer, "eval", eval_metrics, global_step)
                logger.write(f"eval step={global_step} {json.dumps(eval_metrics, sort_keys=True)}")
                metric = eval_metrics.get("loss")
                if metric is not None and (best_metric is None or float(metric) < float(best_metric)):
                    best_metric = float(metric)
                    path = save_checkpoint(model, optimizer, scaler, scheduler, args, epoch, global_step, best_metric, best_filename)
                    logger.write(f"saved best checkpoint {path} metric={best_metric:.6f}")
                if int(args.eval_vis_every_steps) > 0 and global_step % int(args.eval_vis_every_steps) == 0:
                    render_eval_visualizations(
                        model,
                        eval_loader,
                        device,
                        args,
                        global_step,
                        motion_mean,
                        motion_std,
                        sqrt_alpha_bar,
                        sqrt_one_minus_alpha_bar,
                        logger,
                    )
                model.train()

            if global_step % int(args.checkpoint_every_steps) == 0:
                path = save_checkpoint(model, optimizer, scaler, scheduler, args, epoch, global_step, best_metric, "latest.pt")
                logger.write(f"saved latest checkpoint {path}")

        path = save_checkpoint(model, optimizer, scaler, scheduler, args, epoch + 1, global_step, best_metric, "latest.pt")
        logger.write(f"epoch end epoch={epoch} global_step={global_step} latest={path}")

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
