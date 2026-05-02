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

from datasets.hsi_method import HSI_METHOD_SPECS, HSIUnifiedDataset, hsi_collate_fn
from models.methods import build_hsi_method_model
from train_stage2 import build_lr_scheduler, worker_kwargs

PROJECT_ROOT = Path(__file__).resolve().parent


def output_root(args: argparse.Namespace) -> Path:
    return Path("outputs") / str(args.run_name)


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


class TextLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, file=sys.stderr, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


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


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return out


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def build_writer(args: argparse.Namespace):
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


def build_loader(args: argparse.Namespace, split: str, shuffle: bool, max_records: int = 0) -> tuple[HSIUnifiedDataset, DataLoader]:
    dataset = HSIUnifiedDataset(
        method=args.method,
        dataset=args.dataset,
        split=split,
        max_target_frames=int(args.max_target_frames),
        nb_voxels=int(args.nb_voxels),
        max_records=max_records,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=hsi_collate_fn,
        drop_last=shuffle,
        **worker_kwargs(args),
    )
    return dataset, loader


def loss_step(
    model: torch.nn.Module,
    batch: dict[str, Any],
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    if hasattr(model, "loss_step"):
        return model.loss_step(batch, args)
    x0 = batch["motion"]
    B = x0.shape[0]
    t = torch.randint(0, int(args.num_diffusion_steps), (B,), device=x0.device, dtype=torch.long)
    noise = torch.randn_like(x0)
    history = batch["history_mask"].to(x0.dtype)[..., None]
    noise = noise * (1.0 - history)
    x_t = q_sample(x0, t, noise, sqrt_alpha_bar, sqrt_one_minus_alpha_bar)
    x_t = x_t * (1.0 - history) + x0 * history
    batch = dict(batch)
    batch["x_t"] = x_t
    eps_hat = model(batch, t)["eps_hat"]
    mask = (batch["target_mask"] * batch["valid_mask"]).to(x0.dtype)
    values = (eps_hat - noise).square().mean(dim=-1)
    loss = (values * mask).sum() / mask.sum().clamp_min(1.0)
    with torch.no_grad():
        pred_mm = ((eps_hat - noise).square().mean(dim=-1).sqrt() * mask).sum() / mask.sum().clamp_min(1.0) * 1000.0
        metrics = {
            "loss": float(loss.detach().cpu()),
            "eps_rmse_x1000": float(pred_mm.detach().cpu()),
        }
    return loss, metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch_idx, batch in enumerate(loader):
        if int(args.max_eval_batches) > 0 and batch_idx >= int(args.max_eval_batches):
            break
        batch = move_to_device(batch, device)
        _, metrics = loss_step(model, batch, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, args)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
    return {key: value / max(count, 1) for key, value in sums.items()}


def sampled_motion_metrics(pred_raw: torch.Tensor, gt_raw: torch.Tensor, batch: dict[str, Any]) -> dict[str, float]:
    pred = pred_raw.detach().float().cpu()
    gt = gt_raw.detach().float().cpu()
    mask = (batch["target_mask"].detach().float().cpu() * batch["valid_mask"].detach().float().cpu()).clamp(0.0, 1.0)
    denom = mask.sum().clamp_min(1.0)
    values = (pred - gt).square().mean(dim=-1).sqrt()
    motion_rmse = (values * mask).sum() / denom

    pred_root = pred.reshape(pred.shape[0], pred.shape[1], 28, 3)[:, :, 0]
    gt_root = gt.reshape(gt.shape[0], gt.shape[1], 28, 3)[:, :, 0]
    root_dist = (pred_root - gt_root).square().sum(dim=-1).sqrt()
    root_rmse = (root_dist * mask).sum() / denom

    if pred.shape[1] > 1:
        vel_mask = mask[:, 1:] * mask[:, :-1]
        vel_denom = vel_mask.sum().clamp_min(1.0)
        pred_vel = pred_root[:, 1:] - pred_root[:, :-1]
        gt_vel = gt_root[:, 1:] - gt_root[:, :-1]
        vel_rmse = ((pred_vel - gt_vel).square().sum(dim=-1).sqrt() * vel_mask).sum() / vel_denom
    else:
        vel_rmse = torch.tensor(0.0)

    transitions = []
    history_frames = batch["history_frames"].detach().cpu().long()
    for i in range(pred.shape[0]):
        h = int(history_frames[i].item())
        if 0 < h < pred.shape[1]:
            transitions.append((pred[i, h] - gt[i, h]).square().mean())
    transition = torch.stack(transitions).mean() if transitions else torch.tensor(0.0)
    return {
        "sample_motion_rmse_mm": float(motion_rmse.item() * 1000.0),
        "sample_root_rmse_mm": float(root_rmse.item() * 1000.0),
        "sample_root_vel_rmse_mm_per_frame": float(vel_rmse.item() * 1000.0),
        "sample_transition_mse": float(transition.item()),
    }


@torch.no_grad()
def render_eval_visualizations(
    model: torch.nn.Module,
    dataset: HSIUnifiedDataset,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    logger: TextLogger,
    writer: Any,
) -> None:
    if bool(args.no_eval_vis) or int(args.eval_vis_samples) <= 0:
        return
    if not hasattr(model, "sample"):
        logger.write(f"eval vis skipped step={step}: model has no sample()")
        return
    try:
        from vis.stage2_eval_render import render_stage2_pair
    except Exception as exc:
        logger.write(f"eval vis import failed step={step}: {type(exc).__name__}: {exc}")
        return

    model.eval()
    out_root = output_root(args) / "eval_vis" / f"step_{int(step):08d}"
    rng = random.Random(int(args.seed) + int(step))
    num_items = len(dataset)
    if num_items <= 0:
        return
    num_candidates = min(num_items, max(int(args.eval_vis_samples), int(args.eval_vis_batch_size)))
    indices = rng.sample(range(num_items), num_candidates)
    vis_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(args.eval_vis_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=hsi_collate_fn,
        drop_last=False,
        **worker_kwargs(args),
    )
    sample_steps = int(args.eval_vis_sampling_steps)
    if sample_steps <= 0:
        sample_steps = int(args.num_diffusion_steps)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(step))

    saved = 0
    metric_sums: dict[str, float] = {}
    metric_batches = 0
    for batch in vis_loader:
        if saved >= int(args.eval_vis_samples):
            break
        batch = move_to_device(batch, device)
        try:
            sample = model.sample(batch, num_steps=sample_steps, generator=generator)
        except Exception as exc:
            logger.write(f"eval vis sampling failed step={step}: {type(exc).__name__}: {exc}")
            return
        pred_raw = sample["pred_raw"].detach().cpu()
        gt_raw = sample["gt_raw"].detach().cpu()
        metric_batch = dict(batch)
        for key, value in metric_batch.items():
            if isinstance(value, torch.Tensor):
                metric_batch[key] = value.detach().cpu()
        for key, value in sampled_motion_metrics(pred_raw, gt_raw, metric_batch).items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        metric_batches += 1

        lengths = batch["length"].detach().cpu().long()
        for local_idx in range(pred_raw.shape[0]):
            if saved >= int(args.eval_vis_samples):
                break
            length = int(lengths[local_idx].item())
            sample_dataset = batch.get("dataset_name", [args.dataset] * pred_raw.shape[0])[local_idx]
            meta = {
                "dataset": sample_dataset,
                "training_dataset": args.dataset,
                "method": args.method,
                "step": int(step),
                "sample_index": int(saved),
                "scene_id": batch["scene_id"][local_idx],
                "sequence_id": batch["sequence_id"][local_idx],
                "segment_id": batch["segment_id"][local_idx],
                "goal_type": batch["goal_type"][local_idx],
                "text": batch["text"][local_idx],
                "length": length,
                "sampling_steps": int(sample_steps),
                "fit_smooth_weight": float(args.eval_vis_fit_smooth_weight),
            }
            sample_dir = out_root / f"sample_{saved:03d}"
            try:
                render_stage2_pair(
                    gt_raw[local_idx, :length],
                    pred_raw[local_idx, :length],
                    dataset=sample_dataset,
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
            except Exception as exc:
                logger.write(f"eval vis sample failed step={step} sample={saved}: {type(exc).__name__}: {exc}")
            saved += 1

    metrics = {key: value / max(metric_batches, 1) for key, value in metric_sums.items()}
    if metrics:
        logger.write(f"eval vis metrics step={step} {json.dumps(metrics, sort_keys=True)}")
        write_scalars(writer, "eval_vis", metrics, step)
    logger.write(f"eval vis step={step} saved={saved} dir={out_root}")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    dataset: HSIUnifiedDataset,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_metric": best_metric,
            "args": vars(args),
            "method_spec": HSI_METHOD_SPECS[str(args.method)].__dict__,
            "motion_mean": dataset.motion_mean,
            "motion_std": dataset.motion_std,
            "coord_norm_meta": dataset.coord_norm_meta,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int, float | None]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)) + 1, int(ckpt.get("global_step", 0)), ckpt.get("best_metric")


def add_common_args(parser: argparse.ArgumentParser, method: str) -> None:
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=["none", "warmup_cosine"], default="warmup_cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=2000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-diffusion-steps", type=int, default=100)
    parser.add_argument("--max-target-frames", type=int, default=300)
    parser.add_argument("--nb-voxels", type=int, default=32)
    parser.add_argument("--log-every-steps", type=int, default=2000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=2000)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--eval-vis-every-steps", type=int, default=0)
    parser.add_argument("--eval-vis-samples", type=int, default=10)
    parser.add_argument("--eval-vis-sampling-steps", type=int, default=0)
    parser.add_argument("--eval-vis-batch-size", type=int, default=10)
    parser.add_argument("--eval-vis-device", default="cuda")
    parser.add_argument("--eval-vis-image-size", type=int, default=384)
    parser.add_argument("--eval-vis-fps", type=int, default=30)
    parser.add_argument("--eval-vis-frame-stride", type=int, default=1)
    parser.add_argument("--eval-vis-fit-smooth-weight", type=float, default=0.0)
    parser.add_argument("--smplx-model-dir", default="human_models")
    parser.add_argument("--no-eval-vis", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-original-checkpoint", type=Path, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(method=method)


def run_training(args: argparse.Namespace) -> None:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if args.dry_run:
        dataset, loader = build_loader(args, args.split, shuffle=False, max_records=max(int(args.batch_size), 4))
        batch = next(iter(loader))
        print(f"method={args.method} dataset={args.dataset} split={args.split} records={len(dataset)}")
        print(f"method_spec={HSI_METHOD_SPECS[str(args.method)]}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
            elif isinstance(value, list):
                print(f"{key}: list len={len(value)} first={value[0] if value else None}")
        return

    out_dir = output_root(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = TextLogger(out_dir / "train.log")
    logger.write(f"args {json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, sort_keys=True)}")
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    train_dataset, train_loader = build_loader(args, args.split, shuffle=True)
    eval_dataset, eval_loader = build_loader(args, args.eval_split, shuffle=False)
    logger.write(
        f"train samples {len(train_dataset)} eval samples {len(eval_dataset)} "
        f"motion_dim {train_dataset.motion_dim} spec {HSI_METHOD_SPECS[str(args.method)]}"
    )
    model = build_hsi_method_model(
        args.method,
        motion_dim=train_dataset.motion_dim,
        hidden_dim=int(args.hidden_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        num_timesteps=int(args.num_diffusion_steps),
    ).to(device)
    if args.init_original_checkpoint is not None:
        if not hasattr(model, "load_original_checkpoint"):
            raise ValueError(f"{args.method} does not support --init-original-checkpoint")
        model.load_original_checkpoint(args.init_original_checkpoint, map_location=device)
        logger.write(f"loaded original checkpoint {args.init_original_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    total_steps = max(1, len(train_loader) * int(args.epochs))
    scheduler = build_lr_scheduler(args, optimizer, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")
    writer = build_writer(args)
    sqrt_alpha_bar, sqrt_one_minus_alpha_bar = diffusion_schedule(int(args.num_diffusion_steps), device)
    start_epoch = 0
    global_step = 0
    best_metric: float | None = None
    if args.resume is not None:
        start_epoch, global_step, best_metric = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)
        logger.write(f"resumed {args.resume} epoch={start_epoch} global_step={global_step} best={best_metric}")

    for epoch in range(start_epoch, int(args.epochs)):
        model.train()
        progress = tqdm(train_loader, desc=f"{args.method} epoch {epoch}", unit="batch")
        for batch in progress:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, bool(args.amp) and device.type == "cuda"):
                loss, metrics = loss_step(model, batch, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, args)
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            progress.set_postfix(loss=f"{metrics['loss']:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            if global_step % int(args.log_every_steps) == 0:
                eval_metrics = evaluate(model, eval_loader, device, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, args)
                logger.write(f"eval step={global_step} {json.dumps(eval_metrics, sort_keys=True)}")
                write_scalars(writer, "train", metrics, global_step)
                write_scalars(writer, "eval", eval_metrics, global_step)
                metric = float(eval_metrics.get("loss", math.inf))
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    save_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer, scheduler, scaler, args, epoch, global_step, best_metric, train_dataset)
                    logger.write(f"saved best checkpoint step={global_step} metric={best_metric:.6f}")
                model.train()
            eval_vis_every = int(args.eval_vis_every_steps) if int(args.eval_vis_every_steps) > 0 else int(args.log_every_steps)
            if eval_vis_every > 0 and global_step % eval_vis_every == 0:
                render_eval_visualizations(model, eval_dataset, device, args, global_step, logger, writer)
                model.train()
            if global_step % int(args.checkpoint_every_steps) == 0:
                save_checkpoint(out_dir / "checkpoints" / "latest.pt", model, optimizer, scheduler, scaler, args, epoch, global_step, best_metric, train_dataset)
                logger.write(f"saved latest checkpoint step={global_step}")
        save_checkpoint(out_dir / "checkpoints" / "latest.pt", model, optimizer, scheduler, scaler, args, epoch, global_step, best_metric, train_dataset)
        logger.write(f"epoch end epoch={epoch} global_step={global_step}")
    if writer is not None:
        writer.close()
