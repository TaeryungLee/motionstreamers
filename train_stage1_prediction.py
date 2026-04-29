from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import LingoPlanning, TrumansPlanning, planning_collate_fn


DATASET_CLASSES = {
    "trumans": TrumansPlanning,
    "lingo": LingoPlanning,
}

MODEL_ALIASES = {
    "Stage1Predictor": "models.stage1_predictor:Stage1Predictor",
}

MODEL_REGISTRY = {
    "v2": "models.stage1_predictor:Stage1Predictor",
}

PREPROCESSED_ROOT = Path("data/preprocessed")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def output_root(args: argparse.Namespace) -> Path:
    return Path("outputs") / str(args.run_name)


def checkpoint_dir(args: argparse.Namespace) -> Path:
    return output_root(args) / "checkpoints"


def build_summary_writer(args: argparse.Namespace):
    if args.no_tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        log("tensorboard is not available; pass --no-tensorboard or install tensorboard")
        return None
    log_dir = output_root(args) / "tensorboard"
    log(f"tensorboard log dir: {log_dir}")
    return SummaryWriter(log_dir=str(log_dir))


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


def write_scalars(writer: Any, prefix: str, metrics: dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", float(value), int(step))


def import_object(spec: str) -> type[torch.nn.Module]:
    spec = MODEL_ALIASES.get(spec, spec)
    module_name, object_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def build_dataset(args: argparse.Namespace, split: str | None = None) -> torch.utils.data.Dataset:
    dataset_split = split or args.split
    root = getattr(args, "preprocessed_root", PREPROCESSED_ROOT)
    return DATASET_CLASSES[args.dataset](
        root=root,
        split=dataset_split,
        scene_id=args.scene_id,
        max_others=args.max_others,
        include_distance_map=not args.no_distance_map,
        model_past_frames=args.past_frames,
        model_future_frames=args.future_frames,
    )


def build_dataset_from_sample_paths(
    args: argparse.Namespace,
    split: str,
    sample_paths_file: Path,
) -> torch.utils.data.Dataset:
    return DATASET_CLASSES[args.dataset](
        root=PREPROCESSED_ROOT,
        split=split,
        scene_id=args.scene_id,
        max_others=args.max_others,
        include_distance_map=not args.no_distance_map,
        sample_paths_file=sample_paths_file,
        model_past_frames=args.past_frames,
        model_future_frames=args.future_frames,
    )


def dataloader_worker_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_workers <= 0:
        return {}
    return {
        "persistent_workers": not args.no_persistent_workers,
        "prefetch_factor": int(args.prefetch_factor),
    }


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


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


def common_eval_samples_path(args: argparse.Namespace, split: str = "test") -> Path:
    scene = args.scene_id or "all"
    name = (
        f"{split}_p{int(args.past_frames)}_f{int(args.future_frames)}"
        f"_scene-{scene}_n{int(args.eval_num_samples)}_seed{int(args.eval_seed)}.json"
    )
    return PREPROCESSED_ROOT / args.dataset / "stage1_eval_samples_v2" / name


def build_common_eval_sample_file(args: argparse.Namespace, split: str = "test") -> Path:
    path = common_eval_samples_path(args, split=split)
    if path.exists():
        return path
    dataset = build_dataset(args, split=split)
    sample_paths = list(getattr(dataset, "sample_paths"))
    rng = random.Random(int(args.eval_seed))
    rng.shuffle(sample_paths)
    selected = sorted(sample_paths[: min(int(args.eval_num_samples), len(sample_paths))])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": split,
                "scene_id": args.scene_id,
                "past_frames": int(args.past_frames),
                "future_frames": int(args.future_frames),
                "seed": args.eval_seed,
                "requested_num_samples": int(args.eval_num_samples),
                "num_samples": len(selected),
                "sample_paths": selected,
            },
            indent=2,
        )
    )
    return path


def build_model(args: argparse.Namespace, sample_batch: dict[str, torch.Tensor]) -> torch.nn.Module:
    model_cls = import_object(MODEL_REGISTRY[args.model_version])
    return model_cls(
        slots=int(sample_batch["x0"].shape[1]),
        past_frames=int(sample_batch["past_rel_pos"].shape[2]),
        future_frames=int(sample_batch["x0"].shape[2]),
        scene_channels=int(sample_batch["scene_maps"].shape[1]),
        hidden_dim=int(args.model_hidden_dim),
        num_layers=int(args.model_layers),
        num_heads=int(args.model_heads),
        dropout=float(args.model_dropout),
        num_timesteps=int(args.num_diffusion_steps),
    )


def call_model(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    diffusion_t: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    output = model(batch) if diffusion_t is None and noise is None else model(batch, diffusion_t=diffusion_t, noise=noise)
    if isinstance(output, torch.Tensor):
        return {"x0_hat": output}
    if not isinstance(output, dict) or "x0_hat" not in output:
        raise TypeError("Stage-1 model must return a tensor or a dict containing x0_hat.")
    return output


def masked_mean_per_slot(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def prediction_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    vel_loss_weight: float,
    smooth_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = call_model(model, batch)
    pred = output["x0_hat"]
    target = batch["target_x0"]
    mask = batch["entity_valid"]

    pos_per_slot = (pred - target).square().mean(dim=(2, 3))
    pos_loss = masked_mean_per_slot(pos_per_slot, mask)
    ego_pos_loss = pos_per_slot[:, 0].mean()
    others_pos_loss = masked_mean_per_slot(pos_per_slot[:, 1:], mask[:, 1:])

    pred_vel = pred[:, :, 1:] - pred[:, :, :-1]
    target_vel = target[:, :, 1:] - target[:, :, :-1]
    vel_per_slot = (pred_vel - target_vel).square().mean(dim=(2, 3))
    vel_loss = masked_mean_per_slot(vel_per_slot, mask)

    if pred.shape[2] > 2:
        pred_acc = pred[:, :, 2:] - 2.0 * pred[:, :, 1:-1] + pred[:, :, :-2]
        smooth_per_slot = pred_acc.square().mean(dim=(2, 3))
        smooth_loss = masked_mean_per_slot(smooth_per_slot, mask)
    else:
        smooth_loss = pred.new_zeros(())

    loss = pos_loss + float(vel_loss_weight) * vel_loss + float(smooth_loss_weight) * smooth_loss
    metrics = {
        "loss": float(loss.detach().cpu()),
        "pos_mse": float(pos_loss.detach().cpu()),
        "ego_pos_mse": float(ego_pos_loss.detach().cpu()),
        "others_pos_mse": float(others_pos_loss.detach().cpu()),
        "vel_mse": float(vel_loss.detach().cpu()),
        "smooth_mse": float(smooth_loss.detach().cpu()),
        "pos_rmse_mm": float(pos_loss.detach().sqrt().cpu() * 1000.0),
        "ego_pos_rmse_mm": float(ego_pos_loss.detach().sqrt().cpu() * 1000.0),
        "others_pos_rmse_mm": float(others_pos_loss.detach().sqrt().cpu() * 1000.0),
        "vel_rmse_mm_per_frame": float(vel_loss.detach().sqrt().cpu() * 1000.0),
        "smooth_rmse_mm_per_frame2": float(smooth_loss.detach().sqrt().cpu() * 1000.0),
    }
    return loss, metrics


def trajectory_metrics(pred: torch.Tensor, target: torch.Tensor, entity_valid: torch.Tensor) -> dict[str, float]:
    dist = (pred - target).norm(dim=-1)
    valid = entity_valid.to(dist.dtype)
    mean_error_per_slot = dist.mean(dim=-1)
    final_error_per_slot = dist[:, :, -1]
    mean_error = masked_mean_per_slot(mean_error_per_slot, entity_valid) * 1000.0
    final_error = masked_mean_per_slot(final_error_per_slot, entity_valid) * 1000.0
    ego_mean_error = mean_error_per_slot[:, 0].mean() * 1000.0
    ego_final_error = final_error_per_slot[:, 0].mean() * 1000.0
    others_mean_error = masked_mean_per_slot(mean_error_per_slot[:, 1:], entity_valid[:, 1:]) * 1000.0
    others_final_error = masked_mean_per_slot(final_error_per_slot[:, 1:], entity_valid[:, 1:]) * 1000.0
    return {
        "mean_error_mm": float(mean_error.detach().cpu()),
        "final_error_mm": float(final_error.detach().cpu()),
        "ego_mean_error_mm": float(ego_mean_error.detach().cpu()),
        "ego_final_error_mm": float(ego_final_error.detach().cpu()),
        "others_mean_error_mm": float(others_mean_error.detach().cpu()),
        "others_final_error_mm": float(others_final_error.detach().cpu()),
    }


def reduce_metric_sums(metric_sums: dict[str, float], count: int) -> dict[str, float]:
    denom = max(int(count), 1)
    return {key: value / denom for key, value in metric_sums.items()}


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    filename: str,
) -> Path:
    ckpt_dir = checkpoint_dir(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / filename
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
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
    scaler: Any | None,
    device: torch.device,
) -> tuple[int, int, float | None]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0)), ckpt.get("best_metric")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    metric_sums: dict[str, float] = defaultdict(float)
    count = 0
    max_batches = int(args.max_eval_batches)
    iterator = tqdm(loader, desc="stage1 prediction eval", unit="batch", leave=False)
    for batch_idx, batch in enumerate(iterator):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = move_to_device(batch, device)
        with autocast_context(device, amp_dtype, amp_enabled):
            denoise = call_model(model, batch)
        denoise_metrics = trajectory_metrics(denoise["x0_hat"], batch["target_x0"], batch["entity_valid"])
        for key, value in denoise_metrics.items():
            metric_sums[f"denoise_{key}"] += value
        if int(args.eval_sampling_steps) > 0:
            with autocast_context(device, amp_dtype, amp_enabled):
                sample = model.sample(batch, num_steps=int(args.eval_sampling_steps), deterministic=True)
            sample_metrics = trajectory_metrics(sample["x0_hat"], batch["target_x0"], batch["entity_valid"])
            for key, value in sample_metrics.items():
                metric_sums[f"sample_{key}"] += value
        count += 1
    return reduce_metric_sums(metric_sums, count)


@torch.no_grad()
def save_visualizations(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    writer: Any,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> None:
    from vis.stage1_prediction_vis import plot_prediction_side_by_side

    model.eval()
    out_dir = output_root(args) / "visualizations" / f"step_{int(step):08d}"
    saved = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        with autocast_context(device, amp_dtype, amp_enabled):
            pred = model.sample(batch, num_steps=int(args.eval_sampling_steps), deterministic=True)["x0_hat"]
        batch_cpu = {key: value.detach().cpu() for key, value in batch.items()}
        pred_cpu = pred.detach().cpu()
        B = pred_cpu.shape[0]
        for idx in range(B):
            if saved >= int(args.num_vis_samples):
                return
            path = out_dir / f"sample_{saved:03d}.png"
            plot_prediction_side_by_side(
                output_path=path,
                clearance_map=batch_cpu["scene_maps"][idx],
                origin_xy=tuple(batch_cpu["map_origin"][idx].tolist()),
                resolution=float(batch_cpu["map_resolution"][idx].item()),
                history_xy=batch_cpu["past_rel_pos"][idx],
                gt_xy=batch_cpu["target_x0"][idx],
                pred_xy=pred_cpu[idx],
                valid_slots=batch_cpu["entity_valid"][idx],
                goal_map=batch_cpu["goal_map"][idx],
                goal_rel_xy=batch_cpu["goal_rel_xy"][idx],
                goal_valid=batch_cpu["goal_valid"][idx],
                goal_in_crop=batch_cpu["goal_in_crop"][idx],
                title=f"step {int(step)} sample {saved}",
            )
            if writer is not None and saved < 4:
                try:
                    import matplotlib.image as mpimg

                    image = mpimg.imread(path)
                    writer.add_image(f"eval_vis/sample_{saved}", image, int(step), dataformats="HWC")
                except Exception as exc:
                    log(f"failed to add visualization to tensorboard: {exc}")
            saved += 1


def dry_run(args: argparse.Namespace) -> None:
    dataset = build_dataset(args)
    item = dataset[0]
    log(f"dataset length: {len(dataset)}")
    for key, value in item.items():
        log(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 trajectory prediction training entrypoint")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--max-others", type=int, default=3)
    parser.add_argument("--past-frames", type=int, default=30)
    parser.add_argument("--future-frames", type=int, default=72)
    parser.add_argument("--no-distance-map", action="store_true")

    parser.add_argument("--model-version", choices=sorted(MODEL_REGISTRY.keys()), default="v2")
    parser.add_argument("--model-hidden-dim", type=int, default=256)
    parser.add_argument("--model-layers", type=int, default=6)
    parser.add_argument("--model-heads", type=int, default=8)
    parser.add_argument("--model-dropout", type=float, default=0.0)
    parser.add_argument("--num-diffusion-steps", type=int, default=50)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--vel-loss-weight", type=float, default=0.5)
    parser.add_argument("--smooth-loss-weight", type=float, default=0.05)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")

    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--log-every-steps", type=int, default=10000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=10000)
    parser.add_argument("--eval-num-samples", type=int, default=10000)
    parser.add_argument("--eval-seed", type=int, default=2025)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--eval-sampling-steps", type=int, default=50)
    parser.add_argument("--num-vis-samples", type=int, default=20)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    output_root(args).mkdir(parents=True, exist_ok=True)
    text_logger = TextLogger(output_root(args) / "train.log")
    text_logger.write(f"args {json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, sort_keys=True)}")

    if args.dry_run:
        dry_run(args)
        return

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    amp_enabled = device.type == "cuda" and bool(args.amp)

    text_logger.write("building train dataset")
    train_dataset = build_dataset(args, split=args.split)
    text_logger.write(f"train samples {len(train_dataset)}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=planning_collate_fn,
        drop_last=True,
        **dataloader_worker_kwargs(args),
    )

    text_logger.write("building fixed eval dataset")
    eval_sample_file = build_common_eval_sample_file(args, split=args.eval_split)
    text_logger.write(f"eval sample file {eval_sample_file}")
    eval_dataset = build_dataset_from_sample_paths(args, split=args.eval_split, sample_paths_file=eval_sample_file)
    text_logger.write(f"eval samples {len(eval_dataset)}")
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=planning_collate_fn,
        drop_last=False,
        **dataloader_worker_kwargs(args),
    )

    sample_batch = next(iter(train_loader))
    model = build_model(args, sample_batch).to(device)
    text_logger.write(
        "model "
        + json.dumps(
            {
                "version": args.model_version,
                "hidden_dim": args.model_hidden_dim,
                "layers": args.model_layers,
                "heads": args.model_heads,
                "dropout": args.model_dropout,
                "num_diffusion_steps": args.num_diffusion_steps,
                "slots": int(sample_batch["x0"].shape[1]),
                "past_frames": int(sample_batch["past_rel_pos"].shape[2]),
                "future_frames": int(sample_batch["x0"].shape[2]),
                "scene_channels": int(sample_batch["scene_maps"].shape[1]),
            },
            sort_keys=True,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    writer = build_summary_writer(args)

    start_epoch = 0
    global_step = 0
    best_metric: float | None = None
    if args.resume is not None:
        start_epoch, global_step, best_metric = load_checkpoint(args.resume, model, optimizer, scaler, device)
        text_logger.write(f"resumed from {args.resume}: epoch={start_epoch} step={global_step} best={best_metric}")

    precision = f"amp_{args.amp_dtype}" if amp_enabled else "fp32"
    text_logger.write(f"training batch={args.batch_size} workers={args.num_workers} precision={precision}")
    model.train()
    for epoch in range(start_epoch, int(args.epochs)):
        text_logger.write(f"epoch start epoch={epoch} global_step={global_step}")
        progress = tqdm(train_loader, desc=f"stage1 prediction epoch {epoch}", unit="batch")
        for batch in progress:
            global_step += 1
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_dtype, amp_enabled):
                loss, metrics = prediction_loss(
                    model,
                    batch,
                    vel_loss_weight=float(args.vel_loss_weight),
                    smooth_loss_weight=float(args.smooth_loss_weight),
                )
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            progress.set_postfix(loss=f"{metrics['loss']:.4f}", pos_rmse_mm=f"{metrics['pos_rmse_mm']:.1f}")

            if global_step % int(args.log_every_steps) == 0:
                write_scalars(writer, "train", metrics, global_step)
                text_logger.write(f"train step={global_step} {json.dumps(metrics, sort_keys=True)}")
                eval_metrics = evaluate(model, eval_loader, device, args, amp_dtype, amp_enabled)
                write_scalars(writer, "eval", eval_metrics, global_step)
                text_logger.write(f"eval step={global_step} {json.dumps(eval_metrics, sort_keys=True)}")
                save_visualizations(model, eval_loader, device, args, global_step, writer, amp_dtype, amp_enabled)
                metric = eval_metrics.get("sample_mean_error_mm", eval_metrics.get("denoise_mean_error_mm"))
                if metric is not None and (best_metric is None or float(metric) < float(best_metric)):
                    best_metric = float(metric)
                    path = save_checkpoint(model, optimizer, scaler, args, epoch, global_step, best_metric, "best.pt")
                    text_logger.write(f"saved best checkpoint {path} metric={best_metric:.6f}")
                model.train()

            if global_step % int(args.checkpoint_every_steps) == 0:
                path = save_checkpoint(model, optimizer, scaler, args, epoch, global_step, best_metric, "latest.pt")
                text_logger.write(f"saved latest checkpoint {path}")

        path = save_checkpoint(model, optimizer, scaler, args, epoch + 1, global_step, best_metric, "latest.pt")
        text_logger.write(f"saved latest checkpoint {path}")
        text_logger.write(f"epoch end epoch={epoch} global_step={global_step}")

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
