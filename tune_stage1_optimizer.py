from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets import LingoPlanning, TrumansPlanning, planning_collate_fn
from models.stage1_planner import (
    Stage1OptimizerV2Config,
    build_stage1_static_fields_v2,
    optimize_stage1_trajectory_batch_v2,
)
from models.stage1_predictor import Stage1Predictor


PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSED_ROOT = PROJECT_ROOT / "data" / "preprocessed"
DATASET_CLASSES = {
    "trumans": TrumansPlanning,
    "lingo": LingoPlanning,
}


def log(message: str) -> None:
    print(message, flush=True)


def load_checkpoint_args(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    args = ckpt.get("args") or {}
    return dict(args)


def build_model_from_checkpoint_args(sample_batch: dict[str, torch.Tensor], ckpt_args: dict[str, Any]) -> Stage1Predictor:
    return Stage1Predictor(
        slots=int(sample_batch["x0"].shape[1]),
        past_frames=int(sample_batch["past_rel_pos"].shape[2]),
        future_frames=int(sample_batch["x0"].shape[2]),
        scene_channels=int(sample_batch["scene_maps"].shape[1]),
        hidden_dim=int(ckpt_args.get("model_hidden_dim", 256)),
        num_layers=int(ckpt_args.get("model_layers", 6)),
        num_heads=int(ckpt_args.get("model_heads", 8)),
        dropout=float(ckpt_args.get("model_dropout", 0.0)),
        num_timesteps=int(ckpt_args.get("num_diffusion_steps", 50)),
    )


def load_model(checkpoint: Path, sample_batch: dict[str, torch.Tensor], device: torch.device) -> Stage1Predictor:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model_from_checkpoint_args(sample_batch, dict(ckpt.get("args") or {}))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def select_indices(num_items: int, num_samples: int, seed: int) -> list[int]:
    if int(num_samples) <= 0:
        return list(range(num_items))
    rng = random.Random(int(seed))
    indices = list(range(num_items))
    rng.shuffle(indices)
    return sorted(indices[: min(int(num_samples), num_items)])


def speed_stats_path(dataset: str) -> Path:
    return PREPROCESSED_ROOT / dataset / "stage1_velocity_stats_v2.json"


def load_or_compute_speed_stats(
    dataset_name: str,
    dataset: torch.utils.data.Dataset,
    max_samples: int,
    dt: float,
) -> dict[str, Any]:
    path = speed_stats_path(dataset_name)
    if path.exists():
        existing = json.loads(path.read_text())
        requested = len(dataset) if int(max_samples) <= 0 else min(int(max_samples), len(dataset))
        if int(existing.get("num_sampled_windows", 0)) >= int(requested):
            return existing
        log(
            "existing velocity stats use fewer samples "
            f"({existing.get('num_sampled_windows')}) than requested ({requested}); recomputing"
        )
    sample_count = len(dataset) if int(max_samples) <= 0 else min(int(max_samples), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num=sample_count, dtype=np.int64)
    speeds: list[np.ndarray] = []
    accs: list[np.ndarray] = []
    for idx in tqdm(indices, desc="compute velocity stats", unit="sample"):
        item = dataset[int(idx)]
        future = item["future_rel_pos"][0].numpy()
        past = item["past_rel_pos"][0].numpy()
        path_xy = np.concatenate([past[-1:][None, 0] if past.ndim == 3 else past[-1:], future], axis=0)
        if path_xy.ndim != 2 or path_xy.shape[0] < 2:
            continue
        vel = (path_xy[1:] - path_xy[:-1]) / float(dt)
        speed = np.linalg.norm(vel, axis=-1)
        speeds.append(speed.astype(np.float32))
        if vel.shape[0] > 1:
            acc = np.linalg.norm((vel[1:] - vel[:-1]) / float(dt), axis=-1)
            accs.append(acc.astype(np.float32))
    speed_values = np.concatenate(speeds) if speeds else np.asarray([1.4], dtype=np.float32)
    acc_values = np.concatenate(accs) if accs else np.asarray([0.0], dtype=np.float32)
    stats = {
        "dataset": dataset_name,
        "num_sampled_windows": int(sample_count),
        "dt": float(dt),
        "move_speed_mean": float(np.mean(speed_values)),
        "move_speed_std": float(np.std(speed_values)),
        "move_speed_p95": float(np.percentile(speed_values, 95)),
        "move_speed_p99": float(np.percentile(speed_values, 99)),
        "move_acc_p95": float(np.percentile(acc_values, 95)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    return stats


def build_dataset(args: argparse.Namespace) -> torch.utils.data.Dataset:
    dataset_cls = DATASET_CLASSES[args.dataset]
    return dataset_cls(
        root=PREPROCESSED_ROOT,
        split=args.split,
        scene_id=args.scene_id,
        max_others=3,
        include_distance_map=True,
        model_past_frames=30,
        model_future_frames=72,
    )


def eval_sample_file(args: argparse.Namespace) -> Path:
    scene = args.scene_id or "all"
    return (
        PREPROCESSED_ROOT
        / args.dataset
        / "stage1_eval_samples_v2"
        / f"{args.split}_p30_f72_scene-{scene}_n{int(args.eval_sample_count)}_seed{int(args.eval_seed)}.json"
    )


def load_eval_sample_paths(args: argparse.Namespace) -> list[str] | None:
    path = eval_sample_file(args)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    sample_paths = payload.get("sample_paths")
    if not isinstance(sample_paths, list):
        raise ValueError(f"eval sample file missing sample_paths list: {path}")
    return [str(item) for item in sample_paths]


def build_common_eval_sample_file(args: argparse.Namespace) -> Path:
    path = eval_sample_file(args)
    if path.exists():
        return path
    dataset = build_dataset(args)
    sample_paths = list(getattr(dataset, "sample_paths"))
    rng = random.Random(int(args.eval_seed))
    rng.shuffle(sample_paths)
    selected = sorted(sample_paths[: min(int(args.eval_sample_count), len(sample_paths))])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "scene_id": args.scene_id,
                "past_frames": 30,
                "future_frames": 72,
                "seed": int(args.eval_seed),
                "requested_num_samples": int(args.eval_sample_count),
                "num_samples": len(selected),
                "sample_paths": selected,
            },
            indent=2,
        )
    )
    return path


def build_dataset_for_split(args: argparse.Namespace, split: str) -> torch.utils.data.Dataset:
    dataset_cls = DATASET_CLASSES[args.dataset]
    return dataset_cls(
        root=PREPROCESSED_ROOT,
        split=split,
        scene_id=args.scene_id,
        max_others=3,
        include_distance_map=True,
        model_past_frames=30,
        model_future_frames=72,
    )


def build_eval_dataset(args: argparse.Namespace) -> tuple[torch.utils.data.Dataset, list[int]]:
    if int(args.num_samples) <= 0:
        build_common_eval_sample_file(args)
    sample_paths = load_eval_sample_paths(args)
    if sample_paths is None:
        dataset = build_dataset(args)
        indices = select_indices(len(dataset), int(args.num_samples), int(args.seed))
        return dataset, indices
    dataset_cls = DATASET_CLASSES[args.dataset]
    dataset = dataset_cls(
        root=PREPROCESSED_ROOT,
        split=args.split,
        scene_id=args.scene_id,
        max_others=3,
        include_distance_map=True,
        sample_paths_file=eval_sample_file(args),
        model_past_frames=30,
        model_future_frames=72,
    )
    if int(args.num_samples) > 0 and int(args.num_samples) < len(dataset):
        indices = select_indices(len(dataset), int(args.num_samples), int(args.seed))
    else:
        indices = list(range(len(dataset)))
    return dataset, indices


def dataloader_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.num_workers) <= 0:
        return {}
    return {
        "persistent_workers": True,
        "prefetch_factor": int(args.prefetch_factor),
    }


def prediction_cache_matches(cache: dict[str, Any], args: argparse.Namespace, indices: list[int]) -> bool:
    meta = cache.get("meta", {})
    return (
        meta.get("dataset") == args.dataset
        and meta.get("split") == args.split
        and meta.get("checkpoint") == str(args.checkpoint)
        and int(meta.get("num_samples", -1)) == len(indices)
        and list(meta.get("indices", [])) == [int(idx) for idx in indices]
        and int(meta.get("eval_sampling_steps", -1)) == int(args.eval_sampling_steps)
    )


@torch.no_grad()
def build_prediction_cache(
    args: argparse.Namespace,
    dataset: torch.utils.data.Dataset,
    indices: list[int],
    device: torch.device,
    cache_path: Path,
) -> dict[str, Any]:
    if cache_path.exists() and not args.rebuild_prediction_cache:
        cache = torch.load(cache_path, map_location="cpu")
        if prediction_cache_matches(cache, args, indices):
            log(f"loaded prediction cache: {cache_path}")
            return cache
        log(f"prediction cache metadata mismatch, rebuilding: {cache_path}")

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=planning_collate_fn,
        drop_last=False,
        **dataloader_kwargs(args),
    )
    sample_batch = next(iter(loader))
    model = load_model(args.checkpoint, sample_batch, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))

    tensors: dict[str, list[torch.Tensor]] = {
        "pred_x0": [],
        "target_x0": [],
        "past_rel_pos": [],
        "past_vel": [],
        "scene_maps": [],
        "goal_map": [],
        "goal_rel_xy": [],
        "entity_valid": [],
    }
    for batch in tqdm(loader, desc="prediction sampling", unit="batch"):
        batch_device = move_to_device(batch, device)
        sample = model.sample(
            batch_device,
            num_steps=int(args.eval_sampling_steps),
            deterministic=True,
            generator=generator,
        )
        tensors["pred_x0"].append(sample["x0_hat"].detach().cpu())
        for key in ("target_x0", "past_rel_pos", "past_vel", "scene_maps", "goal_map", "goal_rel_xy", "entity_valid"):
            tensors[key].append(batch[key].detach().cpu())

    cache = {
        "meta": {
            "dataset": args.dataset,
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "num_samples": len(indices),
            "indices": [int(idx) for idx in indices],
            "eval_sampling_steps": int(args.eval_sampling_steps),
        },
        "pred_x0": torch.cat(tensors["pred_x0"], dim=0),
        "target_x0": torch.cat(tensors["target_x0"], dim=0),
        "past_rel_pos": torch.cat(tensors["past_rel_pos"], dim=0),
        "past_vel": torch.cat(tensors["past_vel"], dim=0),
        "scene_maps": torch.cat(tensors["scene_maps"], dim=0),
        "goal_map": torch.cat(tensors["goal_map"], dim=0),
        "goal_rel_xy": torch.cat(tensors["goal_rel_xy"], dim=0),
        "entity_valid": torch.cat(tensors["entity_valid"], dim=0),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    log(f"wrote prediction cache: {cache_path}")
    return cache


def body_goal_distance_field(
    goal_rel_xy: np.ndarray,
    height: int,
    width: int,
    origin_xy: tuple[float, float],
    resolution: float,
) -> np.ndarray:
    rows, cols = np.indices((int(height), int(width)), dtype=np.float32)
    x = float(origin_xy[0]) + (cols + 0.5) * float(resolution)
    z = float(origin_xy[1]) + ((int(height) - 1 - rows) + 0.5) * float(resolution)
    goal = np.asarray(goal_rel_xy, dtype=np.float32)
    return np.sqrt((x - float(goal[0])) ** 2 + (z - float(goal[1])) ** 2).astype(np.float32)


def build_planner_fields(
    cache: dict[str, Any],
    config: Stage1OptimizerV2Config,
) -> dict[str, torch.Tensor]:
    scene_maps = cache["scene_maps"]
    distance_fields = scene_maps[:, 1].float()
    goal_maps = cache["goal_map"][:, 0].float()
    goal_rel_xy = cache["goal_rel_xy"].float()
    static_fields: list[torch.Tensor] = []
    goal_distance_fields: list[torch.Tensor] = []
    corridor_ratios: list[float] = []
    start_unsafe_count = 0
    goal_unsafe_count = 0
    for idx in tqdm(range(distance_fields.shape[0]), desc="build static fields", unit="sample", leave=False):
        fields = build_stage1_static_fields_v2(
            distance_fields[idx].numpy(),
            goal_maps[idx].numpy(),
            config,
            origin_xy=(-4.0, -3.0),
            resolution=0.1,
            start_xy=(0.0, 0.0),
        )
        static_fields.append(torch.from_numpy(fields["final_static"]))
        goal_distance_fields.append(
            torch.from_numpy(
                body_goal_distance_field(
                    goal_rel_xy[idx].numpy(),
                    height=int(distance_fields.shape[1]),
                    width=int(distance_fields.shape[2]),
                    origin_xy=(-4.0, -3.0),
                    resolution=0.1,
                )
            )
        )
        corridor_ratios.append(float(np.asarray(fields["corridor"], dtype=np.float32).mean()))
        start_unsafe_count += int(bool(fields["start_unsafe"]))
        goal_unsafe_count += int(bool(fields["goal_unsafe"]))
    return {
        "distance_fields": distance_fields,
        "static_fields": torch.stack(static_fields, dim=0).float(),
        "goal_distance_fields": torch.stack(goal_distance_fields, dim=0).float(),
        "corridor_ratio": torch.tensor(corridor_ratios, dtype=torch.float32),
        "start_unsafe_count": torch.tensor(start_unsafe_count),
        "goal_unsafe_count": torch.tensor(goal_unsafe_count),
    }


def grid_values(args: argparse.Namespace) -> dict[str, list[float]]:
    if args.grid == "full":
        return {
            "w_prior": [0.5, 1.0, 2.0, 4.0, 8.0],
            "w_goal": [0.1, 0.3, 1.0, 3.0, 10.0],
            "w_static": [1.0, 3.0, 10.0, 30.0, 100.0],
            "w_dyn": [1.0, 3.0, 10.0, 30.0, 100.0],
            "smooth_scale": [0.25, 0.5, 1.0, 2.0, 4.0],
            "terminal_weight_ratio": [2.0],
            "goal_progress_weight": [0.0],
            "static_topk_weight": [0.0],
            "static_topk_frac": [0.2],
        }
    if args.grid == "loop_a":
        return {
            "w_prior": [0.0, 0.05, 0.1],
            "w_goal": [3.0, 10.0, 30.0],
            "w_static": [3.0, 10.0, 30.0],
            "w_dyn": [10.0],
            "smooth_scale": [0.5],
            "terminal_weight_ratio": [5.0],
            "goal_progress_weight": [1.0],
            "static_topk_weight": [3.0],
            "static_topk_frac": [0.2],
        }
    return {
        "w_prior": [1.0, 2.0, 4.0],
        "w_goal": [0.8, 1.0, 1.2],
        "w_static": [3.0, 10.0, 30.0],
        "w_dyn": [3.0, 10.0, 30.0],
        "smooth_scale": [0.5, 1.0, 2.0],
        "terminal_weight_ratio": [2.0],
        "goal_progress_weight": [0.0],
        "static_topk_weight": [0.0],
        "static_topk_frac": [0.2],
    }


def iter_grid(values: dict[str, list[float]]) -> list[dict[str, float]]:
    configs = []
    for w_prior in values["w_prior"]:
        for w_goal in values["w_goal"]:
            for w_static in values["w_static"]:
                for w_dyn in values["w_dyn"]:
                    for smooth_scale in values["smooth_scale"]:
                        for terminal_weight_ratio in values["terminal_weight_ratio"]:
                            for goal_progress_weight in values["goal_progress_weight"]:
                                for static_topk_weight in values["static_topk_weight"]:
                                    for static_topk_frac in values["static_topk_frac"]:
                                        configs.append(
                                            {
                                                "w_prior": float(w_prior),
                                                "w_goal": float(w_goal),
                                                "w_static": float(w_static),
                                                "w_dyn": float(w_dyn),
                                                "smooth_scale": float(smooth_scale),
                                                "terminal_weight_ratio": float(terminal_weight_ratio),
                                                "goal_progress_weight": float(goal_progress_weight),
                                                "static_topk_weight": float(static_topk_weight),
                                                "static_topk_frac": float(static_topk_frac),
                                            }
                                        )
    return configs


def score_metrics(metrics: dict[str, float]) -> float:
    return (
        5.0 * metrics["goal_final_dist_mm"] / 1000.0
        + 2.0 * metrics["goal_mean_dist_mm"] / 1000.0
        + 10.0 * metrics["static_collision_ratio"]
        + 10.0 * metrics["dynamic_collision_ratio"]
        + 0.1 * metrics.get("debug_jerk", 0.0)
    )


def run_one_config(
    cache: dict[str, Any],
    field_cache: dict[str, torch.Tensor],
    config: Stage1OptimizerV2Config,
    speed_bound: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    pred_x0 = cache["pred_x0"]
    target_x0 = cache["target_x0"]
    entity_valid = cache["entity_valid"].bool()
    past_vel = cache["past_vel"]
    totals: dict[str, float] = {}
    count = 0
    for start in range(0, pred_x0.shape[0], int(args.optimizer_batch_size)):
        end = min(start + int(args.optimizer_batch_size), pred_x0.shape[0])
        batch_slice = slice(start, end)
        pred = pred_x0[batch_slice].to(device)
        target = target_x0[batch_slice].to(device)
        valid = entity_valid[batch_slice].to(device)
        current_vel = (past_vel[batch_slice, 0, -1].to(device) / float(config.dt)).float()
        fields = {key: value[batch_slice].to(device) for key, value in field_cache.items() if isinstance(value, torch.Tensor) and value.ndim >= 1}
        _, metrics = optimize_stage1_trajectory_batch_v2(
            pred_ego=pred[:, 0].float(),
            pred_others=pred[:, 1:].float(),
            others_valid=valid[:, 1:],
            distance_fields=fields["distance_fields"].float(),
            static_fields=fields["static_fields"].float(),
            goal_distance_fields=fields["goal_distance_fields"].float(),
            gt_ego=target[:, 0].float(),
            gt_others=target[:, 1:].float(),
            current_vel_xy=current_vel,
            speed_bound=float(speed_bound),
            config=config,
            origin_xy=(-4.0, -3.0),
            resolution=0.1,
        )
        batch_n = end - start
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_n
        count += batch_n
    out = {key: value / max(count, 1) for key, value in totals.items()}
    out["score"] = score_metrics(out)
    out["num_samples"] = float(count)
    return out


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Stage-1 v2 prediction-guided optimizer hyperparameters.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], default="trumans")
    parser.add_argument("--split", default="test")
    parser.add_argument("--velocity-stat-split", default="train")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/stage1_v2_trumans_001/checkpoints/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--eval-sample-count", type=int, default=10000)
    parser.add_argument("--eval-seed", type=int, default=2025)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--optimizer-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--eval-sampling-steps", type=int, default=50)
    parser.add_argument("--grid", choices=["coarse", "full", "loop_a"], default="coarse")
    parser.add_argument("--rebuild-prediction-cache", action="store_true")
    parser.add_argument("--velocity-stat-max-samples", type=int, default=50000)
    parser.add_argument("--max-configs", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path("outputs") / f"stage1_optimizer_tuning_{args.dataset}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log(f"output dir: {args.output_dir}")
    log(f"checkpoint: {args.checkpoint}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset, indices = build_eval_dataset(args)
    (args.output_dir / "selected_indices.json").write_text(json.dumps({"indices": indices}, indent=2))
    stats_dataset = build_dataset_for_split(args, args.velocity_stat_split)
    speed_stats = load_or_compute_speed_stats(args.dataset, stats_dataset, int(args.velocity_stat_max_samples), dt=1.0 / 30.0)
    speed_bound = float(speed_stats["move_speed_p95"])
    log(f"speed bound p95: {speed_bound:.4f} m/s")

    cache_path = args.output_dir / "prediction_cache.pt"
    cache = build_prediction_cache(args, dataset, indices, device, cache_path)
    base_config = Stage1OptimizerV2Config()
    field_cache = build_planner_fields(cache, base_config)
    log(
        "static field summary: "
        f"start_unsafe={int(field_cache['start_unsafe_count'])} "
        f"goal_unsafe={int(field_cache['goal_unsafe_count'])} "
        f"corridor_ratio={float(field_cache['corridor_ratio'].mean()):.4f}"
    )

    configs = iter_grid(grid_values(args))
    if int(args.max_configs) > 0:
        configs = configs[: int(args.max_configs)]
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    progress = tqdm(configs, desc="optimizer grid", unit="config")
    for config_idx, values in enumerate(progress):
        config = Stage1OptimizerV2Config(**values)
        metrics = run_one_config(cache, field_cache, config, speed_bound, args, device)
        row = {"config_idx": config_idx, **values, **metrics}
        rows.append(row)
        if best is None or float(row["score"]) < float(best["score"]):
            best = dict(row)
            progress.set_postfix(score=f"{row['score']:.4f}", ade=f"{row['gt_ade_30_mm']:.1f}")
            best_payload = {
                "dataset": args.dataset,
                "split": args.split,
                "checkpoint": str(args.checkpoint),
                "speed_stats": speed_stats,
                "score": float(row["score"]),
                "metrics": {
                    key: float(value)
                    for key, value in row.items()
                    if key
                    not in {
                        "config_idx",
                        "w_prior",
                        "w_goal",
                        "w_static",
                        "w_dyn",
                        "smooth_scale",
                        "terminal_weight_ratio",
                        "goal_progress_weight",
                        "static_topk_weight",
                        "static_topk_frac",
                    }
                },
                "config": {
                    **asdict(config),
                    "speed_bound_mps": speed_bound,
                    "w_init": float(config.smooth_scale),
                    "w_acc": float(config.smooth_scale),
                    "w_jerk": 0.3 * float(config.smooth_scale),
                },
            }
            (args.output_dir / "best_config.json").write_text(json.dumps(best_payload, indent=2, sort_keys=True))
        if (config_idx + 1) % 10 == 0:
            write_results_csv(args.output_dir / "results.csv", rows)

    write_results_csv(args.output_dir / "results.csv", rows)
    if best is not None:
        log(f"best score={float(best['score']):.6f}")
        log(
            json.dumps(
                {
                    key: best[key]
                    for key in (
                        "w_prior",
                        "w_goal",
                        "w_static",
                        "w_dyn",
                        "smooth_scale",
                        "terminal_weight_ratio",
                        "goal_progress_weight",
                        "static_topk_weight",
                        "static_topk_frac",
                    )
                },
                sort_keys=True,
            )
        )
        log(f"best config: {args.output_dir / 'best_config.json'}")
    log(f"results: {args.output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
