from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worldstreamers_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/worldstreamers_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from models.stage1_planner import (
    Stage1OptimizerV2Config,
    build_stage1_static_fields_v2,
    optimize_stage1_trajectory_batch_v2,
)


LOCAL_X_RANGE = (-4.0, 4.0)
LOCAL_Z_RANGE = (-3.0, 3.0)
LOCAL_RESOLUTION = 0.1
ORIGIN_XY = (-4.0, -3.0)
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
LABELS = ["ego", "other1", "other2", "other3"]


def as_numpy(value: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def load_config(tuning_dir: Path) -> tuple[Stage1OptimizerV2Config, float, dict[str, Any]]:
    payload = json.loads((tuning_dir / "best_config.json").read_text())
    config_data = dict(payload["config"])
    speed_bound = float(config_data.pop("speed_bound_mps"))
    config_data.pop("w_init", None)
    config_data.pop("w_acc", None)
    config_data.pop("w_jerk", None)
    allowed = Stage1OptimizerV2Config.__dataclass_fields__.keys()
    config = Stage1OptimizerV2Config(**{key: value for key, value in config_data.items() if key in allowed})
    return config, speed_bound, payload


def selected_index_labels(tuning_dir: Path, num_items: int) -> list[int]:
    path = tuning_dir / "selected_indices.json"
    if not path.exists():
        return list(range(num_items))
    payload = json.loads(path.read_text())
    indices = payload.get("indices")
    if not isinstance(indices, list):
        return list(range(num_items))
    return [int(item) for item in indices[:num_items]]


def compute_metrics(
    opt_ego: np.ndarray,
    pred_ego: np.ndarray,
    gt_ego: np.ndarray,
    pred_others: np.ndarray,
    gt_others: np.ndarray,
    valid_others: np.ndarray,
    distance_field: np.ndarray,
    goal_distance: np.ndarray,
    config: Stage1OptimizerV2Config,
) -> dict[str, float]:
    gt_dist = np.linalg.norm(opt_ego - gt_ego, axis=-1)
    pred_dist = np.linalg.norm(pred_ego - gt_ego, axis=-1)
    static_values = sample_field_numpy(distance_field, opt_ego)
    goal_values = sample_field_numpy(goal_distance, opt_ego)
    if pred_others.size and valid_others.any():
        dyn_dist = np.linalg.norm(opt_ego[None] - gt_others, axis=-1)
        dyn_dist = dyn_dist[valid_others.astype(bool)]
        min_dyn = dyn_dist.min(axis=0) if dyn_dist.size else np.full((opt_ego.shape[0],), np.inf)
    else:
        min_dyn = np.full((opt_ego.shape[0],), np.inf, dtype=np.float32)
    return {
        "opt_ade_mm": float(gt_dist.mean() * 1000.0),
        "opt_fde_mm": float(gt_dist[-1] * 1000.0),
        "pred_ade_mm": float(pred_dist.mean() * 1000.0),
        "pred_fde_mm": float(pred_dist[-1] * 1000.0),
        "goal_final_mm": float(goal_values[-1] * 1000.0),
        "static_ratio": float((static_values < float(config.static_margin)).mean()),
        "dyn_ratio": float((min_dyn < float(config.dyn_margin)).mean()) if np.isfinite(min_dyn).any() else 0.0,
    }


def sample_field_numpy(field: np.ndarray, xy: np.ndarray) -> np.ndarray:
    H, W = field.shape
    cols = np.floor((xy[:, 0] - LOCAL_X_RANGE[0]) / LOCAL_RESOLUTION).astype(np.int64)
    rows_from_bottom = np.floor((xy[:, 1] - LOCAL_Z_RANGE[0]) / LOCAL_RESOLUTION).astype(np.int64)
    rows = H - 1 - rows_from_bottom
    rows = np.clip(rows, 0, H - 1)
    cols = np.clip(cols, 0, W - 1)
    return field[rows, cols]


def draw_scene(ax: Any, scene_maps: np.ndarray, goal_map: np.ndarray, goal_rel_xy: np.ndarray, title: str) -> None:
    if scene_maps.shape[0] >= 3:
        background = 1.0 - np.clip(scene_maps[2], 0.0, 1.0)
    else:
        background = scene_maps[0]
    extent = [LOCAL_X_RANGE[0], LOCAL_X_RANGE[1], LOCAL_Z_RANGE[0], LOCAL_Z_RANGE[1]]
    ax.imshow(np.flipud(background), cmap="gray", origin="lower", extent=extent, vmin=0.0, vmax=1.0)
    if scene_maps.shape[0] >= 2:
        try:
            ax.contour(
                np.linspace(extent[0], extent[1], scene_maps.shape[-1]),
                np.linspace(extent[2], extent[3], scene_maps.shape[-2]),
                np.flipud(scene_maps[1]),
                levels=[0.25, 0.5, 0.75],
                colors="#777777",
                linewidths=0.45,
                alpha=0.7,
            )
        except Exception:
            pass
    if float(goal_map.max()) > 1e-6:
        masked = np.ma.masked_less_equal(np.flipud(goal_map), 1e-4)
        ax.imshow(masked, cmap="summer", origin="lower", extent=extent, alpha=0.55, vmin=0.0, vmax=max(1.0, float(goal_map.max())))
        ax.contour(goal_map, levels=[0.2, 0.5], origin="upper", extent=extent, colors=["#ffd400", "white"], linewidths=[1.0, 0.75])
    ax.scatter([goal_rel_xy[0]], [goal_rel_xy[1]], marker="*", s=150, c="#ffd400", edgecolors="black", linewidths=0.8, zorder=8)
    ax.scatter([0.0], [0.0], s=42, c="red", edgecolors="black", linewidths=0.6, zorder=8)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(LOCAL_X_RANGE)
    ax.set_ylim(LOCAL_Z_RANGE)
    ax.set_xlabel("x rel (m)")
    ax.set_ylabel("z rel (m)")


def plot_paths(ax: Any, history: np.ndarray, future: np.ndarray, valid: np.ndarray, future_label: str, style: str = "-") -> None:
    for slot in range(min(future.shape[0], len(COLORS))):
        if slot >= valid.shape[0] or not bool(valid[slot]):
            continue
        color = COLORS[slot]
        label = LABELS[slot]
        if history.shape[1] > 0:
            ax.plot(history[slot, :, 0], history[slot, :, 1], color=color, linewidth=3.0 if slot == 0 else 1.6, alpha=0.95, marker="o", markersize=2.2, label=f"{label} hist")
        ax.plot(future[slot, :, 0], future[slot, :, 1], style, color=color, linewidth=1.7 if slot == 0 else 1.1, alpha=0.9, label=f"{label} {future_label}")


def plot_one(
    out_path: Path,
    scene_maps: np.ndarray,
    goal_map: np.ndarray,
    goal_rel_xy: np.ndarray,
    history: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    opt_ego: np.ndarray,
    valid: np.ndarray,
    fields: dict[str, Any],
    metrics: dict[str, float],
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    draw_scene(axes[0, 0], scene_maps, goal_map, goal_rel_xy, f"{title} / GT")
    plot_paths(axes[0, 0], history, gt, valid, "gt", "-")
    draw_scene(axes[0, 1], scene_maps, goal_map, goal_rel_xy, "Prediction")
    plot_paths(axes[0, 1], history, pred, valid, "pred", "--")
    draw_scene(axes[0, 2], scene_maps, goal_map, goal_rel_xy, "Optimized ego")
    opt_future = pred.copy()
    opt_future[0, : opt_ego.shape[0]] = opt_ego
    plot_paths(axes[0, 2], history, opt_future, valid, "opt", "--")

    extent = [LOCAL_X_RANGE[0], LOCAL_X_RANGE[1], LOCAL_Z_RANGE[0], LOCAL_Z_RANGE[1]]
    debug_panels = [
        (fields["raw_static"], "raw static barrier", "inferno", 0.0, 1.0),
        (fields["corridor"].astype(np.float32), "corridor", "Blues", 0.0, 1.0),
        (fields["final_static"], "final static barrier", "inferno", 0.0, 1.0),
    ]
    for ax, (image, panel_title, cmap, vmin, vmax) in zip(axes[1], debug_panels):
        im = ax.imshow(image, origin="upper", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.contour(goal_map, levels=[0.2], origin="upper", extent=extent, colors=["#ffd400"], linewidths=[1.0])
        ax.plot(opt_ego[:, 0], opt_ego[:, 1], "--", color="red", linewidth=1.7)
        if fields["path"]:
            xs = [LOCAL_X_RANGE[0] + (col + 0.5) * LOCAL_RESOLUTION for row, col in fields["path"]]
            ys = [LOCAL_Z_RANGE[1] - (row + 0.5) * LOCAL_RESOLUTION for row, col in fields["path"]]
            ax.plot(xs, ys, color="cyan", linewidth=1.4)
        ax.scatter([0.0], [0.0], s=36, c="red", edgecolors="black", linewidths=0.5)
        ax.scatter([goal_rel_xy[0]], [goal_rel_xy[1]], marker="*", s=120, c="#ffd400", edgecolors="black", linewidths=0.7)
        ax.set_title(panel_title)
        ax.set_xlim(LOCAL_X_RANGE)
        ax.set_ylim(LOCAL_Z_RANGE)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.ravel()[:3]:
        ax.legend(loc="upper right", fontsize=7)
    metric_text = (
        f"opt ADE/FDE {metrics['opt_ade_mm']:.1f}/{metrics['opt_fde_mm']:.1f}mm | "
        f"pred ADE/FDE {metrics['pred_ade_mm']:.1f}/{metrics['pred_fde_mm']:.1f}mm | "
        f"goal {metrics['goal_final_mm']:.1f}mm | "
        f"static {metrics['static_ratio']:.2f} dyn {metrics['dyn_ratio']:.2f} | "
        f"start_unsafe={fields['start_unsafe']} goal_unsafe={fields['goal_unsafe']}"
    )
    fig.suptitle(metric_text, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize final Stage-1 optimizer outputs from tuning cache.")
    parser.add_argument("--tuning-dir", type=Path, default=Path("outputs/stage1_optimizer_tuning_trumans"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-vis", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optimizer-batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tuning_dir = args.tuning_dir
    out_dir = args.output_dir or (tuning_dir / "visualizations")
    config, speed_bound, best_payload = load_config(tuning_dir)
    cache = torch.load(tuning_dir / "prediction_cache.pt", map_location="cpu")
    if "past_rel_pos" in cache:
        past_rel_pos = cache["past_rel_pos"]
    else:
        pred_shape = cache["pred_x0"].shape
        past_rel_pos = torch.zeros((pred_shape[0], pred_shape[1], 0, 2), dtype=cache["pred_x0"].dtype)
    index_labels = selected_index_labels(tuning_dir, int(cache["pred_x0"].shape[0]))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    num_total = int(cache["pred_x0"].shape[0])
    start = max(0, int(args.start_index))
    end = num_total if int(args.num_vis) <= 0 else min(num_total, start + int(args.num_vis))
    selected = list(range(start, end))

    for offset in tqdm(range(0, len(selected), int(args.optimizer_batch_size)), desc="visualize optimizer", unit="batch"):
        batch_ids = selected[offset : offset + int(args.optimizer_batch_size)]
        pred = cache["pred_x0"][batch_ids].to(device).float()
        target = cache["target_x0"][batch_ids].to(device).float()
        valid = cache["entity_valid"][batch_ids].to(device).bool()
        current_vel = (cache["past_vel"][batch_ids, 0, -1].to(device).float() / float(config.dt))
        scene_maps = cache["scene_maps"][batch_ids].float()
        goal_maps = cache["goal_map"][batch_ids, 0].float()
        distance_fields = scene_maps[:, 1]
        static_fields = []
        goal_distance_fields = []
        field_debug = []
        for local_idx in range(len(batch_ids)):
            fields = build_stage1_static_fields_v2(
                distance_fields[local_idx].numpy(),
                goal_maps[local_idx].numpy(),
                config,
                origin_xy=ORIGIN_XY,
                resolution=LOCAL_RESOLUTION,
                start_xy=(0.0, 0.0),
            )
            static_fields.append(torch.from_numpy(fields["final_static"]))
            goal_distance_fields.append(torch.from_numpy(fields["goal_distance"]))
            field_debug.append(fields)
        opt, _ = optimize_stage1_trajectory_batch_v2(
            pred_ego=pred[:, 0],
            pred_others=pred[:, 1:],
            others_valid=valid[:, 1:],
            distance_fields=distance_fields.to(device),
            static_fields=torch.stack(static_fields).to(device),
            goal_distance_fields=torch.stack(goal_distance_fields).to(device),
            gt_ego=target[:, 0],
            gt_others=target[:, 1:],
            current_vel_xy=current_vel,
            speed_bound=float(speed_bound),
            config=config,
            origin_xy=ORIGIN_XY,
            resolution=LOCAL_RESOLUTION,
        )
        opt_np = opt.cpu().numpy()
        for local_idx, sample_idx in enumerate(batch_ids):
            cache_idx = int(sample_idx)
            original_idx = index_labels[cache_idx] if cache_idx < len(index_labels) else cache_idx
            metrics = compute_metrics(
                opt_np[local_idx],
                pred[local_idx, 0].detach().cpu().numpy()[: config.horizon],
                target[local_idx, 0].detach().cpu().numpy()[: config.horizon],
                pred[local_idx, 1:].detach().cpu().numpy()[:, : config.horizon],
                target[local_idx, 1:].detach().cpu().numpy()[:, : config.horizon],
                valid[local_idx, 1:].detach().cpu().numpy(),
                distance_fields[local_idx].numpy(),
                field_debug[local_idx]["goal_distance"],
                config,
            )
            plot_one(
                out_dir / f"sample_{cache_idx:05d}_idx_{original_idx}.png",
                scene_maps[local_idx].numpy(),
                goal_maps[local_idx].numpy(),
                cache["goal_rel_xy"][batch_ids[local_idx]].numpy(),
                past_rel_pos[batch_ids[local_idx]].numpy(),
                cache["target_x0"][batch_ids[local_idx]].numpy(),
                cache["pred_x0"][batch_ids[local_idx]].numpy(),
                opt_np[local_idx],
                cache["entity_valid"][batch_ids[local_idx]].numpy(),
                field_debug[local_idx],
                metrics,
                f"sample {cache_idx} / data idx {original_idx}",
            )
    (out_dir / "visualization_config.json").write_text(
        json.dumps(
            {
                "tuning_dir": str(tuning_dir),
                "best_score": best_payload.get("score"),
                "num_visualized": len(selected),
                "start_index": start,
                "speed_bound_mps": speed_bound,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(out_dir)


if __name__ == "__main__":
    main()
