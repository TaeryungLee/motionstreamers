from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worldstreamers_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/worldstreamers_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
LABELS = ["ego", "other1", "other2", "other3"]


def as_numpy(value: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value, dtype=dtype)


def grid_to_world(rows: np.ndarray, cols: np.ndarray, origin_xy: tuple[float, float], resolution: float, height: int) -> np.ndarray:
    x = float(origin_xy[0]) + (cols.astype(np.float32) + 0.5) * float(resolution)
    y_from_bottom = (height - 1 - rows.astype(np.float32)) + 0.5
    y = float(origin_xy[1]) + y_from_bottom * float(resolution)
    return np.stack([x, y], axis=-1)


def soft_argmax_occ(occ: Any, origin_xy: tuple[float, float], resolution: float, eps: float = 1e-8) -> np.ndarray:
    """Return world XY path from occupancy maps.

    occ can be [T,H,W] or [K,T,H,W]. Positive heatmap mass is normalized.
    If a frame has no positive mass, it falls back to a softmax over logits.
    """
    maps = as_numpy(occ)
    squeeze = maps.ndim == 3
    if squeeze:
        maps = maps[None]
    if maps.ndim != 4:
        raise ValueError(f"Expected occ [T,H,W] or [K,T,H,W], got {maps.shape}")
    K, T, H, W = maps.shape
    rows, cols = np.indices((H, W), dtype=np.float32)
    xy = grid_to_world(rows.reshape(-1), cols.reshape(-1), origin_xy, resolution, H).reshape(H * W, 2)
    out = np.zeros((K, T, 2), dtype=np.float32)
    flat = maps.reshape(K, T, H * W)
    for k in range(K):
        for t in range(T):
            weights = np.maximum(flat[k, t], 0.0)
            total = float(weights.sum())
            if total <= eps:
                logits = flat[k, t] - float(flat[k, t].max())
                weights = np.exp(logits)
                total = float(weights.sum())
            out[k, t] = (xy * (weights / max(total, eps))[:, None]).sum(axis=0)
    return out[0] if squeeze else out


def _setup_axis(
    ax: Any,
    scene_map: np.ndarray,
    origin_xy: tuple[float, float],
    resolution: float,
    title: str,
    goal_map: np.ndarray | None = None,
    goal_rel_xy: np.ndarray | None = None,
    goal_valid: bool = False,
    goal_in_crop: bool = False,
) -> None:
    scene = as_numpy(scene_map)
    if scene.ndim == 3 and scene.shape[0] >= 3:
        obstacle = scene[2]
        background = 1.0 - obstacle.clip(0.0, 1.0)
    else:
        background = scene[0] if scene.ndim == 3 else scene
    H, W = background.shape
    extent = [
        float(origin_xy[0]),
        float(origin_xy[0]) + W * float(resolution),
        float(origin_xy[1]),
        float(origin_xy[1]) + H * float(resolution),
    ]
    ax.imshow(
        np.flipud(background),
        cmap="gray",
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=max(1.0, float(background.max())),
        alpha=1.0,
    )
    if scene.ndim == 3 and scene.shape[0] >= 3:
        distance = scene[1]
        try:
            ax.contour(
                np.linspace(extent[0], extent[1], W),
                np.linspace(extent[2], extent[3], H),
                np.flipud(distance),
                levels=6,
                colors="#666666",
                linewidths=0.35,
                alpha=0.45,
            )
        except Exception:
            pass
    if goal_map is not None:
        goal = as_numpy(goal_map)
        if goal.ndim == 3:
            goal = goal[0]
        if float(goal.max()) > 1e-6:
            masked = np.ma.masked_less_equal(np.flipud(goal), 1e-4)
            ax.imshow(masked, cmap="summer", origin="lower", extent=extent, alpha=0.55, vmin=0.0, vmax=max(1.0, float(goal.max())))
    if goal_valid and goal_rel_xy is not None:
        goal_xy = as_numpy(goal_rel_xy).reshape(-1)
        x0, x1, y0, y1 = extent
        if goal_in_crop:
            ax.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=170, c="#ffd400", edgecolors="black", linewidths=0.8, zorder=8, label="goal")
        else:
            clipped_x = float(np.clip(goal_xy[0], x0, x1))
            clipped_y = float(np.clip(goal_xy[1], y0, y1))
            ax.scatter([clipped_x], [clipped_y], marker="*", s=150, c="#ffd400", edgecolors="black", linewidths=0.8, zorder=8, label="goal outside")
    suffix = "" if (not goal_valid or goal_in_crop) else " / goal outside crop"
    ax.set_title(title + suffix)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _plot_paths(
    ax: Any,
    history_xy: np.ndarray | None,
    future_xy: np.ndarray | None,
    valid_slots: np.ndarray | None,
    *,
    future_style: str,
    future_label: str,
) -> None:
    if future_xy is None and history_xy is None:
        return
    if future_xy is not None:
        future_xy = as_numpy(future_xy)
        K = future_xy.shape[0]
    else:
        history_xy = as_numpy(history_xy)
        K = history_xy.shape[0]
    if valid_slots is None:
        valid = np.ones((K,), dtype=np.bool_)
    else:
        valid = as_numpy(valid_slots, dtype=np.float32).astype(bool)
    for slot in range(K):
        if slot >= len(valid) or not valid[slot]:
            continue
        color = COLORS[slot % len(COLORS)]
        label = LABELS[slot] if slot < len(LABELS) else f"slot{slot}"
        if history_xy is not None:
            hist = as_numpy(history_xy)[slot]
            ax.plot(hist[:, 0], hist[:, 1], color=color, linewidth=3.2, alpha=0.95, marker="o", markersize=2.6, zorder=5, label=f"{label} hist")
            ax.scatter(hist[-1:, 0], hist[-1:, 1], color=color, s=34, zorder=6)
        if future_xy is not None:
            fut = future_xy[slot]
            ax.plot(fut[:, 0], fut[:, 1], future_style, color=color, linewidth=1.4, alpha=0.85, zorder=4, label=f"{label} {future_label}")


def plot_prediction_side_by_side(
    output_path: Path | str,
    clearance_map: Any,
    origin_xy: tuple[float, float],
    resolution: float,
    history_xy: Any | None,
    gt_xy: Any,
    pred_xy: Any | None,
    valid_slots: Any | None = None,
    goal_map: Any | None = None,
    goal_rel_xy: Any | None = None,
    goal_valid: Any | None = None,
    goal_in_crop: Any | None = None,
    title: str = "stage1 prediction",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = None if history_xy is None else as_numpy(history_xy)
    valid = None if valid_slots is None else as_numpy(valid_slots, dtype=np.float32)
    goal_is_valid = bool(as_numpy(goal_valid, dtype=np.float32).reshape(-1)[0]) if goal_valid is not None else False
    goal_is_in_crop = bool(as_numpy(goal_in_crop, dtype=np.float32).reshape(-1)[0]) if goal_in_crop is not None else False
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    _setup_axis(axes[0], clearance_map, origin_xy, resolution, f"{title} / GT", goal_map, goal_rel_xy, goal_is_valid, goal_is_in_crop)
    _plot_paths(axes[0], history, as_numpy(gt_xy), valid, future_style="-", future_label="gt")
    _setup_axis(axes[1], clearance_map, origin_xy, resolution, f"{title} / Pred", goal_map, goal_rel_xy, goal_is_valid, goal_is_in_crop)
    _plot_paths(axes[1], history, None if pred_xy is None else as_numpy(pred_xy), valid, future_style="--", future_label="pred")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=7)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_planner_three_panel(
    output_path: Path | str,
    clearance_map: Any,
    origin_xy: tuple[float, float],
    resolution: float,
    history_xy: Any | None,
    gt_xy: Any,
    optimizer_xy: Any,
    planner_xy: Any,
    valid_slots: Any | None = None,
    goal_map: Any | None = None,
    goal_rel_xy: Any | None = None,
    goal_valid: Any | None = None,
    goal_in_crop: Any | None = None,
    title: str = "stage1 planner",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = None if history_xy is None else as_numpy(history_xy)
    valid = None if valid_slots is None else as_numpy(valid_slots, dtype=np.float32)
    goal_is_valid = bool(as_numpy(goal_valid, dtype=np.float32).reshape(-1)[0]) if goal_valid is not None else False
    goal_is_in_crop = bool(as_numpy(goal_in_crop, dtype=np.float32).reshape(-1)[0]) if goal_in_crop is not None else False
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    panels = [
        ("GT", as_numpy(gt_xy), "-"),
        ("Optimizer", as_numpy(optimizer_xy), "--"),
        ("Optimizer + Planner", as_numpy(planner_xy), "--"),
    ]
    for ax, (name, path, style) in zip(axes, panels):
        _setup_axis(ax, clearance_map, origin_xy, resolution, f"{title} / {name}", goal_map, goal_rel_xy, goal_is_valid, goal_is_in_crop)
        _plot_paths(ax, history, path, valid, future_style=style, future_label=name.lower())
        ax.legend(loc="upper right", fontsize=7)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
