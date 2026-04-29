from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import planning_collate_fn
from models.stage1_planner import (
    ActiveAction,
    OptimizerConfig,
    PlannerState,
    build_plan_fields,
    load_speed_profile,
    optimize_root,
    planner_step,
)
from train_stage1_prediction import build_dataset
from vis.stage1_prediction_vis import plot_planner_three_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug Stage-1 optimizer and planner with GT trajectory input.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], default="trumans")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--preprocessed-root", type=Path, default=Path("data/preprocessed"))
    parser.add_argument("--max-others", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1_prediction_v2/debug_planner"))
    parser.add_argument("--speed-profile", type=Path, default=None)
    parser.add_argument("--planner-mode", choices=["MOVE", "ACT", "WAIT"], default="MOVE")
    parser.add_argument("--optimizer-steps", type=int, default=80)
    parser.add_argument("--no-distance-map", action="store_true")
    return parser.parse_args()


def gaussian_occ_from_paths(paths: np.ndarray, H: int, W: int, origin_xy: tuple[float, float], resolution: float, sigma: float = 0.18) -> np.ndarray:
    K, T, _ = paths.shape
    rows, cols = np.indices((H, W), dtype=np.float32)
    x = float(origin_xy[0]) + (cols + 0.5) * float(resolution)
    y = float(origin_xy[1]) + ((H - 1 - rows) + 0.5) * float(resolution)
    out = np.zeros((K, T, H, W), dtype=np.float32)
    for k in range(K):
        for t in range(T):
            dx = x - float(paths[k, t, 0])
            dy = y - float(paths[k, t, 1])
            out[k, t] = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).astype(np.float32)
    return out


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args)
    indices = list(range(int(args.start_index), min(len(dataset), int(args.start_index) + int(args.num_samples))))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=planning_collate_fn,
    )
    speed_profile = load_speed_profile(args.speed_profile) if args.speed_profile is not None else None
    config = OptimizerConfig(steps=int(args.optimizer_steps), speed_profile=speed_profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    active_action = ActiveAction(label="debug", goal_area_id="debug", goal_type_id=0, interruptible=True)

    for idx, batch in enumerate(loader):
        origin = tuple(float(v) for v in batch["map_origin"][0].tolist())
        resolution = float(batch["map_resolution"][0].item())
        scene = batch["scene_maps"][0]
        H, W = scene.shape[-2:]
        gt_xy = batch["target_x0"][0].numpy()
        occ = gaussian_occ_from_paths(gt_xy[:, :60], H, W, origin, resolution)
        fields = build_plan_fields(
            prediction={"ego_future_occ": occ[0], "others_future_occ": occ[1:]},
            scene_maps=scene,
            goal_map=batch["goal_map"][0],
            map_origin_xy=origin,
            map_resolution=resolution,
        )

        roots = batch["past_rel_pos"][0, 0].numpy()
        current_pos = (float(roots[-1, 0]), float(roots[-1, 1]))
        current_vel = (float((roots[-1, 0] - roots[-2, 0]) * 30.0), float((roots[-1, 1] - roots[-2, 1]) * 30.0))
        state = PlannerState(mode=args.planner_mode)
        optimizer_plan = optimize_root(fields, state, active_action, current_pos, current_vel, config=config)
        planned_state = planner_step(optimizer_plan, fields, state, active_action)
        planner_plan = optimize_root(fields, planned_state, active_action, current_pos, current_vel, config=config)

        optimizer_xy = gt_xy[:, :60].copy()
        planner_xy = gt_xy[:, :60].copy()
        optimizer_xy[0] = optimizer_plan.pos_xy
        planner_xy[0] = planner_plan.pos_xy
        plot_planner_three_panel(
            args.output_dir / f"planner_sample_{idx:03d}.png",
            clearance_map=scene,
            origin_xy=origin,
            resolution=resolution,
            history_xy=batch["past_rel_pos"][0],
            gt_xy=gt_xy,
            optimizer_xy=optimizer_xy,
            planner_xy=planner_xy,
            valid_slots=batch["entity_valid"][0],
            goal_map=batch["goal_map"][0],
            goal_rel_xy=batch["goal_rel_xy"][0],
            goal_valid=batch["goal_valid"][0],
            goal_in_crop=batch["goal_in_crop"][0],
            title=f"planner sample {idx} {state.mode}->{planned_state.mode}",
        )
    print(f"saved {len(indices)} planner visualization(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
