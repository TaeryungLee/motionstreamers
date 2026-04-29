from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import planning_collate_fn
from train_stage1_prediction import build_dataset
from vis.stage1_prediction_vis import plot_prediction_side_by_side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug Stage-1 v2 GT trajectory visualization.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], default="trumans")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--preprocessed-root", type=Path, default=Path("data/preprocessed"))
    parser.add_argument("--max-others", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1_prediction_v2/debug_gt_vis"))
    parser.add_argument("--no-distance-map", action="store_true")
    return parser.parse_args()


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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx, batch in enumerate(loader):
        plot_prediction_side_by_side(
            args.output_dir / f"gt_sample_{idx:03d}.png",
            clearance_map=batch["scene_maps"][0],
            origin_xy=tuple(float(v) for v in batch["map_origin"][0].tolist()),
            resolution=float(batch["map_resolution"][0].item()),
            history_xy=batch["past_rel_pos"][0],
            gt_xy=batch["target_x0"][0],
            pred_xy=batch["target_x0"][0],
            valid_slots=batch["entity_valid"][0],
            goal_map=batch["goal_map"][0],
            goal_rel_xy=batch["goal_rel_xy"][0],
            goal_valid=batch["goal_valid"][0],
            goal_in_crop=batch["goal_in_crop"][0],
            title=f"GT sample {idx}",
        )
    print(f"saved {len(indices)} GT visualization(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
