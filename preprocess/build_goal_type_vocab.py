from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSED_ROOT = PROJECT_ROOT / "data" / "preprocessed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset-level goal_type vocab from scenes_v2.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PREPROCESSED_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes_root = args.output_root / args.dataset / "scenes_v2"
    if not scenes_root.exists():
        raise FileNotFoundError(scenes_root)

    goal_types: set[str] = set()
    for scene_dir in sorted(path for path in scenes_root.iterdir() if path.is_dir()):
        seq_dir = scene_dir / "sequences"
        if not seq_dir.exists():
            continue
        for seq_path in sorted(seq_dir.glob("*.json")):
            seq_info = json.loads(seq_path.read_text())
            for segment in seq_info.get("segment_list", []):
                goal_type = str(segment.get("goal_type") or "").strip()
                if goal_type:
                    goal_types.add(goal_type)

    vocab = {"__background__": 0}
    for idx, goal_type in enumerate(sorted(goal_types), start=1):
        vocab[goal_type] = idx

    out_path = scenes_root / "goal_type_vocab.json"
    out_path.write_text(json.dumps(vocab, indent=2))
    print(out_path)
    print(json.dumps(vocab, indent=2))


if __name__ == "__main__":
    main()
