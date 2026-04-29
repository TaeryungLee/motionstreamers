from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
EPISODE_RUNNER = ROOT_DIR / "vis" / "run_episode_blender.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render full multi-character episodes with Blender.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--episodes-dir", type=Path, default=None, help="Directory containing <scene_id>/episode_*.json")
    parser.add_argument("--scene-id", default=None, help="Render a single scene id.")
    parser.add_argument("--scene-list-file", type=Path, default=None, help="Line-separated scene ids.")
    parser.add_argument("--scene-ids", nargs="*", default=None, help="Scene ids.")
    parser.add_argument("--camera-mode", choices=["scene", "topdown", "oblique"], default=None, help="Common camera mode alias for dataset-specific mode flags.")
    parser.add_argument("--camera-modes", nargs="+", default=None, help="Render multiple camera modes for each episode.")
    parser.add_argument("--episode-id", default=None, help="Render a single episode file name (e.g. episode_0001.json).")
    parser.add_argument("--episode-index", type=int, default=None, help="Render a single episode by sorted index within one scene.")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs" / "blender_episodes")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render-mode", choices=["preview", "final"], default="preview")
    parser.add_argument("--camera-mode-trumans", choices=["scene", "topdown", "oblique"], default="scene")
    parser.add_argument("--camera-mode-lingo", choices=["oblique", "topdown"], default="oblique")
    parser.add_argument(
        "--camera-scale",
        type=float,
        default=None,
        help="Optional camera scale override. When omitted, a dataset/mode-specific default is used.",
    )
    parser.add_argument(
        "--clear-existing-characters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="For TRUMANS, remove characters already embedded in the source scene. Default: enabled.",
    )
    parser.add_argument(
        "--bright-preview",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add debug lighting for preview renders. Default: enabled for preview mode.",
    )
    parser.add_argument(
        "--show-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render goal target markers and text labels inside Blender.",
    )
    parser.add_argument("--render-mp4", action="store_true", default=True, help="Render mp4 clips (this is the default).")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_scene_ids(args: argparse.Namespace) -> list[str]:
    if args.scene_id is not None:
        return [args.scene_id]
    if args.scene_ids:
        return [s.strip() for s in args.scene_ids if s.strip()]
    if args.scene_list_file is not None:
        return [line.strip() for line in args.scene_list_file.read_text().splitlines() if line.strip()]
    episodes_dir = args.episodes_dir
    if episodes_dir is None:
        return []
    return sorted([p.name for p in episodes_dir.iterdir() if p.is_dir()])


def iter_episode_files(
    episodes_dir: Path,
    scene_id: str,
    episode_id: str | None = None,
    episode_index: int | None = None,
) -> list[Path]:
    scene_dir = episodes_dir / scene_id
    if not scene_dir.exists():
        return []
    if episode_id is not None:
        normalized = episode_id if episode_id.endswith(".json") else f"{episode_id}.json"
        episode_file = scene_dir / normalized
        return [episode_file] if episode_file.exists() else []

    episodes = sorted(scene_dir.glob("episode_*.json"))
    if episode_index is None:
        return list(episodes)
    if not (0 <= int(episode_index) < len(episodes)):
        return []
    return [episodes[int(episode_index)]]


def resolve_camera_mode(args: argparse.Namespace) -> str:
    if args.dataset == "trumans":
        if args.camera_mode is None:
            return args.camera_mode_trumans
        if args.camera_mode not in {"scene", "topdown", "oblique"}:
            raise ValueError(f"Invalid camera mode for trumans: {args.camera_mode}. Use scene, topdown, or oblique.")
        return args.camera_mode

    if args.camera_mode is None:
        return args.camera_mode_lingo
    if args.camera_mode not in {"oblique", "topdown"}:
        raise ValueError(f"Invalid camera mode for lingo: {args.camera_mode}. Use oblique or topdown.")
    return args.camera_mode


def resolve_camera_scale(args: argparse.Namespace, camera_mode: str) -> float:
    if args.camera_scale is not None:
        return float(args.camera_scale)
    if args.dataset == "lingo" and camera_mode == "topdown":
        return 1.7
    return 1.0


def resolve_bright_preview(args: argparse.Namespace) -> bool:
    if args.bright_preview is not None:
        return bool(args.bright_preview)
    return args.render_mode == "preview"


def resolve_clear_existing_characters(args: argparse.Namespace) -> bool:
    if args.clear_existing_characters is not None:
        return bool(args.clear_existing_characters)
    return args.dataset == "trumans"


def build_runner_command(
    args: argparse.Namespace,
    scene_id: str,
    episode_path: Path,
    camera_mode: str,
    camera_scale: float,
    bright_preview: bool,
    clear_existing_characters: bool,
) -> list[str]:
    cmd = [
        "python",
        str(EPISODE_RUNNER),
        "--dataset",
        args.dataset,
        "--scene-id",
        scene_id,
        "--episode-json",
        str(episode_path),
        "--render-mode",
        args.render_mode,
        "--camera-scale",
        str(camera_scale),
        "--fps",
        str(int(args.fps)),
        "--output-dir",
        str(args.output_dir.resolve()),
    ]
    if args.camera_modes:
        cmd.extend(["--camera-modes", *args.camera_modes])
    else:
        cmd.extend(["--camera-mode", camera_mode])
    if args.render_mp4:
        cmd.append("--render-mp4")
    if args.show_targets:
        cmd.append("--show-targets")
    if bright_preview:
        cmd.append("--bright-preview")
    if args.dataset == "trumans" and clear_existing_characters:
        cmd.append("--clear-existing-characters")
    return cmd


def main() -> None:
    args = parse_args()
    episodes_root = args.episodes_dir or (ROOT_DIR / "data" / "preprocessed" / args.dataset / "episodes_v2")
    if not episodes_root.exists():
        raise FileNotFoundError(f"episodes_dir not found: {episodes_root}")

    if args.episode_id is not None and args.episode_index is not None:
        raise ValueError("--episode-id and --episode-index cannot be used together.")

    scene_ids = collect_scene_ids(args)
    if not scene_ids:
        raise ValueError("No scene ids provided and no auto-discoverable scenes found.")
    if args.episode_index is not None and len(scene_ids) != 1:
        raise ValueError("--episode-index can only be used with a single scene.")

    camera_mode = resolve_camera_mode(args)
    if args.camera_modes:
        allowed = {"scene", "topdown", "oblique"} if args.dataset == "trumans" else {"oblique", "topdown"}
        invalid = [mode for mode in args.camera_modes if mode not in allowed]
        if invalid:
            raise ValueError(f"Invalid camera mode(s) for {args.dataset}: {', '.join(invalid)}")
    camera_scale = resolve_camera_scale(args, camera_mode)
    bright_preview = resolve_bright_preview(args)
    clear_existing_characters = resolve_clear_existing_characters(args)

    total_episodes = 0
    for scene_id in scene_ids:
        episode_files = iter_episode_files(
            episodes_root,
            scene_id,
            episode_id=args.episode_id,
            episode_index=args.episode_index,
        )
        if not episode_files:
            if args.episode_id is not None:
                raise FileNotFoundError(f"Episode not found: {args.episode_id} in scene {scene_id}.")
            if args.episode_index is not None:
                raise IndexError(f"Episode index {args.episode_index} is out of range for scene {scene_id}.")
            raise ValueError(f"No episode files found in scene {scene_id}.")

        for episode_path in episode_files:
            cmd = build_runner_command(
                args=args,
                scene_id=scene_id,
                episode_path=episode_path,
                camera_mode=camera_mode,
                camera_scale=camera_scale,
                bright_preview=bright_preview,
                clear_existing_characters=clear_existing_characters,
            )
            print("run:", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)
            total_episodes += 1

    print(f"dispatched {total_episodes} blender episode renders")


if __name__ == "__main__":
    main()
