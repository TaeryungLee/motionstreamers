from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, total=None, desc=None, unit=None):  # type: ignore
        return iterable


ROOT_DIR = Path(__file__).resolve().parent.parent
EPISODE_RUNNER = ROOT_DIR / "vis" / "run_episode_blender.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render all sampled episodes into outputs/<dataset>_episode_vis/<scene>/episode_XXXX.mp4")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--episodes-dir", type=Path, default=None, help="Directory containing <scene>/episode_*.json")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--scene-list-file", type=Path, default=None)
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--episode-id", default=None, help="Optional single episode file name.")
    parser.add_argument("--output-root", type=Path, default=None, help="Final mp4 root. Default: outputs/<dataset>_episode_vis")
    parser.add_argument("--work-dir", type=Path, default=ROOT_DIR / "outputs", help="Intermediate render workspace.")
    parser.add_argument("--render-mode", choices=["preview", "final"], default="preview")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--camera-modes", nargs="+", default=None, help="Default: oblique topdown")
    parser.add_argument("--camera-scale", type=float, default=None)
    parser.add_argument("--camera-scale-scene", type=float, default=None)
    parser.add_argument("--camera-scale-oblique", type=float, default=None)
    parser.add_argument("--camera-scale-topdown", type=float, default=None)
    parser.add_argument(
        "--show-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render goal target markers and text labels inside Blender.",
    )
    parser.add_argument(
        "--bright-preview",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: enabled in preview mode.",
    )
    parser.add_argument(
        "--clear-existing-characters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="TRUMANS only. Default: enabled.",
    )
    parser.add_argument(
        "--quiet-subprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide per-episode Blender/stdout logs and show only tqdm progress. Default: enabled.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Number of episodes to render in parallel.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_scene_ids(args: argparse.Namespace, episodes_dir: Path) -> list[str]:
    if args.scene_id is not None:
        return [args.scene_id]
    if args.scene_ids:
        return [s.strip() for s in args.scene_ids if s.strip()]
    if args.scene_list_file is not None:
        return [line.strip() for line in args.scene_list_file.read_text().splitlines() if line.strip()]
    return sorted([p.name for p in episodes_dir.iterdir() if p.is_dir()])


def iter_episode_files(episodes_dir: Path, scene_id: str, episode_id: str | None) -> list[Path]:
    scene_dir = episodes_dir / scene_id
    if not scene_dir.exists():
        return []
    if episode_id is not None:
        name = episode_id if episode_id.endswith(".json") else f"{episode_id}.json"
        path = scene_dir / name
        return [path] if path.exists() else []
    return sorted(scene_dir.glob("episode_*.json"))


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root if args.output_root.is_absolute() else (ROOT_DIR / args.output_root)
    return ROOT_DIR / "outputs" / f"{args.dataset}_episode_vis"


def resolve_camera_modes(args: argparse.Namespace) -> list[str]:
    if args.camera_modes:
        return list(args.camera_modes)
    return ["oblique", "topdown"]


def build_command(
    args: argparse.Namespace,
    scene_id: str,
    episode_path: Path,
    final_output: Path,
    camera_modes: list[str],
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
        "--output-dir",
        str((args.work_dir if args.work_dir.is_absolute() else (ROOT_DIR / args.work_dir)).resolve()),
        "--final-output",
        str(final_output.resolve()),
        "--render-mode",
        args.render_mode,
        "--fps",
        str(int(args.fps)),
        "--camera-modes",
        *camera_modes,
        "--render-mp4",
    ]
    if args.frame_limit is not None:
        cmd.extend(["--frame-limit", str(int(args.frame_limit))])
    if args.camera_scale is not None:
        cmd.extend(["--camera-scale", str(float(args.camera_scale))])
    if args.camera_scale_scene is not None:
        cmd.extend(["--camera-scale-scene", str(float(args.camera_scale_scene))])
    if args.camera_scale_oblique is not None:
        cmd.extend(["--camera-scale-oblique", str(float(args.camera_scale_oblique))])
    if args.camera_scale_topdown is not None:
        cmd.extend(["--camera-scale-topdown", str(float(args.camera_scale_topdown))])
    if args.show_targets:
        cmd.append("--show-targets")
    if args.bright_preview is not None:
        cmd.append("--bright-preview" if args.bright_preview else "--no-bright-preview")
    if args.clear_existing_characters is not None:
        cmd.append("--clear-existing-characters" if args.clear_existing_characters else "--no-clear-existing-characters")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def run_render_task(task: tuple[list[str], bool, str]) -> str:
    cmd, quiet_subprocess, task_name = task
    if quiet_subprocess:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True)
    return task_name


def main() -> None:
    args = parse_args()
    episodes_dir = args.episodes_dir or (ROOT_DIR / "data" / "preprocessed" / args.dataset / "episodes_v3")
    if not episodes_dir.exists():
        raise FileNotFoundError(episodes_dir)
    output_root = resolve_output_root(args)
    camera_modes = resolve_camera_modes(args)

    tasks: list[tuple[str, Path, Path]] = []
    for scene_id in collect_scene_ids(args, episodes_dir):
        episode_files = iter_episode_files(episodes_dir, scene_id, args.episode_id)
        if not episode_files:
            continue
        for episode_path in episode_files:
            final_output = output_root / scene_id / f"{episode_path.stem}.mp4"
            if args.skip_existing and final_output.exists():
                continue
            tasks.append((scene_id, episode_path, final_output))

    worker_count = max(1, int(args.workers))
    total = 0
    task_specs: list[tuple[list[str], bool, str]] = []
    for scene_id, episode_path, final_output in tasks:
        cmd = build_command(args, scene_id, episode_path, final_output, camera_modes)
        task_name = f"{scene_id}/{episode_path.stem}"
        if args.dry_run or not args.quiet_subprocess:
            print("run:", " ".join(cmd))
        if not args.dry_run:
            final_output.parent.mkdir(parents=True, exist_ok=True)
            task_specs.append((cmd, bool(args.quiet_subprocess), task_name))

    if args.dry_run:
        total = len(tasks)
    elif worker_count == 1:
        progress = tqdm(task_specs, total=len(task_specs), desc=f"{args.dataset} episodes", unit="episode")
        for cmd, quiet_subprocess, task_name in progress:
            if hasattr(progress, "set_postfix_str"):
                progress.set_postfix_str(task_name)
            try:
                run_render_task((cmd, quiet_subprocess, task_name))
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Episode render failed for {task_name}. "
                    f"Rerun with --no-quiet-subprocess to inspect Blender logs."
                ) from exc
            total += 1
    else:
        progress = tqdm(total=len(task_specs), desc=f"{args.dataset} episodes", unit="episode")
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_name = {
                executor.submit(run_render_task, spec): spec[2]
                for spec in task_specs
            }
            for future in concurrent.futures.as_completed(future_to_name):
                task_name = future_to_name[future]
                try:
                    future.result()
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f"Episode render failed for {task_name}. "
                        f"Rerun with --workers 1 --no-quiet-subprocess to inspect Blender logs."
                    ) from exc
                progress.update(1)
                total += 1
        progress.close()

    print(f"dispatched {total} episode visualizations")


if __name__ == "__main__":
    main()
