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
    parser = argparse.ArgumentParser(
        description="Render one 1-frame TRUMANS debug video per scene for camera/framing inspection."
    )
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        default=ROOT_DIR / "data" / "preprocessed" / "trumans" / "episodes_v2",
        help="Directory containing <scene>/episode_*.json",
    )
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--scene-list-file", type=Path, default=None)
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--episode-id", default=None, help="Optional fixed episode file name to use for every scene.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT_DIR / "data" / "preprocessed" / "trumans" / "scene_debug_vis",
        help="Final 1-frame mp4 root.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/worldstreamers_scene_debug_vis"),
        help="Intermediate render workspace.",
    )
    parser.add_argument("--render-mode", choices=["preview", "final"], default="preview")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-modes", nargs="+", default=["oblique", "topdown"])
    parser.add_argument("--camera-scale", type=float, default=None)
    parser.add_argument("--camera-scale-scene", type=float, default=None)
    parser.add_argument("--camera-scale-oblique", type=float, default=None)
    parser.add_argument("--camera-scale-topdown", type=float, default=None)
    parser.add_argument(
        "--show-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render current active goal marker/text overlays.",
    )
    parser.add_argument(
        "--bright-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default: enabled.",
    )
    parser.add_argument(
        "--clear-existing-characters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove characters already embedded in the TRUMANS scene.",
    )
    parser.add_argument(
        "--quiet-subprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide per-scene Blender logs and show only tqdm progress. Default: enabled.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Number of scene debug renders to run in parallel.")
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


def choose_episode_file(episodes_dir: Path, scene_id: str, episode_id: str | None) -> Path | None:
    scene_dir = episodes_dir / scene_id
    if not scene_dir.exists():
        return None
    if episode_id is not None:
        name = episode_id if episode_id.endswith(".json") else f"{episode_id}.json"
        path = scene_dir / name
        return path if path.exists() else None
    candidates = sorted(scene_dir.glob("episode_*.json"))
    return candidates[0] if candidates else None


def build_command(args: argparse.Namespace, scene_id: str, episode_path: Path, final_output: Path) -> list[str]:
    cmd = [
        "python",
        str(EPISODE_RUNNER),
        "--dataset",
        "trumans",
        "--scene-id",
        scene_id,
        "--episode-json",
        str(episode_path),
        "--output-dir",
        str((args.work_dir if args.work_dir.is_absolute() else (ROOT_DIR / args.work_dir)).resolve()),
        "--final-output",
        str(final_output.resolve()),
        "--frame-limit",
        "1",
        "--render-mode",
        args.render_mode,
        "--fps",
        str(int(args.fps)),
        "--camera-modes",
        *list(args.camera_modes),
        "--render-mp4",
    ]
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
    episodes_dir = args.episodes_dir if args.episodes_dir.is_absolute() else (ROOT_DIR / args.episodes_dir)
    if not episodes_dir.exists():
        raise FileNotFoundError(episodes_dir)
    output_root = args.output_root if args.output_root.is_absolute() else (ROOT_DIR / args.output_root)

    tasks: list[tuple[str, Path, Path]] = []
    for scene_id in collect_scene_ids(args, episodes_dir):
        episode_path = choose_episode_file(episodes_dir, scene_id, args.episode_id)
        if episode_path is None:
            continue
        final_output = output_root / f"{scene_id}__{episode_path.stem}.mp4"
        if args.skip_existing and final_output.exists():
            continue
        tasks.append((scene_id, episode_path, final_output))

    worker_count = max(1, int(args.workers))
    total = 0
    task_specs: list[tuple[list[str], bool, str]] = []
    for scene_id, episode_path, final_output in tasks:
        cmd = build_command(args, scene_id, episode_path, final_output)
        task_name = f"{scene_id}/{episode_path.stem}"
        if args.dry_run or not args.quiet_subprocess:
            print("run:", " ".join(cmd))
        if not args.dry_run:
            final_output.parent.mkdir(parents=True, exist_ok=True)
            task_specs.append((cmd, bool(args.quiet_subprocess), task_name))

    if args.dry_run:
        total = len(tasks)
    elif worker_count == 1:
        progress = tqdm(task_specs, total=len(task_specs), desc="trumans scene debug", unit="scene")
        for cmd, quiet_subprocess, task_name in progress:
            if hasattr(progress, "set_postfix_str"):
                progress.set_postfix_str(task_name)
            try:
                run_render_task((cmd, quiet_subprocess, task_name))
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Scene debug render failed for {task_name}. "
                    f"Rerun with --no-quiet-subprocess to inspect Blender logs."
                ) from exc
            total += 1
    else:
        progress = tqdm(total=len(task_specs), desc="trumans scene debug", unit="scene")
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
                        f"Scene debug render failed for {task_name}. "
                        f"Rerun with --workers 1 --no-quiet-subprocess to inspect Blender logs."
                    ) from exc
                progress.update(1)
                total += 1
        progress.close()

    print(f"dispatched {total} trumans scene debug renders")


if __name__ == "__main__":
    main()
