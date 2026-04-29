from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from episode_blender_common import (
    DEFAULT_BLENDER,
    ROOT_DIR,
    export_episode_character_clips,
    load_json,
    render_side_by_side_mp4,
    resolve_lingo_scene_obj,
    resolve_trumans_scene_blend,
    to_abs,
)


SCRIPT_DIR = Path(__file__).resolve().parent
TRUMANS_APPLY_SCRIPT = SCRIPT_DIR / "blender_apply_trumans_episode.py"
LINGO_APPLY_SCRIPT = SCRIPT_DIR / "blender_apply_lingo_episode.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Blender episode renderer for TRUMANS and LINGO.")
    parser.add_argument("--dataset", choices=["trumans", "lingo"], required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--episode-json", type=Path, required=True)

    parser.add_argument("--scene-blend", type=Path, default=None)
    parser.add_argument("--blend-root", type=Path, default=Path("data/raw/trumans/Recordings_blend"))
    parser.add_argument("--scene-obj", type=Path, default=None)
    parser.add_argument("--mesh-root", type=Path, default=Path("data/raw/lingo/Scene_mesh"))

    parser.add_argument("--object-name", default="SMPLX-mesh-male")
    parser.add_argument("--smplx-gender", default="male", choices=["female", "male", "neutral"])
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs")
    parser.add_argument("--final-output", type=Path, default=None, help="Optional final mp4 path to copy after render.")
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-limit", type=int, default=None, help="Optional max number of frames from the episode start.")
    parser.add_argument("--render-mode", default="preview", choices=["preview", "final"])
    parser.add_argument("--camera-mode", default=None, help="Single camera mode.")
    parser.add_argument("--camera-modes", nargs="+", default=None, help="Multiple camera modes to render and optionally stack.")
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
        help="Add debug lighting. Default: enabled in preview mode.",
    )
    parser.add_argument("--render-mp4", action="store_true")
    parser.add_argument("--save-blend", action="store_true")
    parser.add_argument(
        "--clear-existing-characters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="TRUMANS only. Remove characters already embedded in the scene. Default: enabled.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def allowed_camera_modes(dataset: str) -> set[str]:
    if dataset == "trumans":
        return {"scene", "topdown", "oblique"}
    return {"oblique", "topdown"}


def resolve_camera_modes(args: argparse.Namespace) -> list[str]:
    if args.camera_modes:
        modes = list(args.camera_modes)
    elif args.camera_mode is not None:
        modes = [str(args.camera_mode)]
    elif args.dataset == "trumans":
        modes = ["scene"]
    else:
        modes = ["oblique"]

    invalid = [mode for mode in modes if mode not in allowed_camera_modes(args.dataset)]
    if invalid:
        raise ValueError(f"Invalid camera mode(s) for {args.dataset}: {', '.join(invalid)}")
    return modes


def resolve_camera_scale(args: argparse.Namespace, camera_mode: str) -> float:
    if camera_mode == "scene" and args.camera_scale_scene is not None:
        return float(args.camera_scale_scene)
    if camera_mode == "oblique" and args.camera_scale_oblique is not None:
        return float(args.camera_scale_oblique)
    if camera_mode == "topdown" and args.camera_scale_topdown is not None:
        return float(args.camera_scale_topdown)
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


def resolve_episode_work_dir(args: argparse.Namespace, episode_json: Path) -> Path:
    return to_abs(args.output_dir) / f"{args.dataset}_episode_vis" / args.scene_id / episode_json.stem


def resolve_final_output(args: argparse.Namespace) -> Path | None:
    if args.final_output is None:
        return None
    return to_abs(args.final_output)


def resolve_scene_asset(args: argparse.Namespace) -> tuple[Path, dict]:
    if args.dataset == "trumans":
        scene_blend = resolve_trumans_scene_blend(args.scene_id, args.scene_blend, args.blend_root)
        return scene_blend, {"scene_blend": str(scene_blend)}

    scene_obj, resolved_scene_mesh_id = resolve_lingo_scene_obj(args.scene_id, args.scene_obj, args.mesh_root)
    return scene_obj, {
        "scene_obj": str(scene_obj),
        "resolved_scene_mesh_id": resolved_scene_mesh_id,
    }


def build_meta(
    args: argparse.Namespace,
    episode_json: Path,
    scene_asset: Path,
    scene_extra: dict,
    char_records: list[dict],
    episode: dict,
) -> dict:
    meta = {
        "dataset": args.dataset,
        "scene_id": args.scene_id,
        "episode_id": episode_json.stem,
        "episode_json": str(episode_json),
        "smplx_gender": args.smplx_gender,
        "scenario_type": episode.get("scenario_type"),
        "num_characters": len(char_records),
        "characters": char_records,
        "target_overlays": build_target_overlays(episode, char_records),
    }
    if args.dataset == "trumans":
        meta["scene_blend"] = str(scene_asset)
    else:
        meta["scene_obj"] = str(scene_asset)
    meta.update(scene_extra)
    return meta


def goal_overlay_position(goal: dict) -> tuple[list[float], str] | tuple[None, None]:
    body_goal = goal.get("body_goal")
    hand_goal = goal.get("hand_goal")
    goal_category = str(goal.get("goal_category") or "")
    goal_type = str(goal.get("goal_type") or "")
    prefer_hand = goal_category in {"fixed", "support_needed"} or goal_type in {
        "open",
        "close",
        "pick_up",
        "put_down",
        "type",
        "wipe",
        "drink",
        "answer",
        "play",
    }
    if prefer_hand and isinstance(hand_goal, list) and len(hand_goal) == 3:
        return hand_goal, "hand"
    if isinstance(body_goal, list) and len(body_goal) == 3:
        return body_goal, "body"
    if isinstance(hand_goal, list) and len(hand_goal) == 3:
        return hand_goal, "hand"
    return None, None


def goal_overlay_label(character_index: int, step_idx: int, goal: dict) -> str:
    source_segment = goal.get("source_segment", {})
    if isinstance(source_segment, dict):
        text = str(source_segment.get("text") or "").strip()
    else:
        text = ""
    if text:
        base = text
    else:
        goal_type = str(goal.get("goal_type") or "").strip()
        target_name = str(goal.get("target_name") or "").strip()
        if goal_type and target_name:
            base = f"{goal_type} {target_name}"
        else:
            base = goal_type or target_name or "goal"
    return f"● Char {character_index}-{step_idx}: {base}"


def build_target_overlays(episode: dict, char_records: list[dict]) -> list[dict]:
    characters = episode.get("character_assignments", [])
    if not characters:
        return []

    color_by_char = {str(rec.get("character_id")): str(rec.get("color")) for rec in char_records}
    clip_start_by_char = {str(rec.get("character_id")): int(rec.get("local_start", 0)) for rec in char_records}
    clip_end_by_char = {str(rec.get("character_id")): int(rec.get("local_end", 0)) for rec in char_records}
    overlays: list[dict] = []
    for char_idx, char in enumerate(characters):
        character_id = str(char.get("character_id", f"char_{char_idx:02d}"))
        clip_local_start = int(clip_start_by_char.get(character_id, 0))
        clip_local_end = int(clip_end_by_char.get(character_id, clip_local_start))
        goals = list(char.get("goal_sequence", []))
        goal_starts: list[int] = []
        for goal in goals:
            source_segment = goal.get("source_segment", {})
            if isinstance(source_segment, dict):
                goal_starts.append(int(source_segment.get("start", clip_local_start)))
            else:
                goal_starts.append(clip_local_start)

        for step_idx, goal in enumerate(goals):
            position, position_kind = goal_overlay_position(goal)
            if position is None:
                continue
            raw_start = goal_starts[step_idx]
            raw_end_exclusive = goal_starts[step_idx + 1] if step_idx + 1 < len(goal_starts) else (clip_local_end + 1)
            active_frame_start = max(0, raw_start - clip_local_start)
            active_frame_end = max(active_frame_start, raw_end_exclusive - clip_local_start - 1)
            overlays.append(
                {
                    "character_id": character_id,
                    "character_index": int(char_idx),
                    "step_index": int(step_idx),
                    "position": [float(position[0]), float(position[1]), float(position[2])],
                    "position_kind": position_kind,
                    "label": goal_overlay_label(char_idx, step_idx, goal),
                    "color": color_by_char.get(character_id),
                    "goal_type": goal.get("goal_type"),
                    "goal_category": goal.get("goal_category"),
                    "target_name": goal.get("target_name"),
                    "active_frame_start": int(active_frame_start),
                    "active_frame_end": int(active_frame_end),
                }
            )
    return overlays


def build_blender_command(
    args: argparse.Namespace,
    scene_asset: Path,
    meta_json: Path,
    camera_mode: str,
    camera_scale: float,
    render_path: Path,
    blend_path: Path,
    bright_preview: bool,
    clear_existing_characters: bool,
) -> list[str]:
    cmd = [str(to_abs(args.blender_bin)), "-b"]
    if args.dataset == "trumans":
        cmd.extend(
            [
                str(scene_asset),
                "--python",
                str(TRUMANS_APPLY_SCRIPT),
                "--",
            ]
        )
    else:
        cmd.extend(
            [
                "--factory-startup",
                "--python",
                str(LINGO_APPLY_SCRIPT),
                "--",
            ]
        )

    cmd.extend(
        [
            "--meta-json",
            str(meta_json),
            "--render-mode",
            args.render_mode,
            "--camera-mode",
            camera_mode,
            "--camera-scale",
            str(camera_scale),
            "--fps",
            str(int(args.fps)),
        ]
    )
    if args.frame_limit is not None:
        cmd.extend(["--frame-limit", str(int(args.frame_limit))])
    if args.show_targets:
        cmd.append("--show-targets")
    if bright_preview:
        cmd.append("--bright-preview")
    if args.render_mp4:
        cmd.extend(["--render-mp4", str(render_path)])
    if args.save_blend:
        cmd.extend(["--save-blend", str(blend_path)])
    if args.dataset == "trumans" and clear_existing_characters:
        cmd.append("--clear-existing-characters")
    return cmd


def copy_final_video(src: Path, dst: Path, dry_run: bool) -> None:
    print("final_output:", dst)
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    episode_json = to_abs(args.episode_json)
    if not episode_json.exists():
        raise FileNotFoundError(episode_json)

    camera_modes = resolve_camera_modes(args)
    bright_preview = resolve_bright_preview(args)
    clear_existing_characters = resolve_clear_existing_characters(args)
    final_output = resolve_final_output(args)

    episode = load_json(episode_json)
    scene_asset, scene_extra = resolve_scene_asset(args)
    work_dir = resolve_episode_work_dir(args, episode_json)
    work_dir.mkdir(parents=True, exist_ok=True)

    char_records = export_episode_character_clips(
        dataset=args.dataset,
        scene_id=args.scene_id,
        episode=episode,
        out_dir=work_dir,
        base_object_name=args.object_name,
        load_hand=(args.dataset == "trumans"),
    )
    if not char_records:
        raise ValueError(f"No character clips could be exported from {episode_json}")

    meta = build_meta(
        args=args,
        episode_json=episode_json,
        scene_asset=scene_asset,
        scene_extra=scene_extra,
        char_records=char_records,
        episode=episode,
    )
    meta_json = work_dir / "meta.json"
    meta_json.write_text(json.dumps(meta, indent=2))

    print("episode_json:", episode_json)
    print("meta_json:", meta_json)
    rendered_paths: list[Path] = []
    for camera_mode in camera_modes:
        render_path = work_dir / f"render_{camera_mode}.mp4"
        blend_path = work_dir / f"episode_{camera_mode}.blend"
        cmd = build_blender_command(
            args=args,
            scene_asset=scene_asset,
            meta_json=meta_json,
            camera_mode=camera_mode,
            camera_scale=resolve_camera_scale(args, camera_mode),
            render_path=render_path,
            blend_path=blend_path,
            bright_preview=bright_preview,
            clear_existing_characters=clear_existing_characters,
        )
        print("blender_cmd:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
        if args.render_mp4:
            rendered_paths.append(render_path)

    stacked_path: Path | None = None
    if len(camera_modes) > 1 and args.render_mp4:
        stacked_path = work_dir / "render.mp4"
        stacked_cmd = render_side_by_side_mp4(
            rendered_paths,
            stacked_path,
            text_overlays=meta.get("target_overlays") if args.show_targets else None,
            dry_run=args.dry_run,
        )
        if stacked_cmd is not None:
            print("ffmpeg_cmd:", " ".join(stacked_cmd))

    if final_output is not None and args.render_mp4:
        source = stacked_path if stacked_path is not None else (work_dir / f"render_{camera_modes[0]}.mp4")
        copy_final_video(source, final_output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
