from __future__ import annotations

import json
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BLENDER = ROOT_DIR / "tools" / "blender-3.6.17-linux-x64" / "blender"

PALETTE_RGB = [
    "220,38,38",
    "16,185,129",
    "45,115,230",
    "245,158,11",
    "168,85,247",
    "14,165,233",
    "234,88,12",
    "236,72,153",
    "79,70,229",
    "217,119,6",
    "6,182,212",
    "249,115,22",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def to_abs(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT_DIR / path)


def to_abs_or_none(path: Path | None) -> Path | None:
    if path is None:
        return None
    return to_abs(path)


def make_color_for_index(index: int) -> str:
    return PALETTE_RGB[index % len(PALETTE_RGB)]


def rgb_spec_to_hex(color_spec: str | None, fallback: str = "FFFFFF") -> str:
    if not color_spec:
        return fallback
    parts = [part.strip() for part in str(color_spec).split(",")]
    if len(parts) != 3:
        return fallback
    try:
        vals = [max(0, min(255, int(float(part)))) for part in parts]
    except ValueError:
        return fallback
    return "".join(f"{value:02X}" for value in vals)


def rgb_spec_to_text_hex(
    color_spec: str | None,
    fallback: str = "FFFFFF",
    white_mix: float = 0.45,
) -> str:
    if not color_spec:
        return fallback
    parts = [part.strip() for part in str(color_spec).split(",")]
    if len(parts) != 3:
        return fallback
    try:
        vals = [max(0, min(255, int(float(part)))) for part in parts]
    except ValueError:
        return fallback
    mix = max(0.0, min(1.0, float(white_mix)))
    brightened = [int(round((value * (1.0 - mix)) + (255.0 * mix))) for value in vals]
    return "".join(f"{value:02X}" for value in brightened)


def ffmpeg_escape_drawtext(text: str) -> str:
    escaped = str(text)
    escaped = escaped.replace("\\", "\\\\")
    escaped = escaped.replace(":", r"\:")
    escaped = escaped.replace("'", r"\'")
    escaped = escaped.replace(",", r"\,")
    escaped = escaped.replace("%", r"\%")
    escaped = escaped.replace("[", r"\[")
    escaped = escaped.replace("]", r"\]")
    return escaped


def probe_video_size(path: Path) -> tuple[int, int]:
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin is None:
        raise FileNotFoundError("ffprobe not found in PATH.")

    result = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found in {path}")
    width = int(streams[0]["width"])
    height = int(streams[0]["height"])
    return width, height


def build_side_by_side_filter(
    input_count: int,
    stack_width: int,
    stack_height: int,
    text_overlays: list[dict] | None = None,
) -> tuple[str, str]:
    stack_inputs = "".join(f"[{idx}:v]" for idx in range(input_count))
    current_label = "stack"
    filters = [f"{stack_inputs}hstack=inputs={input_count}[{current_label}]"]

    if text_overlays:
        box_x = int(stack_width // 4)
        box_y = 0
        box_w = int(stack_width // 2)
        box_h = int(stack_height // 4)
        text_x = int(box_x + 24)
        boxed_label = "boxed"
        filters.append(
            f"[{current_label}]drawbox=x={box_x}:y={box_y}:w={box_w}:h={box_h}:color=black@0.62:t=fill[{boxed_label}]"
        )
        current_label = boxed_label
        for overlay_idx, overlay in enumerate(text_overlays):
            label = str(overlay.get("label") or "").strip()
            if not label:
                continue
            line_idx = int(overlay.get("character_index", 0))
            next_label = f"txt{overlay_idx:03d}"
            text_expr = ffmpeg_escape_drawtext(label)
            color_hex = rgb_spec_to_text_hex(overlay.get("color"), fallback="FFFFFF")
            start_frame = max(0, int(overlay.get("active_frame_start", 0)))
            end_frame = max(start_frame, int(overlay.get("active_frame_end", start_frame)))
            y_offset = 26 + (line_idx * 40)
            filters.append(
                f"[{current_label}]drawtext="
                f"text='{text_expr}':"
                f"fontcolor=0x{color_hex}:"
                f"fontsize=20:"
                f"borderw=2:"
                f"bordercolor=black@0.85:"
                f"x={text_x}:"
                f"y={y_offset}:"
                f"enable='between(n,{start_frame},{end_frame})'"
                f"[{next_label}]"
            )
            current_label = next_label

    return ";".join(filters), current_label


def load_sequence_json(dataset: str, scene_id: str, sequence_id: str) -> dict:
    seq_path = ROOT_DIR / "data" / "preprocessed" / dataset / "scenes" / scene_id / "sequences" / f"{sequence_id}.json"
    if not seq_path.exists():
        raise FileNotFoundError(seq_path)
    return load_json(seq_path)


def export_motion_window(sequence: dict, local_start: int, local_end: int, out_path: Path) -> int:
    motion_ref = sequence["human_motion_ref"]
    smpl = motion_ref["smplx"]
    seq_start = int(motion_ref["start"])
    clip_start = seq_start + int(local_start)
    clip_end = seq_start + int(local_end) + 1
    if clip_end <= clip_start:
        raise ValueError(f"Invalid clip range: start={local_start}, end={local_end}")

    payload = {
        "global_orient": np.asarray(
            np.load(resolve_repo_path(smpl["global_orient_path"]), mmap_mode="r")[clip_start:clip_end],
            dtype=np.float32,
        ),
        "body_pose": np.asarray(
            np.load(resolve_repo_path(smpl["body_pose_path"]), mmap_mode="r")[clip_start:clip_end],
            dtype=np.float32,
        ),
        "transl": np.asarray(
            np.load(resolve_repo_path(smpl["transl_path"]), mmap_mode="r")[clip_start:clip_end],
            dtype=np.float32,
        ),
    }
    if "left_hand_pose_path" in smpl:
        payload["left_hand_pose"] = np.asarray(
            np.load(resolve_repo_path(smpl["left_hand_pose_path"]), mmap_mode="r")[clip_start:clip_end],
            dtype=np.float32,
        )
    if "right_hand_pose_path" in smpl:
        payload["right_hand_pose"] = np.asarray(
            np.load(resolve_repo_path(smpl["right_hand_pose_path"]), mmap_mode="r")[clip_start:clip_end],
            dtype=np.float32,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return int(payload["transl"].shape[0])


def resolve_trumans_scene_blend(scene_id: str, scene_blend: Path | None, blend_root: Path) -> Path:
    if scene_blend is not None:
        resolved = to_abs(scene_blend)
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved

    candidate_root = to_abs(blend_root)
    candidates: list[str] = []

    def add(name: str) -> None:
        if name and name not in candidates:
            candidates.append(name)

    add(scene_id)
    if "_" in scene_id:
        add(scene_id.split("_", 1)[0])
    if scene_id.endswith("-copy"):
        add(scene_id[: -len("-copy")])

    checked: list[str] = []
    for candidate in candidates:
        direct = candidate_root / f"{candidate}.blend"
        nested = candidate_root / candidate / f"{candidate}.blend"
        checked.extend([str(direct), str(nested)])
        if direct.exists():
            return direct
        if nested.exists():
            return nested

    base_uuid = candidates[-1] if candidates else scene_id
    prefix_matches = sorted(candidate_root.glob(f"{base_uuid}*"))
    for match in prefix_matches:
        checked.append(str(match))
        if match.is_file() and match.suffix == ".blend":
            return match
        if match.is_dir():
            nested_blends = sorted(match.glob("*.blend"))
            if nested_blends:
                return nested_blends[0]
    raise FileNotFoundError(
        f"Scene blend not found for scene_id={scene_id}. "
        f"Checked candidates: {checked}."
    )


def candidate_scene_ids(scene_id: str) -> list[str]:
    candidates: list[str] = []

    def add(name: str) -> None:
        if name and name not in candidates:
            candidates.append(name)

    add(scene_id)
    add(scene_id.replace("_", "-"))
    add(scene_id.replace("-", "_"))
    if scene_id.endswith("_mirror"):
        base = scene_id[:-7]
        add(base)
        add(base.replace("_", "-"))
        add(base.replace("-", "_"))
    return candidates


def resolve_lingo_scene_obj(scene_id: str, scene_obj: Path | None, mesh_root: Path) -> tuple[Path, str]:
    if scene_obj is not None:
        resolved = to_abs(scene_obj)
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved, resolved.parent.name

    mesh_root = to_abs(mesh_root)
    for candidate in candidate_scene_ids(scene_id):
        scene_mesh = mesh_root / candidate / "mesh_low.obj"
        if scene_mesh.exists():
            return scene_mesh, candidate

    raise FileNotFoundError(
        f"No scene mesh found for scene_id={scene_id}. "
        f"Checked under {mesh_root} using candidates {candidate_scene_ids(scene_id)}."
    )


def _clip_bounds_from_goal_sequence(char: dict) -> tuple[str, int, int]:
    goals = char.get("goal_sequence", [])
    if not goals:
        raise ValueError(f"Character {char.get('character_id')} has no goal_sequence.")

    seq_ids = {
        str(goal.get("source_segment", {}).get("sequence_id", "")).strip()
        for goal in goals
        if goal.get("source_segment") is not None
    }
    seq_ids.discard("")
    if len(seq_ids) != 1:
        raise ValueError(f"Character {char.get('character_id')} spans multiple sequences: {sorted(seq_ids)}")

    starts = []
    ends = []
    for goal in goals:
        source = goal.get("source_segment", {})
        if "start" not in source or "end" not in source:
            continue
        starts.append(int(source["start"]))
        ends.append(int(source["end"]))
    if not starts or not ends:
        raise ValueError(f"Character {char.get('character_id')} has no usable source segment bounds.")
    return next(iter(seq_ids)), min(starts), max(ends)


def character_clip_spec(char: dict) -> dict:
    if char.get("source_window") is not None:
        window = char["source_window"]
        sequence_id = str(window["sequence_id"])
        local_start = int(window["local_start"])
        local_end = int(window["local_end"])
        return {
            "sequence_id": sequence_id,
            "local_start": local_start,
            "local_end": local_end,
            "source_kind": "source_window",
        }

    sequence_id, local_start, local_end = _clip_bounds_from_goal_sequence(char)
    return {
        "sequence_id": sequence_id,
        "local_start": local_start,
        "local_end": local_end,
        "source_kind": "goal_sequence_window",
    }


def export_episode_character_clips(
    dataset: str,
    scene_id: str,
    episode: dict,
    out_dir: Path,
    base_object_name: str,
    load_hand: bool,
) -> list[dict]:
    characters = episode.get("character_assignments", [])
    if not characters:
        return []

    char_records: list[dict] = []
    for char_idx, char in enumerate(characters):
        clip_spec = character_clip_spec(char)
        sequence = load_sequence_json(dataset, scene_id, clip_spec["sequence_id"])
        char_id = str(char.get("character_id", f"char_{char_idx:02d}"))
        motion_pkl = out_dir / f"{char_id}_smplx_results.pkl"
        num_frames = export_motion_window(sequence, clip_spec["local_start"], clip_spec["local_end"], motion_pkl)
        char_records.append(
            {
                "character_id": char_id,
                "object_name": f"{base_object_name}_{char_id}",
                "sequence_id": clip_spec["sequence_id"],
                "local_start": int(clip_spec["local_start"]),
                "local_end": int(clip_spec["local_end"]),
                "num_frames": int(num_frames),
                "motion_pkl": str(motion_pkl),
                "color": make_color_for_index(char_idx),
                "load_hand": bool(load_hand),
                "source_kind": clip_spec["source_kind"],
            }
        )
    return char_records


def render_side_by_side_mp4(
    input_paths: list[Path],
    output_path: Path,
    text_overlays: list[dict] | None = None,
    dry_run: bool = False,
) -> list[str] | None:
    resolved_inputs = [Path(path).resolve() for path in input_paths]
    if len(resolved_inputs) < 2:
        return None

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise FileNotFoundError("ffmpeg not found in PATH.")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_args: list[str] = []
    for idx, path in enumerate(resolved_inputs):
        input_args.extend(["-i", str(path)])
    if dry_run:
        stack_width = 1920
        stack_height = 540
    else:
        sizes = [probe_video_size(path) for path in resolved_inputs]
        stack_width = sum(width for width, _ in sizes)
        stack_height = max(height for _, height in sizes)

    filter_complex, output_label = build_side_by_side_filter(
        input_count=len(resolved_inputs),
        stack_width=stack_width,
        stack_height=stack_height,
        text_overlays=text_overlays,
    )

    cmd = [
        ffmpeg_bin,
        "-y",
        *input_args,
        "-filter_complex",
        filter_complex,
        "-map",
        f"[{output_label}]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    if dry_run:
        return cmd

    missing = [str(path) for path in resolved_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot create side-by-side render. Missing inputs: {missing}")

    subprocess.run(cmd, check=True)
    return cmd
