from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a multi-character LINGO episode to a Blender scene.")
    parser.add_argument("--meta-json", type=Path, required=True)
    parser.add_argument("--render-mode", default="preview", choices=["preview", "final"])
    parser.add_argument("--camera-mode", default="oblique", choices=["oblique", "topdown"])
    parser.add_argument("--camera-scale", type=float, default=1.0)
    parser.add_argument("--show-targets", action="store_true")
    parser.add_argument("--bright-preview", action="store_true")
    parser.add_argument("--save-blend", type=Path, default=None)
    parser.add_argument("--render-mp4", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-limit", type=int, default=None)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def rename_character_hierarchy(obj: bpy.types.Object, name_prefix: str) -> bpy.types.Object:
    obj.name = name_prefix
    if obj.parent is not None and obj.parent.type == "ARMATURE":
        obj.parent.name = f"{name_prefix}_armature"
    return obj


def create_character_instance(
    idx: int,
    object_name: str,
    gender: str,
    get_or_create_smplx_object,
    create_new_smplx_object,
) -> bpy.types.Object:
    if idx == 0:
        obj = get_or_create_smplx_object(object_name, gender)
    else:
        obj = create_new_smplx_object(gender)
    return rename_character_hierarchy(obj, object_name)


def main() -> None:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    args = parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from blender_apply_lingo_segment import (
        apply_scene_material,
        clear_default_scene,
        clip_scene_meshes_for_topdown,
        ensure_oblique_camera,
        ensure_preview_lighting,
        ensure_topdown_camera,
        ensure_world,
        import_scene_mesh,
    )
    from blender_apply_smplx_segment import (
        apply_character_color,
        configure_final_render,
        configure_preview_render,
        create_target_overlays,
        create_new_smplx_object,
        ensure_local_smplx_addon,
        get_or_create_smplx_object,
    )
    from visualize_smplx_motion.load_smplx_animatioin_clear import load_smplx_animation_new

    meta = load_json(args.meta_json.resolve())
    characters = meta.get("characters", [])
    if not characters:
        raise ValueError(f"No characters found in {args.meta_json}")

    clear_default_scene()
    ensure_world()
    ensure_local_smplx_addon(project_root)

    scene_meshes = import_scene_mesh(Path(meta["scene_obj"]).resolve())
    apply_scene_material(scene_meshes)

    max_frames = 0
    gender = str(meta.get("smplx_gender", "male"))
    for idx, char in enumerate(characters):
        obj = create_character_instance(
            idx=idx,
            object_name=str(char["object_name"]),
            gender=gender,
            get_or_create_smplx_object=get_or_create_smplx_object,
            create_new_smplx_object=create_new_smplx_object,
        )
        apply_character_color(obj, char.get("color"), material_name=f"CodexCharacterMaterial_{idx:02d}")
        load_smplx_animation_new(
            str(Path(char["motion_pkl"]).resolve()),
            obj,
            load_hand=bool(char.get("load_hand", False)),
            load_betas=False,
        )
        max_frames = max(max_frames, int(char.get("num_frames", 0)))

    scene = bpy.context.scene
    clip_scene_meshes_for_topdown(scene_meshes)
    if args.camera_mode == "topdown":
        ensure_topdown_camera(scene_meshes, args.camera_scale)
    else:
        ensure_oblique_camera(scene_meshes, bpy.data.objects[str(characters[0]["object_name"])], args.camera_scale)
    if args.bright_preview:
        ensure_preview_lighting(scene_meshes)

    if args.frame_limit is not None:
        max_frames = min(max_frames, max(1, int(args.frame_limit)))
    scene.frame_start = 0
    scene.frame_end = max(0, max_frames - 1)
    if hasattr(scene, "frame_step"):
        scene.frame_step = 1
    if hasattr(scene.render, "frame_map_old"):
        scene.render.frame_map_old = 1
    if hasattr(scene.render, "frame_map_new"):
        scene.render.frame_map_new = 1
    scene.render.fps = int(args.fps)
    if args.show_targets:
        create_target_overlays(meta.get("target_overlays", []))

    if args.render_mp4 is not None:
        out_path = args.render_mp4.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.render_mode == "preview":
            configure_preview_render(scene, resolution_percentage=100)
        else:
            configure_final_render(scene)
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.ffmpeg.audio_codec = "NONE"
        scene.render.filepath = str(out_path)
        bpy.ops.render.render(animation=True)

    if args.save_blend is not None:
        out_blend = args.save_blend.resolve()
        out_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))


if __name__ == "__main__":
    main()
