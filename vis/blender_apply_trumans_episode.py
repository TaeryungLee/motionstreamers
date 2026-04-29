from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a multi-character TRUMANS episode to a Blender scene.")
    parser.add_argument("--meta-json", type=Path, required=True)
    parser.add_argument("--render-mode", default="preview", choices=["preview", "final"])
    parser.add_argument("--camera-mode", default="scene", choices=["scene", "topdown", "oblique"])
    parser.add_argument("--camera-scale", type=float, default=1.0)
    parser.add_argument("--show-targets", action="store_true")
    parser.add_argument("--bright-preview", action="store_true")
    parser.add_argument("--clear-existing-characters", action="store_true")
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


def character_focus(obj: bpy.types.Object) -> Vector:
    focus_obj = obj.parent if obj.parent is not None else obj
    focus = Vector(focus_obj.matrix_world.translation)
    focus.z += 1.0
    return focus


def average_focus(objs: list[bpy.types.Object]) -> Vector:
    if not objs:
        return Vector((0.0, 0.0, 1.0))
    focus = Vector((0.0, 0.0, 0.0))
    for obj in objs:
        focus += character_focus(obj)
    return focus / float(len(objs))


def apply_camera_scale_to_targets(scene: bpy.types.Scene, targets: list[bpy.types.Object], camera_scale: float) -> None:
    camera = scene.camera
    if camera is None or not targets:
        return
    focus = average_focus(targets)
    cam_loc = Vector(camera.location)
    offset = cam_loc - focus
    if offset.length < 1e-6:
        return
    camera.location = focus + (offset * float(camera_scale))
    direction = focus - Vector(camera.location)
    if direction.length >= 1e-6:
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    if camera.data is not None:
        camera.data.clip_start = min(float(camera.data.clip_start), 0.01)
        camera.data.clip_end = max(float(camera.data.clip_end), 1000.0)


def ensure_bright_preview_lighting_for_targets(targets: list[bpy.types.Object]) -> None:
    scene = bpy.context.scene
    focus = average_focus(targets)

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    if scene.world.node_tree is not None:
        nodes = scene.world.node_tree.nodes
        links = scene.world.node_tree.links
        nodes.clear()
        output = nodes.new(type="ShaderNodeOutputWorld")
        output.location = (300, 0)
        background = nodes.new(type="ShaderNodeBackground")
        background.location = (0, 0)
        background.inputs[0].default_value = (0.18, 0.18, 0.18, 1.0)
        background.inputs[1].default_value = 0.9
        links.new(background.outputs["Background"], output.inputs["Surface"])

    sun = bpy.data.objects.get("CodexPreviewSun")
    if sun is None:
        light_data = bpy.data.lights.new(name="CodexPreviewSun", type="SUN")
        sun = bpy.data.objects.new("CodexPreviewSun", light_data)
        scene.collection.objects.link(sun)
    sun.data.energy = 2.5
    sun.rotation_euler = (0.7, 0.0, 0.8)

    area = bpy.data.objects.get("CodexPreviewArea")
    if area is None:
        light_data = bpy.data.lights.new(name="CodexPreviewArea", type="AREA")
        area = bpy.data.objects.new("CodexPreviewArea", light_data)
        scene.collection.objects.link(area)
    area.data.energy = 2600.0
    area.data.shape = "RECTANGLE"
    area.data.size = 12.0
    area.data.size_y = 12.0
    area.location = focus + Vector((0.0, 0.0, 8.0))
    area.rotation_euler = (0.0, 0.0, 0.0)

    if hasattr(scene, "view_settings"):
        scene.view_settings.exposure = 1.1


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

    from blender_apply_smplx_segment import (
        apply_character_color,
        average_focus,
        clear_existing_characters,
        clip_scene_meshes_for_debug,
        collect_scene_mesh_objects,
        configure_final_render,
        configure_preview_render,
        create_target_overlays,
        crop_scene_meshes_for_debug,
        create_new_smplx_object,
        ensure_local_smplx_addon,
        get_or_create_smplx_object,
        hide_oblique_occluders,
        hide_topdown_occluders,
        prepare_topdown_floor_debug_view,
        setup_oblique_camera,
        setup_topdown_camera,
    )
    from visualize_smplx_motion.load_smplx_animatioin_clear import load_smplx_animation_new

    meta = load_json(args.meta_json.resolve())
    scene_id = str(meta.get("scene_id") or "")
    characters = meta.get("characters", [])
    if not characters:
        raise ValueError(f"No characters found in {args.meta_json}")
    gender = str(meta.get("smplx_gender", "male"))

    ensure_local_smplx_addon(project_root)
    if args.clear_existing_characters:
        clear_existing_characters()

    created_objects: list[bpy.types.Object] = []
    max_frames = 0
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
            load_hand=bool(char.get("load_hand", True)),
            load_betas=False,
        )
        created_objects.append(obj)
        max_frames = max(max_frames, int(char.get("num_frames", 0)))

    scene = bpy.context.scene
    character_roots = [obj.parent if obj.parent is not None else obj for obj in created_objects]
    scene_meshes = collect_scene_mesh_objects(excluded_roots=character_roots)
    focus = average_focus(created_objects)
    if args.camera_mode == "topdown":
        clip_scene_meshes_for_debug(scene_meshes)
        crop_scene_meshes_for_debug(scene_meshes, focus)
        hide_topdown_occluders()
        prepare_topdown_floor_debug_view()
        setup_topdown_camera(scene, created_objects, args.camera_scale, scene_id=scene_id)
    elif args.camera_mode == "oblique":
        clip_scene_meshes_for_debug(scene_meshes)
        crop_scene_meshes_for_debug(scene_meshes, focus)
        hide_oblique_occluders()
        setup_oblique_camera(scene, created_objects, args.camera_scale, scene_id=scene_id)
    else:
        apply_camera_scale_to_targets(scene, created_objects, args.camera_scale)
    if args.bright_preview:
        ensure_bright_preview_lighting_for_targets(created_objects)

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
