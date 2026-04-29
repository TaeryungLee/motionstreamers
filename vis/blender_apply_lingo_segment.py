from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a LINGO segment clip to an imported scene mesh in Blender.")
    parser.add_argument("--motion-pkl", type=Path, required=True)
    parser.add_argument("--scene-obj", type=Path, required=True)
    parser.add_argument("--object-name", default="SMPLX-mesh-male")
    parser.add_argument("--smplx-gender", default="male", choices=["female", "male", "neutral"])
    parser.add_argument("--render-mode", default="preview", choices=["preview", "final"])
    parser.add_argument("--camera-mode", default="oblique", choices=["oblique", "topdown"])
    parser.add_argument("--camera-scale", type=float, default=1.0)
    parser.add_argument("--bright-preview", action="store_true")
    parser.add_argument("--character-color", default=None)
    parser.add_argument("--save-blend", type=Path, default=None)
    parser.add_argument("--render-mp4", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args(argv)


def clear_default_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def ensure_world() -> bpy.types.World:
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    background = nodes.get("Background")
    if background is not None:
        background.inputs[0].default_value = (0.04, 0.04, 0.04, 1.0)
        background.inputs[1].default_value = 1.0
    return scene.world


def import_scene_mesh(scene_obj_path: Path) -> list[bpy.types.Object]:
    scene_obj_path = scene_obj_path.resolve()
    before = {obj.name for obj in bpy.data.objects}
    if hasattr(bpy.ops.wm, "obj_import"):
        result = bpy.ops.wm.obj_import(filepath=str(scene_obj_path))
    else:
        result = bpy.ops.import_scene.obj(filepath=str(scene_obj_path))
    if "FINISHED" not in set(result):
        raise RuntimeError(f"Failed to import scene mesh: {scene_obj_path}")

    imported = [obj for obj in bpy.data.objects if obj.name not in before and obj.type == "MESH"]
    if not imported:
        raise RuntimeError(f"OBJ import succeeded but no mesh objects were created: {scene_obj_path}")
    return imported


def apply_scene_material(meshes: list[bpy.types.Object]) -> None:
    material = bpy.data.materials.get("CodexLingoSceneMaterial")
    if material is None:
        material = bpy.data.materials.new(name="CodexLingoSceneMaterial")
        material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (300, 0)
    shader = nodes.new(type="ShaderNodeBsdfPrincipled")
    shader.location = (0, 0)
    shader.inputs["Base Color"].default_value = (0.62, 0.64, 0.67, 1.0)
    shader.inputs["Roughness"].default_value = 0.9
    if "Specular IOR Level" in shader.inputs:
        shader.inputs["Specular IOR Level"].default_value = 0.15
    elif "Specular" in shader.inputs:
        shader.inputs["Specular"].default_value = 0.15
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(material)


def world_bbox_points(obj: bpy.types.Object) -> list[Vector]:
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]


def compute_bounds(meshes: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    mins = Vector((float("inf"), float("inf"), float("inf")))
    maxs = Vector((float("-inf"), float("-inf"), float("-inf")))
    for obj in meshes:
        for point in world_bbox_points(obj):
            mins.x = min(mins.x, point.x)
            mins.y = min(mins.y, point.y)
            mins.z = min(mins.z, point.z)
            maxs.x = max(maxs.x, point.x)
            maxs.y = max(maxs.y, point.y)
            maxs.z = max(maxs.z, point.z)
    return mins, maxs


def clip_scene_meshes_for_topdown(
    meshes: list[bpy.types.Object],
    clip_above_height: float = 1.7,
    wall_normal_z_threshold: float = 0.25,
) -> None:
    mins, maxs = compute_bounds(meshes)
    cut_height = mins.z + float(clip_above_height)
    cap_band = 0.08

    for obj in meshes:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        plane_point_world = Vector((0.0, 0.0, cut_height))
        plane_no_world = Vector((0.0, 0.0, 1.0))
        inv = obj.matrix_world.inverted()
        plane_point_local = inv @ plane_point_world
        plane_no_local = (inv.to_3x3() @ plane_no_world).normalized()
        bmesh.ops.bisect_plane(
            bm,
            geom=list(bm.verts) + list(bm.edges) + list(bm.faces),
            plane_co=plane_point_local,
            plane_no=plane_no_local,
            clear_outer=True,
            clear_inner=False,
        )
        bm.normal_update()
        rot_world = obj.matrix_world.to_3x3()
        wall_faces = []
        cap_faces = []
        for face in bm.faces:
            normal_world = (rot_world @ face.normal).normalized()
            center_world = obj.matrix_world @ face.calc_center_median()
            if float(center_world.z) >= (cut_height - cap_band) and float(normal_world.z) > 0.35:
                cap_faces.append(face)
                continue
            if abs(float(normal_world.z)) < float(wall_normal_z_threshold):
                wall_faces.append(face)
        if cap_faces:
            bmesh.ops.delete(bm, geom=cap_faces, context="FACES")
        if wall_faces:
            bmesh.ops.delete(bm, geom=wall_faces, context="FACES")
        loose_verts = [vert for vert in bm.verts if not vert.link_faces]
        if loose_verts:
            bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")
        bm.to_mesh(mesh)
        mesh.update()
        bm.free()


def ensure_oblique_camera(scene_meshes: list[bpy.types.Object], target_obj: bpy.types.Object, camera_scale: float) -> None:
    scene = bpy.context.scene
    mins, maxs = compute_bounds(scene_meshes)
    center = (mins + maxs) * 0.5
    extent_x = maxs.x - mins.x
    extent_y = maxs.y - mins.y
    extent_z = maxs.z - mins.z
    radius = max(extent_x, extent_y, 2.0) * max(float(camera_scale), 0.3)
    height = max(extent_z, 2.0) * 1.1 + radius * 0.45

    camera = bpy.data.objects.get("CodexLingoCamera")
    if camera is None:
        cam_data = bpy.data.cameras.new("CodexLingoCamera")
        camera = bpy.data.objects.new("CodexLingoCamera", cam_data)
        scene.collection.objects.link(camera)

    focus = Vector(center)
    focus.z = mins.z + max(extent_z * 0.45, 1.0)
    camera.location = focus + Vector((-radius * 0.85, -radius * 1.05, height))
    direction = focus - Vector(camera.location)
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "PERSP"
    camera.data.lens = 28
    camera.data.clip_start = 0.01
    camera.data.clip_end = 1000.0
    scene.camera = camera


def ensure_topdown_camera(scene_meshes: list[bpy.types.Object], camera_scale: float) -> None:
    mins, maxs = compute_bounds(scene_meshes)
    center = (mins + maxs) * 0.5
    extent_x = maxs.x - mins.x
    extent_y = maxs.y - mins.y
    ortho_scale = max(extent_x, extent_y, 2.0) * 1.1 * max(float(camera_scale), 0.3)

    camera = bpy.data.objects.get("CodexLingoTopDownCamera")
    if camera is None:
        cam_data = bpy.data.cameras.new("CodexLingoTopDownCamera")
        camera = bpy.data.objects.new("CodexLingoTopDownCamera", cam_data)
        bpy.context.scene.collection.objects.link(camera)

    camera.data.type = "ORTHO"
    camera.data.ortho_scale = ortho_scale
    camera.data.clip_start = 0.01
    camera.data.clip_end = 1000.0
    camera.location = center + Vector((0.0, 0.0, max(8.0, (maxs.z - mins.z) + 6.0)))
    camera.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = camera


def ensure_preview_lighting(scene_meshes: list[bpy.types.Object]) -> None:
    mins, maxs = compute_bounds(scene_meshes)
    center = (mins + maxs) * 0.5
    height = maxs.z + 6.0

    sun = bpy.data.objects.get("CodexLingoSun")
    if sun is None:
        light_data = bpy.data.lights.new(name="CodexLingoSun", type="SUN")
        sun = bpy.data.objects.new("CodexLingoSun", light_data)
        bpy.context.scene.collection.objects.link(sun)
    sun.data.energy = 2.8
    sun.rotation_euler = (0.7, 0.0, 0.8)

    area = bpy.data.objects.get("CodexLingoArea")
    if area is None:
        light_data = bpy.data.lights.new(name="CodexLingoArea", type="AREA")
        area = bpy.data.objects.new("CodexLingoArea", light_data)
        bpy.context.scene.collection.objects.link(area)
    area.data.energy = 2800.0
    area.data.shape = "RECTANGLE"
    area.data.size = 12.0
    area.data.size_y = 12.0
    area.location = center + Vector((0.0, 0.0, height))
    area.rotation_euler = (0.0, 0.0, 0.0)


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
        configure_final_render,
        configure_preview_render,
        ensure_local_smplx_addon,
        get_or_create_smplx_object,
    )
    from visualize_smplx_motion.load_smplx_animatioin_clear import load_smplx_animation_new

    clear_default_scene()
    ensure_world()
    ensure_local_smplx_addon(project_root)

    scene_meshes = import_scene_mesh(args.scene_obj)
    apply_scene_material(scene_meshes)

    obj = get_or_create_smplx_object(args.object_name, args.smplx_gender)
    apply_character_color(obj, args.character_color)

    load_smplx_animation_new(str(args.motion_pkl), obj, load_hand=False, load_betas=False)

    scene = bpy.context.scene
    clip_scene_meshes_for_topdown(scene_meshes)
    if args.camera_mode == "topdown":
        ensure_topdown_camera(scene_meshes, args.camera_scale)
    else:
        ensure_oblique_camera(scene_meshes, obj, args.camera_scale)
    if args.bright_preview:
        ensure_preview_lighting(scene_meshes)

    with args.motion_pkl.open("rb") as handle:
        payload = pickle.load(handle)
    num_frames = int(payload["transl"].shape[0])
    scene.frame_start = 0
    scene.frame_end = max(0, num_frames - 1)
    if hasattr(scene, "frame_step"):
        scene.frame_step = 1
    if hasattr(scene.render, "frame_map_old"):
        scene.render.frame_map_old = 1
    if hasattr(scene.render, "frame_map_new"):
        scene.render.frame_map_new = 1
    scene.render.fps = int(args.fps)

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
        print(
            "Render config:",
            {
                "mode": args.render_mode,
                "camera_mode": args.camera_mode,
                "engine": scene.render.engine,
                "resolution": (
                    scene.render.resolution_x,
                    scene.render.resolution_y,
                    scene.render.resolution_percentage,
                ),
                "fps": scene.render.fps,
            },
        )
        bpy.ops.render.render(animation=True)

    if args.save_blend is not None:
        out_blend = args.save_blend.resolve()
        out_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))


if __name__ == "__main__":
    main()
