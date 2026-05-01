from __future__ import annotations

import argparse
import sys
from math import atan2, cos, radians, sin
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply an exported SMPL-X segment clip to a Blender scene.")
    parser.add_argument("--motion-pkl", type=Path, required=True)
    parser.add_argument("--object-name", default="SMPLX-mesh-male")
    parser.add_argument("--smplx-gender", default="male", choices=["female", "male", "neutral"])
    parser.add_argument("--render-mode", default="preview", choices=["preview", "final"])
    parser.add_argument("--camera-mode", default="scene", choices=["scene", "topdown", "oblique"])
    parser.add_argument("--camera-scale", type=float, default=1.0)
    parser.add_argument("--bright-preview", action="store_true")
    parser.add_argument("--character-color", default=None)
    parser.add_argument("--clear-existing-characters", action="store_true")
    parser.add_argument("--save-blend", type=Path, default=None)
    parser.add_argument("--render-mp4", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args(argv)


def ensure_local_smplx_addon(project_root: Path) -> None:
    sys.path.insert(0, str(project_root))
    import smplx_blender_addon

    if not hasattr(bpy.types.WindowManager, "smplx_tool"):
        smplx_blender_addon.register()


def configure_smplx_tool(gender: str) -> None:
    wm = bpy.context.window_manager
    if not hasattr(wm, "smplx_tool"):
        raise RuntimeError("SMPL-X addon is not registered in this Blender session.")

    wm.smplx_tool.smplx_gender = gender
    if hasattr(wm.smplx_tool, "smplx_version"):
        if "v1.1" in {item.identifier for item in wm.smplx_tool.bl_rna.properties["smplx_version"].enum_items}:
            wm.smplx_tool.smplx_version = "v1.1"
    if hasattr(wm.smplx_tool, "smplx_uv"):
        if "UV_2023" in {item.identifier for item in wm.smplx_tool.bl_rna.properties["smplx_uv"].enum_items}:
            wm.smplx_tool.smplx_uv = "UV_2023"
    if hasattr(wm.smplx_tool, "smplx_handpose"):
        wm.smplx_tool.smplx_handpose = "relaxed"


def create_new_smplx_object(gender: str) -> bpy.types.Object:
    before = {item.name for item in bpy.data.objects}
    configure_smplx_tool(gender)
    result = bpy.ops.scene.smplx_add_gender("EXEC_DEFAULT")
    if "FINISHED" not in set(result):
        raise RuntimeError(f"Failed to create SMPL-X object via addon: {result}")

    obj = bpy.context.active_object
    if obj is not None:
        return obj

    target_prefix = f"SMPLX-mesh-{gender}"
    candidates = [
        item
        for item in bpy.data.objects
        if item.name not in before and (item.name == target_prefix or item.name.startswith(f"{target_prefix}."))
    ]
    if not candidates:
        raise RuntimeError(f"SMPL-X addon ran but no object matching '{target_prefix}' was created.")
    return sorted(candidates, key=lambda item: item.name)[-1]


def get_or_create_smplx_object(object_name: str, gender: str) -> bpy.types.Object:
    obj = bpy.data.objects.get(object_name)
    if obj is not None:
        return obj

    target_prefix = f"SMPLX-mesh-{gender}"
    candidates = [item for item in bpy.data.objects if item.name == target_prefix or item.name.startswith(f"{target_prefix}.")]
    if candidates:
        return sorted(candidates, key=lambda item: item.name)[0]

    return create_new_smplx_object(gender)


def parse_rgba(color_spec: str | None) -> tuple[float, float, float, float] | None:
    if not color_spec:
        return None
    text = color_spec.strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) in {6, 8} and all(ch in "0123456789abcdefABCDEF" for ch in text):
        if len(text) == 6:
            text += "FF"
        vals = [int(text[i : i + 2], 16) / 255.0 for i in range(0, 8, 2)]
        return (vals[0], vals[1], vals[2], vals[3])
    parts = [part.strip() for part in text.split(",")]
    if len(parts) in {3, 4}:
        vals = [float(part) for part in parts]
        if len(vals) == 3:
            vals.append(1.0)
        if max(vals) > 1.0:
            vals = [min(max(v / 255.0, 0.0), 1.0) for v in vals]
        else:
            vals = [min(max(v, 0.0), 1.0) for v in vals]
        return (vals[0], vals[1], vals[2], vals[3])
    raise ValueError(f"Unsupported color format: {color_spec}")


def iter_descendants(obj: bpy.types.Object) -> list[bpy.types.Object]:
    stack = list(obj.children)
    out: list[bpy.types.Object] = []
    while stack:
        cur = stack.pop()
        out.append(cur)
        stack.extend(cur.children)
    return out


def iter_character_meshes(obj: bpy.types.Object) -> list[bpy.types.Object]:
    armature = obj.parent if obj.parent is not None and obj.parent.type == "ARMATURE" else None
    roots = [obj]
    if armature is not None:
        roots.extend(iter_descendants(armature))

    meshes: list[bpy.types.Object] = []
    seen: set[str] = set()
    for item in roots:
        if item.type != "MESH":
            continue
        if item.name in seen:
            continue
        seen.add(item.name)
        meshes.append(item)
    return meshes


def apply_character_color(
    obj: bpy.types.Object,
    color_spec: str | None,
    material_name: str = "CodexCharacterMaterial",
) -> None:
    rgba = parse_rgba(color_spec)
    if rgba is None:
        return

    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (300, 0)
    shader = nodes.new(type="ShaderNodeBsdfPrincipled")
    shader.location = (0, 0)
    shader.inputs["Base Color"].default_value = rgba
    shader.inputs["Roughness"].default_value = 0.65
    if "Specular IOR Level" in shader.inputs:
        shader.inputs["Specular IOR Level"].default_value = 0.2
    elif "Specular" in shader.inputs:
        shader.inputs["Specular"].default_value = 0.2
    if "Emission" in shader.inputs:
        shader.inputs["Emission"].default_value = rgba
    if "Emission Color" in shader.inputs:
        shader.inputs["Emission Color"].default_value = rgba
    if "Emission Strength" in shader.inputs:
        shader.inputs["Emission Strength"].default_value = 0.8
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    material.blend_method = "OPAQUE"
    material.shadow_method = "OPAQUE"

    for mesh_obj in iter_character_meshes(obj):
        mesh_obj.data.materials.clear()
        mesh_obj.data.materials.append(material)


def apply_debug_material_to_prefixes(
    prefixes: tuple[str, ...],
    material_name: str,
    rgba: tuple[float, float, float, float],
    roughness: float = 0.9,
) -> None:
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (300, 0)
    shader = nodes.new(type="ShaderNodeBsdfPrincipled")
    shader.location = (0, 0)
    shader.inputs["Base Color"].default_value = rgba
    shader.inputs["Roughness"].default_value = float(roughness)
    if "Specular IOR Level" in shader.inputs:
        shader.inputs["Specular IOR Level"].default_value = 0.12
    elif "Specular" in shader.inputs:
        shader.inputs["Specular"].default_value = 0.12
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not obj.name.startswith(prefixes):
            continue
        obj.data.materials.clear()
        obj.data.materials.append(material)


def is_existing_character_root(obj: bpy.types.Object) -> bool:
    if obj.type != "ARMATURE":
        return False
    if obj.name.startswith("SMPLX-"):
        return False
    child_names = {child.name for child in obj.children}
    if any(name.startswith("CC_Base_") for name in child_names):
        return True
    if obj.name.startswith("zzy"):
        return True
    return False


def clear_existing_characters() -> None:
    roots = [obj for obj in bpy.data.objects if is_existing_character_root(obj)]
    if not roots:
        return

    to_remove: list[bpy.types.Object] = []
    seen: set[str] = set()
    for root in roots:
        for obj in [root, *iter_descendants(root)]:
            if obj.name in seen:
                continue
            seen.add(obj.name)
            to_remove.append(obj)

    for obj in reversed(to_remove):
        bpy.data.objects.remove(obj, do_unlink=True)


def freeze_scene_animation(frame: int = 0) -> None:
    scene = bpy.context.scene
    scene.frame_set(int(frame))
    protected_prefixes = ("SMPLX-", "Codex")
    for obj in bpy.data.objects:
        if obj.name.startswith(protected_prefixes):
            continue
        obj.animation_data_clear()
        if obj.data is not None:
            obj.data.animation_data_clear()
            shape_keys = getattr(obj.data, "shape_keys", None)
            if shape_keys is not None:
                shape_keys.animation_data_clear()


def apply_camera_scale(scene: bpy.types.Scene, target_obj: bpy.types.Object, camera_scale: float) -> None:
    camera = scene.camera
    if camera is None:
        return

    focus_obj = target_obj.parent if target_obj.parent is not None else target_obj
    focus = Vector(focus_obj.matrix_world.translation)
    focus.z += 1.0
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


def world_bbox_points(obj: bpy.types.Object) -> list[Vector]:
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]


def collect_scene_mesh_objects(excluded_roots: list[bpy.types.Object] | None = None) -> list[bpy.types.Object]:
    excluded_names: set[str] = set()
    if excluded_roots is not None:
        for root in excluded_roots:
            excluded_names.add(root.name)
            for child in iter_descendants(root):
                excluded_names.add(child.name)

    meshes: list[bpy.types.Object] = []
    for obj in bpy.data.objects:
        if obj.hide_render or obj.type != "MESH":
            continue
        if obj.name in excluded_names:
            continue
        meshes.append(obj)
    return meshes


def compute_scene_bounds(excluded_roots: list[bpy.types.Object] | None = None) -> tuple[Vector, Vector] | None:
    meshes = collect_scene_mesh_objects(excluded_roots=excluded_roots)
    if not meshes:
        return None

    mins = Vector((float("inf"), float("inf"), float("inf")))
    maxs = Vector((float("-inf"), float("-inf"), float("-inf")))
    found = False
    for obj in meshes:
        for point in world_bbox_points(obj):
            mins.x = min(mins.x, point.x)
            mins.y = min(mins.y, point.y)
            mins.z = min(mins.z, point.z)
            maxs.x = max(maxs.x, point.x)
            maxs.y = max(maxs.y, point.y)
            maxs.z = max(maxs.z, point.z)
            found = True
    if not found:
        return None
    return mins, maxs


def compute_bounds_for_meshes(meshes: list[bpy.types.Object]) -> tuple[Vector, Vector] | None:
    if not meshes:
        return None

    mins = Vector((float("inf"), float("inf"), float("inf")))
    maxs = Vector((float("-inf"), float("-inf"), float("-inf")))
    found = False
    for obj in meshes:
        if obj.type != "MESH" or obj.hide_render:
            continue
        for point in world_bbox_points(obj):
            mins.x = min(mins.x, point.x)
            mins.y = min(mins.y, point.y)
            mins.z = min(mins.z, point.z)
            maxs.x = max(maxs.x, point.x)
            maxs.y = max(maxs.y, point.y)
            maxs.z = max(maxs.z, point.z)
            found = True
    if not found:
        return None
    return mins, maxs


def bbox_intersects_crop(obj: bpy.types.Object, crop_mins: Vector, crop_maxs: Vector) -> bool:
    points = world_bbox_points(obj)
    obj_mins = Vector(
        (
            min(point.x for point in points),
            min(point.y for point in points),
            min(point.z for point in points),
        )
    )
    obj_maxs = Vector(
        (
            max(point.x for point in points),
            max(point.y for point in points),
            max(point.z for point in points),
        )
    )
    return not (
        obj_maxs.x < crop_mins.x
        or obj_mins.x > crop_maxs.x
        or obj_maxs.y < crop_mins.y
        or obj_mins.y > crop_maxs.y
        or obj_maxs.z < crop_mins.z
        or obj_mins.z > crop_maxs.z
    )


def point_to_bbox_xy_distance(point: Vector, obj: bpy.types.Object) -> float:
    points = world_bbox_points(obj)
    obj_min_x = min(p.x for p in points)
    obj_max_x = max(p.x for p in points)
    obj_min_y = min(p.y for p in points)
    obj_max_y = max(p.y for p in points)
    dx = max(obj_min_x - point.x, 0.0, point.x - obj_max_x)
    dy = max(obj_min_y - point.y, 0.0, point.y - obj_max_y)
    return float((dx * dx + dy * dy) ** 0.5)


def compute_room_crop_bounds(meshes: list[bpy.types.Object], focus: Vector) -> tuple[Vector, Vector] | None:
    floor_candidates = [obj for obj in meshes if obj.name.startswith("Floor")]
    if floor_candidates:
        floor_candidates = sorted(floor_candidates, key=lambda obj: point_to_bbox_xy_distance(focus, obj))
        min_dist = point_to_bbox_xy_distance(focus, floor_candidates[0])
        selected = [obj for obj in floor_candidates if point_to_bbox_xy_distance(focus, obj) <= (min_dist + 0.6)]
        bounds = compute_bounds_for_meshes(selected)
        if bounds is not None:
            mins, maxs = bounds
            margin_xy = 0.8
            margin_z_low = 0.5
            margin_z_high = 2.4
            return (
                Vector((mins.x - margin_xy, mins.y - margin_xy, mins.z - margin_z_low)),
                Vector((maxs.x + margin_xy, maxs.y + margin_xy, mins.z + margin_z_high)),
            )

    bounds = compute_bounds_for_meshes(meshes)
    if bounds is None:
        return None
    mins, maxs = bounds
    half_extent = max(maxs.x - mins.x, maxs.y - mins.y, 4.0) * 0.22
    return (
        Vector((focus.x - half_extent, focus.y - half_extent, mins.z - 0.5)),
        Vector((focus.x + half_extent, focus.y + half_extent, mins.z + 2.4)),
    )


def hide_meshes_outside_crop(meshes: list[bpy.types.Object], crop_mins: Vector, crop_maxs: Vector) -> None:
    for obj in meshes:
        if obj.type != "MESH":
            continue
        if not bbox_intersects_crop(obj, crop_mins, crop_maxs):
            obj.hide_render = True
            obj.hide_viewport = True


def clip_scene_meshes_for_debug(
    meshes: list[bpy.types.Object],
    clip_above_height: float = 1.7,
    wall_normal_z_threshold: float = 0.25,
) -> None:
    bounds = compute_bounds_for_meshes(meshes)
    if bounds is None:
        return

    mins, _ = bounds
    cut_height = mins.z + float(clip_above_height)
    cap_band = 0.08

    for obj in meshes:
        if obj.type != "MESH" or obj.hide_render:
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


def crop_scene_meshes_for_debug(meshes: list[bpy.types.Object], focus: Vector) -> None:
    crop_bounds = compute_room_crop_bounds(meshes, focus)
    if crop_bounds is None:
        return
    crop_mins, crop_maxs = crop_bounds
    hide_meshes_outside_crop(meshes, crop_mins, crop_maxs)


def average_focus(target_objs: list[bpy.types.Object]) -> Vector:
    if not target_objs:
        return Vector((0.0, 0.0, 1.0))
    focus = Vector((0.0, 0.0, 0.0))
    for obj in target_objs:
        focus_obj = obj.parent if obj.parent is not None else obj
        cur = Vector(focus_obj.matrix_world.translation)
        cur.z += 1.0
        focus += cur
    return focus / float(len(target_objs))


def trumans_scene_camera_overrides(scene_id: str | None) -> dict[str, float]:
    prefix = str(scene_id or "").lower()[:3]

    topdown_zoom_in = {"0ab", "0ac", "1d5", "1e1", "2d3"}
    topdown_zoom_out = {"2a6", "2a7", "2b4", "3a5", "4a8", "4a0", "4ac", "5a2"}
    topdown_shift_down = {"0ab", "0ac", "1d5", "2b4", "3a5", "4a8", "4a0"}
    oblique_look_down = {"0ab", "0ac", "1d5", "1e1", "2a6", "2a7", "2b4", "2d3", "3a5", "4a8", "4a0", "4ac", "5a2"}
    oblique_zoom_in = {"0ab", "0ac", "1d5", "1e1", "2d3", "4a8", "4ac"}
    oblique_zoom_out = {"2a6", "2a7", "2b4", "3a5", "4a0"}

    topdown_scale_mult = 0.91 if prefix in topdown_zoom_in else (1.12 if prefix in topdown_zoom_out else 1.0)
    topdown_shift_mult = 1.95 if prefix in topdown_shift_down else 1.0
    oblique_look_down_mult = 1.85 if prefix in oblique_look_down else 1.0
    oblique_radius_mult = 0.89 if prefix in oblique_zoom_in else (1.12 if prefix in oblique_zoom_out else 1.0)

    extra_both_zoom_in = {"0ab", "0ac", "1d5", "1e1"}
    extra_oblique_look_down = {"2a6", "2a7", "3a5", "4a8", "4a0", "5a2"}
    extra_oblique_look_down_heavy = {"2b4"}
    extra_topdown_zoom_out = {"2a6", "2a7"}
    extra_topdown_shift_down = {"3a5", "4a8", "4a0"}
    extra_topdown_shift_down_heavy = {"2b4"}

    if prefix in extra_both_zoom_in:
        topdown_scale_mult *= 0.90
        oblique_radius_mult *= 0.90
    if prefix in extra_oblique_look_down:
        oblique_look_down_mult *= 1.24
    if prefix in extra_oblique_look_down_heavy:
        oblique_look_down_mult *= 1.72
    if prefix in extra_topdown_zoom_out:
        topdown_scale_mult *= 1.08
    if prefix in extra_topdown_shift_down:
        topdown_shift_mult *= 1.32
    if prefix in extra_topdown_shift_down_heavy:
        topdown_shift_mult *= 1.96

    third_both_zoom_in_heavy = {"1d5"}
    third_both_zoom_out = {"2a6", "2a7"}
    third_topdown_shift_down_more = {"4a0"}
    third_topdown_shift_down_heavy = {"2b4"}
    third_topdown_zoom_in = {"2b4"}
    third_oblique_look_down_more = {"4a0"}
    third_oblique_look_down_heavy = {"2b4", "3a5"}
    third_oblique_zoom_in = {"2b4", "3a5"}

    if prefix in third_both_zoom_in_heavy:
        topdown_scale_mult *= 0.90
        oblique_radius_mult *= 0.90
    if prefix in third_both_zoom_out:
        topdown_scale_mult *= 1.08
        oblique_radius_mult *= 1.08
    if prefix in third_topdown_shift_down_more:
        topdown_shift_mult *= 1.22
    if prefix in third_topdown_shift_down_heavy:
        topdown_shift_mult *= 1.45
    if prefix in third_topdown_zoom_in:
        topdown_scale_mult *= 0.92
    if prefix in third_oblique_look_down_more:
        oblique_look_down_mult *= 1.16
    if prefix in third_oblique_look_down_heavy:
        oblique_look_down_mult *= 1.38
    if prefix in third_oblique_zoom_in:
        oblique_radius_mult *= 0.92

    return {
        "topdown_scale_mult": topdown_scale_mult,
        "topdown_shift_mult": topdown_shift_mult,
        "oblique_look_down_mult": oblique_look_down_mult,
        "oblique_radius_mult": oblique_radius_mult,
    }


def setup_oblique_camera(
    scene: bpy.types.Scene,
    target_objs: list[bpy.types.Object],
    camera_scale: float,
    scene_id: str | None = None,
) -> None:
    if not target_objs:
        return

    overrides = trumans_scene_camera_overrides(scene_id)
    roots = [obj.parent if obj.parent is not None else obj for obj in target_objs]
    bounds = compute_scene_bounds(excluded_roots=roots)
    if bounds is None:
        focus = average_focus(target_objs)
        mins = focus - Vector((2.0, 2.0, 1.0))
        maxs = focus + Vector((2.0, 2.0, 2.0))
    else:
        mins, maxs = bounds
        focus = average_focus(target_objs)
        focus.z = mins.z + max((maxs.z - mins.z) * 0.45, 1.0)

    extent_x = maxs.x - mins.x
    extent_y = maxs.y - mins.y
    extent_z = max(maxs.z - mins.z, 2.0)
    base_scale = max(float(camera_scale), 0.3)
    orbit_radius = max(extent_x, extent_y, 2.5) * base_scale * 0.90 * float(overrides["oblique_radius_mult"])
    azimuth = atan2(-0.84, -0.66)
    elevation = radians(56.0)
    horizontal_radius = orbit_radius * cos(elevation)
    look_at = Vector(focus)
    look_down_amount = max(extent_z * 0.10, 0.18) * float(overrides["oblique_look_down_mult"])
    look_at.z -= look_down_amount

    camera = bpy.data.objects.get("CodexTrumansObliqueCamera")
    if camera is None:
        cam_data = bpy.data.cameras.new("CodexTrumansObliqueCamera")
        camera = bpy.data.objects.new("CodexTrumansObliqueCamera", cam_data)
        scene.collection.objects.link(camera)

    camera.location = focus + Vector(
        (
            horizontal_radius * cos(azimuth),
            horizontal_radius * sin(azimuth),
            max(orbit_radius * sin(elevation), extent_z * 0.87, 2.65),
        )
    )
    direction = look_at - Vector(camera.location)
    if direction.length >= 1e-6:
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "PERSP"
    camera.data.lens = 38
    camera.data.clip_start = 0.01
    camera.data.clip_end = 1000.0
    scene.camera = camera


def hide_topdown_occluders() -> None:
    occluder_prefixes = (
        "Ceiling",
        "SmartCustomizedCeiling",
        "ExtrusionCustomizedCeilingModel",
        "LightBand",
        "Cornice",
        "WallTop",
        "WallOuter",
        "WallInner",
        "Front",
        "Back",
        "Window",
        "Window_Object",
        "BayWindow",
        "Hole",
        "CustomizedPersonalizedModel",
        "CustomizedPlatform",
        "SmartCustomizedCeiling",
        "Flue",
    )
    for obj in bpy.data.objects:
        if obj.name.startswith(occluder_prefixes):
            obj.hide_render = True
            obj.hide_viewport = True


def hide_oblique_occluders() -> None:
    # Geometry clipping handles the heavy lifting; this only hides obvious shell / window overlays.
    occluder_prefixes = (
        "Ceiling",
        "SmartCustomizedCeiling",
        "ExtrusionCustomizedCeilingModel",
        "LightBand",
        "Cornice",
        "Front",
        "Door",
        "Window",
        "Window_Object",
        "BayWindow",
        "CustomizedPersonalizedModel",
        "CustomizedPlatform",
        "Hole",
        "Flue",
    )
    for obj in bpy.data.objects:
        if obj.name.startswith(occluder_prefixes):
            obj.hide_render = True
            obj.hide_viewport = True


def compute_topdown_room_bounds(target_objs: list[bpy.types.Object] | bpy.types.Object) -> tuple[Vector, float]:
    objs = target_objs if isinstance(target_objs, list) else [target_objs]
    roots = [obj.parent if obj.parent is not None else obj for obj in objs]
    bounds = compute_scene_bounds(excluded_roots=roots)
    focus = average_focus(objs)
    if bounds is None:
        return focus, 6.0
    mins, maxs = bounds

    center = (mins + maxs) * 0.5
    focus.z = center.z
    extent_x = maxs.x - mins.x
    extent_y = maxs.y - mins.y
    ortho_scale = max(extent_x, extent_y) * 1.0
    return focus, max(ortho_scale, 2.0)


def setup_topdown_camera(
    scene: bpy.types.Scene,
    target_objs: list[bpy.types.Object] | bpy.types.Object,
    camera_scale: float,
    scene_id: str | None = None,
) -> None:
    overrides = trumans_scene_camera_overrides(scene_id)
    focus, base_ortho_scale = compute_topdown_room_bounds(target_objs)
    base_scale = max(float(camera_scale), 0.25)
    radius = 12.0 * base_scale
    slide_angle = radians(0.0)
    sphere_offset = Vector((0.0, -radius * sin(slide_angle), radius * cos(slide_angle)))
    focus.y -= max(base_ortho_scale * 0.0225, 0.08) * float(overrides["topdown_shift_mult"])

    camera = bpy.data.objects.get("CodexTopDownCamera")
    if camera is None:
        cam_data = bpy.data.cameras.new("CodexTopDownCamera")
        camera = bpy.data.objects.new("CodexTopDownCamera", cam_data)
        scene.collection.objects.link(camera)

    camera.data.type = "ORTHO"
    camera.data.ortho_scale = base_ortho_scale * base_scale * 0.84 * float(overrides["topdown_scale_mult"])
    camera.data.clip_start = 0.01
    camera.data.clip_end = 1000.0
    camera.location = focus + sphere_offset
    camera.rotation_euler = (slide_angle, 0.0, 0.0)
    scene.camera = camera


def ensure_bright_preview_lighting(target_obj: bpy.types.Object) -> None:
    focus_obj = target_obj.parent if target_obj.parent is not None else target_obj
    focus = Vector(focus_obj.matrix_world.translation)

    scene = bpy.context.scene
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
        background.inputs[0].default_value = (0.62, 0.62, 0.62, 1.0)
        background.inputs[1].default_value = 0.8
        links.new(background.outputs["Background"], output.inputs["Surface"])

    sun = bpy.data.objects.get("CodexPreviewSun")
    if sun is None:
        light_data = bpy.data.lights.new(name="CodexPreviewSun", type="SUN")
        sun = bpy.data.objects.new("CodexPreviewSun", light_data)
        bpy.context.scene.collection.objects.link(sun)
    sun.data.energy = 2.5
    sun.rotation_euler = (0.7, 0.0, 0.8)

    area = bpy.data.objects.get("CodexPreviewArea")
    if area is None:
        light_data = bpy.data.lights.new(name="CodexPreviewArea", type="AREA")
        area = bpy.data.objects.new("CodexPreviewArea", light_data)
        bpy.context.scene.collection.objects.link(area)
    area.data.energy = 2600.0
    area.data.shape = "RECTANGLE"
    area.data.size = 10.0
    area.data.size_y = 10.0
    area.location = focus + Vector((0.0, 0.0, 7.5))
    area.rotation_euler = (0.0, 0.0, 0.0)

    if hasattr(scene, "view_settings"):
        scene.view_settings.exposure = 1.1


def prepare_topdown_floor_debug_view() -> None:
    return


def smpl_point_to_blender(point: list[float] | tuple[float, float, float]) -> Vector:
    return Vector((float(point[0]), -float(point[2]), float(point[1])))


def ensure_overlay_material(
    material_name: str,
    color_spec: str | None,
    fallback_rgba: tuple[float, float, float, float] = (1.0, 0.6, 0.2, 1.0),
) -> bpy.types.Material:
    rgba = parse_rgba(color_spec) or fallback_rgba

    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (300, 0)
    shader = nodes.new(type="ShaderNodeBsdfPrincipled")
    shader.location = (0, 0)
    shader.inputs["Base Color"].default_value = rgba
    shader.inputs["Roughness"].default_value = 0.35
    if "Specular IOR Level" in shader.inputs:
        shader.inputs["Specular IOR Level"].default_value = 0.18
    elif "Specular" in shader.inputs:
        shader.inputs["Specular"].default_value = 0.18
    if "Emission" in shader.inputs:
        shader.inputs["Emission"].default_value = rgba
    if "Emission Color" in shader.inputs:
        shader.inputs["Emission Color"].default_value = rgba
    if "Emission Strength" in shader.inputs:
        shader.inputs["Emission Strength"].default_value = 1.6
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    material.blend_method = "OPAQUE"
    material.shadow_method = "OPAQUE"
    return material


def ensure_overlay_collection(name: str = "CodexTargetOverlays") -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def clear_overlay_collection(collection: bpy.types.Collection) -> None:
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def move_object_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    if collection not in obj.users_collection:
        collection.objects.link(obj)
    for other in list(obj.users_collection):
        if other != collection:
            other.objects.unlink(obj)


def keyframe_overlay_visibility(
    obj: bpy.types.Object,
    visible_start: int,
    visible_end: int,
    scene_start: int,
    scene_end: int,
) -> None:
    visible_start = max(int(scene_start), int(visible_start))
    visible_end = min(int(scene_end), int(visible_end))
    if visible_end < visible_start:
        obj.hide_viewport = True
        obj.hide_render = True
        return

    hidden_before = max(int(scene_start), visible_start - 1)
    hidden_after = min(int(scene_end), visible_end + 1)

    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_viewport", frame=int(scene_start))
    obj.keyframe_insert(data_path="hide_render", frame=int(scene_start))

    if hidden_before >= scene_start and hidden_before < visible_start:
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=int(hidden_before))
        obj.keyframe_insert(data_path="hide_render", frame=int(hidden_before))

    obj.hide_viewport = False
    obj.hide_render = False
    obj.keyframe_insert(data_path="hide_viewport", frame=int(visible_start))
    obj.keyframe_insert(data_path="hide_render", frame=int(visible_start))
    obj.keyframe_insert(data_path="hide_viewport", frame=int(visible_end))
    obj.keyframe_insert(data_path="hide_render", frame=int(visible_end))

    if hidden_after <= scene_end and hidden_after > visible_end:
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=int(hidden_after))
        obj.keyframe_insert(data_path="hide_render", frame=int(hidden_after))


def create_target_overlays(
    overlays: list[dict],
    marker_radius: float = 0.21,
) -> None:
    if not overlays:
        return

    scene = bpy.context.scene
    collection = ensure_overlay_collection()
    clear_overlay_collection(collection)

    for overlay_idx, overlay in enumerate(overlays):
        position = overlay.get("position")
        if not isinstance(position, list) or len(position) != 3:
            continue

        base_loc = smpl_point_to_blender(position)

        marker_name = f"CodexTargetMarker_{overlay_idx:03d}"
        bpy.ops.mesh.primitive_uv_sphere_add(radius=marker_radius, location=base_loc)
        marker = bpy.context.active_object
        marker.name = marker_name
        move_object_to_collection(marker, collection)
        marker_material = ensure_overlay_material(
            material_name=f"CodexTargetMaterial_{overlay.get('character_id', 'char')}",
            color_spec=overlay.get("color"),
        )
        if marker.data is not None:
            marker.data.materials.clear()
            marker.data.materials.append(marker_material)
        active_start = int(overlay.get("active_frame_start", 0))
        active_end = int(overlay.get("active_frame_end", active_start))
        keyframe_overlay_visibility(
            marker,
            visible_start=active_start,
            visible_end=active_end,
            scene_start=int(scene.frame_start),
            scene_end=int(scene.frame_end),
        )


def configure_preview_render(scene: bpy.types.Scene, resolution_percentage: int = 50) -> None:
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = int(resolution_percentage)
    scene.render.use_motion_blur = False
    if hasattr(scene.render, "use_compositing"):
        scene.render.use_compositing = False
    if hasattr(scene.render, "use_sequencer"):
        scene.render.use_sequencer = False
    if hasattr(scene.render, "film_transparent"):
        scene.render.film_transparent = False
    scene.eevee.taa_render_samples = 8
    scene.eevee.taa_samples = 8
    if hasattr(scene, "view_settings"):
        scene.view_settings.exposure = max(float(scene.view_settings.exposure), 0.6)
    if hasattr(scene.eevee, "use_bloom"):
        scene.eevee.use_bloom = False
    if hasattr(scene.eevee, "use_gtao"):
        scene.eevee.use_gtao = False
    if hasattr(scene.eevee, "use_ssr"):
        scene.eevee.use_ssr = False
    if hasattr(scene.eevee, "shadow_cube_size"):
        scene.eevee.shadow_cube_size = "256"
    if hasattr(scene.eevee, "shadow_cascade_size"):
        scene.eevee.shadow_cascade_size = "256"


def configure_final_render(scene: bpy.types.Scene) -> None:
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    if hasattr(scene.render, "use_compositing"):
        scene.render.use_compositing = False
    if hasattr(scene.render, "use_sequencer"):
        scene.render.use_sequencer = False
    if hasattr(scene, "cycles"):
        scene.cycles.samples = 256
        if hasattr(scene.cycles, "use_denoising"):
            scene.cycles.use_denoising = True


def main() -> None:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    args = parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    ensure_local_smplx_addon(project_root)
    from visualize_smplx_motion.load_smplx_animatioin_clear import load_smplx_animation_new

    if args.clear_existing_characters:
        clear_existing_characters()

    obj = get_or_create_smplx_object(args.object_name, args.smplx_gender)
    apply_character_color(obj, args.character_color)

    load_smplx_animation_new(str(args.motion_pkl), obj, load_hand=True, load_betas=False)
    character_root = obj.parent if obj.parent is not None else obj
    scene_meshes = collect_scene_mesh_objects(excluded_roots=[character_root])
    if args.camera_mode == "topdown":
        clip_scene_meshes_for_debug(scene_meshes)
        crop_scene_meshes_for_debug(scene_meshes, average_focus([obj]))
        hide_topdown_occluders()
        prepare_topdown_floor_debug_view()
        setup_topdown_camera(bpy.context.scene, obj, args.camera_scale)
    elif args.camera_mode == "oblique":
        clip_scene_meshes_for_debug(scene_meshes)
        crop_scene_meshes_for_debug(scene_meshes, average_focus([obj]))
        hide_oblique_occluders()
        setup_oblique_camera(bpy.context.scene, [obj], args.camera_scale)
    else:
        apply_camera_scale(bpy.context.scene, obj, args.camera_scale)
    if args.bright_preview:
        ensure_bright_preview_lighting(obj)

    import pickle

    with args.motion_pkl.open("rb") as handle:
        payload = pickle.load(handle)
    num_frames = int(payload["transl"].shape[0])
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = max(0, num_frames - 1)
    bpy.context.scene.render.fps = int(args.fps)

    if args.render_mp4 is not None:
        out_path = args.render_mp4.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scene = bpy.context.scene
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
                "eevee_samples": getattr(scene.eevee, "taa_render_samples", None),
                "cycles_samples": getattr(scene.cycles, "samples", None) if hasattr(scene, "cycles") else None,
            },
        )
        bpy.ops.render.render(animation=True)

    if args.save_blend is not None:
        out_blend = args.save_blend.resolve()
        out_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))


if __name__ == "__main__":
    main()
