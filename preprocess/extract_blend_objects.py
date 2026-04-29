import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


EXPORT_SCRIPT_TEMPLATE = r"""
import bpy
import json
import sys
from pathlib import Path
from mathutils import Vector

INCLUDE_BBOX = __INCLUDE_BBOX__
INCLUDE_NAMES = __INCLUDE_NAMES__


def matrix_to_list(mat):
    return [[float(v) for v in row] for row in mat]


def vector_to_list(vec):
    return [float(v) for v in vec]


def object_record(obj):
    world_loc, world_rot, _world_scale = obj.matrix_world.decompose()
    record = {
        "name": obj.name,
        "type": obj.type,
        "parent": obj.parent.name if obj.parent else None,
        "location": vector_to_list(obj.location),
        "rotation_euler": vector_to_list(obj.rotation_euler),
        "world_location": vector_to_list(world_loc),
        "world_rotation_euler": vector_to_list(world_rot.to_euler()),
        "scale": vector_to_list(obj.scale),
        "matrix_world": matrix_to_list(obj.matrix_world),
        "has_mesh_data": obj.type == "MESH" and obj.data is not None,
        "data_name": obj.data.name if obj.data else None,
    }
    if INCLUDE_BBOX:
        world_corners = [obj.matrix_world @ Vector(corner[:]) for corner in obj.bound_box]
        min_corner = [
            min(float(c[i]) for c in world_corners)
            for i in range(3)
        ]
        max_corner = [
            max(float(c[i]) for c in world_corners)
            for i in range(3)
        ]
        record["bbox_min"] = min_corner
        record["bbox_max"] = max_corner
        record["dimensions"] = vector_to_list(obj.dimensions)
    return record


argv = sys.argv
if "--" not in argv:
    raise SystemExit("Expected '-- <output_json>'")
out_path = Path(argv[argv.index("--") + 1])
records = []
for obj in bpy.data.objects:
    if INCLUDE_NAMES is not None and obj.name not in INCLUDE_NAMES:
        continue
    records.append(object_record(obj))

payload = {
    "blend_file": bpy.data.filepath,
    "num_objects": len(records),
    "objects": records,
}
out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"wrote {out_path}")
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract object metadata from a Blender .blend file."
    )
    parser.add_argument("blend_file", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <blend_file>.objects.json",
    )
    parser.add_argument(
        "--blender-bin",
        default="blender",
        help="Blender executable to use",
    )
    parser.add_argument(
        "--noaudio",
        action="store_true",
        help="Pass -noaudio to Blender to avoid audio init issues in headless mode.",
    )
    parser.add_argument(
        "--factory-startup",
        action="store_true",
        help="Pass --factory-startup to Blender for a cleaner headless startup.",
    )
    parser.add_argument(
        "--no-bbox",
        action="store_true",
        help="Skip bbox and dimensions export. Keeps only object transforms and basic metadata.",
    )
    parser.add_argument(
        "--include-name",
        action="append",
        default=[],
        help="If provided, only export objects with these exact names. Can be repeated.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    blend_file = args.blend_file.resolve()
    if not blend_file.exists():
        raise FileNotFoundError(f"Blend file not found: {blend_file}")

    output_path = args.output
    if output_path is None:
        output_path = blend_file.with_suffix(".objects.json")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_script = EXPORT_SCRIPT_TEMPLATE.replace(
        "__INCLUDE_BBOX__",
        "False" if args.no_bbox else "True",
    )
    export_script = export_script.replace(
        "__INCLUDE_NAMES__",
        repr(sorted(set(args.include_name))) if args.include_name else "None",
    )

    with tempfile.NamedTemporaryFile("w", suffix="_blend_export.py", delete=False) as tmp:
        tmp.write(export_script)
        script_path = Path(tmp.name)

    cmd = [
        args.blender_bin,
    ]
    if args.noaudio:
        cmd.append("-noaudio")
    if args.factory_startup:
        cmd.append("--factory-startup")
    cmd.extend(
        [
            "-b",
            str(blend_file),
            "--python",
            str(script_path),
            "--",
            str(output_path),
        ]
    )
    try:
        subprocess.run(cmd, check=True)
    finally:
        script_path.unlink(missing_ok=True)

    print(output_path)


if __name__ == "__main__":
    main()
