import argparse
import math
import os
import sys
import importlib
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

try:
    bpy = importlib.import_module("bpy")
    bmesh = importlib.import_module("bmesh")
except Exception as e:
    raise ImportError("This script must be run inside Blender (missing bpy/bmesh)") from e


def _tri_count_for_mesh(mesh: Any) -> int:
    # Mesh polygons can be quads/ngons; estimate triangle count by fan triangulation.
    tris = 0
    for poly in mesh.polygons:
        n = len(poly.vertices)
        if n >= 3:
            tris += n - 2
    return int(tris)


def _tri_count_for_object(obj: Any) -> int:
    if obj is None or getattr(obj, "type", None) != "MESH":
        return 0
    return _tri_count_for_mesh(obj.data)


def _get_mesh_objects() -> List[Any]:
    return [o for o in bpy.context.scene.objects if getattr(o, "type", None) == "MESH"]


def _cleanup_topology(
    obj: Any,
    merge_dist: float,
    limited_dissolve_angle_deg: float,
    do_delete_loose: bool,
    do_limited_dissolve: bool,
) -> None:
    mesh = obj.data

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)

        if do_delete_loose:
            loose_verts = [v for v in bm.verts if len(v.link_faces) == 0]
            loose_edges = [e for e in bm.edges if len(e.link_faces) == 0]
            if loose_edges:
                bmesh.ops.delete(bm, geom=loose_edges, context="EDGES")
            if loose_verts:
                bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")

        if merge_dist and merge_dist > 0:
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=float(merge_dist))

        # Recalculate normals (outside)
        if bm.faces:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

        if do_limited_dissolve and limited_dissolve_angle_deg and limited_dissolve_angle_deg > 0:
            angle_limit = math.radians(float(limited_dissolve_angle_deg))
            # Delimit by UV/material seams to reduce damage.
            bmesh.ops.dissolve_limit(
                bm,
                angle_limit=angle_limit,
                use_dissolve_boundaries=False,
                delimit={"UV", "MATERIAL"},
                edges=bm.edges,
            )

        bm.to_mesh(mesh)
        mesh.update()
    finally:
        bm.free()


def _apply_shading(obj: Any, auto_smooth_angle_deg: float) -> None:
    mesh = obj.data

    # Shade smooth (per-face flag)
    for poly in mesh.polygons:
        poly.use_smooth = True

    # Auto smooth for hard edges.
    mesh.use_auto_smooth = True
    mesh.auto_smooth_angle = math.radians(float(auto_smooth_angle_deg))

    try:
        mesh.calc_normals_split()
    except Exception:
        pass


def _recalc_normals(obj: Any) -> None:
    mesh = obj.data
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        if bm.faces:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        mesh.update()
    finally:
        bm.free()


def _apply_decimate(obj: Any, ratio: float) -> None:
    ratio = float(ratio)
    if ratio >= 1.0:
        return
    if ratio <= 0.0:
        ratio = 0.01

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mod = obj.modifiers.new(name="Trellis_Decimate", type="DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio

    # Apply modifier
    bpy.ops.object.modifier_apply(modifier=mod.name)

    obj.select_set(False)


def _reset_scene() -> None:
    # Start with a clean scene.
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    except Exception:
        # Fallback: remove objects
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)


def _import_glb(path: str) -> None:
    bpy.ops.import_scene.gltf(filepath=path)


def _export_glb(path: str) -> None:
    # Ensure output directory exists.
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        export_apply=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    # Blender passes args before/after '--'
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-tris", type=int, default=15000)
    parser.add_argument("--merge-dist", type=float, default=0.0005)
    parser.add_argument("--auto-smooth-angle", type=float, default=45.0)
    parser.add_argument("--limited-dissolve-angle", type=float, default=1.0)
    parser.add_argument("--no-delete-loose", action="store_true")
    parser.add_argument("--no-limited-dissolve", action="store_true")
    parser.add_argument("--no-decimate", action="store_true")
    parser.add_argument("--max-decimate-iters", type=int, default=3)

    args = parser.parse_args(argv)

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    _reset_scene()
    _import_glb(input_path)

    mesh_objects = _get_mesh_objects()
    if not mesh_objects:
        print("[cleanup_glb] No mesh objects found after import.")
        _export_glb(output_path)
        return 0

    # Ensure object mode
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    # Initial cleanup: delete loose -> merge -> normals -> shading
    for obj in mesh_objects:
        _cleanup_topology(
            obj,
            merge_dist=float(args.merge_dist),
            limited_dissolve_angle_deg=float(args.limited_dissolve_angle),
            do_delete_loose=not args.no_delete_loose,
            do_limited_dissolve=not args.no_limited_dissolve,
        )
        _apply_shading(obj, auto_smooth_angle_deg=float(args.auto_smooth_angle))

    # Decimate to reach target tris (total across all mesh objects)
    if not args.no_decimate:
        for _ in range(int(args.max_decimate_iters)):
            total_tris = sum(_tri_count_for_object(o) for o in mesh_objects)
            print(f"[cleanup_glb] Total triangles before decimate iter: {total_tris}")
            if total_tris <= int(args.target_tris):
                break
            ratio = float(args.target_tris) / float(total_tris)
            ratio = max(0.01, min(1.0, ratio))

            for obj in mesh_objects:
                _apply_decimate(obj, ratio=ratio)

            # Normals again after decimate (key pro step)
            for obj in mesh_objects:
                _recalc_normals(obj)
                _apply_shading(obj, auto_smooth_angle_deg=float(args.auto_smooth_angle))

    total_tris_after = sum(_tri_count_for_object(o) for o in mesh_objects)
    print(f"[cleanup_glb] Total triangles after cleanup: {total_tris_after}")

    _export_glb(output_path)
    print(f"[cleanup_glb] Exported cleaned GLB: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
