from __future__ import annotations

import sys
from pathlib import Path

import bpy


def _parse_args() -> tuple[Path, Path]:
    if "--" not in sys.argv:
        raise RuntimeError("Expected Blender script arguments after '--': <input.bvh> <output.fbx>")
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 2:
        raise RuntimeError("Usage: blender --background --python blender_bvh_to_fbx.py -- <input.bvh> <output.fbx>")

    bvh_path = Path(args[0]).resolve()
    fbx_path = Path(args[1]).resolve()
    return bvh_path, fbx_path


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _ensure_finished(op_name: str, result: object) -> None:
    if "FINISHED" not in set(result):
        raise RuntimeError(f"{op_name} failed with result: {result}")


def main() -> None:
    bvh_path, fbx_path = _parse_args()

    if not bvh_path.exists():
        raise FileNotFoundError(f"BVH file not found: {bvh_path}")

    fbx_path.parent.mkdir(parents=True, exist_ok=True)

    _clear_scene()

    import_result = bpy.ops.import_anim.bvh(
        filepath=str(bvh_path),
        axis_forward="-Z",
        axis_up="Y",
    )
    _ensure_finished("import_anim.bvh", import_result)

    export_result = bpy.ops.export_scene.fbx(
        filepath=str(fbx_path),
        use_selection=False,
        add_leaf_bones=False,
        bake_anim=True,
        use_armature_deform_only=True,
        axis_forward="-Z",
        axis_up="Y",
    )
    _ensure_finished("export_scene.fbx", export_result)

    if (not fbx_path.exists()) or fbx_path.stat().st_size == 0:
        raise RuntimeError(f"FBX export reported success but file is missing/empty: {fbx_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[blender_bvh_to_fbx] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
