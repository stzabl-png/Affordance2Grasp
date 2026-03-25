#!/usr/bin/env python3
"""
批量 OBJ/PLY → USD 转换 (Isaac Sim)
====================================
将 Affordance2Grasp 的所有物体 mesh 转换为 Isaac Sim USD 格式。

支持:
  - v1 OBJ 文件 (data_hub/meshes/v1/)
  - GRAB PLY 文件 (data_hub/meshes/grab/) — 先转 OBJ 再转 USD

用法 (需要 Isaac Sim 环境):
    cd /home/lyh/Project/mano2gripper
    sim45 Pipeline/convert_batch_usd.py

    # 只转 GRAB 物体
    sim45 Pipeline/convert_batch_usd.py --grab-only

    # 跳过已有 USD 的物体
    sim45 Pipeline/convert_batch_usd.py --skip-existing
"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import os
import sys
import asyncio
import glob
import numpy as np
import omni.kit.asset_converter as converter

# ---- Paths ----
AFFORDANCE_ROOT = "/home/lyh/Project/Affordance2Grasp"
MESH_V1_DIR = os.path.join(AFFORDANCE_ROOT, "data_hub", "meshes", "v1")
MESH_GRAB_DIR = os.path.join(AFFORDANCE_ROOT, "data_hub", "meshes", "grab")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


async def convert_to_usd(input_path, output_path):
    """Async USD conversion."""
    task_manager = converter.get_instance()
    context = converter.AssetConverterContext()
    context.ignore_materials = False
    context.ignore_animations = True
    context.ignore_camera = True
    context.ignore_light = True
    context.single_mesh = True
    context.smooth_normals = True
    context.export_preview_surface = False
    context.use_meter_as_world_unit = True
    context.embed_textures = True

    task = task_manager.create_converter_task(
        input_path, output_path, progress_callback=None,
        asset_converter_context=context
    )
    success = await task.wait_until_finished()
    return success


def ply_to_obj(ply_path, obj_path):
    """Convert PLY to OBJ using trimesh."""
    import trimesh
    mesh = trimesh.load(ply_path, force='mesh')
    mesh.export(obj_path)
    return obj_path


def discover_all_objects(grab_only=False):
    """Discover all mesh files. Returns [(obj_id, mesh_path), ...]"""
    objects = []

    if not grab_only:
        # v1 meshes
        if os.path.exists(MESH_V1_DIR):
            for f in sorted(os.listdir(MESH_V1_DIR)):
                name, ext = os.path.splitext(f)
                if ext.lower() in ('.obj', '.ply'):
                    objects.append((name, os.path.join(MESH_V1_DIR, f)))

    # GRAB meshes
    if os.path.exists(MESH_GRAB_DIR):
        for f in sorted(os.listdir(MESH_GRAB_DIR)):
            name, ext = os.path.splitext(f)
            if ext.lower() in ('.obj', '.ply') and '_low' not in name:
                if not any(o[0] == name for o in objects):
                    objects.append((name, os.path.join(MESH_GRAB_DIR, f)))

    return objects


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--grab-only", action="store_true",
                        help="Only convert GRAB objects")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip objects that already have USD files")
    args = parser.parse_args()

    objects = discover_all_objects(grab_only=args.grab_only)

    print("=" * 60)
    print("Batch USD Conversion")
    print("=" * 60)
    print(f"  Objects:    {len(objects)}")
    print(f"  Output:     {ASSETS_DIR}")
    print(f"  Skip exist: {args.skip_existing}")

    success = 0
    skipped = 0
    failed = []

    loop = asyncio.get_event_loop()

    for i, (obj_id, mesh_path) in enumerate(objects, 1):
        usd_path = os.path.join(ASSETS_DIR, f"{obj_id}.usd")

        if args.skip_existing and os.path.exists(usd_path):
            skipped += 1
            continue

        print(f"\n  [{i}/{len(objects)}] {obj_id}", end="", flush=True)

        # PLY → OBJ first
        obj_path = mesh_path
        if mesh_path.endswith('.ply'):
            obj_path = os.path.join(ASSETS_DIR, f"{obj_id}.obj")
            if not os.path.exists(obj_path):
                ply_to_obj(mesh_path, obj_path)
                print(f" (PLY→OBJ)", end="", flush=True)

        # OBJ → USD
        ok = loop.run_until_complete(convert_to_usd(obj_path, usd_path))

        if ok:
            success += 1
            print(f" ✅")
        else:
            failed.append(obj_id)
            print(f" ❌")

    print(f"\n{'=' * 60}")
    print(f"  ✅ Success: {success}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"  ❌ Failed:  {len(failed)}")
    if failed:
        print(f"  Failed objects: {', '.join(failed)}")
    print(f"  Total USD: {len(glob.glob(os.path.join(ASSETS_DIR, '*.usd')))}")
    print(f"{'=' * 60}")


main()
simulation_app.close()
