#!/usr/bin/env python3
"""
批量转换 OakInk-v2 OBJ → USD

使用方式 (需要 Isaac Sim 环境):
    sim45 assets/convert_all_v2_usd.py
"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import os
import sys
import asyncio
import omni.kit.asset_converter as converter

# v2 OBJ 文件目录
V2_OBJ_DIR = os.path.expanduser("~/Project/Affordance2Grasp/output/meshes_v2")
USD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd")
os.makedirs(USD_DIR, exist_ok=True)


async def convert_one(input_path, output_path):
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
        input_path, output_path, progress_callback=None, asset_converter_context=context
    )
    success = await task.wait_until_finished()
    if not success:
        print(f"  FAIL: {task.get_status()} - {task.get_detailed_error()}")
    return success


def main():
    # 找所有 OBJ 文件
    obj_files = sorted([
        f for f in os.listdir(V2_OBJ_DIR)
        if f.endswith('.obj')
    ])
    print(f"Found {len(obj_files)} OBJ files in {V2_OBJ_DIR}")

    success = 0
    skipped = 0
    failed = []

    for i, obj_file in enumerate(obj_files, 1):
        name = os.path.splitext(obj_file)[0]
        input_path = os.path.abspath(os.path.join(V2_OBJ_DIR, obj_file))
        output_path = os.path.join(USD_DIR, f"{name}.usd")

        if os.path.exists(output_path):
            print(f"[{i}/{len(obj_files)}] {name}: skip (exists)")
            skipped += 1
            continue

        print(f"[{i}/{len(obj_files)}] {name}: converting...", end=" ", flush=True)
        ok = asyncio.get_event_loop().run_until_complete(convert_one(input_path, output_path))
        if ok:
            print("✅")
            success += 1
        else:
            print("❌")
            failed.append(name)

    print(f"\nDone: {success} converted, {skipped} skipped, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")


main()
simulation_app.close()
