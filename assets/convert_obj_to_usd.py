"""
将 OakInk 的 .obj 模型转换为 Isaac Sim 可用的 .usd 格式

使用方式 (从 mano2gripper 根目录):
    isaac python Oakink2DP3/convert_obj_to_usd.py
    isaac python Oakink2DP3/convert_obj_to_usd.py --input Oakink2DP3/Assets/A16013.obj
"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import os
import sys
import argparse
import asyncio
import omni.kit.asset_converter as converter

async def convert(input_path, output_path):
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
        print(f"  Failed: {task.get_status()} - {task.get_detailed_error()}")
    return success

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, 
                        default=os.path.join(os.path.dirname(__file__), "Assets", "A16013.obj"))
    args = parser.parse_args()
    
    input_path = os.path.abspath(args.input)
    # 保存到 assets/usd/ 目录
    usd_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd")
    os.makedirs(usd_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_path))[0] + ".usd"
    output_path = os.path.join(usd_dir, basename)
    
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return
    
    success = asyncio.get_event_loop().run_until_complete(convert(input_path, output_path))
    if success:
        print(f"✅ OBJ → USD done: {output_path}")
    else:
        print("❌ Conversion failed")

main()
simulation_app.close()
