#!/usr/bin/env python3
"""
Batch Process All OakInk Objects
=================================
处理 /home/lyh/Project/OakInk/image/obj/ 下的所有 OBJ/PLY 文件:
  1. PLY → OBJ 转换 (C 系列)
  2. Affordance 推理 → grasp HDF5
  3. 可视化图 PNG
  4. 复制 OBJ → Pipeline/assets/
  5. 生成 convert_all_usd.sh

用法:
    cd /home/lyh/Project/Affordance2Grasp
    python batch_process.py
"""

import os
import sys
import glob
import shutil
import subprocess
import time

# ============================================================
# 配置
# ============================================================
OAKINK_OBJ_DIR = "/home/lyh/Project/OakInk/image/obj"
AFFORDANCE_ROOT = "/home/lyh/Project/Affordance2Grasp"
MANO2GRIPPER_ROOT = "/home/lyh/Project/mano2gripper"

GRASPS_DIR = os.path.join(AFFORDANCE_ROOT, "output", "grasps")
VIS_DIR = os.path.join(AFFORDANCE_ROOT, "output", "analysis")
ASSETS_DIR = os.path.join(MANO2GRIPPER_ROOT, "Pipeline", "assets")
USD_SCRIPT_PATH = os.path.join(MANO2GRIPPER_ROOT, "convert_all_usd.sh")

os.makedirs(GRASPS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


def discover_objects():
    """发现所有 OBJ/PLY 文件, 返回 [(obj_id, file_path, ext), ...]"""
    objects = []
    for f in sorted(os.listdir(OAKINK_OBJ_DIR)):
        name, ext = os.path.splitext(f)
        if ext.lower() in ('.obj', '.ply'):
            objects.append((name, os.path.join(OAKINK_OBJ_DIR, f), ext.lower()))
    return objects


def convert_ply_to_obj(obj_id, ply_path):
    """PLY → OBJ 转换"""
    obj_path = os.path.join(OAKINK_OBJ_DIR, f"{obj_id}.obj")
    if os.path.exists(obj_path):
        return obj_path

    print(f"    Converting PLY → OBJ...")
    import trimesh
    mesh = trimesh.load(ply_path, force='mesh')
    mesh.export(obj_path)
    print(f"    ✅ Saved: {obj_path}")
    return obj_path


def generate_grasp(obj_id, mesh_path, force=False):
    """生成抓取数据 HDF5"""
    hdf5_path = os.path.join(GRASPS_DIR, f"{obj_id}_grasp.hdf5")
    if os.path.exists(hdf5_path) and not force:
        print(f"    ⏭️  Grasp HDF5 already exists")
        return True

    print(f"    Generating grasp data...")
    result = subprocess.run(
        [sys.executable, "-m", "inference.grasp_pose", "--mesh", mesh_path],
        cwd=AFFORDANCE_ROOT,
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"    ❌ Grasp failed: {result.stderr[-200:]}")
        return False

    print(f"    ✅ Grasp HDF5 saved")
    return True


def generate_vis(obj_id):
    """生成可视化图"""
    vis_path = os.path.join(VIS_DIR, f"{obj_id}_affordance_vis.png")
    if os.path.exists(vis_path):
        print(f"    ⏭️  Vis PNG already exists")
        return True

    print(f"    Generating visualization...")
    result = subprocess.run(
        [sys.executable, "analysis/vis_affordance.py", "--obj_id", obj_id],
        cwd=AFFORDANCE_ROOT,
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"    ❌ Vis failed: {result.stderr[-200:]}")
        return False

    print(f"    ✅ Vis PNG saved")
    return True


def copy_to_assets(obj_id, mesh_path):
    """复制 OBJ 到 Pipeline/assets/"""
    dst = os.path.join(ASSETS_DIR, f"{obj_id}.obj")
    if os.path.exists(dst):
        return
    # 确保源是 OBJ (PLY 已转换)
    obj_path = mesh_path
    if mesh_path.endswith('.ply'):
        obj_path = os.path.join(OAKINK_OBJ_DIR, f"{obj_id}.obj")
    if os.path.exists(obj_path):
        shutil.copy2(obj_path, dst)


def generate_usd_script(objects):
    """生成 convert_all_usd.sh"""
    lines = [
        "#!/bin/bash",
        "# Auto-generated: batch OBJ → USD conversion",
        "# 用法: cd ~/Project/mano2gripper && sim45 convert_all_usd.sh",
        "#   或: bash convert_all_usd.sh  (在 sim45 环境下)",
        f"# 共 {len(objects)} 个物体",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
        'CONVERT="$SCRIPT_DIR/Oakink2DP3/convert_obj_to_usd.py"',
        'ASSETS="$SCRIPT_DIR/Pipeline/assets"',
        "",
    ]

    for obj_id, _, _ in objects:
        obj_file = os.path.join(ASSETS_DIR, f"{obj_id}.obj")
        usd_file = os.path.join(ASSETS_DIR, f"{obj_id}.usd")
        lines.append(f'if [ ! -f "{usd_file}" ]; then')
        lines.append(f'  echo "Converting {obj_id}..."')
        lines.append(f'  python "$CONVERT" --input "{obj_file}"')
        lines.append(f'  # Move USD to assets if needed')
        lines.append(f'  OBJ_USD="{os.path.join(OAKINK_OBJ_DIR, obj_id + ".usd")}"')
        lines.append(f'  [ -f "$OBJ_USD" ] && mv "$OBJ_USD" "{usd_file}"')
        lines.append(f'fi')
        lines.append("")

    lines.append('echo "✅ All USD conversions complete!"')
    lines.append(f'echo "Total USD files: $(ls "$ASSETS"/*.usd 2>/dev/null | wc -l)"')

    with open(USD_SCRIPT_PATH, 'w') as f:
        f.write('\n'.join(lines))
    os.chmod(USD_SCRIPT_PATH, 0o755)
    print(f"\n📝 USD conversion script: {USD_SCRIPT_PATH}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="强制重新生成抓取数据")
    cli_args = parser.parse_args()

    print("=" * 60)
    print("Batch Process All OakInk Objects")
    print("=" * 60)

    objects = discover_objects()
    print(f"\n📦 Found {len(objects)} objects:")
    ply_count = sum(1 for _, _, ext in objects if ext == '.ply')
    obj_count = sum(1 for _, _, ext in objects if ext == '.obj')
    print(f"   OBJ: {obj_count}  |  PLY: {ply_count}")
    if cli_args.force:
        print(f"   ⚡ FORCE mode: regenerating all grasp HDF5")

    success = 0
    failed = []
    t0 = time.time()

    for i, (obj_id, file_path, ext) in enumerate(objects, 1):
        print(f"\n[{i}/{len(objects)}] {obj_id} ({ext})")
        print("-" * 40)

        try:
            # Step 1: PLY → OBJ
            mesh_path = file_path
            if ext == '.ply':
                mesh_path = convert_ply_to_obj(obj_id, file_path)

            # Step 2: Generate grasp
            grasp_ok = generate_grasp(obj_id, mesh_path, force=cli_args.force)

            # Step 3: Generate vis (only if grasp succeeded)
            vis_ok = False
            if grasp_ok:
                vis_ok = generate_vis(obj_id)

            # Step 4: Copy to assets
            copy_to_assets(obj_id, file_path)

            if grasp_ok and vis_ok:
                success += 1
            else:
                failed.append((obj_id, "grasp" if not grasp_ok else "vis"))

        except Exception as e:
            print(f"    ❌ Error: {e}")
            failed.append((obj_id, str(e)[:50]))

    elapsed = time.time() - t0

    # Step 5: Generate USD script
    generate_usd_script(objects)

    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Complete: {success}/{len(objects)} objects ({elapsed:.0f}s)")
    if failed:
        print(f"❌ Failed ({len(failed)}):")
        for obj_id, reason in failed:
            print(f"   {obj_id}: {reason}")
    print("=" * 60)
    print(f"\n下一步:")
    print(f"  1. 查看可视化: ls {VIS_DIR}/*.png")
    print(f"  2. USD 转换:   cd {MANO2GRIPPER_ROOT} && sim45 convert_all_usd.sh")
    print(f"  3. Sim 抓取:   sim45 Pipeline/run_grasp_sim.py --hdf5 <path>")


if __name__ == "__main__":
    main()
