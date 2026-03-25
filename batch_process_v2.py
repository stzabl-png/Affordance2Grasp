#!/usr/bin/env python3
"""
Batch Process OakInk-v2 Objects
===============================
处理 OakInk-v2 的 object_raw/align_ds/ 下的所有物体:
  1. PLY → OBJ 转换
  2. Affordance 推理 → grasp HDF5
  3. 可视化图 PNG
  4. 复制 OBJ → Pipeline/assets/
  5. 生成 convert_all_usd.sh

用法:
    cd /home/lyh/Project/Affordance2Grasp
    python batch_process_v2.py
    python batch_process_v2.py --force     # 重新生成
    python batch_process_v2.py --max 10    # 只处理前10个
"""

import os
import sys
import shutil
import subprocess
import time
import argparse

# ============================================================
# 配置 — 使用 config 中的 data_hub 路径
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

OAKINK2_OBJ_DIR = config.MESH_V2_DIR
AFFORDANCE_ROOT = config.PROJECT_DIR

GRASPS_DIR = config.GRASPS_DIR
VIS_DIR = os.path.join(config.OUTPUT_DIR, "analysis")
ASSETS_DIR = os.path.join(config.ASSETS_DIR, "usd")

# v2 物体的 OBJ 转换输出目录
V2_OBJ_DIR = os.path.join(config.OUTPUT_DIR, "meshes_v2")

os.makedirs(GRASPS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(V2_OBJ_DIR, exist_ok=True)


def discover_objects():
    """发现 OakInk-v2 的所有物体, 返回 [(obj_id, ply_path), ...]"""
    objects = []
    if not os.path.isdir(OAKINK2_OBJ_DIR):
        print(f"❌ 目录不存在: {OAKINK2_OBJ_DIR}")
        return objects

    for obj_dir in sorted(os.listdir(OAKINK2_OBJ_DIR)):
        full_dir = os.path.join(OAKINK2_OBJ_DIR, obj_dir)
        if not os.path.isdir(full_dir):
            continue
        # 在子目录中找 PLY/OBJ 文件
        for f in os.listdir(full_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in ('.ply', '.obj'):
                objects.append((obj_dir, os.path.join(full_dir, f)))
                break  # 每个目录只取第一个 mesh
    return objects


def safe_obj_id(obj_id):
    """将 O02@0015@00019 → O02_0015_00019 (文件名安全)"""
    return obj_id.replace("@", "_")


def convert_to_obj(obj_id, mesh_path):
    """PLY/OBJ → output/meshes_v2/xxx.obj"""
    safe_id = safe_obj_id(obj_id)
    obj_path = os.path.join(V2_OBJ_DIR, f"{safe_id}.obj")
    if os.path.exists(obj_path):
        return obj_path

    print(f"    Converting → OBJ...")
    import trimesh
    mesh = trimesh.load(mesh_path, force='mesh')
    mesh.export(obj_path)
    print(f"    ✅ {obj_path}")
    return obj_path


def generate_grasp(obj_id, mesh_path, force=False):
    """生成抓取数据 HDF5"""
    safe_id = safe_obj_id(obj_id)
    hdf5_path = os.path.join(GRASPS_DIR, f"{safe_id}_grasp.hdf5")
    if os.path.exists(hdf5_path) and not force:
        print(f"    ⏭️  Grasp HDF5 exists")
        return True

    print(f"    Generating grasp data...")
    result = subprocess.run(
        [sys.executable, "-m", "inference.grasp_pose", "--mesh", mesh_path],
        cwd=AFFORDANCE_ROOT,
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        err = result.stderr[-300:] if result.stderr else result.stdout[-300:]
        print(f"    ❌ Failed: {err}")
        return False

    print(f"    ✅ Grasp HDF5 saved")
    return True


def generate_vis(obj_id):
    """生成可视化图"""
    safe_id = safe_obj_id(obj_id)
    vis_path = os.path.join(VIS_DIR, f"{safe_id}_affordance_vis.png")
    if os.path.exists(vis_path):
        print(f"    ⏭️  Vis PNG exists")
        return True

    print(f"    Generating visualization...")
    result = subprocess.run(
        [sys.executable, "analysis/vis_affordance.py", "--obj_id", safe_id],
        cwd=AFFORDANCE_ROOT,
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"    ⚠️  Vis failed (non-critical)")
        return False

    print(f"    ✅ Vis PNG saved")
    return True


def copy_to_assets(obj_id, obj_path):
    """复制 OBJ 到 Pipeline/assets/"""
    safe_id = safe_obj_id(obj_id)
    dst = os.path.join(ASSETS_DIR, f"{safe_id}.obj")
    if not os.path.exists(dst):
        shutil.copy2(obj_path, dst)


def main():
    parser = argparse.ArgumentParser(description="Batch process OakInk-v2 objects")
    parser.add_argument("--force", action="store_true", help="Force regenerate")
    parser.add_argument("--max", type=int, default=0, help="Max objects to process")
    args = parser.parse_args()

    print("=" * 60)
    print("Batch Process OakInk-v2 Objects")
    print("=" * 60)

    objects = discover_objects()
    if args.max > 0:
        objects = objects[:args.max]

    print(f"\n📦 Found {len(objects)} objects")
    if args.force:
        print(f"   ⚡ FORCE mode")

    success = 0
    failed = []
    t0 = time.time()

    for i, (obj_id, mesh_path) in enumerate(objects, 1):
        safe_id = safe_obj_id(obj_id)
        print(f"\n[{i}/{len(objects)}] {obj_id}")
        print("-" * 40)

        try:
            # Step 1: Convert to OBJ
            obj_path = convert_to_obj(obj_id, mesh_path)

            # Step 2: Generate grasp
            grasp_ok = generate_grasp(obj_id, obj_path, force=args.force)

            # Step 3: Vis (optional)
            if grasp_ok:
                generate_vis(obj_id)

            # Step 4: Copy to assets
            copy_to_assets(obj_id, obj_path)

            if grasp_ok:
                success += 1
            else:
                failed.append((obj_id, "grasp"))

        except Exception as e:
            print(f"    ❌ Error: {e}")
            failed.append((obj_id, str(e)[:50]))

    elapsed = time.time() - t0

    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Complete: {success}/{len(objects)} objects ({elapsed:.0f}s)")
    if failed:
        print(f"❌ Failed ({len(failed)}):")
        for obj_id, reason in failed:
            print(f"   {obj_id}: {reason}")
    print("=" * 60)
    print(f"\n下一步:")
    print(f"  1. 查看可视化: ls {VIS_DIR}/*_affordance_vis.png")
    print(f"  2. USD 转换:   cd {MANO2GRIPPER_ROOT}")
    print(f"  3. Sim 抓取:   sim45 sim/run_grasp.py --hdf5 <path>")


if __name__ == "__main__":
    main()
