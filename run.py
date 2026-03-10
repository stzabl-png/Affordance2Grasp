#!/usr/bin/env python3
"""
Affordance2Grasp — 统一入口
============================
一个命令完成: 物体 mesh → Affordance 推理 → 抓取位姿 → Isaac Sim 执行

用法:
    # 推理 + 仿真 (最常用)
    python run.py --mesh /path/to/object.obj

    # 只生成抓取位姿 (不启动 Sim)
    python run.py --mesh /path/to/object.obj --no-sim

    # 从零训练 (换新数据集时)
    python run.py --prepare
    python run.py --train --epochs 150

    # 直接执行已有 HDF5
    python run.py --execute output/grasps/A16013_grasp.hdf5
"""

import os
import sys
import argparse
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
import config


def cmd_prepare(args):
    """Stage 1-2: 提取接触点 + 生成训练集."""
    print("=" * 60)
    print("Stage 1: 提取接触点")
    print("=" * 60)
    cmd = [sys.executable, "-m", "data.extract_contacts"]
    if args.threshold:
        cmd += ["--threshold", str(args.threshold)]
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)

    print("\n" + "=" * 60)
    print("Stage 2: 生成训练集")
    print("=" * 60)
    cmd = [sys.executable, "-m", "data.build_dataset"]
    if args.intent:
        cmd += ["--intent", args.intent]
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)


def cmd_train(args):
    """Stage 3: 训练 Affordance 模型."""
    print("=" * 60)
    print("Stage 3: 训练 Affordance 模型")
    print("=" * 60)
    cmd = [sys.executable, "-m", "model.train",
           "--epochs", str(args.epochs),
           "--batch_size", str(args.batch_size)]
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)


def cmd_infer(args):
    """Stage 4: Affordance 推理 + 抓取位姿生成."""
    print("=" * 60)
    print("Stage 4: Grasp Pose Generation")
    print("=" * 60)
    cmd = [sys.executable, "-m", "inference.grasp_pose",
           "--mesh", args.mesh,
           "--threshold", str(args.affordance_threshold)]
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    return result.returncode == 0


def cmd_sim(hdf5_path, args):
    """Stage 5: Isaac Sim 抓取执行."""
    print("\n" + "=" * 60)
    print("Stage 5: Isaac Sim 抓取执行")
    print("=" * 60)

    # sim 模块必须用 Isaac Sim 的 Python
    isaac_python = os.path.join(config.ISAAC_SIM_PATH, "python.sh")
    if not os.path.exists(isaac_python):
        print(f"❌ Isaac Sim python not found: {isaac_python}")
        print(f"   请修改 config.py 中的 ISAAC_SIM_PATH")
        return False

    sim_script = os.path.join(PROJECT_DIR, "sim", "run_grasp.py")
    cmd = [isaac_python, sim_script,
           "--hdf5", hdf5_path,
           "--object_scale", str(args.object_scale)]
    if args.headless:
        cmd.append("--headless")

    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Affordance2Grasp — 从物体 mesh 到仿真抓取的完整 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py --mesh /path/to/object.obj          # 推理 + 仿真
  python run.py --mesh /path/to/object.obj --no-sim # 只生成位姿
  python run.py --prepare                           # 准备训练数据
  python run.py --train                             # 训练模型
  python run.py --execute output/grasps/A16013_grasp.hdf5  # 直接执行
        """,
    )

    # 模式选择
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mesh", type=str, help="物体 mesh 路径 → 推理 + 仿真")
    mode.add_argument("--prepare", action="store_true", help="Stage 1-2: 提取接触点 + 生成训练集")
    mode.add_argument("--train", action="store_true", help="Stage 3: 训练模型")
    mode.add_argument("--execute", type=str, help="Stage 5: 直接执行已有 HDF5")

    # 通用参数
    parser.add_argument("--no-sim", action="store_true", help="不启动 Isaac Sim (只生成 HDF5)")
    parser.add_argument("--headless", action="store_true", help="Isaac Sim 无头模式")
    parser.add_argument("--object_scale", type=float, default=config.OBJECT_SCALE)
    parser.add_argument("--affordance_threshold", type=float, default=config.AFFORDANCE_THRESHOLD)

    # 训练参数
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.TRAIN_BATCH_SIZE)

    # 数据参数
    parser.add_argument("--threshold", type=float, default=None, help="接触距离阈值")
    parser.add_argument("--intent", type=str, default=None, help="OakInk intent 过滤 (hold/use)")

    args = parser.parse_args()
    config.ensure_dirs()

    # ---- 路由到对应功能 ----
    if args.prepare:
        cmd_prepare(args)
    elif args.train:
        cmd_train(args)
    elif args.execute:
        if not os.path.exists(args.execute):
            print(f"❌ HDF5 not found: {args.execute}")
            return
        cmd_sim(args.execute, args)
    elif args.mesh:
        # 主流程: 推理 → 仿真
        if not os.path.exists(args.mesh):
            print(f"❌ Mesh not found: {args.mesh}")
            return

        # Stage 4: 推理
        obj_name = os.path.splitext(os.path.basename(args.mesh))[0]
        hdf5_path = os.path.join(config.GRASPS_DIR, f"{obj_name}_grasp.hdf5")

        if not cmd_infer(args):
            print("❌ 推理失败")
            return

        if args.no_sim:
            print(f"\n✅ 位姿已保存: {hdf5_path}")
            print(f"   手动执行: sim45 sim/run_grasp.py --hdf5 {hdf5_path}")
            return

        # Stage 5: 仿真
        cmd_sim(hdf5_path, args)


if __name__ == "__main__":
    main()
