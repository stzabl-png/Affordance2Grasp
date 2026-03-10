"""
Affordance2Grasp — 全局配置
============================
所有外部路径和默认参数集中在此。
换机器只需修改这一个文件。
"""

import os

# ============================================================
# 路径配置 (按需修改)
# ============================================================

# OakInk 数据集根目录
OAKINK_DIR = "/home/lyh/Project/OakInk"

# Isaac Sim 路径 (sim 模块自动检测, 一般不需要改)
ISAAC_SIM_PATH = os.environ.get("ISAAC_SIM_PATH", "/home/lyh/isaac-sim")

# ============================================================
# 项目路径 (自动计算, 不需要改)
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")

# 各阶段输出子目录
CONTACTS_DIR = os.path.join(OUTPUT_DIR, "contacts")
DATASET_DIR = os.path.join(OUTPUT_DIR, "dataset")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
GRASPS_DIR = os.path.join(OUTPUT_DIR, "grasps")

# OakInk 子路径
OAKINK_OBJ_DIR = os.path.join(OAKINK_DIR, "image", "obj")
OAKINK_ANNO_DIR = os.path.join(OAKINK_DIR, "image", "anno")
OAKINK_FILTERED_DIR = os.path.join(OAKINK_DIR, "filtered")

# ============================================================
# 默认参数
# ============================================================

# 数据提取
CONTACT_THRESHOLD = 0.005    # 5mm 接触距离阈值
FRAME_STEP = 5               # 每 N 帧采样

# 数据集
NUM_POINTS = 1024            # 点云采样数
CONTACT_RADIUS = 0.005       # 接触标签半径

# 训练
TRAIN_EPOCHS = 150
TRAIN_BATCH_SIZE = 32
TRAIN_LR = 0.001

# 推理
AFFORDANCE_THRESHOLD = 0.3  # 接触概率阈值

# 仿真
OBJECT_SCALE = 1.5           # 物体 Sim 缩放
TABLE_TOP_Z = 0.80
ROBOT_POSITION = [0.2, -0.05, 0.8]
ROBOT_ORIENTATION = [0.0, 0.0, 90.0]

# ============================================================
# 辅助函数
# ============================================================
def ensure_dirs():
    """创建所有输出目录。"""
    for d in [CONTACTS_DIR, DATASET_DIR, CHECKPOINT_DIR, GRASPS_DIR]:
        os.makedirs(d, exist_ok=True)
