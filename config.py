"""
Affordance2Grasp — 全局配置
============================
所有外部路径和默认参数集中在此。
换机器只需修改这一个文件。
"""

import os

# ============================================================
# 项目路径
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")

# Isaac Sim 路径 (sim 模块自动检测, 一般不需要改)
ISAAC_SIM_PATH = os.environ.get("ISAAC_SIM_PATH", "/home/lyh/isaac-sim")

# ============================================================
# data_hub — 统一数据中心
# ============================================================
DATA_HUB = os.path.join(PROJECT_DIR, "data_hub")
MESH_V1_DIR = os.path.join(DATA_HUB, "meshes", "v1")
MESH_V2_DIR = os.path.join(DATA_HUB, "meshes", "v2")
SEQUENCES_V1_DIR = os.path.join(DATA_HUB, "sequences", "v1")
HUMAN_PRIOR_DIR = os.path.join(DATA_HUB, "human_prior")
ROBOT_GT_DIR = os.path.join(DATA_HUB, "robot_gt")
TRAINING_DIR = os.path.join(DATA_HUB, "training")
REGISTRY_PATH = os.path.join(DATA_HUB, "registry.json")

# 兼容旧代码 (逐步淘汰)
OAKINK_OBJ_DIR = MESH_V1_DIR
OAKINK_FILTERED_DIR = SEQUENCES_V1_DIR
OAKINK2_OBJ_DIR = MESH_V2_DIR

# 各阶段输出子目录
CONTACTS_DIR = os.path.join(OUTPUT_DIR, "contacts")
CONTACTS_V2_DIR = os.path.join(OUTPUT_DIR, "contacts_v2")
DATASET_DIR = os.path.join(OUTPUT_DIR, "dataset")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
GRASPS_DIR = os.path.join(OUTPUT_DIR, "grasps")

# ============================================================
# 默认参数
# ============================================================

# 数据提取
CONTACT_THRESHOLD = 0.005    # 5mm 接触距离阈值
FRAME_STEP = 5               # 每 N 帧采样

# 稳定性过滤 (M1 新增)
MIN_FINGERS = 3              # 至少 N 根手指同时接触
MIN_STABLE_FRAMES = 10       # 连续 N 帧保持稳定接触

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

# Robot GT 聚合
GAUSSIAN_SIGMA = 0.005       # 高斯核半径 5mm

# ============================================================
# 辅助函数
# ============================================================
def ensure_dirs():
    """创建所有输出目录。"""
    for d in [CONTACTS_DIR, DATASET_DIR, CHECKPOINT_DIR, GRASPS_DIR,
              HUMAN_PRIOR_DIR, ROBOT_GT_DIR, TRAINING_DIR]:
        os.makedirs(d, exist_ok=True)

