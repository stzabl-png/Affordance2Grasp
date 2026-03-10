# Affordance2Grasp

从物体 mesh 到仿真抓取执行的完整 Pipeline。

基于 OakInk 人手抓取数据训练 PointNet++ Affordance 模型，预测物体接触区域，计算 Franka 二指夹爪抓取位姿，在 Isaac Sim 中用 cuRobo 执行无碰撞抓取。

```
OakInk 数据 → 接触点提取 → 训练集 → PointNet++ 训练
                                                ↓
物体 Mesh → Affordance 推理 → 抓取位姿 → Isaac Sim 执行
```

## 快速开始

### 有训练好的模型（推荐）

```bash
cd /home/lyh/Project/Affordance2Grasp

# 一条命令: mesh → affordance → 抓取位姿 → Isaac Sim 执行
python run.py --mesh /home/lyh/Project/OakInk/image/obj/A16013.obj
```

这会自动执行:
1. PointNet++ 预测接触区域
2. PCA 计算抓取位姿
3. 保存 HDF5 到 `output/grasps/`
4. 启动 Isaac Sim + cuRobo 执行抓取

### 只生成位姿（不启动 Sim）

```bash
python run.py --mesh /path/to/object.obj --no-sim
```

### 直接执行已有位姿

```bash
sim45 sim/run_grasp.py --hdf5 output/grasps/A16013_grasp.hdf5
```

---

## 从零搭建（换新数据集/物体）

### Step 1: 提取接触点 + 生成训练集

```bash
# 需要 OakInk 数据集
python run.py --prepare
python run.py --prepare --intent hold   # 只用 hold 抓取意图
```

### Step 2: 训练 Affordance 模型

```bash
python run.py --train --epochs 150
```

训练好的 checkpoint 自动保存到 `output/checkpoints/best_model.pth`。

### Step 3: 推理 + 执行

```bash
python run.py --mesh /path/to/new_object.obj
```

---

## 扩展到新物体

### 1. 准备 USD 文件

Isaac Sim 需要 USD 格式的 3D 模型:

```bash
# 在 Isaac Sim 环境下
sim45 assets/convert_obj_to_usd.py --input /path/to/object.obj
# USD 会保存到 assets/usd/
```

### 2. 执行

```bash
python run.py --mesh /path/to/object.obj
```

---

## 项目结构

```
Affordance2Grasp/
├── run.py                      # 统一入口
├── config.py                   # 全局配置 (路径、参数)
│
├── data/                       # Stage 1-2: 数据处理
│   ├── extract_contacts.py     #   OakInk → 接触点
│   └── build_dataset.py        #   接触点 → HDF5 训练集
│
├── model/                      # Stage 3: 模型
│   ├── pointnet2.py            #   PointNet++ 架构
│   └── train.py                #   训练脚本
│
├── inference/                  # Stage 4: 推理 + 抓取位姿
│   ├── predictor.py            #   Affordance 推理 API
│   └── grasp_pose.py           #   Mesh → 抓取位姿 HDF5
│
├── sim/                        # Stage 5: Isaac Sim 执行
│   ├── run_grasp.py            #   Franka + cuRobo 抓取
│   └── env_config/             #   仿真环境配置
│       ├── franka.py           #     Franka 机器人
│       ├── real_ground.py      #     地面
│       ├── rigid_object.py     #     刚体物体
│       ├── set_drive.py        #     关节驱动
│       ├── transforms.py       #     坐标变换
│       └── code_tools.py       #     工具函数
│
├── assets/                     # 资源文件
│   ├── usd/                    #   物体 USD 模型
│   ├── scene/                  #   场景环境 USD
│   └── convert_obj_to_usd.py   #   OBJ → USD 转换
│
└── output/                     # 输出 (gitignored)
    ├── contacts/               #   Stage 1 接触点
    ├── dataset/                #   Stage 2 训练集
    ├── checkpoints/            #   Stage 3 模型权重
    └── grasps/                 #   Stage 4 抓取位姿
```

## 环境依赖

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 核心 |
| PyTorch | 2.0+ | PointNet++ |
| Isaac Sim | 4.5.0 | 物理仿真 |
| cuRobo | 0.7.x | 运动规划 |
| trimesh | any | Mesh 处理 |
| h5py | any | 数据存储 |

```bash
pip install -r requirements.txt
```

> Isaac Sim 和 cuRobo 通过 Isaac Sim 内置 Python 运行，不需要额外 pip 安装。

## 配置

所有路径和参数集中在 `config.py`:

```python
# 换机器只需修改这几行
OAKINK_DIR = "/home/lyh/Project/OakInk"
ISAAC_SIM_PATH = "/home/lyh/isaac-sim"
```

## 命令参考

| 命令 | 说明 |
|------|------|
| `python run.py --mesh X.obj` | 推理 + 仿真 (全流程) |
| `python run.py --mesh X.obj --no-sim` | 只生成位姿 |
| `python run.py --prepare` | 提取接触点 + 生成训练集 |
| `python run.py --train` | 训练模型 |
| `python run.py --execute X.hdf5` | 直接执行 HDF5 |
| `python run.py --mesh X.obj --headless` | 无头仿真 |
