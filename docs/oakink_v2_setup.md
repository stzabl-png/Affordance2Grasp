# OakInk-v2 集成指南

在另一台电脑上复现 OakInk-v2 → Affordance2Grasp 全流程。

## 前提条件
- Affordance2Grasp 项目已克隆并可运行
- Python 3.10+
- MANO 模型已下载到 `~/Project/mano_v1_2/`

---

## 第 1 步：安装依赖

```bash
pip install oakink2-toolkit manotorch chumpy huggingface_hub
```

如果 Python ≥ 3.12，需要修复 chumpy 兼容性：
```bash
# 找到 chumpy 安装位置
python -c "import chumpy; print(chumpy.__file__)"
# 编辑 ch.py，把 inspect.getargspec 改为 inspect.getfullargspec
```

## 第 2 步：下载 OakInk-v2 数据 (~37GB)

```bash
mkdir -p ~/Project/OakInk2 && cd ~/Project/OakInk2
git clone https://github.com/oakink/OakInk2.git OakInk-v2-hub
cd OakInk-v2-hub

# 下载标注数据
python script/download.py

# 需要下载的数据包（按提示选择）:
#   - anno_preview (标注数据, 必需)
#   - object_raw   (物体 mesh, 必需)
#   - object_affordance (可选)
```

## 第 3 步：提取接触图 (~21小时)

```bash
cd ~/Project/Affordance2Grasp
nohup python data/extract_contacts_v2.py \
  --oakink2_dir ~/Project/OakInk2/OakInk-v2-hub \
  --workers 4 \
  > extract_v2_full.log 2>&1 &

# 查看进度
tail -f extract_v2_full.log
```

输出: `output/contacts_v2/` 下 ~57,000 个 NPZ 文件
支持断点续传，中断后重新运行同一命令即可。

## 第 4 步：构建合并训练集 (~30min)

```bash
cd ~/Project/Affordance2Grasp
python data/build_dataset.py --output_dir output/dataset_v1v2
```

输出: `output/dataset_v1v2/affordance_train.h5` + `affordance_val.h5`
自动合并 `output/contacts/` (v1) + `output/contacts_v2/` (v2)

## 第 5 步：训练模型 (~2小时)

```bash
python model/train_v5.py --save_dir output/checkpoints_v1v2
```

输出: `output/checkpoints_v1v2/best_model.pth` (F1 ~44%)

## 第 6 步：用新模型生成抓取位姿

```bash
# 设为默认模型
cp output/checkpoints_v1v2/best_model.pth output/checkpoints/best_model.pth

# 批量生成 v1 物体抓取
python batch_process.py --force

# 批量生成 v2 物体抓取
python batch_process_v2.py --force
```

输出: `output/grasps/*.hdf5` (每个物体一个)

## 第 7 步：USD 转换 + Sim 测试

```bash
# OBJ → USD 转换 (需要 Isaac Sim)
sim45 assets/convert_all_v2_usd.py

# 单物体 Sim 抓取测试
sim45 sim/run_grasp.py --hdf5 output/grasps/O02_0010_00003_grasp.hdf5
```

---

## 关键路径

| 目录 | 内容 |
|---|---|
| `~/Project/OakInk2/OakInk-v2-hub/` | OakInk-v2 原始数据 |
| `~/Project/mano_v1_2/` | MANO 手部模型 |
| `output/contacts_v2/` | 提取的 v2 接触图 NPZ |
| `output/dataset_v1v2/` | 合并后的 HDF5 训练集 |
| `output/checkpoints_v1v2/` | 训练好的模型 |
| `output/grasps/` | 生成的抓取位姿 HDF5 |

## config.py 关键配置

确保以下路径正确：
```python
OAKINK_DIR = "~/Project/OakInk"           # v1 数据
OAKINK2_OBJ_DIR = "~/Project/OakInk2/OakInk-v2-hub/object_raw/align_ds"  # v2 mesh
CONTACTS_V2_DIR = "output/contacts_v2"    # v2 接触图输出
```
