#!/usr/bin/env python3
"""
3-Panel 可视化: Human Prior | Robot GT | Model Prediction

每个物体输出一张 3 列拼合图:
  左: Human Prior (二值 0/1, 蓝=0 红=1)
  中: Robot GT (连续 0~1, jet colormap)
  右: 模型预测 (连续 0~1, jet colormap)

用法:
    cd /home/lyh/Project/Affordance2Grasp
    # 单个物体
    python3 tools/vis_3panel.py --obj A01001
    # 批量 (输出到文件夹)
    python3 tools/vis_3panel.py --batch --out output/vis_3panel
"""
import os, sys, argparse, glob
import numpy as np
import trimesh
import h5py
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model'))
from pointnet2 import PointNet2Seg

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(PROJ, 'data_hub', 'meshes', 'v1')
TRAIN_DIR = os.path.join(PROJ, 'data_hub', 'training_m5')
CKPT = os.path.join(PROJ, 'output', 'checkpoints_m5', 'best_m5_model.pth')
N_POINTS = 4096


def load_training_data(obj_id):
    """从 training_m5 加载预处理好的训练数据."""
    path = os.path.join(TRAIN_DIR, f'{obj_id}.hdf5')
    if not os.path.exists(path):
        return None
    with h5py.File(path, 'r') as f:
        return {
            'pc': f['point_cloud'][()].astype(np.float32),
            'normals': f['normals'][()].astype(np.float32),
            'human_prior': f['human_prior'][()].astype(np.float32),
            'robot_gt': f['robot_gt'][()].astype(np.float32),
        }


def predict_affordance(model, pc, normals, hp, device):
    """模型推理."""
    features = np.concatenate([pc, normals, hp.reshape(-1, 1)], axis=-1)
    pts_t = torch.from_numpy(pc).unsqueeze(0).to(device)
    feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(pts_t, feat_t).squeeze(0).cpu().numpy()
    return pred


def render_pointcloud(ax, points, values, title, cmap_name='jet', vmin=0, vmax=1, 
                      elev=25, azim=135, binary=False):
    """在 matplotlib 3D subplot 上渲染点云."""
    cmap = plt.get_cmap(cmap_name)
    
    if binary:
        # 二值显示: 灰=0, 红=1
        colors = np.zeros((len(values), 4))
        colors[values < 0.5] = [0.75, 0.75, 0.85, 1.0]  # 浅灰蓝
        colors[values >= 0.5] = [0.9, 0.15, 0.15, 1.0]   # 红
    else:
        colors = cmap(np.clip(values, vmin, vmax))
    
    # 排序: 低值先画, 高值后画 (高值覆盖)
    order = np.argsort(values)
    points = points[order]
    colors = colors[order]
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors, s=1.5, alpha=0.9, edgecolors='none')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.view_init(elev=elev, azim=azim)
    
    # 等比例
    extents = points.max(axis=0) - points.min(axis=0)
    max_ext = extents.max() * 0.6
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    ax.set_xlim(center[0] - max_ext, center[0] + max_ext)
    ax.set_ylim(center[1] - max_ext, center[1] + max_ext)
    ax.set_zlim(center[2] - max_ext, center[2] + max_ext)
    ax.set_axis_off()


def vis_single(obj_id, model, device, out_dir=None, show=False):
    """处理并可视化单个物体."""
    data = load_training_data(obj_id)
    if data is None:
        print(f"  ⚠️ {obj_id}: 无训练数据, 跳过")
        return False
    
    pc = data['pc']
    normals = data['normals']
    hp = data['human_prior']
    rgt = data['robot_gt']
    
    # 模型预测
    pred = predict_affordance(model, pc, normals, hp, device)
    
    has_hp = np.any(hp > 0)
    has_rgt = np.any(rgt > 0.01)
    
    # 密集采样用于可视化
    mesh_path = os.path.join(MESH_DIR, f'{obj_id}.obj')
    if os.path.exists(mesh_path):
        mesh = trimesh.load(mesh_path, force='mesh')
        N_VIS = 20000
        vis_pc, _ = trimesh.sample.sample_surface(mesh, N_VIS)
        vis_pc = vis_pc.astype(np.float32)
        
        # KNN 插值到密集点云
        from scipy.spatial import cKDTree
        tree = cKDTree(pc)
        _, idx = tree.query(vis_pc, k=3)
        dists = np.linalg.norm(vis_pc[:, None, :] - pc[idx], axis=2)
        weights = 1.0 / (dists + 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        
        hp_dense = np.sum(hp[idx] * weights, axis=1)
        rgt_dense = np.sum(rgt[idx] * weights, axis=1)
        pred_dense = np.sum(pred[idx] * weights, axis=1)
    else:
        vis_pc = pc
        hp_dense = hp
        rgt_dense = rgt
        pred_dense = pred
    
    # 画图
    fig = plt.figure(figsize=(18, 6), facecolor='white')
    
    hp_label = f'Human Prior' + ('' if has_hp else ' (N/A)')
    rgt_label = f'Robot GT' + ('' if has_rgt else ' (N/A)')
    pred_label = f'Model Prediction\n(max={pred_dense.max():.2f})'
    
    ax1 = fig.add_subplot(131, projection='3d')
    render_pointcloud(ax1, vis_pc, hp_dense, hp_label, binary=True)
    
    ax2 = fig.add_subplot(132, projection='3d')
    render_pointcloud(ax2, vis_pc, rgt_dense, rgt_label)
    
    ax3 = fig.add_subplot(133, projection='3d')
    render_pointcloud(ax3, vis_pc, pred_dense, pred_label)
    
    fig.suptitle(f'{obj_id}', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{obj_id}_3panel.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✅ {obj_id}: saved → {out_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description='3-Panel 可视化')
    parser.add_argument('--obj', type=str, help='单个物体 ID')
    parser.add_argument('--batch', action='store_true', help='批量处理所有训练物体')
    parser.add_argument('--out', type=str, default=os.path.join(PROJ, 'output', 'vis_3panel'),
                        help='输出目录')
    parser.add_argument('--show', action='store_true', help='交互显示 (单个模式)')
    args = parser.parse_args()
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet2Seg(num_classes=1, in_channel=7).to(device)
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"✅ 模型加载: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    
    if args.obj:
        vis_single(args.obj, model, device, out_dir=args.out, show=args.show)
    elif args.batch:
        files = sorted(glob.glob(os.path.join(TRAIN_DIR, '*.hdf5')))
        total = 0
        success = 0
        for f in files:
            obj_id = os.path.splitext(os.path.basename(f))[0]
            total += 1
            if vis_single(obj_id, model, device, out_dir=args.out):
                success += 1
        print(f"\n{'='*50}")
        print(f"  完成! {success}/{total} 个物体可视化")
        print(f"  输出: {args.out}")
        print(f"{'='*50}")
    else:
        print("请指定 --obj 或 --batch")


if __name__ == '__main__':
    main()
