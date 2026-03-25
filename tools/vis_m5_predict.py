#!/usr/bin/env python3
"""
用 M5 模型对物体做 affordance 预测, Open3D 可视化.

用法:
    cd /home/lyh/Project/Affordance2Grasp
    python3 tools/vis_m5_predict.py --obj A01001
    python3 tools/vis_m5_predict.py --obj A16013
"""
import os, sys, argparse
import numpy as np
import trimesh
import h5py
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model'))
from pointnet2 import PointNet2Seg

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(PROJ, 'data_hub', 'meshes', 'v1')
HP_DIR = os.path.join(PROJ, 'data_hub', 'human_prior')
CKPT = os.path.join(PROJ, 'output', 'checkpoints_m5', 'best_m5_model.pth')
N_POINTS = 4096


def load_human_prior(obj_id):
    for name in [f'{obj_id}.hdf5', f'grab_{obj_id}.hdf5']:
        path = os.path.join(HP_DIR, name)
        if os.path.exists(path):
            with h5py.File(path, 'r') as f:
                return f['point_cloud'][()].astype(np.float32), \
                       f['normals'][()].astype(np.float32), \
                       f['human_prior'][()].astype(np.float32)
    return None, None, None


def predict(obj_id):
    import open3d as o3d
    from scipy.spatial import cKDTree

    mesh_path = os.path.join(MESH_DIR, f'{obj_id}.obj')
    if not os.path.exists(mesh_path):
        print(f"❌ mesh 不存在: {mesh_path}"); return

    mesh = trimesh.load(mesh_path, force='mesh')

    # 采样 4096 点用于模型推理
    hp_pc, hp_nrm, hp_labels = load_human_prior(obj_id)
    if hp_pc is not None:
        pc, normals, hp = hp_pc, hp_nrm, hp_labels
        if len(pc) != N_POINTS:
            idx = np.random.choice(len(pc), N_POINTS, replace=len(pc) < N_POINTS)
            pc, normals, hp = pc[idx], normals[idx], hp[idx]
        print(f"📂 使用 Human Prior 点云")
    else:
        pc, face_idx = trimesh.sample.sample_surface(mesh, N_POINTS)
        pc = pc.astype(np.float32)
        normals = mesh.face_normals[face_idx].astype(np.float32)
        hp = np.zeros(N_POINTS, dtype=np.float32)
        print(f"📂 从 mesh 采样点云")

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet2Seg(num_classes=1, in_channel=7).to(device)
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"✅ 模型加载: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    # 推理 (4096 点)
    pts_t = torch.from_numpy(pc).unsqueeze(0).to(device)
    features = np.concatenate([pc, normals, hp.reshape(-1, 1)], axis=-1)
    feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_sparse = model(pts_t, feat_t).squeeze(0).cpu().numpy()

    # 密集采样 30K 点用于可视化
    N_DENSE = 30000
    dense_pc, dense_fidx = trimesh.sample.sample_surface(mesh, N_DENSE)
    dense_pc = dense_pc.astype(np.float32)

    # 用 KNN 把 4096 点的预测插值到 30K 点
    tree = cKDTree(pc)
    dists, idx = tree.query(dense_pc, k=3)
    weights = 1.0 / (dists + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    pred = np.sum(pred_sparse[idx] * weights, axis=1)

    print(f"\n📊 预测统计 ({N_DENSE} 点):")
    print(f"   min={pred.min():.3f}  max={pred.max():.3f}  mean={pred.mean():.3f}")
    print(f"   > 0.3: {(pred > 0.3).sum()} 点")
    print(f"   > 0.5: {(pred > 0.5).sum()} 点")

    # Open3D: 热力图 蓝→红
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dense_pc)

    # matplotlib jet colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap('jet')
    colors = cmap(pred)[:, :3]  # (N, 3) RGB
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"M5 — {obj_id} (max={pred.max():.2f})", width=1200, height=800)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 4.0
    opt.background_color = np.array([1, 1, 1])
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', required=True, help='物体 ID')
    args = parser.parse_args()
    predict(args.obj)
