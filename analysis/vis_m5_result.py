#!/usr/bin/env python3
"""
M5 Phase 3 Interactive Result Visualization
===========================================
对比同一个物体上的：
1. Human Prior (输入意图)
2. Robot GT (实际仿真抓取真实值)
3. Model Prediction (模型预测的机器抓取可行度)
"""

import os
import sys
import argparse
import h5py
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
from pointnet2 import PointNet2Seg

def get_color_map(values, cmap_name='jet', vmin=0, vmax=1):
    cmap = plt.get_cmap(cmap_name)
    # 归一化
    norm_vals = np.clip((values - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colors = cmap(norm_vals)[:, :3]  # Nx3 RGB
    return colors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', type=str, default='A16013')
    parser.add_argument('--model_path', type=str, default='output/checkpoints_m5/best_m5_model.pth')
    parser.add_argument('--data_dir', type=str, default='data_hub/training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Model
    model = PointNet2Seg(num_classes=1, in_channel=7).to(device)
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Warning: Model not found at {args.model_path}. Using random weights.")
    model.eval()

    # Allow reading from human_prior if not in training
    data_path = os.path.join(args.data_dir, f"{args.obj_id}.hdf5")
    if not os.path.exists(data_path):
        # Try grab prefix
        data_path = os.path.join(args.data_dir, f"grab_{args.obj_id}.hdf5")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        sys.exit(1)

    with h5py.File(data_path, 'r') as f:
        pc = f['point_cloud'][()]
        nrm = f['normals'][()]
        hp = f['human_prior'][()]
        has_gt = 'robot_gt' in f
        robot_gt = f['robot_gt'][()] if has_gt else np.zeros_like(hp)

    
    # Pre-process for inference
    hp_feat = hp.reshape(-1, 1).astype(np.float32)
    features = np.concatenate([pc, nrm, hp_feat], axis=-1)
    
    pc_t = torch.from_numpy(pc).unsqueeze(0).float().to(device)
    feat_t = torch.from_numpy(features).unsqueeze(0).float().to(device)

    # Inference
    with torch.no_grad():
        pred_logits = model(pc_t, feat_t)
        pred = pred_logits.squeeze().cpu().numpy()  # (N,)

    # 将稀疏点云预测映射到超高精度 Mesh 上
    mesh_paths = [
        os.path.join("data_hub", "meshes", "v1", f"{args.obj_id}.obj"),
        os.path.join("data_hub", "meshes", "v2", f"{args.obj_id}.ply"),
        os.path.join("data_hub", "meshes", "grab", f"{args.obj_id}.stl"),
        os.path.join("data_hub", "meshes", "grab", f"grab_{args.obj_id}.stl"),
    ]
    
    mesh = None
    for mp in mesh_paths:
        if os.path.exists(mp):
            mesh = o3d.io.read_triangle_mesh(mp)
            break
            
    if mesh is None or len(mesh.vertices) == 0:
        print("未找到原模型 Mesh，回退到稀疏点云显示")
        # 3. Prediction (Only)
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(pc)
        pred_colors = get_color_map(pred, cmap_name='jet', vmin=0, vmax=1)
        pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)
        disp_geom = pcd_pred
    else:
        # Mesh 映射: 给 Mesh 的每一个顶点寻找最近的预测点
        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices)
        
        # KDTree
        pcd_sparse = o3d.geometry.PointCloud()
        pcd_sparse.points = o3d.utility.Vector3dVector(pc)
        kdtree = o3d.geometry.KDTreeFlann(pcd_sparse)
        
        mesh_pred = np.zeros(len(vertices))
        for i, v in enumerate(vertices):
            _, idx, _ = kdtree.search_knn_vector_3d(v, 1) # 找最近的 1 个点
            mesh_pred[i] = pred[idx[0]]
            
        mesh_colors = get_color_map(mesh_pred, cmap_name='jet', vmin=0, vmax=1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
        disp_geom = mesh

    mae = np.abs(pred - robot_gt).mean()

    # Coordinate Frames for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

    print("=========================================")
    print(" 交互式可视化说明 ")
    print(" 已将预测热力图裹在 高清 Mesh 模型 上！")
    print(f"\n >>> MAE 均绝对误差: {mae:.4f}")
    print("=========================================")

    window_name = f"{args.obj_id} | High-Res Mesh Affordance (MAE={mae:.3f})"
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(disp_geom)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.95, 0.95, 0.95]) # 干净的白底
    opt.mesh_show_back_face = True
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
