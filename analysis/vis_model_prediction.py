#!/usr/bin/env python3
"""
交互式可视化 — 模型预测的 Affordance (Model Prediction)

用 PointNet++ checkpoint 推理物体 mesh, 展示预测的接触概率热力图。
支持同时对比 human contact GT (如果有)。

用法:
    cd /home/lyh/Project/Affordance2Grasp

    # 单独看模型预测
    python analysis/vis_model_prediction.py mug --source grab
    python analysis/vis_model_prediction.py A16013 --source v1
    python analysis/vis_model_prediction.py apple --source grab

    # 同时看 GT (左边) + 预测 (右边)
    python analysis/vis_model_prediction.py mug --source grab --compare

颜色 (预测):
    红色 = 高 affordance (预测接触概率高)
    黄色 = 中等 affordance
    灰色 = 低 affordance (非接触)
    绿球 = 预测的受力中心

颜色 (GT, --compare 模式):
    红色 = 真实接触区域
    蓝色 = 原始接触点
"""
import numpy as np
import trimesh
import h5py
import glob
import os
import sys
import argparse
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ============================================================
# Mesh lookup (same as vis_human_contacts.py)
# ============================================================
GRAB_MESH_DIR = os.path.join(config.DATA_HUB, 'meshes', 'grab')
GRAB_CONTACTS_DIR = os.path.join(os.path.dirname(config.CONTACTS_DIR), 'contacts_grab')

GRAB_MESH_MAP = {
    'mug': 'coffeemug',
    'phone': 'phone',
    'fryingpan': 'fryingpan',
    'gamecontroller': 'gamecontroller',
    'stanfordbunny': 'stanfordbunny',
    'piggybank': 'piggybank',
    'doorknob': 'doorknob',
    'waterbottle': 'waterbottle',
    'lightbulb': 'lightbulb',
    'alarmclock': 'alarmclock',
    'eyeglasses': 'eyeglasses',
}


def find_mesh_path(obj_id, source):
    """Find mesh file path."""
    if source == 'grab':
        for name in [obj_id, GRAB_MESH_MAP.get(obj_id, obj_id)]:
            p = os.path.join(GRAB_MESH_DIR, f"{name}.ply")
            if os.path.exists(p):
                return p
        # Fuzzy match
        for f in os.listdir(GRAB_MESH_DIR):
            if obj_id.replace('_', '').lower() in f.replace('_', '').lower():
                return os.path.join(GRAB_MESH_DIR, f)
    else:  # v1
        for ext in ['.obj', '.ply']:
            p = os.path.join(config.MESH_V1_DIR, f"{obj_id}{ext}")
            if os.path.exists(p):
                return p
    return None


def load_gt_contacts(obj_id, source, max_frames=200):
    """Load ground truth contact points for comparison."""
    if source == 'grab':
        pattern = os.path.join(GRAB_CONTACTS_DIR, f"{obj_id}/*/*.hdf5")
    else:
        pattern = os.path.join(config.CONTACTS_DIR, f"**/{obj_id}_*/*.hdf5")

    files = sorted(glob.glob(pattern, recursive=True))[:max_frames]
    pts = []
    for f in files:
        try:
            with h5py.File(f, 'r') as h:
                if 'finger_contact_pts' in h:
                    pts.append(h['finger_contact_pts'][:])
                elif 'finger_contact_points' in h:
                    pts.append(h['finger_contact_points'][:])
        except Exception:
            continue

    if pts:
        return np.vstack(pts)
    return None


def vis_model_prediction(obj_id, source='grab', checkpoint=None, compare=False):
    """Run model inference and visualize prediction on mesh."""

    # ---- Find and load mesh ----
    mesh_path = find_mesh_path(obj_id, source)
    if mesh_path is None:
        print(f"❌ Mesh not found for '{obj_id}' (source={source})")
        return

    print(f"\n{'=' * 50}")
    print(f"  Model Prediction Visualization")
    print(f"{'=' * 50}")
    print(f"  Object:     {obj_id}")
    print(f"  Source:      {source}")
    print(f"  Mesh:        {os.path.basename(mesh_path)}")

    # ---- Load model ----
    if checkpoint is None:
        # Try common checkpoint paths
        candidates = [
            os.path.join(config.OUTPUT_DIR, "checkpoints_v1v2", "best_model.pth"),
            os.path.join(config.OUTPUT_DIR, "checkpoints", "best_model.pth"),
            os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
        ]
        for c in candidates:
            if os.path.exists(c):
                checkpoint = c
                break

    if checkpoint is None or not os.path.exists(checkpoint):
        print(f"❌ Checkpoint not found")
        return

    print(f"  Checkpoint:  {os.path.basename(checkpoint)}")

    from inference.predictor import AffordancePredictor
    predictor = AffordancePredictor(checkpoint=checkpoint)

    # ---- Run inference ----
    points, normals, contact_prob, force_center = predictor.predict(mesh_path)

    n_high = (contact_prob > 0.5).sum()
    n_medium = ((contact_prob > 0.3) & (contact_prob <= 0.5)).sum()
    print(f"\n  Prediction results:")
    print(f"    Points:       {len(points)}")
    print(f"    High (>0.5):  {n_high} ({100*n_high/len(points):.1f}%)")
    print(f"    Medium (>0.3): {n_medium}")
    print(f"    Max prob:     {contact_prob.max():.3f}")
    print(f"    Mean prob:    {contact_prob.mean():.3f}")
    if force_center is not None:
        print(f"    Force center: [{force_center[0]:.4f}, {force_center[1]:.4f}, {force_center[2]:.4f}]")

    # ---- Build visualization ----
    geometries = []

    # Load full mesh for display
    mesh = trimesh.load(mesh_path, force='mesh')
    full_verts = np.array(mesh.vertices)
    full_faces = np.array(mesh.faces)

    # Map sampled point predictions back to mesh vertices
    tree = cKDTree(points)
    dists, indices = tree.query(full_verts)

    # Each mesh vertex gets the prediction of nearest sampled point
    vert_probs = contact_prob[indices]
    # Fade out distant vertices
    vert_probs[dists > 0.01] *= 0.5

    # Color: heatmap  gray(0) → yellow(0.3) → red(1.0)
    colors = np.ones((len(full_verts), 3)) * 0.75  # base gray
    for vi in range(len(full_verts)):
        p = vert_probs[vi]
        if p > 0.5:
            # Red
            colors[vi] = [1.0, 0.15 * (1 - p), 0.05]
        elif p > 0.3:
            # Yellow-orange
            t = (p - 0.3) / 0.2
            colors[vi] = [1.0, 0.8 - 0.6 * t, 0.1]
        elif p > 0.1:
            # Light yellow
            t = (p - 0.1) / 0.2
            colors[vi] = [0.8 + 0.2 * t, 0.8 + 0.0 * t, 0.6 - 0.5 * t]

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(full_verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(full_faces)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    geometries.append(o3d_mesh)

    # Force center (green sphere)
    if force_center is not None:
        fc_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        fc_sphere.translate(force_center)
        fc_sphere.paint_uniform_color([0.0, 1.0, 0.3])
        geometries.append(fc_sphere)

    # ---- Compare with GT ----
    if compare:
        gt_contacts = load_gt_contacts(obj_id, source)
        if gt_contacts is not None:
            # Offset GT mesh to the right for side-by-side
            offset = np.array([full_verts.ptp(axis=0)[0] * 1.5, 0, 0])

            gt_mesh = o3d.geometry.TriangleMesh()
            gt_mesh.vertices = o3d.utility.Vector3dVector(full_verts + offset)
            gt_mesh.triangles = o3d.utility.Vector3iVector(full_faces)
            gt_mesh.compute_vertex_normals()

            # GT coloring
            gt_dists, _ = cKDTree(gt_contacts).query(full_verts)
            gt_mask = gt_dists < config.CONTACT_RADIUS
            gt_colors = np.ones((len(full_verts), 3)) * 0.8
            gt_colors[gt_mask] = [1.0, 0.15, 0.05]
            gt_mesh.vertex_colors = o3d.utility.Vector3dVector(gt_colors)
            geometries.append(gt_mesh)

            # GT contact point cloud
            step = max(1, len(gt_contacts) // 3000)
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(gt_contacts[::step] + offset)
            gt_pcd.paint_uniform_color([0.0, 0.4, 1.0])
            geometries.append(gt_pcd)

            print(f"\n  GT comparison: {gt_mask.sum()}/{len(full_verts)} contact verts")
            print(f"  (Left = Model Prediction, Right = Human GT)")
        else:
            print(f"\n  ⚠️ No GT data found for '{obj_id}'")

    # ---- Display ----
    title = f"Model Prediction — {obj_id} [{source}]"
    if compare:
        title += " | Left=Pred Right=GT"

    print(f"\n🖱️ 交互操作:")
    print(f"  左键拖动 = 旋转   滚轮 = 缩放   Shift+左键 = 平移   Q/Esc = 关闭")
    print(f"\n  🔴 红色 = 高 affordance (模型预测)")
    print(f"  🟡 黄色 = 中等 affordance")
    print(f"  ⬜ 灰色 = 低 affordance")
    print(f"  🟢 绿球 = 预测受力中心\n")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1400,
        height=800,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="可视化模型预测的 Affordance (Model Prediction)")
    parser.add_argument("obj_id", type=str,
                        help="Object name (e.g. mug, apple, A16013)")
    parser.add_argument("--source", type=str, default=None,
                        choices=['v1', 'grab'],
                        help="Data source (default: auto-detect)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path")
    parser.add_argument("--compare", action="store_true",
                        help="Show GT comparison side-by-side")
    args = parser.parse_args()

    # Auto-detect source
    if args.source is None:
        if find_mesh_path(args.obj_id, 'v1'):
            args.source = 'v1'
        elif find_mesh_path(args.obj_id, 'grab'):
            args.source = 'grab'
        else:
            print(f"❌ Cannot find mesh for '{args.obj_id}'")
            sys.exit(1)

    vis_model_prediction(args.obj_id, args.source, args.checkpoint, args.compare)
