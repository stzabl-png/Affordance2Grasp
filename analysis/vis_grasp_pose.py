#!/usr/bin/env python3
"""
交互式可视化 — 抓取位姿 (Grasp Pose) — Open3D 版

展示: mesh + affordance热力图 + 夹爪线框 + 受力中心

用法:
    cd /home/lyh/Project/Affordance2Grasp

    python analysis/vis_grasp_pose.py coffeemug
    python analysis/vis_grasp_pose.py apple
    python analysis/vis_grasp_pose.py waterbottle
    python analysis/vis_grasp_pose.py wineglass
    python analysis/vis_grasp_pose.py A16013

颜色:
    红色→黄色→灰色 = affordance 热力图 (高→中→低)
    绿球 = 受力中心
    彩色线框 = 夹爪候选 (最优=红, 其余=蓝/橙/紫)
"""
import numpy as np
import trimesh
import h5py
import os
import sys
import argparse
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

GRAB_MESH_DIR = os.path.join(config.DATA_HUB, 'meshes', 'grab')
GRAB_MESH_MAP = {'mug': 'coffeemug'}

CANDIDATE_COLORS = [
    [1.0, 0.2, 0.1],   # 红 (best)
    [0.2, 0.5, 1.0],   # 蓝
    [1.0, 0.6, 0.1],   # 橙
    [0.6, 0.2, 0.8],   # 紫
    [0.2, 0.8, 0.4],   # 绿
    [0.8, 0.8, 0.2],   # 黄
]


def find_mesh_path(obj_id):
    """Find mesh file."""
    for ext in ['.obj', '.ply']:
        p = os.path.join(config.MESH_V1_DIR, f"{obj_id}{ext}")
        if os.path.exists(p):
            return p
    search_name = GRAB_MESH_MAP.get(obj_id, obj_id)
    for ext in ['.ply', '.obj']:
        p = os.path.join(GRAB_MESH_DIR, f"{search_name}{ext}")
        if os.path.exists(p):
            return p
    if os.path.exists(GRAB_MESH_DIR):
        for f in os.listdir(GRAB_MESH_DIR):
            if obj_id.replace('_', '').lower() in f.replace('_', '').lower():
                return os.path.join(GRAB_MESH_DIR, f)
    return None


def make_gripper(grasp_point, rotation, gripper_width, color):
    """Create gripper wireframe as LineSet."""
    x_open = rotation[:, 0]
    z_approach = rotation[:, 2]
    half_w = gripper_width / 2
    finger_len = 0.04

    lines = []
    points = []

    for sign in [1, -1]:
        tip = grasp_point + sign * x_open * half_w
        base = tip - z_approach * finger_len
        idx = len(points)
        points.extend([tip, base])
        lines.append([idx, idx + 1])

    # Crossbar at finger base
    base_left = grasp_point + x_open * half_w - z_approach * finger_len
    base_right = grasp_point - x_open * half_w - z_approach * finger_len
    idx = len(points)
    points.extend([base_left, base_right])
    lines.append([idx, idx + 1])

    # Approach axis (dashed — just one line)
    approach_start = grasp_point - z_approach * finger_len
    approach_end = approach_start - z_approach * 0.05
    idx = len(points)
    points.extend([approach_start, approach_end])
    lines.append([idx, idx + 1])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(points))
    ls.lines = o3d.utility.Vector2iVector(np.array(lines))
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def vis_grasp_pose(obj_id):
    """Interactive Open3D visualization of grasp candidates."""

    mesh_path = find_mesh_path(obj_id)
    if mesh_path is None:
        print(f"❌ Mesh not found: {obj_id}")
        return

    grasp_path = os.path.join(config.GRASPS_DIR, f"{obj_id}_grasp.hdf5")
    if not os.path.exists(grasp_path):
        print(f"❌ Grasp HDF5 not found: {grasp_path}")
        print(f"   先跑: python -m inference.grasp_pose --mesh {mesh_path}")
        return

    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # Load grasp data
    with h5py.File(grasp_path, 'r') as f:
        points = f['affordance/points'][:]
        probs = f['affordance/contact_prob'][:]
        force_center = f['affordance/force_center'][:] if 'affordance/force_center' in f else None

        candidates = []
        n_cands = f['candidates'].attrs.get('n_candidates', 0)
        for i in range(n_cands):
            key = f'candidates/candidate_{i}'
            if key not in f:
                continue
            c = f[key]
            candidates.append({
                'name': str(c.attrs['name']),
                'score': float(c.attrs['score']),
                'gripper_width': float(c.attrs['gripper_width']),
                'grasp_point': c['grasp_point'][:],
                'rotation': c['rotation'][:],
            })

    print(f"\n{'=' * 50}")
    print(f"  Grasp Pose Visualization (Open3D)")
    print(f"{'=' * 50}")
    print(f"  Object:      {obj_id}")
    print(f"  Mesh:        {os.path.basename(mesh_path)} ({len(verts)} verts)")
    print(f"  Candidates:  {len(candidates)}")
    for i, c in enumerate(candidates):
        marker = "⭐" if i == 0 else "  "
        print(f"  {marker} [{i+1}] {c['name']:>12s}  score={c['score']:.1f}  width={c['gripper_width']*100:.1f}cm")

    # ---- Build affordance heatmap on mesh ----
    tree = cKDTree(points)
    dists, indices = tree.query(verts)
    vert_probs = probs[indices]
    vert_probs[dists > 0.01] *= 0.5

    colors = np.ones((len(verts), 3)) * 0.75
    for vi in range(len(verts)):
        p = vert_probs[vi]
        if p > 0.5:
            colors[vi] = [1.0, 0.15 * (1 - p), 0.05]
        elif p > 0.3:
            t = (p - 0.3) / 0.2
            colors[vi] = [1.0, 0.8 - 0.6 * t, 0.1]
        elif p > 0.1:
            t = (p - 0.1) / 0.2
            colors[vi] = [0.8 + 0.2 * t, 0.8, 0.6 - 0.5 * t]

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    geometries = [o3d_mesh]

    # ---- Force center (green sphere) ----
    if force_center is not None:
        fc_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
        fc_sphere.translate(force_center)
        fc_sphere.paint_uniform_color([0.1, 0.9, 0.3])
        fc_sphere.compute_vertex_normals()
        geometries.append(fc_sphere)

    # ---- Gripper wireframes ----
    for i, c in enumerate(candidates):
        color = CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)]
        gripper = make_gripper(c['grasp_point'], c['rotation'],
                               c['gripper_width'], color)
        geometries.append(gripper)

        # Grasp point sphere
        gp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        gp_sphere.translate(c['grasp_point'])
        gp_sphere.paint_uniform_color(color)
        gp_sphere.compute_vertex_normals()
        geometries.append(gp_sphere)

    # ---- Display ----
    print(f"\n🖱️ 交互操作:")
    print(f"  左键拖动 = 旋转   滚轮 = 缩放   Shift+左键 = 平移   Q/Esc = 关闭")
    print(f"\n  🔴 红色 mesh = 高 affordance")
    print(f"  🟡 黄色 mesh = 中等 affordance")
    print(f"  ⬜ 灰色 mesh = 低 affordance")
    print(f"  🟢 绿球 = 受力中心")
    print(f"  线框 = 夹爪候选 (红=最优)\n")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Grasp Pose — {obj_id} | {len(candidates)} candidates",
        width=1400, height=800,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="交互式抓取位姿可视化 (Open3D)")
    parser.add_argument("obj_id", type=str, help="Object ID (e.g. coffeemug, A16013)")
    args = parser.parse_args()
    vis_grasp_pose(args.obj_id)
