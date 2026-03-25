#!/usr/bin/env python3
"""
交互式可视化 M1 接触提取结果

用 Open3D 展示物体 mesh，手指接触区域标红。
支持鼠标旋转、缩放、平移。

用法:
    cd /home/lyh/Project/Affordance2Grasp
    python analysis/vis_contacts.py A16013
    python analysis/vis_contacts.py A16013 --max_frames 100
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


def vis_contacts(obj_id, max_frames=100):
    # ---- 加载 mesh ----
    mesh_path = None
    for ext in ['.obj', '.ply']:
        p = os.path.join(config.MESH_V1_DIR, f"{obj_id}{ext}")
        if os.path.exists(p):
            mesh_path = p
            break
    if mesh_path is None:
        print(f"❌ Mesh not found: {obj_id}")
        return

    mesh = trimesh.load(mesh_path, force='mesh')
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    print(f"Mesh: {mesh_path} ({len(verts)} verts, {len(faces)} faces)")

    # ---- 收集 HDF5 接触点 ----
    pattern = os.path.join(config.CONTACTS_DIR, f"**/{obj_id}_*/*.hdf5")
    hdf5_files = sorted(glob.glob(pattern, recursive=True))
    if not hdf5_files:
        print(f"❌ No HDF5 files found for {obj_id}")
        return

    all_contacts = []
    finger_stats = {f: 0 for f in ["thumb", "index", "middle", "ring", "pinky"]}
    total_frames = 0

    for f in hdf5_files[:max_frames]:
        try:
            with h5py.File(f, 'r') as h:
                all_contacts.append(h['finger_contact_points'][:])
                for fn in finger_stats:
                    if h.attrs.get(f'finger_{fn}', False):
                        finger_stats[fn] += 1
                total_frames += 1
        except Exception:
            continue

    all_contacts = np.vstack(all_contacts)

    # ---- 标记顶点接触 ----
    dists, _ = cKDTree(all_contacts).query(verts)
    contact_mask = dists < config.CONTACT_RADIUS
    n_contact = int(contact_mask.sum())

    print(f"\nObject: {obj_id}")
    print(f"  Frames loaded: {total_frames}")
    print(f"  Raw contact pts: {all_contacts.shape[0]}")
    print(f"  Contact verts: {n_contact}/{len(verts)} ({100*n_contact/len(verts):.1f}%)")
    print(f"  Finger participation:")
    for fn, cnt in finger_stats.items():
        print(f"    {fn:>8s}: {cnt}/{total_frames} frames")

    # ---- 构建 Open3D mesh ----
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()

    # 颜色: 灰色 = 非接触, 红色 = 手指接触
    colors = np.ones((len(verts), 3)) * 0.8  # 灰色
    colors[contact_mask] = [1.0, 0.15, 0.05]  # 红色
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # ---- 接触点点云 (小蓝点, 原始接触位置) ----
    contact_pcd = o3d.geometry.PointCloud()
    contact_pcd.points = o3d.utility.Vector3dVector(all_contacts)
    contact_pcd.paint_uniform_color([0.0, 0.4, 1.0])  # 蓝色

    # ---- 可视化 ----
    print(f"\n🖱️ Open3D 交互窗口:")
    print(f"  左键拖动 = 旋转")
    print(f"  滚轮 = 缩放")
    print(f"  Shift+左键 = 平移")
    print(f"  Q/Esc = 关闭")
    print(f"\n  红色 = 手指接触区域 (mesh 顶点)")
    print(f"  蓝色 = 原始接触点 (物体表面)")
    print(f"  灰色 = 非接触表面")

    o3d.visualization.draw_geometries(
        [o3d_mesh, contact_pcd],
        window_name=f"{obj_id} — Finger Contacts (palm discarded) | {n_contact}/{len(verts)} verts",
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive contact visualization")
    parser.add_argument("obj_id", type=str, help="Object ID (e.g. A16013)")
    parser.add_argument("--max_frames", type=int, default=100, help="Max frames to load")
    args = parser.parse_args()
    vis_contacts(args.obj_id, args.max_frames)
