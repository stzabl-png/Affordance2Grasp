#!/usr/bin/env python3
"""
交互式可视化 — 人手真实接触点 (Human Contact Ground Truth)

支持两种数据源:
  1. OakInk v1: mesh 在 data_hub/meshes/v1/,  contacts 在 output/contacts/
  2. GRAB:      mesh 在 data_hub/meshes/grab/, contacts 在 output/contacts_grab/

用法:
    cd /home/lyh/Project/Affordance2Grasp

    # OakInk v1 物体
    python analysis/vis_human_contacts.py A16013
    python analysis/vis_human_contacts.py A02029 --max_frames 200

    # GRAB 物体
    python analysis/vis_human_contacts.py mug --source grab
    python analysis/vis_human_contacts.py apple --source grab
    python analysis/vis_human_contacts.py wineglass --source grab

    # 自动检测 (先找 v1, 再找 grab)
    python analysis/vis_human_contacts.py mug

颜色:
    红色 = 被人手指接触的 mesh 顶点区域
    蓝色 = 原始接触点 (从 HDF5 读取)
    灰色 = 未接触的表面
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
# Data source definitions
# ============================================================
SOURCES = {
    'v1': {
        'mesh_dir': config.MESH_V1_DIR,
        'contacts_dir': config.CONTACTS_DIR,
        'contact_key': 'finger_contact_points',   # OakInk v1 HDF5 key
        'pattern': '**/{obj_id}_*/*.hdf5',
        'mesh_exts': ['.obj', '.ply'],
    },
    'grab': {
        'mesh_dir': os.path.join(config.DATA_HUB, 'meshes', 'grab'),
        'contacts_dir': os.path.join(os.path.dirname(config.CONTACTS_DIR), 'contacts_grab'),
        'contact_key': 'finger_contact_pts',       # GRAB HDF5 key
        'pattern': '{obj_id}/*/*.hdf5',
        'mesh_exts': ['.ply'],
    },
}

# GRAB mesh name mapping (object name → mesh filename, without extension)
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


def find_mesh(obj_id, source_cfg):
    """Find mesh file for the given object."""
    mesh_dir = source_cfg['mesh_dir']

    # Direct name match
    for ext in source_cfg['mesh_exts']:
        p = os.path.join(mesh_dir, f"{obj_id}{ext}")
        if os.path.exists(p):
            return p

    # GRAB name mapping
    mapped = GRAB_MESH_MAP.get(obj_id, obj_id)
    for ext in source_cfg['mesh_exts']:
        p = os.path.join(mesh_dir, f"{mapped}{ext}")
        if os.path.exists(p):
            return p

    # Fuzzy match (remove underscores)
    if os.path.exists(mesh_dir):
        for f in os.listdir(mesh_dir):
            base = os.path.splitext(f)[0]
            if obj_id.replace('_', '').lower() == base.replace('_', '').lower():
                return os.path.join(mesh_dir, f)

    return None


def find_source(obj_id):
    """Auto-detect data source for the given object."""
    for name, cfg in SOURCES.items():
        if find_mesh(obj_id, cfg) is not None:
            pattern = os.path.join(cfg['contacts_dir'],
                                   cfg['pattern'].format(obj_id=obj_id))
            files = glob.glob(pattern, recursive=True)
            if files:
                return name
    return None


def vis_human_contacts(obj_id, source_name=None, max_frames=200,
                       contact_radius=None):
    """Visualize human contact ground truth on object mesh."""

    if contact_radius is None:
        contact_radius = config.CONTACT_RADIUS

    # Auto-detect source
    if source_name is None:
        source_name = find_source(obj_id)
        if source_name is None:
            print(f"❌ Cannot find data for '{obj_id}' in any source (v1, grab)")
            print(f"   Available v1 objects: ls {config.MESH_V1_DIR}")
            print(f"   Available GRAB objects: ls {SOURCES['grab']['mesh_dir']}")
            return
        print(f"  Auto-detected source: {source_name}")

    src = SOURCES[source_name]

    # ---- Load mesh ----
    mesh_path = find_mesh(obj_id, src)
    if mesh_path is None:
        print(f"❌ Mesh not found for '{obj_id}' in {src['mesh_dir']}")
        return

    mesh = trimesh.load(mesh_path, force='mesh')
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    print(f"  Mesh: {os.path.basename(mesh_path)} ({len(verts)} verts, {len(faces)} faces)")

    # ---- Collect HDF5 contact points ----
    pattern = os.path.join(src['contacts_dir'],
                           src['pattern'].format(obj_id=obj_id))
    hdf5_files = sorted(glob.glob(pattern, recursive=True))
    if not hdf5_files:
        print(f"❌ No contact HDF5 files found")
        print(f"   Pattern: {pattern}")
        return

    contact_key = src['contact_key']
    # Try fallback key if primary doesn't exist
    fallback_key = 'finger_contact_pts' if contact_key == 'finger_contact_points' else 'finger_contact_points'

    all_contacts = []
    loaded = 0

    for f in hdf5_files[:max_frames]:
        try:
            with h5py.File(f, 'r') as h:
                if contact_key in h:
                    all_contacts.append(h[contact_key][:])
                elif fallback_key in h:
                    all_contacts.append(h[fallback_key][:])
                else:
                    continue
                loaded += 1
        except Exception:
            continue

    if not all_contacts:
        print(f"❌ No valid contact data loaded (tried {len(hdf5_files)} files)")
        return

    all_contacts = np.vstack(all_contacts)

    # ---- Compute contact region on mesh ----
    dists, _ = cKDTree(all_contacts).query(verts)
    contact_mask = dists < contact_radius
    n_contact = int(contact_mask.sum())

    print(f"\n{'=' * 50}")
    print(f"  Human Contact Visualization")
    print(f"{'=' * 50}")
    print(f"  Object:          {obj_id}")
    print(f"  Source:           {source_name}")
    print(f"  Frames loaded:   {loaded}/{len(hdf5_files)}")
    print(f"  Raw contact pts: {all_contacts.shape[0]:,}")
    print(f"  Contact verts:   {n_contact}/{len(verts)} ({100 * n_contact / len(verts):.1f}%)")
    print(f"  Contact radius:  {contact_radius * 1000:.1f}mm")

    # ---- Build Open3D mesh ----
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()

    # Colors: gray=no contact, red=finger contact
    colors = np.ones((len(verts), 3)) * 0.8
    colors[contact_mask] = [1.0, 0.15, 0.05]
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # ---- Contact point cloud (blue dots) ----
    # Subsample for performance
    step = max(1, len(all_contacts) // 5000)
    contact_pcd = o3d.geometry.PointCloud()
    contact_pcd.points = o3d.utility.Vector3dVector(all_contacts[::step])
    contact_pcd.paint_uniform_color([0.0, 0.4, 1.0])

    # ---- Visualize ----
    print(f"\n🖱️ 交互操作:")
    print(f"  左键拖动 = 旋转   滚轮 = 缩放   Shift+左键 = 平移   Q/Esc = 关闭")
    print(f"\n  🔴 红色 = 人手指接触区域")
    print(f"  🔵 蓝色 = 原始接触点")
    print(f"  ⬜ 灰色 = 未接触表面\n")

    o3d.visualization.draw_geometries(
        [o3d_mesh, contact_pcd],
        window_name=(f"Human Contacts — {obj_id} [{source_name}] | "
                     f"{n_contact}/{len(verts)} contact verts"),
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="可视化人手真实接触点 (Human Contact Ground Truth)")
    parser.add_argument("obj_id", type=str,
                        help="Object ID (e.g. A16013, mug, apple, wineglass)")
    parser.add_argument("--source", type=str, default=None,
                        choices=['v1', 'grab'],
                        help="Data source (auto-detect if omitted)")
    parser.add_argument("--max_frames", type=int, default=200,
                        help="Max frames to load")
    parser.add_argument("--radius", type=float, default=None,
                        help="Contact radius in meters (default: config.CONTACT_RADIUS)")
    args = parser.parse_args()
    vis_human_contacts(args.obj_id, args.source, args.max_frames, args.radius)
