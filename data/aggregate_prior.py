#!/usr/bin/env python3
"""
Multi-Sequence Contact Aggregation → Human Prior (M1)

将同一物体的多次抓取（多序列、多帧）合并为一份 human prior:
  1. 收集该物体所有序列的接触点
  2. 在物体表面采样 N 个点
  3. 对每个点，计算到所有接触点的最小距离
  4. 距离 < contact_radius → 标为接触 (1)
  5. 保存为 human_prior/{obj_id}.hdf5

用法:
    cd /home/lyh/Project/Affordance2Grasp
    python data/aggregate_prior.py                       # 处理所有有接触数据的物体
    python data/aggregate_prior.py --obj_id A16013       # 只处理一个物体
    python data/aggregate_prior.py --source v1           # 只处理 v1
"""

import os
import sys
import json
import argparse
import glob
import numpy as np
import trimesh
import h5py
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# GRAB contacts directory
CONTACTS_GRAB_DIR = os.path.join(os.path.dirname(config.CONTACTS_DIR), 'contacts_grab')


def collect_contacts(obj_id, contacts_dirs, search_name=None):
    """收集一个物体所有序列的手指接触点.

    Args:
        obj_id: 物体 ID (registry key, e.g. 'A16013' or 'grab_mug')
        contacts_dirs: list of (dir_path, pattern_type) tuples
        search_name: override name for file search (e.g. strip 'grab_' prefix)

    Returns:
        all_points: (M, 3) 所有接触点
        all_force_centers: (K, 3) 所有帧的 force_center
        n_frames: 帧数
    """
    all_points = []
    all_force_centers = []
    n_frames = 0
    name = search_name or obj_id

    for cdir, pat_type in contacts_dirs:
        files = []
        if pat_type == 'v1':
            # v1: contacts_dir/{category}/{obj_id}_*/*.hdf5
            pattern = os.path.join(cdir, f"**/{name}_*/*.hdf5")
            files = glob.glob(pattern, recursive=True)
        elif pat_type == 'grab':
            # GRAB: contacts_grab/{obj_name}/*/*.hdf5
            pattern = os.path.join(cdir, f"{name}/*/*.hdf5")
            files = glob.glob(pattern, recursive=True)

        for f in files:
            try:
                with h5py.File(f, 'r') as h:
                    # Support both key names
                    if 'finger_contact_points' in h:
                        pts = h['finger_contact_points'][:]
                    elif 'finger_contact_pts' in h:
                        pts = h['finger_contact_pts'][:]
                    else:
                        continue
                    fc = h['force_center'][:]
                    all_points.append(pts)
                    all_force_centers.append(fc)
                    n_frames += 1
            except Exception:
                continue

    if not all_points:
        return None, None, 0

    all_points = np.vstack(all_points)
    all_force_centers = np.array(all_force_centers)
    return all_points, all_force_centers, n_frames


def aggregate_to_prior(mesh, contact_pts, force_centers,
                       num_points=1024, contact_radius=0.005):
    """将接触点聚合为 human prior.

    Args:
        mesh: trimesh.Trimesh
        contact_pts: (M, 3) 所有接触点
        force_centers: (K, 3) 所有 force_center
        num_points: 采样点数
        contact_radius: 接触半径

    Returns:
        points: (N, 3) 采样点
        normals: (N, 3) 法线
        labels: (N,) 0/1 标签
        force_center: (3,) 聚合后的 force_center
    """
    # 均匀采样
    points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
    points = np.array(points, dtype=np.float32)
    normals = np.array(mesh.face_normals[face_idx], dtype=np.float32)

    # 标注: 距离接触点 < contact_radius → 接触
    tree = cKDTree(contact_pts)
    dists, _ = tree.query(points)
    labels = (dists < contact_radius).astype(np.float32)

    # 聚合 force_center
    force_center = force_centers.mean(axis=0).astype(np.float32)

    return points, normals, labels, force_center


def find_mesh(obj_id, registry):
    """从 registry 找 mesh 路径并加载."""
    if obj_id not in registry:
        return None
    info = registry[obj_id]
    mesh_path = os.path.join(config.DATA_HUB, info["mesh_path"])
    if not os.path.exists(mesh_path):
        return None
    return trimesh.load(mesh_path, force='mesh')


def main():
    parser = argparse.ArgumentParser(description="Aggregate contacts → human prior (M1)")
    parser.add_argument("--obj_id", type=str, default=None, help="Only process one object")
    parser.add_argument("--source", type=str, default="all",
                        choices=["v1", "grab", "all"],
                        help="Which contact data to use")
    parser.add_argument("--num_points", type=int, default=config.NUM_POINTS)
    parser.add_argument("--contact_radius", type=float, default=config.CONTACT_RADIUS)
    args = parser.parse_args()

    config.ensure_dirs()

    # Load registry
    with open(config.REGISTRY_PATH) as f:
        registry = json.load(f)

    # Contact directories: (path, type)
    contacts_dirs = []
    if args.source in ["v1", "all"]:
        contacts_dirs.append((config.CONTACTS_DIR, 'v1'))
    if args.source in ["grab", "all"]:
        contacts_dirs.append((CONTACTS_GRAB_DIR, 'grab'))

    # Determine which objects to process
    if args.obj_id:
        obj_ids = [args.obj_id]
    else:
        # Use registry to determine all objects
        obj_ids = sorted(registry.keys())

    print("=" * 60)
    print("M1: Multi-Sequence Contact Aggregation → Human Prior")
    print("=" * 60)
    print(f"  Source:         {args.source}")
    print(f"  Objects:        {len(obj_ids)}")
    print(f"  Points:         {args.num_points}")
    print(f"  Contact radius: {args.contact_radius}m")
    print(f"  Output:         {config.HUMAN_PRIOR_DIR}")
    sys.stdout.flush()

    success = 0
    skipped = 0
    errors = []

    for i, obj_id in enumerate(obj_ids):
        # Determine search name (strip 'grab_' prefix for GRAB objects)
        source = registry.get(obj_id, {}).get('source', 'v1')
        if source == 'grab':
            search_name = obj_id.replace('grab_', '', 1)
        else:
            search_name = obj_id

        # Collect contacts
        contact_pts, force_centers, n_frames = collect_contacts(
            obj_id, contacts_dirs, search_name=search_name
        )
        if contact_pts is None or n_frames == 0:
            skipped += 1
            continue

        # Load mesh
        mesh = find_mesh(obj_id, registry)
        if mesh is None:
            print(f"  [{i+1}/{len(obj_ids)}] {obj_id}: ❌ mesh not found")
            errors.append(obj_id)
            continue

        # Aggregate
        points, normals, labels, force_center = aggregate_to_prior(
            mesh, contact_pts, force_centers,
            args.num_points, args.contact_radius
        )

        n_contact = int(labels.sum())

        # Save HDF5
        out_path = os.path.join(config.HUMAN_PRIOR_DIR, f"{obj_id}.hdf5")
        with h5py.File(out_path, 'w') as f:
            f.create_dataset("point_cloud", data=points, compression="gzip")
            f.create_dataset("normals", data=normals, compression="gzip")
            f.create_dataset("human_prior", data=labels)
            f.create_dataset("force_center", data=force_center)
            f.attrs["obj_id"] = obj_id
            f.attrs["source"] = registry.get(obj_id, {}).get("source", "unknown")
            f.attrs["category"] = registry.get(obj_id, {}).get("category", "unknown")
            f.attrs["num_points"] = args.num_points
            f.attrs["contact_radius"] = args.contact_radius
            f.attrs["n_contact_verts"] = n_contact
            f.attrs["n_raw_contact_pts"] = len(contact_pts)
            f.attrs["n_frames"] = n_frames
            f.attrs["n_force_centers"] = len(force_centers)

        print(f"  [{i+1}/{len(obj_ids)}] {obj_id}: ✅ {n_contact}/{args.num_points} contact "
              f"({n_frames} frames, {len(contact_pts)} raw pts)")
        success += 1

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {len(errors)}")
    print(f"  Output:  {config.HUMAN_PRIOR_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
