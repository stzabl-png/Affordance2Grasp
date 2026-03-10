#!/usr/bin/env python3
"""
Batch Contact Map Extraction from OakInk

遍历所有 OakInk filtered 序列,提取每帧的 contact map 数据。

用法:
    python batch_extract.py
    python batch_extract.py --threshold 0.005 --frame_step 5
"""

import os
import sys
import json
import pickle
import argparse
import time
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ============================================================
# Config
# ============================================================
OAKINK_DIR = config.OAKINK_DIR
FILTERED_DIR = config.OAKINK_FILTERED_DIR
ANNO_DIR = config.OAKINK_ANNO_DIR
OBJ_DIR = config.OAKINK_OBJ_DIR
OUTPUT_DIR = config.CONTACTS_DIR


# ============================================================
# Helpers (reused from extract_contact.py)
# ============================================================

def load_object_mesh(obj_id):
    """加载物体 mesh (支持 .obj 和 .ply)."""
    for ext in [".obj", ".ply"]:
        path = os.path.join(OBJ_DIR, f"{obj_id}{ext}")
        if os.path.exists(path):
            return trimesh.load(path, force='mesh')
    return None


def find_sbj_flag(seq_id, timestamp, frame, cam_id=0):
    """找到正确的 sbj_flag."""
    for sbj in [0, 1]:
        pkl = f"{seq_id}__{timestamp}__{sbj}__{frame}__{cam_id}.pkl"
        if os.path.exists(os.path.join(ANNO_DIR, "hand_v", pkl)):
            return sbj
    return None


def load_pkl(seq_id, timestamp, sbj, frame, cam_id, anno_type):
    """加载标注 pkl 文件."""
    pkl = f"{seq_id}__{timestamp}__{sbj}__{frame}__{cam_id}.pkl"
    path = os.path.join(ANNO_DIR, anno_type, pkl)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# MANO palm/finger separation threshold
# Palm = vertices within 5cm of wrist (vertex 0)
# Finger = vertices >= 5cm from wrist
PALM_DIST_THRESHOLD = 0.05  # 5cm


def classify_hand_vertices(hand_v):
    """将 MANO 778 顶点分为手掌和手指区域.
    
    Returns:
        palm_mask: (778,) bool, True=手掌
        finger_mask: (778,) bool, True=手指
    """
    wrist = hand_v[0]
    dists = np.linalg.norm(hand_v - wrist, axis=1)
    palm_mask = dists < PALM_DIST_THRESHOLD
    finger_mask = ~palm_mask
    return palm_mask, finger_mask


def compute_force_center(finger_pts, finger_normals, palm_surface_pt, palm_normal):
    """计算受力中心: 手指力线在掌心法线上的投影中点.
    
    掌心法线穿过物体内部, 是力的"骨架".
    每条手指力线与掌心法线的最近点 → 投影到掌心法线上的参数 t.
    受力中心 = palm_surface_pt + mean(t) * palm_normal
    
    Args:
        finger_pts: (N, 3) 手指在物体表面的接触点
        finger_normals: (N, 3) 手指接触点的内法线
        palm_surface_pt: (3,) 掌心在物体表面的最近点
        palm_normal: (3,) 掌心法线 (指向物体内部)
    """
    palm_d = palm_normal / (np.linalg.norm(palm_normal) + 1e-8)
    
    t_values = []
    for i in range(len(finger_pts)):
        p = finger_pts[i]
        d = finger_normals[i]
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-8:
            continue
        d = d / d_norm
        
        # 求两条直线的最近点
        # 线1: palm_surface_pt + t * palm_d
        # 线2: p + s * d
        w = palm_surface_pt - p
        a = np.dot(palm_d, palm_d)  # = 1
        b_val = np.dot(palm_d, d)
        c = np.dot(d, d)  # = 1
        d_val = np.dot(palm_d, w)
        e = np.dot(d, w)
        
        denom = a * c - b_val * b_val
        if abs(denom) < 1e-8:
            # 平行线, 取 w 在 palm_d 上的投影
            t = -d_val
        else:
            t = (b_val * e - c * d_val) / denom
        
        t_values.append(t)
    
    if len(t_values) == 0:
        return palm_surface_pt.copy().astype(np.float32)
    
    t_mean = np.mean(t_values)
    # 限制 t >= 0 (受力中心应在物体内部, 不能在表面外)
    t_mean = max(t_mean, 0.0)
    
    force_center = palm_surface_pt + t_mean * palm_d
    return force_center.astype(np.float32)


def compute_contacts(hand_v, obj_transf, obj_mesh, threshold):
    """计算接触点,区分手指/手掌,返回手指接触点、手掌中心和受力中心.
    
    力线方向 = 物体表面内法线 (每个接触点向内施力)
    掌心也作为一个力点参与受力中心计算
    
    Returns:
        finger_contact_idx: 手指区域的接触顶点索引
        finger_contact_pts_obj: 手指接触点在物体表面的位置
        palm_center_obj: 手掌中心在物体坐标系的位置
        all_contact_idx: 所有接触顶点索引 (兼容旧格式)
        all_contact_pts_obj: 所有接触点
        dists: 所有顶点的距离
        force_center_obj: 受力中心 (力线汇聚点)
        force_normals: 力线方向 (内法线)
    """
    hand_v = np.array(hand_v)
    T_obj_cam = np.linalg.inv(np.array(obj_transf))
    hand_homo = np.hstack([hand_v, np.ones((len(hand_v), 1))])
    hand_obj = (T_obj_cam @ hand_homo.T).T[:, :3]

    closest_pts, dists, face_idx = trimesh.proximity.closest_point(obj_mesh, hand_obj)

    # 分类手掌/手指
    palm_mask, finger_mask = classify_hand_vertices(hand_obj)
    
    # 手掌中心 (手掌区域手顶点平均位置)
    palm_center_obj = hand_obj[palm_mask].mean(axis=0).astype(np.float32)
    
    # 所有接触 (兼容)
    contact_mask = dists < threshold
    all_contact_idx = np.where(contact_mask)[0]
    all_contact_pts_obj = closest_pts[contact_mask]
    
    # 手指接触
    finger_contact_mask = contact_mask & finger_mask
    finger_contact_idx = np.where(finger_contact_mask)[0]
    finger_contact_pts_obj = closest_pts[finger_contact_mask]

    # ---- 受力中心计算 ----
    # 获取物体表面法线 (面法线 → 内法线)
    face_normals = np.array(obj_mesh.face_normals)
    
    # 收集所有力点: 手指接触点 + 手掌接触点在物体表面的投影
    force_pts = []
    force_normals_list = []
    
    # 手指接触的表面点和内法线
    finger_face_idx = face_idx[finger_contact_mask]
    for i in range(len(finger_contact_pts_obj)):
        pt = finger_contact_pts_obj[i]
        fn = face_normals[finger_face_idx[i]]
        # 判断法线方向: 应该指向物体内部
        # 如果法线指向手 (远离物体中心), 取反
        obj_center = np.array(obj_mesh.centroid)
        to_center = obj_center - pt
        if np.dot(fn, to_center) < 0:
            fn = -fn  # 翻转为内法线
        force_pts.append(pt)
        force_normals_list.append(fn)
    
    # 手掌中心也投射力线: 找到掌心在物体表面的最近点
    palm_surface_pt, palm_dist, palm_face = trimesh.proximity.closest_point(
        obj_mesh, palm_center_obj.reshape(1, -1))
    if palm_dist[0] < threshold * 3:  # 掌心足够近
        palm_fn = face_normals[palm_face[0]]
        obj_center = np.array(obj_mesh.centroid)
        to_center = obj_center - palm_surface_pt[0]
        if np.dot(palm_fn, to_center) < 0:
            palm_fn = -palm_fn
        force_pts.append(palm_surface_pt[0])
        force_normals_list.append(palm_fn)
    
    # 计算受力中心 (掌心法线投影方案)
    if len(force_pts) >= 2 and palm_dist[0] < threshold * 3:
        # 手指力点 = 不含掌心的那些
        n_finger_force = len(force_pts) - 1 if palm_dist[0] < threshold * 3 else len(force_pts)
        finger_force_pts = np.array(force_pts[:n_finger_force], dtype=np.float32)
        finger_force_norms = np.array(force_normals_list[:n_finger_force], dtype=np.float32)
        force_pts_arr = np.array(force_pts, dtype=np.float32)
        force_normals_arr = np.array(force_normals_list, dtype=np.float32)
        force_center = compute_force_center(
            finger_force_pts, finger_force_norms,
            palm_surface_pt[0].astype(np.float32),
            palm_fn.astype(np.float32)
        )
    else:
        force_pts_arr = np.array(force_pts, dtype=np.float32) if force_pts else all_contact_pts_obj
        force_normals_arr = np.zeros_like(force_pts_arr)
        force_center = all_contact_pts_obj.mean(axis=0).astype(np.float32) if len(all_contact_pts_obj) > 0 else palm_center_obj

    return (finger_contact_idx, finger_contact_pts_obj, palm_center_obj,
            all_contact_idx, all_contact_pts_obj, dists,
            force_center, force_pts_arr, force_normals_arr)


def get_frames(seq_path):
    """获取序列的所有帧号."""
    frames = set()
    for f in os.listdir(seq_path):
        if f.endswith('.png'):
            try:
                num = int(f.rsplit('_', 1)[1].replace('.png', ''))
                frames.add(num)
            except (ValueError, IndexError):
                pass
    return sorted(frames)


def parse_seq_folder(folder_name):
    """解析序列文件夹名."""
    parts = folder_name.split("__")
    seq_id = parts[0]
    timestamp = parts[1]
    obj_id = seq_id.split("_")[0]
    return seq_id, timestamp, obj_id


# ============================================================
# Main batch processing
# ============================================================

def process_sequence(seq_path, category, intent, obj_mesh, threshold, frame_step):
    """处理一个序列,返回处理的帧数和结果列表."""
    folder_name = os.path.basename(seq_path)
    seq_id, timestamp, obj_id = parse_seq_folder(folder_name)

    frames = get_frames(seq_path)
    if not frames:
        return 0, []

    # 找到有标注的帧
    cam_id = 0
    valid_frames = []
    for frame in frames:
        sbj = find_sbj_flag(seq_id, timestamp, frame, cam_id)
        if sbj is not None:
            valid_frames.append((frame, sbj))

    if not valid_frames:
        return 0, []

    # 找到首次接触帧
    first_contact_frame = None
    for frame, sbj in valid_frames:
        hv = load_pkl(seq_id, timestamp, sbj, frame, cam_id, "hand_v")
        ot = load_pkl(seq_id, timestamp, sbj, frame, cam_id, "obj_transf")
        if hv is None or ot is None:
            continue
        contact_result = compute_contacts(hv, ot, obj_mesh, threshold)
        (finger_idx, finger_pts, palm_center, all_idx, all_pts, dists,
         force_center, force_pts, force_norms) = contact_result
        if (dists < threshold).sum() > 0:
            first_contact_frame = frame
            break

    if first_contact_frame is None:
        # 没有接触帧,跳过
        return 0, []

    # 从首次接触帧开始,每隔 frame_step 取一帧
    contact_frames = [(f, s) for f, s in valid_frames if f >= first_contact_frame]
    sampled = contact_frames[::frame_step]

    # 输出目录
    out_dir = os.path.join(OUTPUT_DIR, category, seq_id)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for frame, sbj in sampled:
        hv = load_pkl(seq_id, timestamp, sbj, frame, cam_id, "hand_v")
        ot = load_pkl(seq_id, timestamp, sbj, frame, cam_id, "obj_transf")
        if hv is None or ot is None:
            continue

        contact_result = compute_contacts(hv, ot, obj_mesh, threshold)
        (finger_contact_idx, finger_contact_pts, palm_center,
         all_contact_idx, all_contact_pts, dists,
         force_center, force_pts, force_normals) = contact_result

        if len(all_contact_idx) == 0:
            continue

        # 保存
        out_path = os.path.join(out_dir, f"frame_{frame}.npz")
        np.savez_compressed(out_path,
            # 手指接触 (新)
            finger_contact_idx=finger_contact_idx,
            finger_contact_points_obj=finger_contact_pts,
            palm_center_obj=palm_center,
            # 受力中心 (新)
            force_center_obj=force_center,
            force_pts_obj=force_pts,
            force_normals=force_normals,
            # 所有接触 (兼容旧格式)
            contact_idx=all_contact_idx,
            contact_points_obj=all_contact_pts,
            distances=dists,
            # 元数据
            obj_id=obj_id,
            category=category,
            intent=intent,
            seq_id=seq_id,
            frame=frame,
            threshold=threshold
        )
        results.append({
            "frame": int(frame),
            "n_contacts": int(len(all_contact_idx)),
            "n_finger_contacts": int(len(finger_contact_idx)),
            "min_dist": float(dists.min())
        })

    return len(results), results


def main():
    parser = argparse.ArgumentParser(description="Batch extract contact maps from OakInk")
    parser.add_argument("--threshold", type=float, default=0.005, help="Contact distance threshold (m)")
    parser.add_argument("--frame_step", type=int, default=3, help="Sample every N frames after first contact")
    parser.add_argument("--obj_id", type=str, default=None, help="Only process sequences for this object ID (e.g. A16013)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Batch Contact Map Extraction")
    print("=" * 60)
    print(f"  OakInk dir: {OAKINK_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Threshold:  {args.threshold}m")
    print(f"  Frame step: every {args.frame_step} frames")
    sys.stdout.flush()

    # Discover all sequences
    categories = sorted(os.listdir(FILTERED_DIR))
    all_sequences = []
    for cat in categories:
        cat_path = os.path.join(FILTERED_DIR, cat)
        if not os.path.isdir(cat_path):
            continue
        for intent in sorted(os.listdir(cat_path)):
            intent_path = os.path.join(cat_path, intent)
            if not os.path.isdir(intent_path):
                continue
            for seq_folder in sorted(os.listdir(intent_path)):
                seq_path = os.path.join(intent_path, seq_folder)
                if os.path.isdir(seq_path) and "__" in seq_folder:
                    all_sequences.append((seq_path, cat, intent))

    print(f"\n  Found {len(all_sequences)} sequences across {len(categories)} categories")
    sys.stdout.flush()

    # Cache loaded meshes
    mesh_cache = {}
    summary = {
        "total_sequences": len(all_sequences),
        "processed": 0,
        "skipped": 0,
        "total_frames": 0,
        "categories": {},
        "threshold": args.threshold,
        "frame_step": args.frame_step
    }

    total_start = time.time()

    for i, (seq_path, cat, intent) in enumerate(all_sequences):
        folder_name = os.path.basename(seq_path)
        seq_id, _, obj_id = parse_seq_folder(folder_name)

        # --obj_id 过滤
        if args.obj_id and obj_id != args.obj_id:
            continue

        # Load or cache mesh
        if obj_id not in mesh_cache:
            mesh = load_object_mesh(obj_id)
            if mesh is None:
                print(f"  [{i+1}/{len(all_sequences)}] SKIP {cat}/{seq_id} - mesh {obj_id} not found")
                sys.stdout.flush()
                summary["skipped"] += 1
                continue
            mesh_cache[obj_id] = mesh

        obj_mesh = mesh_cache[obj_id]

        t0 = time.time()
        n_frames, results = process_sequence(
            seq_path, cat, intent, obj_mesh, args.threshold, args.frame_step
        )
        elapsed = time.time() - t0

        if n_frames > 0:
            summary["processed"] += 1
            summary["total_frames"] += n_frames
            if cat not in summary["categories"]:
                summary["categories"][cat] = {"sequences": 0, "frames": 0}
            summary["categories"][cat]["sequences"] += 1
            summary["categories"][cat]["frames"] += n_frames
            status = f"{n_frames} frames"
        else:
            summary["skipped"] += 1
            status = "no contacts"

        print(f"  [{i+1}/{len(all_sequences)}] {cat}/{seq_id}: {status} ({elapsed:.1f}s)")
        sys.stdout.flush()

    total_time = time.time() - total_start

    # Save summary
    summary["total_time_seconds"] = round(total_time, 1)
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DONE in {total_time:.1f}s")
    print(f"  Processed: {summary['processed']} sequences")
    print(f"  Skipped:   {summary['skipped']} sequences")
    print(f"  Total frames: {summary['total_frames']}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Summary: {summary_path}")
    print(f"{'=' * 60}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
