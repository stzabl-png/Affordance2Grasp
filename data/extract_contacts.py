#!/usr/bin/env python3
"""
Batch Contact Map Extraction from OakInk (M1 改造版)

改动对比旧版:
  1. 3指/10帧 滑动窗口稳定性过滤
  2. 丢弃掌面接触，只保留手指接触
  3. 输出改为 HDF5
  4. 路径改为 data_hub/

用法:
    cd /home/lyh/Project/Affordance2Grasp
    python data/extract_contacts.py
    python data/extract_contacts.py --obj_id A16013
    python data/extract_contacts.py --threshold 0.005 --min_fingers 3 --min_stable_frames 10
"""

import os
import sys
import json
import pickle
import argparse
import time
import numpy as np
import trimesh
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ============================================================
# Config — 使用 data_hub 路径
# ============================================================
FILTERED_DIR = config.SEQUENCES_V1_DIR
OBJ_DIR = config.MESH_V1_DIR
OUTPUT_DIR = config.CONTACTS_DIR

# OakInk v1 标注路径 (anno 还在原始位置, 因为太大不搬)
# 如果已移入 data_hub, 改这里即可
ANNO_DIR = os.path.join(os.path.dirname(config.PROJECT_DIR), "OakInk", "image", "anno")
if not os.path.exists(ANNO_DIR):
    # Fallback: 尝试 data_hub 内部
    ANNO_DIR = os.path.join(config.DATA_HUB, "annotations", "v1")


# ============================================================
# MANO 手指拓扑 — 5个手指的顶点索引范围
# ============================================================
# MANO 模型共 778 个顶点, 按区域划分:
# 使用基于距离的 5 指判定 (相对于 5 个指尖关键点)

# MANO 指尖顶点 ID (拇指→小指)
FINGERTIP_VERTEX_IDS = [744, 320, 443, 555, 672]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# 掌面距离阈值 (距手腕)
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


def identify_finger_contacts(hand_v, contact_mask):
    """判断每根手指是否有接触.

    将每个接触顶点分配到最近的指尖, 统计哪几根手指有接触.

    Args:
        hand_v: (778, 3) 手顶点 (物体坐标系)
        contact_mask: (778,) bool 接触掩码

    Returns:
        finger_contact: dict {finger_name: bool}
        n_fingers_in_contact: int
        per_vertex_finger: (778,) int, -1=非指, 0-4=哪根指
    """
    # 指尖位置
    tips = hand_v[FINGERTIP_VERTEX_IDS]  # (5, 3)

    # 排除掌面
    palm_mask, finger_mask = classify_hand_vertices(hand_v)

    # 为每个手指区域的顶点分配最近指尖
    per_vertex_finger = np.full(len(hand_v), -1, dtype=np.int32)
    finger_verts_mask = finger_mask  # 只处理手指区域

    if finger_verts_mask.sum() > 0:
        finger_vert_ids = np.where(finger_verts_mask)[0]
        finger_vert_pos = hand_v[finger_vert_ids]
        # 到每个指尖的距离
        dists_to_tips = np.linalg.norm(
            finger_vert_pos[:, None, :] - tips[None, :, :], axis=2
        )  # (N_finger_verts, 5)
        nearest_finger = np.argmin(dists_to_tips, axis=1)  # (N_finger_verts,)
        per_vertex_finger[finger_vert_ids] = nearest_finger

    # 统计每根手指是否有接触
    finger_contact = {}
    for i, name in enumerate(FINGER_NAMES):
        # 该手指的顶点中有接触的
        finger_verts = (per_vertex_finger == i)
        has_contact = np.any(contact_mask & finger_verts)
        finger_contact[name] = bool(has_contact)

    n_fingers = sum(finger_contact.values())
    return finger_contact, n_fingers, per_vertex_finger


# ============================================================
# Contact computation (保留原有逻辑, 微调输出)
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


def compute_contacts(hand_v, obj_transf, obj_mesh, threshold):
    """计算接触点, 只返回手指接触 (丢弃掌面).

    Returns:
        finger_contact_pts_obj: (M, 3) 手指接触点在物体表面的位置
        dists: (778,) 所有顶点到物体表面的距离
        contact_mask: (778,) bool 全部接触掩码 (含掌面, 用于统计)
        finger_contact_mask: (778,) bool 手指接触掩码
        n_fingers: int 有接触的手指数量
        finger_detail: dict 每指接触详情
        force_center: (3,) 受力中心
    """
    hand_v = np.array(hand_v)
    T_obj_cam = np.linalg.inv(np.array(obj_transf))
    hand_homo = np.hstack([hand_v, np.ones((len(hand_v), 1))])
    hand_obj = (T_obj_cam @ hand_homo.T).T[:, :3]

    closest_pts, dists, face_idx = trimesh.proximity.closest_point(obj_mesh, hand_obj)

    # 分类
    palm_mask, finger_mask = classify_hand_vertices(hand_obj)
    contact_mask = dists < threshold

    # 手指接触 (丢弃掌面)
    finger_contact_mask = contact_mask & finger_mask
    finger_contact_pts = closest_pts[finger_contact_mask]

    # 判断每根手指
    finger_detail, n_fingers, _ = identify_finger_contacts(hand_obj, contact_mask)

    # 受力中心 (简化: 手指接触点均值)
    if len(finger_contact_pts) > 0:
        force_center = finger_contact_pts.mean(axis=0).astype(np.float32)
    else:
        force_center = np.zeros(3, dtype=np.float32)

    return (finger_contact_pts, dists, contact_mask, finger_contact_mask,
            n_fingers, finger_detail, force_center)


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
# 核心改动: 滑动窗口稳定性过滤
# ============================================================

def find_stable_windows(frames_data, min_fingers=3, min_frames=10):
    """滑动窗口: 找连续 ≥min_frames 帧中 ≥min_fingers 同时接触的窗口.

    Args:
        frames_data: list of (frame_idx, n_fingers, frame_data_dict)
        min_fingers: 至少几根手指同时接触
        min_frames: 至少连续多少帧

    Returns:
        stable_ranges: list of (start_idx, end_idx) 在 frames_data 中的索引范围
    """
    if len(frames_data) < min_frames:
        return []

    # 判断每帧是否满足手指数条件
    meets_condition = [fd[1] >= min_fingers for fd in frames_data]

    # 滑动窗口找连续满足的段
    stable_ranges = []
    start = None
    for i, ok in enumerate(meets_condition):
        if ok:
            if start is None:
                start = i
        else:
            if start is not None and (i - start) >= min_frames:
                stable_ranges.append((start, i))
            start = None

    # 处理末尾
    if start is not None and (len(meets_condition) - start) >= min_frames:
        stable_ranges.append((start, len(meets_condition)))

    return stable_ranges


# ============================================================
# 序列处理 (改造版)
# ============================================================

def process_sequence(seq_path, category, intent, obj_mesh, threshold,
                     min_fingers, min_stable_frames):
    """处理一个序列, 应用稳定性过滤.

    Returns:
        n_saved: 保存的帧数
        results: 结果摘要列表
        stats: 统计信息 dict
    """
    folder_name = os.path.basename(seq_path)
    seq_id, timestamp, obj_id = parse_seq_folder(folder_name)

    frames = get_frames(seq_path)
    if not frames:
        return 0, [], {"reason": "no_frames"}

    cam_id = 0

    # Step 1: 扫描所有帧, 记录每帧的手指接触数
    all_frames_data = []
    for frame in frames:
        sbj = find_sbj_flag(seq_id, timestamp, frame, cam_id)
        if sbj is None:
            continue

        hv = load_pkl(seq_id, timestamp, sbj, frame, cam_id, "hand_v")
        ot = load_pkl(seq_id, timestamp, sbj, frame, cam_id, "obj_transf")
        if hv is None or ot is None:
            continue

        (finger_pts, dists, contact_mask, finger_contact_mask,
         n_fingers, finger_detail, force_center) = compute_contacts(
            hv, ot, obj_mesh, threshold)

        all_frames_data.append({
            "frame": frame,
            "sbj": sbj,
            "n_fingers": n_fingers,
            "finger_detail": finger_detail,
            "finger_pts": finger_pts,
            "finger_contact_mask": finger_contact_mask,
            "force_center": force_center,
            "n_all_contacts": int(contact_mask.sum()),
            "n_finger_contacts": int(finger_contact_mask.sum()),
        })

    if not all_frames_data:
        return 0, [], {"reason": "no_valid_frames"}

    # Step 2: 滑动窗口过滤
    indexed_data = [(d["frame"], d["n_fingers"], d) for d in all_frames_data]
    stable_ranges = find_stable_windows(indexed_data, min_fingers, min_stable_frames)

    if not stable_ranges:
        total_frames = len(all_frames_data)
        avg_fingers = np.mean([d["n_fingers"] for d in all_frames_data])
        return 0, [], {
            "reason": "no_stable_window",
            "total_frames": total_frames,
            "avg_fingers": round(float(avg_fingers), 1),
        }

    # Step 3: 从稳定窗口中保存帧 (只保存手指接触)
    out_dir = os.path.join(OUTPUT_DIR, category, seq_id)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    saved_frames = set()

    for start_idx, end_idx in stable_ranges:
        for i in range(start_idx, end_idx):
            d = indexed_data[i][2]
            frame = d["frame"]

            if frame in saved_frames:
                continue
            saved_frames.add(frame)

            if d["n_finger_contacts"] == 0:
                continue

            # 保存为 HDF5
            out_path = os.path.join(out_dir, f"frame_{frame}.hdf5")
            with h5py.File(out_path, 'w') as f:
                # 手指接触点 (丢弃掌面)
                f.create_dataset("finger_contact_points",
                                 data=d["finger_pts"].astype(np.float32),
                                 compression="gzip")
                f.create_dataset("force_center",
                                 data=d["force_center"])

                # 元数据
                f.attrs["obj_id"] = obj_id
                f.attrs["category"] = category
                f.attrs["intent"] = intent
                f.attrs["seq_id"] = seq_id
                f.attrs["frame"] = int(frame)
                f.attrs["threshold"] = threshold
                f.attrs["n_finger_contacts"] = d["n_finger_contacts"]
                f.attrs["n_fingers"] = d["n_fingers"]

                # 每指接触详情
                for fname, has_contact in d["finger_detail"].items():
                    f.attrs[f"finger_{fname}"] = bool(has_contact)

            results.append({
                "frame": int(frame),
                "n_finger_contacts": d["n_finger_contacts"],
                "n_fingers": d["n_fingers"],
            })

    stats = {
        "total_scanned": len(all_frames_data),
        "stable_windows": len(stable_ranges),
        "stable_frames": sum(e - s for s, e in stable_ranges),
        "saved_frames": len(results),
    }
    return len(results), results, stats


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Batch extract contacts (M1: stability filter + finger-only + HDF5)")
    parser.add_argument("--threshold", type=float, default=config.CONTACT_THRESHOLD,
                        help="Contact distance threshold (m)")
    parser.add_argument("--min_fingers", type=int, default=config.MIN_FINGERS,
                        help="Min fingers in contact for stability")
    parser.add_argument("--min_stable_frames", type=int, default=config.MIN_STABLE_FRAMES,
                        help="Min consecutive frames for stability")
    parser.add_argument("--obj_id", type=str, default=None,
                        help="Only process sequences for this object ID")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("M1: Contact Extraction (stability filter + finger-only + HDF5)")
    print("=" * 60)
    print(f"  Sequences:     {FILTERED_DIR}")
    print(f"  Meshes:        {OBJ_DIR}")
    print(f"  Annotations:   {ANNO_DIR}")
    print(f"  Output:        {OUTPUT_DIR}")
    print(f"  Threshold:     {args.threshold}m")
    print(f"  Min fingers:   {args.min_fingers}")
    print(f"  Min stable:    {args.min_stable_frames} frames")
    if args.obj_id:
        print(f"  Filter obj_id: {args.obj_id}")
    sys.stdout.flush()

    # 检查 ANNO_DIR
    if not os.path.exists(ANNO_DIR):
        print(f"\n  ❌ Annotation dir not found: {ANNO_DIR}")
        print(f"  提示: OakInk v1 的 image/anno/ 目录需要在这个路径可访问")
        return

    # Discover sequences
    if not os.path.exists(FILTERED_DIR):
        print(f"\n  ❌ Filtered dir not found: {FILTERED_DIR}")
        return

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

    # Process
    mesh_cache = {}
    summary = {
        "total_sequences": len(all_sequences),
        "processed": 0,
        "skipped": 0,
        "no_stable": 0,
        "total_frames": 0,
        "categories": {},
        "params": {
            "threshold": args.threshold,
            "min_fingers": args.min_fingers,
            "min_stable_frames": args.min_stable_frames,
        },
    }

    total_start = time.time()

    for i, (seq_path, cat, intent) in enumerate(all_sequences):
        folder_name = os.path.basename(seq_path)
        seq_id, _, obj_id = parse_seq_folder(folder_name)

        if args.obj_id and obj_id != args.obj_id:
            continue

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
        n_frames, results, stats = process_sequence(
            seq_path, cat, intent, obj_mesh, args.threshold,
            args.min_fingers, args.min_stable_frames
        )
        elapsed = time.time() - t0

        if n_frames > 0:
            summary["processed"] += 1
            summary["total_frames"] += n_frames
            if cat not in summary["categories"]:
                summary["categories"][cat] = {"sequences": 0, "frames": 0}
            summary["categories"][cat]["sequences"] += 1
            summary["categories"][cat]["frames"] += n_frames
            status = f"✅ {n_frames} frames ({stats.get('stable_windows', 0)} windows)"
        else:
            reason = stats.get("reason", "unknown")
            if reason == "no_stable_window":
                summary["no_stable"] += 1
                status = f"⚠️ no stable window (avg {stats.get('avg_fingers', 0)} fingers)"
            else:
                summary["skipped"] += 1
                status = f"⏭️ {reason}"

        print(f"  [{i+1}/{len(all_sequences)}] {cat}/{seq_id}: {status} ({elapsed:.1f}s)")
        sys.stdout.flush()

    total_time = time.time() - total_start

    # Save summary
    summary["total_time_seconds"] = round(total_time, 1)
    summary_path = os.path.join(OUTPUT_DIR, "summary_m1.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DONE in {total_time:.1f}s")
    print(f"  Processed:  {summary['processed']} sequences")
    print(f"  No stable:  {summary['no_stable']} sequences")
    print(f"  Skipped:    {summary['skipped']} sequences")
    print(f"  Total frames: {summary['total_frames']}")
    print(f"  Output:     {OUTPUT_DIR}")
    print(f"  Summary:    {summary_path}")
    print(f"{'=' * 60}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
