#!/usr/bin/env python3
"""
Contact Map Extraction from OakInk-v2 (M1 改造版)

改动对比旧版:
  1. 适配 v1 extract_contacts.py 的新 compute_contacts API
  2. 3指/10帧 滑动窗口稳定性过滤
  3. 丢弃掌面接触，只保留手指接触
  4. 输出改为 HDF5
  5. 路径改为 data_hub/

用法:
    cd /home/lyh/Project/Affordance2Grasp
    python data/extract_contacts_v2.py
    python data/extract_contacts_v2.py --workers 4
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import trimesh
import h5py
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from data.extract_contacts import (
    compute_contacts,
    classify_hand_vertices,
    identify_finger_contacts,
    find_stable_windows,
    FINGER_NAMES,
)


# ============================================================
# MANO Forward Kinematics
# ============================================================

class MANOForward:
    def __init__(self, mano_root, device='cpu'):
        from manotorch.manolayer import ManoLayer
        self.device = device
        self.rh_layer = ManoLayer(rot_mode='quat', side='right', mano_assets_root=mano_root).to(device)
        self.lh_layer = ManoLayer(rot_mode='quat', side='left', mano_assets_root=mano_root).to(device)

    @torch.no_grad()
    def forward(self, mano_param, side='right', tsl=None):
        layer = self.rh_layer if side == 'right' else self.lh_layer
        pose = mano_param['pose_coeffs'].to(self.device)
        betas = mano_param['betas'].to(self.device)
        N = pose.shape[0]
        pose_flat = pose.reshape(N, -1)
        output = layer(pose_flat, betas)
        verts = output.verts.cpu().numpy()
        if tsl is not None:
            tsl_np = tsl.cpu().numpy() if isinstance(tsl, torch.Tensor) else tsl
            verts = verts + tsl_np[:, None, :]
        return verts


# ============================================================
# Contact extraction for one frame (新 API)
# ============================================================

def extract_frame_contacts(hand_v, obj_transf, obj_mesh, threshold):
    """提取单帧接触, 适配新 compute_contacts API."""
    result = compute_contacts(hand_v, obj_transf, obj_mesh, threshold)
    (finger_pts, dists, contact_mask, finger_contact_mask,
     n_fingers, finger_detail, force_center) = result

    if finger_contact_mask.sum() == 0:
        return None

    return {
        'finger_contact_points': finger_pts.astype(np.float32),
        'force_center': force_center,
        'n_fingers': n_fingers,
        'n_finger_contacts': int(finger_contact_mask.sum()),
        'finger_detail': finger_detail,
    }


# ============================================================
# Worker: process one sequence (with stability filter)
# ============================================================

_worker_oi2 = None
_worker_mano = None

def _init_worker(oakink2_dir, mano_root):
    global _worker_oi2, _worker_mano
    from oakink2_toolkit.dataset import OakInk2__Dataset
    _worker_oi2 = OakInk2__Dataset(
        dataset_prefix=oakink2_dir,
        return_instantiated=True,
        anno_offset='anno_preview',
        obj_offset='object_raw',
        affordance_offset='object_affordance',
    )
    _worker_mano = MANOForward(mano_root, device='cpu')


def _process_one_seq(task):
    """Process one sequence with stability filter. Returns (seq_key, n_frames, elapsed, error)."""
    seq_idx, seq_key, frame_step, threshold, min_fingers, min_stable_frames, output_dir = task
    global _worker_oi2, _worker_mano

    t0 = time.time()
    mesh_cache = {}

    def get_mesh(obj_id):
        if obj_id not in mesh_cache:
            try:
                mesh = _worker_oi2._load_obj(obj_id)
                if mesh is not None:
                    mesh_cache[obj_id] = mesh
            except Exception:
                pass
        return mesh_cache.get(obj_id, None)

    try:
        complex_task = _worker_oi2.load_complex_task(seq_key)
        primitives = _worker_oi2.load_primitive_task(complex_task)
    except Exception as e:
        return (seq_key, 0, time.time() - t0, str(e)[:100])

    seq_frames = 0

    for prim_idx, prim in enumerate(primitives):
        if not prim.instantiated:
            continue

        obj_list = prim.task_obj_list or []
        if not obj_list:
            continue

        for side, param, obj_ids in [
            ('right', prim.rh_param, prim.rh_obj_list),
            ('left', prim.lh_param, prim.lh_obj_list),
        ]:
            if param is None or obj_ids is None:
                continue

            mask = prim.rh_in_range_mask if side == 'right' else prim.lh_in_range_mask
            if mask is None:
                continue

            try:
                tsl = param.get('tsl', None)
                hand_verts_all = _worker_mano.forward(param, side=side, tsl=tsl)
            except Exception:
                continue

            N_frames = hand_verts_all.shape[0]

            for obj_id in obj_ids:
                obj_mesh = get_mesh(obj_id)
                if obj_mesh is None:
                    continue

                if obj_id not in prim.obj_transf:
                    continue
                obj_transf_all = prim.obj_transf[obj_id]

                parts = obj_id.split("@")
                category = parts[1] if len(parts) >= 2 else "unknown"

                # Step 1: 扫描帧 (每 frame_step 帧采样一次)
                all_frames_data = []
                for fi in range(0, N_frames, frame_step):
                    if not mask[fi]:
                        continue

                    hand_v = hand_verts_all[fi]
                    if isinstance(obj_transf_all, torch.Tensor):
                        obj_tf = obj_transf_all[fi].numpy()
                    elif isinstance(obj_transf_all, np.ndarray):
                        obj_tf = obj_transf_all[fi]
                    else:
                        continue

                    contacts = extract_frame_contacts(hand_v, obj_tf, obj_mesh, threshold)
                    if contacts is None:
                        all_frames_data.append((fi, 0, None))
                    else:
                        all_frames_data.append((fi, contacts['n_fingers'], contacts))

                if not all_frames_data:
                    continue

                # Step 2: 滑动窗口
                stable_ranges = find_stable_windows(
                    all_frames_data, min_fingers, min_stable_frames
                )

                if not stable_ranges:
                    continue

                # Step 3: 保存稳定窗口内的帧为 HDF5
                seq_token = f"{seq_key}__p{prim_idx}__{side}"
                out_dir = os.path.join(output_dir, category, seq_token.replace("/", "__"))
                os.makedirs(out_dir, exist_ok=True)

                for start_idx, end_idx in stable_ranges:
                    for i in range(start_idx, end_idx):
                        fi, n_fingers, contacts = all_frames_data[i]
                        if contacts is None:
                            continue

                        frame_id = prim.frame_range[0] + fi if prim.frame_range else fi
                        out_path = os.path.join(out_dir, f"frame_{frame_id}.hdf5")
                        with h5py.File(out_path, 'w') as f:
                            f.create_dataset("finger_contact_points",
                                             data=contacts['finger_contact_points'],
                                             compression="gzip")
                            f.create_dataset("force_center",
                                             data=contacts['force_center'])
                            f.attrs["obj_id"] = obj_id
                            f.attrs["category"] = category
                            f.attrs["intent"] = prim.primitive_task or "grasp"
                            f.attrs["seq_id"] = seq_token
                            f.attrs["frame"] = int(frame_id)
                            f.attrs["threshold"] = threshold
                            f.attrs["n_finger_contacts"] = contacts['n_finger_contacts']
                            f.attrs["n_fingers"] = contacts['n_fingers']
                            for fn in FINGER_NAMES:
                                f.attrs[f"finger_{fn}"] = contacts['finger_detail'].get(fn, False)

                        seq_frames += 1

    elapsed = time.time() - t0
    return (seq_key, seq_frames, elapsed, None)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Extract contacts from OakInk-v2 (M1: stability + HDF5)")
    parser.add_argument("--oakink2_dir", type=str,
                        default=os.path.expanduser("~/Project/OakInk2/OakInk-v2-hub"))
    parser.add_argument("--mano_root", type=str,
                        default=os.path.expanduser("~/Project/mano_v1_2"))
    parser.add_argument("--threshold", type=float, default=config.CONTACT_THRESHOLD)
    parser.add_argument("--min_fingers", type=int, default=config.MIN_FINGERS)
    parser.add_argument("--min_stable_frames", type=int, default=config.MIN_STABLE_FRAMES)
    parser.add_argument("--frame_step", type=int, default=5,
                        help="Sample every N-th frame (default: 5)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_seqs", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = args.output_dir or config.CONTACTS_V2_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("M1: OakInk-v2 Contact Extraction (stability + finger-only + HDF5)")
    print("=" * 60)
    print(f"  OakInk-v2 dir: {args.oakink2_dir}")
    print(f"  MANO root:     {args.mano_root}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Threshold:     {args.threshold}m")
    print(f"  Min fingers:   {args.min_fingers}")
    print(f"  Min stable:    {args.min_stable_frames} sampled frames")
    print(f"  Frame step:    every {args.frame_step} frames")
    print(f"  Workers:       {args.workers}")
    sys.stdout.flush()

    # Load dataset to get sequence list
    print("\n  Loading sequence list...")
    sys.stdout.flush()
    from oakink2_toolkit.dataset import OakInk2__Dataset
    oi2 = OakInk2__Dataset(
        dataset_prefix=args.oakink2_dir,
        return_instantiated=True,
        anno_offset='anno_preview',
        obj_offset='object_raw',
        affordance_offset='object_affordance',
    )
    all_seqs = oi2.all_seq_list
    print(f"  Total sequences: {len(all_seqs)}")
    del oi2

    n_seqs = len(all_seqs)
    if args.max_seqs > 0:
        n_seqs = min(n_seqs, args.max_seqs)
    seqs_to_process = all_seqs[:n_seqs]

    # Resume
    done_marker_dir = os.path.join(output_dir, ".done")
    os.makedirs(done_marker_dir, exist_ok=True)

    remaining = []
    skipped_resume = 0
    for seq_key in seqs_to_process:
        marker = os.path.join(done_marker_dir, seq_key.replace("/", "__") + ".done")
        if os.path.exists(marker):
            skipped_resume += 1
        else:
            remaining.append(seq_key)

    if skipped_resume > 0:
        print(f"  ⏭️  Resuming: skipped {skipped_resume} already-processed sequences")
    print(f"  Remaining: {len(remaining)} sequences")
    sys.stdout.flush()

    if not remaining:
        print("  ✅ All sequences already processed!")
        return

    tasks = [
        (i, seq_key, args.frame_step, args.threshold, args.min_fingers, args.min_stable_frames, output_dir)
        for i, seq_key in enumerate(remaining)
    ]

    # Process
    total_start = time.time()
    completed = 0
    total_frames = 0
    errors = []
    no_stable = 0

    if args.workers <= 1:
        _init_worker(args.oakink2_dir, args.mano_root)
        for task in tasks:
            result = _process_one_seq(task)
            seq_key, n_frames, elapsed, error = result
            completed += 1
            if error:
                errors.append((seq_key, error))
                label = f"ERROR: {error[:60]}"
            elif n_frames > 0:
                total_frames += n_frames
                label = f"✅ {n_frames} frames"
            else:
                no_stable += 1
                label = "⚠️ no stable window"

            wall = time.time() - total_start
            rate = completed / wall if wall > 0 else 1
            eta = (len(remaining) - completed) / rate
            eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.0f}min"

            print(f"  [{completed}/{len(remaining)}] {seq_key}: {label} ({elapsed:.0f}s) ETA={eta_str}")
            sys.stdout.flush()

            marker = os.path.join(done_marker_dir, seq_key.replace("/", "__") + ".done")
            with open(marker, 'w') as f:
                f.write(f"{n_frames}")
    else:
        print(f"\n  Starting {args.workers} workers...")
        sys.stdout.flush()

        with Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.oakink2_dir, args.mano_root),
        ) as pool:
            for result in pool.imap_unordered(_process_one_seq, tasks):
                seq_key, n_frames, elapsed, error = result
                completed += 1
                if error:
                    errors.append((seq_key, error))
                    label = f"ERROR: {error[:60]}"
                elif n_frames > 0:
                    total_frames += n_frames
                    label = f"✅ {n_frames} frames"
                else:
                    no_stable += 1
                    label = "⚠️ no stable window"

                wall = time.time() - total_start
                rate = completed / wall if wall > 0 else 1
                eta = (len(remaining) - completed) / rate
                eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.0f}min"

                print(f"  [{completed}/{len(remaining)}] {seq_key}: {label} ({elapsed:.0f}s) ETA={eta_str}")
                sys.stdout.flush()

                marker = os.path.join(done_marker_dir, seq_key.replace("/", "__") + ".done")
                with open(marker, 'w') as f:
                    f.write(f"{n_frames}")

    total_time = time.time() - total_start

    summary = {
        "total_sequences": len(all_seqs),
        "processed": completed,
        "skipped_resume": skipped_resume,
        "no_stable": no_stable,
        "errors": len(errors),
        "total_frames": total_frames,
        "total_time_seconds": round(total_time, 1),
        "params": {
            "threshold": args.threshold,
            "min_fingers": args.min_fingers,
            "min_stable_frames": args.min_stable_frames,
        },
    }
    summary_path = os.path.join(output_dir, "summary_m1.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DONE in {total_time/3600:.1f}h")
    print(f"  Processed:  {completed} sequences")
    print(f"  No stable:  {no_stable} sequences")
    print(f"  Resumed:    {skipped_resume} sequences")
    print(f"  Errors:     {len(errors)} sequences")
    print(f"  Total frames: {total_frames}")
    print(f"  Output:     {output_dir}")
    if errors:
        print(f"\n  Errors:")
        for seq, err in errors[:10]:
            print(f"    {seq}: {err}")
    print(f"{'=' * 60}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
