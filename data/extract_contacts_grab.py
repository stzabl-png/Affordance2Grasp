#!/usr/bin/env python3
"""
Extract contact data from GRAB dataset.

GRAB provides per-vertex contact annotations directly (contact.object),
so we don't need MANO forward kinematics or distance computation.

Each object vertex has a label:
  0     = no contact
  1-40  = body parts (torso, legs, etc.)
  41-55 = hand/finger parts

We extract frames where hand contacts exist, and output HDF5 files
in the same format as extract_contacts.py (OakInk v1).

Usage:
    python data/extract_contacts_grab.py
    python data/extract_contacts_grab.py --obj_name mug
"""

import os
import sys
import json
import time
import glob
import argparse
import numpy as np
import trimesh
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ============================================================
# Config
# ============================================================
GRAB_DATA = os.path.expanduser("~/Project/GRAB_data/_unzipped")
GRAB_SEQS = os.path.join(GRAB_DATA, "grab")
GRAB_MESHES = os.path.join(GRAB_DATA, "tools", "object_meshes", "contact_meshes")
OUTPUT_DIR = os.path.join(config.CONTACTS_DIR, "..", "contacts_grab")

# GRAB contact IDs for hand parts (41-55)
# Any value > 0 means contact; values 41+ are hand/finger regions
HAND_CONTACT_MIN = 41
HAND_CONTACT_MAX = 55

# Right hand finger contact IDs (from SMPL-X joint mapping)
# 41=right index1, 42=right index2, 43=right index3,
# 44=right middle1, 45=right middle2, 46=right middle3, ...
# For simplicity, we treat ALL hand contact (41-55) as valid
# since GRAB's contact is already high-quality

# Minimum frames with hand contact to consider a sequence valid
MIN_CONTACT_FRAMES = 5


def load_grab_mesh(obj_name):
    """Load object mesh from GRAB contact_meshes directory."""
    # GRAB names: 'mug' -> 'coffeemug.ply', 'ps_controller' -> 'pscontroller.ply'
    # Try exact match first, then common variants
    candidates = [
        os.path.join(GRAB_MESHES, f"{obj_name}.ply"),
        os.path.join(GRAB_MESHES, f"{obj_name.replace('_', '')}.ply"),
        os.path.join(GRAB_MESHES, f"coffee{obj_name}.ply"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return trimesh.load(path, process=False)

    # Fuzzy match
    all_meshes = os.listdir(GRAB_MESHES)
    for m in all_meshes:
        if obj_name.replace("_", "") in m.replace("_", "").lower():
            return trimesh.load(os.path.join(GRAB_MESHES, m), process=False)

    return None


def process_sequence(npz_path, min_contact_frames=MIN_CONTACT_FRAMES):
    """
    Process one GRAB sequence.

    Returns:
        results: list of dicts with contact data per frame
        obj_name: object name
        intent: motion intent (grab/pass/use/etc)
    """
    data = dict(np.load(npz_path, allow_pickle=True))

    obj_name = str(data['obj_name'])
    intent = str(data.get('motion_intent', 'unknown'))
    n_frames = int(data['n_frames'])

    # Get contact data
    contact_data = data['contact'].item()
    contact_obj = contact_data['object']  # (T, M) - per-vertex contact labels

    # Find frames with hand contact (values >= HAND_CONTACT_MIN)
    hand_contact_per_frame = (contact_obj >= HAND_CONTACT_MIN) & (contact_obj <= HAND_CONTACT_MAX)
    # Also include any non-zero contact as potentially useful
    any_contact_per_frame = contact_obj > 0

    # Count frames with hand contact
    frames_with_hand_contact = hand_contact_per_frame.any(axis=1).sum()

    if frames_with_hand_contact < min_contact_frames:
        return None, obj_name, intent, n_frames, int(frames_with_hand_contact)

    # Get object mesh to compute force center
    obj_mesh = load_grab_mesh(obj_name)
    if obj_mesh is None:
        return None, obj_name, intent, n_frames, -1

    obj_verts = np.array(obj_mesh.vertices)  # (M_mesh, 3)

    # Object transform per frame
    obj_params = data['object'].item()['params']
    obj_transl = obj_params['transl']  # (T, 3)
    obj_orient = obj_params['global_orient']  # (T, 3) axis-angle

    results = []

    for fi in range(n_frames):
        if not hand_contact_per_frame[fi].any():
            continue

        # Get contacted vertex indices
        contact_mask = hand_contact_per_frame[fi]  # (M,)

        # Contact vertices in object's canonical frame
        # GRAB's contact.object corresponds to the mesh vertices directly
        contacted_indices = np.where(contact_mask)[0]

        if len(contacted_indices) == 0:
            continue

        # Map to mesh vertices (GRAB contact mesh may have more vertices than display mesh)
        # Use the indices that are within mesh vertex range
        valid_indices = contacted_indices[contacted_indices < len(obj_verts)]
        if len(valid_indices) == 0:
            continue

        contact_pts = obj_verts[valid_indices]
        force_center = contact_pts.mean(axis=0)

        # Get normals at contact points
        if hasattr(obj_mesh, 'vertex_normals') and len(obj_mesh.vertex_normals) > 0:
            contact_normals = np.array(obj_mesh.vertex_normals)[valid_indices]
        else:
            contact_normals = np.zeros_like(contact_pts)

        # Count distinct finger parts
        contact_parts = contact_obj[fi, contacted_indices]
        n_distinct_parts = len(np.unique(contact_parts))

        results.append({
            'frame': fi,
            'contact_pts': contact_pts.astype(np.float32),
            'normals': contact_normals.astype(np.float32),
            'force_center': force_center.astype(np.float32),
            'n_contact_pts': len(contact_pts),
            'n_parts': n_distinct_parts,
        })

    return results, obj_name, intent, n_frames, int(frames_with_hand_contact)


def save_hdf5(results, obj_name, intent, seq_name, output_dir):
    """Save extracted contacts as HDF5 (same format as OakInk extraction)."""
    seq_dir = os.path.join(output_dir, obj_name, f"{seq_name}")
    os.makedirs(seq_dir, exist_ok=True)

    saved = 0
    for r in results:
        fi = r['frame']
        out_path = os.path.join(seq_dir, f"frame_{fi:06d}.hdf5")
        with h5py.File(out_path, 'w') as f:
            f.create_dataset("finger_contact_pts", data=r['contact_pts'])
            f.create_dataset("finger_contact_normals", data=r['normals'])
            f.create_dataset("force_center", data=r['force_center'])
            f.attrs['obj_id'] = obj_name
            f.attrs['frame'] = fi
            f.attrs['n_contact_pts'] = r['n_contact_pts']
            f.attrs['n_parts'] = r['n_parts']
            f.attrs['source'] = 'grab'
            f.attrs['intent'] = intent
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract contacts from GRAB dataset")
    parser.add_argument("--min_frames", type=int, default=MIN_CONTACT_FRAMES,
                        help="Min frames with hand contact to keep sequence")
    parser.add_argument("--obj_name", type=str, default=None,
                        help="Only process sequences for this object")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_seqs", type=int, default=0,
                        help="Max sequences to process (0=all)")
    args = parser.parse_args()

    output_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("M1: GRAB Contact Extraction")
    print("=" * 60)
    print(f"  GRAB data:     {GRAB_SEQS}")
    print(f"  GRAB meshes:   {GRAB_MESHES}")
    print(f"  Output:        {output_dir}")
    print(f"  Min frames:    {args.min_frames}")
    if args.obj_name:
        print(f"  Filter obj:    {args.obj_name}")
    print()

    # Discover sequences
    all_seqs = []
    for subj in sorted(os.listdir(GRAB_SEQS)):
        subj_dir = os.path.join(GRAB_SEQS, subj)
        if not os.path.isdir(subj_dir):
            continue
        for npz in sorted(os.listdir(subj_dir)):
            if not npz.endswith('.npz'):
                continue
            if args.obj_name:
                # Filter: npz name starts with obj_name_
                if not npz.startswith(args.obj_name + "_"):
                    # Also try without underscore
                    if not npz.startswith(args.obj_name.replace("_", "") + "_"):
                        continue
            all_seqs.append((subj, npz, os.path.join(subj_dir, npz)))

    if args.max_seqs > 0:
        all_seqs = all_seqs[:args.max_seqs]

    print(f"  Found {len(all_seqs)} sequences")
    print()

    total_frames = 0
    total_saved = 0
    success = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for i, (subj, npz, npz_path) in enumerate(all_seqs, 1):
        seq_name = f"{subj}_{os.path.splitext(npz)[0]}"
        t1 = time.time()

        try:
            results, obj_name, intent, n_frames, n_contact_frames = process_sequence(
                npz_path, min_contact_frames=args.min_frames
            )
        except Exception as e:
            print(f"  [{i}/{len(all_seqs)}] {subj}/{npz}: ❌ Error: {e}")
            failed += 1
            continue

        elapsed = time.time() - t1

        if results is None:
            if n_contact_frames == -1:
                print(f"  [{i}/{len(all_seqs)}] {subj}/{npz}: ⚠️ mesh not found for '{obj_name}' ({elapsed:.1f}s)")
            else:
                print(f"  [{i}/{len(all_seqs)}] {subj}/{npz}: ⚠️ too few contact frames "
                      f"({n_contact_frames}/{n_frames}) ({elapsed:.1f}s)")
            skipped += 1
            continue

        saved = save_hdf5(results, obj_name, intent, seq_name, output_dir)
        total_frames += len(results)
        total_saved += saved
        success += 1

        # ETA
        avg_time = (time.time() - t0) / i
        eta = avg_time * (len(all_seqs) - i)
        eta_str = f"{eta / 60:.1f}min" if eta < 3600 else f"{eta / 3600:.1f}h"

        print(f"  [{i}/{len(all_seqs)}] {subj}/{npz}: ✅ {len(results)} frames, "
              f"{saved} saved ({elapsed:.1f}s) ETA={eta_str}")

    elapsed_total = time.time() - t0

    # Summary
    print()
    print("=" * 60)
    print(f"  ✅ Success:  {success}")
    print(f"  ⚠️ Skipped:  {skipped}")
    print(f"  ❌ Failed:   {failed}")
    print(f"  📊 Total frames saved: {total_saved}")
    print(f"  ⏱️ Total time: {elapsed_total:.0f}s ({elapsed_total / 60:.1f}min)")
    print("=" * 60)

    # Save summary
    summary = {
        'source': 'GRAB',
        'success': success,
        'skipped': skipped,
        'failed': failed,
        'total_frames': total_saved,
        'elapsed_seconds': round(elapsed_total, 1),
    }
    summary_path = os.path.join(output_dir, "summary_grab.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {summary_path}")


if __name__ == "__main__":
    main()
