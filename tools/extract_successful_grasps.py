#!/usr/bin/env python3
"""
从 robot_gt + robot_gt_random 提取成功抓取 → 生成 replay HDF5.
用于二次验证 + 接触点提取.

用法:
    cd /home/lyh/Project/Affordance2Grasp
    python3 tools/extract_successful_grasps.py
"""
import os, glob, h5py, numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIRS = [
    os.path.join(PROJ, 'output', 'robot_gt'),
    os.path.join(PROJ, 'output', 'robot_gt_random'),
]
MESH_DIR = os.path.join(PROJ, 'data_hub', 'meshes', 'v1')
OUTPUT_DIR = os.path.join(PROJ, 'output', 'grasps_verified')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 收集所有成功物体
all_meshes = sorted(glob.glob(os.path.join(MESH_DIR, '*.obj')))
total = 0
total_candidates = 0

for mp in all_meshes:
    obj_id = os.path.splitext(os.path.basename(mp))[0]
    candidates = []

    for gt_dir in GT_DIRS:
        gt_path = os.path.join(gt_dir, f'{obj_id}_robot_gt.hdf5')
        if not os.path.exists(gt_path):
            continue
        with h5py.File(gt_path, 'r') as f:
            if not f.attrs.get('success', False):
                continue
            if 'successful_grasps' not in f:
                continue
            for key in f['successful_grasps'].keys():
                g = f[f'successful_grasps/{key}']
                candidates.append({
                    'name': g.attrs['name'],
                    'score': g.attrs['score'],
                    'position': g['grasp_point'][:],
                    'rotation': g['rotation'][:],
                    'gripper_width': g.attrs['gripper_width'],
                    'approach_type': g.attrs['approach_type'],
                })

    if not candidates:
        continue

    # 写 replay HDF5
    out_path = os.path.join(OUTPUT_DIR, f'{obj_id}_grasp.hdf5')
    with h5py.File(out_path, 'w') as f:
        m = f.create_group('metadata')
        m.attrs['obj_id'] = obj_id
        m.attrs['mesh_path'] = os.path.abspath(mp)
        m.attrs['method'] = 'second_verification'
        cg = f.create_group('candidates')
        cg.attrs['n_candidates'] = len(candidates)
        for i, c in enumerate(candidates):
            ci = cg.create_group(f'candidate_{i}')
            ci.attrs['name'] = c['name']
            ci.attrs['score'] = c['score']
            ci.attrs['gripper_width'] = c['gripper_width']
            ci.attrs['approach_type'] = c['approach_type']
            ci.create_dataset('position', data=c['position'])
            ci.create_dataset('rotation', data=c['rotation'])
        a = f.create_group('affordance')
        a.attrs['n_contact'] = 0

    print(f"  {obj_id}: {len(candidates)} 个成功抓取 → {os.path.basename(out_path)}")
    total += 1
    total_candidates += len(candidates)

print(f"\n总计: {total} 个物体, {total_candidates} 个候选抓取")
print(f"输出: {OUTPUT_DIR}")
