#!/usr/bin/env python3
"""可视化某个物体的全部 grasp candidates (3D 箭头图)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', required=True, help='Object ID, e.g. A16013')
    parser.add_argument('--out', default=None, help='Output PNG path')
    args = parser.parse_args()

    obj_id = args.obj
    mesh_path = f"data_hub/meshes/v1/{obj_id}.obj"
    hdf5_path = f"output/grasps/{obj_id}_grasp.hdf5"
    out_path = args.out or f"output/vis/{obj_id}_candidates.png"

    if not os.path.exists(mesh_path):
        print(f"❌ Mesh not found: {mesh_path}")
        return
    if not os.path.exists(hdf5_path):
        print(f"❌ HDF5 not found: {hdf5_path}")
        return

    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    verts = np.array(mesh.vertices)

    # Load candidates
    with h5py.File(hdf5_path, 'r') as f:
        n_cands = f.attrs.get('num_candidates', 0)
        print(f"物体: {obj_id}  候选数: {n_cands}")
        candidates = []
        for i in range(n_cands):
            g = f[f'candidate_{i}']
            cand = {
                'name': g.attrs.get('name', f'cand_{i}'),
                'position': np.array(g['position']),
                'rotation': np.array(g['rotation']),
                'gripper_width': float(g.attrs.get('gripper_width', 0.04)),
                'grasp_point': np.array(g['grasp_point']),
                'approach_type': g.attrs.get('approach_type', 'horizontal'),
                'score': float(g.attrs.get('score', 0)),
            }
            candidates.append(cand)
            print(f"  [{i}] {cand['name']:25s} | type={cand['approach_type']:10s} | "
                  f"width={cand['gripper_width']:.3f} | score={cand['score']:.1f}")

    # ---- 绘图 ----
    fig = plt.figure(figsize=(18, 6))

    # 3 个视角
    views = [
        ("正面 (XZ)", 0, 0),
        ("侧面 (YZ)", 0, 90),
        ("俯视 (XY)", 90, -90),
    ]

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(candidates), 1)))

    for vi, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, vi+1, projection='3d')
        ax.set_title(title, fontsize=12)

        # 画 mesh (简化: 只画点云)
        sample_pts, _ = mesh.sample(800, return_index=True)
        ax.scatter(sample_pts[:, 0], sample_pts[:, 1], sample_pts[:, 2],
                   c='lightgray', s=1, alpha=0.4)

        # 画每个候选
        TCP_OFFSET = 0.105
        for ci, cand in enumerate(candidates):
            gp = cand['grasp_point']
            rot = cand['rotation']
            approach = rot[:, 2]  # 第3列 = approach 方向
            finger_dir = rot[:, 0]  # 第1列 = finger open 方向
            w = cand['gripper_width'] / 2

            color = colors[ci]

            # 画 approach 箭头 (从 grasp_point 往外延伸)
            arrow_len = 0.06
            ax.quiver(gp[0], gp[1], gp[2],
                      -approach[0], -approach[1], -approach[2],
                      length=arrow_len, color=color, arrow_length_ratio=0.3, linewidth=2)

            # 画两个指尖位置
            tip1 = gp + finger_dir * w
            tip2 = gp - finger_dir * w
            ax.plot([tip1[0], tip2[0]], [tip1[1], tip2[1]], [tip1[2], tip2[2]],
                    color=color, linewidth=2, marker='o', markersize=3)

            # 标注名字
            label_pt = gp - approach * arrow_len * 1.2
            ax.text(label_pt[0], label_pt[1], label_pt[2], str(ci),
                    fontsize=6, color=color, ha='center')

        ax.view_init(elev=elev, azim=azim)

        # 保持等比例
        ranges = np.array([verts[:, i].max() - verts[:, i].min() for i in range(3)])
        max_range = ranges.max() * 0.7
        mid = verts.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # 图例
    fig.suptitle(f'{obj_id} — {len(candidates)} Grasp Candidates', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化已保存: {out_path}")

if __name__ == '__main__':
    main()
