#!/usr/bin/env python3
"""
Combined 3-Panel Visualization (Best-of-All)
=============================================
Panel 1: 3D Object Model (带光照渲染)
Panel 2: 半透明 mesh + 可抓区域(浅绿) + Force Center (绿星) + Grasp Point (红十字) + 连线
Panel 3: 密集 affordance 热力图 + Franka 真实夹爪 + Force Center

用法:
    cd /home/lyh/Project/Affordance2Grasp
    python analysis/vis_combined.py --obj_id A16013
"""

import os, sys, argparse
import numpy as np
import trimesh
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Paths ----
OBJ_DIR = "/home/lyh/Project/OakInk/image/obj"
GRASP_DIR = "/home/lyh/Project/Affordance2Grasp/output/grasps"
OUT_DIR = "/home/lyh/Project/Affordance2Grasp/output/analysis"
FRANKA_MESH_DIR = "/home/lyh/Project/curobo/src/curobo/content/assets/robot/franka_description/meshes/visual"


# ===================== Utility =====================

def set_axes_equal(ax, pts, scale=1.3):
    """等比例坐标轴."""
    center = pts.mean(axis=0)
    span = (pts.max(axis=0) - pts.min(axis=0)).max() / 2 * scale
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)


def get_dense_affordance(mesh, points_1k, probs_1k, n_dense=8000):
    """从 mesh 表面密集采样, 用 KNN 距离加权插值 affordance."""
    from scipy.spatial import cKDTree
    dense_pts, _ = trimesh.sample.sample_surface(mesh, n_dense)
    tree = cKDTree(points_1k)
    dists, idxs = tree.query(dense_pts, k=3)
    weights = 1.0 / (dists + 1e-8)
    weights = weights / weights.sum(axis=1, keepdims=True)
    dense_probs = (probs_1k[idxs] * weights).sum(axis=1)
    return dense_pts.astype(np.float32), dense_probs.astype(np.float32)


def compute_grippable_region(mesh, finger_depth=0.04, n_surface=4000, n_layers=8):
    """
    计算夹爪可抓区域: 物体内部距表面 <= finger_depth 的空间.
    通过沿法线向内采样多层点, 过滤保留在 mesh 内部的点.
    """
    # 密集采样表面点 + 法线
    surface_pts, face_idx = trimesh.sample.sample_surface(mesh, n_surface)
    face_normals = mesh.face_normals[face_idx]

    # 沿法线向内偏移多层 (从 0.002m 到 finger_depth)
    depths = np.linspace(0.002, finger_depth, n_layers)
    all_pts = []
    for d in depths:
        interior_pts = surface_pts - face_normals * d  # 沿法线反方向 = 向内
        all_pts.append(interior_pts)

    all_pts = np.vstack(all_pts)

    # 只保留真正在 mesh 内部的点
    inside_mask = mesh.contains(all_pts)
    grippable_pts = all_pts[inside_mask]

    return grippable_pts.astype(np.float32)


# ===================== Panel 1: Mesh =====================

def render_mesh(ax, mesh, elev=15, azim=-60):
    """渲染 mesh — 带简单光照的灰色面 + 细边缘线."""
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-8)

    light_dir = np.array([0.3, -0.5, 0.8])
    light_dir /= np.linalg.norm(light_dir)
    intensity = np.clip(normals @ light_dir, 0.15, 1.0)

    base_color = np.array([0.82, 0.82, 0.82])
    face_colors = np.outer(intensity, base_color)
    face_colors = np.clip(face_colors, 0, 1)
    alpha_col = np.ones((len(face_colors), 1)) * 0.95
    face_colors = np.hstack([face_colors, alpha_col])

    tri_verts = verts[faces]
    poly = Poly3DCollection(tri_verts)
    poly.set_facecolor(face_colors)
    poly.set_edgecolor((0.35, 0.35, 0.35, 0.25))
    poly.set_linewidth(0.15)
    ax.add_collection3d(poly)

    set_axes_equal(ax, verts)
    ax.view_init(elev=elev, azim=azim)


# ===================== Panel 2: Grasp Pose Detail =====================

def draw_mesh_transparent(ax, mesh, alpha=0.12, color='lightskyblue'):
    """半透明 mesh."""
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    max_faces = 2000
    if len(faces) > max_faces:
        idx = np.random.choice(len(faces), max_faces, replace=False)
        faces = faces[idx]
    polygons = verts[faces]
    poly = Poly3DCollection(polygons, alpha=alpha, facecolor=color,
                            edgecolor='gray', linewidth=0.1)
    ax.add_collection3d(poly)


def draw_gripper_wireframe(ax, grasp_pt, rot, width):
    """画夹爪线框 — grasp_pt=指尖中点, 手指向后延伸."""
    x_open = rot[:, 0]
    y_body = rot[:, 1]
    z_approach = rot[:, 2]
    half_w = width / 2
    fl = 0.04
    fw = 0.004

    for sign, color_f in [(1, '#333333'), (-1, '#666666')]:
        tip = grasp_pt + sign * x_open * (half_w - fw)
        base = tip - z_approach * fl  # 手指从指尖向后延伸
        c1 = base - y_body * fw
        c2 = base + y_body * fw
        c3 = tip + y_body * fw
        c4 = tip - y_body * fw
        for a, b in [(c1, c2), (c2, c3), (c3, c4), (c4, c1)]:
            ax.plot3D([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                      c=color_f, linewidth=1.5, alpha=0.8)

    # 横梁在手指根部 (后方)
    crossbar = grasp_pt - z_approach * fl
    left_base = crossbar + x_open * half_w
    right_base = crossbar - x_open * half_w
    ax.plot3D([left_base[0], right_base[0]],
              [left_base[1], right_base[1]],
              [left_base[2], right_base[2]],
              c='#444444', linewidth=2, alpha=0.7)


# 每个方向的颜色
CANDIDATE_COLORS = {
    'horizontal_front': '#e74c3c',   # 红
    'horizontal_right': '#2980b9',   # 蓝
    'horizontal_left':  '#f39c12',   # 橙
    'top_down':         '#8e44ad',   # 紫
}
CANDIDATE_LABELS = {
    'horizontal_front': 'Front',
    'horizontal_right': 'Right',
    'horizontal_left':  'Left',
    'top_down':         'Top-down',
}


def render_grasp_detail(ax, mesh, candidates, force_center, best_idx=0,
                        elev=15, azim=-60):
    """中间面板: 半透明 mesh + 所有可行抓取候选 + Force Center."""
    verts = np.array(mesh.vertices)

    draw_mesh_transparent(ax, mesh, alpha=0.10, color='lightskyblue')

    # Force Center (绿色大星)
    if force_center is not None:
        ax.scatter(*force_center, c='lime', s=200, marker='*',
                   edgecolors='darkgreen', linewidth=1.2, zorder=10,
                   label='Force center')

    # 画每个候选的夹爪 + 抓取点
    for i, c in enumerate(candidates):
        name = c['name']
        color = CANDIDATE_COLORS.get(name, '#888888')
        label = CANDIDATE_LABELS.get(name, name)
        rot = c['rotation']
        width = c['gripper_width']
        gp = c.get('grasp_point')  # 指尖中点
        is_best = (i == best_idx)

        # 夹爪线框 (画在 grasp_point 指尖中点处)
        if gp is not None:
            draw_gripper_wireframe_colored(ax, gp, rot, width, color=color)

        # 抓取点标记
        if gp is not None:
            marker = '★' if is_best else '+'
            size = 120 if is_best else 80
            ax.scatter(*gp, c=color, s=size, marker='+',
                       linewidth=2.5, zorder=10,
                       label=f'{label} ({c["cross_section_width"]*100:.1f}cm)'
                             + (' ⭐' if is_best else ''))

        # 连线: Force Center → Grasp Point
        if force_center is not None and gp is not None:
            ax.plot3D([force_center[0], gp[0]],
                      [force_center[1], gp[1]],
                      [force_center[2], gp[2]],
                      '--', color=color, linewidth=1.2, alpha=0.5)

    set_axes_equal(ax, verts)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(fontsize=6, loc='upper left', framealpha=0.8)


def draw_gripper_wireframe_colored(ax, grasp_pt, rot, width, color='#333333'):
    """画彩色夹爪线框 — grasp_pt=指尖中点, 手指向后延伸."""
    x_open = rot[:, 0]
    y_body = rot[:, 1]
    z_approach = rot[:, 2]
    half_w = width / 2
    fl = 0.04
    fw = 0.004

    for sign in [1, -1]:
        tip = grasp_pt + sign * x_open * (half_w - fw)
        base = tip - z_approach * fl  # 手指从指尖向后延伸
        c1 = base - y_body * fw
        c2 = base + y_body * fw
        c3 = tip + y_body * fw
        c4 = tip - y_body * fw
        for a, b in [(c1, c2), (c2, c3), (c3, c4), (c4, c1)]:
            ax.plot3D([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                      c=color, linewidth=1.5, alpha=0.8)

    # 横梁在手指根部 (后方)
    crossbar = grasp_pt - z_approach * fl
    left_base = crossbar + x_open * half_w
    right_base = crossbar - x_open * half_w
    ax.plot3D([left_base[0], right_base[0]],
              [left_base[1], right_base[1]],
              [left_base[2], right_base[2]],
              c=color, linewidth=2, alpha=0.7)


# ===================== Panel 3: Heatmap + Franka Gripper =====================

def draw_gripper_3d(ax, panda_hand_pos, grasp_rot, gripper_width, approach_offset=0.0):
    """用真实 Franka mesh 画夹爪 (hand + 2x finger)."""
    x_open = grasp_rot[:, 0]
    y_body = grasp_rot[:, 1]
    z_approach = grasp_rot[:, 2]

    R = np.column_stack([-y_body, z_approach, -x_open])
    gc = panda_hand_pos - z_approach * approach_offset + np.array([0, -0.01, 0])

    def render_mesh_part(ax, obj_path, offset, R, scale=1.0, color='#c8c8c8', alpha=0.8):
        m = trimesh.load(obj_path, force='mesh')
        verts = np.array(m.vertices) * scale
        faces = np.array(m.faces)
        verts_world = (R @ verts.T).T + offset

        v0, v1, v2 = verts_world[faces[:, 0]], verts_world[faces[:, 1]], verts_world[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-8)
        light = np.array([0.3, -0.5, 0.8])
        light /= np.linalg.norm(light)
        intensity = np.clip(np.abs(normals @ light), 0.25, 1.0)

        base = np.array(matplotlib.colors.to_rgb(color))
        fc = np.outer(intensity, base)
        fc = np.clip(fc, 0, 1)
        fc = np.hstack([fc, np.ones((len(fc), 1)) * alpha])

        tri = verts_world[faces]
        poly = Poly3DCollection(tri)
        poly.set_facecolor(fc)
        poly.set_edgecolor((0.5, 0.5, 0.5, 0.2))
        poly.set_linewidth(0.1)
        ax.add_collection3d(poly)

    finger_path = os.path.join(FRANKA_MESH_DIR, "finger.obj")
    if os.path.exists(finger_path):
        finger_gap = 0.04
        lf_offset = gc + x_open * finger_gap
        render_mesh_part(ax, finger_path, lf_offset, R, color='#d0d0d0', alpha=0.80)

        R_flip = np.column_stack([-y_body, z_approach, x_open])
        rf_offset = gc - x_open * finger_gap
        render_mesh_part(ax, finger_path, rf_offset, R_flip, color='#d0d0d0', alpha=0.80)


def render_heatmap_gripper(ax, dense_pts, dense_probs, panda_hand_pos, grasp_rot,
                           gripper_width, force_center, grasp_point=None,
                           elev=15, azim=-60):
    """右面板: 密集热力图 + Franka 真实夹爪 + Force Center."""
    # 热力图 (按概率排序, 高概率在前)
    order = np.argsort(dense_probs)
    ax.scatter(
        dense_pts[order, 0], dense_pts[order, 1], dense_pts[order, 2],
        c=cm.jet(dense_probs[order]), s=3, alpha=0.9,
        edgecolors='none', depthshade=True
    )

    # Franka 真实夹爪 (position 已是 panda_hand 位置)
    draw_gripper_3d(ax, panda_hand_pos, grasp_rot, gripper_width)

    # 虚线: panda_hand → grasp_point (指尖中点)
    z_approach = grasp_rot[:, 2]
    if grasp_point is not None:
        line_start = panda_hand_pos
        line_end = grasp_point
    else:
        line_start = panda_hand_pos
        line_end = panda_hand_pos + z_approach * 0.1034
    n_seg = 15
    for i in range(n_seg):
        if i % 2 == 0:
            t0 = i / n_seg
            t1 = (i + 1) / n_seg
            p0 = line_start + t0 * (line_end - line_start)
            p1 = line_start + t1 * (line_end - line_start)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    'k-', linewidth=1.5, alpha=0.4, zorder=15)

    # Force Center (绿星)
    if force_center is not None:
        ax.scatter(*force_center, c='lime', s=150, marker='*',
                   edgecolors='darkgreen', linewidth=1.2, zorder=10,
                   label='Force center')
        ax.legend(fontsize=7, loc='upper left')

    set_axes_equal(ax, dense_pts, scale=1.8)
    ax.view_init(elev=elev, azim=azim)


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser(description="Combined 3-Panel Visualization")
    parser.add_argument("--obj_id", type=str, default="A16013")
    parser.add_argument("--elev", type=float, default=15)
    parser.add_argument("--azim", type=float, default=-60)
    parser.add_argument("--n_dense", type=int, default=8000, help="密集采样点数")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    obj_id = args.obj_id
    mesh_path = os.path.join(OBJ_DIR, f"{obj_id}.obj")
    grasp_path = os.path.join(GRASP_DIR, f"{obj_id}_grasp.hdf5")

    print(f"Loading {obj_id}...")
    mesh = trimesh.load(mesh_path, force='mesh')

    with h5py.File(grasp_path, 'r') as f:
        points = f['affordance/points'][:]
        probs = f['affordance/contact_prob'][:]
        grasp_pos = f['grasp/position'][:]       # panda_hand EE 位置
        grasp_rot = f['grasp/rotation'][:]
        gripper_width = f['grasp'].attrs['gripper_width']
        best_name = f['grasp'].attrs.get('candidate_name', '')
        force_center = f['affordance/force_center'][:] if 'affordance/force_center' in f else None
        # 指尖中点 (用于可视化)
        grasp_point_best = f['grasp/grasp_point'][:] if 'grasp/grasp_point' in f else grasp_pos

        # 读取所有候选
        all_candidates = []
        if 'candidates' in f:
            n_cand = f['candidates'].attrs.get('n_candidates', 0)
            for i in range(n_cand):
                cg = f[f'candidates/candidate_{i}']
                all_candidates.append({
                    'name': str(cg.attrs['name']),
                    'position': cg['position'][:],           # panda_hand EE 位置
                    'grasp_point': cg['grasp_point'][:] if 'grasp_point' in cg else cg['position'][:],
                    'rotation': cg['rotation'][:],
                    'gripper_width': float(cg.attrs['gripper_width']),
                    'cross_section_width': float(cg.attrs['cross_section_width']),
                    'approach_type': str(cg.attrs['approach_type']),
                    'score': float(cg.attrs['score']),
                })

    # Fallback: 从 affordance 数据计算 force center (概率加权质心)
    if force_center is None:
        contact_mask = probs > 0.3
        if contact_mask.sum() > 0:
            weights = probs[contact_mask]
            force_center = np.average(points[contact_mask], weights=weights, axis=0)
            print(f"  Force center (computed): [{force_center[0]:.4f}, {force_center[1]:.4f}, {force_center[2]:.4f}]")

    # grasp_point 已在 HDF5 中单独存储, 不需要从 position 复制

    # 找到 best 候选的索引
    best_idx = 0
    for i, c in enumerate(all_candidates):
        if c['name'] == best_name:
            best_idx = i
            break

    print(f"  Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print(f"  Contacts (>0.3): {(probs > 0.3).sum()}/1024, Best: {best_name}")
    print(f"  Candidates: {len(all_candidates)}")
    for i, c in enumerate(all_candidates):
        marker = '⭐' if i == best_idx else '  '
        print(f"    {marker} {c['name']:>20s}  score={c['score']:.1f}  width={c['cross_section_width']*100:.1f}cm")
    if force_center is not None:
        print(f"  Force center: [{force_center[0]:.4f}, {force_center[1]:.4f}, {force_center[2]:.4f}]")

    # Dense sampling
    print(f"  Dense sampling: {args.n_dense} points...")
    dense_pts, dense_probs = get_dense_affordance(mesh, points, probs, args.n_dense)
    print(f"  Dense contacts (>0.3): {(dense_probs > 0.3).sum()}/{args.n_dense}")

    # ---- 3-Panel Figure ----
    fig = plt.figure(figsize=(21, 7), dpi=150)
    fig.patch.set_facecolor('white')
    fig.suptitle(f'Object: {obj_id}   |   Intent: Hold   |   {len(all_candidates)} candidates',
                 fontsize=16, fontweight='bold', y=0.97)

    # Panel 1: 3D Object Model
    ax1 = fig.add_subplot(131, projection='3d')
    render_mesh(ax1, mesh, elev=args.elev, azim=args.azim)
    ax1.set_title('3D Object Model', fontsize=12, pad=5)
    ax1.axis('off')
    ax1.set_facecolor('white')

    # Panel 2: All Grasp Candidates
    ax2 = fig.add_subplot(132, projection='3d')
    render_grasp_detail(ax2, mesh, all_candidates, force_center,
                        best_idx=best_idx,
                        elev=args.elev, azim=args.azim)
    ax2.set_title(f'Grasp Candidates ({len(all_candidates)})', fontsize=12, pad=5)
    ax2.axis('off')
    ax2.set_facecolor('white')

    # Panel 3: Affordance + Franka Gripper
    ax3 = fig.add_subplot(133, projection='3d')
    render_heatmap_gripper(ax3, dense_pts, dense_probs, grasp_pos, grasp_rot,
                           gripper_width, force_center,
                           grasp_point=grasp_point_best,
                           elev=args.elev, azim=args.azim)
    ax3.set_title('Affordance + Gripper', fontsize=12, pad=5)
    ax3.axis('off')
    ax3.set_facecolor('white')

    # Colorbar
    sm = cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.65])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label('Contact Probability', fontsize=10)

    plt.subplots_adjust(left=0.01, right=0.91, wspace=0.02, top=0.90, bottom=0.03)

    out_path = args.output or os.path.join(OUT_DIR, f"{obj_id}_combined.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
