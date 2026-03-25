#!/usr/bin/env python3
"""
M2 Random Grasp Sampler v2
==========================
内部采样 + ±XYZ 6方向 + Raycast 居中 + 评分系统 + HP引导

迭代生成: 每批20点×6方向 → 评分 → 不够20个>60分 → 再来一批
最终输出: top 20 高质量候选 (分数>60)

用法:
    cd /home/lyh/Project/Affordance2Grasp
    python3 tools/random_grasp_sampler.py --obj A01001           # 单个物体
    python3 tools/random_grasp_sampler.py --obj A01001 --vis     # 生成 + 可视化
    python3 tools/random_grasp_sampler.py --all                  # 全部物体
"""
import os, sys, glob, argparse
import numpy as np
import trimesh
import h5py
from scipy.spatial.transform import Rotation

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(PROJ, 'data_hub', 'meshes', 'v1')
HP_DIR = os.path.join(PROJ, 'data_hub', 'human_prior')
OUTPUT_DIR = os.path.join(PROJ, 'output', 'grasps_random')
MAX_GRIPPER_OPEN = 0.08
MIN_GRIPPER_WIDTH = 0.005
N_POINTS_PER_BATCH = 20     # 每批采样点数
TARGET_HIGH_QUALITY = 20     # 目标高质量候选数
SCORE_THRESHOLD = 60.0       # 高质量门槛
MAX_BATCHES = 10             # 最大迭代批次 (防止死循环)

# 6 个固定接近方向
APPROACH_DIRS = [
    np.array([0, 0, -1], dtype=np.float32),  # 从上方
    np.array([0, 0, 1], dtype=np.float32),   # 从下方
    np.array([0, 1, 0], dtype=np.float32),   # 从正面
    np.array([0, -1, 0], dtype=np.float32),  # 从背面
    np.array([1, 0, 0], dtype=np.float32),   # 从左侧
    np.array([-1, 0, 0], dtype=np.float32),  # 从右侧
]


def load_human_prior(obj_id):
    for name in [f'{obj_id}.hdf5', f'grab_{obj_id}.hdf5']:
        path = os.path.join(HP_DIR, name)
        if os.path.exists(path):
            with h5py.File(path, 'r') as f:
                return f['point_cloud'][()].astype(np.float32), f['human_prior'][()].astype(np.float32)
    return None, None


def sample_points(mesh, hp_pc, hp_labels, n_total, has_hp):
    """采样内部点: 有HP时一半引导 一半随机."""
    bbox_min, bbox_max = mesh.bounds[0], mesh.bounds[1]
    points = []
    
    # HP 引导采样
    if has_hp:
        contact_idx = np.where(hp_labels > 0.5)[0]
        if len(contact_idx) > 0:
            n_hp = n_total // 2
            for _ in range(n_hp * 5):
                seed = hp_pc[np.random.choice(contact_idx)]
                pt = seed + np.random.uniform(-0.01, 0.01, 3).astype(np.float32)
                if np.all(pt >= bbox_min) and np.all(pt <= bbox_max):
                    if mesh.contains([pt])[0]:
                        points.append(pt)
                        if len(points) >= n_hp:
                            break
    
    # 随机采样补齐
    n_remain = n_total - len(points)
    if n_remain > 0:
        all_pts = np.random.uniform(bbox_min, bbox_max, size=(n_remain * 10, 3))
        inside = mesh.contains(all_pts)
        for p in all_pts[inside][:n_remain]:
            points.append(p.astype(np.float32))
    
    return points


def choose_finger_dir(approach):
    up = np.array([0, 0, 1], dtype=np.float32)
    if abs(np.dot(approach, up)) > 0.9:
        return np.array([1, 0, 0], dtype=np.float32)
    else:
        finger = np.cross(approach, up)
        return (finger / (np.linalg.norm(finger) + 1e-8)).astype(np.float32)


def make_rotation_matrix(approach, finger_dir):
    z = approach / (np.linalg.norm(approach) + 1e-8)
    x = finger_dir / (np.linalg.norm(finger_dir) + 1e-8)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-8)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-8)
    R = np.column_stack([x, y, z]).astype(np.float32)
    if np.linalg.det(R) < 0:
        R = np.column_stack([-x, y, z]).astype(np.float32)
    return R


def score_candidate(width, approach, grasp_center, z_min, z_max):
    """评分: 宽度(40%) + 高度(30%) + approach偏好(15%) + 预留HP(15%)."""
    # 宽度分: 2-5cm 最优
    ws = max(0, 1.0 - abs(width - 0.035) / 0.045)
    # 高度分: 物体中间最好
    zr = (grasp_center[2] - z_min) / (z_max - z_min + 1e-8)
    hs = max(0, 1.0 - abs(zr - 0.5) * 2)
    # approach偏好: 水平略好于从上 (从上方已经通过clearance验证, 不需要额外加分)
    # 从上方=0.7, 水平=1.0, 从下方=0.3
    if abs(approach[2]) > 0.9:
        ap = 0.5 if approach[2] < 0 else 0.3  # 从上=0.5, 从下=0.3
    else:
        ap = 1.0  # 水平方向
    
    return (0.40 * ws + 0.30 * hs + 0.15 * ap + 0.15 * 0.0) * 100


def check_finger_reachable(mesh, grasp_center, approach, max_finger_depth=0.04):
    """检查手指能否从 approach 方向到达抓取中心.
    
    从 grasp_center 沿 -approach (向外) 射线 → 打到物体表面
      距离 ≤ 4cm (手指长度) → 手指够得到 ✅
      距离 > 4cm → 手指伸不到 ❌
    """
    hits, _, _ = mesh.ray.intersects_location([grasp_center], [-approach])
    if len(hits) == 0:
        return True  # 没打到表面 = 抓取中心在物体外部边缘, 一定够得到
    
    dists = np.linalg.norm(hits - grasp_center, axis=1)
    nearest_dist = np.min(dists)
    
    return nearest_dist <= max_finger_depth


def generate_one_batch(mesh, points, z_min, z_max):
    """从一批采样点生成候选并评分."""
    candidates = []
    for pt in points:
        for approach in APPROACH_DIRS:
            finger_dir = choose_finger_dir(approach)
            
            hits_pos, _, _ = mesh.ray.intersects_location([pt], [finger_dir])
            hits_neg, _, _ = mesh.ray.intersects_location([pt], [-finger_dir])
            
            if len(hits_pos) == 0 or len(hits_neg) == 0:
                continue
            
            d_pos = np.linalg.norm(hits_pos - pt, axis=1)
            d_neg = np.linalg.norm(hits_neg - pt, axis=1)
            nearest_pos = hits_pos[np.argmin(d_pos)]
            nearest_neg = hits_neg[np.argmin(d_neg)]
            
            width = np.linalg.norm(nearest_pos - nearest_neg)
            if width > MAX_GRIPPER_OPEN or width < MIN_GRIPPER_WIDTH:
                continue
            
            grasp_center = ((nearest_pos + nearest_neg) / 2.0).astype(np.float32)
            
            # ⭐ 手指深度检查: 手指能否从这个方向到达抓取点 (≤4cm)
            if not check_finger_reachable(mesh, grasp_center, approach):
                continue
            
            R = make_rotation_matrix(approach, finger_dir)
            gripper_width = float(np.clip(width + 0.005, 0.01, MAX_GRIPPER_OPEN))
            score = score_candidate(width, approach, grasp_center, z_min, z_max)
            
            candidates.append({
                'name': '',  # filled later
                'position': grasp_center,
                'grasp_point': grasp_center,
                'rotation': R,
                'gripper_width': gripper_width,
                'approach': approach.copy(),
                'finger_dir': finger_dir.copy(),
                'contact_L': nearest_neg.astype(np.float32),
                'contact_R': nearest_pos.astype(np.float32),
                'score': score,
                'cross_section_width': float(width),
            })
    return candidates


def generate_candidates_iterative(mesh, obj_id):
    """迭代生成候选, 直到有 TARGET_HIGH_QUALITY 个分数 > SCORE_THRESHOLD."""
    hp_pc, hp_labels = load_human_prior(obj_id)
    has_hp = hp_pc is not None and np.any(hp_labels > 0.5)
    
    z_min, z_max = mesh.bounds[0][2], mesh.bounds[1][2]
    all_candidates = []
    
    for batch in range(MAX_BATCHES):
        pts = sample_points(mesh, hp_pc, hp_labels, N_POINTS_PER_BATCH, has_hp)
        new_cands = generate_one_batch(mesh, pts, z_min, z_max)
        all_candidates.extend(new_cands)
        
        # 统计高质量候选
        high_quality = [c for c in all_candidates if c['score'] >= SCORE_THRESHOLD]
        hp_tag = f"({len(pts)} pts: {'HP+rnd' if has_hp else 'rnd'})"
        print(f"    batch {batch+1}: +{len(new_cands)} 候选, "
              f"高质量≥{SCORE_THRESHOLD:.0f}分: {len(high_quality)}/{TARGET_HIGH_QUALITY} {hp_tag}")
        
        if len(high_quality) >= TARGET_HIGH_QUALITY:
            break
    
    # 按分数排序, 取 top TARGET_HIGH_QUALITY
    all_candidates.sort(key=lambda c: -c['score'])
    selected = all_candidates[:TARGET_HIGH_QUALITY]
    
    # 重命名
    for i, c in enumerate(selected):
        c['name'] = f'raycast_{i}'
    
    print(f"  → 最终选出 {len(selected)} 个候选 "
          f"(分数: {selected[0]['score']:.1f} ~ {selected[-1]['score']:.1f})")
    
    return selected


def save_candidates_hdf5(candidates, obj_id, mesh_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{obj_id}_grasp.hdf5')
    
    with h5py.File(path, 'w') as f:
        m = f.create_group('metadata')
        m.attrs['obj_id'] = obj_id
        m.attrs['mesh_path'] = os.path.abspath(mesh_path)
        m.attrs['method'] = 'raycast_scored_v2'
        
        cg = f.create_group('candidates')
        cg.attrs['n_candidates'] = len(candidates)
        for i, c in enumerate(candidates):
            ci = cg.create_group(f'candidate_{i}')
            ci.create_dataset('position', data=c['position'])
            ci.create_dataset('grasp_point', data=c['grasp_point'])
            ci.create_dataset('rotation', data=c['rotation'])
            ci.attrs['name'] = c['name']
            ci.attrs['score'] = c['score']
            ci.attrs['gripper_width'] = c['gripper_width']
            ci.attrs['cross_section_width'] = c.get('cross_section_width', 0)
        
        if candidates:
            best = candidates[0]
            g = f.create_group('grasp')
            g.create_dataset('position', data=best['position'])
            g.create_dataset('grasp_point', data=best['grasp_point'])
            g.create_dataset('rotation', data=best['rotation'])
            quat_xyzw = Rotation.from_matrix(best['rotation']).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            g.create_dataset('quaternion_wxyz', data=quat_wxyz.astype(np.float32))
            g.attrs['gripper_width'] = best['gripper_width']
        
        aff = f.create_group('affordance')
        aff.attrs['n_contact'] = 0
    return path


def visualize_candidates(mesh, candidates, obj_id):
    import open3d as o3d
    
    geometries = []
    
    N_VIS = 30000
    vis_pc, _ = trimesh.sample.sample_surface(mesh, N_VIS)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vis_pc)
    pcd.paint_uniform_color([0.75, 0.75, 0.82])
    geometries.append(pcd)
    
    for i, c in enumerate(candidates):
        center = c['grasp_point']
        
        sphere_L = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        sphere_L.translate(c['contact_L'])
        sphere_L.paint_uniform_color([0.9, 0.1, 0.1])
        geometries.append(sphere_L)
        
        sphere_R = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        sphere_R.translate(c['contact_R'])
        sphere_R.paint_uniform_color([0.1, 0.3, 0.9])
        geometries.append(sphere_R)
        
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
        center_sphere.translate(center)
        center_sphere.paint_uniform_color([1.0, 0.85, 0.0])
        geometries.append(center_sphere)
        
        pts = np.array([c['contact_L'], c['contact_R']])
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(pts)
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[0.6, 0.6, 0.6]])
        geometries.append(line)
        
        arrow_end = center - c['approach'] * 0.05
        arrow_line = o3d.geometry.LineSet()
        arrow_line.points = o3d.utility.Vector3dVector([center, arrow_end])
        arrow_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        arrow_line.colors = o3d.utility.Vector3dVector([[0.2, 0.8, 0.2]])
        geometries.append(arrow_line)
    
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    geometries.append(coord)
    
    print(f"\n  🔍 Open3D: {obj_id} (top {len(candidates)} 候选)")
    print(f"     🔴 红=左接触  🔵 蓝=右接触  🟡 黄=中心  🟢 绿=approach")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Top {len(candidates)} — {obj_id}", width=1200, height=800)
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    opt.point_size = 2.0
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description='Grasp Sampler v2 (Scored + Iterative)')
    parser.add_argument('--obj', help='单个物体 ID')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    args = parser.parse_args()
    
    if args.obj:
        obj_list = [(args.obj, os.path.join(MESH_DIR, f'{args.obj}.obj'))]
    elif args.all:
        all_meshes = sorted(glob.glob(os.path.join(MESH_DIR, '*.obj')))
        obj_list = [(os.path.splitext(os.path.basename(m))[0], m) for m in all_meshes]
    else:
        print("用法: python3 tools/random_grasp_sampler.py --obj A01001 [--vis]")
        return
    
    print("=" * 60)
    print("  Grasp Sampler v2 (Raycast + Scored + Iterative)")
    print(f"  Target: {TARGET_HIGH_QUALITY} candidates ≥ {SCORE_THRESHOLD} pts")
    print("=" * 60)
    
    generated = 0
    for idx, (obj_id, mesh_path) in enumerate(obj_list):
        print(f"\n[{idx+1}/{len(obj_list)}] {obj_id}")
        
        if not args.force:
            out_path = os.path.join(args.output_dir, f'{obj_id}_grasp.hdf5')
            if os.path.exists(out_path):
                print(" ⏭️ (已生成)")
                continue
        
        if not os.path.exists(mesh_path):
            print(f" ❌ mesh 不存在")
            continue
        
        mesh = trimesh.load(mesh_path, force='mesh')
        ext = mesh.bounding_box.extents * 100
        print(f"  尺寸: {ext[0]:.1f}×{ext[1]:.1f}×{ext[2]:.1f} cm")
        
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
        
        candidates = generate_candidates_iterative(mesh, obj_id)
        
        if candidates:
            path = save_candidates_hdf5(candidates, obj_id, mesh_path, args.output_dir)
            print(f"  ✅ → {os.path.basename(path)} ({len(candidates)} 候选)")
            generated += 1
            
            if args.vis:
                visualize_candidates(mesh, candidates, obj_id)
        else:
            print(f"  ⚠️ 无有效候选")
    
    print(f"\n{'='*60}")
    print(f"  完成! 生成 {generated} 个物体的候选")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
