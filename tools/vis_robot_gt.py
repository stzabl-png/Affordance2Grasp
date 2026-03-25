#!/usr/bin/env python3
"""
Robot GT 可视化: 显示成功抓取的夹爪位姿叠加在物体上.
用法:
    python3 tools/vis_robot_gt.py --obj A01001        # 单个物体
    python3 tools/vis_robot_gt.py --all                # 汇总统计
"""
import os, sys, glob, argparse
import numpy as np
import trimesh
import h5py

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(PROJ, 'data_hub', 'meshes', 'v1')
GT_DIRS = [
    os.path.join(PROJ, 'output', 'robot_gt_verified'),  # ★ 二次验证 (优先)
    os.path.join(PROJ, 'output', 'robot_gt'),
    os.path.join(PROJ, 'output', 'robot_gt_random'),
]


def vis_single(obj_id):
    """可视化单个物体的 Robot GT: 用 raycast 计算真实表面接触点."""
    import open3d as o3d
    mesh_path = os.path.join(MESH_DIR, f'{obj_id}.obj')
    if not os.path.exists(mesh_path):
        print(f"❌ mesh 不存在: {mesh_path}"); return
    
    # 加载 mesh 用于 raycast
    mesh = trimesh.load(mesh_path, force='mesh')
    
    grasp_data = []  # list of {mid, c1, c2, name}
    
    for gt_dir in GT_DIRS:
        gt_path = os.path.join(gt_dir, f'{obj_id}_robot_gt.hdf5')
        if not os.path.exists(gt_path): continue
        
        with h5py.File(gt_path, 'r') as f:
            if not f.attrs.get('success', False): continue
            if 'successful_grasps' not in f: continue
            
            src = os.path.basename(gt_dir)
            ns = f.attrs.get('n_successful', 0)
            print(f"📂 {src}: ✅ {ns} 成功")
            
            for key in f['successful_grasps'].keys():
                g = f[f'successful_grasps/{key}']
                name = g.attrs.get('name', key)
                gp = g['grasp_point'][:]  # TCP 位置
                ad = g['approach_dir'][:] if 'approach_dir' in g else None
                fd = g['finger_dir'][:] if 'finger_dir' in g else None
                w = g.attrs.get('gripper_width', 0.04)
                
                if ad is None or fd is None:
                    print(f"    ⚠️ {name}: 缺少方向数据, 跳过")
                    continue
                
                # 手指中点 = TCP + approach × 10.5cm
                finger_mid = gp + ad * 0.105
                
                # Raycast: 从中点沿 ±finger_dir 射出, 找表面交点
                c1, c2 = None, None
                
                # 方向1: +finger_dir
                locs1, _, _ = mesh.ray.intersects_location(
                    ray_origins=[finger_mid],
                    ray_directions=[fd]
                )
                if len(locs1) > 0:
                    dists = np.linalg.norm(locs1 - finger_mid, axis=1)
                    c1 = locs1[np.argmin(dists)]
                
                # 方向2: -finger_dir
                locs2, _, _ = mesh.ray.intersects_location(
                    ray_origins=[finger_mid],
                    ray_directions=[-fd]
                )
                if len(locs2) > 0:
                    dists = np.linalg.norm(locs2 - finger_mid, axis=1)
                    c2 = locs2[np.argmin(dists)]
                
                if c1 is not None and c2 is not None:
                    print(f"    🤖 {name}: w={w*100:.1f}cm ✅ 两个接触点")
                elif c1 is not None or c2 is not None:
                    print(f"    🤖 {name}: w={w*100:.1f}cm ⚠️ 只找到1个接触点")
                else:
                    print(f"    🤖 {name}: w={w*100:.1f}cm ❌ 无接触点")
                
                grasp_data.append({'mid': finger_mid, 'c1': c1, 'c2': c2, 'name': name})
        
        if grasp_data:
            break  # 优先使用第一个有数据的 GT 源
    
    if not grasp_data:
        print(f"⚠️ {obj_id} 没有成功的 Robot GT"); return
    
    print(f"\n打开 Open3D ({len(grasp_data)} 个GT)...")
    
    geometries = []
    
    # 物体线框
    o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)
    o3d_mesh.compute_vertex_normals()
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)
    wireframe.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(wireframe)
    
    for gd in grasp_data:
        mid = gd['mid']
        # 蓝色球: 抓取中点
        sp = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        sp.translate(mid)
        sp.paint_uniform_color([0.0, 0.3, 1.0])
        sp.compute_vertex_normals()
        geometries.append(sp)
        
        c1, c2 = gd['c1'], gd['c2']
        pts_for_lines = [mid]
        
        # 红色球: 表面接触点
        for cp in [c1, c2]:
            if cp is not None:
                s = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
                s.translate(cp)
                s.paint_uniform_color([1.0, 0.0, 0.0])
                s.compute_vertex_normals()
                geometries.append(s)
                pts_for_lines.append(cp)
        
        # 绿线: 中点 ↔ 接触点
        if len(pts_for_lines) > 1:
            pts = np.array(pts_for_lines)
            lines = [[0, i] for i in range(1, len(pts))]
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts)
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(lines))
            geometries.append(ls)
    
    o3d.visualization.draw_geometries(geometries, 
        window_name=f"Robot GT — {obj_id} ({len(grasp_data)} grasps)")


def summary_all():
    """统计所有物体的 Robot GT 状态."""
    all_meshes = sorted(glob.glob(os.path.join(MESH_DIR, '*.obj')))
    
    results = {'success': [], 'fail': [], 'missing': []}
    
    for mp in all_meshes:
        obj_id = os.path.splitext(os.path.basename(mp))[0]
        best_status = 'missing'
        
        for gt_dir in GT_DIRS:
            gt_path = os.path.join(gt_dir, f'{obj_id}_robot_gt.hdf5')
            if os.path.exists(gt_path):
                try:
                    with h5py.File(gt_path, 'r') as f:
                        if f.attrs.get('success', False):
                            best_status = 'success'
                            break
                        else:
                            best_status = 'fail'
                except:
                    pass
        
        results[best_status].append(obj_id)
    
    total = len(all_meshes)
    print("=" * 60)
    print("  Robot GT 汇总")
    print("=" * 60)
    print(f"  ✅ 成功: {len(results['success'])}/{total}")
    print(f"  ❌ 失败: {len(results['fail'])}/{total}")
    print(f"  🆕 缺失: {len(results['missing'])}/{total}")
    print()
    
    if results['success']:
        print("✅ 成功的物体:")
        for i, oid in enumerate(results['success']):
            print(f"  {oid}", end='')
            if (i + 1) % 8 == 0: print()
        print()
    
    if results['fail']:
        print(f"\n❌ 失败的物体 ({len(results['fail'])}):")
        for oid in results['fail']:
            print(f"  {oid}")
    
    if results['missing']:
        print(f"\n🆕 缺失的物体 ({len(results['missing'])}):")
        for oid in results['missing']:
            print(f"  {oid}")


def main():
    parser = argparse.ArgumentParser(description='Robot GT 可视化')
    parser.add_argument('--obj', help='可视化单个物体')
    parser.add_argument('--all', action='store_true', help='汇总所有统计')
    args = parser.parse_args()
    
    if args.all:
        summary_all()
    elif args.obj:
        vis_single(args.obj)
    else:
        print("用法: --all 查看汇总, --obj ID 可视化单个")


if __name__ == '__main__':
    main()
