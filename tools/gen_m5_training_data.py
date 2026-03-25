#!/usr/bin/env python3
"""
生成 M5 训练数据: 合并 Human Prior + Robot GT → per-point affordance HDF5.

每个 HDF5 包含:
    point_cloud:  (N, 3) — 表面采样点 xyz
    normals:      (N, 3) — 法向量
    human_prior:  (N,)   — 人类抓取接触标签 (0/1)
    robot_gt:     (N,)   — 机器人抓取连续 affordance (0~1)
    force_center: (N, 3) — 力中心位置

用法:
    cd /home/lyh/Project/Affordance2Grasp
    python3 tools/gen_m5_training_data.py
"""
import os, sys, glob, h5py, numpy as np, trimesh

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(PROJ, 'data_hub', 'meshes', 'v1')
HP_DIR = os.path.join(PROJ, 'data_hub', 'human_prior')
GT_DIRS = [
    os.path.join(PROJ, 'output', 'robot_gt_verified'),
    os.path.join(PROJ, 'sim', 'output', 'robot_gt'),    # 新手动标注 GT
    os.path.join(PROJ, 'output', 'robot_gt'),
    os.path.join(PROJ, 'output', 'robot_gt_random'),
]
OUT_DIR = os.path.join(PROJ, 'data_hub', 'training_m5')
os.makedirs(OUT_DIR, exist_ok=True)

N_POINTS = 4096
CONTACT_SIGMA = 0.025  # 高斯平滑 sigma (25mm, 扩大标注覆盖)


def load_human_prior(obj_id):
    """加载 Human Prior: point_cloud, normals, human_prior."""
    for name in [f'{obj_id}.hdf5', f'grab_{obj_id}.hdf5']:
        path = os.path.join(HP_DIR, name)
        if os.path.exists(path):
            with h5py.File(path, 'r') as f:
                pc = f['point_cloud'][()]
                nrm = f['normals'][()]
                hp = f['human_prior'][()]
                return pc.astype(np.float32), nrm.astype(np.float32), hp.astype(np.float32)
    return None, None, None


def load_robot_gt_grasps(obj_id):
    """加载 verified Robot GT 抓取参数."""
    grasps = []
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
                gp = g['grasp_point'][:]
                ad = g['approach_dir'][:] if 'approach_dir' in g else None
                fd = g['finger_dir'][:] if 'finger_dir' in g else None
                w = g.attrs.get('gripper_width', 0.04)
                if ad is not None and fd is not None:
                    grasps.append({'gp': gp, 'ad': ad, 'fd': fd, 'w': w})
        if grasps:
            break  # 优先使用第一个有数据的目录
    return grasps


def compute_robot_gt_label(mesh, points, grasps):
    """用 raycast 找物体表面接触点, 计算连续 affordance."""
    labels = np.zeros(len(points), dtype=np.float32)
    force_center_acc = np.zeros(3, dtype=np.float64)
    n_contacts = 0

    for g in grasps:
        finger_mid = g['gp'] + g['ad'] * 0.105
        y_dir = np.cross(g['ad'], g['fd'])
        y_dir = y_dir / (np.linalg.norm(y_dir) + 1e-8)

        # 用柱状切片法 (同 aggregate_robot_gt.py 的方法)
        offsets = points - finger_mid
        proj_approach = np.dot(offsets, g['ad'])
        proj_y = np.dot(offsets, y_dir)
        proj_finger = np.dot(offsets, g['fd'])

        mask = (np.abs(proj_approach) < 0.025) & (np.abs(proj_y) < 0.015)
        if np.sum(mask) < 2:
            mask = (np.abs(proj_approach) < 0.04) & (np.abs(proj_y) < 0.03)

        if np.sum(mask) < 2:
            # 退回 raycast 方法
            contact_pts = []
            for direction in [g['fd'], -g['fd']]:
                locs, _, _ = mesh.ray.intersects_location(
                    ray_origins=[finger_mid], ray_directions=[direction])
                if len(locs) > 0:
                    dists = np.linalg.norm(locs - finger_mid, axis=1)
                    contact_pts.append(locs[np.argmin(dists)])
            if not contact_pts:
                continue
            contact_pts = np.array(contact_pts)
        else:
            # 用柱状切片内的最大/最小点作为接触中心
            f_max = np.max(proj_finger[mask])
            f_min = np.min(proj_finger[mask])
            c1 = finger_mid + f_max * g['fd']
            c2 = finger_mid + f_min * g['fd']
            contact_pts = np.array([c1, c2])
            force_center_acc += finger_mid + ((f_max + f_min) / 2.0) * g['fd']
            n_contacts += 1

        # 高斯衰减
        for cp in contact_pts:
            dists = np.linalg.norm(points - cp, axis=1)
            local_score = np.exp(-(dists ** 2) / (2 * CONTACT_SIGMA ** 2))
            labels = np.maximum(labels, local_score)

    force_center = (force_center_acc / max(n_contacts, 1)).astype(np.float32)
    return labels, force_center


def process_object(obj_id, mesh_path):
    """处理单个物体."""
    mesh = trimesh.load(mesh_path, force='mesh')

    # 优先使用 Human Prior 的点云
    hp_pc, hp_nrm, hp_labels = load_human_prior(obj_id)
    if hp_pc is not None:
        pc = hp_pc
        normals = hp_nrm
        human_prior = hp_labels
        # 如果点数不是 N_POINTS，随机下采样 / 上采样
        if len(pc) != N_POINTS:
            if len(pc) > N_POINTS:
                idx = np.random.choice(len(pc), N_POINTS, replace=False)
            else:
                idx = np.random.choice(len(pc), N_POINTS, replace=True)
            pc = pc[idx]
            normals = normals[idx]
            human_prior = human_prior[idx]
        has_hp = True
    else:
        # 没有 HP 时从 mesh 采样
        pc, face_idx = trimesh.sample.sample_surface(mesh, N_POINTS)
        pc = pc.astype(np.float32)
        normals = mesh.face_normals[face_idx].astype(np.float32)
        human_prior = np.zeros(N_POINTS, dtype=np.float32)
        has_hp = False

    # Robot GT
    grasps = load_robot_gt_grasps(obj_id)
    if grasps:
        robot_gt, force_center = compute_robot_gt_label(mesh, pc, grasps)
        has_rgt = np.any(robot_gt > 0.01)
    else:
        robot_gt = np.zeros(N_POINTS, dtype=np.float32)
        force_center = np.zeros(3, dtype=np.float32)
        has_rgt = False

    return pc, normals, human_prior, robot_gt, force_center, has_hp, has_rgt, len(grasps)


def main():
    all_meshes = sorted(glob.glob(os.path.join(MESH_DIR, '*.obj')))
    skip = {'body', 'cubelarge', 'eyeglasses', 'teapot'}

    total = 0
    has_hp = 0
    has_rgt = 0
    has_both = 0

    for mp in all_meshes:
        obj_id = os.path.splitext(os.path.basename(mp))[0]
        if obj_id in skip:
            continue

        pc, normals, hp, rgt, fc, hp_ok, rgt_ok, n_grasps = process_object(obj_id, mp)

        # 至少需要一个数据源
        if not hp_ok and not rgt_ok:
            continue

        out_path = os.path.join(OUT_DIR, f'{obj_id}.hdf5')
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('point_cloud', data=pc, compression='gzip')
            f.create_dataset('normals', data=normals, compression='gzip')
            f.create_dataset('human_prior', data=hp, compression='gzip')
            f.create_dataset('robot_gt', data=rgt, compression='gzip')
            f.create_dataset('force_center', data=fc, compression='gzip')

        flag = ''
        if hp_ok: flag += '🧑 '
        if rgt_ok: flag += f'🤖({n_grasps}g) '
        print(f"  ✅ {obj_id}: {flag}")

        total += 1
        if hp_ok: has_hp += 1
        if rgt_ok: has_rgt += 1
        if hp_ok and rgt_ok: has_both += 1

    print(f"\n{'='*50}")
    print(f"  完成! {total} 个训练样本")
    print(f"  🧑 Human Prior: {has_hp}")
    print(f"  🤖 Robot GT:    {has_rgt}")
    print(f"  🧑+🤖 两者:    {has_both}")
    print(f"  输出: {OUT_DIR}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
