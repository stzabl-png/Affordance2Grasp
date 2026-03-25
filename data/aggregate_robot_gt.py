import os
import glob
import h5py
import numpy as np

ROBOT_GT_DIR = "output/robot_gt"
HUMAN_PRIOR_DIR = "data_hub/human_prior"
TRAINING_OUT_DIR = "data_hub/training"

os.makedirs(TRAINING_OUT_DIR, exist_ok=True)

def find_human_prior_file(obj_id):
    p1 = os.path.join(HUMAN_PRIOR_DIR, f"{obj_id}.hdf5")
    if os.path.exists(p1): return p1
    p2 = os.path.join(HUMAN_PRIOR_DIR, f"grab_{obj_id}.hdf5")
    if os.path.exists(p2): return p2
    return None

def compute_robot_gt_affordance(point_cloud, panda_hand_pos, approach_dir, finger_dir):
    """
    已知成功抓取的机械臂 TCP 位置 (panda_hand_pos)、接近方向 (approach_dir) 和手指张开方向 (finger_dir)，
    在物体点云上找到两个接触面的中心，然后通过高斯平滑生成连续 affordance。
    """
    # 模拟时的 TCP_OFFSET 为 0.105，真实的指尖中心在这里：
    grasp_point = panda_hand_pos + approach_dir * 0.105

    # 计算正交坐标系的第三个轴
    y_dir = np.cross(approach_dir, finger_dir)
    y_dir = y_dir / (np.linalg.norm(y_dir) + 1e-8)

    # 把点云投影到夹爪局部坐标系
    offsets = point_cloud - grasp_point
    proj_approach = np.dot(offsets, approach_dir)
    proj_y = np.dot(offsets, y_dir)
    proj_finger = np.dot(offsets, finger_dir)

    # 选取夹爪指尖所在的一个 柱状/盒状 区域
    # 指尖深度（沿接近方向）约 2~3cm
    # 指尖宽度（沿侧向）约 2cm
    mask = (np.abs(proj_approach) < 0.025) & (np.abs(proj_y) < 0.015)
    
    if np.sum(mask) < 2:
        # 如果选不出来点（比如太薄了 / 预估有偏差），放宽条件
        mask = (np.abs(proj_approach) < 0.04) & (np.abs(proj_y) < 0.03)
        
    if np.sum(mask) < 2:
        return None, None # 物点太少，失败

    # 在该柱状切片内，沿着手指张开方向的最外侧即为接触点
    f_max = np.max(proj_finger[mask])
    f_min = np.min(proj_finger[mask])

    c1 = grasp_point + f_max * finger_dir
    c2 = grasp_point + f_min * finger_dir

    # 计算所有点到两个接触中心的最小距离
    dist_c1 = np.linalg.norm(point_cloud - c1, axis=1)
    dist_c2 = np.linalg.norm(point_cloud - c2, axis=1)
    min_dist = np.minimum(dist_c1, dist_c2)

    # 高斯平滑 (sigma = 10mm)
    sigma = 0.01
    affordance = np.exp(-(min_dist**2) / (2 * sigma**2))

    # 更新对称后的真实受力中心
    real_force_center = grasp_point + ((f_max + f_min) / 2.0) * finger_dir

    return affordance, real_force_center

def main():
    robot_files = sorted(glob.glob(os.path.join(ROBOT_GT_DIR, "*_robot_gt.hdf5")))
    print(f"找到 {len(robot_files)} 个 Robot GT 文件")

    success_count = 0
    missing_prior = 0
    processed = 0

    for rf in robot_files:
        obj_id = os.path.basename(rf).replace("_robot_gt.hdf5", "")
        
        try:
            with h5py.File(rf, 'r') as f_robot:
                if not f_robot.attrs.get('success', False):
                    continue
                w_cand = f_robot['winning_candidate']
                grasp_point = w_cand['grasp_point'][()]
                approach_dir = w_cand['approach_dir'][()]
                finger_dir = w_cand['finger_dir'][()]
        except Exception as e:
            print(f"[{obj_id}] 读取失败: {e}")
            continue

        success_count += 1

        prior_file = find_human_prior_file(obj_id)
        if prior_file is None:
            missing_prior += 1
            print(f"[{obj_id}] 找不到对应的 Human Prior: {obj_id}")
            continue

        # 读取 Human Prior
        with h5py.File(prior_file, 'r') as f_prior:
            point_cloud = f_prior['point_cloud'][()]
            normals = f_prior['normals'][()]
            human_prior = f_prior['human_prior'][()]
            
        robot_gt, force_center = compute_robot_gt_affordance(point_cloud, grasp_point, approach_dir, finger_dir)
        if robot_gt is None:
             print(f"[{obj_id}] 无法在点云上找到接触点")
             continue

        # 保存 训练集 HDF5
        out_file = os.path.join(TRAINING_OUT_DIR, f"{obj_id}.hdf5")
        with h5py.File(out_file, 'w') as f_out:
            f_out.create_dataset('point_cloud', data=point_cloud, dtype=np.float32)
            f_out.create_dataset('normals', data=normals, dtype=np.float32)
            # human_prior 可能本来是 bool 或 float，统一为 float32
            f_out.create_dataset('human_prior', data=human_prior.astype(np.float32))
            f_out.create_dataset('robot_gt', data=robot_gt.astype(np.float32))
            f_out.create_dataset('force_center', data=force_center.astype(np.float32))
        
        processed += 1
        print(f"✅ [{obj_id}] -> {out_file}")

    print("============================================================")
    print(f" 总 Robot GT 成功数: {success_count}")
    print(f" 找不到 Human Prior: {missing_prior}")
    print(f" 最终训练集生成数:   {processed}")
    print("============================================================")

if __name__ == "__main__":
    main()
