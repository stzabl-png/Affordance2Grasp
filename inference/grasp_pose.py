#!/usr/bin/env python3
"""
Grasp Pose Generation (Multi-Candidate)
========================================
输入: 物体 mesh (.obj / .ply)
输出: HDF5 文件 (多候选抓取位姿 + affordance 数据)

用法:
    cd /home/lyh/Project/Affordance2Grasp
    python -m inference.grasp_pose --mesh /home/lyh/Project/OakInk/image/obj/A16013.obj
"""

import os
import sys
import argparse
import numpy as np
import trimesh
import h5py
from scipy.spatial.transform import Rotation

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from inference.predictor import AffordancePredictor


# ============================================================
# 抓取位姿计算 (多候选)
# ============================================================

def compute_principal_axis(mesh):
    """PCA 计算物体主轴 (对瓶子=竖直方向)."""
    verts = np.array(mesh.vertices)
    centered = verts - verts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axis = Vt[0]
    if axis[2] < 0:
        axis = -axis
    return axis


def compute_cross_section_width(verts, grasp_pos, principal_axis, approach_dir, slice_thickness=0.01):
    """计算物体在指定接近方向上的截面宽度."""
    # 取抓取高度附近的切片
    proj_pa = verts @ principal_axis
    contact_height = grasp_pos @ principal_axis
    height_mask = np.abs(proj_pa - contact_height) < slice_thickness
    if height_mask.sum() < 10:
        height_mask = np.ones(len(verts), dtype=bool)

    slice_verts = verts[height_mask]

    # 在接近方向上投影得到宽度
    open_dir = np.cross(principal_axis, approach_dir)
    open_dir = open_dir / (np.linalg.norm(open_dir) + 1e-8)
    proj = slice_verts @ open_dir
    width = proj.max() - proj.min()
    return float(width)


def clamp_grasp_depth(grasp_point, verts, approach_dir, max_depth=0.035, mesh=None):
    """
    限制抓取点深度: 从接近方向的入口表面算起, 最多 max_depth.

    approach_dir 指向物体内部 (夹爪前进方向).
    使用 ray casting 找 LOCAL 入口表面 (而非全局顶点极值),
    解决有突出部件 (泵头、把手) 时深度估算错误的问题。
    """
    # 方法1: 用 trimesh ray cast 找 LOCAL 入口 (更准确)
    if mesh is not None:
        origin = np.array(grasp_point, dtype=np.float64).reshape(1, 3)
        ray_dir = np.array(-approach_dir, dtype=np.float64).reshape(1, 3)  # 向外/上方

        try:
            locations, index_ray, _ = mesh.ray.intersects_location(
                ray_origins=origin, ray_directions=ray_dir, multiple_hits=True)
            if len(locations) > 0:
                # 找最远的命中点 (从内到外, 最远 = 出口 = 入口表面)
                dists = np.linalg.norm(locations - grasp_point, axis=1)
                entry_pt = locations[np.argmax(dists)]
                depth = np.linalg.norm(grasp_point - entry_pt)
                if depth > max_depth:
                    # 沿接近方向回退到 max_depth 深度
                    grasp_point = entry_pt + approach_dir * max_depth
                return grasp_point
        except Exception:
            pass  # fall through to method 2

    # 方法2: fallback — 全局投影 (对简单形状仍然有效)
    proj_verts = verts @ approach_dir
    proj_gp = grasp_point @ approach_dir
    entry_proj = proj_verts.min()  # 入口表面投影值

    depth = proj_gp - entry_proj
    if depth > max_depth:
        new_proj = entry_proj + max_depth
        delta = new_proj - proj_gp
        grasp_point = grasp_point + delta * approach_dir

    return grasp_point


def correct_to_cross_section_center(grasp_point, verts, approach_dir, finger_open_dir,
                                     slice_thickness=0.01):
    """
    截面中心修正: 沿 finger_open_dir 方向居中。

    在 grasp_point 处取垂直于 approach_dir 的薄切片,
    用切片中心的 finger_open_dir 分量替换 grasp_point 的对应分量。
    保留 approach_dir 方向的深度不变, 第三轴也不变。

    Args:
        grasp_point: (3,)
        verts: (V, 3) mesh 顶点
        approach_dir: (3,) 接近方向 (单位)
        finger_open_dir: (3,) 夹爪指尖连线方向 (单位)
        slice_thickness: 切片半厚度 (m)
    """
    # 沿 approach_dir 投影, 取切片
    proj = verts @ approach_dir
    gp_proj = float(np.dot(grasp_point, approach_dir))
    mask = np.abs(proj - gp_proj) < slice_thickness
    if mask.sum() < 3:
        return grasp_point.copy()

    slice_pts = verts[mask]
    # 沿 finger_open_dir 投影
    finger_proj = slice_pts @ finger_open_dir
    slice_center_finger = float(finger_proj.mean())
    gp_finger = float(np.dot(grasp_point, finger_open_dir))

    # 只修正 finger_open_dir 分量
    corrected = grasp_point.copy()
    corrected += (slice_center_finger - gp_finger) * finger_open_dir
    return corrected


def verify_gripper_closure(grasp_point, finger_open_dir, mesh, max_width=0.08):
    """
    Ray casting 验证夹爪闭合: 从 grasp_point 沿 ±finger_open_dir 发射 ray,
    检查两侧都能碰到物体表面, 且总宽度 < max_width。

    Returns:
        valid: bool
        actual_width: float (两侧接触距离之和, 若无效则 0)
    """
    origins = np.array([grasp_point, grasp_point], dtype=np.float64)
    directions = np.array([finger_open_dir, -finger_open_dir], dtype=np.float64)

    try:
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=True,
        )
    except Exception:
        return False, 0.0

    if len(locations) == 0:
        return False, 0.0

    # 每条 ray 找最近命中点
    best_dist = [np.inf, np.inf]
    for loc, ri in zip(locations, index_ray):
        d = np.linalg.norm(loc - grasp_point)
        if d < best_dist[ri]:
            best_dist[ri] = d

    # 两侧都必须命中
    if best_dist[0] == np.inf or best_dist[1] == np.inf:
        return False, 0.0

    width = best_dist[0] + best_dist[1]
    if width > max_width:
        return False, float(width)

    return True, float(width)


def generate_grasp_candidates(contact_pts, mesh, force_center=None):
    """
    生成多个候选抓取位姿 (世界轴对齐接近方向)。

    接近方向严格对齐世界坐标轴:
      - 右侧: approach = -X  (从右往左)
      - 左侧: approach = +X  (从左往右)
      - 正面: approach = -Y  (从前往后)
      - 上方: approach = -Z  (从上往下)

    指尖连线始终与接触面平行 (不抓棱角):
      - 水平接近 (±X, -Y): 指尖连线沿 Z (竖直方向)
      - 上方接近 (-Z):      指尖连线沿最窄水平方向 (X 或 Y)

    Args:
        contact_pts: (N, 3) 接触点
        mesh: trimesh.Trimesh 物体 mesh
        force_center: (3,) 可选, 受力中心

    Returns:
        candidates: list of dict
    """
    if len(contact_pts) < 3:
        raise ValueError(f"接触点太少: {len(contact_pts)}, 需要至少 3 个")

    FINGER_LENGTH = 0.04       # 夹爪手指长度 4cm
    MAX_GRIPPER_OPEN = 0.08    # 夹爪最大开口 8cm
    MAX_INSERT_DEPTH = 0.035   # 最大插入深度 3.5cm (手指4cm - 安全余量0.5cm)
    # TCP 偏移: panda_hand → 指尖中心 (ee_link), 来自 Franka URDF
    # panda_hand → finger_base = 5.84cm, finger_length ≈ 4.5cm
    # panda_hand → ee_link = 10.34cm
    TCP_OFFSET = 0.1034

    verts = np.array(mesh.vertices)
    obj_center = verts.mean(axis=0)
    principal_axis = compute_principal_axis(mesh)

    # 抓取点 (夹爪指尖中点)
    if force_center is not None:
        grasp_point = np.array(force_center, dtype=np.float64)

        # 边界情况: 受力中心太深
        closest_pt, dist, _ = trimesh.proximity.closest_point(
            mesh, grasp_point.reshape(1, -1))
        surface_depth = float(dist[0])
        if surface_depth > FINGER_LENGTH:
            surface_pt = closest_pt[0]
            inward_dir = grasp_point - surface_pt
            inward_norm = np.linalg.norm(inward_dir)
            if inward_norm > 1e-8:
                inward_dir = inward_dir / inward_norm
                grasp_point = surface_pt + FINGER_LENGTH * inward_dir
    else:
        contact_centroid = contact_pts.mean(axis=0)
        pa_offset = np.dot(contact_centroid - obj_center, principal_axis)
        grasp_point = obj_center + pa_offset * principal_axis

    # 物体高度
    proj_pa = verts @ principal_axis
    obj_height = proj_pa.max() - proj_pa.min()

    candidates = []

    # ---- 水平接近 ----
    horizontal_approaches = [
        # (name, approach_dir, finger_open_dir)
        ("right", np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),  # approach -X, finger along Y
        ("left",  np.array([ 1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),  # approach +X, finger along Y
        ("front", np.array([ 0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])),  # approach +Y, finger along X
    ]

    for name, approach, x_open in horizontal_approaches:
        # x_open = 指尖连线方向 (per-direction)
        y_body = np.cross(approach, x_open)
        y_body = y_body / (np.linalg.norm(y_body) + 1e-8)
        x_open = np.cross(y_body, approach)
        x_open = x_open / (np.linalg.norm(x_open) + 1e-8)

        rot = np.column_stack([x_open, y_body, approach])
        if np.linalg.det(rot) < 0:
            x_open = -x_open
            rot = np.column_stack([x_open, y_body, approach])

        # 截面宽度 (在 grasp_point 处)
        width = compute_cross_section_width(verts, grasp_point, principal_axis, approach)

        # 如果截面 > 8cm, 沿主轴上下搜索更窄的截面
        adjusted_gp = grasp_point.copy()
        if width > MAX_GRIPPER_OPEN:
            proj_vals = verts @ principal_axis
            pa_min, pa_max = proj_vals.min(), proj_vals.max()
            cur_h = np.dot(grasp_point, principal_axis)
            best_w, best_off = width, 0.0
            for off in np.linspace(-0.05, 0.05, 21):  # ±5cm, 每0.5cm
                h = cur_h + off
                if h < pa_min or h > pa_max:
                    continue
                w = compute_cross_section_width(
                    verts, grasp_point + off * principal_axis, principal_axis, approach)
                if w < best_w:
                    best_w, best_off = w, off
            if best_w < width:
                adjusted_gp = grasp_point + best_off * principal_axis
                width = best_w

        # ★ 搜索后仍 > 8cm → 放弃此方向
        if width > MAX_GRIPPER_OPEN:
            continue

        # ★ 深度限制: 手指最多伸进去 3.5cm
        adjusted_gp = clamp_grasp_depth(adjusted_gp, verts, approach, MAX_INSERT_DEPTH, mesh=mesh)

        # ★ 截面中心修正: 沿 finger 方向居中
        adjusted_gp = correct_to_cross_section_center(adjusted_gp, verts, approach, x_open)

        # ★ Ray casting 闭合验证
        closure_ok, ray_width = verify_gripper_closure(adjusted_gp, x_open, mesh, MAX_GRIPPER_OPEN)
        if not closure_ok:
            continue
        # 用 ray casting 的实际宽度 (比截面估算更准确)
        gripper_width = float(np.clip(ray_width + 0.005, 0.01, MAX_GRIPPER_OPEN))

        # panda_hand 位置 = 指尖中点沿接近方向后退 TCP_OFFSET
        panda_hand_pos = adjusted_gp - approach * TCP_OFFSET

        candidates.append({
            "name": f"horizontal_{name}",
            "position": panda_hand_pos.astype(np.float32),  # panda_hand EE 位置
            "rotation": rot.astype(np.float32),
            "gripper_width": gripper_width,
            "approach_type": "horizontal",
            "angle_deg": {"right": 0, "left": 180, "front": 90}.get(name, 0),
            "cross_section_width": float(width),
            "obj_height": float(obj_height),
            "grasp_point": adjusted_gp.astype(np.float32),  # 指尖中点 (可视化用)
        })

    # ---- Top-Down: approach = -Z ----
    z_down = np.array([0.0, 0.0, -1.0])
    z_up = np.array([0.0, 0.0, 1.0])

    # 指尖方向 = 垂直于物体整体最长的水平轴 (用 bounding box 判断)
    # 注意: compute_cross_section_width 返回的是 cross(principal_axis, approach_dir) 方向的宽度
    # 即: approach=[1,0,0] → 测量 Y 方向宽度; approach=[0,1,0] → 测量 X 方向宽度
    span_x = verts[:, 0].max() - verts[:, 0].min()
    span_y = verts[:, 1].max() - verts[:, 1].min()
    if span_x > span_y:
        # X 更长 → 指尖沿 Y (垂直于长轴X) → 测 Y 方向宽度 → approach=[1,0,0]
        x_open_td = np.array([0.0, 1.0, 0.0])
        td_width = compute_cross_section_width(verts, grasp_point, z_up, np.array([1, 0, 0.0]))
    else:
        # Y 更长 → 指尖沿 X (垂直于长轴Y) → 测 X 方向宽度 → approach=[0,1,0]
        x_open_td = np.array([1.0, 0.0, 0.0])
        td_width = compute_cross_section_width(verts, grasp_point, z_up, np.array([0, 1, 0.0]))




    # 如果 > 8cm, 沿 Z 轴上下搜索更窄截面 (自动选最窄方向)
    td_gp = grasp_point.copy()
    if td_width > MAX_GRIPPER_OPEN:
        proj_z = verts[:, 2]
        z_min, z_max = proj_z.min(), proj_z.max()
        best_w, best_z = td_width, grasp_point[2]
        best_dir = x_open_td.copy()
        for z_off in np.linspace(-0.05, 0.05, 21):
            test_z = grasp_point[2] + z_off
            if test_z < z_min or test_z > z_max:
                continue
            test_pt = grasp_point.copy()
            test_pt[2] = test_z
            w1 = compute_cross_section_width(verts, test_pt, z_up, np.array([1, 0, 0.0]))
            w2 = compute_cross_section_width(verts, test_pt, z_up, np.array([0, 1, 0.0]))
            w = min(w1, w2)
            if w < best_w:
                best_w, best_z = w, test_z
                best_dir = np.array([1.0, 0.0, 0.0]) if w1 < w2 else np.array([0.0, 1.0, 0.0])
        if best_w < td_width:
            td_gp = grasp_point.copy()
            td_gp[2] = best_z
            td_width = best_w
            x_open_td = best_dir


    # ★ 搜索后仍 > 8cm → 放弃 top-down
    if td_width <= MAX_GRIPPER_OPEN:
        # ★ 深度限制: 手指最多伸进去 3.5cm
        td_gp = clamp_grasp_depth(td_gp, verts, z_down, MAX_INSERT_DEPTH, mesh=mesh)

        # ★ 截面中心修正: 沿 finger 方向居中
        td_gp = correct_to_cross_section_center(td_gp, verts, z_down, x_open_td)

        # ★ Ray casting 闭合验证
        closure_ok, ray_width = verify_gripper_closure(td_gp, x_open_td, mesh, MAX_GRIPPER_OPEN)
        if not closure_ok:
            pass  # top-down 不因为 ray 失败就跳过, 保留截面宽度
        else:
            td_width = ray_width  # 用实际 ray 宽度

        y_body_td = np.cross(z_down, x_open_td)
        y_body_td = y_body_td / (np.linalg.norm(y_body_td) + 1e-8)
        rot_td = np.column_stack([x_open_td, y_body_td, z_down])
        if np.linalg.det(rot_td) < 0:
            x_open_td = -x_open_td
            rot_td = np.column_stack([x_open_td, y_body_td, z_down])

        td_gw = float(np.clip(td_width + 0.005, 0.01, MAX_GRIPPER_OPEN))

        # panda_hand 位置 = 指尖中点沿接近方向后退 TCP_OFFSET
        td_panda_hand_pos = td_gp - z_down * TCP_OFFSET

        candidates.append({
            "name": "top_down",
            "position": td_panda_hand_pos.astype(np.float32),  # panda_hand EE 位置
            "rotation": rot_td.astype(np.float32),
            "gripper_width": td_gw,
            "approach_type": "top_down",
            "angle_deg": -1,
            "cross_section_width": float(td_width),
            "obj_height": float(obj_height),
            "grasp_point": td_gp.astype(np.float32),  # 指尖中点 (可视化用)
        })

    if len(candidates) == 0:
        raise ValueError(f"物体所有方向截面都 > {MAX_GRIPPER_OPEN*100:.0f}cm, 无法抓取")

    return candidates




def score_candidates(candidates):
    """
    对候选打分排序。

    打分规则:
    1. 宽度可抓 (40分): 截面宽度 < 8cm
    2. Robot-Facing (30分): 接近方向朝 +Y (机器人从正面接近物体)
    3. 高度自适应 (20分): 矮物体优先 top-down
    4. 方向稳定 (10分): front 和 right 方向更优
    """
    scored = []
    for c in candidates:
        score = 0.0
        width = c["cross_section_width"]
        obj_h = c["obj_height"]
        approach = c["rotation"][:, 2]  # z 列 = 接近方向

        # 1. 宽度可抓 (40分)
        if width < 0.06:
            score += 40
        elif width < 0.08:
            score += 40 * (0.08 - width) / 0.02
        else:
            score += 0  # 太宽, 夹不住

        # 2. Robot-Facing (30分)
        # 机器人从 +Y 方向面对物体, approach = +Y = 最佳
        robot_approach = np.array([0, 1, 0])  # 正面接近 = +Y
        cos_robot = np.dot(approach, robot_approach)
        score += 30 * max(0, (cos_robot + 1) / 2)  # [-1,1] → [0,1]

        # 3. 高度自适应 (20分)
        if c["approach_type"] == "top_down":
            if obj_h < 0.05:
                score += 20
            elif obj_h < 0.08:
                score += 10
            else:
                score += 5
        else:
            if obj_h > 0.08:
                score += 20
            elif obj_h > 0.05:
                score += 15
            else:
                score += 5

        # 4. 方向稳定 (10分)
        name = c.get("name", "")
        if "front" in name:
            score += 10  # 正面接近最稳定
        elif "right" in name or "left" in name:
            score += 7   # 左右次之
        else:
            score += 3   # top_down

        c["score"] = round(score, 1)
        scored.append(c)

    # 按分数排序
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored



# ============================================================
# 兼容: 保留旧接口
# ============================================================

def compute_grasp_pose(contact_pts, mesh, threshold=0.5):
    """兼容旧接口: 返回最优候选的位姿。"""
    candidates = generate_grasp_candidates(contact_pts, mesh)
    scored = score_candidates(candidates)
    best = scored[0]
    return best["position"], best["rotation"], best["gripper_width"]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Grasp Pose Generation (Multi-Candidate)")
    parser.add_argument("--mesh", type=str, required=True, help="物体 mesh 路径 (.obj/.ply)")
    parser.add_argument("--num_points", type=int, default=config.NUM_POINTS)
    parser.add_argument("--threshold", type=float, default=config.AFFORDANCE_THRESHOLD,
                        help="Affordance 接触阈值")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None, help="输出 HDF5 路径 (默认自动)")
    args = parser.parse_args()

    config.ensure_dirs()

    obj_name = os.path.splitext(os.path.basename(args.mesh))[0]

    print("=" * 60)
    print("Affordance2Grasp — Grasp Pose Generation (Multi-Candidate)")
    print("=" * 60)
    print(f"  Mesh:      {args.mesh}")
    print(f"  Points:    {args.num_points}")
    print(f"  Threshold: {args.threshold}")
    print()

    # ---- Step 1: 加载 Mesh ----
    mesh = trimesh.load(args.mesh, force='mesh')
    print(f"  [1/4] Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    verts = np.array(mesh.vertices)
    span = verts.max(axis=0) - verts.min(axis=0)
    print(f"         Span: X={span[0]*100:.1f}cm, Y={span[1]*100:.1f}cm, Z={span[2]*100:.1f}cm")

    # ---- Step 2: Affordance 预测 ----
    print(f"\n  [2/4] Running Affordance prediction...")
    predictor = AffordancePredictor(device=args.device, num_points=args.num_points)
    points, normals, contact_prob, force_center = predictor.predict(args.mesh, args.num_points)
    if force_center is not None:
        print(f"         Force center (predicted): [{force_center[0]:.4f}, {force_center[1]:.4f}, {force_center[2]:.4f}]")

    contact_mask = contact_prob > args.threshold
    n_contact = contact_mask.sum()
    print(f"         Contact points: {n_contact}/{args.num_points} (threshold={args.threshold})")
    print(f"         Prob range: [{contact_prob.min():.3f}, {contact_prob.max():.3f}]")

    if n_contact < 5:
        print(f"\n  ⚠️ 接触点太少 ({n_contact} < 5), 降低阈值...")
        args.threshold = max(0.1, contact_prob.mean())
        contact_mask = contact_prob > args.threshold
        n_contact = contact_mask.sum()
        print(f"         新阈值: {args.threshold:.3f}, 接触点: {n_contact}")

    contact_pts = points[contact_mask]

    # ---- Step 3: 生成多候选抓取位姿 ----
    print(f"\n  [3/4] Generating grasp candidates from {n_contact} contact points...")
    candidates = generate_grasp_candidates(contact_pts, mesh, force_center=force_center)
    scored = score_candidates(candidates)

    print(f"\n  📊 Candidates (sorted by score):")
    for i, c in enumerate(scored):
        euler = Rotation.from_matrix(c["rotation"]).as_euler('xyz', degrees=True)
        marker = "⭐" if i == 0 else "  "
        print(f"  {marker} [{i+1}] {c['name']:>16s}  score={c['score']:5.1f}  "
              f"width={c['cross_section_width']*100:.1f}cm  "
              f"gripper={c['gripper_width']*100:.1f}cm  "
              f"euler=[{euler[0]:.0f}°,{euler[1]:.0f}°,{euler[2]:.0f}°]")

    best = scored[0]
    print(f"\n  ✅ Best: {best['name']} (score={best['score']})")

    # ---- Step 4: 保存 HDF5 ----
    h5_path = args.output or os.path.join(config.GRASPS_DIR, f"{obj_name}_grasp.hdf5")
    with h5py.File(h5_path, 'w') as f:
        # 保持兼容: grasp/ 存最优候选
        g = f.create_group("grasp")
        g.create_dataset("position", data=best["position"])  # panda_hand EE 位置
        g.create_dataset("grasp_point", data=best["grasp_point"])  # 指尖中点
        g.create_dataset("rotation", data=best["rotation"])
        quat_xyzw = Rotation.from_matrix(best["rotation"]).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        euler_best = Rotation.from_matrix(best["rotation"]).as_euler('xyz', degrees=True)
        g.create_dataset("quaternion_wxyz", data=quat_wxyz)
        g.create_dataset("euler_deg", data=euler_best.astype(np.float32))
        g.attrs["gripper_width"] = best["gripper_width"]
        g.attrs["approach_type"] = best["approach_type"]
        g.attrs["candidate_name"] = best["name"]
        g.attrs["score"] = best["score"]

        # 保存所有候选
        cg = f.create_group("candidates")
        cg.attrs["n_candidates"] = len(scored)
        for i, c in enumerate(scored):
            ci = cg.create_group(f"candidate_{i}")
            ci.create_dataset("position", data=c["position"])  # panda_hand EE 位置
            ci.create_dataset("grasp_point", data=c["grasp_point"])  # 指尖中点
            ci.create_dataset("rotation", data=c["rotation"])
            ci.attrs["name"] = c["name"]
            ci.attrs["score"] = c["score"]
            ci.attrs["gripper_width"] = c["gripper_width"]
            ci.attrs["approach_type"] = c["approach_type"]
            ci.attrs["cross_section_width"] = c["cross_section_width"]
            ci.attrs["obj_height"] = c["obj_height"]

        # Affordance 数据
        a = f.create_group("affordance")
        a.create_dataset("points", data=points, compression="gzip")
        a.create_dataset("normals", data=normals, compression="gzip")
        a.create_dataset("contact_prob", data=contact_prob.astype(np.float32))
        a.create_dataset("contact_mask", data=contact_mask.astype(np.uint8))
        if force_center is not None:
            a.create_dataset("force_center", data=force_center.astype(np.float32))
        a.attrs["threshold"] = args.threshold
        a.attrs["n_contact"] = int(n_contact)


        # Metadata
        m = f.create_group("metadata")
        m.attrs["obj_id"] = obj_name
        m.attrs["mesh_path"] = os.path.abspath(args.mesh)
        m.attrs["num_points"] = args.num_points
        m.attrs["coordinate_system"] = "OBJ_local"
        m.attrs["version"] = "2.0_multi_candidate"

    print(f"\n  [4/4] ✅ Saved: {h5_path}")
    print(f"         {len(scored)} candidates stored")
    print(f"\n{'=' * 60}")
    print(f"  下一步: sim45 Pipeline/run_grasp_sim.py --hdf5 {h5_path}")
    print(f"{'=' * 60}")

    return h5_path


if __name__ == "__main__":
    main()
