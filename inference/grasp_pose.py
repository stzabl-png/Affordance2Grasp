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
    Human-Guided Sampling (PCA + Jitter).

    1. PCA 分析接触点 → 手指张开方向
    2. 叉乘物体主轴 → 接近方向
    3. 角度抖动 ±30° 增加多样性
    4. Top-down 兜底

    Args:
        contact_pts: (N, 3) 接触点
        mesh: trimesh.Trimesh 物体 mesh
        force_center: (3,) 可选, 受力中心

    Returns:
        candidates: list of dict
    """
    if len(contact_pts) < 3:
        raise ValueError(f"接触点太少: {len(contact_pts)}, 需要至少 3 个")

    FINGER_LENGTH = 0.04
    MAX_GRIPPER_OPEN = 0.08
    MAX_INSERT_DEPTH = 0.035
    TCP_OFFSET = 0.105

    verts = np.array(mesh.vertices)
    obj_center = verts.mean(axis=0)
    principal_axis = compute_principal_axis(mesh)

    # 抓取点
    if force_center is not None:
        grasp_point = np.array(force_center, dtype=np.float64)
        closest_pt, dist, _ = trimesh.proximity.closest_point(
            mesh, grasp_point.reshape(1, -1))
        if float(dist[0]) > FINGER_LENGTH:
            surface_pt = closest_pt[0]
            inward_dir = grasp_point - surface_pt
            inward_norm = np.linalg.norm(inward_dir)
            if inward_norm > 1e-8:
                grasp_point = surface_pt + FINGER_LENGTH * (inward_dir / inward_norm)
    else:
        contact_centroid = contact_pts.mean(axis=0)
        pa_offset = np.dot(contact_centroid - obj_center, principal_axis)
        grasp_point = obj_center + pa_offset * principal_axis

    proj_pa = verts @ principal_axis
    obj_height = proj_pa.max() - proj_pa.min()

    # PCA on contact points
    centered_contacts = contact_pts - contact_pts.mean(axis=0)
    _, _, Vt_c = np.linalg.svd(centered_contacts, full_matrices=False)
    contact_spread_dir = Vt_c[0]
    contact_spread_dir /= (np.linalg.norm(contact_spread_dir) + 1e-8)

    base_approach = np.cross(principal_axis, contact_spread_dir)
    base_approach /= (np.linalg.norm(base_approach) + 1e-8)

    BASE_DIRS = [
        ("dynamic_front", base_approach),
        ("dynamic_back", -base_approach),
        ("top_down", -principal_axis)
    ]

    JITTER_DEGREES = [-30, -15, 0, 15, 30]
    generated_directions = []

    for name, b_dir in BASE_DIRS:
        if name == "top_down":
            generated_directions.append(("top_down", b_dir))
            for rot_deg in [-15, 15]:
                rot_rad = np.radians(rot_deg)
                idx_axis = np.argmin(np.abs(b_dir))
                axis = np.zeros(3); axis[idx_axis] = 1.0
                rot_vec = Rotation.from_rotvec(rot_rad * axis)
                generated_directions.append((f"top_down_{rot_deg}", rot_vec.apply(b_dir)))
        else:
            for yaw_deg in JITTER_DEGREES:
                yaw_rad = np.radians(yaw_deg)
                rot_yaw = Rotation.from_rotvec(yaw_rad * principal_axis)
                jittered_dir = rot_yaw.apply(b_dir)
                jittered_dir /= (np.linalg.norm(jittered_dir) + 1e-8)
                generated_directions.append((f"{name}_y{yaw_deg}", jittered_dir))

    print(f"         ✅ 动态聚类生成了 {len(generated_directions)} 个具有人体先验的探测方向 (带 Jitter)")

    candidates = []

    def _find_best_finger_dir(approach, gp):
        if abs(approach[2]) < 0.9:
            up = np.array([0, 0, 1.0])
        else:
            up = np.array([1, 0, 0.0])
        u = np.cross(approach, up)
        u = u / (np.linalg.norm(u) + 1e-8)
        v = np.cross(approach, u)
        v = v / (np.linalg.norm(v) + 1e-8)
        best_width = 1e10
        best_dir = u.copy()
        for angle_deg in range(0, 180, 15):
            angle = np.radians(angle_deg)
            finger_dir = np.cos(angle) * u + np.sin(angle) * v
            finger_dir = finger_dir / (np.linalg.norm(finger_dir) + 1e-8)
            offsets = (verts - gp) @ finger_dir
            w = offsets.max() - offsets.min()
            if w < best_width:
                best_width = w
                best_dir = finger_dir.copy()
        return best_dir, best_width

    for name, approach in generated_directions:
        approach = np.array(approach, dtype=np.float64)
        is_topdown = "top_down" in name
        approach_type = "top_down" if is_topdown else "horizontal"

        x_open, width = _find_best_finger_dir(approach, grasp_point)

        y_body = np.cross(approach, x_open)
        y_body = y_body / (np.linalg.norm(y_body) + 1e-8)
        x_open = np.cross(y_body, approach)
        x_open = x_open / (np.linalg.norm(x_open) + 1e-8)
        rot = np.column_stack([x_open, y_body, approach])
        if np.linalg.det(rot) < 0:
            x_open = -x_open
            rot = np.column_stack([x_open, y_body, approach])

        adjusted_gp = grasp_point.copy()
        if width > MAX_GRIPPER_OPEN:
            proj_vals = verts @ principal_axis
            pa_min, pa_max = proj_vals.min(), proj_vals.max()
            cur_h = np.dot(grasp_point, principal_axis)
            best_w, best_off = width, 0.0
            for off in np.linspace(-0.05, 0.05, 11):
                h = cur_h + off
                if h < pa_min or h > pa_max:
                    continue
                test_pt = grasp_point + off * principal_axis
                d, w = _find_best_finger_dir(approach, test_pt)
                if w < best_w:
                    best_w, best_off = w, off
            if best_w < width:
                adjusted_gp = grasp_point + best_off * principal_axis
                width = best_w

        if width > MAX_GRIPPER_OPEN:
            continue

        adjusted_gp = clamp_grasp_depth(adjusted_gp, verts, approach, MAX_INSERT_DEPTH, mesh=mesh)
        adjusted_gp = correct_to_cross_section_center(adjusted_gp, verts, approach, x_open)

        closure_ok, ray_width = verify_gripper_closure(adjusted_gp, x_open, mesh, MAX_GRIPPER_OPEN)
        if not closure_ok:
            continue
        width = ray_width
        gripper_width = float(np.clip(ray_width + 0.005, 0.01, MAX_GRIPPER_OPEN))

        panda_hand_pos = adjusted_gp - approach * TCP_OFFSET

        candidates.append({
            "name": name,
            "position": panda_hand_pos.astype(np.float32),
            "rotation": rot.astype(np.float32),
            "gripper_width": gripper_width,
            "approach_type": approach_type,
            "angle_deg": int(np.degrees(np.arctan2(approach[0], approach[1]))),
            "cross_section_width": float(width),
            "obj_height": float(obj_height),
            "grasp_point": adjusted_gp.astype(np.float32),
        })

    print(f"         ✅ 有效候选数量: {len(candidates)}")
    if len(candidates) == 0:
        raise ValueError(f"物体所有方向截面都 > {MAX_GRIPPER_OPEN*100:.0f}cm, 无法抓取")

    return candidates




def score_candidates(candidates):
    """
    对候选打分排序 (纯几何学, 不依赖方向名称)。

    打分规则:
    1. 宽度可抓 (40分): 截面宽度 < 8cm, 越窄越好
    2. 接近可达 (30分): 接近方向与 +Y 对齐 (机器人正面) 得分最高
    3. 高度间距 (20分): panda_hand 越高于桌面越好 (避免撞桌面)
    4. 抓取稳定 (10分): 接近方向与物体表面越垂直越好 (法线方向)
    """
    scored = []
    for c in candidates:
        score = 0.0
        width = c["cross_section_width"]
        approach = c["rotation"][:, 2]  # z 列 = 接近方向

        # 1. 宽度可抓 (40分): 越窄越好
        if width < 0.06:
            score += 40
        elif width < 0.08:
            score += 40 * (0.08 - width) / 0.02
        else:
            score += 0

        # 2. 接近可达 (30分): 偏好 +Y (正面), 其次水平, 最后纯垂直
        robot_approach = np.array([0.0, 1.0, 0.0])
        cos_robot = np.dot(approach, robot_approach)
        score += 30 * max(0, (cos_robot + 1) / 2)

        # 3. 高度间距 (20分): 接近方向的 Z 分量越正 (从上往下/水平), 越安全
        # approach_z ≈ 0 → 水平 (好), approach_z ≈ -1 → top-down (也好)
        # approach_z ≈ +1 → bottom-up (差, 撞桌面)
        if approach[2] < 0:  # 有向下分量 (top-down 类)
            score += 15
        elif abs(approach[2]) < 0.3:  # 近似水平
            score += 20
        else:  # 向上分量大 → 可能撞桌面
            score += 5

        # 4. 抓取稳定 (10分): 偏好 top-down 或纯水平 (而非倾斜)
        # 接近方向与水平面/竖直的对齐度
        horiz_component = np.sqrt(approach[0]**2 + approach[1]**2)
        if horiz_component > 0.9 or abs(approach[2]) > 0.9:
            score += 10  # 纯水平或纯竖直
        else:
            score += 5   # 倾斜方向

        c["score"] = round(score, 1)
        scored.append(c)

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
