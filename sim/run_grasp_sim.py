#!/usr/bin/env python3
"""
Pipeline Stage B: Isaac Sim 抓取执行
======================================
输入: Stage A 生成的 HDF5 (抓取位姿)
输出: Isaac Sim 中 Franka 执行无碰撞抓取

运行:
    cd /home/lyh/Project/mano2gripper
    sim45 Pipeline/run_grasp_sim.py --hdf5 Pipeline/output/A16013_grasp.hdf5
    sim45 Pipeline/run_grasp_sim.py --hdf5 Pipeline/output/A16013_grasp.hdf5 --headless

Pipeline:
    HDF5 → 搭建场景 → 坐标变换 → cuRobo 规划 → Franka 执行抓取
"""
from isaacsim import SimulationApp
import argparse
import os
import sys

# Parse args before SimulationApp (因为 SimulationApp 修改了 sys.argv)
parser = argparse.ArgumentParser(description="Pipeline Stage B: Isaac Sim Grasp Execution")
parser.add_argument("--hdf5", type=str, required=True, help="Stage A 输出的 HDF5 路径")
parser.add_argument("--headless", action="store_true", help="无头模式")
parser.add_argument("--object_scale", type=float, default=1.0, help="物体缩放 (默认 1.0)")
parser.add_argument("--save-result", action="store_true", default=True,
                    help="保存 Robot GT 结果到 HDF5 (默认开启)")
parser.add_argument("--result-dir", type=str, default=None,
                    help="结果保存目录 (默认 Pipeline/output/robot_gt/)")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import h5py
import torch
from termcolor import cprint
from scipy.spatial.transform import Rotation

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
import omni.replicator.core as rep

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
from env_config.robot.Franka import Franka
from env_config.room.Real_Ground import Real_Ground
from env_config.rigid.RigidObject import RigidObject


# ============================================================
# Scene Config (和 Replay_Grasp / Scene_Env 一致)
# ============================================================
ROBOT_POSITION = [0.2, -0.05, 0.8]
ROBOT_ORIENTATION = [0.0, 0.0, 90.0]
TABLE_POSITION = [0.0, 1.0, 0.75]
TABLE_ORIENTATION = [0.0, 0.0, 0.0]
TABLE_SCALE = [2.0, 2.0, 0.1]
TABLE_TOP_Z = 0.80

OBJECT_POSITION = [0.0, 0.55, TABLE_TOP_Z]
OBJECT_ORIENTATION = [0.0, 0.0, 0.0]

APPROACH_HEIGHT = 0.15  # 物体上方 15cm
LIFT_HEIGHT = 0.15      # 提起 15cm (Franka workspace 限制)


# ============================================================
# Scene Setup
# ============================================================
def setup_scene(obj_id, object_scale):
    """搭建仿真场景."""
    world = World(backend="numpy")

    physics = world.get_physics_context()
    physics.enable_ccd(True)
    physics.enable_gpu_dynamics(True)
    physics.set_broadphase_type("gpu")
    physics.enable_stablization(True)
    physics.set_solver_type("TGS")

    set_camera_view(
        eye=[0.0, 4.5, 3.5], target=[0.0, 0.0, 0.0],
        camera_prim_path="/OmniverseKit_Persp",
    )

    delete_prim("/Replicator/DomeLight_Xform")
    rep.create.light(position=[0, 0, 0], light_type="dome")

    Real_Ground(world.scene, visual_material_usd=None)

    delete_prim("/World/Table")
    FixedCuboid(
        prim_path="/World/Table", name="table",
        position=TABLE_POSITION,
        orientation=euler_angles_to_quat(np.array(TABLE_ORIENTATION), degrees=True),
        scale=TABLE_SCALE, size=1.0, visible=True,
    )

    delete_prim("/World/Franka")
    franka = Franka(world, np.array(ROBOT_POSITION), np.array(ROBOT_ORIENTATION))

    world.reset()
    for _ in range(50):
        world.step(render=True)

    franka.open_gripper()
    for _ in range(10):
        world.step(render=True)

    # 加载物体 USD — 搜索顺序: output/assets/ → sim/assets/
    a2g_root = os.path.dirname(SIM_DIR)  # Affordance2Grasp root
    usd_search_dirs = [
        os.path.join(a2g_root, "output", "assets"),
        os.path.join(SIM_DIR, "assets"),
        os.path.join(os.path.expanduser('~'), 'Project', 'mano2gripper', 'Pipeline', 'assets'),
    ]
    usd_path = None
    for d in usd_search_dirs:
        p = os.path.join(d, f"{obj_id}.usd")
        if os.path.exists(p):
            usd_path = p
            break

    if usd_path is None:
        cprint(f"❌ USD not found: {obj_id}.usd", "red")
        cprint(f"   搜索目录: {usd_search_dirs}", "yellow")
        cprint(f"   先运行: sim45 sim/convert_batch_usd.py", "yellow")
        return None

    obj_z_offset = 0.075 * object_scale
    obj_pos = list(OBJECT_POSITION)
    obj_pos[2] += obj_z_offset

    for i in range(10):
        delete_prim(f"/World/Rigid/rigid_{i}")
    delete_prim("/World/Rigid/rigid")

    obj = RigidObject(
        world, usd_path=usd_path,
        pos=np.array(obj_pos), ori=np.array(OBJECT_ORIENTATION),
        scale=np.array([object_scale] * 3), mass=0.05,  # 空瓶 ~50g
    )

    # 碰撞 + 摩擦力设置
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf, UsdShade
    stage = world.stage

    # 1. 创建高摩擦材料
    material_path = "/World/PhysicsMaterials/BottleMaterial"
    UsdShade.Material.Define(stage, material_path)
    mat_prim = stage.GetPrimAtPath(material_path)
    physics_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    physics_mat.CreateStaticFrictionAttr(1.0)    # 高静摩擦
    physics_mat.CreateDynamicFrictionAttr(0.8)   # 高动摩擦
    physics_mat.CreateRestitutionAttr(0.0)       # 不弹跳
    cprint(f"   ✅ Physics material: friction=1.0/0.8", "green")

    # 2. 物体碰撞 + 绑定摩擦材料
    obj_prim = stage.GetPrimAtPath(obj.rigid_prim_path)
    for prim in Usd.PrimRange(obj_prim):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_col.GetApproximationAttr().Set("convexHull")
            col_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            col_api.GetContactOffsetAttr().Set(0.02)
            col_api.GetRestOffsetAttr().Set(0.001)
            # 绑定摩擦材料
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(
                UsdShade.Material(mat_prim),
                UsdShade.Tokens.weakerThanDescendants,
                "physics"
            )

    # 3. 给夹爪指尖也加摩擦
    finger_material_path = "/World/PhysicsMaterials/FingerMaterial"
    UsdShade.Material.Define(stage, finger_material_path)
    finger_mat_prim = stage.GetPrimAtPath(finger_material_path)
    finger_physics_mat = UsdPhysics.MaterialAPI.Apply(finger_mat_prim)
    finger_physics_mat.CreateStaticFrictionAttr(1.2)   # 更高摩擦 (橡胶指尖)
    finger_physics_mat.CreateDynamicFrictionAttr(1.0)
    finger_physics_mat.CreateRestitutionAttr(0.0)

    for finger_name in ["panda_leftfinger", "panda_rightfinger"]:
        finger_path = f"/World/Franka/{finger_name}"
        finger_prim = stage.GetPrimAtPath(finger_path)
        if finger_prim.IsValid():
            for child in Usd.PrimRange(finger_prim):
                if child.IsA(UsdGeom.Mesh) or child.IsA(UsdGeom.Gprim):
                    binding = UsdShade.MaterialBindingAPI.Apply(child)
                    binding.Bind(
                        UsdShade.Material(finger_mat_prim),
                        UsdShade.Tokens.weakerThanDescendants,
                        "physics"
                    )
            cprint(f"   ✅ Finger friction on {finger_name}", "green")

    for _ in range(100):
        world.step(render=True)

    cprint("✅ Scene Ready", "green")
    return {"world": world, "franka": franka, "obj": obj}


# ============================================================
# Coordinate Transforms
# ============================================================
def get_robot_base_transform():
    """T_world_robot 和 T_robot_world."""
    yaw_rad = np.deg2rad(ROBOT_ORIENTATION[2])
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T[:3, 3] = ROBOT_POSITION
    return T, np.linalg.inv(T)


def world_to_robot_pose(pos_w, quat_wxyz_w, T_robot_world):
    """世界坐标 → 机器人底座坐标."""
    pos_r = (T_robot_world @ np.append(pos_w, 1.0))[:3]

    R_w = Rotation.from_quat([quat_wxyz_w[1], quat_wxyz_w[2],
                               quat_wxyz_w[3], quat_wxyz_w[0]])
    R_rw = Rotation.from_matrix(T_robot_world[:3, :3])
    R_r = R_rw * R_w
    q = R_r.as_quat()  # xyzw
    quat_r = np.array([q[3], q[0], q[1], q[2]])  # wxyz

    return pos_r, quat_r


def transform_grasp_to_world(grasp_pos_obj, grasp_rot_obj, T_world_obj):
    """OBJ 坐标 → 世界坐标."""
    pos_w = (T_world_obj @ np.append(grasp_pos_obj, 1.0))[:3]
    rot_w = T_world_obj[:3, :3] @ grasp_rot_obj
    return pos_w, rot_w


def make_transform(pos, quat_wxyz):
    """位置+四元数 → 4x4 变换矩阵."""
    T = np.eye(4)
    r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = pos
    return T


# ============================================================
# cuRobo Motion Planning
# ============================================================
_CUROBO_MG = None

def init_curobo():
    """初始化 cuRobo MotionGen."""
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    _, T_robot_world = get_robot_base_transform()

    # 桌面障碍物 (变换到机器人底座坐标)
    table_pos_r = (T_robot_world @ np.append(TABLE_POSITION, 1.0))[:3]
    ground_pos_r = (T_robot_world @ np.array([0, 0, -0.005, 1.0]))[:3]

    world_config = {
        "cuboid": {
            "table": {
                "dims": [TABLE_SCALE[0], TABLE_SCALE[1], TABLE_SCALE[2]],
                "pose": [*table_pos_r.tolist(), 1, 0, 0, 0],
            },
            "ground": {
                "dims": [5.0, 5.0, 0.01],
                "pose": [*ground_pos_r.tolist(), 1, 0, 0, 0],
            },
        },
    }

    mg_config = MotionGenConfig.load_from_robot_config(
        "franka.yml", world_config, interpolation_dt=0.02,
    )
    mg = MotionGen(mg_config)

    cprint("   → cuRobo warmup...", "yellow")
    mg.warmup()
    cprint("   ✅ cuRobo ready", "green")
    return mg


def plan_trajectory(motion_gen, franka, target_pos_world, target_quat_wxyz_world, label=""):
    """cuRobo 规划无碰撞轨迹. 返回关节轨迹 (numpy) 或 None."""
    from curobo.types.math import Pose
    from curobo.types.robot import JointState as CuJointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    _, T_robot_world = get_robot_base_transform()
    pos_r, quat_r = world_to_robot_pose(target_pos_world, target_quat_wxyz_world, T_robot_world)

    euler_r = Rotation.from_quat([quat_r[1], quat_r[2], quat_r[3], quat_r[0]]).as_euler('xyz', degrees=True)

    cprint(f"      [{label}] target (world): pos=[{target_pos_world[0]:.4f}, {target_pos_world[1]:.4f}, {target_pos_world[2]:.4f}]", "magenta")
    cprint(f"      [{label}] target (robot): pos=[{pos_r[0]:.4f}, {pos_r[1]:.4f}, {pos_r[2]:.4f}]", "magenta")
    cprint(f"      [{label}] target (robot): euler=[{euler_r[0]:.1f}°, {euler_r[1]:.1f}°, {euler_r[2]:.1f}°]", "magenta")
    cprint(f"      [{label}] target (robot): quat_wxyz=[{quat_r[0]:.4f}, {quat_r[1]:.4f}, {quat_r[2]:.4f}, {quat_r[3]:.4f}]", "magenta")

    current_joints = franka.get_joint_positions()[:7]
    joint_names = [f"panda_joint{i}" for i in range(1, 8)]

    start_state = CuJointState.from_position(
        torch.tensor(current_joints, dtype=torch.float32).unsqueeze(0).cuda(),
        joint_names=joint_names,
    )

    goal_pose = Pose.from_list([
        float(pos_r[0]), float(pos_r[1]), float(pos_r[2]),
        float(quat_r[0]), float(quat_r[1]), float(quat_r[2]), float(quat_r[3]),
    ])

    plan_config = MotionGenPlanConfig(
        max_attempts=10, enable_graph=True, enable_opt=True,
    )

    result = motion_gen.plan_single(start_state, goal_pose, plan_config)

    if result.success.item():
        traj = result.get_interpolated_plan()
        cprint(f"      [{label}] ✅ Plan OK: {traj.position.shape[0]} steps", "green")
        return traj.position.cpu().numpy()

    # ---- 详细失败诊断 ----
    cprint(f"      [{label}] ❌ Plan FAILED. Diagnostics:", "red")

    # 1. 检查 IK 是否可达
    try:
        ik_result = motion_gen.ik_solver.solve_single(goal_pose)
        if ik_result.success.item():
            cprint(f"      [{label}]   IK: ✅ 可达 (位置误差={ik_result.position_error[0].item()*1000:.2f}mm)", "yellow")
        else:
            cprint(f"      [{label}]   IK: ❌ 不可达! 这个位姿在机械臂工作空间外", "red")
            cprint(f"      [{label}]   IK 位置误差: {ik_result.position_error[0].item()*1000:.2f}mm", "red")
    except Exception as e:
        cprint(f"      [{label}]   IK check error: {e}", "red")

    # 2. 检查 result 的其他属性
    try:
        if hasattr(result, 'valid_query') and result.valid_query is not None:
            cprint(f"      [{label}]   valid_query: {result.valid_query.item()}", "yellow")
        if hasattr(result, 'status') and result.status is not None:
            cprint(f"      [{label}]   status: {result.status}", "yellow")
        if hasattr(result, 'attempts') and result.attempts is not None:
            cprint(f"      [{label}]   attempts: {result.attempts}", "yellow")
        if hasattr(result, 'position_error') and result.position_error is not None:
            cprint(f"      [{label}]   position_error: {result.position_error.item()*1000:.2f}mm", "yellow")
        if hasattr(result, 'rotation_error') and result.rotation_error is not None:
            cprint(f"      [{label}]   rotation_error: {result.rotation_error.item():.4f}rad", "yellow")
        if hasattr(result, 'cspace_error') and result.cspace_error is not None:
            cprint(f"      [{label}]   cspace_error: {result.cspace_error.item():.6f}", "yellow")
    except Exception as e:
        cprint(f"      [{label}]   (error reading result attrs: {e})", "yellow")

    # 3. 打印桌面位置供参考
    table_pos_r = (T_robot_world @ np.append(TABLE_POSITION, 1.0))[:3]
    cprint(f"      [{label}]   Table in robot frame: z={table_pos_r[2]:.4f}", "yellow")
    cprint(f"      [{label}]   Target z in robot frame: {pos_r[2]:.4f}", "yellow")
    cprint(f"      [{label}]   距离桌面: {(pos_r[2] - table_pos_r[2])*100:.1f}cm", "yellow")

    return None


def execute_trajectory(franka, world, traj):
    """执行关节轨迹."""
    for joint_pos in traj:
        gripper = franka.get_joint_positions()[7:9]
        franka.set_joint_positions(np.concatenate([joint_pos, gripper]))
        world.step(render=True)


# ============================================================
# Main Grasp Execution
# ============================================================
def execute_grasp(scene, grasp_pos_obj, grasp_rot_obj, gripper_width, object_scale):
    """完整抓取流程."""
    global _CUROBO_MG

    franka = scene["franka"]
    world = scene["world"]

    # ---- 获取物体世界位姿 ----
    obj_pos_world, obj_quat_wxyz = scene["obj"].get_obj_pos()
    T_world_obj = make_transform(obj_pos_world, obj_quat_wxyz)

    # scale 影响: OBJ 坐标中的位置需要乘以 scale
    grasp_pos_scaled = grasp_pos_obj * object_scale
    pos_world, rot_world = transform_grasp_to_world(grasp_pos_scaled, grasp_rot_obj, T_world_obj)

    # ⭐ 关键: 坐标系约定转换
    # 我们的抓取坐标系:     x = 夹爪开合,  y = 沿瓶身(竖直),  z = 接近
    # Franka panda_hand:     y = 夹爪开合,                      z = 接近
    # 需要绕 Z 轴(接近方向)旋转 -90°, 让我们的 x → panda_hand 的 y
    R_adapt = np.array([
        [0, 1, 0],   # new_x = old_y (沿瓶身 → panda x)
        [-1, 0, 0],  # new_y = -old_x (夹爪开合 → panda y, 取反保持右手系)
        [0, 0, 1],   # new_z = old_z (接近方向不变)
    ], dtype=np.float64)
    rot_world = rot_world @ R_adapt

    # TCP 偏移处理
    # ⭐ 手动标注的 position = 指尖夹持中心, 需要在 sim 内减去 TCP 偏移
    # ⭐ 自动生成的 position = panda_hand 位置, 已包含 TCP 偏移
    TCP_OFFSET = 0.105
    approach_dir = rot_world[:, 2]  # 接近方向 = 旋转矩阵第3列
    pos_world = pos_world - approach_dir * TCP_OFFSET

    # Z 安全限制 — 侧面抓取时手臂关节比夹爪低, 需要间距
    # 桌面 z=0.80, 最低抓取 z=0.88 (+8cm)
    MIN_GRASP_Z = TABLE_TOP_Z + 0.08
    if pos_world[2] < MIN_GRASP_Z:
        cprint(f"   ⚠️ Z={pos_world[2]:.3f} 太低 (需 >{MIN_GRASP_Z:.3f}), clamp up", "yellow")
        pos_world[2] = MIN_GRASP_Z

    # 旋转矩阵 → 四元数 wxyz
    q_xyzw = Rotation.from_matrix(rot_world).as_quat()
    quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    euler = Rotation.from_matrix(rot_world).as_euler('xyz', degrees=True)

    lift_pos = pos_world.copy()
    lift_pos[2] += LIFT_HEIGHT

    cprint(f"\n🤖 Executing grasp:", "cyan")
    cprint(f"   World pos:  [{pos_world[0]:.4f}, {pos_world[1]:.4f}, {pos_world[2]:.4f}]", "cyan")
    cprint(f"   Euler:      [{euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°]", "cyan")
    cprint(f"   Gripper:    {gripper_width*100:.1f}cm", "cyan")

    # 记录初始物体 Z
    obj_init, _ = scene["obj"].get_obj_pos()
    initial_z = obj_init[2]

    # ---- 初始化 cuRobo ----
    if _CUROBO_MG is None:
        try:
            _CUROBO_MG = init_curobo()
        except Exception as e:
            cprint(f"   ❌ cuRobo init failed: {e}", "red")
            return False

    # ---- 打开夹爪 ----
    franka.open_gripper()
    for _ in range(30):
        world.step(render=True)

    # ---- Phase 1: 规划到预抓取点 (Pre-grasp) ----
    # 预抓取点 = 抓取点沿接近方向后退 12cm, 避免轨迹穿过瓶子
    approach_dir = rot_world[:, 2]  # z 轴 = 接近方向 (已做 R_adapt)
    pre_grasp_offset = 0.15  # 15cm
    pre_grasp_pos = pos_world - approach_dir * pre_grasp_offset

    cprint(f"   → [1/5] Planning to pre-grasp point...", "yellow")
    cprint(f"      pre-grasp: [{pre_grasp_pos[0]:.4f}, {pre_grasp_pos[1]:.4f}, {pre_grasp_pos[2]:.4f}]", "magenta")
    traj = plan_trajectory(_CUROBO_MG, franka, pre_grasp_pos, quat_wxyz, label="pre-grasp")

    if traj is None:
        # Fallback: 直接规划到抓取点
        cprint(f"   → Pre-grasp 失败, 直接规划到抓取点...", "yellow")
        traj = plan_trajectory(_CUROBO_MG, franka, pos_world, quat_wxyz, label="direct")

    if traj is None:
        cprint(f"   ❌ cuRobo 规划全部失败", "red")
        return False

    cprint(f"   ✅ Pre-grasp trajectory: {len(traj)} steps", "green")

    # ---- Phase 2: 执行到预抓取点 ----
    cprint(f"   → [2/5] Moving to pre-grasp...", "yellow")
    for joint_pos in traj:
        gripper = franka.get_joint_positions()[7:9]
        franka.set_joint_positions(np.concatenate([joint_pos, gripper]))
        world.step(render=True)

    for _ in range(10):
        world.step(render=True)

    # ---- Phase 3: 从预抓取点缓慢推进到抓取点 ----
    cprint(f"   → [3/5] Final approach (slow)...", "yellow")
    # 用 IK 求抓取点的关节角, 然后线性插值推进
    traj_final = plan_trajectory(_CUROBO_MG, franka, pos_world, quat_wxyz, label="final")
    if traj_final is not None:
        # 慢速推进, 每步多做物理模拟 (让碰撞反应自然)
        for joint_pos in traj_final:
            gripper = franka.get_joint_positions()[7:9]
            franka.set_joint_positions(np.concatenate([joint_pos, gripper]))
            for _ in range(3):
                world.step(render=True)
    else:
        # 如果规划失败, 用简单线性插值
        cprint(f"   → Final approach 规划失败, 用线性插值...", "yellow")
        current_joints = franka.get_joint_positions()[:7]
        # 保持当前位置, 不额外推进


    # ---- Phase 4: 闭合夹爪 + 力传感器 ----
    cprint(f"   → [4/5] Closing gripper with force sensing...", "yellow")

    # 设置接触力传感器 (PhysX contact report)
    from pxr import UsdPhysics, PhysxSchema
    stage = world.stage
    left_finger_path = "/World/Franka/panda_leftfinger"
    right_finger_path = "/World/Franka/panda_rightfinger"

    # 开启 contact report
    for finger_path in [left_finger_path, right_finger_path]:
        prim = stage.GetPrimAtPath(finger_path)
        if prim.IsValid():
            cr = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            cr.CreateThresholdAttr(0.0)  # 报告所有接触力
            cprint(f"      ✅ Contact sensor on {finger_path.split('/')[-1]}", "green")

    franka.close_gripper()

    # 闭合过程中记录力
    force_log = []
    for step in range(80):
        world.step(render=True)

        # 读取接触力
        from omni.physx import get_physx_scene_query_interface
        try:
            contact_data = get_physx_scene_query_interface().overlap_shape_any(
                left_finger_path, None)
        except:
            pass

        # 记录夹爪位置
        finger_pos = franka.get_joint_positions()[7:9]
        if step % 20 == 0:
            cprint(f"      Step {step}: fingers=[{finger_pos[0]:.4f}, {finger_pos[1]:.4f}]", "cyan")
        force_log.append(finger_pos.copy())

    # 读取闭合后的实际手指位置 (仅用于诊断)
    finger_pos_after = franka.get_joint_positions()[7:9]
    cprint(f"      夹爪位置: [{finger_pos_after[0]:.4f}, {finger_pos_after[1]:.4f}] (宽{(finger_pos_after[0]+finger_pos_after[1])*100:.2f}cm)", "magenta")

    # ⭐ 不再用 set_joint_positions 控制夹爪！
    # close_gripper() 的控制器会持续施加闭合力, 这才是真正产生物理夹持力的方式
    # set_joint_positions 是运动学瞬移, 不产生力

    # 检查夹爪是否在变化 (松开?)
    if len(force_log) > 10:
        early = np.mean([f[0] for f in force_log[:10]])
        late = np.mean([f[0] for f in force_log[-10:]])
        delta = late - early
        if delta > 0.001:
            cprint(f"      ⚠️ 夹爪在过程中张开了 {delta*100:.2f}cm!", "red")
        else:
            cprint(f"      ✅ 夹爪稳定 (变化 {delta*100:.3f}cm)", "green")

    # ---- Phase 5: 提起 (夹爪由控制器维持力) ----
    cprint(f"   → [5/5] Planning lift...", "yellow")
    traj_lift = plan_trajectory(_CUROBO_MG, franka, lift_pos, quat_wxyz, label="lift")

    if traj_lift is not None:
        cprint(f"   ✅ Lift trajectory: {len(traj_lift)} steps", "green")
        # close_gripper() 只需调一次, 控制器会持续施力
        franka.close_gripper()
        for joint_pos in traj_lift:
            # 用 apply_action 或直接设 DC target, 只设手臂
            from omni.isaac.core.utils.types import ArticulationAction
            action = ArticulationAction(
                joint_positions=np.concatenate([joint_pos, np.array([None, None])]),
            )
            franka.apply_action(action)
            for _ in range(2):
                world.step(render=True)
    else:
        cprint(f"   ⚠️ Lift planning failed, skipping", "yellow")

    # 稳定 — 夹爪控制器持续施力
    for _ in range(80):
        world.step(render=True)

    # ---- 检查结果 ----
    obj_after, _ = scene["obj"].get_obj_pos()
    z_delta = obj_after[2] - initial_z
    success = z_delta > 0.03

    cprint(f"   📍 obj Z: {initial_z:.4f} → {obj_after[2]:.4f} (Δ={z_delta:.4f}m)", "cyan")

    result = {'success': success, 'contact_points_local': None, 'finger_width_actual': None}

    if success:
        cprint(f"   ✅ GRASP SUCCESS!", "green", "on_green")
        # ★ 提取手指尖实际位置 (世界坐标)
        try:
            from pxr import UsdGeom
            stage = world.stage
            # 获取左右手指的世界位置
            left_xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Franka/panda_leftfinger"))
            right_xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Franka/panda_rightfinger"))
            left_world = np.array(left_xform.ComputeLocalToWorldTransform(0).ExtractTranslation())
            right_world = np.array(right_xform.ComputeLocalToWorldTransform(0).ExtractTranslation())

            # 转换到物体初始局部坐标系
            # 物体初始位置 = OBJECT_POSITION + z_offset
            obj_init_pos = np.array(OBJECT_POSITION)
            obj_init_pos[2] += 0.075 * object_scale  # z offset
            left_local = left_world - obj_init_pos
            right_local = right_world - obj_init_pos

            finger_width = np.linalg.norm(left_world - right_world)
            cprint(f"   📐 手指接触: L={left_local}, R={right_local}, width={finger_width*100:.2f}cm", "magenta")

            result['contact_points_local'] = np.array([left_local, right_local], dtype=np.float32)
            result['finger_width_actual'] = float(finger_width)
        except Exception as e:
            cprint(f"   ⚠️ 无法提取手指位置: {e}", "yellow")
    else:
        cprint(f"   ❌ GRASP FAILED", "red")

    return result


# ============================================================
# Main
# ============================================================
def main():
    h5_path = args.hdf5
    if not os.path.exists(h5_path):
        cprint(f"❌ HDF5 not found: {h5_path}", "red")
        cprint(f"   先运行: python Pipeline/generate_grasp_pose.py --mesh ...", "yellow")
        simulation_app.close()
        return

    # ---- 读取 HDF5 ----
    cprint("=" * 60, "cyan")
    cprint("Pipeline Stage B: Isaac Sim Grasp Execution", "cyan")
    cprint("=" * 60, "cyan")
    cprint(f"  HDF5: {h5_path}", "cyan")

    with h5py.File(h5_path, 'r') as f:
        # 兼容两种格式: 自动生成 vs 手动标注
        if "metadata" in f:
            # v2 自动生成格式
            obj_id = f["metadata"].attrs["obj_id"]
            n_contact = f["affordance"].attrs.get("n_contact", 0)
        else:
            # 手动标注格式
            obj_id = f.attrs.get('object_id', f.attrs.get('obj_id', 'unknown'))
            n_contact = 0

        grasp_candidates = []

        if "candidates" in f:
            # v2 自动生成: candidates/candidate_N
            n_cand = f["candidates"].attrs["n_candidates"]
            for i in range(n_cand):
                ci = f[f"candidates/candidate_{i}"]
                grasp_candidates.append({
                    "name": ci.attrs["name"],
                    "score": ci.attrs["score"],
                    "position": ci["position"][:],
                    "rotation": ci["rotation"][:],
                    "gripper_width": ci.attrs["gripper_width"],
                    "approach_type": ci.attrs["approach_type"],
                })
            cprint(f"  Candidates: {n_cand} (v2 multi-candidate)", "cyan")

        elif "candidate_0" in f:
            # 手动标注格式: candidate_N 在根目录
            n_cand = f.attrs.get('num_candidates', 0)
            for i in range(n_cand):
                key = f'candidate_{i}'
                if key not in f: continue
                ci = f[key]
                grasp_candidates.append({
                    "name": ci.attrs.get("name", f"manual_{i}"),
                    "score": ci.attrs.get("score", 80.0),
                    "position": ci["position"][:],
                    "rotation": ci["rotation"][:],
                    "gripper_width": ci.attrs["gripper_width"],
                    "approach_type": ci.attrs.get("approach_type", "horizontal"),
                })
            cprint(f"  Candidates: {len(grasp_candidates)} (manual annotation)", "cyan")

        else:
            # v1 兼容: 只有一个位姿
            grasp_candidates.append({
                "name": "legacy",
                "score": 0.0,
                "position": f["grasp/position"][:],
                "rotation": f["grasp/rotation"][:],
                "gripper_width": f["grasp"].attrs["gripper_width"],
                "approach_type": "horizontal",
            })
            cprint(f"  Candidates: 1 (v1 legacy)", "cyan")

    cprint(f"  Object:    {obj_id}", "cyan")
    cprint(f"  Contacts:  {n_contact}", "cyan")
    cprint(f"  Scale:     {args.object_scale}x", "cyan")

    for i, c in enumerate(grasp_candidates):
        euler = Rotation.from_matrix(c["rotation"]).as_euler('xyz', degrees=True)
        marker = "⭐" if i == 0 else "  "
        cprint(f"  {marker} [{i+1}] {c['name']:>16s}  score={c['score']:5.1f}  "
               f"gripper={c['gripper_width']*100:.1f}cm", "cyan")

    # ---- 搭建场景 ----
    cprint(f"\n📦 Setting up scene...", "yellow")
    scene = setup_scene(obj_id, args.object_scale)
    if scene is None:
        simulation_app.close()
        return

    # ---- 等待用户调整视角 (Sim 保持交互) ----
    if not args.headless:
        import sys, select
        cprint("🎥 在 Sim 中调整视角, 终端按 Enter 开始抓取...", "cyan")
        while True:
            scene["world"].step(render=True)
            # 非阻塞检测 Enter
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                break

    # ---- 逐候选尝试抓取 ----
    success = False
    candidate_results = []  # 记录每个候选的结果
    winning_candidate = None

    for attempt, cand in enumerate(grasp_candidates):
        cprint(f"\n{'='*40}", "yellow")
        cprint(f"  🔄 Attempt {attempt+1}/{len(grasp_candidates)}: {cand['name']} (score={cand['score']:.1f})", "yellow")
        cprint(f"{'='*40}", "yellow")

        try:
            grasp_result = execute_grasp(
                scene,
                cand["position"],
                cand["rotation"],
                cand["gripper_width"],
                args.object_scale
            )
            success = grasp_result['success']
        except Exception as e:
            cprint(f"  ❌ Error: {e}", "red")
            success = False
            grasp_result = {'success': False, 'contact_points_local': None, 'finger_width_actual': None}

        candidate_results.append({
            'name': cand['name'],
            'score': cand['score'],
            'success': success,
            'grasp_point': cand.get('position', np.zeros(3)),
            'rotation': cand.get('rotation', np.eye(3)),
            'gripper_width': cand['gripper_width'],
            'approach_type': cand.get('approach_type', 'unknown'),
            'contact_points_local': grasp_result.get('contact_points_local'),
            'finger_width_actual': grasp_result.get('finger_width_actual'),
        })

        if success:
            cprint(f"\n  ✅ SUCCESS with candidate: {cand['name']}", "green")
            if winning_candidate is None:
                winning_candidate = cand  # 记录第一个成功的
        else:
            cprint(f"  ❌ FAILED with candidate: {cand['name']}", "red")

        # 每次尝试后都重置场景 (最后一个除外)
        if attempt < len(grasp_candidates) - 1:
            cprint(f"  → 重置场景, 尝试下一个候选...", "yellow")
            # 重置物体位置+朝向 (竖直)
            obj_z_offset = 0.075 * args.object_scale
            reset_pos = list(OBJECT_POSITION)
            reset_pos[2] += obj_z_offset
            scene["obj"].set_obj_pose(np.array(reset_pos), ori=np.array([0.0, 0.0, 0.0]))
            # 机械臂直接回 home 关节角 (Franka 默认)
            home_joints = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
            scene["franka"].set_joint_positions(home_joints)
            # 等物理稳定
            for _ in range(80):
                scene["world"].step(render=True)

    # ---- 保存 Robot GT 结果 (所有成功的都保存) ----
    successful_grasps = [cr for cr in candidate_results if cr['success']]
    any_success = len(successful_grasps) > 0

    if args.save_result:
        result_dir = args.result_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output", "robot_gt")
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, f"{obj_id}_robot_gt.hdf5")

        with h5py.File(result_path, 'w') as rf:
            rf.attrs['obj_id'] = obj_id
            rf.attrs['success'] = any_success
            rf.attrs['n_candidates_tried'] = len(candidate_results)
            rf.attrs['n_candidates_total'] = len(grasp_candidates)
            rf.attrs['n_successful'] = len(successful_grasps)
            rf.attrs['object_scale'] = args.object_scale

            # 兼容: 保留 winning_candidate (第一个成功的)
            if winning_candidate is not None:
                wg = rf.create_group('winning_candidate')
                wg.attrs['name'] = winning_candidate['name']
                wg.attrs['score'] = winning_candidate['score']
                wg.attrs['gripper_width'] = winning_candidate['gripper_width']
                wg.attrs['approach_type'] = winning_candidate.get('approach_type', '')
                wg.create_dataset('grasp_point', data=winning_candidate['position'])
                wg.create_dataset('rotation', data=winning_candidate['rotation'])
                wg.create_dataset('approach_dir', data=winning_candidate['rotation'][:, 2])
                wg.create_dataset('finger_dir', data=winning_candidate['rotation'][:, 0])

            # ★ 所有成功的抓取都保存为 GT
            sg = rf.create_group('successful_grasps')
            sg.attrs['count'] = len(successful_grasps)
            for i, cr in enumerate(successful_grasps):
                gi = sg.create_group(f'grasp_{i}')
                gi.attrs['name'] = cr['name']
                gi.attrs['score'] = cr['score']
                gi.attrs['gripper_width'] = cr['gripper_width']
                gi.attrs['approach_type'] = cr['approach_type']
                gi.create_dataset('grasp_point', data=cr['grasp_point'])
                gi.create_dataset('rotation', data=cr['rotation'])
                gi.create_dataset('approach_dir', data=cr['rotation'][:, 2])
                gi.create_dataset('finger_dir', data=cr['rotation'][:, 0])
                # ★ 真实接触点 (物体局部坐标系)
                if cr.get('contact_points_local') is not None:
                    gi.create_dataset('contact_points_local', data=cr['contact_points_local'])
                    gi.attrs['finger_width_actual'] = cr.get('finger_width_actual', 0.0)
                    gi.attrs['has_contact_points'] = True
                else:
                    gi.attrs['has_contact_points'] = False

            # 所有候选的结果 (含失败的)
            cg = rf.create_group('candidate_results')
            for i, cr in enumerate(candidate_results):
                ci = cg.create_group(f'candidate_{i}')
                ci.attrs['name'] = cr['name']
                ci.attrs['score'] = cr['score']
                ci.attrs['success'] = cr['success']
                ci.attrs['gripper_width'] = cr['gripper_width']
                ci.attrs['approach_type'] = cr['approach_type']
                ci.create_dataset('grasp_point', data=cr['grasp_point'])
                ci.create_dataset('rotation', data=cr['rotation'])

        cprint(f"\n  📁 Saved: {result_path}  ({len(successful_grasps)} 个成功GT)", "green")

    # ---- 等待观察 ----
    if not args.headless:
        n_success = len(successful_grasps)
        cprint(f"\n{'=' * 60}", "cyan")
        cprint(f"  结果: {'✅ SUCCESS' if any_success else '❌ ALL FAILED'}  ({n_success}/{len(candidate_results)} 成功, {n_success} 条GT)", "green" if any_success else "red")
        for cr in candidate_results:
            icon = '✅' if cr['success'] else '❌'
            cprint(f"    {icon} {cr['name']:>12s}  score={cr['score']:.1f}", "green" if cr['success'] else "red")
        cprint(f"  保持 5 秒...", "cyan")
        cprint(f"{'=' * 60}", "cyan")
        hold_arm = scene["franka"].get_joint_positions()[:7]
        for _ in range(250):  # ~5秒
            all_j = scene["franka"].get_joint_positions()
            all_j[:7] = hold_arm
            scene["franka"].set_joint_positions(all_j)
            scene["franka"].close_gripper()
            scene["world"].step(render=True)

    simulation_app.close()


main()

