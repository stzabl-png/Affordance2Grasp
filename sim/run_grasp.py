#!/usr/bin/env python3
"""
Affordance2Grasp — Isaac Sim 抓取执行
======================================
输入: 抓取位姿 HDF5 (inference/grasp_pose.py 生成)
输出: Isaac Sim 中 Franka 执行无碰撞抓取

运行:
    cd /home/lyh/Project/Affordance2Grasp
    sim45 sim/run_grasp.py --hdf5 output/grasps/A16013_grasp.hdf5

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
parser.add_argument("--object_scale", type=float, default=1.0, help="物体缩放 (默认 1.0, 原始尺寸)")
parser.add_argument("--object_pos", type=float, nargs=2, default=None,
                    help="物体 XY 位置 (默认 0.0 0.55), 例: --object_pos -0.1 0.5")
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

# 项目根目录加入 path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from sim.env_config.franka import Franka
from sim.env_config.real_ground import Real_Ground
from sim.env_config.rigid_object import RigidObject


# ============================================================
# Scene Config (和 Replay_Grasp / Scene_Env 一致)
# ============================================================
ROBOT_POSITION = [0.2, -0.05, 0.8]
ROBOT_ORIENTATION = [0.0, 0.0, 90.0]
TABLE_POSITION = [0.0, 1.0, 0.75]
TABLE_ORIENTATION = [0.0, 0.0, 0.0]
TABLE_SCALE = [2.0, 2.0, 0.1]
TABLE_TOP_Z = 0.80

# 物体位置 (可通过 --object_pos X Y 覆盖)
# 机器人底座: [0.2, -0.05, 0.8] 朝 Y+ 方向
# ┌─────────────── 可达范围 (桌面俯视) ──────────────┐
# │  X 范围: [-0.3,  0.4]   (左右, 0.2±0.5m)        │
# │  Y 范围: [ 0.2,  0.75]  (远近, 机器人前方)       │
# │  推荐测试: (0,0.55) (±0.1,0.55) (0,0.45) (0,0.65)│
# └──────────────────────────────────────────────────┘
if args.object_pos:
    OBJECT_POSITION = [args.object_pos[0], args.object_pos[1], TABLE_TOP_Z]
else:
    OBJECT_POSITION = [0.0, 0.55, TABLE_TOP_Z]
OBJECT_ORIENTATION = [0.0, 0.0, 0.0]

APPROACH_HEIGHT = 0.15  # 物体上方 15cm
MAX_LIFT_Z = TABLE_TOP_Z + 0.40  # 提升的最大 Z 高度 (1.20m, Franka 工作范围内)


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

    # 加载物体 USD
    usd_path = os.path.join(_PROJECT_ROOT, "assets", "usd", f"{obj_id}.usd")

    if not os.path.exists(usd_path):
        cprint(f"❌ USD not found: {usd_path}", "red")
        cprint(f"   先运行: python assets/convert_obj_to_usd.py --input /path/to/{obj_id}.obj", "yellow")
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

    # ⭐ position 已经是 panda_hand EE 位置 (grasp_pose.py 中已做 TCP 偏移)
    # 无需额外偏移, pos_world 直接用作 cuRobo 目标

    # Z 安全限制 — 桌面 z=0.80, 夹爪不能低于桌面
    # 1.0x 物体较矮, 合理抓取高度可能只比桌面高 5cm
    MIN_GRASP_Z = TABLE_TOP_Z + 0.05  # 0.85
    if pos_world[2] < MIN_GRASP_Z:
        cprint(f"   ⚠️ Z={pos_world[2]:.3f} 太低 (需 >{MIN_GRASP_Z:.3f}), clamp up", "yellow")
        pos_world[2] = MIN_GRASP_Z

    # 旋转矩阵 → 四元数 wxyz
    q_xyzw = Rotation.from_matrix(rot_world).as_quat()
    quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    euler = Rotation.from_matrix(rot_world).as_euler('xyz', degrees=True)

    lift_pos = pos_world.copy()
    lift_pos[2] = MAX_LIFT_Z  # 提到固定的最大安全高度

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
    # 预抓取点 = 抓取点沿接近方向后退 20cm
    approach_dir = rot_world[:, 2]  # z 轴 = 接近方向 (已做 R_adapt)
    pre_grasp_offset = 0.20  # 20cm
    pre_grasp_pos = pos_world - approach_dir * pre_grasp_offset

    cprint(f"   → [1/5] Planning to pre-grasp point...", "yellow")
    cprint(f"      pre-grasp: [{pre_grasp_pos[0]:.4f}, {pre_grasp_pos[1]:.4f}, {pre_grasp_pos[2]:.4f}]", "magenta")
    traj = plan_trajectory(_CUROBO_MG, franka, pre_grasp_pos, quat_wxyz, label="pre-grasp")

    if traj is None:
        cprint(f"   ❌ Pre-grasp 不可达, 跳过此候选", "red")
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

    # ---- Phase 3: 从预抓取点沿接近方向直线推进到抓取点 ----
    cprint(f"   → [3/5] Final approach (linear)...", "yellow")
    # 用 cuRobo 的轨迹规划器求抓取点的关节角 (取末尾点作为 IK 解)
    traj_final = plan_trajectory(_CUROBO_MG, franka, pos_world, quat_wxyz, label="final")
    if traj_final is not None and len(traj_final) > 0:
        target_joints = traj_final[-1]  # IK 解: 抓取点的关节角
        current_joints = franka.get_joint_positions()[:7]
        # 关节空间线性插值 (短距离+同姿态 → 笛卡尔近似直线)
        N_INTERP = 40  # 插值步数
        for i in range(1, N_INTERP + 1):
            alpha = i / N_INTERP
            interp_joints = current_joints * (1 - alpha) + target_joints * alpha
            gripper = franka.get_joint_positions()[7:9]
            franka.set_joint_positions(np.concatenate([interp_joints, gripper]))
            for _ in range(3):
                world.step(render=True)
    else:
        cprint(f"   ⚠️ Final approach IK 失败, 保持当前位置", "yellow")


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
    if success:
        cprint(f"   ✅ GRASP SUCCESS!", "green", "on_green")
    else:
        cprint(f"   ❌ GRASP FAILED", "red")

    return success


# ============================================================
# Main
# ============================================================
def main():
    h5_path = args.hdf5
    if not os.path.exists(h5_path):
        cprint(f"❌ HDF5 not found: {h5_path}", "red")
        cprint(f"   先运行: python -m inference.grasp_pose --mesh ...", "yellow")
        simulation_app.close()
        return

    # ---- 读取 HDF5 ----
    cprint("=" * 60, "cyan")
    cprint("Pipeline Stage B: Isaac Sim Grasp Execution", "cyan")
    cprint("=" * 60, "cyan")
    cprint(f"  HDF5: {h5_path}", "cyan")

    with h5py.File(h5_path, 'r') as f:
        obj_id = f["metadata"].attrs["obj_id"]
        n_contact = f["affordance"].attrs["n_contact"]

        # 读取所有候选 (按 score 降序)
        all_candidates = []
        if 'candidates' in f:
            n_cand = f['candidates'].attrs.get('n_candidates', 0)
            for i in range(n_cand):
                cg = f[f'candidates/candidate_{i}']
                all_candidates.append({
                    'name': str(cg.attrs['name']),
                    'position': cg['position'][:],
                    'rotation': cg['rotation'][:],
                    'gripper_width': float(cg.attrs['gripper_width']),
                    'score': float(cg.attrs['score']),
                })
        else:
            # 兼容旧格式: 只有 grasp/
            all_candidates.append({
                'name': 'best',
                'position': f["grasp/position"][:],
                'rotation': f["grasp/rotation"][:],
                'gripper_width': float(f["grasp"].attrs["gripper_width"]),
                'score': 0,
            })

    cprint(f"  Object:    {obj_id}", "cyan")
    cprint(f"  Contacts:  {n_contact}", "cyan")
    cprint(f"  Scale:     {args.object_scale}x", "cyan")
    cprint(f"  Candidates: {len(all_candidates)}", "cyan")
    for i, c in enumerate(all_candidates):
        euler_c = Rotation.from_matrix(c['rotation']).as_euler('xyz', degrees=True)
        cprint(f"    [{i+1}] {c['name']:>20s}  score={c['score']:.1f}  "
               f"gripper={c['gripper_width']*100:.1f}cm  "
               f"euler=[{euler_c[0]:.0f}°,{euler_c[1]:.0f}°,{euler_c[2]:.0f}°]", "cyan")

    # ---- 搭建场景 ----
    cprint(f"\n📦 Setting up scene...", "yellow")
    scene = setup_scene(obj_id, args.object_scale)
    if scene is None:
        simulation_app.close()
        return

    # ---- 等待用户调整视角 (Sim 保持交互) ----
    if not args.headless:
        import sys, select
        cprint("🎥 在 Sim 中调整视角, 终端按 Enter 开始可达性预检...", "cyan")
        while True:
            scene["world"].step(render=True)
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                break

    # ---- cuRobo 可达性预检 ----
    global _CUROBO_MG
    if _CUROBO_MG is None:
        try:
            _CUROBO_MG = init_curobo()
        except Exception as e:
            cprint(f"❌ cuRobo init failed: {e}", "red")
            simulation_app.close()
            return

    cprint(f"\n🔍 可达性预检 (cuRobo)...", "yellow")
    franka = scene["franka"]
    _, T_robot_world = get_robot_base_transform()

    for candidate in all_candidates:
        # 复现 execute_grasp 的坐标变换
        pos_obj = np.array(candidate['position'], dtype=np.float64)
        rot_obj = np.array(candidate['rotation'], dtype=np.float64)

        obj_pos_w, obj_quat_w = scene["obj"].get_obj_pos()
        T_world_obj = make_transform(obj_pos_w, obj_quat_w)
        grasp_pos_scaled = pos_obj * args.object_scale
        pos_w, rot_w = transform_grasp_to_world(grasp_pos_scaled, rot_obj, T_world_obj)

        # R_adapt (同 execute_grasp)
        R_adapt = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        rot_w = rot_w @ R_adapt

        # position 已是 panda_hand 位置, 无需 TCP 偏移
        approach_dir = rot_w[:, 2]  # 接近方向 (pre-grasp 计算用)

        # Z 安全限制
        MIN_GRASP_Z = TABLE_TOP_Z + 0.05
        if pos_w[2] < MIN_GRASP_Z:
            pos_w[2] = MIN_GRASP_Z

        # pre-grasp = 后退 20cm
        pre_grasp_pos = pos_w - approach_dir * 0.20
        q_xyzw = Rotation.from_matrix(rot_w).as_quat()
        quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

        # 尝试规划到 pre-grasp
        traj = plan_trajectory(_CUROBO_MG, franka, pre_grasp_pos, quat_wxyz, label=f"check-{candidate['name']}")
        if traj is None:
            candidate['score'] = 0.0
            candidate['reachable'] = False
        else:
            candidate['reachable'] = True

    # 重新按 score 排序
    all_candidates.sort(key=lambda c: c['score'], reverse=True)

    # 显示预检结果
    cprint(f"\n📊 可达性预检结果:", "cyan")
    cprint(f"{'─' * 60}", "cyan")
    for i, c in enumerate(all_candidates):
        status = "✅ 可达" if c.get('reachable', False) else "❌ 不可达"
        star = "⭐" if i == 0 and c['score'] > 0 else "  "
        cprint(f"  {star} [{i+1}] {c['name']:<20s}  score={c['score']:>5.1f}  "
               f"gripper={c['gripper_width']*100:.1f}cm  {status}", "cyan")
    cprint(f"{'─' * 60}", "cyan")

    reachable_count = sum(1 for c in all_candidates if c.get('reachable', False))
    if reachable_count == 0:
        cprint(f"❌ 没有可达的候选, 退出", "red")
        simulation_app.close()
        return

    # 等待用户确认
    if not args.headless:
        cprint(f"\n按 Enter 从最高分开始执行 ({reachable_count} 个可达候选)...", "cyan")
        while True:
            scene["world"].step(render=True)
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                break

    # ---- 依次尝试可达的候选 ----
    overall_success = False
    for ci, candidate in enumerate(all_candidates):
        if not candidate.get('reachable', False):
            continue  # 跳过不可达的

        cprint(f"\n{'─' * 60}", "cyan")
        cprint(f"  🔄 尝试候选 [{ci+1}/{len(all_candidates)}]: {candidate['name']} "
               f"(score={candidate['score']:.1f})", "cyan")
        cprint(f"{'─' * 60}", "cyan")

        success = execute_grasp(
            scene,
            candidate['position'],
            candidate['rotation'],
            candidate['gripper_width'],
            args.object_scale,
        )

        if success:
            overall_success = True
            cprint(f"\n  ✅ 候选 {candidate['name']} 成功!", "green")
            break
        else:
            cprint(f"\n  ❌ 候选 {candidate['name']} 失败", "red")
            remaining = sum(1 for c in all_candidates[ci+1:] if c.get('reachable', False))
            if remaining > 0:
                cprint(f"  → 尝试下一个候选 (剩余 {remaining} 个可达)...", "yellow")
                # 重置机械臂到初始位置
                world = scene["world"]
                default_joints = np.array([0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854])
                franka.set_joint_positions(np.concatenate([default_joints, [0.04, 0.04]]))
                for _ in range(60):
                    world.step(render=True)

    # ---- 结果展示 ----
    if not args.headless:
        cprint(f"\n{'=' * 60}", "cyan")
        cprint(f"  结果: {'✅ SUCCESS' if overall_success else '❌ ALL FAILED'}", "green" if overall_success else "red")
        cprint(f"  保持 5 秒...", "cyan")
        cprint(f"{'=' * 60}", "cyan")
        hold_arm = scene["franka"].get_joint_positions()[:7]
        for _ in range(250):
            all_j = scene["franka"].get_joint_positions()
            all_j[:7] = hold_arm
            scene["franka"].set_joint_positions(all_j)
            scene["franka"].close_gripper()
            scene["world"].step(render=True)

    simulation_app.close()


main()
