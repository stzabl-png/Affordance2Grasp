#!/usr/bin/env python3
"""
交互式夹爪标注工具 — Open3D 3D Viewer + 可拖动夹爪.

键盘控制:
  W/S: 沿 Y 轴移动 (前/后)
  A/D: 沿 X 轴移动 (左/右)
  Q/E: 沿 Z 轴移动 (上/下)
  J/L: 绕 Z 轴旋转 (yaw)
  I/K: 绕 X 轴旋转 (pitch)
  U/O: 绕 Y 轴旋转 (roll)
  +/-: 调整夹爪张开宽度
  1:   切换精细/粗略移动模式
  Space: 保存当前抓取到列表
  Enter: 导出所有抓取到 HDF5 并退出
  Esc:   取消退出

用法:
    cd /home/lyh/Project/Affordance2Grasp
    python3 tools/interactive_grasp.py --obj mug
"""
import os, sys, argparse, copy
import numpy as np
import trimesh
import h5py
import open3d as o3d

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(PROJ, 'data_hub', 'meshes', 'v1')
OUT_DIR = os.path.join(PROJ, 'output', 'grasps')
TCP_OFFSET = 0.105
MAX_GRIPPER_OPEN = 0.08


def create_gripper_geo(width=0.04):
    """创建简化 Panda 夹爪几何 (两根手指 + 基座)."""
    geos = []

    # 手指尺寸
    finger_length = 0.04
    finger_thickness = 0.008
    half_w = width / 2

    # 左手指 (红)
    left = o3d.geometry.TriangleMesh.create_box(finger_thickness, finger_thickness, finger_length)
    left.translate([-half_w - finger_thickness/2, -finger_thickness/2, 0])
    left.paint_uniform_color([0.9, 0.2, 0.2])
    geos.append(left)

    # 右手指 (红)
    right = o3d.geometry.TriangleMesh.create_box(finger_thickness, finger_thickness, finger_length)
    right.translate([half_w - finger_thickness/2, -finger_thickness/2, 0])
    right.paint_uniform_color([0.9, 0.2, 0.2])
    geos.append(right)

    # 基座 (灰)
    base_w = width + finger_thickness * 2
    base = o3d.geometry.TriangleMesh.create_box(base_w, finger_thickness, finger_thickness)
    base.translate([-base_w/2, -finger_thickness/2, -finger_thickness])
    base.paint_uniform_color([0.5, 0.5, 0.5])
    geos.append(base)

    # 手腕杆 (灰)
    wrist = o3d.geometry.TriangleMesh.create_cylinder(0.005, TCP_OFFSET)
    wrist.translate([0, 0, -TCP_OFFSET/2 - finger_thickness])
    wrist.paint_uniform_color([0.6, 0.6, 0.6])
    geos.append(wrist)

    # 接近方向箭头 (蓝)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.003, cone_radius=0.006,
        cylinder_height=0.03, cone_height=0.01)
    arrow.paint_uniform_color([0.2, 0.2, 0.9])
    geos.append(arrow)

    return geos


class InteractiveGraspAnnotator:
    def __init__(self, obj_id, mesh_path):
        self.obj_id = obj_id
        self.mesh_path = mesh_path
        self.mesh = trimesh.load(mesh_path, force='mesh')

        # 夹爪状态
        self.pos = np.array(self.mesh.centroid, dtype=np.float64)
        self.rot = np.eye(3, dtype=np.float64)  # 列: [finger_dir, y_body, approach]
        self.gripper_width = 0.04
        self.fine_mode = False  # 精细模式

        # 已保存的抓取
        self.saved_grasps = []

        # 构建场景
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=f"夹爪标注 — {obj_id}", width=1400, height=900)

        # 物体 mesh
        self.o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.o3d_mesh.compute_vertex_normals()
        self.o3d_mesh.paint_uniform_color([0.8, 0.85, 0.9])
        self.vis.add_geometry(self.o3d_mesh)

        # 坐标轴
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        self.vis.add_geometry(axes)

        # 夹爪几何 - 保存原始顶点用于变换
        self.gripper_geos = create_gripper_geo(self.gripper_width)
        self.gripper_originals = []
        for g in self.gripper_geos:
            g.compute_vertex_normals()
            self.gripper_originals.append(np.asarray(g.vertices).copy())
            self.vis.add_geometry(g)

        # 已保存抓取的可视化球
        self.saved_spheres = []

        # 渲染设置
        opt = self.vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])
        opt.mesh_show_wireframe = True

        # 注册按键
        self._register_keys()
        self._update_gripper()

        # 提示
        print("\n" + "="*60)
        print("  交互式夹爪标注工具")
        print("="*60)
        print("  W/S A/D Q/E: 移动 (Y/X/Z)")
        print("  J/L I/K U/O: 旋转 (Yaw/Pitch/Roll)")
        print("  +/-:         夹爪张开/合拢")
        print("  1:           切换精细/粗略模式")
        print("  Space:       保存当前抓取")
        print("  Enter:       导出 HDF5 并退出")
        print("  Esc:         取消退出")
        print("="*60)
        print(f"  当前模式: {'精细' if self.fine_mode else '粗略'}")
        print(f"  夹爪宽度: {self.gripper_width*100:.1f}cm")
        print()

    def _step(self):
        return 0.002 if self.fine_mode else 0.005

    def _angle_step(self):
        return np.radians(3) if self.fine_mode else np.radians(10)

    def _register_keys(self):
        # 移动
        self.vis.register_key_callback(ord('W'), lambda v: self._move([0, self._step(), 0]))
        self.vis.register_key_callback(ord('S'), lambda v: self._move([0, -self._step(), 0]))
        self.vis.register_key_callback(ord('A'), lambda v: self._move([-self._step(), 0, 0]))
        self.vis.register_key_callback(ord('D'), lambda v: self._move([self._step(), 0, 0]))
        self.vis.register_key_callback(ord('Q'), lambda v: self._move([0, 0, self._step()]))
        self.vis.register_key_callback(ord('E'), lambda v: self._move([0, 0, -self._step()]))

        # 旋转
        self.vis.register_key_callback(ord('J'), lambda v: self._rotate(2, self._angle_step()))   # yaw
        self.vis.register_key_callback(ord('L'), lambda v: self._rotate(2, -self._angle_step()))
        self.vis.register_key_callback(ord('I'), lambda v: self._rotate(0, self._angle_step()))   # pitch
        self.vis.register_key_callback(ord('K'), lambda v: self._rotate(0, -self._angle_step()))
        self.vis.register_key_callback(ord('U'), lambda v: self._rotate(1, self._angle_step()))   # roll
        self.vis.register_key_callback(ord('O'), lambda v: self._rotate(1, -self._angle_step()))

        # 夹爪宽度
        self.vis.register_key_callback(ord('='), lambda v: self._adjust_width(0.005))
        self.vis.register_key_callback(ord('-'), lambda v: self._adjust_width(-0.005))

        # 精细模式
        self.vis.register_key_callback(ord('1'), lambda v: self._toggle_fine())

        # 保存/导出
        self.vis.register_key_callback(ord(' '), lambda v: self._save_current())
        self.vis.register_key_callback(257, lambda v: self._export())  # Enter

    def _move(self, delta):
        self.pos += np.array(delta)
        self._update_gripper()
        self._print_status()

    def _rotate(self, axis, angle):
        c, s = np.cos(angle), np.sin(angle)
        if axis == 0:  # X
            R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        elif axis == 1:  # Y
            R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        else:  # Z
            R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        self.rot = self.rot @ R
        self._update_gripper()
        self._print_status()

    def _adjust_width(self, delta):
        self.gripper_width = float(np.clip(self.gripper_width + delta, 0.01, MAX_GRIPPER_OPEN))
        # 重建夹爪几何
        for g in self.gripper_geos:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self.gripper_geos = create_gripper_geo(self.gripper_width)
        self.gripper_originals = []
        for g in self.gripper_geos:
            g.compute_vertex_normals()
            self.gripper_originals.append(np.asarray(g.vertices).copy())
            self.vis.add_geometry(g, reset_bounding_box=False)
        self._update_gripper()
        self._print_status()

    def _toggle_fine(self):
        self.fine_mode = not self.fine_mode
        mode = '精细' if self.fine_mode else '粗略'
        print(f"  🔧 切换到{mode}模式 (步长: {self._step()*1000:.1f}mm)")

    def _update_gripper(self):
        """更新夹爪的位置和朝向."""
        R = self.rot
        t = self.pos
        for g, orig_verts in zip(self.gripper_geos, self.gripper_originals):
            new_verts = (R @ orig_verts.T).T + t
            g.vertices = o3d.utility.Vector3dVector(new_verts)
            g.compute_vertex_normals()
            self.vis.update_geometry(g)

    def _print_status(self):
        p = self.pos
        print(f"\r  pos=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})  "
              f"width={self.gripper_width*100:.1f}cm  "
              f"saved={len(self.saved_grasps)}  "
              f"{'[精细]' if self.fine_mode else '[粗略]'}",
              end='', flush=True)

    def _save_current(self):
        approach = self.rot[:, 2]

        # 指尖夹持中心 = 夹爪原点 + approach * 手指半长
        FINGER_HALF = 0.02  # 手指长 4cm 的一半
        finger_center = self.pos + approach * FINGER_HALF

        grasp = {
            'position': finger_center.astype(np.float32),    # ⭐ 指尖夹持中心 (sim 内部处理 TCP 偏移)
            'rotation': self.rot.astype(np.float32),
            'grasp_point': finger_center.astype(np.float32),
            'gripper_width': self.gripper_width,
            'approach_type': 'manual_interactive',
            'score': 90.0,
        }
        self.saved_grasps.append(grasp)

        # 可视化: 绿球标记在指尖夹持中心
        sphere = o3d.geometry.TriangleMesh.create_sphere(0.005)
        sphere.translate(finger_center)
        sphere.paint_uniform_color([0.1, 0.9, 0.1])
        self.vis.add_geometry(sphere, reset_bounding_box=False)
        self.saved_spheres.append(sphere)

        print(f"\n  ✅ 抓取 #{len(self.saved_grasps)} 已保存!")
        self._print_status()

    def _export(self):
        if not self.saved_grasps:
            print("\n  ⚠️  没有保存的抓取!")
            return

        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, f"{self.obj_id}_grasp.hdf5")
        with h5py.File(out_path, 'w') as f:
            f.attrs['object_id'] = self.obj_id
            f.attrs['mesh_path'] = self.mesh_path
            f.attrs['num_candidates'] = len(self.saved_grasps)
            f.attrs['source'] = 'interactive_annotation'
            for i, g in enumerate(self.saved_grasps):
                grp = f.create_group(f'candidate_{i}')
                grp.create_dataset('position', data=g['position'])
                grp.create_dataset('rotation', data=g['rotation'])
                grp.create_dataset('grasp_point', data=g['grasp_point'])
                grp.attrs['gripper_width'] = g['gripper_width']
                grp.attrs['name'] = f'interactive_{i}'
                grp.attrs['approach_type'] = g['approach_type']
                grp.attrs['score'] = g['score']

        print(f"\n\n  ✅ {len(self.saved_grasps)} 个抓取 → {out_path}")
        self.vis.destroy_window()

    def run(self):
        self.vis.run()


def main():
    parser = argparse.ArgumentParser(description='交互式夹爪标注工具')
    parser.add_argument('--obj', required=True, help='物体 ID')
    args = parser.parse_args()

    mesh_path = os.path.join(MESH_DIR, f'{args.obj}.obj')
    if not os.path.exists(mesh_path):
        # 尝试其他格式
        for ext in ['.ply', '.stl']:
            alt = os.path.join(MESH_DIR, f'{args.obj}{ext}')
            if os.path.exists(alt):
                mesh_path = alt
                break
        else:
            print(f"❌ 找不到: {mesh_path}")
            return

    annotator = InteractiveGraspAnnotator(args.obj, mesh_path)
    annotator.run()


if __name__ == '__main__':
    main()
