import os 
import sys
import torch
import carb
import numpy as np
from typing import List, Optional
from termcolor import cprint

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim, SingleRigidPrim
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, quat_to_rot_matrix, rot_matrix_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver, Franka
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.robot_motion.motion_generation.lula.motion_policies import RmpFlow, RmpFlowSmoothed
from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_motion_policy_config
from isaacsim.robot_motion.motion_generation.articulation_motion_policy import ArticulationMotionPolicy

sys.path.append(os.getcwd())
from sim.env_config.set_drive import set_drive
from sim.env_config.transforms import quat_diff_rad, Rotation, get_pose_relat, get_pose_world
from sim.env_config.code_tools import float_truncate, dense_trajectory_points_generation

class Franka(Robot):
    def __init__(
        self, 
        world:World, 
        position:np.ndarray, 
        orientation:np.ndarray, 
        robot_name:str="Franka"
    )->None:
        # define world
        self.world = world
        # define Franka name
        self._name = robot_name
        # define Franka prim
        self._prim_path = "/World/"+self._name
        # get Franka usd file
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.asset_file = os.path.join(_project_root, "assets", "robots", "franka", "franka.usd")
        # define Franka positon
        self.position = position
        # define Franka orientation
        self.orientation = euler_angles_to_quat(orientation,degrees=True)
        
        # add Franka USD to stage
        add_reference_to_stage(self.asset_file, self._prim_path)
        # set Franka
        super().__init__(
            prim_path=self._prim_path,
            name=self._name,
            position=self.position,
            orientation=self.orientation,
            articulation_controller = None
        )
        # set Franka end effector
        self._end_effector_prim_path = self._prim_path + "/panda_rightfinger"
        gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"]
        gripper_open_position = np.array([0.05, 0.05]) / get_stage_units()
        gripper_closed_position = np.array([0.0, 0.0])
        deltas = np.array([0.05, 0.05]) / get_stage_units()
        self._gripper = ParallelGripper(
            end_effector_prim_path=self._end_effector_prim_path,
            joint_prim_names=gripper_dof_names,
            joint_opened_positions=gripper_open_position,
            joint_closed_positions=gripper_closed_position,
            action_deltas=deltas,
        )
        # add Franka into world (important!)
        self.world.scene.add(self)
        
        self.rmp_flow_config = load_supported_motion_policy_config("Franka", "RMPflow")
        self.rmp_flow = RmpFlow(**self.rmp_flow_config)
        self.rmp_flow.set_robot_base_pose(
            self.position, self.orientation
        )
        self.articulation_rmp = ArticulationMotionPolicy(self, self.rmp_flow, 1.0 / 60.0)
        self.articulation_controller = self.get_articulation_controller()
        
        # check whether point is reachable or not
        self.pre_error = 0.0
        self.error_nochange_epoch = 0
        
        # self.debug_tool = VisualCuboid(
        #     prim_path="/World/Debug_cube",
        #     name="debug_cube",
        #     scale=np.array([0.01, 0.01, 0.01]),
        #     color=np.array([1.0, 0.0, 0.0]),
        #     translation=np.array([0.0, 0.0, 0.0]),
        #     orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        #     visible=True,
        # )
        
        # self.debug_tool_2 = VisualCuboid(
        #     prim_path="/World/Debug_cube_1",
        #     name="debug_cube_1",
        #     scale=np.array([0.01, 0.01, 0.01]),
        #     color=np.array([0.0, 1.0, 0.0]),
        #     translation=np.array([0.0, 0.0, 0.0]),
        #     orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        #     visible=True,
        # )
        
        return
    
    def initialize(self, physics_sim_view=None) -> None:
        """[summary]"""
        super().initialize(physics_sim_view)
        # self._end_effector = SingleRigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector = SingleRigidPrim(prim_path=self._prim_path+"/panda_hand", name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )
        self.disable_gravity()
        
        return
    
    def post_reset(self) -> None:
        """[summary]"""
        super().post_reset()
        self._gripper.post_reset()
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[0], mode="position"
        )
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[1], mode="position"
        )
        return
    
    @property
    def end_effector(self) -> SingleRigidPrim:
        """[summary]

        Returns:
            SingleRigidPrim: [description]
        """
        return self._end_effector
    
    @property
    def gripper(self) -> ParallelGripper:
        """[summary]

        Returns:
            ParallelGripper: [description]
        """
        return self._gripper
    
    def open_gripper(self) -> None:
        """[summary]"""
        self.gripper.open()
        for i in range(20):
            self.world.step()
        return
    
    def close_gripper(self) -> None:
        """[summary]"""
        self.gripper.close()
        for i in range(20):
            self.world.step()
        return
    
    def get_cur_ee_pos(self,local_frame:bool=False):
        """
        get current end_effector_position and end_effector orientation
        """
        
        if local_frame:
            position, orientation = self.end_effector.get_local_pose()
        else:
            position, orientation = self.end_effector.get_world_pose()

        return position, orientation

    def get_cur_eef_pos(self, local_frame: bool = False):
        """
        get current gripper position and end_effector orientation
        """
        pos, ori = self.get_cur_ee_pos(local_frame=local_frame)
        gripper_pos = pos + Rotation(ori, np.array([0.0, 0.0, 0.1]))
        return gripper_pos, ori
    
    def add_obstacle(self, obstacle):
        """
        add obstacle to franka motion
        make franka avoid potential collision smartly
        """
        self.rmp_flow.add_obstacle(obstacle, False)
        for i in range(10):
            self.world.step(render=True)
        return
    
    def Rmpflow_Step_Action(self, position, orientation=None):
        """
        Use RMPflow_controller to move the Franka
        """
        self.world.step(render=True)
        # set end effector target
        self.rmp_flow.set_end_effector_target(
            target_position=position, target_orientation=orientation
        )
        # update obstacle position and get target action
        self.rmp_flow.update_world()
        actions = self.articulation_rmp.get_next_articulation_action()
        # apply actions
        self._articulation_controller.apply_action(actions)
        
    def Rmpflow_Move(self, target_position, target_orientation=np.array([180.0, 0.0, 180.0])):
        """
        Use RMPflow_controller to move the Franka
        """
        target_ee_position = target_position
        if target_orientation is None:
            target_ee_orientation = None
        else:
            target_ee_orientation = euler_angles_to_quat(target_orientation, degrees=True)
        
        while True:
            # get current end effector position
            pos, ori = self.get_cur_ee_pos()
            # get current gripper position
            gripper_pos = pos + Rotation(ori, np.array([0.0, 0.0, 0.1]))
            
            # # debug
            # self.debug_tool.set_world_pose(gripper_pos, ori)
            # for i in range(10):
            #     self.world.step()
            
            # compute distance error
            error = np.linalg.norm(target_position - gripper_pos)
            
            error_gap = abs(error - self.pre_error)
            self.pre_error = error
            
            if error_gap < 1e-4:
                self.error_nochange_epoch += 1
            
            if self.error_nochange_epoch > 100:
                cprint("Single Franka RMPflow controller failed", "red")
                return False
            if error < 0.001:
                cprint("Single Franka RMPflow controller success", "green")
                return True
            
            # print("error: ", error)
            # print("error_gap: ", self.error_nochange_epoch)
            
            self.Rmpflow_Step_Action(target_ee_position, target_ee_orientation)
                
    def Dense_Rmpflow_Move(
        self, 
        target_position, 
        target_orientation=np.array([180.0, 0.0, 180.0]),
        dense_sample_scale=0.02,
    ):
        if target_orientation is not None:
            target_orientation = euler_angles_to_quat(target_orientation, degrees=True)
        
        pos, ori = self.get_cur_ee_pos()
        gripper_pos = pos + Rotation(ori, np.array([0.0, 0.0, 0.1]))
        
        # self.debug_tool.set_world_pose(gripper_pos, ori)
        
        dense_sample_num = int(np.linalg.norm(gripper_pos - target_position) // dense_sample_scale)
        
        interp_pos = dense_trajectory_points_generation(
            start_pos=gripper_pos, 
            end_pos=target_position,
            num_points=dense_sample_num,
        )
        
        cprint("--Single Franka Dense_Rmpflow_Move Begin", "green")
        for i in range(len(interp_pos)):
            print(f"-------step {i}-------")
            for j in range(5):
                self.Rmpflow_Step_Action(interp_pos[i], target_orientation)
                self.world.step()
        cprint("--Single Franka Dense_Rmpflow_Move End", "green")
        return
        
        
        
    
    