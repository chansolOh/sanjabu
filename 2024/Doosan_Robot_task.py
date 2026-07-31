# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
from typing import Optional

import numpy as np
import omni.isaac.core.tasks as tasks
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.motion_generation.lula.kinematics import LulaKinematicsSolver
import torch


# Inheriting from the base class Follow Target
class My_Robot_Task(tasks.BaseTask):
    def __init__(
        self,
        name: str = "Default_name",
        usd_prim_path :str = "/World/Robot",
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.BaseTask.__init__(self, name=name, offset=offset)
        self.usd_prim_path = usd_prim_path
        return

    
    def set_up_scene(self, scene: Scene) -> None:
        """[summary]

        Args:
            scene (Scene): [description]
        """
        super().set_up_scene(scene)
        #scene.add_default_ground_plane(z_position=0)


        self._robot = self.set_robot()
        self.kinematics_solver = self.set_solver()

        scene.add(self._robot)
        self._task_objects[self._robot.name] = self._robot
        self._move_task_objects_to_their_frame()
        return
    
    def set_robot(self) -> SingleManipulator:

        asset_path = "/media/nia/6d737125-0a20-46a5-94bc-b44a6aec1a2e/ochansol/isaac_sim/USD/robots/Doosan_M1013/M1013_robotiq_2F140.usd"
        # asset_path = "/ochansol/isaac_sim/USD/robots/Doosan_M1013/M1013_robotiq_gripper.usd"
        # asset_path = "/ochansol/isaac_sim/USD/robots/Doosan_M1013/M1013_origin.usd"

        add_reference_to_stage(usd_path=asset_path, prim_path=self.usd_prim_path)
        gripper = ParallelGripper(
            end_effector_prim_path="/World/Doosan_M1013/robotiq_2F_140/_F_Body",
            # end_effector_prim_path="/World/Doosan_M1013/robotiq_arg2f_base_link",
            joint_prim_names=["Left", "Right"],
            joint_opened_positions=np.array([0, 0]),
            joint_closed_positions=np.array([0.628, -0.628]),
            action_deltas=np.array([-0., 0.]),
        )
        manipulator = SingleManipulator(
            prim_path="/World/Doosan_M1013",
            name="doosan",
            end_effector_prim_name="robotiq_2F_140/_F_Body",
            gripper=gripper,
        )
        joints_default_positions = np.zeros(14)
        # joints_default_positions[6] = 0
        # joints_default_positions[7] = 0
        manipulator.set_joints_default_state(positions=torch.tensor(joints_default_positions, dtype=torch.float32))
        return manipulator


    @property
    def get_robot_name(self):
        return self._robot.name
    



    def set_solver(self):
        kinematics_solver = LulaKinematicsSolver(
                            # robot_description_path="/ochansol/isaac_sim/USD/robots/Doosan_M1013/Doosan_M1013_description.yaml",
                            # urdf_path="/ochansol/isaac_sim/USD/robots/Doosan_M1013/M1013_robotiq_gripper.urdf",
                            robot_description_path="/media/nia/6d737125-0a20-46a5-94bc-b44a6aec1a2e/ochansol/isaac_sim/USD/robots/Doosan_M1013/M1013_origin_description.yaml",
                            urdf_path="/media/nia/6d737125-0a20-46a5-94bc-b44a6aec1a2e/ochansol/isaac_sim/USD/robots/Doosan_M1013/M1013_origin.urdf",
        )

        return kinematics_solver


    def compute_ik(self,
        target_position : Optional[list],
        target_orientation : Optional[list], 
        frame_name : str = None,
        warm_start : np.ndarray = np.array([0.3,0.3,0.3,0.3,0.3,0.3])
    ):
        if frame_name == None : 
            frame_name = self.kinematics_solver.get_all_frame_names()[7]; print(frame_name)
        if type(target_orientation) == list:
            target_orientation = np.array(target_orientation)
        ik = self.kinematics_solver.compute_inverse_kinematics(
            frame_name = frame_name,
            target_position = target_position ,
            target_orientation = euler_angles_to_quat( target_orientation/180*np.pi),
            warm_start=warm_start
            )
        return ik

    def compute_fk(self, 
        frame_name:str, 
        joint_positions : Optional[np.ndarray] 
        ):
        if frame_name == None : 
            frame_name = self.kinematics_solver.get_all_frame_names()[7]; print(frame_name)
        fk = self.kinematics_solver.compute_forward_kinematics(frame_name =frame_name, joint_positions = joint_positions)
        print("fk : ", fk)
        return fk


    def pre_step(self, current_time_step_index, current_time):
        return
    
    def post_reset(self):
        
        return