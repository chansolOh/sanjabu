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
from pxr import Gf, PhysxSchema
import cs_rep_utils as csr
import omni.isaac.core.utils.rotations as rot_utils
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.debug_draw import _debug_draw
import carb
import omni
from omni.isaac.sensor import _sensor
from omni.physx import get_physx_simulation_interface
from omni.physx.scripts.physicsUtils import *
from omni.isaac.sensor import ContactSensor

# Inheriting from the base class Follow Target
class RobotTask(tasks.BaseTask):
    def __init__(
        self,
        name: str = "Default_name",
        usd_prim_path :str = "/World/Robot",
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.BaseTask.__init__(self, name=name, offset=offset)
        self.usd_prim_path = usd_prim_path
        self.world = SimulationContext()
        self.stage = omni.usd.get_context().get_stage()

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
        self.robot_position = [0,0,0]
        self.robot_rotation = [0,0,0]
        self.gripper_force_L = 0
        self.gripper_force_R = 0
        self.gripper_width = 0.11
        self.gripper_height = 0.014
        self.gripper_depth = 0.10
        self.palm_depth = 0.084
        self.J6_offset = 0.12
        self.finger_thickness = 0.006
        return
    
    def set_robot(self) -> SingleManipulator:

        asset_path = "/media/nia/6d737125-0a20-46a5-94bc-b44a6aec1a2e/ochansol/isaac_sim/USD/robots/Doosan_M1013/M1013_onrobot_2fg14.usd"
        # asset_path = "/ochansol/isaac_sim/USD/robots/Doosan_M1013/M1013_robotiq_gripper.usd"
        # asset_path = "/ochansol/isaac_sim/USD/robots/Doosan_M1013/M1013_origin.usd"

        add_reference_to_stage(usd_path=asset_path, prim_path=self.usd_prim_path)
        gripper = ParallelGripper(
            end_effector_prim_path="/World/Doosan_M1013/gripper/onrobot_2fg_14/base",
            # end_effector_prim_path="/World/Doosan_M1013/robotiq_arg2f_base_link",
            joint_prim_names=["left_joint", "right_joint"],
            joint_opened_positions=np.array([-0.22, 0.22]),
            joint_closed_positions=np.array([0.33, -0.33]),
            action_deltas=np.array([-0., 0.]),
        )
        manipulator = SingleManipulator(
            prim_path="/World/Doosan_M1013",
            name="doosan",
            end_effector_prim_name="gripper/onrobot_2fg_14/base",
            gripper=gripper,
        )
        joints_default_positions = np.zeros(8)
        # joints_default_positions[6] = 0
        # joints_default_positions[7] = 0
        manipulator.set_joints_default_state(positions=torch.tensor(joints_default_positions, dtype=torch.float32))
        return manipulator

    def set_init_pose(self,position, rotation):
        self.robot_position = position
        self.robot_rotation = rotation 
        self._robot.prim.GetAttribute("xformOp:translate").Set(csr.np_to_GfVec3d(np.array(position)))
        if self._robot.prim.HasAttribute("xformOp:rotateXYZ"):
            self._robot.prim.GetAttribute("xformOp:rotateXYZ").Set(csr.np_to_GfVec3d(np.array(rotation)))
        elif self._robot.prim.HasAttribute("xformOp:orient"):
            self._robot.prim.GetAttribute("xformOp:orient").Set(csr.np_to_GfQuatd(rot_utils.euler_angles_to_quat(np.array(rotation), degrees=True )))
        return
    

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
        position, rotation = self.kinematics_solver.compute_forward_kinematics(frame_name =frame_name, joint_positions = joint_positions)
        
        return {
            "position":position,
            "rotation":rot_utils.matrix_to_euler_angles(rotation, degrees = True)
        }


    def picking(self, target_position, target_orientation, frame_name = "J6"):    #### exact offset J6 : 0.118
        target_position[2] = target_position[2] + self.J6_offset +self.palm_depth
        target_position[2] = np.clip(target_position[2], self.gripper_depth+self.J6_offset+self.palm_depth-self.robot_position +0.003, 1)
        self.open(width = 0.01, time_out = 0.3)
        if not self.action(target_position + np.array([0,0,0.15]), target_orientation, frame_name = frame_name, th = 0.005) : return
        if not self.action(target_position, target_orientation, frame_name = frame_name, th = 0.001) : return
        self.grasp(time_out = 0.6)
        self.action(target_position + np.array([0,0,0.20]), target_orientation, frame_name = frame_name, th = 0.01, gripper_force_th=1000)
        
    def action(self,target_position, target_orientation, frame_name = "J6", th = 0.008, time_out = 2, gripper_force_th = 50):
        ik,_ = self.compute_ik(target_position = target_position,
            target_orientation = target_orientation, # x,y,z 순서로 회전
            frame_name = frame_name,
            )
        init_time = self.world.current_time
        while True:
            if self.gripper_force_L>gripper_force_th or self.gripper_force_R>gripper_force_th:
                return False
            self.world.step(render=True)
            self._robot.apply_action(ArticulationAction(
                                    joint_indices=[0,1,2,3,4,5] ,
                                joint_positions = ik,
                                ) )
            joints = self._robot.get_joints_state().positions[:6]
            pos_err = np.sqrt(np.sum((target_position -self.compute_fk(frame_name="J6", joint_positions=joints )["position"] )**2))
            if pos_err<th or self.world.current_time - init_time > time_out:
                break
        return True
            
    def grasp(self,time_out=1):
        target_position = np.array([-0.045,0.045])
        init_time = self.world.current_time
        while True:
            self.world.step(render=True)
            self._robot.apply_action(ArticulationAction(
                                    joint_indices=[6,7] ,
                                joint_positions =target_position,
                                ) )
        
            if self.world.current_time - init_time > time_out:
                break
        return
    
    def open(self,width=0.01,time_out =0.3):
        self._robot.apply_action(ArticulationAction(
                                joint_indices=[6,7] ,
                            joint_positions = np.array([width,-width]),
                            ) )
        return

    def _on_contact_report_event(self, contact_headers, contact_data):
        # for contact_header in contact_headers:         
        #     collider_1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
        #     collider_2 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))
        #     print(collider_1, collider_2)
        
        self.gripper_sensor_L = self.sensor_L.get_current_frame()
        self.gripper_sensor_R = self.sensor_R.get_current_frame()
        self.gripper_force_L = self.gripper_sensor_L["force"]
        self.gripper_force_R = self.gripper_sensor_R["force"]
        if self.gripper_sensor_L["in_contact"]:
            print("sensor_L : ", self.gripper_sensor_L["force"])
        if self.gripper_sensor_R["in_contact"]:
            print("sensor_R : ", self.gripper_sensor_R["force"])
            

        
    def set_contact_sensor(self):
        gripper_L_path = "/World/Doosan_M1013/gripper/onrobot_2fg_14/Left/_114008_1_STEP_26/NONE_15"
        gripper_R_path = "/World/Doosan_M1013/gripper/onrobot_2fg_14/Right/_114008_1_STEP_14/NONE_15"
        # gripper_palm_path = "/World/Doosan_M1013/gripper/onrobot_2fg_14/base/_114001_1_STEP_88/NONE_89"
        
        self._contact_sensor_path_L = os.path.join(gripper_L_path, "Contact_Sensor")
        self._contact_sensor_path_R = os.path.join(gripper_R_path, "Contact_Sensor")
        # self._contact_sensor_path_palm = os.path.join(gripper_palm_path, "Contact_Sensor")
        self._contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)
        self.sensor_L = ContactSensor(
            prim_path=self._contact_sensor_path_L,
            name="Contact_Sensor",
            frequency=120,
            translation=np.array([0, 0, 0]),
            min_threshold=0,
            max_threshold=10000000,
            radius=1
            )
        self.sensor_R = ContactSensor(
            prim_path=self._contact_sensor_path_R,
            name="Contact_Sensor",
            frequency=120,
            translation=np.array([0, 0, 0]),
            min_threshold=0,
            max_threshold=10000000,
            radius=1
            )
        # self.stage.GetPrimAtPath(gripper_palm_path + "/Mesh_67").GetAttribute("physxCollision:contactOffset").Set(0.005)
        # self.stage.GetPrimAtPath(gripper_palm_path + "/Mesh_67").GetAttribute("physxCollision:restOffset").Set(0.000000001)
        # self.sensor_palm = ContactSensor(
        #     prim_path=self._contact_sensor_path_palm,
        #     name="Contact_Sensor",
        #     frequency=120,
        #     translation=np.array([0, 0, 0]),
        #     min_threshold=0,
        #     max_threshold=10000000,
        #     radius=1
        #     )

    def pre_step(self, current_time_step_index, current_time):
        # aa = self._contact_sensor_interface.get_sensor_reading(self._contact_sensor_path_L, use_latest_data = True)
        # print(aa.in_contact)
        return
    
    def post_reset(self):
        self._robot.set_world_pose(position = self.robot_position, orientation = rot_utils.euler_angles_to_quat(self.robot_rotation, degrees = True))
        self.gripper_force_L = 0
        self.gripper_force_R = 0
        print("reset")
        return
    
