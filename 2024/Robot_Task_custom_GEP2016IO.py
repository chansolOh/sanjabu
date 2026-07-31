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
from pxr import Gf, PhysxSchema, UsdPhysics
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
from omni.physx.scripts import utils as physx_utils

from omni.isaac.core.materials.physics_material import PhysicsMaterial


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

        self.gripper_grasp_position = np.array([0,0])
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
        self.gripper_height = 0.016
        self.gripper_depth = 0.14
        self.finger_thickness = 0.03
        self.plot = []
        self.grasp_state = False
        self.target_prim_path = ""
        self.render = True
        self.draw = _debug_draw.acquire_debug_draw_interface()
        return
    
    def set_robot(self) -> SingleManipulator:
        self.gripper_model = "Custom_GEP2016IO"
        asset_path = f"/media/nia/6d737125-0a20-46a5-94bc-b44a6aec1a2e/ochansol/isaac_sim/sanjabu/Robot/{self.gripper_model}/{self.gripper_model}.usd"
        # asset_path = "/ochansol/isaac_sim/USD/robots/Doosan_M1013/M1013_robotiq_gripper.usd"
        # asset_path = "/ochansol/isaac_sim/USD/robots/Doosan_M1013/M1013_origin.usd"
        finger_physics_mat_path = f"/World/{self.gripper_model}/Body/GEP2016IO/PhysicsMaterial"
        add_reference_to_stage(usd_path=asset_path, prim_path=self.usd_prim_path)
        gripper = ParallelGripper(
            end_effector_prim_path=f"/World/{self.gripper_model}/Body/GEP2016IO/base",
            # end_effector_prim_path="/World/Doosan_M1013/robotiq_arg2f_base_link",
            joint_prim_names=["joint_L", "joint_R"],
            joint_opened_positions=np.array([-1, 1]),
            joint_closed_positions=self.gripper_grasp_position,
            action_deltas=np.array([-0., 0.]),
        )
        manipulator = SingleManipulator(
            prim_path=f"/World/{self.gripper_model}/Body",
            name="doosan",
            end_effector_prim_name="GEP2016IO/base",
            gripper=gripper,
        )
        
        self.finger_material_prim = self.stage.GetPrimAtPath(finger_physics_mat_path)
        self.joints_default_positions = np.zeros(3)
        self.joints_default_positions[0] = -0.65
        manipulator.set_joints_default_state(positions=torch.tensor(self.joints_default_positions, dtype=torch.float32))
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
    
    def set_finger_material(self,static_friction=0.5, dynamic_friction=0.2, restitution=0):
        self.finger_material_prim.GetAttribute("physics:staticFriction").Set(static_friction)
        self.finger_material_prim.GetAttribute("physics:dynamicFriction").Set(dynamic_friction)
        self.finger_material_prim.GetAttribute("physics:restitution").Set(restitution)

        # self.finger_material_prim.GetAttribute("physxMaterial:frictionCombineMode").Set("min")
    

    @property
    def get_robot_name(self):
        return self._robot.name
    



    def set_solver(self):
        kinematics_solver = LulaKinematicsSolver(
                            # robot_description_path="/ochansol/isaac_sim/USD/robots/Doosan_M1013/Doosan_M1013_description.yaml",
                            # urdf_path="/ochansol/isaac_sim/USD/robots/Doosan_M1013/M1013_robotiq_gripper.urdf",
                            robot_description_path=f"/media/nia/6d737125-0a20-46a5-94bc-b44a6aec1a2e/ochansol/isaac_sim/sanjabu/Robot/{self.gripper_model}/{self.gripper_model}_description.yaml",
                            urdf_path=f"/media/nia/6d737125-0a20-46a5-94bc-b44a6aec1a2e/ochansol/isaac_sim/sanjabu/Robot/{self.gripper_model}/{self.gripper_model}.urdf",
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


    def picking(self, target_points, target_orientation, target_width, target_prim_path):    #### exact offset J6 : 0.118
        return_dict = {
            "success": False,
            "quality": 0,
            "width": 0,
            "last_position": np.array([0,0,0]),
            "LR_contact_position": [None, None]
        }
        self.target_prim_path = target_prim_path
        # target_position[2] = target_position[2] + self.J6_offset +self.palm_depth
        # target_position[2] = np.clip(target_position[2], self.gripper_depth+0.004, 1) - 1
        target_position = target_points.copy()
        target_position[2] = target_position[2] + self.gripper_depth  -1
        self.set_init_pose([target_position[0],target_position[1],0.0], [0,0,target_orientation[2]])
        
        
        self.open(width = target_width, time_out=0.1)
        if not self.action(target_position, th = 0.001,time_out=1, gripper_force_th=100, call_back=self.check_gripper_force) : 
            return return_dict
        self.grasp(time_out = 0.4)
        
        self.grasp_state = self.check_contact_periodic(duration=5, method="mean")
            
        if not self.grasp_state:return return_dict
        self.action(target_position + np.array([0,0,+0.03]), th = 0.02, gripper_force_th=1000)
        self.action(target_position + np.array([0,0,+0.07]), th = 0.02, gripper_force_th=1000)
        self.action(target_position + np.array([0,0,+0.12]), th = 0.02, gripper_force_th=1000)
        self.action(target_position + np.array([0,0,+0.15]), th = 0.005, gripper_force_th=1000, time_out=0.5)
        
        L_contact_position, R_contact_position = self.check_contact()
        self.grasp_state = self.check_contact_periodic(duration=50, method="and")
        if self.grasp_state: 
            return_dict["success"] = True
            return_dict["width"] =  (self._robot.get_joints_state().positions[1] - self.gripper_grasp_position[0]) -\
                                    (self._robot.get_joints_state().positions[2] - self.gripper_grasp_position[1])
            # target_position[2] = target_position[2] - self.gripper_depth +1
            last_z = (list(L_contact_position)[2] + list(R_contact_position)[2])/2
            target_position[2] = last_z 
            return_dict["last_position"] = target_position# + np.array([0,0,+0.15])
            return return_dict
        else : 
            
            return return_dict
        
    def action(self,target_position,  th = 0.008, time_out = 2, gripper_force_th = 100, call_back = None):
        target_z = target_position[2]
        init_time = self.world.current_time

        while True:
            if call_back != None:
                call_back()
                
            self.plot.append(self._robot.get_measured_joint_forces([1,2,3])[2][:3] )
            if self.gripper_force_L>gripper_force_th or self.gripper_force_R>gripper_force_th:
                return False
            self.world.step(render=self.render)
            self._robot.apply_action(ArticulationAction(
                                    joint_indices=[0] ,
                                joint_positions = [target_z],
                                ) )
            joints = self._robot.get_joints_state().positions[0]
            pos_err = np.sqrt(np.sum((target_z -joints)**2))
            if pos_err<th or self.world.current_time - init_time > time_out:
                break
        return True
    

    def grasp(self,time_out=1):
        init_time = self.world.current_time
        while True:
            self.plot.append(self._robot.get_measured_joint_forces([1,2,3])[2][:3] )
            self.world.step(render=self.render)
            self._robot.apply_action(ArticulationAction(
                                    joint_indices=[1,2] ,
                                joint_positions =self.gripper_grasp_position,
                                ) )
        
            if self.world.current_time - init_time > time_out:
                break
        return
    
    def open(self,width, time_out=0.2):
        init_time = self.world.current_time
        while True:
            self.world.step(render=self.render)
            self._robot.apply_action(ArticulationAction(
                                    joint_indices=[1,2] ,
                                joint_positions =self.gripper_grasp_position + np.array([width/2,-width/2]),
                                ) )
            if self.world.current_time - init_time > time_out:
                break
        return
        # self.gripper_width = width
        # self.joints_default_positions[1] = width/2
        # self.joints_default_positions[2] = -width/2
        # self._robot.set_joints_default_state(positions=torch.tensor(self.joints_default_positions, dtype=torch.float32))
        # self.world.reset()
    def check_gripper_force(self):
        self.gripper_force_L,self.gripper_force_R = self._robot.get_measured_joint_forces().round(3)[[2,3],:3 ]
        self.gripper_force_L = np.sqrt(np.sum(self.gripper_force_L**2))
        self.gripper_force_R = np.sqrt(np.sum(self.gripper_force_R**2))
        

    def check_contact(self):
        sensor_L_data = self._contact_sensor_interface.get_contact_sensor_raw_data(self._contact_sensor_path_L)
        sensor_R_data = self._contact_sensor_interface.get_contact_sensor_raw_data(self._contact_sensor_path_R)
        self.L_contact_check = False
        self.R_contact_check = False
        # print(sensor_L_data)
        
        if len(sensor_L_data)>0:
            L_body1 = str(PhysicsSchemaTools.intToSdfPath(int(sensor_L_data[0][2])))
            L_body2 = str(PhysicsSchemaTools.intToSdfPath(int(sensor_L_data[0][3])))
            L_contact_point = sensor_L_data[0][4]
            # self.draw.draw_points([carb.Float3(i) for i in [sensor_L_data[0][4]] ] , 
            #         [carb.ColorRgba(0.0,0.0,1.0,1.0)]*len([sensor_L_data[0][4]]),
            #         [15]*len([sensor_L_data[0][4]])     )
            if L_body2 == self.target_prim_path:
                self.L_contact_check = True
            # print(L_body1, L_body2)
        if len(sensor_R_data)>0:
            R_body1 = str(PhysicsSchemaTools.intToSdfPath(int(sensor_R_data[0][2])))
            R_body2 = str(PhysicsSchemaTools.intToSdfPath(int(sensor_R_data[0][3])))
            R_contact_point = sensor_R_data[0][4]
            if R_body2 == self.target_prim_path:
                self.R_contact_check = True
            # print(R_body1, R_body2)

        if self.L_contact_check and self.R_contact_check:
            self.grasp_state = True
            return [R_contact_point, L_contact_point]
        else:
            self.grasp_state = False
            return [None, None]
    
    def check_contact_periodic(self, duration=5,method="mean"):
        check_cache = []
        for i in range(duration):
            self.world.step(render=self.render)
            self.check_contact()
            check_cache.append(self.grasp_state)
        if method =="mean":
            if check_cache.count(True)>=duration/2:
                return True
        if method == "and":
            if check_cache.count(True)==duration:
                return True
        

        
        
        
    def _on_contact_report_event(self, contact_headers, contact_data):
        self.gripper_force_L,self.gripper_force_R = self._robot.get_measured_joint_forces().round(3)[[2,3],:3 ]
        self.gripper_force_L = np.sqrt(np.sum(self.gripper_force_L**2))
        self.gripper_force_R = np.sqrt(np.sum(self.gripper_force_R**2))
        L_contact_check = False
        R_contact_check = False
        if self.gripper_force_L>0.1 and self.gripper_force_R>0.1:

            for contact_header in contact_headers:         
                collider_1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
                collider_2 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))
                if collider_1 == self.gripper_L_path and collider_2 == self.target_prim_path:
                    L_contact_check = True
                    
                elif collider_2 == self.gripper_L_path and collider_1 == self.target_prim_path:
                    L_contact_check = True
                
                
                if collider_1 == self.gripper_R_path and collider_2 == self.target_prim_path:
                    R_contact_check = True
                elif collider_2 == self.gripper_R_path and collider_1 == self.target_prim_path:
                    R_contact_check = True
                
            if L_contact_check and R_contact_check:
                self.grasp_state = True
            else:
                self.grasp_state = False
                # print("contact failed", L_contact_check, R_contact_check)
        else:
            # print("week force", self.gripper_force_L, self.gripper_force_R)

            self.grasp_state = False

        # print(self.grasp_state)
        # if self.gripper_sensor_L["in_contact"]:
        #     print("sensor_L : ", self.gripper_sensor_L["force"])
        # if self.gripper_sensor_R["in_contact"]:
        #     print("sensor_R : ", self.gripper_sensor_R["force"])
            

        
    def set_contact_sensor(self):
        
        self.gripper_L_path = f"/World/{self.gripper_model}/Body/GEP2016IO/Left"
        self.gripper_R_path = f"/World/{self.gripper_model}/Body/GEP2016IO/Right"
        # gripper_palm_path = "/World/Doosan_M1013/gripper/onrobot_2fg_14/base/_114001_1_STEP_88/NONE_89"
        
        self._contact_sensor_path_L = self.gripper_L_path+"/Contact_Sensor"
        self._contact_sensor_path_R = self.gripper_R_path+"/Contact_Sensor"
        # self._contact_sensor_path_palm = os.path.join(gripper_palm_path, "Contact_Sensor")
        # self._contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)
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
        self._contact_sensor_interface = _sensor.acquire_contact_sensor_interface()
        
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
    def set_filter_target_gripper(self, target_prim_path):
        physx_utils.addPairFilter(self.stage, [target_prim_path, self.gripper_L_path])
        physx_utils.addPairFilter(self.stage, [target_prim_path, self.gripper_R_path])
        return

    def pre_step(self, current_time_step_index, current_time):
        # self.check_contact()
        # aa = self._contact_sensor_interface.get_sensor_reading(self._contact_sensor_path_L, use_latest_data = True)
        # print(aa.in_contact)
        return
    
    def post_reset(self):
        self._robot.set_world_pose(position = self.robot_position, orientation = rot_utils.euler_angles_to_quat(self.robot_rotation, degrees = True))
        self.gripper_force_L = 0
        self.gripper_force_R = 0
        self.plot = []
        # print("reset")
        return
    
