from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
import numpy as np
# import Robot_task_suction_2cup as Robot_task

import omni.isaac.core.prims as Prims
import omni.isaac.core.utils.rotations as rot_utils
import omni.isaac.core.utils.prims as prim_utils
import omni
from omni.physx.scripts import utils
from pxr import Usd, UsdShade, Sdf, Gf, UsdGeom, PhysxSchema, UsdPhysics
from omni.isaac.core.objects import GroundPlane, DynamicCuboid
from omni.physx.scripts import utils as physx_utils

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.kit.asset_converter")
# enable_extension("omni.kit.renderer.iray")

import omni.kit.asset_converter
import sys
import asyncio
import omni.replicator.core as rep

import omni.graph.core as og
import omni.kit.commands
import os
import shutil
import json

import carb.settings
settings = carb.settings.get_settings()



from scipy.io import loadmat
import numpy as np

import omni.isaac.core.utils.bounds as bounds_utils
from omni.isaac.debug_draw import _debug_draw
import scan_rep
import Robot_Task_custom as Robot_Task
from omni.physx import get_physx_interface, get_physx_simulation_interface
from omni.physx import get_physx_scene_query_interface
from omni.physx.scripts.physicsUtils import *
import cs_utils as cs
import cs_rep_utils as csr
from PIL import Image
import ast
import matplotlib.pyplot as plt
import time

import argparse
parser = argparse.ArgumentParser(description='파라미터 예시')
parser.add_argument('--num', type=int, default =2 ,help='directory')
args = parser.parse_args()




print("world_set")
my_world = World(stage_units_in_meters=1.0,
                 physics_dt  = 0.001,
                 rendering_dt = 0.008)
stage = omni.usd.get_context().get_stage()
robot_task = Robot_Task.RobotTask(
    name="robot_task",
    usd_prim_path="/World/Custom_onrobot",)
my_world.add_task(robot_task)
my_world.reset()


print("robot_set")
robot = robot_task._robot
robot_task.set_init_pose(position =[0.0,0.0,0.0], rotation = [0.0,0.0,0.0])
robot_tf = np.array(csr.find_parents_tf(robot.prim, include_self=True)).T
robot_task.set_contact_sensor()
robot_task.set_finger_material(static_friction=0.55, dynamic_friction=0.25, restitution=0.0)


print("env_set")
plane_object = GroundPlane(prim_path = "/World/ground_plane",
                           name = "ground_plane",
                           scale = [1,1,1],
                           z_position = 0,
                           )
plane_col_prim = stage.GetPrimAtPath(os.path.join(plane_object.prim_path,"collisionPlane"))
plane_prim = stage.GetPrimAtPath(os.path.join(plane_object.prim_path,"geom"))
plane_col_prim.GetAttribute('physics:collisionEnabled').Set(False)

robot_task.set_filter_target_gripper(plane_object.prim_path)

light_1 = prim_utils.create_prim(
    "/World/Light_1",
    "DiskLight",
    position=np.array([2.0, 2.0, 100.0]),
    attributes={
        "inputs:radius": 0.5,
        "inputs:intensity": 5e4,
       "inputs:color": (255, 250, 245),
       "inputs:exposure" : 2,
    }
)






print("config_set")
########  load config
# scene_num = args.num
scene_num = 0
root_path = "/home/nia/ochansol/isaac/sanjabu/data/scene/Home/"
grasp_ann_path = os.path.join(root_path,"grasp")

with open( os.path.join(root_path,"conf",f"{scene_num:04d}"+".json"), 'r') as f:
    config= json.load(f)
obj_conf = config["objects"]
cam_conf = config["cameras"]
physx_conf = config["physics_scene"]
top_cam_config = [ i for i in cam_conf if i["name"] == "top_view_camera"][0]
side_cam_config = [ i for i in cam_conf if i["name"] == "side_view_camera"][0]

# depth_img_cam00 = np.load(os.path.join(root_path,"depth","cam_00",f"{scene_num:04d}"+".npy"))
# rgb_img_cam00 = Image.open(os.path.join(root_path,"rgb","cam_00",f"{scene_num:04d}"+".png"))
# inst_img_cam00 = np.array(Image.open(os.path.join(root_path,"inst_seg","cam_00",f"{scene_num:04d}"+".png")))
# with open( os.path.join(root_path,"inst_seg","cam_00","semantics_mapping_"+f"{scene_num:04d}"+".json"), 'r') as f:
#     inst_label_cam00= json.load(f)


print("cam_set")
################ cam set

top_view_camera = rep.create.camera(
    position            = np.array(top_cam_config["cam_poses"])[:3,3],
    rotation            = rot_utils.matrix_to_euler_angles(np.array(top_cam_config["cam_poses"]), degrees=True ),
    focal_length        = top_cam_config["focal_length_isaac"], 
    focus_distance      = top_cam_config["focus_distance"], 
    f_stop              = top_cam_config["f_stop"], 
    horizontal_aperture = top_cam_config["horizontal_aperture"],
    clipping_range      = tuple(top_cam_config["clipping_range"]))

side_view_camera = rep.create.camera(
    position            = np.array(side_cam_config["cam_poses"])[:3,3],
    rotation            = rot_utils.matrix_to_euler_angles(np.array(side_cam_config["cam_poses"]), degrees=True ),
    focal_length        = side_cam_config["focal_length_isaac"], 
    focus_distance      = side_cam_config["focus_distance"], 
    f_stop              = side_cam_config["f_stop"], 
    horizontal_aperture = side_cam_config["horizontal_aperture"],
    clipping_range      = tuple(side_cam_config["clipping_range"]))



# render_product = rep.create.render_product(top_view_camera, top_cam_config["output_size"])
# depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
# depth_annotator.attach([render_product])
# rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
# rgb_annotator.attach([render_product])





# print("physx_set")
# stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:enableGPUDynamics').Set(True)
# stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:broadphaseType').Set("GPU")
# stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:collisionSystem').Set("PCM") 
# stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:gpuTotalAggregatePairsCapacity').Set(20000)
# stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:gpuFoundLostAggregatePairsCapacity').Set(20000) 

# stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:timeStepsPerSecond").Set(350)
stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:minPositionIterationCount").Set(150) 
stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:minVelocityIterationCount").Set(10)


print("obj_set")
######object pose set
obj_rep_list = []
for obj in obj_conf:

    path = obj["usd_path"]
    path = "/media/nia/Data/peel3"+path[20:]
    scan_obj = scan_rep.Scan_Rep(usd_path = path,
                                          class_name = obj['class'])    
    # scan_obj = scan_rep.Scan_Rep(usd_path = obj['usd_path'],
    #                                       class_name = obj['class'])
    obj_rep_list.append(scan_obj)
    
    scan_obj.set_pose(position = obj["translate"], rotation = rot_utils.quat_to_euler_angles(obj["orient"], degrees=True) )
    scan_obj.set_scale(scale = obj["scale"])
    scan_obj.set_rigidbody_collider()
    # scan_obj.set_contact_sensor()
    
    # scan_obj.set_physics_material(
    #     dynamic_friction=0.1,
    #     static_friction=0.3,
    #     restitution=0.0
    # )
    scan_obj.set_physics_material(
    )
    print("scan_obj ready : ",obj['class'])

# X = (ds_x-cx)/focal_length*depth_img[ds_y,ds_x].flatten()
# Y = -(ds_y-cy)/focal_length*depth_img[ds_y,ds_x].flatten()

my_world.reset()


csr.set_cam_zero_rotate(top_view_camera)
csr.set_cam_zero_rotate(side_view_camera)







with open(os.path.join(grasp_ann_path,f"{scene_num:04d}"+".json"), 'r') as f:
    grasp_output_dict = json.load(f)


        
start_time = time.time()

draw = _debug_draw.acquire_debug_draw_interface()

from pynput import keyboard
key_state=""
def on_press(key):
    global key_state
    try:
        if key.char == 'r':
            key_state = "r"
        elif key.char == 'n':
            key_state = "n"
        elif key.char == 'p':
            key_state = "p"
        elif key.char == 'm':
            key_state = "m"
    except AttributeError:
        pass
listener = keyboard.Listener(on_press=on_press)
listener.start()

for key in grasp_output_dict.keys():
    cnt=0
    while True:
        try:
            print("cnt : ",cnt)
            grasp = grasp_output_dict[key][cnt]
        except:
            break
        my_world.reset()
        target_conf = grasp["isaac_env"]
        target_points = target_conf["target_points"]
        target_orientation = target_conf["target_orientation"]
        target_width = target_conf["target_width"]
        target_prim_path = target_conf["target_prim_path"]
        grasp_success_dict = robot_task.picking(target_points =target_points, 
                                        target_orientation = target_orientation, 
                                        target_width = target_width,
                                        target_prim_path= target_prim_path)
        print("grasp_success : ",grasp_success_dict["success"])
        # print(grasp_success_dict)
        
        cnt+=1
        while True:
            my_world.step(render=True)
            if key_state == "r":
                cnt-=1
                key_state = ""
                break
            if key_state == "n":
                key_state = ""
                break
            if key_state == "p":
                key_state = ""
                cnt-=2
                break
            if key_state == "m":
                cnt=0
                break
        if key_state == "m":
            key_state = ""
            break
        # if not my_world.is_playing():
        #     break
    



        
simulation_app.close()

