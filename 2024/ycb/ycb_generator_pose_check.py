from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
# import Robot_task_suction_2cup as Robot_task

import omni.isaac.core.prims as Prims
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import omni.isaac.core.utils.rotations as rot_utils
import omni.isaac.core.utils.prims as prim_utils
import omni
from omni.physx.scripts import utils
from pxr import Usd, UsdShade, Sdf, Gf, UsdGeom
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
from sanjabu import my_rep



my_world = World(stage_units_in_meters=1.0)
my_world.reset()
stage = omni.usd.get_context().get_stage()

light_1 = prim_utils.create_prim(
    "/World/Light_1",
    "DiskLight",
    position=np.array([2.0, 2.0, 100.0]),
    attributes={
        "inputs:radius": 0.5,
        "inputs:intensity": 5e4,
       "inputs:color": (255, 250, 245),
       "inputs:exposure" : 4,
    }
)

dir_num ="/0003/"
with open("/home/nia/ochansol/isaac/sanjabu/ycb_isaac_conf.json",'r') as f:
    usd_conf = json.load(f)
    
    
house_usd = add_reference_to_stage(usd_path=usd_conf[dir_num]["usd_path"], 
                                    prim_path="/World/House")

if len(usd_conf[dir_num]["usd_pose"]["orientation"])==3:

    house_prim = Prims.XFormPrim(name ="House", prim_path="/World/House",
                                position = usd_conf[dir_num]["usd_pose"]["position"], 
                                orientation = rot_utils.euler_angles_to_quat( usd_conf[dir_num]["usd_pose"]["orientation"], degrees = True), 
                                # orientation =  usd_conf[dir_num]["usd_pose"]["orientation"],
                                scale = usd_conf[dir_num]["usd_pose"]["scale"] )
else:
    house_prim = Prims.XFormPrim(name ="House", prim_path="/World/House",
                                position = usd_conf[dir_num]["usd_pose"]["position"], 
                                #orientation = rot_utils.euler_angles_to_quat( usd_conf[dir_num]["usd_pose"]["orientation"], degrees = True), 
                                orientation =  usd_conf[dir_num]["usd_pose"]["orientation"],
                                scale = usd_conf[dir_num]["usd_pose"]["scale"] )





# house_usd_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(house_quat[0],Gf.Vec3d([i for i in house_quat[1:]]) ))

# house_usd = add_reference_to_stage(usd_path="/media/nia/6d737125-0a20-46a5-94bc-b44a6aec1a2e/ochansol/isaac_sim/sanjabu/Home/Home_scene1_2204_flatten.usd", 
#                                     prim_path="/World/House")
# house_prim = Prims.XFormPrim(name ="House", prim_path="/World/House", 
#                              position =[6.6, -0.1, 0.889], 
#                              orientation = rot_utils.euler_angles_to_quat([-90,0,0], degrees = True), 
#                              scale = [0.01, 0.01, 0.01] )


ycb_model_dir_path = "/media/nia/Data/peel3/scan_data_2204/ycb_video_object_final/"
ycb_model_dir_list = sorted(os.listdir(ycb_model_dir_path))
ycb_model_usd_path_list = [ os.path.join(ycb_model_dir_path, i) for i in ycb_model_dir_list]
ycb_model_usd_list = []

for path in ycb_model_usd_path_list:
    ycb_model_usd_list += [i for i in os.listdir(path) if i.split(".")[-1] in ["usd"]]

dir_path = "/media/nia/Data/dataset/ycb_video/data"+dir_num
ann = loadmat(dir_path+"000001-meta.mat")
cls_indexes = ann["cls_indexes"].T[0]-1  ## class num : 1~21
ycb_usd_poses = ann["poses"].transpose(2,0,1) ## pose shape : (obj num, 3, 4)





######## cam set
intrinsic_mat = ann["intrinsic_matrix"]
th = 180/180*np.pi
rot_x = np.array([[1,0,0,0],
                [0,np.cos(th),-np.sin(th),0],
                [0,np.sin(th),np.cos(th),0],
                [0,0,0,1]])
cam_tf = np.vstack(( ann["rotation_translation_matrix"], np.array([0,0,0,1])))
cam_position = cam_tf[:-1,-1]
cam_euler = rot_utils.matrix_to_euler_angles(cam_tf[:3,:3], degrees = True)


output_img_size = (640,480)
cam_pixel_size = 0.003
camera = rep.create.camera(position = cam_position, rotation = cam_euler,
    focal_length= intrinsic_mat[0,0]*cam_pixel_size,focus_distance=0, f_stop=0, 
    horizontal_aperture = output_img_size[0]*cam_pixel_size,
clipping_range=(0.0001, 100000))

camera2 = rep.create.camera(position = cam_position, rotation = cam_euler,
    focal_length= intrinsic_mat[0,0]*cam_pixel_size,focus_distance=0, f_stop=0, 
    horizontal_aperture = output_img_size[0]*cam_pixel_size,
clipping_range=(0.0001, 100000))

camera_prim = stage.GetPrimAtPath("/Replicator/Camera_Xform/Camera")
camera_xform_prim = stage.GetPrimAtPath("/Replicator/Camera_Xform")


######object pose set

ycb_usd_prim_dict={}
for i, idx in enumerate(cls_indexes):
    ycb_tf = cam_tf.dot( rot_x).dot(np.vstack(( ycb_usd_poses[i], np.array([0,0,0,1]) )) )
    ycb_position = ycb_tf[:-1,-1]
    # ycb_quat = rot_utils.rot_matrix_to_quat(ycb_tf[:3,:3])
    ycb_euler = rot_utils.matrix_to_euler_angles(ycb_tf[:3,:3], degrees = True)
    ycb_rep = my_rep.rep_usd(usd_path = os.path.join(ycb_model_usd_path_list[idx], ycb_model_usd_list[idx]),
                                prim_path = "ycb_object/"+ycb_model_usd_list[idx].split(".")[0],count =1,
                                rigidbody_collider=False,
                                particle_cloth = False)
    
    # with ycb_rep.node:
    #     rep.modify.pose(position = ycb_position, rotation = ycb_euler)
        
    ycb_rep.set_semantic("class",ycb_model_usd_list[idx].split(".")[0] )
    ycb_usd_prim_dict[i] ={
        "rep" : ycb_rep,
        "prim" : ycb_rep.get_prims()[0],
        "object_name":ycb_model_usd_list[idx],
        "objct_idx":idx,
        "TF":ycb_tf,
        }







######## render set

render_product = rep.create.render_product(camera, output_img_size)
output_path = f"/home/nia/ochansol/isaac/sanjabu/dataset"+dir_num
basic_writer = rep.WriterRegistry.get("BasicWriter")
basic_writer.initialize(
    output_dir              =output_path,
    rgb                     =True,
    bounding_box_2d_loose   =False,
    bounding_box_2d_tight   =True,
    bounding_box_3d         =False,
    distance_to_camera      =True,
    distance_to_image_plane =False,
    instance_segmentation   =True,
    normals                 =False,
    semantic_segmentation   =False,
    use_common_output_dir   =True,
)

# # Attach render_product to the writer
basic_writer.attach([render_product])



my_world.reset()
my_world.stop()


base_tf = np.linalg.inv(ycb_usd_prim_dict[0]["TF"])

for i in ycb_usd_prim_dict:
    obj = ycb_usd_prim_dict[i]
    new_tf = base_tf.dot(obj["TF"])
    
    ycb_position = new_tf[:-1,-1]
    ycb_euler = rot_utils.matrix_to_euler_angles(new_tf[:3,:3], degrees = True)
    with obj["rep"].node:
        rep.modify.pose(position = ycb_position, rotation = ycb_euler)
        
    
cam_new_tf = base_tf.dot(cam_tf)


# settings.set("/rtx/rendermode", "PathTracing")
# settings.set("/rtx/pathtracing/spp", 1) 
# settings.set("/rtx/pathtracing/totalSpp", 128)
# settings.set("/rtx/pathtracing/maxBounces", 12)
# settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 12)
# camera_prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(0.0,0.0,0.0))


camera_xform_prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f([i for i in rot_utils.matrix_to_euler_angles(cam_new_tf[:3,:3],degrees=True ) ] ) )
camera_xform_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f([i for i in cam_new_tf[:-1,-1]] ))








while simulation_app.is_running():
    my_world.step(render=True)
    # import pdb;pdb.set_trace()

    # if cnt==200:
    #     my_world.play()

#     if my_world.is_playing():

#         if my_world.current_time_step_index <= 1:
#             my_world.reset()



simulation_app.close()