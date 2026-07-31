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
import json

import carb.settings
settings = carb.settings.get_settings()

from scipy.io import loadmat
import numpy as np

import omni.isaac.core.utils.bounds as bounds_utils



my_world = World(stage_units_in_meters=1.0)
my_world.reset()
stage = omni.usd.get_context().get_stage()

plane = GroundPlane(prim_path="/World/GroundPlane", z_position=0)
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
house_usd = add_reference_to_stage(usd_path="/media/nia/6d737125-0a20-46a5-94bc-b44a6aec1a2e/ochansol/isaac_sim/sanjabu/Home/Home_scene1_2204.usd", 
                                    prim_path="/World/House")
house_prim = Prims.XFormPrim(name ="House", prim_path="/World/House", position = (5.9, -0.8, -0.847), 
                             orientation= rot_utils.euler_angles_to_quat((90,0,0), degrees = True), scale=(0.01,0.01,0.01))


ycb_model_dir_path = "/media/nia/Data/peel3/scan_data_2204/ycb_video_object/"
ycb_model_dir_list = sorted(os.listdir(ycb_model_dir_path))
ycb_model_usd_path_list = [ os.path.join(ycb_model_dir_path, i  + "/editted") for i in ycb_model_dir_list]
ycb_model_usd_list = []
for path in ycb_model_usd_path_list:
    ycb_model_usd_list += [i for i in os.listdir(path) if i.split(".")[-1] in ["usd"]]
 

ann = loadmat("/media/nia/Data/dataset/ycb_video/data/0005/000001-meta.mat")
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
cam_quat = rot_utils.rot_matrix_to_quat(cam_tf[:3,:3])

# cam_tf = Gf.Matrix4d( np.vstack(( ann["rotation_translation_matrix"], np.array([0.0, 0.0, 0.0, 1.0]) )).T )
# cam_position = np.array(cam_tf.ExtractTranslation())
# cam_quat = rot_utils.gf_quat_to_np_array(cam_tf.ExtractRotationQuat())

output_img_size = (640,480)
cam_pixel_size = 0.003
camera = rep.create.camera(position = cam_position, rotation = cam_euler,
    focal_length= intrinsic_mat[0,0]*cam_pixel_size,focus_distance=0, f_stop=0, 
    horizontal_aperture = output_img_size[0]*cam_pixel_size,
clipping_range=(0.0001, 100000))

camera_prim = stage.GetPrimAtPath("/Replicator/Camera_Xform/Camera")
camera_xform_prim = stage.GetPrimAtPath("/Replicator/Camera_Xform")









######object pose set

ycb_usd_prim_dict={}
for i, idx in enumerate(cls_indexes):
    ycb_usd = add_reference_to_stage(usd_path=os.path.join(ycb_model_usd_path_list[idx], ycb_model_usd_list[idx]), 
                                    prim_path="/World/ycb_object/"+ycb_model_usd_list[idx].split(".")[0])
   
    ycb_tf = cam_tf.dot( rot_x).dot(np.vstack(( ycb_usd_poses[i], np.array([0,0,0,1]) )) )
    ycb_position = ycb_tf[:-1,-1]
    ycb_quat = rot_utils.rot_matrix_to_quat(ycb_tf[:3,:3])
    # ycb_tf = Gf.Matrix4d( np.vstack(( ycb_usd_poses[i], np.array([0,0,0,1]) )).T ) #* cam_tf
    # ycb_position = np.array(ycb_tf.ExtractTranslation())
    # ycb_quat = rot_utils.gf_quat_to_np_array(ycb_tf.ExtractRotationQuat())
    # ycb_quat = np.hstack((np.array(ycb_tf.ExtractRotationQuat().real),  np.array(ycb_tf.ExtractRotationQuat().imaginary)  ))

    ycb_prim = Prims.XFormPrim(name =ycb_model_usd_list[idx], prim_path="/World/ycb_object/"+ycb_model_usd_list[idx].split(".")[0], 
                        scale = [1,1,1])
    
    cache = bounds_utils.create_bbox_cache()
    rot_mat = ycb_tf[:3,:3]
    obb = bounds_utils.compute_obb_corners(cache, ycb_prim.prim_path )
    center = np.mean(obb,axis=0)
    
    ycb_obb = rot_mat.dot((obb).T).T +ycb_position

    ycb_prim.set_world_pose(position = ycb_position,
                            orientation = ycb_quat)
    ycb_usd_prim_dict[i] ={
        "xform_prim" : ycb_prim,
        "object_name":ycb_model_usd_list[idx],
        "objct_idx":idx,
        "TF":ycb_tf,
        "position":ycb_position,
        "quat":ycb_quat,
        "obb":ycb_obb,
        }




########bbox debuging
object_obb_arr = np.array([ycb_usd_prim_dict[i]["obb"] for i in ycb_usd_prim_dict])
from omni.isaac.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()
for obj_num in ycb_usd_prim_dict:
    pt1 = object_obb_arr[obj_num].astype(np.float32)
    pt2 = np.vstack((object_obb_arr[obj_num][-1], object_obb_arr[obj_num][:-1])).astype(np.float32)
    draw.draw_lines([carb.Float3(i) for i in pt1] , [carb.Float3(i) for i in pt2], [carb.ColorRgba(1.0,0.0,0.0,1.0)]*8,[5.0]*8     )



##### ransac base plane detection

obb_arr = np.array([ ycb_usd_prim_dict[i]["obb"] for i in ycb_usd_prim_dict]).reshape(-1,3)
index = np.arange(obb_arr.shape[0])
combi_list = []
max_inlier_cnt=0
max_inlier_pt =None
max_inlier_norm =None
dist_sign = 1

for _ in range(500):
    ransac_idx = np.sort(np.random.choice(index,3, replace = False))
    if ransac_idx.tolist() in combi_list:continue
    combi_list.append(ransac_idx.tolist())
    sample_point = obb_arr[ransac_idx]
    v12 = sample_point[1]-sample_point[0]
    v13 = sample_point[2]-sample_point[0]
    norm_vec = np.cross(v12,v13)
    D = -sample_point[0].dot(norm_vec)

    
    idx = np.delete(index,ransac_idx)
    dist_mat = np.abs(obb_arr[idx].dot(norm_vec) + D)/np.sqrt(np.sum(norm_vec**2))
    idx_inlier = np.where(dist_mat<0.005)[0]
    if (len(idx_inlier)+3)/len(index)>=0.25:
        if len(idx_inlier)+ 3>max_inlier_cnt:
            max_inlier_cnt = len(idx_inlier)+3
            max_inlier_pt = obb_arr[ransac_idx]
            sign =  np.sign( obb_arr[idx].dot(norm_vec) + D )
            val,cnts = np.unique(sign, return_counts = True)
            dist_sign = val[cnts.argmax()]
            max_inlier_norm = norm_vec * dist_sign

    # for idx in np.delete(index,ransac_idx):
    #     dist = np.abs(obb_arr[idx].dot(norm_vec) + D) / np.sqrt(np.sum( norm_vec**2 ))
    #     if dist<0.008:
    #         inlier.append(idx)
    # if (len(inlier) + 3)/len(index) >=0.45:
    #     print(inlier)
    #     if len(inlier)+ 3>max_inlier_cnt:
    #         max_inlier_cnt = len(inlier)+3
    #         max_inlier_pt = obb_arr[ransac_idx]
        ###debug
        pt1 = obb_arr[ransac_idx].astype(np.float32)
        pt2 = np.vstack((obb_arr[ransac_idx][-1], obb_arr[ransac_idx][:-1])).astype(np.float32)
        draw.draw_lines([carb.Float3(i) for i in pt1] , [carb.Float3(i) for i in pt2], [carb.ColorRgba(0.0,1.0,0.0,1.0)]*3,[5.0]*3     )
          
        
norm = max_inlier_norm/np.linalg.norm(max_inlier_norm)
# roll = np.arctan(norm[2]/norm[1])
# pitch = np.arctan(-norm[0]/np.sqrt(norm[1]**2+norm[2]**2))
# yaw = 0
# norm_rot_mat = rot_utils.euler_to_rot_matrix(np.array([roll,pitch,yaw]))

roll  = -np.arctan(norm[1]/np.sqrt(norm[0]**2 + norm[2]**2))

pitch = np.arccos(norm[2]/np.sqrt(norm[0]**2 + norm[2]**2))
rot_x = np.array([[1,0,0],
                  [0,np.cos(roll),-np.sin(roll)],
                  [0,np.sin(roll),np.cos(roll)]])
rot_y = np.array([[np.cos(pitch),0,np.sin(pitch)],
                  [0,1,0],
                  [-np.sin(pitch),0,np.cos(pitch)]])
norm_rot_mat = rot_y.dot(rot_x)
norm_tf = np.vstack(( np.hstack((norm_rot_mat,np.mean(max_inlier_pt,axis=0)[:,None] )), np.array([[0,0,0,1]]) ))


# draw.draw_lines([carb.Float3(norm)] , [carb.Float3(0.0,0.0,0.0)], [carb.ColorRgba(0.0,0.0,1.0,1.0)],[5.0]     ) 
# draw.draw_lines([carb.Float3(np.linalg.inv(rot_y.dot(rot_x)).dot(norm))] , [carb.Float3(0.0,0.0,0.0)], [carb.ColorRgba(0.0,1.0,1.0,1.0)],[5.0]     ) 

# draw.draw_lines([carb.Float3(rot_x.dot(norm))] , [carb.Float3(0.0,0.0,0.0)], [carb.ColorRgba(0.0,1.0,1.0,1.0)],[5.0]     ) 
print("norm_vec", max_inlier_norm/np.linalg.norm(max_inlier_norm))
# if max_inlier_pt==None: raise ValueError






######## render set

render_product = rep.create.render_product(camera, output_img_size)
basic_writer = rep.WriterRegistry.get("BasicWriter")
basic_writer.initialize(
    output_dir=f"~/ochansol/isaac/sanjabu/rendering_image",
    rgb                     =True,
    bounding_box_2d_loose   =False,
    bounding_box_2d_tight   =False,
    bounding_box_3d         =False,
    distance_to_camera      =False,
    distance_to_image_plane =False,
    instance_segmentation   =False,
    normals                 =False,
    semantic_segmentation   =False,
)

# # Attach render_product to the writer
basic_writer.attach([render_product])



my_world.reset()
my_world.stop()


base_tf = np.linalg.inv(ycb_usd_prim_dict[0]["TF"])
base_tf = np.linalg.inv(norm_tf)
for i in ycb_usd_prim_dict:
    obj = ycb_usd_prim_dict[i]
    new_tf = base_tf.dot(obj["TF"])
    
    ycb_position = new_tf[:-1,-1]
    ycb_quat = rot_utils.rot_matrix_to_quat(new_tf[:3,:3])
    obj["xform_prim"].set_world_pose(position = ycb_position, orientation=ycb_quat)
    
cam_new_tf = base_tf.dot(cam_tf)






settings.set("/rtx/rendermode", "PathTracing")
settings.set("/rtx/pathtracing/spp", 1) 
settings.set("/rtx/pathtracing/totalSpp", 256)
settings.set("/rtx/pathtracing/maxBounces", 12)
settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 12)
camera_prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(0.0,0.0,0.0))


camera_xform_prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f([i for i in rot_utils.matrix_to_euler_angles(cam_new_tf[:3,:3],degrees=True ) ] ) )
camera_xform_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f([i for i in cam_new_tf[:-1,-1]] ))

rep.orchestrator.step()
rep.orchestrator.step()
rep.orchestrator.step()
# basic_writer.write()





while simulation_app.is_running():
    my_world.step(render=True)


    # if cnt==200:
    #     my_world.play()

#     if my_world.is_playing():

#         if my_world.current_time_step_index <= 1:
#             my_world.reset()



simulation_app.close()