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
import Robot_Task_custom_GEP2016IO as Robot_Task
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
parser.add_argument('--scene_num', type=int, default =0 ,help='directory')
args = parser.parse_args()




print("world_set")
my_world = World(stage_units_in_meters=1.0,
                 physics_dt  = 0.001,
                 rendering_dt = 0.008)
stage = omni.usd.get_context().get_stage()
robot_task = Robot_Task.RobotTask(
    name="robot_task",
    usd_prim_path="/World/Custom_GEP2016IO",)
my_world.add_task(robot_task)
my_world.reset()


print("robot_set")
robot = robot_task._robot
robot_task.set_init_pose(position =[0.0,0.0,0.0], rotation = [0.0,0.0,0.0])
robot_tf = np.array(csr.find_parents_tf(robot.prim, include_self=True)).T
robot_task.set_contact_sensor()
# robot_task.set_finger_material(self,static_friction=0.5, dynamic_friction=0.2, restitution=0)

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
# plane_object = scan_rep.Scan_Rep(usd_path = "/media/nia/6d737125-0a20-46a5-94bc-b44a6aec1a2e/ochansol/isaac_sim/USD/etc_assets/plane_object.usd",
#                                  class_name = "plane",
#                                  prim_path = "plane",
#                                  visible=False,
#                                  scale = [2,2,2]
#                                  )
# robot_task.set_filter_target_gripper(plane_object.prim.GetPrimPath())
# plane_col_prim.GetAttribute('physxCollision:contactOffset').Set(0.000001)
# plane_col_prim.GetAttribute('physxCollision:restOffset').Set(0)

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
scene_num = args.scene_num
# scene_num = 0
root_path = "/home/nia/ochansol/isaac/sanjabu/data/scene_bak_20241128/Home/"
grasp_ann_path = os.path.join(root_path,"grasp")
if not os.path.exists(grasp_ann_path):
    os.makedirs(grasp_ann_path)
with open( os.path.join(root_path,"conf",f"{scene_num:04d}"+".json"), 'r') as f:
    config= json.load(f)
obj_conf = config["objects"]
cam_conf = config["cameras"]
physx_conf = config["physics_scene"]
top_cam_config = [ i for i in cam_conf if i["name"] == "top_view_camera"][0]
side_cam_config = [ i for i in cam_conf if i["name"] == "side_view_camera"][0]

depth_img_cam00 = np.load(os.path.join(root_path,"depth","cam_00",f"{scene_num:04d}"+".npy"))
rgb_img_cam00 = Image.open(os.path.join(root_path,"rgb","cam_00",f"{scene_num:04d}"+".png"))
inst_img_cam00 = np.array(Image.open(os.path.join(root_path,"inst_seg","cam_00",f"{scene_num:04d}"+".png")))
with open( os.path.join(root_path,"inst_seg","cam_00","semantics_mapping_"+f"{scene_num:04d}"+".json"), 'r') as f:
    inst_label_cam00= json.load(f)


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



render_product = rep.create.render_product(top_view_camera, top_cam_config["output_size"])
depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
depth_annotator.attach([render_product])
rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach([render_product])





# print("physx_set")
# stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:enableGPUDynamics').Set(True)
# stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:broadphaseType').Set("GPU")
# stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:collisionSystem').Set("PCM") 
# stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:gpuTotalAggregatePairsCapacity').Set(20000)
# stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:gpuFoundLostAggregatePairsCapacity').Set(20000) 

# stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:timeStepsPerSecond").Set(350)
stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:minPositionIterationCount").Set(200) 
stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:minVelocityIterationCount").Set(50)


print("obj_set")
######object pose set
obj_rep_list = []
for obj in obj_conf:
    
    ###### specified path for each PC
    path = obj["usd_path"]
    path = "/media/nia/Data/peel3"+path[20:]
    scan_obj = scan_rep.Scan_Rep(usd_path = path,
                                          class_name = obj['class'])
    #########
    
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



print("picking_ready")
########### picking point detection

cx,cy = np.array(top_cam_config["intrinsic_isaac"])[0,2], np.array(top_cam_config["intrinsic_isaac"])[1,2]
focal_length = np.array(top_cam_config["intrinsic_isaac"])[0,0]

cam_tf = np.array(top_cam_config["cam_poses"])





# plt.scatter(X_all,Y_all, c = depth_img_cam00[IDX[0],IDX[1]], cmap = "jet", s=0.1)
# plt.show()


def align_bbox(bbox):
    y_min = np.min(bbox[:,1],axis=0)
    y_min_idx = np.argwhere(bbox[:,1]==y_min)
    if y_min_idx.shape[0]>=2:
        return np.roll(bbox,-max(y_min_idx),axis=0)
    else:
        return np.roll(bbox,-y_min_idx[0],axis=0)

def lin_func(bbox,points):# bbox=2x2, points = 2xN
    x1,y1 = bbox[0]
    x2,y2 = bbox[1]
    if x2-x1 == 0:
        return points[0] - x1
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1    
    return a*points[0] - points[1] + b 

def select_point_in_bbox(bbox,points): # bbox = 4x2, points = 2xN
    aligned_bbox = align_bbox(bbox)
    p1 = np.where(lin_func(aligned_bbox[[3,2]],points)>=0)
    p2 = np.where(lin_func(aligned_bbox[[2,1]],points)>=0)
    p3 = np.where(lin_func(aligned_bbox[[1,0]],points)<=0)
    p4 = np.where(lin_func(aligned_bbox[[0,3]],points)<=0)

    filtered_idx = np.intersect1d(np.intersect1d(p1,p2),np.intersect1d(p3,p4))
    return filtered_idx

def select_points(bboxes,points) : # bboxes = bboxes x 3 x 13, points = 3 x N

    idx = []
    for bbox in bboxes:
        tmp = [] # 3x2xN
        tmp.append(select_point_in_bbox(bbox[:2,  :4].T, points[:2]))
        tmp.append(select_point_in_bbox(bbox[:2, 4:8].T, points[:2]))
        tmp.append(select_point_in_bbox(bbox[:2, 8:12].T,points[:2]))
        idx.append(tmp)

    return idx #boxes x 3 x 2 x N

print("picking_set")




###### point cloud sampling
IDX = np.argwhere(depth_img_cam00==depth_img_cam00).T
Z_all = depth_img_cam00[IDX[0],IDX[1]]
sample_idx = np.random.choice(len(Z_all),int(len(Z_all)*0.08),replace=False)
Z_sample = Z_all[sample_idx]
X_sample = (IDX[1][sample_idx]-cx)/focal_length*Z_sample
Y_sample = (IDX[0][sample_idx]-cy)/focal_length*Z_sample
PCD = np.array([X_sample,Y_sample,Z_sample,np.ones_like(X_sample)])
cam_to_pt_all = np.array([X_sample,Y_sample,Z_sample,np.ones_like(X_sample)])
PCD = np.array(cam_tf).dot(cs.rot_x(180)).dot(cam_to_pt_all)[:3]


###### grasp point sampling
labeled_depth_dict = {}
for obj in obj_rep_list:
    idx_part = None
    for color_key in inst_label_cam00.keys():
        if obj.class_name == inst_label_cam00[color_key]["class"]:
            color_arr = np.array(ast.literal_eval(color_key))

            idx_part = np.argwhere( (inst_img_cam00[IDX[0], IDX[1], 0] == color_arr[0]) &\
                                    (inst_img_cam00[IDX[0], IDX[1], 1] == color_arr[1]) &\
                                    (inst_img_cam00[IDX[0], IDX[1], 2] == color_arr[2]) &\
                                    (inst_img_cam00[IDX[0], IDX[1], 3] == color_arr[3])).T
    if idx_part is None:
        continue
    labeled_depth_dict[obj.class_name] = idx_part
    


if os.path.exists(os.path.join(grasp_ann_path,f"{scene_num:04d}"+".json")):
    with open(os.path.join(grasp_ann_path,f"{scene_num:04d}"+".json"), 'r') as f:
        grasp_output_dict = json.load(f)
        if grasp_output_dict.keys() == []:
            for obj in obj_rep_list:
                grasp_output_dict[obj.class_name] = []
else:
    with open(os.path.join(grasp_ann_path,f"{scene_num:04d}"+".json"), 'w') as f:
        grasp_output_dict = {}
        for obj in obj_rep_list:
            grasp_output_dict[obj.class_name] = []
        json.dump(grasp_output_dict,f, indent = 4)
  
start_time = time.time()
all_obj_idx = np.hstack([labeled_depth_dict[key] for key in labeled_depth_dict.keys()])[0]   ##### x,y 
draw = _debug_draw.acquire_debug_draw_interface()
for obj in obj_rep_list:
    draw.clear_points()
    print("####################### obj : ",obj.class_name)
    idx_part= labeled_depth_dict[obj.class_name][0]  ##### shape = 1 * idx 
    
    sample_num = 2000
    ##### distance based sampling in IMG
    idx = idx_part[ np.random.choice(np.arange(len(idx_part)),sample_num if len(idx_part)>sample_num else len(idx_part) )   ]
    sampled_idx = IDX[:,idx]
    
    idx_dist = np.tile(sampled_idx[:,None,:],(1,len(idx),1)) - np.tile(sampled_idx[...,None],(1,1,len(idx)))
    idx_dist = np.sqrt(np.sum(idx_dist**2,axis=0))
    dist_th = 50
    cnt = 0
    while len(idx_dist)>cnt:
        min_idx = np.argwhere((idx_dist[cnt]<=dist_th) & (idx_dist[cnt]!=0))
        sampled_idx = np.delete(sampled_idx, min_idx, axis=1)
        idx_dist = np.delete(idx_dist, min_idx, axis=0)
        idx_dist = np.delete(idx_dist, min_idx, axis=1)
        cnt+=1
    print("sample_ num : ",len(sampled_idx.T))
    ###### convert to world coordinate
    temp_depth = depth_img_cam00[sampled_idx[0],sampled_idx[1]]
    X = (sampled_idx[1]-cx)/focal_length*temp_depth
    Y = (sampled_idx[0]-cy)/focal_length*temp_depth
    cam_to_grasp = np.array([X,Y,temp_depth,np.ones_like(X)])
    # cam_to_grasp = np.hstack((cam_to_grasp,np.array([[0,0,-1.2,1]]).T ))
    world_to_grasp = np.array(cam_tf).dot(cs.rot_x(180)).dot(cam_to_grasp)[:3].T
            
    gripper_width_list = [0.15,0.11,0.05,]#0.15,0.11,0.05
    for gripper_width in gripper_width_list:
        print("gripper_width : ",gripper_width)
        gripper_finger_bbox = np.array([[ robot_task.gripper_height/2, -gripper_width/2 -  robot_task.finger_thickness],
                                        [ robot_task.gripper_height/2, -gripper_width/2 ],
                                        [-robot_task.gripper_height/2, -gripper_width/2 ],
                                        [-robot_task.gripper_height/2, -gripper_width/2 -  robot_task.finger_thickness],
                                        [ robot_task.gripper_height/2,  gripper_width/2 ],
                                        [ robot_task.gripper_height/2,  gripper_width/2 +  robot_task.finger_thickness],
                                        [-robot_task.gripper_height/2,  gripper_width/2 +  robot_task.finger_thickness],
                                        [-robot_task.gripper_height/2,  gripper_width/2 ]])
        gripper_width_bbox = np.array([[ robot_task.gripper_height/2, -gripper_width/2],
                                    [ robot_task.gripper_height/2,  gripper_width/2],
                                    [-robot_task.gripper_height/2,  gripper_width/2],
                                    [-robot_task.gripper_height/2, -gripper_width/2],]) 
        gripper_center_point = np.array([[ 0,0 ]])
        gripper_bbox = np.vstack((gripper_finger_bbox,gripper_width_bbox,gripper_center_point)).T
        gripper_bbox = np.vstack((gripper_bbox,np.zeros_like(gripper_bbox[0])))

        #### point in bbox
        for rot_deg in range(180,0,-30):
            print("gripper_rot : ",rot_deg)
            gripper_bbox_3d = cs.rot_z(rot_deg).dot(np.vstack((gripper_bbox,np.ones_like(gripper_bbox[0]))))[:3]
            gripper_bbox_3d = np.tile(world_to_grasp[...,None], (1,1,gripper_bbox_3d.shape[1])) + np.tile(gripper_bbox_3d[None,...],(len(world_to_grasp),1,1))
            # if world_to_grasp[0][2]<0:
            #     import pdb;pdb.set_trace()
            X_all = (IDX[1,all_obj_idx]-cx)/focal_length*depth_img_cam00[IDX[0,all_obj_idx],IDX[1,all_obj_idx]]
            Y_all = (IDX[0,all_obj_idx]-cy)/focal_length*depth_img_cam00[IDX[0,all_obj_idx],IDX[1,all_obj_idx]]
            cam_to_obj_part = np.array([X_all,Y_all,depth_img_cam00[IDX[0,all_obj_idx],IDX[1,all_obj_idx]],np.ones_like(X_all)])
            world_to_obj_part = np.array(cam_tf).dot(cs.rot_x(180)).dot(cam_to_obj_part)[:3]
            point_in_bboxes_idx = select_points(gripper_bbox_3d,world_to_obj_part)
            for bbox_num in range(len(gripper_bbox_3d)):
                finger_max_depth = np.max([0 if len(point_in_bboxes_idx[bbox_num][0])==0 else world_to_obj_part[2][point_in_bboxes_idx[bbox_num][0]].max(),  
                                            0 if len(point_in_bboxes_idx[bbox_num][1])==0 else world_to_obj_part[2][point_in_bboxes_idx[bbox_num][1]].max()])
                palm_max_depth   = np.max( 0 if len(point_in_bboxes_idx[bbox_num][2])==0 else world_to_obj_part[2][point_in_bboxes_idx[bbox_num][2]].max())
                
                
                ########### visualization
                ##############
                crit = palm_max_depth - finger_max_depth
                margin = 0.003 # default = 0.005
                if crit<0.008:
                    continue
                    
                if crit>robot_task.gripper_depth/3*2:
                    target_z = palm_max_depth + margin - robot_task.gripper_depth/3*2
                else:
                    target_z = finger_max_depth + margin
                
                world_points = world_to_grasp[bbox_num].copy()
                ###### surface point
                draw.draw_points([carb.Float3(i) for i in [world_points] ] , 
                                    [carb.ColorRgba(1.0,1.0,0.0,1.0)]*len([world_points]),
                                    [15]*len([world_points])     )
                ###### grasp point
                world_points[2] = target_z

                draw.draw_points([carb.Float3(i) for i in [world_points] ] , 
                                    [carb.ColorRgba(1.0,0.0,1.0,1.0)]*len([world_points]),
                                    [15]*len([world_points])     )

                
                ###############
                # plt.scatter(PCD[0],PCD[1], c = PCD[2], cmap = "jet", s=0.1)
                # plt.scatter(world_to_grasp.T[0],world_to_grasp.T[1], c = 'yellow', s=10)
                # pt = gripper_bbox_3d[bbox_num]
                # plt.plot(pt[0,[0,1,2,3,0]],pt[1,[0,1,2,3,0]],c = "g")
                # plt.plot(pt[0,[4,5,6,7,4]],pt[1,[4,5,6,7,4]], c = "g")
                # plt.plot(pt[0,[8,9,10,11,8]],pt[1,[8,9,10,11,8]], c = "r")
                # plt.scatter(pt[0,12],pt[1,12], c = "b",s=5)

                # pt_in_bbx = point_in_bboxes_idx[bbox_num]
                # plt.scatter(world_to_obj_part[0][pt_in_bbx[0]], world_to_obj_part[1][pt_in_bbx[0]], c = "g",s=2)
                # plt.scatter(world_to_obj_part[0][pt_in_bbx[1]], world_to_obj_part[1][pt_in_bbx[1]], c = "g",s=2)
                # plt.scatter(world_to_obj_part[0][pt_in_bbx[2]], world_to_obj_part[1][pt_in_bbx[2]], c = "r",s=2)
                # plt.show()

                
                my_world.reset()
                # print(world_points)
                grasp_success_dict = robot_task.picking(target_points = world_points, 
                                                target_orientation = [0,0,rot_deg], 
                                                target_width = gripper_width,
                                                target_prim_path=str(obj.prim.GetPath()))

                # draw.clear_points()
                
                print("grasp_success : ",grasp_success_dict["success"])
                if grasp_success_dict["success"]:
                    grasp_success_dict["width"] +=0.005
                    org_grasp_bbox = np.array([[ robot_task.gripper_height/2, -grasp_success_dict["width"]/2 -  robot_task.finger_thickness, 0],
                                                [-robot_task.gripper_height/2, -grasp_success_dict["width"]/2 -  robot_task.finger_thickness, 0],
                                                [-robot_task.gripper_height/2,  grasp_success_dict["width"]/2 +  robot_task.finger_thickness, 0],
                                                [ robot_task.gripper_height/2,  grasp_success_dict["width"]/2 +  robot_task.finger_thickness, 0] ]).T
                    org_grasp_bbox = cs.rot_z(rot_deg).dot(np.vstack((org_grasp_bbox,np.ones_like(org_grasp_bbox[0]))) )[:3].T
                    ##### no restore
                    org_grasp_bbox_shifted = world_points + org_grasp_bbox
                    org_grasp_bbox_shifted = np.vstack((org_grasp_bbox_shifted,np.zeros_like(org_grasp_bbox_shifted[0])))
                    # draw.draw_lines([carb.Float3(i) for i in org_grasp_bbox_shifted[[0,1,2,3]] ],
                    #                 [carb.Float3(i) for i in org_grasp_bbox_shifted[[1,2,3,0]] ],
                    #                 [carb.ColorRgba(0.0,1.0,1.0,1.0),
                    #                  carb.ColorRgba(1.0,0.0,1.0,1.0),
                    #                  carb.ColorRgba(0.0,1.0,1.0,1.0),
                    #                  carb.ColorRgba(1.0,0.0,1.0,1.0)],
                    #                 [1]*4     )
                    
                    
                    ##### resetore
                    # import pdb;pdb.set_trace()
                    suc_gripper_bbx = org_grasp_bbox + grasp_success_dict["last_position"]
                    obj_tf = np.array(csr.find_parents_tf(obj.prim, include_self=True, include_scale=False)).T
                    obj_init_tf = rot_utils.euler_to_rot_matrix(obj.init_rotation,degrees=True)
                    obj_init_tf = np.vstack((np.hstack((obj_init_tf,np.array(obj.init_position)[:,None])),[0,0,0,1]))
                    suc_gripper_bbx_3d = obj_init_tf.dot(np.linalg.inv(obj_tf)).dot(np.vstack((suc_gripper_bbx.T,np.ones_like(suc_gripper_bbx.T[0]))))[:3].T
                    

                    draw.clear_lines()
                    draw.draw_lines([carb.Float3(i) for i in suc_gripper_bbx_3d[[0,1,2,3]] ],
                                    [carb.Float3(i) for i in suc_gripper_bbx_3d[[1,2,3,0]] ],
                                    [carb.ColorRgba(1.0,0.0,0.0,1.0),
                                     carb.ColorRgba(0.0,1.0,0.0,1.0),
                                     carb.ColorRgba(1.0,0.0,0.0,1.0),
                                     carb.ColorRgba(0.0,1.0,0.0,1.0)],
                                    [1]*4     )
                    
                    ## edited
                    bbx_cnt = suc_gripper_bbx_3d.mean(axis=0)
                    width_cnt = np.array([(suc_gripper_bbx_3d[0] + suc_gripper_bbx_3d[1])/2 , (suc_gripper_bbx_3d[2] + suc_gripper_bbx_3d[3])/2])
                    width_cnt[:,2] = bbx_cnt[2]
                    width_sub = (width_cnt[1] - width_cnt[0])[:2]
                    edited_width = np.sqrt(np.sum(width_sub**2))
                    yaw = -np.arctan(width_sub[0]/width_sub[1])
                    
                    #### option
                    org_grasp_bbox = np.array([ [ robot_task.gripper_height/2, -edited_width/2 , 0],
                                                [-robot_task.gripper_height/2, -edited_width/2 , 0],
                                                [-robot_task.gripper_height/2,  edited_width/2 , 0],
                                                [ robot_task.gripper_height/2,  edited_width/2 , 0] ]).T
                    edited_grasp_bbox = np.vstack((org_grasp_bbox, np.ones_like(org_grasp_bbox[0]) ))
                    edited_grasp_bbox =cs.rot_z(yaw/np.pi*180).dot(edited_grasp_bbox)[:3].T + bbx_cnt 
                    
                    draw.draw_lines([carb.Float3(i) for i in edited_grasp_bbox[[0,1,2,3]] ],
                                    [carb.Float3(i) for i in edited_grasp_bbox[[1,2,3,0]] ],
                                    [carb.ColorRgba(1.0,1.0,0.0,1.0),
                                     carb.ColorRgba(0.0,0.0,1.0,1.0),
                                     carb.ColorRgba(1.0,1.0,0.0,1.0),
                                     carb.ColorRgba(0.0,0.0,1.0,1.0)],
                                    [1]*4     )
                    
                    top_cam_tf = np.array(cam_tf).dot(cs.rot_x(180))
                    bbox_3d_to_2d = np.array(top_cam_config["intrinsic_isaac"]).dot(cs.dot([np.linalg.inv(top_cam_tf), np.vstack((edited_grasp_bbox.T, np.ones_like(edited_grasp_bbox.T[0]))) ])[:3])
                    bbox_3d_to_2d = (bbox_3d_to_2d[:2]/bbox_3d_to_2d[2]).T
                    grasp_output_dict[obj.class_name].append(
                        {
                            "bbox_3d_point":suc_gripper_bbx_3d.tolist(),
                            "bbox_3d_point_aligned":edited_grasp_bbox.tolist(),
                            "bbox_2d":{
                                "bbox":bbox_3d_to_2d.tolist(),
                                "center":bbox_3d_to_2d.mean(axis=0).tolist(),
                                "width":np.sqrt(np.sum((bbox_3d_to_2d[1] - bbox_3d_to_2d[2])**2) ).tolist(),
                                "height":np.sqrt(np.sum((bbox_3d_to_2d[0] - bbox_3d_to_2d[1])**2) ).tolist(),
                                "angle":yaw,
                                },
                            "quality":grasp_success_dict["quality"],
                            "isaac_env":{
                                "target_points": world_points.tolist(),
                                "target_orientation": [0,0,rot_deg],
                                "target_width": gripper_width,
                                "target_prim_path":str(obj.prim.GetPath())
                            },
                            "gripper_model":robot_task.gripper_model,
                            
                        }
                    )
    with open(os.path.join(grasp_ann_path,f"{scene_num:04d}"+".json"), 'w') as f:
        json.dump(grasp_output_dict,f, indent=4)


                    ##### save data
  
                    

duration = time.time() - start_time
print("duration : ",duration)


# my_world.reset()
# my_world.stop()
# world_stop_flag = True
# while True:
#     my_world.step(render=True)
#     if my_world.is_stopped():
#         if not world_stop_flag:
#             world_stop_flag = True
#     if my_world.is_playing():
#         if world_stop_flag:
#             my_world.reset()
#             world_stop_flag = False
#             print("reset")
    

        
simulation_app.close()

