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
import Robot_Task
from omni.physx import get_physx_interface, get_physx_simulation_interface
from omni.physx import get_physx_scene_query_interface
from omni.physx.scripts.physicsUtils import *
import cs_utils as cs
import cs_rep_utils as csr
from PIL import Image
import ast
import matplotlib.pyplot as plt

print("world_set")
my_world = World(stage_units_in_meters=1.0,
                 physics_dt  = 0.005,
                 rendering_dt = 0.005)
stage = omni.usd.get_context().get_stage()
robot_task = Robot_Task.RobotTask(
    name="robot_task",
    usd_prim_path="/World/Doosan_M1013",)
my_world.add_task(robot_task)
my_world.reset()

print("robot_set")
robot = robot_task._robot
robot_task.set_init_pose(position =[0.5,0.5,0.056], rotation = [0,0,0])
robot_tf = np.array(csr.find_parents_tf(robot.prim, include_self=True)).T
robot_task.set_contact_sensor()

# robot = robot_task._robot
# robot.set_world_pose(position = [0.5,0.5,11])

# robot_name = robot_task.get_robot_name
# my_robot = my_world.scene.get_object(robot_name)




print("env_set")
plane_object = GroundPlane(prim_path = "/World/ground_plane",
                           name = "ground_plane",
                           scale = [1,1,1],
                           z_position = 0,
                           )
plane_col_prim = stage.GetPrimAtPath(os.path.join(plane_object.prim_path,"collisionPlane"))
plane_prim = stage.GetPrimAtPath(os.path.join(plane_object.prim_path,"geom"))
plane_col_prim.GetAttribute('physics:collisionEnabled').Set(False)

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
       "inputs:exposure" : 4,
    }
)






print("config_set")
########  load config
scene_num = 4
root_path = "/home/nia/ochansol/isaac/sanjabu/data/scene/Home/"
with open( os.path.join(root_path,"conf",f"{scene_num:04d}"+".json"), 'r') as f:
    config= json.load(f)
obj_conf = config["objects"]
cam_conf = config["cameras"]
physx_conf = config["physics_scene"]
top_cam_config = [ i for i in cam_conf if i["name"] == "top_view_camera"][0]
side_cam_config = [ i for i in cam_conf if i["name"] == "side_view_camera"][0]

depth_img_cam00 = np.load(os.path.join(root_path,"depth","cam_00",f"{scene_num:04d}"+".npy"))
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





print("physx_set")
stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:enableGPUDynamics').Set(True)
stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:broadphaseType').Set("GPU")
stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:collisionSystem').Set("PCM") 
stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:gpuTotalAggregatePairsCapacity').Set(20000) 
stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:gpuFoundLostAggregatePairsCapacity').Set(20000) 

# stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:timeStepsPerSecond").Set(350)
# stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:minPositionIterationCount").Set(150) 
# stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:minVelocityIterationCount").Set(150)


print("obj_set")
######object pose set
obj_rep_list = []
for obj in obj_conf:
    scan_obj = scan_rep.Scan_Rep(usd_path = obj['usd_path'],
                                          class_name = obj['class'])
    obj_rep_list.append(scan_obj)
    
    scan_obj.set_pose(position = obj["translate"], rotation = rot_utils.quat_to_euler_angles(obj["orient"], degrees=True) )
    scan_obj.set_scale(scale = obj["scale"])
    scan_obj.set_rigidbody_collider()
    # scan_obj.set_contact_sensor()
    scan_obj.set_physics_material(
        dynamic_friction=0.5,
        static_friction=0.8,
        restitution=0.0
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
point_dict ={}




gripper_finger_bbox = np.array([[-robot_task.gripper_height/2, -robot_task.gripper_width/2 -  robot_task.finger_thickness],
                                [-robot_task.gripper_height/2, -robot_task.gripper_width/2 ],
                                [ robot_task.gripper_height/2, -robot_task.gripper_width/2 ],
                                [ robot_task.gripper_height/2, -robot_task.gripper_width/2 -  robot_task.finger_thickness],
                                [-robot_task.gripper_height/2,  robot_task.gripper_width/2 ],
                                [-robot_task.gripper_height/2,  robot_task.gripper_width/2 +  robot_task.finger_thickness],
                                [ robot_task.gripper_height/2,  robot_task.gripper_width/2 +  robot_task.finger_thickness],
                                [ robot_task.gripper_height/2,  robot_task.gripper_width/2 ]])
gripper_width_bbox = np.array([[-robot_task.gripper_height/2, -robot_task.gripper_width/2],
                               [-robot_task.gripper_height/2,  robot_task.gripper_width/2],
                               [ robot_task.gripper_height/2,  robot_task.gripper_width/2],
                               [ robot_task.gripper_height/2, -robot_task.gripper_width/2],]) 
gripper_center_point = np.array([[ 0,0 ]])
gripper_bbox = np.vstack((gripper_finger_bbox,gripper_width_bbox,gripper_center_point)).T
gripper_bbox = np.vstack((gripper_bbox,np.zeros_like(gripper_bbox[0])))


IDX = np.argwhere(depth_img_cam00==depth_img_cam00).T
sample_idx = np.random.choice(IDX.shape[1],int(IDX.shape[1]*0.3),replace=False)
IDX = IDX[:,sample_idx]
X_all = (IDX[1]-cx)/focal_length*depth_img_cam00[IDX[0],IDX[1]]
Y_all = (IDX[0]-cy)/focal_length*depth_img_cam00[IDX[0],IDX[1]]

cam_to_pt_all = np.array([X_all,Y_all,depth_img_cam00[IDX[0],IDX[1]],np.ones_like(X_all)])
world_to_pt_all = np.array(cam_tf).dot(cs.rot_x(180)).dot(cam_to_pt_all)[:3]

# plt.scatter(X_all,Y_all, c = depth_img_cam00[IDX[0],IDX[1]], cmap = "jet", s=0.1)
# plt.show()


def align_bbox(bbox):
    Min = np.min(bbox,axis=0)
    idx_y,idx_x = np.where(bbox==Min)
    if len(idx_y)>2:
        idx = max(idx_y[0],idx_y[2])
        return np.roll(bbox,-idx,axis=0),False
    else:
        idx = idx_y[0]
        return np.roll(bbox,-idx,axis=0),True

def lin_func(bbox,points):# bbox=2x2, points = 2xN
    x1,y1 = bbox[0]
    x2,y2 = bbox[1]
    if x2-x1 == 0:
        return points[0] - x1
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1    
    return a*points[0] - points[1] + b 

def select_point_in_bbox(bbox,points): # bbox = 4x2, points = 2xN
    aligned_bbox, zero = align_bbox(bbox)
    if zero:
        p1 = np.where(lin_func(aligned_bbox[[0,1]],points)<=0)
        p2 = np.where(lin_func(aligned_bbox[[1,2]],points)>=0)
        p3 = np.where(lin_func(aligned_bbox[[2,3]],points)>=0)
        p4 = np.where(lin_func(aligned_bbox[[3,0]],points)<=0)
    else:
        p1 = np.where(lin_func(aligned_bbox[[0,1]],points)>=0)
        p2 = np.where(lin_func(aligned_bbox[[1,2]],points)<=0)
        p3 = np.where(lin_func(aligned_bbox[[2,3]],points)<=0)
        p4 = np.where(lin_func(aligned_bbox[[3,0]],points)>=0)
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
labeled_depth_dict = {}
for obj in obj_rep_list:
    idx_all = None
    for color_key in inst_label_cam00.keys():
        if obj.class_name == inst_label_cam00[color_key]["class"]:
            color_arr = np.array(ast.literal_eval(color_key))

            idx_all = np.argwhere((inst_img_cam00[...,0] == color_arr[0]) &\
                            (inst_img_cam00[...,1] == color_arr[1]) &\
                            (inst_img_cam00[...,2] == color_arr[2]) &\
                            (inst_img_cam00[...,3] == color_arr[3])).T
    if idx_all is None:
        continue
    labeled_depth_dict[obj.class_name] = idx_all

all_obj_idx = np.hstack([labeled_depth_dict[key] for key in labeled_depth_dict.keys()])
draw = _debug_draw.acquire_debug_draw_interface()
for obj in obj_rep_list:
    idx_all= labeled_depth_dict[obj.class_name]
    
    sample_num = 2000
    idx = idx_all[:,np.random.choice(np.arange(len(idx_all[0])),sample_num)]
    idx_dist = np.tile(idx[:,None,:],(1,sample_num,1)) - np.tile(idx[...,None],(1,1,sample_num))
    idx_dist = np.sqrt(np.sum(idx_dist**2,axis=0))
    dist_th = 40
    cnt = 0
    while len(idx_dist)>cnt:
        min_idx = np.argwhere((idx_dist[cnt]<=dist_th) & (idx_dist[cnt]!=0))
        idx = np.delete(idx, min_idx, axis=1)
        idx_dist = np.delete(idx_dist, min_idx, axis=0)
        idx_dist = np.delete(idx_dist, min_idx, axis=1)
        cnt+=1
    print(len(idx.T))
    temp_depth = depth_img_cam00[idx[0],idx[1]]
    X = (idx[1]-cx)/focal_length*temp_depth
    Y = (idx[0]-cy)/focal_length*temp_depth
    cam_to_pt = np.array([X,Y,temp_depth,np.ones_like(X)])
    cam_to_pt = np.hstack((cam_to_pt,np.array([[0,0,-1.2,1]]).T ))
    world_to_pt = np.array(cam_tf).dot(cs.rot_x(180)).dot(cam_to_pt)[:3].T
    

    
    #### point in bbox
    rot_deg = 0
    gripper_bbox_3d = cs.rot_z(rot_deg).dot(np.vstack((gripper_bbox,np.ones_like(gripper_bbox[0]))))[:3]
    gripper_bbox_3d = np.tile(world_to_pt[...,None], (1,1,gripper_bbox_3d.shape[1])) + np.tile(gripper_bbox_3d[None,...],(len(world_to_pt),1,1))
 
    X_all = (all_obj_idx[1]-cx)/focal_length*depth_img_cam00[all_obj_idx[0],all_obj_idx[1]]
    Y_all = (all_obj_idx[0]-cy)/focal_length*depth_img_cam00[all_obj_idx[0],all_obj_idx[1]]
    cam_to_pt_all = np.array([X_all,Y_all,depth_img_cam00[all_obj_idx[0],all_obj_idx[1]],np.ones_like(X_all)])
    world_to_pt_part = np.array(cam_tf).dot(cs.rot_x(180)).dot(cam_to_pt_all)[:3]
    point_in_bboxes_idx = select_points(gripper_bbox_3d,world_to_pt_part)

    for bbox_num in range(len(gripper_bbox_3d)):
        finger_min_depth = np.max([0 if len(point_in_bboxes_idx[bbox_num][0])==0 else world_to_pt_part[2][point_in_bboxes_idx[bbox_num][0]].max(),  
                                0 if len(point_in_bboxes_idx[bbox_num][1])==0 else world_to_pt_part[2][point_in_bboxes_idx[bbox_num][1]].max()])
        palm_min_depth   = np.max( 0 if len(point_in_bboxes_idx[bbox_num][2])==0 else world_to_pt_part[2][point_in_bboxes_idx[bbox_num][2]].max())
        
        
        ########### visualization
        plt.scatter(world_to_pt_all[0],world_to_pt_all[1], c = world_to_pt_all[2], cmap = "jet", s=0.1)
        pt = gripper_bbox_3d[bbox_num]
        plt.plot(pt[0,[0,1,2,3,0]],pt[1,[0,1,2,3,0]],c = "g")
        plt.plot(pt[0,[4,5,6,7,4]],pt[1,[4,5,6,7,4]], c = "g")
        plt.plot(pt[0,[8,9,10,11,8]],pt[1,[8,9,10,11,8]], c = "r")
        plt.scatter(pt[0,12],pt[1,12], c = "b",s=5)
        pt_in_bbx = point_in_bboxes_idx[bbox_num]
        plt.scatter(world_to_pt_part[0][pt_in_bbx[0]], world_to_pt_part[1][pt_in_bbx[0]], c = "g",s=2)
        plt.scatter(world_to_pt_part[0][pt_in_bbx[1]], world_to_pt_part[1][pt_in_bbx[1]], c = "g",s=2)
        plt.scatter(world_to_pt_part[0][pt_in_bbx[2]], world_to_pt_part[1][pt_in_bbx[2]], c = "r",s=2)
        plt.show()
        crit = palm_min_depth - finger_min_depth
        margin = 0.005
        if crit<0.01:
            continue
        if crit>robot_task.gripper_depth:
            target_z = palm_min_depth + margin
        else:
            target_z = finger_min_depth + margin


        world_points = world_to_pt[bbox_num]
        world_points[2] = target_z
        draw.draw_points([carb.Float3(i) for i in [world_points] ] , 
                            [carb.ColorRgba(0.0,1.0,0.0,1.0)]*len([world_points]),
                            [15]*len([world_points])     )
        # draw.draw_points([carb.Float3(i) for i in [[0,0,0]]] , 
        #                     [carb.ColorRgba(0.0,0.0,1.0,1.0)]*len([[0,0,0]]),
        #                     [15]*len([[0,0,0]])     )
        # draw.draw_points([carb.Float3(i) for i in [world_to_pt[-1]]] , 
        #                     [carb.ColorRgba(1.0,0.0,0.0,1.0)]*len([world_to_pt[-1]]),
        #                     [15]*len([world_to_pt[-1]])     )
        
        ###############
        
        target_points = np.linalg.inv(robot_tf).dot(np.vstack((world_points[None,...].T,np.ones(1) ) ) )[:3].T[0]
        my_world.reset()
        print(world_points)
        robot_task.picking(target_position = target_points, target_orientation = [0,-180,rot_deg], frame_name="J6")
        draw.clear_points()




my_world.reset()
my_world.stop()
world_stop_flag = True
while True:
    my_world.step(render=True)
    if my_world.is_stopped():
        if not world_stop_flag:
            world_stop_flag = True
    if my_world.is_playing():
        if world_stop_flag:
            my_world.reset()
            world_stop_flag = False
            print("reset")
    

        
simulation_app.close()

