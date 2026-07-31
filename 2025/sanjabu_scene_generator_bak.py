from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

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
from pxr import Usd, UsdShade, Sdf, Gf, UsdGeom, PhysxSchema, UsdPhysics
from omni.isaac.core.objects import GroundPlane, DynamicCuboid
from omni.physx.scripts import utils as physx_utils


from omni.isaac.core.utils.extensions import enable_extension

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

from omni.physx.scripts.physicsUtils import *

sys.path.append("/home/cubox/ochansol/isaac_code/python/utils")


import omni.isaac.core.utils.bounds as bounds_utils
from omni.isaac.debug_draw import _debug_draw
import scan_rep

import cs_utils as cs
import cs_rep_utils as csr
import light_set as light
import sanjabu_Writer as sw

import argparse
parser = argparse.ArgumentParser(description="Sanjabu Scene Generator")
parser.add_argument("--scene_num", default=0, help="")
args = parser.parse_args()



############# set params
render_set = False
object_path = "/nas/ochansol/3d_model/peel3_scan_data_2025" #"/scan_data_2204/objects_conf.json"
root_path = "/nas/ochansol/isaac/sanjabu/envs"
env_name = ["Home","Logistic_site","Manufactory"][1]
env_path_dict = {
    "Home" : f"{root_path}/{env_name}/Home_scene1_2204_flatten.usd",
    "Logistic_site" : f"{root_path}/{env_name}/logistic_scene_1f_centric.usd",
    "Manufactory" : f"{root_path}/{env_name}/manufactory_chiken.usd"
}
env_conf = {
    "name": env_name,
    "usd_path": env_path_dict[env_name],
    "position":[0,0,0],
    "orientation":[90,0,0],
    "scale":[0.01,0.01,0.01]
}

output_path =  "/nas/Dataset/Dataset_2025/test_data/" + env_conf["name"] 

cam_model_conf_path = "/home/cubox/ochansol/isaac_code/configure/percipio_FM855-E1_conf.json"

cam_conf = {
    "name":"",
    "cam_model_conf_path" : cam_model_conf_path,
    "pixel_size" : 0.003,
    "output_size" : (1920,1080),# min object 1920*1280 = 96*54( 5% )
    "clipping_range" : (0.0001, 100000),
    "focus_distance" : 0,
    "f_stop" : 0,
    "cam_poses" : [],
}

writer_dict = {
    "rgb"                           : True,
    "bounding_box_2d_loose"         : False,
    "bounding_box_2d_tight"         : True,
    "bounding_box_3d"               : False,
    "distance_to_camera"            : False,
    "distance_to_image_plane"       : True,
    "instance_segmentation"         : True,
    "normals"                       : True,
    "semantic_segmentation"         : False,
    "use_common_output_dir"         : True,
    "pointcloud_include_unlabelled" : True,
    "pointcloud"                    : True
}


######################





my_world = World(stage_units_in_meters=1.0,
                 physics_dt  = 0.001,
                 rendering_dt = 0.005)
stage = omni.usd.get_context().get_stage()

my_world.reset()



######## env set


env_usd = add_reference_to_stage(usd_path=env_conf["usd_path"], 
                                    prim_path="/World/"+env_conf["name"])

env_prim = Prims.XFormPrim(name =env_conf["name"], prim_path="/World/"+env_conf["name"], 
                             position = env_conf["position"], 
                             orientation = rot_utils.euler_angles_to_quat( env_conf["orientation"], degrees = True), 
                             scale = env_conf["scale"] )
light_list = csr.find_lights(env_usd)
Lights = light.Light(light_list)
Lights.random_trans(0.2, [1])
Lights.random_exposure()
Lights.random_intensity()


###### parent끼리 중복검사 해야됨





######object set
with open(os.path.join(object_path, "objects_conf.json"),'r'  ) as f:
    obj_attr = json.load(f)

model_list_freq = []
model_list = []
for obj in obj_attr[:30]:
    # model_list_freq +=  [obj["name"]]*obj["envs"][env_conf["name"]]
    model_list.append(obj)# if obj["envs"][env_conf["name"]] > 0 else None



obj_rep_all_list = []
obj_rep_size_list = []
for model_attr in model_list:
    print("model_attr : ", model_attr["name"])
    scan_obj = scan_rep.Scan_Rep(usd_path =  model_attr["path"],
                               class_name = model_attr["name"],
                               size = model_attr["size_rank"],)
    # scan_obj.prim.SetActive(False)
    obj_rep_all_list.append(scan_obj)
    obj_rep_size_list.append(scan_obj.size)
print("object set complete : ", len(obj_rep_all_list))



######## cam set

with open(cam_model_conf_path, 'r') as f:
    cam_model_conf = json.load(f)

((fx,_,cx),(_,fy,cy),(_,_,_))= cam_model_conf["intrinsic_matrix"]

cam_conf["focal_length_isaac"] = (fx+fy)/2*cam_conf["pixel_size"]
cam_conf["horizontal_aperture"] = cam_conf["output_size"][0]*cam_conf["pixel_size"]
cam_conf["intrinsic_isaac"] = [[(fx+fy)/2, 0,cam_conf["output_size"][0]/2],
                               [0, (fx+fy)/2, cam_conf["output_size"][1]/2],
                               [0,0,1]]

top_view_camera = rep.create.camera(
    position = [0,0,1],
    rotation = [0,-90,0],
    # look_at =obj_rep_list[0].node,
    focal_length = cam_conf["focal_length_isaac"], 
    focus_distance =cam_conf["focus_distance"], 
    f_stop = cam_conf["f_stop"], 
    horizontal_aperture = cam_conf["horizontal_aperture"],
    clipping_range = cam_conf["clipping_range"])

side_view_camera = rep.create.camera(
    position = [0,0,0],
    # rotation = [],
    # look_at = obj_rep_list[0].node,
    focal_length = cam_conf["focal_length_isaac"], 
    focus_distance = cam_conf["focus_distance"], 
    f_stop = cam_conf["f_stop"], 
    horizontal_aperture = cam_conf["horizontal_aperture"],
    clipping_range = cam_conf["clipping_range"])

cam_conf1 = cam_conf.copy()
cam_conf2 = cam_conf.copy()
cam_conf1["name"] = "top_view_camera"
cam_conf2["name"] = "side_view_camera"


print("cam set complete : ", cam_conf1["name"], cam_conf2["name"])


######## render set

render_product_top = rep.create.render_product(top_view_camera, cam_conf["output_size"])
render_product_side = rep.create.render_product(side_view_camera, cam_conf["output_size"])
writer = rep.WriterRegistry.get("SanjabuWriter")
writer.initialize(
    output_dir                      = output_path,
    rgb                             = writer_dict["rgb"],
    bounding_box_2d_loose           = writer_dict["bounding_box_2d_loose"],
    bounding_box_2d_tight           = writer_dict["bounding_box_2d_tight"],
    bounding_box_3d                 = writer_dict["bounding_box_3d"],
    distance_to_camera              = writer_dict["distance_to_camera"],
    distance_to_image_plane         = writer_dict["distance_to_image_plane"],
    instance_segmentation           = writer_dict["instance_segmentation"],
    normals                         = writer_dict["normals"],
    semantic_segmentation           = writer_dict["semantic_segmentation"],
    use_common_output_dir           = writer_dict["use_common_output_dir"],
    pointcloud_include_unlabelled   = writer_dict["pointcloud_include_unlabelled"],
    pointcloud                      = writer_dict["pointcloud"]
)
writer.set_path(output_path,
                rgb_path = "rgb",
                bounding_box_path = "bbox",
                distance_to_image_plane_path = "depth",
                instance_segmentation_path = "inst_seg",
                pointcloud_path = "pointcloud",
                normals_path = "normals",)
writer.set_cam_name_list([cam_conf1["name"], cam_conf2["name"]])

# # Attach render_product to the writer

# instance_seg_annotator = rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast")
# instance_seg_annotator.attach([render_product_top])
# depth_cam_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
# depth_cam_annotator.attach([render_product])
# depth_plane_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
# depth_plane_annotator.attach([render_product])

writer.attach([render_product_top, render_product_side])
rep.orchestrator.pause()
rep.orchestrator.set_capture_on_play(False)

print("render set complete ")


##################################################################################33
my_world.reset()
my_world.stop()
# writer.set_frame(frame_id=0)
os.makedirs(os.path.join(output_path,"conf"), exist_ok=True)
print("dir making complete : ")

physics_scene_conf={
    # 'physxScene:enableGPUDynamics': 1, # True
    # 'physxScene:broadphaseType' : "GPU",
    # 'physxScene:collisionSystem' : "PCM",
    
    # 'physxScene:timeStepsPerSecond' : 1000,
    'physxScene:minPositionIterationCount' : 30,
    'physxScene:minVelocityIterationCount' : 1,
    "physics:gravityMagnitude":35,
    # "physxScene:updateType":"Asynchronous",
}
for key in physics_scene_conf.keys():
    stage.GetPrimAtPath("/physicsScene").GetAttribute(key).Set(physics_scene_conf[key])
    
    

# for ls in del_list:
#     os.remove(os.path.join(output_path,ls))

for OBJ in obj_rep_all_list:
    print("set collider for : ", OBJ.class_name)
    OBJ.set_rigidbody_collider()
    # OBJ.set_contact_sensor()
    OBJ.set_physics_material(
        dynamic_friction=0.25,
        static_friction=0.4,
        restitution=0.0
    )


# plane_object = scan_rep.Scan_Rep(usd_path = plane_object_path,
#                                  class_name = "plane",
#                                  prim_path = "plane",
#                                  visible=True,
#                                  scale = [2,2,2]
#                                  )

platform_area_prims = csr.find_target_name(env_prim.prim,["Mesh"],"platform_area")
platform_area_prims = [i.GetParent() for i in platform_area_prims] #if i.GetParent().GetName() == "Modern_Kitchen_01"]
# platform_path = np.random.choice(platform_area_prims).GetPath().__str__()
platform_rep_list = []
platform_path_list = []
for plf_prim in platform_area_prims:
    platform_path = plf_prim.GetPath().__str__()
    platform_rep = scan_rep.Scan_Rep_Platform(prim_path = platform_path,scale = [1,1,1], class_name = platform_path.split("/")[-1])
    platform_rep_list.append(platform_rep)
    platform_path_list.append(platform_path)

my_world.reset()
for platform_rep, platform_path in zip(platform_rep_list, platform_path_list):
    platform_tf = csr.find_parents_tf(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
    platform_scale = csr.find_parents_scale(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
    platform_rep.set_tf(platform_tf)
    platform_rep.set_scale(platform_scale)


# platform_rep.set_pose(position = env_prim.get_world_pose()[0] ,rotation = rot_utils.quat_to_euler_angles(env_prim.get_world_pose()[1], degrees = True ))
# platform_rep.set_scale(scale = env_prim.get_world_scale())
# platform_rep.set_collider()
print("platform set complete")

scene_num  = 0 
import time
while scene_num<10:
    print("START")
    data_gen_time = time.time()
    print("####################    scene_num : ",scene_num)
    settings.set("/rtx/rendermode", "RayTraced")
    scene_name = f"{scene_num:04d}"
    
    platform_rep = np.random.choice(platform_rep_list)

    print("platform_rep : ", platform_rep.prim)

    
    obj_rep_list = [platform_rep]
    size_rank = np.random.randint(0, 2) # 0: small, 1: medium, 2: large
    print("size_rank : ", size_rank)
    size_idx = np.where(np.array(obj_rep_size_list) == size_rank)[0]
    active_idx = np.random.choice(size_idx, 5, replace=False)
    for i, obj in enumerate(obj_rep_all_list):
        if i in active_idx:
            obj.prim.SetActive(True)
            obj_rep_list.append(obj)
            print(obj.prim)
        else:
            obj.prim.SetActive(False)


    my_world.reset()
    my_world.stop()


    Lights.random_exposure(val = 1.2)#, default_exposure = np.random.uniform(1,2.3) )
    Lights.random_temp(val = 300, default_temp = 5800)



    # my_world.play()
    
    
    # platform_obb = platform_rep.get_obb()
    # csr.debug_draw_obb(platform_rep.get_obb())
    # platform_pos = platform_obb.mean(axis=0)
    # platform_pos[2] = platform_obb.max(axis=0)[2]
    # csr.debug_draw_points([platform_pos], size=10,color=[0,1,0])
    # platform_scale = platform_obb.max(axis=0)-platform_obb.min(axis=0)


    # csr.top_view_cam_move(top_view_camera, writer, obj_rep_list[0], 1)


    # csr.scatter_on_target_object(top_view_camera, writer, obj_rep_list, distance = 1)

    csr.scatter_in_specipic_area(obj_rep_list[0],obj_rep_list[1:])
    
    # csr.scatter3D_obb(obj_rep_list, 
    #                     center_position=platform_pos+np.array([0,0,0.15]),
    #                     scale = (platform_scale[0],platform_scale[1],0.3),
    #                     drop_out = False,
    #                     fixed_first=True)


    # csr.scatter3D_obb(obj_rep_list, 
    #                 center_position=(0,0,0.1),
    #                 scale = (0.25,0.4,0.1),
    #                 drop_out = False,
    #                 fixed_first=True)



    obj_rep_list = obj_rep_list[1:]
    
    my_world.play()
    obj_rotation_buf = []
    obj_location_buf = []

    for i in range(20):
        my_world.step(render = render_set)
        obj_rotation_buf.append([obj.get_local_pose()["rotation"]for obj in obj_rep_list])
        obj_location_buf.append([obj.get_local_pose()["translation"] for obj in obj_rep_list])

    while True:
        my_world.step(render = render_set)
        del(obj_rotation_buf[0])
        del(obj_location_buf[0])
        obj_rotation_buf.append([obj.get_local_pose()["rotation"]for obj in obj_rep_list])
        obj_location_buf.append([obj.get_local_pose()["translation"] for obj in obj_rep_list])
        # print(np.array(obj_rotation_buf).std(axis=0).max())
        # print(np.array(obj_location_buf).std(axis=0).max())
        if np.array(obj_rotation_buf).std(axis=0).max()<=0.00001 and np.array(obj_location_buf).std(axis=0).max()<=0.0001:
            break
        
        if my_world.current_time>6:
            break

    print("current_time : ",my_world.current_time)
    ########  
    obb_list = []
    for obj in obj_rep_list:
        obb_list.append(obj.get_obb())

    obb_arr = np.vstack(obb_list)
    obb_min = obb_arr.min(axis=0)
    obb_max = obb_arr.max(axis=0)
    center = (obb_min+obb_max)/2
        
    ########
    with top_view_camera:
        rep.modify.pose(position = [center[0],center[1],center[2]+1.2])
    rep.orchestrator.step()

    if writer.get_data()["annotators"]["instance_segmentation_fast"]["Replicator"]["idToSemantics"].keys().__len__()<7 or \
        writer.get_data()["annotators"]["instance_segmentation_fast"]["Replicator"]["data"].min() == 0:
        print("scene_reset, 탑뷰 카메라 오류")
        continue

    for _ in range(8):
        with side_view_camera:
            rad  = np.random.randint(0,360)/180*np.pi
            dist = 0.8
            x,y,z = dist*np.cos(rad)+center[0], dist*np.sin(rad)+center[1], center[2]+1
            rep.modify.pose(position=(x,y,z),
                            look_at = center,)
        rep.orchestrator.step()


        side_view_obj_count = writer.get_data()["annotators"]["instance_segmentation_fast"]["Replicator_01"]["idToSemantics"].keys().__len__()
        if side_view_obj_count<7:
            print("side_view_obj_count : ", side_view_obj_count)
            continue
        else:
            break
    if side_view_obj_count<7:
        break
    
    # side_view_bboxes = np.array(writer.get_data()["annotators"]["bounding_box_2d_tight_fast"]["Replicator_01"]["data"].tolist())[:,1:5]
    # side_view_bboxes_xmax = np.max(side_view_bboxes[:,2])>=cam_conf["output_size"][0]
    # side_view_bboxes_ymax = np.max(side_view_bboxes[:,3])>=cam_conf["output_size"][1]
    # side_view_bboxes_min = np.min(side_view_bboxes)<=0
    # if side_view_bboxes_xmax or side_view_bboxes_ymax or side_view_bboxes_min:
    #     continue

    my_world.pause()
  
    print("spp complete")
        ####
    rep.orchestrator.run()
    rep.orchestrator.step()
    rep.orchestrator.pause()

    writer.set_frame(frame_id=scene_num)
    settings.set("/rtx/rendermode", "PathTracing")

    settings.set("/rtx/pathtracing/spp", 1) 
    settings.set("/rtx/pathtracing/totalSpp", 150)
    settings.set("/rtx/pathtracing/maxBounces", 12)
    settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 12)
    rep.orchestrator.step()

    

    obj_conf = []

    for OBJ in obj_rep_list:
        pose = OBJ.get_world_pose()
        scale = OBJ.get_scale()
        obj_conf.append({
            "class" : OBJ.class_name,
            "usd_path" : OBJ.usd_path,
            "translate" : pose["translation"],
            "orient" : pose["rotation"],
            "scale" : scale,
        })
        
    cam_conf1["cam_poses"] = np.array(csr.cal_cam_tf(top_view_camera)).T.tolist()
    cam_conf2["cam_poses"] = np.array(csr.cal_cam_tf(side_view_camera)).T.tolist()
    cam_conf_list = [ cam_conf1, cam_conf2 ]

    Lights_conf = Lights.get_all_state()
    
    platform_rep.usd_path = f"{root_path}/{env_name}/platform_usd/{platform_rep.class_name}.usd"
    
    platform_conf = {
        "name": platform_rep.class_name,
        "usd_path": platform_rep.usd_path,
        "translate": env_conf["position"],
        "orient": euler_angles_to_quat(env_conf["orientation"], degrees=True).tolist(),
        "scale": env_conf["scale"],
        
    }


    save_conf = {
        "envs": env_conf,
        "objects" : obj_conf,
        "platform" :platform_conf,
        "cameras" : cam_conf_list,
        "lights" : Lights_conf,
        "physics_scene" : physics_scene_conf,
    }



    with open(output_path+f"/conf/{scene_name}.json", 'w') as f:
        json.dump(save_conf, f, indent=4)
    
    scene_num+=1

    print("scene save complete : ", scene_name)
    
    print("data_gen_time : ", time.time()-data_gen_time)

simulation_app.close()