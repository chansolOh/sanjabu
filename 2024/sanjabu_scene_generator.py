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
import Doosan_Robot_task as Doosan_Robot_tas
# from omni.isaac.sensor import ContactSensor
from omni.isaac.sensor import _sensor
from omni.physx import get_physx_interface, get_physx_simulation_interface
from omni.physx import get_physx_scene_query_interface
from omni.physx.scripts.physicsUtils import *
import cs_utils as cs
import cs_rep_utils as csr
import light_set as light
import sanjabu_Writer as sw


############# set params
render_set = False
object_path = "/nas/ochansol/3d_model/peel3_scan_data" #"/scan_data_2204/objects_conf.json"
plane_object_path = "/nas/ochansol/isaac/USD/etc_assets/plane_object.usd"
platform_path = "/World/Home/Carpet"
env_conf = {
    "name": "Home",
    "usd_path": "/nas/ochansol/isaac/sanjabu/envs/Home/Home_scene1_2204_flatten.usd",
    "position":[1.5, 1, -0.002],
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
Lights_conf = Lights.get_all_state()
# Lights.random_intensity()





######object set
with open(os.path.join(object_path, "objects_conf.json"),'r'  ) as f:
    obj_attr = json.load(f)
model_list = [i for i in obj_attr if env_conf["name"] in i["envs"]]
model_usd_list = np.array(sorted([os.path.join(object_path, i["path"]) for i in model_list]))[:10]
obj_rep_all_list = []
for i, idx in enumerate(range(len(model_usd_list))):
    print(model_usd_list[idx])
    scan_obj = scan_rep.Scan_Rep(usd_path =  model_usd_list[idx],
                               class_name = model_usd_list[idx].split(".")[0].split("/")[-1] )
    obj_rep_all_list.append(scan_obj)
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
    
settings.set("/rtx/pathtracing/spp", 1) 
settings.set("/rtx/pathtracing/totalSpp", 128)
settings.set("/rtx/pathtracing/maxBounces", 12)
settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 12)
    
    
del_list = [i for i in os.listdir(output_path) if len(i.split('.'))>1]

# for ls in del_list:
#     os.remove(os.path.join(output_path,ls))

for OBJ in obj_rep_all_list:
    print("set collider for : ", OBJ.class_name)
    OBJ.set_rigidbody_collider()
    # OBJ.set_contact_sensor()
    OBJ.set_physics_material(
        dynamic_friction=0.1,
        static_friction=0.3,
        restitution=0.0
    )

# plane_object = scan_rep.Scan_Rep(usd_path = plane_object_path,
#                                  class_name = "plane",
#                                  prim_path = "plane",
#                                  visible=True,
#                                  scale = [2,2,2]
#                                  )

platform_rep = scan_rep.Scan_Rep_Platform(prim_path = platform_path,scale = [1,1,1], class_name = "Carpet")
my_world.reset()
platform_rep.set_pose(position = env_prim.get_world_pose()[0] ,rotation = rot_utils.quat_to_euler_angles(env_prim.get_world_pose()[1], degrees = True ))
platform_rep.set_scale(scale = env_prim.get_world_scale())
platform_rep.set_collider()
print("platform set complete")

scene_num  = 0 
while scene_num<10:
    print("scene_num : ",scene_num)
    settings.set("/rtx/rendermode", "RayTraced")
    scene_name = f"{scene_num:04d}"
    obj_rep_list = [platform_rep]
    active_idx = np.random.choice(len(obj_rep_all_list), 5, replace=False)
    for i, obj in enumerate(obj_rep_all_list):
        if i in active_idx:
            obj.prim.SetActive(True)
            obj_rep_list.append(obj)
            print(obj.prim)
        else:
            obj.prim.SetActive(False)

    Lights.random_exposure(val=0.5)
    my_world.reset()

    my_world.stop()
    # my_world.play()
    
    
    # platform_obb = platform_rep.get_obb()
    # csr.debug_draw_obb(platform_rep.get_obb())
    # platform_pos = platform_obb.mean(axis=0)
    # platform_pos[2] = platform_obb.max(axis=0)[2]
    # csr.debug_draw_points([platform_pos])
    # platform_scale = platform_obb.max(axis=0)-platform_obb.min(axis=0)


    # csr.top_view_cam_move(top_view_camera, writer, obj_rep_list[0], 1)
    # csr.scatter_on_object(obj_rep_list[1:], obj_rep_list[0])
    
    
    # csr.scatter3D_obb(obj_rep_list, 
    #                     center_position=platform_pos+np.array([0,0,0.15]),
    #                     scale = (platform_scale[0],platform_scale[1],0.3),
    #                     drop_out = False,
    #                     fixed_first=True)


    csr.scatter3D_obb(obj_rep_list, 
                    center_position=(0,0,0.1),
                    scale = (0.25,0.4,0.1),
                    drop_out = False,
                    fixed_first=True)



    obj_rep_list = obj_rep_list[1:]
    
    my_world.play()
    obj_rotation_buf = []
    obj_location_buf = []
    for i in range(20):
        my_world.step(render = render_set)
        obj_rotation_buf.append([obj.get_pose()["rotation"]for obj in obj_rep_list])
        obj_location_buf.append([obj.get_pose()["translation"] for obj in obj_rep_list])

    while True:
        my_world.step(render = render_set)
        del(obj_rotation_buf[0])
        del(obj_location_buf[0])
        obj_rotation_buf.append([obj.get_pose()["rotation"]for obj in obj_rep_list])
        obj_location_buf.append([obj.get_pose()["translation"] for obj in obj_rep_list])
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
        rep.modify.pose(position = [center[0],center[1],1.2])
    with side_view_camera:
        rad  = np.random.randint(0,360)/180*np.pi
        dist = 0.8
        x,y,z = dist*np.cos(rad), dist*np.sin(rad), 1.3
        rep.modify.pose(position=(x,y,z),
                        look_at = center,)
                                                       #   look_at=( obj_rep_list[0].node ))
    

    rep.orchestrator.step()
    writer.set_frame(scene_num)
    # if writer.get_data()["annotators"]["instance_segmentation_fast"]["Replicator"]["idToSemantics"].keys().__len__()<7:
    #     continue
    # if writer.get_data()["annotators"]["instance_segmentation_fast"]["Replicator_01"]["idToSemantics"].keys().__len__()<7:
    #     continue
    # side_view_bboxes = np.array(writer.get_data()["annotators"]["bounding_box_2d_tight_fast"]["Replicator_01"]["data"].tolist())[:,1:5]
    # side_view_bboxes_xmax = np.max(side_view_bboxes[:,2])>=cam_conf["output_size"][0]
    # side_view_bboxes_ymax = np.max(side_view_bboxes[:,3])>=cam_conf["output_size"][1]
    # side_view_bboxes_min = np.min(side_view_bboxes)<=0
    # if side_view_bboxes_xmax or side_view_bboxes_ymax or side_view_bboxes_min:
    #     continue
    
    
    my_world.pause()
    settings.set("/rtx/rendermode", "PathTracing")
    for i in range(128):
        my_world.step(render = render_set)
    print("spp complete")
        ####
    rep.orchestrator.run()
    rep.orchestrator.step()
    rep.orchestrator.pause()
    for i in range(5):
        my_world.step(render = render_set)


    

    obj_conf = []
    for OBJ in obj_rep_list:
        pose = OBJ.get_pose()
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

    save_conf = {
        "envs": env_conf,
        "objects" : obj_conf,
        "cameras" : cam_conf_list,
        "lights" : Lights_conf,
        "physics_scene" : physics_scene_conf,
    }



    # with open(output_path+f"/conf/{scene_name}.json", 'w') as f:
    #     json.dump(save_conf, f, indent=4)
    
    scene_num+=1

    



simulation_app.close()