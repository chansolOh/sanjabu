import os
import argparse
import json
parser = argparse.ArgumentParser(description='파라미터 예시')
parser.add_argument('--scene_num', type=int, default =0 ,help='directory')
args = parser.parse_args()

############# set params

render_set = False
object_conf_path = "/media/nia/Data/peel3/scan_data_2024_v1/objects_conf.json"
plane_object_path = "/media/nia/SSD1/ochansol/isaac/USD/etc_assets/plane_object.usd"

env_name = "Home"
output_path =  "/media/nia/SSD1/ochansol/isaac/sanjabu/Dataset_2024/scene_recon/" + env_name
conf_path = "/media/nia/SSD1/ochansol/isaac/sanjabu/Dataset_2024/scene_bak_20241128/" + env_name

writer_dict = {
    "rgb"                           : True,
    "bounding_box_2d_loose"         : False,
    "bounding_box_2d_tight"         : False,
    "bounding_box_3d"               : False,
    "distance_to_camera"            : False,
    "distance_to_image_plane"       : False,
    "instance_segmentation"         : False,
    "normals"                       : False,
    "semantic_segmentation"         : False,
    "use_common_output_dir"         : False,
    "pointcloud_include_unlabelled" : False,
    "pointcloud"                    : False
}



scene_num = args.scene_num


with open( os.path.join(conf_path,"conf",f"{scene_num:04d}"+".json"), 'r') as f:
    config= json.load(f)

env_conf = config["envs"]
obj_conf = config["objects"]
cam_conf = config["cameras"]
light_conf = config["lights"]
physx_conf = config["physics_scene"]
top_cam_config = [ i for i in cam_conf if i["name"] == "top_view_camera"][0]
side_cam_config = [ i for i in cam_conf if i["name"] == "side_view_camera"][0]

flag = False
for obj_c in obj_conf:
    if obj_c["class"] == "gas": flag = True
if not flag: exit()









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



######################





my_world = World(stage_units_in_meters=1.0,
                physics_dt  = 0.001,
                rendering_dt = 0.005)
stage = omni.usd.get_context().get_stage()

my_world.reset()



######## env set

env_conf["usd_path"] = "omniverse://192.168.0.27/ochansol/sanjabu/envs/Home/Home_scene1_2204_flatten.usd"
env_usd = add_reference_to_stage(usd_path=env_conf["usd_path"], 
                                    prim_path="/World/"+env_conf["name"])

env_prim = Prims.XFormPrim(name =env_conf["name"], prim_path="/World/"+env_conf["name"], 
                            position = env_conf["position"], 
                            orientation = rot_utils.euler_angles_to_quat( env_conf["orientation"], degrees = True), 
                            scale = env_conf["scale"] )

light_list = csr.find_lights(env_usd)
Lights = light.Light(light_list)
Lights.set_each_setting(light_conf)

# Lights.random_intensity()





######object set
with open(object_conf_path,'r'  ) as f:
    obj_attr = json.load(f)
model_list = [i for i in obj_attr if env_conf["name"] in i["envs"]]
# model_list = [i for i in obj_attr if env_conf["name"] in i["envs"]]
model_usd_list = np.array(sorted([i["path"] for i in model_list]))[:5]
obj_rep_all_list = []
for obj in obj_conf:
    
    ###### specified path for each PC
    path = obj["usd_path"]
    if not os.path.exists(path):
        path = "/media/nia/Data/peel3"+path[20:]
    scan_obj = scan_rep.Scan_Rep(usd_path = path,
                                        class_name = obj['class'])

    
    obj_rep_all_list.append(scan_obj)
    
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



######## cam set

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






######## render set

render_product_top = rep.create.render_product(top_view_camera, top_cam_config["output_size"])
render_product_side = rep.create.render_product(side_view_camera, side_cam_config["output_size"])
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
                pointcloud_path = "pointcloud")
writer.set_cam_name_list([top_cam_config["name"], side_cam_config["name"]])

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


##################################################################################33
my_world.reset()
my_world.stop()
csr.set_cam_zero_rotate(top_view_camera)
csr.set_cam_zero_rotate(side_view_camera)
# writer.set_frame(frame_id=0)
os.makedirs(os.path.join(output_path,"conf"), exist_ok=True)



for key in physx_conf.keys():
    stage.GetPrimAtPath("/physicsScene").GetAttribute(key).Set(physx_conf[key])
    
settings.set("/rtx/pathtracing/spp", 1) 
settings.set("/rtx/pathtracing/totalSpp", 128)
settings.set("/rtx/pathtracing/maxBounces", 12)
settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 12)
    
    
del_list = [i for i in os.listdir(output_path) if len(i.split('.'))>1]

# for ls in del_list:
#     os.remove(os.path.join(output_path,ls))


plane_object = scan_rep.Scan_Rep(usd_path = plane_object_path,
                                class_name = "plane",
                                prim_path = "plane",
                                visible=False,
                                scale = [2,2,2]
                                )



print("scene_num : ",scene_num)
settings.set("/rtx/rendermode", "RayTraced")

writer.set_frame(scene_num)
my_world.reset()

my_world.stop()


rep.orchestrator.step()
rep.orchestrator.step()
# writer.set_frame(scene_num)


# while True:
#     my_world.step(render = render_set)

my_world.pause()
settings.set("/rtx/rendermode", "PathTracing")
for i in range(128):
    my_world.step(render = render_set)
print("spp complete")
    ####

# rep.orchestrator.run()
writer.set_frame(scene_num)
rep.orchestrator.step()
# import pdb;pdb.set_trace()
# rep.orchestrator.pause()
for i in range(5):
    my_world.step(render = render_set)

        
    
    


simulation_app.close()