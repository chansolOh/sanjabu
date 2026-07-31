import os
import argparse
import json
parser = argparse.ArgumentParser(description='파라미터 예시')
parser.add_argument('--scene_num', type=int, default =0 ,help='directory')
args = parser.parse_args()

############# set params

render_set = False
data_path  = "/nas/Dataset/Dataset_2025/test_data/Home" 
output_path =  "/nas/Dataset/Dataset_2025/test_data/Home_test"
obj_conf_path = "/nas/ochansol/3d_model/peel3_scan_data_2025/objects_conf.json"

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


with open( os.path.join(data_path,"conf",f"{scene_num:04d}"+".json"), 'r') as f:
    data_conf= json.load(f)

env_conf = data_conf["envs"]
obj_conf = data_conf["objects"]
platform_conf = data_conf["platform"]
cam_conf = data_conf["cameras"]
light_conf = data_conf["lights"]
physx_conf = data_conf["physics_scene"]

top_cam_config = [ i for i in cam_conf if i["name"] == "top_view_camera"][0]
side_cam_config = [ i for i in cam_conf if i["name"] == "side_view_camera"][0]



from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np

import omni.isaac.core.prims as Prims
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import omni.isaac.core.utils.rotations as rot_utils
import omni.isaac.core.utils.prims as prim_utils
import omni




import sys
sys.path.append("/home/cubox/ochansol/isaac_code/python/utils")
import asyncio
import omni.replicator.core as rep

import omni.graph.core as og
import omni.kit.commands
import os
import shutil
import json

import carb.settings
settings = carb.settings.get_settings()

import numpy as np


from omni.isaac.debug_draw import _debug_draw
import scan_rep

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
obj_rep_all_list = []
for obj in obj_conf:
    scan_obj = scan_rep.Scan_Rep(usd_path = obj["usd_path"],
                                        class_name = obj['class'])
    obj_rep_all_list.append(scan_obj)

    scan_obj.set_pose(position = obj["translate"], rotation = rot_utils.quat_to_euler_angles(obj["orient"], degrees=True) )
    scan_obj.set_scale(scale = obj["scale"])
    scan_obj.set_rigidbody_collider()

######## platform set



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
    





print("scene_num : ",scene_num)
settings.set("/rtx/rendermode", "RayTraced")

my_world.reset()

my_world.stop()


rep.orchestrator.step()
rep.orchestrator.step()
# writer.set_frame(scene_num)


while True:
    my_world.step(render = True)

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