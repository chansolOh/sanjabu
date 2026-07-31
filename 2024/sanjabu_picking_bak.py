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
import Doosan_Robot_task as Doosan_Robot_task
# from omni.isaac.sensor import ContactSensor
from omni.isaac.sensor import _sensor
from omni.physx import get_physx_interface, get_physx_simulation_interface
from omni.physx import get_physx_scene_query_interface
from omni.physx.scripts.physicsUtils import *
import cs_utils as cs
import cs_rep_utils as csr


my_world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
# my_robot_task = Doosan_Robot_task.My_Robot_Task(name="robot_task" )
# my_world.add_task(my_robot_task)
my_world.reset()


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




# robot_name = my_robot_task.get_robot_name
# my_robot = my_world.scene.get_object(robot_name)
# my_robot.set_world_pose(position = [0,0,0.056], orientation= euler_angles_to_quat([0,0,0]))



house_usd = add_reference_to_stage(usd_path="/media/nia/6d737125-0a20-46a5-94bc-b44a6aec1a2e/ochansol/isaac_sim/sanjabu/Home/Home_scene1_2204_flatten.usd", 
                                    prim_path="/World/House")

house_prim = Prims.XFormPrim(name ="House", prim_path="/World/House", 
                             position = [1.5,1,0], 
                             orientation = rot_utils.euler_angles_to_quat( [90,0,0], degrees = True), 
                             scale = [0.01,0.01,0.01] )

model_dir_path = "/media/nia/Data/peel3/scan_data_2204"
model_dir_list = sorted(os.listdir(model_dir_path))
model_usd_path_list = [ os.path.join(model_dir_path, i  + "/edited") for i in model_dir_list if len(i.split("."))==1 ][:5]
model_usd_list = []
for path in model_usd_path_list:
    model_usd_list += [i for i in os.listdir(path) if i.split(".")[-1] in ["usd"]]
 




######object pose set

cls_indexes = [0,1,2,3]
obj_rep_list = []
for i, idx in enumerate(cls_indexes):
    obj_rep_list.append(scan_rep.Scan_Rep(usd_path = os.path.join(model_usd_path_list[idx], model_usd_list[idx]),
                               class_name = model_usd_list[idx].split(".")[0]))



# _contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(_on_contact_report_event)

# _contact_sensor_interface = _sensor.acquire_contact_sensor_interface()
# scan_rep.make_scatter_group()



with open("/home/nia/ochansol/isaac/sanjabu/configure/percipio_FM855-E1_conf.json", 'r') as f:
    cam_conf = json.load(f)


######## cam set

((fx,_,cx),(_,fy,cy),(_,_,_))= cam_conf["intrinsic_matrix"]

cam_pixel_size = 0.003
focal_length = (fx+fy)*cam_pixel_size/2

output_img_size = (1920,1080)  # min object 1920*1280 = 96*54( 5% )
# output_img_size = cam_conf["rgb_resolution"]
cam_pixel_size = 0.003
cx,cy = output_img_size[0]/2, output_img_size[1]/2

top_view_camera = rep.create.camera(
    position = [0.5,0.5,1],
    rotation = [0,-90,0],
    # look_at =obj_rep_list[0].node,
    focal_length= focal_length, 
    focus_distance=0, 
    f_stop=0, 
    horizontal_aperture = output_img_size[0]*cam_pixel_size,
    clipping_range=(0.0001, 100000))

camera = rep.create.camera(
    position = [1,1,1],
    # rotation = [],
    look_at =obj_rep_list[0].node,
    focal_length= focal_length, 
    focus_distance=0, 
    f_stop=0, 
    horizontal_aperture = output_img_size[0]*cam_pixel_size,
    clipping_range=(0.0001, 100000))

camera_prim = stage.GetPrimAtPath("/Replicator/Camera_Xform/Camera")
camera_xform_prim = stage.GetPrimAtPath("/Replicator/Camera_Xform")

focal_length = (fx+fy)/2







######## render set

render_product = rep.create.render_product(camera, output_img_size)
output_path = "/home/nia/ochansol/isaac/sanjabu/dataset/cubox"
basic_writer = rep.WriterRegistry.get("BasicWriter")
basic_writer.initialize(
    output_dir              =output_path,
    rgb                     =True,
    bounding_box_2d_loose   =False,
    bounding_box_2d_tight   =True,
    bounding_box_3d         =False,
    distance_to_camera      =True,
    distance_to_image_plane =True,
    instance_segmentation   =True,
    normals                 =False,
    semantic_segmentation   =False,
    use_common_output_dir   =True,
)

# # Attach render_product to the writer
instance_seg_annotator = rep.AnnotatorRegistry.get_annotator("instance_segmentation")
instance_seg_annotator.attach([render_product])
depth_plane_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
depth_plane_annotator.attach([render_product])
depth_cam_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
depth_cam_annotator.attach([render_product])

# basic_writer.attach([render_product])


##################################################################################33
my_world.reset()
my_world.stop()

csr.scatter3D_obb(obj_rep_list, 
                      center_position=(0.5,0.5,0.5),
                      scale = (0.3,0.3,0.3),
                      drop_out = False)


stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:enableGPUDynamics').Set(True)
stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:broadphaseType').Set("GPU")
stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:collisionSystem').Set("PCM")
stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:timeStepsPerSecond').Set(300)
stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:minPositionIterationCount').Set(200)
stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:minVelocityIterationCount').Set(200)

for OBJ in obj_rep_list:
    OBJ.set_rigidbody_collider()
    OBJ.set_contact_sensor()
    OBJ.set_physics_material()



my_world.reset()
my_world.stop()

for i in range(100):
    my_world.step(render=True)
inst_seg_img = instance_seg_annotator.get_data()['data']
depth_img = depth_plane_annotator.get_data()
# depth_cam_img = depth_cam_annotator.get_data()
# dy, dx = np.where(depth_cam_img== depth_cam_img)

# depth_img = -(np.cos(np.arctan(np.sqrt((dx-cx)**2+(dy-cy)**2)/( focal_length)  ))*depth_cam_img.flatten()).reshape((output_img_size[1],output_img_size[0]))
ds_y,ds_x = np.where(depth_img>-0.8)
X = (ds_x-cx)/focal_length*depth_img[ds_y,ds_x].flatten()
Y = (ds_y-cy)/focal_length*depth_img[ds_y,ds_x].flatten()


from omni.isaac.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()

cam_to_pt = np.vstack((X,Y,depth_img[ds_y,ds_x].flatten(), np.ones_like(X)))
cam_tf = csr.cal_cam_tf(camera)
# world_to_pt = np.array(cam_tf).T.dot(cs.rot_z(180)).dot(cam_to_pt).T
world_to_pt = np.array(cam_tf).T.dot(cs.rot_x(180)).dot(cam_to_pt).T


idx = np.arange(len(world_to_pt))
idx = np.random.choice(idx,int(len(idx)*0.5),replace=False )
world_to_pt = world_to_pt[idx]

draw.draw_points([carb.Float3(i) for i in world_to_pt] , 
                    [carb.ColorRgba(1.0,0.0,0.0,1.0)]*len(world_to_pt),
                    [0.3]*len(world_to_pt)     )

# import matplotlib.pyplot as plt
# plt.imshow(depth_img)
# plt.show()
# plt.scatter(X,Y,c= depth_img[ds_y,ds_x].flatten(), s=1)
# plt.show()





# settings.set("/rtx/rendermode", "PathTracing")
# settings.set("/rtx/pathtracing/spp", 1) 
# settings.set("/rtx/pathtracing/totalSpp", 128)
# settings.set("/rtx/pathtracing/maxBounces", 12)
# settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 12)
# camera_prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(0.0,0.0,0.0))





# del_list = [i for i in os.listdir(output_path) if len(i.split('.'))>1]

# for ls in del_list: 
#     os.remove(os.path.join(output_path,ls))
# os.makedirs(os.path.join(output_path,"rgb"), exist_ok=True)
# os.makedirs(os.path.join(output_path,"depth"), exist_ok=True)
# os.makedirs(os.path.join(output_path,"inst_seg"), exist_ok=True)
# os.makedirs(os.path.join(output_path,"bbox_2d"), exist_ok=True)


# import pdb;pdb.set_trace()





cnt = 0

while simulation_app.is_running():
    my_world.step(render=True)
    
    
    # import pdb;pdb.set_trace()

    # cnt +=1
    # if cnt>100:

    #     cnt=0


    # cnt+=1
    # if cnt >800:
    #     import pdb;pdb.set_trace()
    # for i in obj_usd_prim_dict.keys():
    #     sensor = _contact_sensor_interface.get_sensor_reading(obj_usd_prim_dict[i]["contact_sensor_path"], use_latest_data = True)
    

    #     print(obj_usd_prim_dict[i]["object_name"]," = " ,sensor.in_contact, end = "")
    # print("\n")
    
    
    
    # if cnt==200:
    #     my_world.play()

#     if my_world.is_playing():

#         if my_world.current_time_step_index <= 1:
#             my_world.reset()



simulation_app.close()