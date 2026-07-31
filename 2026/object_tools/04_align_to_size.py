from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
# config = {
#     "headless": False,
#     "exts": [
#         {"id": "omni.kit.window.viewport", "enabled": True},
#         {"id": "omni.kit.window.stage", "enabled": True},
#         {"id": "omni.usd", "enabled": True},
#         # Composer에서 기본적으로 사용 중인 확장들을 가능한 활성화
#     ]
# }

# simulation_app = SimulationApp(config)


from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage

import numpy as np
# import Robot_task_suction_2cup as Robot_task

import omni.isaac.core.prims as Prims
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import omni.isaac.core.utils.rotations as rot_utils
import omni.isaac.core.utils.prims as prim_utils
import omni


from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.kit.asset_converter")

import omni.kit.asset_converter

from pxr import Usd,UsdShade, Sdf, Gf, UsdGeom


import omni.kit.commands
import os
import shutil
import json

import carb.settings
settings = carb.settings.get_settings()

from scipy.io import loadmat
import numpy as np

import omni.isaac.core.utils.bounds as bounds_utils

from omni.physx.scripts.physicsUtils import *
from omni.physx.scripts import utils as physx_utils

import sys


# # import planning


from omni.isaac.core.utils.stage import open_stage




my_world = World(stage_units_in_meters=1.0,
                 physics_dt  = 0.001,
                 rendering_dt = 0.005)
stage = omni.usd.get_context().get_stage()

my_world.reset()


conf_path = "/nas/ochansol/3d_model/peel3_scan_data_2026/objects_conf.json"
with open(conf_path, "r") as f:
    conf = json.load(f)



conf_list = sorted(conf, key=lambda x: x["size"])
pos_x = 0
for file in conf_list:
    usd_path = file["path"]
    add_reference_to_stage(usd_path, prim_path="/World/"+file["name"])
    print("added : ", file["name"])
    prim = stage.GetPrimAtPath(f"/World/{file['name']}")

    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(pos_x, 0, 0))

    pos_x+=file["size"]

stage.Export(os.path.join( "/nas/ochansol/3d_model/2026_align_to_size.usd" ))





simulation_app.close()



