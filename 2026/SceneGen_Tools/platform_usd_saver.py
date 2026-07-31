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
sys.path.append("/home/uon/ochansol/isaac_code/isaac_chansol")
from Utils.isaac_utils_51 import rep_utils as csr
# # import planning


root_path = "/nas/ochansol/3d_model/peel3_scan_data_2026"
obj_dir_list = sorted([ i for i in os.listdir(root_path) if os.path.isdir(os.path.join(root_path,i)) and not i.startswith(".")    ])

from omni.isaac.core.utils.stage import open_stage


class platform_usd:
    def __init__(self, prim):
        self.prim = prim
        self.prim_path = prim.GetPrimPath().__str__()
        self.name = self.prim_path.split("/")[-1]
        






env_usd_root_path = "/nas/ochansol/isaac/sanjabu/envs"


env_name = "Manufactory" # "Home" # "Logistic_site" # "Manufactory"
section_name = "Seongju_Melon_Processing_Facility"#"LivingRoom_Kitchen" #FOODnamoo # FOODnamoo_poultry_plant
env_path_dict = {
    "Home" : f"{env_usd_root_path}/{env_name}/{section_name}.usd",
    "Logistic_site" : f"{env_usd_root_path}/{env_name}/{section_name}.usd",
    "Manufactory" : f"{env_usd_root_path}/{env_name}/{section_name}.usd"
}



if not os.path.exists(f"{env_usd_root_path}/{env_name}/platform_usd"):
    os.makedirs(f"{env_usd_root_path}/{env_name}/platform_usd")
if not os.path.exists(f"{env_usd_root_path}/{env_name}/platform_usd/{section_name}"):
    os.makedirs(f"{env_usd_root_path}/{env_name}/platform_usd/{section_name}")

for i in range(50):
    usd_path = env_path_dict[env_name]
    save_assets_path = f"{env_usd_root_path}/{env_name}/platform_usd/{section_name}"
    asset_name_list = [ i.split(".")[0] for i in os.listdir(save_assets_path) if i.endswith(".usd") or i.endswith(".usda") or i.endswith(".usdc")]

    open_stage(usd_path)
    stage = omni.usd.get_context().get_stage()
    my_world = World(stage_units_in_meters=1.0)
    my_world.reset()

    world_prim = stage.GetPrimAtPath("/World")
    all_prim = csr.find_target_name(world_prim,["Xform"],"")

    platform_area_prims = csr.find_target_name(world_prim,["Mesh"],"platform_area")
    platform_area_prims = [i.GetParent() for i in platform_area_prims]


    final_target_platform_list = []
    for target in platform_area_prims:
        platform = platform_usd(target)
        if platform.name not in asset_name_list:
            final_target_platform_list.append(platform)

    if len(final_target_platform_list) == 0:
        break


    target = final_target_platform_list[0]

    for prim in all_prim:
        try:
            if "Looks" not in prim.GetName() and prim.GetPath().__str__() not in target.prim_path and target.prim_path not in prim.GetPath().__str__():
                stage.RemovePrim(prim.GetPath())
        except:
            print(f"Error removing prim: {prim.GetPath()}")

    stage.Flatten().Export(os.path.join( save_assets_path, f"{target.name}.usd" ))
    print(f"complete :{target.name}.usd")




simulation_app.close()


