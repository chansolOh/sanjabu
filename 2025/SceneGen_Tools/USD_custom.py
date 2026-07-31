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


root_path = "/nas/ochansol/3d_model/peel3_scan_data_2025"
obj_dir_list = sorted([ i for i in os.listdir(root_path) if os.path.isdir(os.path.join(root_path,i)) and not i.startswith(".")    ])

from omni.isaac.core.utils.stage import open_stage


#### object_custom
# for obj_name in obj_dir_list:
#     try:
#         usd_path = f"{root_path}/{obj_name}/edited/{obj_name}.usd"
#         # stage = Usd.Stage.Open(usd_path)
#         open_stage(usd_path)
#         stage = omni.usd.get_context().get_stage()
#         prims = stage.GetPrimAtPath(f"/World")
#         # meshes = csr.find_targets(prims, ["Mesh"])
#         prims.GetAttribute("xformOp:scale").Set(Gf.Vec3d([1, 1, 1]))

#         stage.GetRootLayer().Save()
#         print("complete : ", obj_name)

#     except Exception as e:
#         print("error : ",obj_name)
    


###### scene spliter ######################33
# for cnt in range(100):
#     usd_path = "/nas/ochansol/isaac/sanjabu/envs/Manufactory/manufactory_chiken.usd"
#     save_assets_path = "/nas/ochansol/isaac/sanjabu/envs/Manufactory/assets"
#     asset_name_list = [ i.split(".")[0] for i in os.listdir(save_assets_path) if i.endswith(".usd") or i.endswith(".usda") or i.endswith(".usdc")]

#     open_stage(usd_path)
#     stage = omni.usd.get_context().get_stage()
#     world_prim = stage.GetPrimAtPath("/World")
#     target_prim_list = []
#     for prim in world_prim.GetChildren():
#         if not prim.HasPayload():
#             target_prim_list.append(prim)

#     final_target_prim_list = []
#     for target in target_prim_list:
#         if target.GetName() not in asset_name_list:
#             final_target_prim_list.append(target)
#     if len(final_target_prim_list) == 0:
#         break

#     target = final_target_prim_list[0]

#     for prim in world_prim.GetChildren():
#         if prim != target:
#             stage.RemovePrim(prim.GetPath())

#     stage.Export(os.path.join( save_assets_path, target.GetName() + ".usd" ))
#     print(f"complete : {target.GetName()}")



############ scale check
import json
conf_path = "/nas/ochansol/3d_model/peel3_scan_data_2025/objects_conf.json"
with open(conf_path, "r") as f:
    conf = json.load(f)

conf_list = []
for file in conf:
    usd_path = file["path"]
    open_stage(usd_path)
    print("open stage : ", file["name"])
    stage = omni.usd.get_context().get_stage()

    prim = stage.GetPrimAtPath("/World").GetChildren()[0]  # Assuming the first child is the one we want


    cache = bounds_utils.create_bbox_cache()
    obb = np.array(bounds_utils.compute_obb_corners(cache,prim.GetPath()))
    max_vec = np.max(obb, axis=0)
    min_vec = np.min(obb, axis=0)
    xy_len = np.linalg.norm((max_vec - min_vec)[:2])
    xz_len = np.linalg.norm((max_vec - min_vec)[[0,2]])
    yz_len = np.linalg.norm((max_vec - min_vec)[2:])
    max_len = max(xy_len, xz_len, yz_len)
    file["size"] = max_len

    conf_list.append(file)



################ viz 
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
# size_list = []
# for file in conf :
#     size_list.append(file["size"])
# # import pdb;pdb.set_trace()
# plt.hist(size_list, bins=50)
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.xlabel('Size')
# plt.ylabel('Frequency')
# plt.title('Histogram of Object Sizes')
# plt.show()


# ################## size rank maiking
# conf_list2 = [] 
# for file in conf_list:
#     if file["size"]<1.5:
#         rank=0
#     elif file["size"]<2.5:
#         rank=1
#     else:
#         rank=2
#     file["size_rank"] = rank
#     conf_list2.append(file)
# with open(conf_path, "w") as f:
#     json.dump(conf_list2, f, indent=4)


simulation_app.close()



