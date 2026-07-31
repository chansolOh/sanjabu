import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import KDTree



from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})


from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.utils.rotations import euler_angles_to_quat


from pxr import Usd,UsdShade, Sdf, Gf, UsdGeom

import omni.kit.commands
import os

import carb.settings
settings = carb.settings.get_settings()

import numpy as np

import omni.isaac.core.utils.bounds as bounds_utils



import sys
import json
sys.path.append("/home/cubox/ochansol/isaac_code/python/utils")
import cs_rep_utils as csr




my_world = World(stage_units_in_meters=1.0,
                 physics_dt  = 0.001,
                 rendering_dt = 0.005)
stage = omni.usd.get_context().get_stage()

my_world.reset()


conf_path = "/nas/ochansol/3d_model/peel3_scan_data_2025/objects_conf.json"
with open(conf_path, "r") as f:
    conf = json.load(f)



conf_list = conf[:2]
obj_list = []
for file in conf_list:
    usd_path = file["path"]
    add_reference_to_stage(usd_path, prim_path="/World/"+file["name"])
    prim = stage.GetPrimAtPath(f"/World/{file['name']}").GetChildren()[0]
    obj_list.append(prim)


obj1 = obj_list[0]
obj1.GetAttribute("xformOp:transform").Set(Gf.Vec3f(0, 0, 0))
obj1_mesh = UsdGeom.Mesh(csr.find_targets(obj1, ["Mesh"])[0])
obj1_pcd = obj1_mesh.GetPointsAttr().Get()

obj2 = obj_list[1]
obj2.GetAttribute("xformOp:transform").Set(Gf.Vec3f(0.8, 0, 0.9))
obj2_mesh = UsdGeom.Mesh(csr.find_targets(obj2, ["Mesh"])[0])
obj2_pcd = obj2_mesh.GetPointsAttr().Get()



tree = KDTree(obj1_pcd)
distances, idx = tree.query(obj2_pcd)



