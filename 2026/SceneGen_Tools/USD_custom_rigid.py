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
from pxr import Usd, PhysxSchema, UsdPhysics
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.physx.scripts.physicsUtils import add_physics_material_to_prim

# import planning


root_path = "/nas/ochansol/3d_model/peel3_scan_data_2025"
# with open(os.path.join(root_path, "objects_conf.json"), "r") as f:
#     obj_conf = json.load(f)

from omni.isaac.core.utils.stage import open_stage

# my_world = World(stage_units_in_meters=1.0)
# stage = omni.usd.get_context().get_stage()

# my_world.reset()
# my_world.stop()



#### import box


def set_rigidbody_collider(prims, meshes):
    for prim in prims:
        # import pdb;pdb.set_trace()
        set_list = []
        for mesh in meshes:
            if mesh.HasAttribute("physxConvexDecompositionCollision:maxConvexHulls"):
                set_list.append({
                        "maxConvexHulls":mesh.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Get(),
                        "hullVertexLimit":mesh.GetAttribute("physxConvexDecompositionCollision:hullVertexLimit").Get(),
                        "voxelResolution":mesh.GetAttribute("physxConvexDecompositionCollision:voxelResolution").Get(),
                        "errorPercentage":mesh.GetAttribute("physxConvexDecompositionCollision:errorPercentage").Get(),
                    })
            else:
                set_list.append({
                        "maxConvexHulls":50,
                        "hullVertexLimit":64,
                        "voxelResolution":200000,
                        "errorPercentage":10,
                    })
        physx_utils.setRigidBody(prim, "convexDecomposition", False)
        # physx_utils.setRigidBody(prim, "meshSimplification", False)
        # physx_utils.setRigidBody(prim, "sdf", False)
        prim.GetAttribute("physxRigidBody:maxAngularVelocity").Set(720)
        prim.GetAttribute("physxRigidBody:maxLinearVelocity").Set(2.5)
        prim.GetAttribute("physxRigidBody:linearDamping").Set(0.7)
        prim.GetAttribute("physxRigidBody:enableCCD").Set(True)
    for mesh, settings in zip(meshes, set_list):
        mesh.GetAttribute('physxCollision:contactOffset').Set(0.000001)
        mesh.GetAttribute('physxCollision:restOffset').Set(0)
        mesh.GetAttribute("physxConvexDecompositionCollision:shrinkWrap").Set(True)
        # mesh.GetAttribute("physxConvexDecompositionCollision:voxelResolution").Set(600000)
        mesh.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Set(settings["maxConvexHulls"])
        mesh.GetAttribute("physxConvexDecompositionCollision:hullVertexLimit").Set(settings["hullVertexLimit"])
        mesh.GetAttribute("physxConvexDecompositionCollision:voxelResolution").Set(settings["voxelResolution"])
        mesh.GetAttribute("physxConvexDecompositionCollision:errorPercentage").Set(settings["errorPercentage"])

def edit_rigidbody_collider(prims, meshes):
    for prim in prims:
        physx_utils.setRigidBody(prim, "convexDecomposition", False)
        # physx_utils.setRigidBody(prim, "meshSimplification", False)
        # physx_utils.setRigidBody(prim, "sdf", False)
        prim.GetAttribute("physxRigidBody:maxAngularVelocity").Set(720)
        prim.GetAttribute("physxRigidBody:maxLinearVelocity").Set(2.5)
        prim.GetAttribute("physxRigidBody:linearDamping").Set(0.7)
        prim.GetAttribute("physxRigidBody:enableCCD").Set(True)
    for mesh in meshes:
        if mesh.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Get() ==240:
            mesh.GetAttribute('physxCollision:contactOffset').Set(0.000001)
            mesh.GetAttribute('physxCollision:restOffset').Set(0)
            mesh.GetAttribute("physxConvexDecompositionCollision:shrinkWrap").Set(True)
            mesh.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Set(80)
            mesh.GetAttribute("physxConvexDecompositionCollision:hullVertexLimit").Set(64)
            mesh.GetAttribute("physxConvexDecompositionCollision:voxelResolution").Set(200000)
            mesh.GetAttribute("physxConvexDecompositionCollision:errorPercentage").Set(10)

def set_physics_material(stage, prims, meshes):
    if  not stage.GetPrimAtPath(os.path.join(str(prims.GetPath()), "PhysicsMaterial")).IsValid():
        physics_material = PhysicsMaterial(
            prim_path=os.path.join(str(prims.GetPath()), "PhysicsMaterial"), 
            dynamic_friction=0.3, 
            static_friction=0.4,
            restitution=0.5
        )

        for mesh in meshes:
            add_physics_material_to_prim(stage, mesh, physics_material.prim.GetPath())
            UsdPhysics.MassAPI.Apply(mesh)


def remove_physics(prim):
    # physx_utils.removeRigidBody(prim)
    physx_utils.removePhysics(prim)
def set_physx(prim):
    physx_utils.setPhysics(prim,False)

cnt=0



obj_dir_list = sorted([i for i in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, i)) and not i.startswith(".")])
for conf in obj_dir_list:
    usd_path = os.path.join(root_path, conf, "edited", f"{conf}.usd")
    open_stage(usd_path)
    # my_world = World(stage_units_in_meters=1.0)
    # my_world.reset()
    stage = omni.usd.get_context().get_stage()
    try:
        prims = stage.GetPrimAtPath("/World")
    except:
        print("err : ",usd_path)
        continue
    child_prims = prims.GetChildren()
    # remove_physics(prims)
    for child in child_prims:
        if child.GetName() == "Looks" : pass #remove_physics(child)
        else: set_physx(child)

    stage.Export(usd_path.split(".usd")[0] + "_rigid.usd")
    print("complete : ", child_prims[0].GetName())

        # org_prim = add_reference_to_stage(usd_path = usd_path, prim_path = f"/{obj_name}")

        # stage.SetDefaultPrim(stage.GetPrimAtPath(f"/{obj_name}"))

        # stage.Flatten().Export(f"{root_path}/{obj_name}/edited/{obj_name}_test.usd")

        # stage.RemovePrim(org_prim.GetPath())
        # print(obj_name)
        # cnt+=1
        # if cnt>3:
        #     break
    





# my_world = World(stage_units_in_meters=1.0)
# my_world.reset()
# my_world.stop()
# while True:
#     my_world.step(True)

simulation_app.close()


