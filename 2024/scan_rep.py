import omni.replicator.core as rep
import omni.usd
from omni.physx.scripts import utils as physx_utils
import omni.isaac.core.utils.bounds as bounds_utils
from omni.isaac.core.utils.rotations import euler_angles_to_quat, euler_to_rot_matrix, quat_to_rot_matrix

from typing import Callable, Dict, List, Tuple, Union
from pxr import Sdf, Tf, Usd, UsdGeom, UsdShade, Gf, PhysxSchema, UsdPhysics
from omni.isaac.debug_draw import _debug_draw
import numpy as np
import carb

import my_rep
from omni.isaac.sensor import _sensor
import os

from omni.physx import get_physx_interface, get_physx_simulation_interface
from omni.physx import get_physx_scene_query_interface
from omni.physx.scripts.physicsUtils import *
from omni.isaac.core.materials.physics_material import PhysicsMaterial
import omni.isaac.core.utils.rotations as rot_utils
import cs_rep_utils as csr



class Scan_Rep(my_rep.rep_usd):
    
    def __init__(self,
                class_name,
                usd_path = None,
                prim_path = "",
                position = [0,0,0], 
                rotation = [0,0,0], 
                scale=[0.1,0.1,0.1],
                semantics: List[Tuple[str, str]] = None, 
                visible = True,
                ) :
        super().__init__(prim_path="object/"+class_name if prim_path == "" else prim_path, 
                        usd_path= usd_path, 
                        semantics=None,
                        count=1, 
                        rigidbody_collider=False,
                        particle_cloth=False,
                        )
        
        self.usd_path = usd_path
        self.class_name = class_name
        self.stage = omni.usd.get_context().get_stage()
        self.prim = self.get_prims()[0]
        self.obb = self.get_init_obb()
        self.contact_prim_path = []
        self.contact_prim_state = 0
        with self.node:
            rep.modify.visibility(visible)
        with self.node:
            rep.modify.pose(position=position, rotation=rotation, scale=scale)
        
        self.set_semantic("class",self.class_name )

        
        self.meshes = self.find_mesh()
        self.mesh_parent_path = self.meshes[0].GetParent().GetPath().pathString

        
        
        
    # def _on_contact_report_event(self, contact_headers, contact_data):
    #     # Check if a collision was because of a player
    #     tmp_list =[]
    #     for contact_header in contact_headers:         
    #         collider_1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
    #         collider_2 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))

    #         if collider_1 == self.prim.GetPath():
    #             if collider_2 not in self.contact_prim_path:
    #                 tmp_list.append(collider_2)
    #         elif collider_2 == self.prim.GetPath():
    #             if collider_1 not in self.contact_prim_path:
    #                 tmp_list.append(collider_1)
                    
    #     if self.contact_prim_state == 0 :
    #         self.contact_prim_path = tmp_list.copy()
    #         self.contact_prim_state = 1
    #     else:
    #         self.contact_prim_path+= tmp_list.copy()
    #         self.contact_prim_state = 0
            

        
    def is_contact(self):
        return self._contact_sensor_interface.get_sensor_reading(self._contact_sensor_path, use_latest_data = True).in_contact




    def find_mesh(self):
        ls = self.get_prims()
        mesh_ls = []
        cnt = 0
        while len(ls)>cnt:
            prim = ls[cnt]
            if prim.GetTypeName() == "Mesh":
                mesh_ls.append(prim)
                cnt+=1
                continue
            child = prim.GetAllChildren()
            if len(child)==0:
                cnt+=1
                continue
            [ls.append(ch) for ch in child]
            cnt+=1
        return mesh_ls
    
    def set_contact_sensor(self):
        success, _isaac_sensor_prim = omni.kit.commands.execute(
            "IsaacSensorCreateContactSensor",
            path="Contact_Sensor",
            parent = self.mesh_parent_path,
            sensor_period=1,
            min_threshold=0.0001,
            max_threshold=100000,
            translation = Gf.Vec3d(0, 0, 0),
        )
        

        self.contact_prim = self.stage.GetPrimAtPath(self.mesh_parent_path)
        self.contact_report = PhysxSchema.PhysxContactReportAPI.Apply(self.contact_prim)
        self.contact_report.CreateThresholdAttr().Set(0)

        self._contact_sensor_interface = _sensor.acquire_contact_sensor_interface()
        self._contact_sensor_path = os.path.join(self.mesh_parent_path, "Contact_Sensor")
        self._contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)

    def set_collider(self, collider_type="convexDecomposition"):
        for prim in self.get_prims():
            physx_utils.setStaticCollider(prim, collider_type)
            for mesh in self.meshes:
                mesh.GetAttribute('physxCollision:contactOffset').Set(0.000001)
                mesh.GetAttribute('physxCollision:restOffset').Set(0)
                mesh.GetAttribute("physxConvexDecompositionCollision:shrinkWrap").Set(True)

            
    def set_rigidbody_collider(self):
        for prim in self.get_prims():
            # import pdb;pdb.set_trace()
            set_list = []
            for mesh in self.meshes:
                if mesh.HasAttribute("physxConvexDecompositionCollision:maxConvexHulls"):
                    set_list.append({
                            "maxConvexHulls":mesh.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Get(),
                            "hullVertexLimit":mesh.GetAttribute("physxConvexDecompositionCollision:hullVertexLimit").Get(),
                            "voxelResolution":mesh.GetAttribute("physxConvexDecompositionCollision:voxelResolution").Get(),
                            "errorPercentage":mesh.GetAttribute("physxConvexDecompositionCollision:errorPercentage").Get(),
                        })
                else:
                    set_list.append({
                            "maxConvexHulls":240,
                            "hullVertexLimit":64,
                            "voxelResolution":700000,
                            "errorPercentage":8,
                        })
            physx_utils.setRigidBody(prim, "convexDecomposition", False)
            # physx_utils.setRigidBody(prim, "meshSimplification", False)
            # physx_utils.setRigidBody(prim, "sdf", False)
            prim.GetAttribute("physxRigidBody:maxAngularVelocity").Set(720)
            prim.GetAttribute("physxRigidBody:maxLinearVelocity").Set(2.5)
            prim.GetAttribute("physxRigidBody:linearDamping").Set(0.7)
            prim.GetAttribute("physxRigidBody:enableCCD").Set(True)
        for mesh, settings in zip(self.meshes, set_list):
            mesh.GetAttribute('physxCollision:contactOffset').Set(0.000001)
            mesh.GetAttribute('physxCollision:restOffset').Set(0)
            mesh.GetAttribute("physxConvexDecompositionCollision:shrinkWrap").Set(True)
            # mesh.GetAttribute("physxConvexDecompositionCollision:voxelResolution").Set(600000)
            mesh.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Set(settings["maxConvexHulls"])
            mesh.GetAttribute("physxConvexDecompositionCollision:hullVertexLimit").Set(settings["hullVertexLimit"])
            mesh.GetAttribute("physxConvexDecompositionCollision:voxelResolution").Set(settings["voxelResolution"])
            mesh.GetAttribute("physxConvexDecompositionCollision:errorPercentage").Set(settings["errorPercentage"])

    def set_physics_material(self, dynamic_friction=0.25, static_friction=0.4, restitution=0.0):
        if  not self.stage.GetPrimAtPath(os.path.join(self.mesh_parent_path, "PhysicsMaterial")).IsValid():
            physics_material = PhysicsMaterial(
                prim_path=os.path.join(self.mesh_parent_path, "PhysicsMaterial"), 
                dynamic_friction=dynamic_friction, 
                static_friction=static_friction, 
                restitution=restitution
            )

            for mesh in self.meshes:
                add_physics_material_to_prim(self.stage, mesh, physics_material.prim.GetPath())
                UsdPhysics.MassAPI.Apply(mesh)
        # import pdb;pdb.set_trace()
        # physics_material_path = Sdf.Path(os.path.join(self.get_prims()[0].GetPath().pathString, "PhysicsMaterial"))
        # material = self.stage.DefinePrim(physics_material_path, "PhysxMaterial")
        # physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material)

    def set_pose(self, position, rotation):
        self.init_position = position
        self.init_rotation = rotation
        with self.node:
            rep.modify.pose(position = position, rotation = rotation)
    def set_scale(self, scale):
        with self.node:
            rep.modify.pose(scale = scale)
    
    def get_pose(self):
        trans = self.prim.GetAttribute("xformOp:translate").Get()
        if self.prim.HasAttribute("xformOp:rotateXYZ"):
            rot = self.prim.GetAttribute("xformOp:rotateXYZ").Get()
            return {
                "translation" :np.array(trans).tolist(),
                "rotation" : np.array(rot).tolist()
            }
        elif self.prim.HasAttribute("xformOp:orient"):
            rot = self.prim.GetAttribute("xformOp:orient").Get()
            return {
                    "translation" :np.array(trans).tolist(),
                    "rotation" : rot_utils.gf_quat_to_np_array(rot).tolist()
            }
            
    def get_scale(self):
        if self.prim.HasAttribute("xformOp:scale"):
            return np.array(self.prim.GetAttribute("xformOp:scale").Get()).tolist()

    def get_init_obb(self):
        cache = bounds_utils.create_bbox_cache()
        obb = np.array(bounds_utils.compute_obb_corners(cache,self.prim.GetPath()))
        return obb
    
    def get_obb(self):
        tf = np.array(csr.find_parents_tf(self.prim, include_self=True)).T
        return tf.dot(np.vstack((self.obb.T, np.ones(len(self.obb))))).T[:,:3]
    


class Scan_Rep_Platform(my_rep.rep_usd):
    
    def __init__(self,
                class_name,
                usd_path = None,
                prim_path = "",
                position = [0,0,0], 
                rotation = [0,0,0], 
                scale=[0.1,0.1,0.1],
                semantics: List[Tuple[str, str]] = None, 
                visible = True,
                ) :

        super().__init__(prim_path=prim_path, 
                        semantics=None,
                        count=1, 
                        rigidbody_collider=False,
                        particle_cloth=False,
                        )

        self.usd_path = usd_path
        self.class_name = class_name
        self.stage = omni.usd.get_context().get_stage()
        self.prim = self.get_prims()[0].GetChildren()[0].GetChildren()[0]
        self.obb = self.get_obb()
        self.contact_prim_path = []
        self.contact_prim_state = 0
        with self.node:
            rep.modify.visibility(visible)
        with self.node:
            rep.modify.pose(position=position, rotation=rotation, scale=scale)
        
        self.set_semantic("class",self.class_name )

        
        self.meshes = self.find_mesh()
        self.mesh_parent_path = self.meshes[0].GetParent().GetPath().pathString

        
    def is_contact(self):
        return self._contact_sensor_interface.get_sensor_reading(self._contact_sensor_path, use_latest_data = True).in_contact


    def find_mesh(self):
        ls = self.get_prims()
        mesh_ls = []
        cnt = 0
        while len(ls)>cnt:
            prim = ls[cnt]
            if prim.GetTypeName() == "Mesh":
                mesh_ls.append(prim)
                cnt+=1
                continue
            child = prim.GetAllChildren()
            if len(child)==0:
                cnt+=1
                continue
            [ls.append(ch) for ch in child]
            cnt+=1
        return mesh_ls
    
    def set_contact_sensor(self):
        success, _isaac_sensor_prim = omni.kit.commands.execute(
            "IsaacSensorCreateContactSensor",
            path="Contact_Sensor",
            parent = self.mesh_parent_path,
            sensor_period=1,
            min_threshold=0.0001,
            max_threshold=100000,
            translation = Gf.Vec3d(0, 0, 0),
        )
        

        self.contact_prim = self.stage.GetPrimAtPath(self.mesh_parent_path)
        self.contact_report = PhysxSchema.PhysxContactReportAPI.Apply(self.contact_prim)
        self.contact_report.CreateThresholdAttr().Set(0)

        self._contact_sensor_interface = _sensor.acquire_contact_sensor_interface()
        self._contact_sensor_path = os.path.join(self.mesh_parent_path, "Contact_Sensor")
        self._contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)

    def set_collider(self, collider_type="convexDecomposition"):
        for prim in self.get_prims():
            physx_utils.setStaticCollider(prim, collider_type)
            for mesh in self.meshes:
                mesh.GetAttribute('physxCollision:contactOffset').Set(0.000001)
                mesh.GetAttribute('physxCollision:restOffset').Set(0)
                mesh.GetAttribute("physxConvexDecompositionCollision:shrinkWrap").Set(True)


    def set_physics_material(self, dynamic_friction=0.25, static_friction=0.4, restitution=0.0):
        if  not self.stage.GetPrimAtPath(os.path.join(self.mesh_parent_path, "PhysicsMaterial")).IsValid():
            physics_material = PhysicsMaterial(
                prim_path=os.path.join(self.mesh_parent_path, "PhysicsMaterial"), 
                dynamic_friction=dynamic_friction, 
                static_friction=static_friction, 
                restitution=restitution
            )

            for mesh in self.meshes:
                add_physics_material_to_prim(self.stage, mesh, physics_material.prim.GetPath())
                UsdPhysics.MassAPI.Apply(mesh)


    def set_pose(self, position, rotation):
        self.init_position = position
        self.init_rotation = rotation
        with self.node:
            rep.modify.pose(position = position, rotation = rotation)
    def set_scale(self, scale):
        with self.node:
            rep.modify.pose(scale = scale)
    
    def get_pose(self):
        trans = self.prim.GetAttribute("xformOp:translate").Get()
        if self.prim.HasAttribute("xformOp:rotateXYZ"):
            rot = self.prim.GetAttribute("xformOp:rotateXYZ").Get()
            return {
                "translation" :np.array(trans).tolist(),
                "rotation" : np.array(rot).tolist()
            }
        elif self.prim.HasAttribute("xformOp:orient"):
            rot = self.prim.GetAttribute("xformOp:orient").Get()
            return {
                    "translation" :np.array(trans).tolist(),
                    "rotation" : rot_utils.gf_quat_to_np_array(rot).tolist()
            }
            
    def get_scale(self):
        if self.prim.HasAttribute("xformOp:scale"):
            return np.array(self.prim.GetAttribute("xformOp:scale").Get()).tolist()


    def get_obb(self):
        cache = bounds_utils.create_bbox_cache()
        obb = np.array(bounds_utils.compute_obb_corners(cache,self.prim.GetPath()))
        return obb
    