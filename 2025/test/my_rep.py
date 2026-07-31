import omni.replicator.core as rep
import omni.usd
from omni.physx.scripts import utils as physx_utils
import omni.isaac.core.utils.bounds as bounds_utils
from omni.isaac.core.utils.rotations import euler_angles_to_quat, euler_to_rot_matrix, quat_to_rot_matrix

from typing import Callable, Dict, List, Tuple, Union
from pxr import Sdf, Tf, Usd, UsdGeom, UsdShade, Gf
import numpy as np

REPLICATOR_SCOPE = "/Replicator"






class rep_usd() :

    def __init__(self, prim_path,
                 usd_path = None,
                 semantics: List[Tuple[str, str]] = None, 
                 count: int = 1,
                 rigidbody_collider : bool = True,
                 particle_cloth : bool = False,
                 ) :
        self.stage = omni.usd.get_context().get_stage()
        self.prim_paths = []
        if usd_path is None:
            for cnt,_ in enumerate(range(count)):
                object_name = prim_path.split('/')[-1]
                #### get_stage_next_free_path : stage에 입력한path 가 존재하면 path맨뒤에 001처럼 숫자 부여해서 path 재생성
                xform_path = omni.usd.get_stage_next_free_path(self.stage, f"{REPLICATOR_SCOPE}/{object_name}_rep", False)
                xform = self.stage.DefinePrim(xform_path, "Xform")
                xform.CreateAttribute("replicatorXform", Sdf.ValueTypeNames.Bool).Set(True)
                ref = self.stage.DefinePrim(f"{xform_path}/Ref/{object_name}")
                ref.GetReferences().AddInternalReference(prim_path)
                
                ref_visibility_attr = UsdGeom.Imageable(ref).GetVisibilityAttr()
                ref_visibility_attr.Set(UsdGeom.Tokens.inherited)  # 독립적 Visibility 유지
                # 원본 Prim 가시성 False 설정
                original_prim = self.stage.GetPrimAtPath(prim_path)
                UsdGeom.Imageable(original_prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                
                
                if semantics:
                    rep.utils._set_semantics(xform, semantics)
                self.prim_paths.append(xform_path)
            
            self.node = rep.create.group(self.prim_paths)
        else : 
            for cnt,_ in enumerate(range(count)):
                #### get_stage_next_free_path : stage에 입력한path 가 존재하면 path맨뒤에 001처럼 숫자 부여해서 path 재생성
                xform_path = omni.usd.get_stage_next_free_path(self.stage, f"{REPLICATOR_SCOPE}/{prim_path}", False)
                xform = self.stage.DefinePrim(xform_path, "Xform")
                xform.CreateAttribute("replicatorXform", Sdf.ValueTypeNames.Bool).Set(True)
                ref_path = f"{xform_path}/Ref"
                ref = self.stage.DefinePrim(ref_path)
                ref.GetReferences().AddReference(usd_path)
                # if semantics:
                #     rep.utils._set_semantics(ref, semantics)
                if semantics:
                    rep.utils._set_semantics(xform, semantics)
                self.prim_paths.append(xform_path)
            
            self.node = rep.create.group(self.prim_paths)
        self.count = count
        if rigidbody_collider:
            self.set_rigidbody_collider()
        if particle_cloth:
            self.merge_particle_system()
    def merge_particle_system(self):
        for prim in self.stage.TraverseAll():
            if prim.GetTypeName() == "PhysxParticleSystem":
                if prim.GetParent().GetParent() == self.get_prims()[0]:
                    target_system = prim
                else:
                    prim.SetActive(False)
                    
        for prim in self.get_child_prims():
            if prim.GetRelationship('physxParticle:particleSystem').GetTargets()[0] != target_system:
                prim.GetRelationship('physxParticle:particleSystem').SetTargets([target_system.GetPath()])
        for mat in self.get_child_prims("Material"):
            if mat.GetParent().GetPath() != target_system.GetParent().GetPath():
                mat.SetActive(False)

        #GetRelationship('physxParticle:particleSystem')
        # for idx,prim in enumerate(self.get_prims()):
        #     if idx==0: continue
        #     prim






    def get_prims(self):
        return self.node.get_output_prims()['prims']
    
    def get_child_prims(self, prim_type : str='Mesh' ):
        path_lists = []
        for prim in self.get_prims():
            childs = prim.GetAllChildren()
            for child in childs:
                if child.GetTypeName() == 'Xform':
                    for childs2 in child.GetAllChildren():
                        path_lists += [childs2]if childs2.GetTypeName() == prim_type else []
                elif child.GetTypeName() == prim_type:
                    path_lists += [child]
        return path_lists


    def get_mesh_paths(self):
        path_lists = []
        for prim in self.get_prims():
            childs = prim.GetAllChildren()
            for child in childs:
                if child.GetTypeName() == 'Xform':
                    for childs2 in child.GetAllChildren():
                        path_lists += [childs2.GetPath()] if childs2.GetTypeName() == "Mesh" else []
                elif child.GetTypeName() == "Mesh":
                    path_lists += [child.GetPath()]
        return path_lists
    
    def extract_to_rotmat(self,prims_list):
        prims_attrs = prims_list[0].GetAttributes()
        if prims_list[0].GetAttribute('xformOp:transform') in prims_attrs:
            return np.array([ np.array(prim.GetAttribute('xformOp:transform').Get()).T for prim in prims_list])
        else:
            prims_list_position = np.array([np.array(prim.GetAttribute('xformOp:translate').Get())  for prim in prims_list])
            if prims_list[0].GetAttribute("xformOp:orient") in prims_attrs :
                tf = np.array([
                                np.vstack((
                                    np.hstack((
                                        quat_to_rot_matrix(np.array([
                                            prim.GetAttribute('xformOp:orient').Get().real,
                                            prim.GetAttribute('xformOp:orient').Get().imaginary[0],
                                            prim.GetAttribute('xformOp:orient').Get().imaginary[1],
                                            prim.GetAttribute('xformOp:orient').Get().imaginary[2],
                                            ])),
                                        pos[:,None] )), 
                                    np.array([0,0,0,1]) ))
                                for prim, pos in zip(prims_list,prims_list_position)])
                return tf
            elif prims_list[0].GetAttribute("xformOp:rotateXYZ") in prims_attrs:
                tf = np.array([
                            np.vstack((
                                np.hstack((
                                    euler_to_rot_matrix(prim.GetAttribute('xformOp:rotateXYZ').Get(), degrees=True),
                                    pos[:,None] )), 
                                np.array([0,0,0,1]) ))
                            for prim, pos in zip(prims_list,prims_list_position)])
                return tf

    def get_position(self):
        child_position  = np.array([ np.array(prim.GetAttribute('xformOp:translate').Get()) for prim in self.get_child_prims()])
        child_position = np.hstack((child_position,np.ones((child_position.shape[0],1))))
        
        ref_parent_tf  = self.extract_to_rotmat([ i.GetParent() for i in self.get_child_prims()])
        xform_parent_tf = self.extract_to_rotmat(self.get_prims())


        child_pos = [xform_tf.dot(ref_tf).dot(child_pos[:,None]).T[0,:-1] for child_pos, ref_tf, xform_tf in zip(child_position,ref_parent_tf, xform_parent_tf)]

        # parent_rotation = np.array([np.array(prim.GetAttribute('xformOp:orient').Get())     for prim in self.get_prims()])
        return child_pos

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
            
            
    
    def set_rigidbody_collider(self):
        for prim in self.get_prims():
            # physx_utils.setRigidBody(prim, "convexDecomposition", False)
            # physx_utils.setRigidBody(prim, "meshSimplification", False)
            physx_utils.setRigidBody(prim, "sdf", False)

    def set_semantic(self, class_name, type_name):
        rep.utils._set_semantics(self.node, [(class_name,type_name)])


    def obb_col_check_pair(self,obj1,obj2): # obj1, obj2 = prim_path, sdf.path
        cache = bounds_utils.create_bbox_cache()
        c1_obb = np.array(bounds_utils.compute_obb_corners(cache,obj1))
        c2_obb = np.array(bounds_utils.compute_obb_corners(cache,obj2))

        c1_cnt = np.mean(c1_obb, axis = 0)
        c2_cnt = np.mean(c2_obb, axis = 0)
        dist_vec = c1_cnt - c2_cnt

        c1_box_vec = c1_obb - c1_obb[0]
        c2_box_vec = c2_obb - c2_obb[0]
        c12_ax_concat = np.concatenate(( (c1_box_vec[1]/np.linalg.norm(c1_box_vec[1])) [:,None],
                                        (c1_box_vec[2]/np.linalg.norm(c1_box_vec[2])) [:,None],
                                        (c1_box_vec[4]/np.linalg.norm(c1_box_vec[4])) [:,None],
                                        (c2_box_vec[1]/np.linalg.norm(c2_box_vec[1])) [:,None],
                                        (c2_box_vec[2]/np.linalg.norm(c2_box_vec[2])) [:,None],
                                        (c2_box_vec[4]/np.linalg.norm(c2_box_vec[4])) [:,None]), axis=1)
        c1_dot = c1_box_vec.dot( c12_ax_concat)
        c2_dot = c2_box_vec.dot( c12_ax_concat)
        dist_dot = dist_vec.dot( c12_ax_concat)
        det = np.abs(dist_dot) >= (np.abs(c1_dot.max(axis=0) - c1_dot.min(axis=0)) + np.abs(c2_dot.max(axis=0) - c2_dot.min(axis=0)) )/2
        return not np.any(det)

    def scatter3D_obb(self, center_position, scale, drop_out:bool = True): # str path list, sdf.path list
        obj_list = self.get_child_prims()
        fixed_list = []
        fixed_list.append(obj_list[0])
        self.set_random_trans(obj_list[0].GetParent(),center_position = center_position,scale = scale)
        loop_th = 100
        for obj in obj_list[1:]:
            self.set_random_trans(obj.GetParent(),center_position = center_position,scale = scale)
            loop = 0
            while loop< loop_th:
                loop +=1
                pass_cnt = 0
                for fixed in fixed_list:
                    if not self.obb_col_check_pair(fixed.GetPath(),obj.GetPath()):
                        pass_cnt+=1
                        if pass_cnt>=len(fixed_list):
                            fixed_list.append(obj)
                            loop = loop_th
                            break
                    else:
                        if loop>= loop_th and drop_out:
                            # print("coll")
                            obj.GetParent().GetParent().SetActive(False)
                            break
                        self.set_random_trans(obj.GetParent(),center_position = center_position,scale = scale)
                        break

    def set_random_trans(self,obj_prim:Usd.Prim, center_position, scale, rotation=360):
        obj_prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d( tuple((np.random.rand(3)-0.5)*np.array(scale) + np.array(center_position))    ))
        qw,qx,qy,qz = euler_angles_to_quat(np.random.rand(3)*rotation, degrees=True )
        obj_prim.GetAttribute('xformOp:orient').Set(Gf.Quatf( qw,qx,qy,qz ))

    def scatter_3d(self, center_position, rotation=(0.0, 0.0, 0.0), scale=(1,1,1), prim_type:str = 'cube'):
        if prim_type == 'cube':
            self.scatter_obj = rep.create.cube(position = center_position,
                                                rotation = rotation,
                                                scale = scale, visible=False)
        elif prim_type == 'sphere':
            self.scatter_obj = rep.create.sphere(position = center_position,
                                                rotation = rotation,
                                                scale = scale, visible=False)
        with self.node:
            rep.randomizer.rotation(tuple([-180.0]*3),tuple([180.0]*3))
            rep.randomizer.scatter_3d(self.scatter_obj, check_for_collisions = True)