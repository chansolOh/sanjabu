import omni.replicator.core as rep
import omni.usd
from omni.physx.scripts import utils as physx_utils
import omni.isaac.core.utils.bounds as bounds_utils
from omni.isaac.core.utils.rotations import euler_angles_to_quat, euler_to_rot_matrix, quat_to_rot_matrix

from typing import Callable, Dict, List, Tuple, Union
from pxr import Sdf, Tf, Usd, UsdGeom, UsdShade, Gf, PhysxSchema, UsdPhysics
try:
    from isaacsim.util.debug_draw import _debug_draw
except :
    from omni.isaac.debug_draw import _debug_draw
import numpy as np
import carb


import my_rep

import os

# from omni.physx import get_physx_interface, get_physx_simulation_interface
# from omni.physx import get_physx_scene_query_interface
from omni.physx.scripts.physicsUtils import *
# from omni.isaac.core.materials.physics_material import PhysicsMaterial
import omni.isaac.core.utils.rotations as rot_utils
import cs_utils as csu
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree, cKDTree
import ast




def find_targets(prims, target_list):
    ls = [prims]
    result_ls = []
    cnt = 0
    while len(ls)>cnt:
        prim = ls[cnt]
        if prim.GetTypeName() in target_list:
            result_ls.append(prim)

        child = prim.GetAllChildren()
        if len(child)==0:
            cnt+=1
            continue
        [ls.append(ch) for ch in child]
        cnt+=1
    return result_ls

def find_target_name(prim, target_type, target_name):
    targets = find_targets(prim, target_type)
    target_list = []
    for target in targets:
        if target_name in target.GetName():
            target_list.append(target)
    return target_list

def find_all_parent_prims(prim, include_self=False):

    prim_list = [prim] if include_self else []
    stage = omni.usd.get_context().get_stage()
    prim = prim.GetParent()
    while prim != stage.GetPrimAtPath("/") :
        prim_list.append(prim)
        prim = prim.GetParent()

    return prim_list

def find_parents_scale(prim, include_self = False):
    scale = np.array([1.0, 1.0, 1.0])
    for pr in find_all_parent_prims(prim, include_self=include_self):
        if pr.HasAttribute("xformOp:scale"):
            scale *= np.array(pr.GetAttribute("xformOp:scale").Get())
        if pr.HasAttribute("xformOp:scale:unitsResolve"):
            scale *= np.array(pr.GetAttribute("xformOp:scale:unitsResolve").Get())
    return scale


def find_parents_tf(prim, include_self = False, include_scale = True):
    tf_list = []
    for pr in find_all_parent_prims(prim, include_self):
        if pr.HasAttribute("xformOp:transform") :
            tf_list.append(Gf.Matrix4f(pr.GetAttribute("xformOp:transform").Get()) )
            continue
        position = Gf.Vec3f(0,0,0)
        scale = Gf.Vec3f(1,1,1)
        scale_res = Gf.Vec3f(1,1,1)
        
        rotX_res = Gf.Rotation(Gf.Vec3d(1, 0, 0),0)
        rotY_res = Gf.Rotation(Gf.Vec3d(0, 1, 0),0)
        rotZ_res = Gf.Rotation(Gf.Vec3d(0, 0, 1),0)
        
        mat3 = Gf.Matrix3f(np.eye(3))
        scale_mat = Gf.Matrix4f(np.eye(4))
        scale_res_mat = Gf.Matrix4f(np.eye(4))
        
        rot_res_mat = Gf.Matrix4f(np.eye(4))
        tf_mat = Gf.Matrix4f(np.eye(4))
        if pr.HasAttribute("xformOp:translate") :
            position = Gf.Vec3f(pr.GetAttribute("xformOp:translate").Get())
        if pr.HasAttribute("xformOp:scale"):
            scale = Gf.Vec3f(pr.GetAttribute("xformOp:scale").Get())
        if pr.HasAttribute("xformOp:scale:unitsResolve"):
            scale_res = Gf.Vec3f(pr.GetAttribute("xformOp:scale:unitsResolve").Get())
        if pr.HasAttribute("xformOp:orient"):
            rotation = pr.GetAttribute("xformOp:orient").Get()
            mat3 = Gf.Matrix3f(np.eye(3)).SetRotate(Gf.Quatf(rotation))
        elif pr.HasAttribute("xformOp:rotateXYZ"):
            rotation = pr.GetAttribute("xformOp:rotateXYZ").Get()
            rot_x = Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation[0])
            rot_y = Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation[1])
            rot_z = Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation[2])
            mat3 = Gf.Matrix3f(rot_x) * Gf.Matrix3f(rot_y) * Gf.Matrix3f(rot_z)
        if pr.HasAttribute("xformOp:rotateX:unitsResolve"):
            rotation = pr.GetAttribute("xformOp:rotateX:unitsResolve").Get()
            rotX_res = Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation)

        if pr.HasAttribute("xformOp:rotateY:unitsResolve"):
            rotation = pr.GetAttribute("xformOp:rotateY:unitsResolve").Get()
            rotY_res = Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation)

        if pr.HasAttribute("xformOp:rotateZ:unitsResolve"):
            rotation = pr.GetAttribute("xformOp:rotateZ:unitsResolve").Get()
            rotZ_res = Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation)

        
        tf_mat = tf_mat.SetRotateOnly(mat3)
        rot_res_mat = rot_res_mat.SetRotateOnly(rotX_res*rotY_res*rotZ_res)
        if include_scale:
            scale_mat = scale_mat.SetScale(scale)
            scale_res_mat = scale_res_mat.SetScale(scale_res)
            tf_mat = tf_mat*rot_res_mat*scale_mat*scale_res_mat  
        
        tf_mat = tf_mat.SetTranslateOnly(position)
        tf_list.append(tf_mat)
    TF = Gf.Matrix4f(np.eye(4))

    for tf in tf_list:   ### tf = real tf.T , A.T * B.T = B*A
        TF =  TF*tf
    return TF

def tf_to_pos_rot_scale(tf:Union[Gf.Matrix4f, Gf.Matrix4d]):
    position = np.array(tf.ExtractTranslation())
    rot_mat = np.array(tf)[:3,:3].T
    scale = np.linalg.norm(rot_mat, axis=0)
    rot_mat /= scale  # Normalize the rotation matrix to get the rotation part
    rotation = rot_utils.rot_matrix_to_quat(rot_mat)

    return{
        "position": position,
        "rotation": rotation,
        "scale": scale
    }

def find_lights(prims):
    ls = [prims]
    light_ls = []
    cnt = 0
    while len(ls)>cnt:
        prim = ls[cnt]
        if prim.GetTypeName()[-5:] == "Light":
            light_ls.append(prim)
            cnt+=1
            continue
        child = prim.GetAllChildren()
        if len(child)==0:
            cnt+=1
            continue
        [ls.append(ch) for ch in child]
        cnt+=1
    return light_ls

    
def cal_cam_tf(camera_rep):
    camera_usd_prim = find_targets(camera_rep.get_input_prims()["primsIn"][0], ["Camera"])
    return find_parents_tf(camera_usd_prim[0], include_self=True)
    
    

def np_to_GfVec3f(arr):
    return Gf.Vec3f(arr[0],arr[1],arr[2])
def np_to_GfVec3d(arr):
    return Gf.Vec3d(arr[0],arr[1],arr[2])

def np_to_GfQuatf(arr):
    return Gf.Quatf(arr[0], np_to_GfVec3f(arr[1:4]) )
def np_to_GfQuatd(arr):
    return Gf.Quatd(arr[0], np_to_GfVec3d(arr[1:4]) )

    


def set_random_tf(obj_prim:Usd.Prim, center_position, scale, rotation=180):
    random_rotation = R.random()
    x,y,z = random_rotation.as_euler('xyz', degrees=True)
    
    obj_prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d( tuple((np.random.rand(3)-0.5 )*np.array(scale) + np.array(center_position))    ))
    if obj_prim.HasAttribute('xformOp:rotateXYZ'):
        obj_prim.GetAttribute('xformOp:rotateXYZ').Set(Gf.Vec3d(x,y,z))
    elif obj_prim.HasAttribute('xformOp:orient'):
        obj_prim.GetAttribute('xformOp:orient').Set( np_to_GfQuatf(rot_utils.euler_angles_to_quat((x,y,z), degrees=True)))


def set_random_tf_in_boundary(obj_prim:Usd.Prim, boundary, rotation=180): # boundary = [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    random_rotation = R.random()
    x,y,z = random_rotation.as_euler('xyz', degrees=True)

    pos_x = np.random.uniform(boundary[0][0], boundary[1][0])
    pos_y = np.random.uniform(boundary[0][1], boundary[1][1])
    pos_z = np.random.uniform(boundary[0][2], boundary[1][2])
    
    obj_prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d( tuple((pos_x, pos_y, pos_z))    ))
    if obj_prim.HasAttribute('xformOp:rotateXYZ'):
        obj_prim.GetAttribute('xformOp:rotateXYZ').Set(Gf.Vec3d(x,y,z))
    elif obj_prim.HasAttribute('xformOp:orient'):
        obj_prim.GetAttribute('xformOp:orient').Set( np_to_GfQuatf(rot_utils.euler_angles_to_quat((x,y,z), degrees=True)))

    
def obb_col_check_pair(obj1,obj2): # obj1, obj2 = prim_path, sdf.path
    c1_obb = obj1.get_obb()
    c2_obb = obj2.get_obb()
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

def scatter3D_obb(rep_list, center_position, scale, drop_out:bool = True, fixed_first = False, auto_expand = True): # str path list, sdf.path list
    scale = np.array(scale)
    fixed_list = []
    fixed_list.append(rep_list[0])
    if not fixed_first:
        set_random_tf(rep_list[0].prim, center_position = center_position, scale = scale)
        
    loop_th = 3000
    for obj_rep in rep_list[1:]:
        set_random_tf(obj_rep.prim, center_position = center_position, scale = scale)
        loop = 0
        err_cnt = 0
        while loop< loop_th:
            loop +=1
            if loop >=loop_th:
                if auto_expand:
                    print("auto_expand")
                    err_cnt+=1
                    scale = scale*1.2
                    loop = 0
                    if err_cnt>4:
                        print("scatter_3d_err, obj dropped out : ", obj_rep.class_name)
                        obj_rep.prim.SetActive(False)
                else:
                    print("scatter_3d_err")
            pass_cnt = 0
            for fixed in fixed_list:
                if not obb_col_check_pair(fixed,obj_rep):
                    pass_cnt+=1
                    if pass_cnt>=len(fixed_list):
                        fixed_list.append(obj_rep)
                        loop = loop_th    
                        break
                else:
                    if loop>= loop_th and drop_out:
                        # print("coll")
                        obj_rep.prim.SetActive(False)
                        break
                    set_random_tf(obj_rep.prim,center_position = center_position,scale = scale)
                    break
    # for fixed in fixed_list:
    #     debug_draw_obb(fixed.get_init_obb())

def scatter3D_obb_in_boundary(rep_list, boundary, drop_out:bool = True, fixed_first = False, auto_expand = ['x','y','z']): # str path list, sdf.path list
    fixed_list = []
    fixed_list.append(rep_list[0])
    bound_cnt = np.random.uniform(boundary[0],boundary[1])#np.mean(boundary, axis=0)
    bound_cnt[2] = boundary[0][2]  # z is fixed to the bottom of the boundary
    expand_size = 0.15
    bound = np.array([bound_cnt-expand_size, bound_cnt+expand_size])
    bound = bound.clip(boundary[0], boundary[1])  # Ensure the boundary is within the specified limits

    if not fixed_first:
        set_random_tf_in_boundary(rep_list[0].prim, bound, rotation=180)
        
    loop_th = 1000
    for obj_rep in rep_list[1:]:
        set_random_tf_in_boundary(obj_rep.prim, bound, rotation=180)

        loop = 0
        err_cnt = 0
        while loop< loop_th:
            loop +=1
            if loop >=loop_th:
                if len(auto_expand)>0:
                    print("auto_expand : ", auto_expand)
                    err_cnt+=1

                    expand_size += 0.15

                    bound = np.array([bound_cnt-expand_size, bound_cnt+expand_size])
                    bound = bound.clip(boundary)

                    loop = 0
                    if err_cnt>6:
                        print("scatter_3d_err, obj dropped out : ", obj_rep.class_name)
                        obj_rep.prim.SetActive(False)
                else:
                    print("scatter_3d_err")
            pass_cnt = 0
            for fixed in fixed_list:
                if not obb_col_check_pair(fixed,obj_rep):
                    pass_cnt+=1
                    if pass_cnt>=len(fixed_list):
                        fixed_list.append(obj_rep)
                        loop = loop_th    
                        break
                else:
                    if loop>= loop_th and drop_out:
                        # print("coll")
                        obj_rep.prim.SetActive(False)
                        break
                    set_random_tf_in_boundary(obj_rep.prim, bound, rotation=180)
                    break
    # for fixed in fixed_list:
    #     debug_draw_obb(fixed.get_obb())



def cam_auto_zoom(camera_rep,writer):
    pass



def top_view_cam_move(camera_rep, writer, target_rep, distance = 1):
    target_points = np.array(get_global_points(target_rep))
    target_points_idx = np.random.choice(np.arange(len(target_points)), 50000 if len(target_points)>50000 else len(target_points), replace=False)
    target_points_dist_samples = kd_tree_sampling_xy(target_points[target_points_idx], 0.03)
    
    while True:
        cam_tpds_idx = np.random.choice(np.arange(len(target_points_dist_samples)), 1, replace=False)
        z_max = target_points_dist_samples.max(axis=0)[2]
        cam_tpds = target_points_dist_samples[cam_tpds_idx][0]
        with camera_rep:
            rep.modify.pose(position = [cam_tpds[0], cam_tpds[1], z_max + distance])
        rep.orchestrator.step()
        
        inst_seg = writer.get_data()["annotators"]['instance_segmentation_fast']["Replicator"]["data"]
        inst_seg_label = writer.get_data()["annotators"]['instance_segmentation_fast']["Replicator"]["idToSemantics"]
        for key in inst_seg_label.keys():
            if inst_seg_label[key]["class"] == target_rep.class_name:
                inst_seg_rgba = (key[3]<<8*3) + (key[2]<<8*2) + (key[1]<<8) + key[0]
                
        uni_val, counts = np.unique(inst_seg, return_counts=True)

        if ((counts[np.where(uni_val == inst_seg_rgba)[0]]/inst_seg.size) <0.6)[0]: ## target object is not enough in camera view
            continue
        else:
            break

        
    pcd_inst = writer.get_data()["annotators"]['pointcloud']["Replicator"]["pointInstance"]
    pcd_data = writer.get_data()["annotators"]['pointcloud']["Replicator"]["data"]
    

    
    pcd_idx = np.where(pcd_inst == inst_seg_rgba)[0]
    target_pcd = pcd_data[pcd_idx]
    target_pcd_sample_idx = np.random.choice(np.arange(len(target_pcd)), 50000 if len(target_pcd)>50000 else len(target_pcd), replace=False)
    target_pcd_sample = target_pcd[target_pcd_sample_idx]
    target_pcd_dist_samples = kd_tree_sampling_xy(target_pcd_sample, 0.03)


    # debug_draw_points(target_pcd_dist_samples,size=10, color =[0.8,0.1,0.6])
    return target_pcd_dist_samples








def scatter_on_target_object(camera_rep, writer, rep_list, distance = 1):
    scatter_base_pos = top_view_cam_move(camera_rep, writer, rep_list[0], distance)
    #### first target fix
    b_max = scatter_base_pos.max(axis=0)
    b_min = scatter_base_pos.min(axis=0)
    b_max[2] += distance
    scatter3D_obb_in_boundary(rep_list, boundary=[b_min,b_max], fixed_first=True, auto_expand=["z"])

def scatter_in_specipic_area( target_rep,rep_list):

    if not hasattr(target_rep,"platform_area") or target_rep.platform_area.__len__()<1: raise ValueError("target_rep must have platform_area attribute")
    area_box = np.random.choice(target_rep.platform_area,1)[0]
    area_obb = get_obb(area_box)
    # debug_draw_obb(area_obb)
    scatter3D_obb_in_boundary(rep_list, boundary=np.array([area_obb.min(0),area_obb.max(0)]), fixed_first=False)
    

def get_obb(prim):
    cache = bounds_utils.create_bbox_cache()
    obb = np.array(bounds_utils.compute_obb_corners(cache,prim.GetPath()))

    return obb




    
def kd_tree_sampling(points, min_distance):
    sampled_points = []
    tree = KDTree(points)
    
    for i, point in enumerate(points):
        if len(sampled_points) == 0 or tree.query(point, k=2)[0][1] >= min_distance:
            sampled_points.append(point)
            tree = KDTree(sampled_points)  # 새롭게 트리를 갱신
    return np.array(sampled_points)

def kd_tree_sampling_xy(points, min_distance):
    # x, y 정보만 추출
    xy_points = points[:, :2]  # 2차원 (x, y)만 사용
    sampled_points = []
    tree = KDTree(xy_points)  # KD-Tree 생성

    for point in points:
        # x, y만 비교
        distances, _ = tree.query(point[:2], k=2)
        if len(sampled_points) == 0 or distances[1] >= min_distance:  # 자기 자신 제외
            sampled_points.append(point)
            xy_points = np.array([p[:2] for p in sampled_points])  # 새롭게 KD-Tree 갱신
            tree = KDTree(xy_points)
    
    return np.array(sampled_points)

def check_overlap_by_distance(points1, points2, threshold=1e-2):
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    # 각 점에서 다른 쪽 점까지의 최소 거리 확인
    dists1, _ = tree1.query(points2, k=1)
    dists2, _ = tree2.query(points1, k=1)

    return np.any(dists1 < threshold) or np.any(dists2 < threshold)

    
def align_to_size(rep_list):
    size_list = []
    for rep_obj in rep_list:
        size_list.append(rep_obj.size)
    #### size list 에 맞게 가장 큰것부터 차례로 정렬해서 output list에 저장
    size_list = np.array(size_list)
    sorted_idx = np.argsort(size_list)[::-1]
    return np.array(rep_list)[sorted_idx].tolist()






def get_global_points(target_rep):
    target_prim = target_rep.prim
    target_meshes = find_targets(target_prim,["Mesh"])
    global_points = []
    for mesh in target_meshes:
        target_mesh = UsdGeom.Mesh(mesh)
        target_points = target_mesh.GetPointsAttr().Get()
        tf = find_parents_tf(mesh, include_self=True)
        # transform_matrix = UsdGeom.Xformable(target_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # global_points+=[transform_matrix.Transform(point) for point in target_points]
        global_points+=[tf.Transform(point) for point in target_points]
    return global_points

def get_global_points_sampled(target_rep, sample_distance = 0.03):
    target_points = np.array(get_global_points(target_rep))
    target_points_idx = np.random.choice(np.arange(len(target_points)), 50000 if len(target_points)>50000 else len(target_points), replace=False)
    target_points_dist_samples = kd_tree_sampling_xy(target_points[target_points_idx], sample_distance)
    return target_points_dist_samples

def debug_draw_obb(obb):
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.draw_lines(
        [carb.Float3(i) for i in obb[[0,1,2,2,0,4,5,5,7,6,7,6]]] , 
        [carb.Float3(i) for i in obb[[1,3,3,0,4,5,1,7,6,4,3,2]]] , 
        [carb.ColorRgba(1.0,0.0,0.0,1.0)]*12,
        [1]*12 )
    return draw
def debug_draw_points(points,size = 3, color=[1,0,0]):
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.draw_points(
        [carb.Float3(i) for i in points] , 
        [carb.ColorRgba(color[0],color[1],color[2],1.0)]*len(points),
        [size]*len(points) )
    return draw



def set_cam_zero_rotate(camera_rep):
    camera_usd_prim = find_targets(camera_rep.get_input_prims()["primsIn"][0], ["Camera"])
    if camera_usd_prim[0].HasAttribute('xformOp:rotateXYZ'):
        camera_usd_prim[0].GetAttribute('xformOp:rotateXYZ').Set(Gf.Vec3d(0,0,0))
    elif camera_usd_prim[0].HasAttribute('xformOp:orient'):
        camera_usd_prim[0].GetAttribute('xformOp:orient').Set( np_to_GfQuatf(rot_utils.euler_angles_to_quat((0,0,0), degrees=True)))