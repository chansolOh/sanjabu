import numpy as np
from PIL import Image


def rot_x(deg):
    deg = deg/180*np.pi
    return np.array([[1,0,0,0],
                     [0,np.cos(deg),-np.sin(deg),0],
                     [0,np.sin(deg),np.cos(deg),0],
                     [0,0,0,1]])
    
def rot_y(deg):
    deg = deg/180*np.pi
    return np.array([[np.cos(deg),0,np.sin(deg),0],
                     [0,1,0,0],
                     [-np.sin(deg),0,np.cos(deg),0],
                     [0,0,0,1]])

def rot_z(deg):
    deg = deg/180*np.pi
    return np.array([[np.cos(deg),-np.sin(deg),0,0],
                     [np.sin(deg),np.cos(deg),0,0],
                     [0,0,1,0],
                     [0,0,0,1]])
def dot(matrix_list):
    result = np.eye(4)
    for mat in matrix_list:
        if type(mat) != np.ndarray:
            mat = np.array(mat)
        if mat.shape != (4,4):
            mat = mat_to_tf(mat)
        result = np.dot(result, mat)
    return result


def mat_to_tf(mat):
    if type(mat) == list:
        mat = np.array(mat)
    elif type(mat) != np.ndarray:
        import torch
        eye = torch.eye(4, device=mat.device)
        row,col = mat.shape
        row_mat = eye[row:,:col]
        col_mat = eye[:,col:]
        return torch.concat((torch.concat((mat,row_mat),dim=0),col_mat),dim=1)
    else :
        eye = np.eye(4)
        row,col = mat.shape
        row_mat = eye[row:,:col]
        col_mat = eye[:,col:]
        return np.hstack((np.vstack((mat,row_mat)),col_mat))















def dist_based_sampling(points, dist_th):
    ##### points : (2,N) image index
    idx_dist = np.tile(points[:,None,:],(1,len(points.T),1)) - np.tile(points[...,None],(1,1,len(points.T)))
    idx_dist = np.sqrt(np.sum(idx_dist**2,axis=0))

    cnt = 0
    while len(idx_dist)>cnt:
        min_idx = np.argwhere((idx_dist[cnt]<=dist_th) & (idx_dist[cnt]!=0))
        points = np.delete(points, min_idx, axis=1)
        idx_dist = np.delete(idx_dist, min_idx, axis=0)
        idx_dist = np.delete(idx_dist, min_idx, axis=1)
        cnt+=1
    return points




############     specific task set  #############################
def align_bbox(bbox):
    y_min = np.min(bbox[:,1],axis=0)
    y_min_idx = np.argwhere(bbox[:,1]==y_min)
    if y_min_idx.shape[0]>=2:
        return np.roll(bbox,-max(y_min_idx),axis=0)
    else:
        return np.roll(bbox,-y_min_idx[0],axis=0)

def lin_func(bbox,points):# bbox=2x2, points = 2xN
    x1,y1 = bbox[0]
    x2,y2 = bbox[1]
    if x2-x1 == 0:
        return points[0] - x1
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1    
    return a*points[0] - points[1] + b 

def select_point_in_bbox(bbox,points): # bbox = 4x2, points = 2xN
    aligned_bbox = align_bbox(bbox)
    p1 = np.where(lin_func(aligned_bbox[[3,2]],points)>=0)
    p2 = np.where(lin_func(aligned_bbox[[2,1]],points)>=0)
    p3 = np.where(lin_func(aligned_bbox[[1,0]],points)<=0)
    p4 = np.where(lin_func(aligned_bbox[[0,3]],points)<=0)

    filtered_idx = np.intersect1d(np.intersect1d(p1,p2),np.intersect1d(p3,p4))
    return filtered_idx

def select_points(bboxes,pt) : # bboxes = bboxes x 3 x 13, points = 3 x N
    points = pt.copy()
    idx = []
    for bbox in bboxes:
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox[0].min(), bbox[1].min(), bbox[0].max(), bbox[1].max()
        pts_idx = np.where((points[0]>=bbox_xmin) & (points[0]<=bbox_xmax) & (points[1]>=bbox_ymin) & (points[1]<=bbox_ymax))[0]
        # import pdb;pdb.set_trace()
        tmp = [] # 3x2xN
        for i in range(bbox.shape[1]//4):
            tmp.append(pts_idx[select_point_in_bbox(bbox[:2, i*4:(i+1)*4].T, points[:2, pts_idx])])
        # tmp.append(pts_idx[select_point_in_bbox(bbox[:2,  :4].T, points[:2,pts_idx])])
        # tmp.append(pts_idx[select_point_in_bbox(bbox[:2, 4:8].T, points[:2,pts_idx])])
        # tmp.append(pts_idx[select_point_in_bbox(bbox[:2, 8:12].T,points[:2,pts_idx])])
        idx.append(tmp)

    return idx #boxes x 3 x 2 x N

###############################