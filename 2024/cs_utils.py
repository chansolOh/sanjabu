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

def point_3d_to_2d(point, cam_extrinsic, cam_intrinsic):
    pass
    