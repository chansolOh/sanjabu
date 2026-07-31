import fnmatch
import os
root_path = "/nas/Dataset/Dataset_2024/scene_bak_20241128/Home"


rgb_path = "rgb/cam_00"
depth_path = "depth/cam_00"
pcd_path = "pointcloud/cam_00"
inst_seg_path = "inst_seg/cam_00"
bbox_path = "bbox/cam_00"
grasp_path = "grasp"


rgb_all_list        = os.listdir(os.path.join(root_path, rgb_path))
depth_all_list      = os.listdir(os.path.join(root_path, depth_path))
pcd_all_list        = os.listdir(os.path.join(root_path, pcd_path))
inst_seg_all_list   = os.listdir(os.path.join(root_path, inst_seg_path))
bbox_all_list       = os.listdir(os.path.join(root_path, bbox_path))
grasp_all_list      = os.listdir(os.path.join(root_path, grasp_path))

rgb_list            = fnmatch.filter(rgb_all_list,      "????.png")
depth_list          = fnmatch.filter(depth_all_list,    "????.npy")
pcd_point_list      = fnmatch.filter(pcd_all_list,      "pointcloud_????.npy")
pcd_rgb_list        = fnmatch.filter(pcd_all_list,      "pointcloud_rgb_????*.npy")
pcd_seg_list        = fnmatch.filter(pcd_all_list,      "pointcloud_instance_????.npy")
inst_seg_img_list   = fnmatch.filter(inst_seg_all_list, "????.png")
inst_seg_label_list = fnmatch.filter(inst_seg_all_list, "semantics_mapping_????.json")
bbox_list           = fnmatch.filter(bbox_all_list,     "bbox_????.json")
grasp_list          = fnmatch.filter(grasp_all_list,    "grasp_????.json")

print("rgb count : ",           len(rgb_list))
print("depth count : ",         len(depth_list))
print("pcd_point count : ",     len(pcd_point_list))
print("pcd_rgb count : ",       len(pcd_rgb_list))
print("pcd_seg count : ",       len(pcd_seg_list))
print("inst_seg_img count : ",  len(inst_seg_img_list))
print("inst_seg_label count : ",len(inst_seg_label_list))
print("bbox count : ",          len(bbox_list))
print("grasp count : ",         len(grasp_list))
