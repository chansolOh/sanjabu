import fnmatch
import os
import json
file_path = os.path.dirname(os.path.abspath(__file__))

rgb_cnt = 0
depth_cnt = 0
normal_cnt = 0
inst_seg_img_cnt = 0
inst_seg_label_cnt = 0
bbox_cnt = 0
grasp_cnt = 0
scene_meta_cnt = 0

root_path = os.path.join(file_path, "../../Data/dataset_v1")#"/nas/Dataset/Dataset_2025/dataset_v1"
result_path = os.path.join(file_path, "../result/Data_counting")


env_list = [i for i in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, i))]
for env in env_list:
    section_list = [i for i in os.listdir(os.path.join(root_path, env)) if os.path.isdir(os.path.join(root_path, env, i))]
    for section in section_list:
        platform_list = [i for i in os.listdir(os.path.join(root_path, env, section)) if os.path.isdir(os.path.join(root_path, env, section, i))]
        for platform in platform_list:
            print("################################################")
            print("Processing : ", os.path.join(env, section, platform))
            print("==============================================")
            rgb_path        = "rgb/top_view_camera"
            depth_path      = "depth/top_view_camera"
            normal_path     = "normals/top_view_camera"
            inst_seg_path   = "inst_seg/top_view_camera"
            bbox_path       = "bbox/top_view_camera"
            grasp_path      = "output_grasp"
            scene_meta_path = "scene_meta"


            rgb_all_list        = os.listdir(os.path.join(root_path,  env,section,platform,      rgb_path))
            depth_all_list      = os.listdir(os.path.join(root_path,  env,section,platform,      depth_path))
            normal_all_list     = os.listdir(os.path.join(root_path,  env,section,platform,      normal_path))
            inst_seg_all_list   = os.listdir(os.path.join(root_path,  env,section,platform,      inst_seg_path))
            bbox_all_list       = os.listdir(os.path.join(root_path,  env,section,platform,      bbox_path))
            grasp_all_list      = os.listdir(os.path.join(root_path,  env,section,platform,      grasp_path))
            scene_meta_all_list = os.listdir(os.path.join(root_path,  env,section,platform,      scene_meta_path))

            rgb_list            = fnmatch.filter(rgb_all_list,      "????.png")
            depth_list          = fnmatch.filter(depth_all_list,    "????.npy")
            normal_list         = fnmatch.filter(normal_all_list,   "????.png")
            inst_seg_img_list   = fnmatch.filter(inst_seg_all_list, "????.png")
            inst_seg_label_list = fnmatch.filter(inst_seg_all_list, "semantics_mapping_????.json")
            bbox_list           = fnmatch.filter(bbox_all_list,     "????.json")
            grasp_list          = fnmatch.filter(grasp_all_list,    "????.json")
            scene_meta_list     = fnmatch.filter(scene_meta_all_list,"????.json")

            print("rgb count : ",           len(rgb_list))
            print("depth count : ",         len(depth_list))
            print("normal count : ",        len(normal_list))
            print("inst_seg_img count : ",  len(inst_seg_img_list))
            print("inst_seg_label count : ",len(inst_seg_label_list))
            print("bbox count : ",          len(bbox_list))
            print("grasp count : ",         len(grasp_list))
            print("scene_meta count : ",    len(scene_meta_list))

            rgb_cnt         += len(rgb_list)
            depth_cnt       += len(depth_list)
            normal_cnt      += len(normal_list)
            inst_seg_img_cnt+= len(inst_seg_img_list)
            inst_seg_label_cnt+= len(inst_seg_label_list)
            bbox_cnt        += len(bbox_list)
            grasp_cnt       += len(grasp_list)
            scene_meta_cnt  += len(scene_meta_list)

print("==============================================")
print("Total rgb count : ",            rgb_cnt)
print("Total depth count : ",          depth_cnt)
print("Total normal count : ",         normal_cnt)
print("Total inst_seg_img count : ",   inst_seg_img_cnt)
print("Total inst_seg_label count : ", inst_seg_label_cnt)
print("Total bbox count : ",           bbox_cnt)
print("Total grasp count : ",          grasp_cnt)
print("Total scene_meta count : ",     scene_meta_cnt)

result = {
    "rgb_count": rgb_cnt,
    "depth_count": depth_cnt,
    "normal_count": normal_cnt,
    "inst_seg_img_count": inst_seg_img_cnt,
    "inst_seg_label_count": inst_seg_label_cnt,
    "bbox_count": bbox_cnt,
    "grasp_count": grasp_cnt,
    "scene_meta_count": scene_meta_cnt
}

with open(os.path.join(result_path, "result.json"), 'w') as f:
    json.dump(result, f, indent=4)