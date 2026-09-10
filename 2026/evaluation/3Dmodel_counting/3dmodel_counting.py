import os
import fnmatch
import json


file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(file_path, "/nas/ochansol/3d_model")
# root_path = os.path.join(file_path, "../../Data/3d_model")
result_path = os.path.join(file_path, "../result/3Dmodel_counting")

obj_name_list = []
# y_dir_list = [i for i in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, i))]
# ["peel3_scan_data_2024", "peel3_scan_data_2025"]

y_dir_list = [os.path.join(root_path,i) for i in ["peel3_scan_data_2024", "peel3_scan_data_2025"]]
for y_dir in y_dir_list:
    obj_dir_list = [i for i in os.listdir(os.path.join(root_path, y_dir)) if os.path.isdir(os.path.join(root_path, y_dir, i)) and not i.startswith('.')  ]
    for obj_dir in obj_dir_list:
        
        obj_list = fnmatch.filter(os.listdir(os.path.join(root_path, y_dir, obj_dir, "edited")), "*.obj")
        if len(obj_list) >1:
            print(obj_list)
        obj_name_list += obj_list
    

# print("Total 3D model count : ", len(obj_name_list))
# print(obj_name_list)

result = {
    "3D_model_count": len(obj_name_list),
    "3D_model_list": sorted(obj_name_list)
}

with open(os.path.join(result_path, "result.json"), 'w') as f:
    json.dump(result, f, indent=4)


