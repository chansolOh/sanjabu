import re
import json
import os
import numpy as np



data_sample_num = 10
ALL_DATA_NUM = 10
root_path = "../Data_samples/scene/Home"

inst_seg_path = "inst_seg/cam_00"
bbox_path = "bbox/cam_00"
grasp_path = "grasp"

with open("syntatic_config.json", "r") as f:
    syntatic_config = json.load(f)



def inst_seg_syntax(data):
    errors = []
    rgba_pattern = re.compile(r"^\(\d{1,3}, \d{1,3}, \d{1,3}, \d{1,3}\)$")
    
    for key, value in data.items():
        # Check if the key matches the RGBA pattern
        # import pdb; pdb.set_trace()
        if not isinstance(key, str) or not rgba_pattern.match(key):
            errors.append(f"Invalid RGBA key: {key}")
            continue
        
        # Check if value is a dictionary
        if not isinstance(value, dict):
            errors.append(f"Value for key {key} is not a dictionary.")
            continue
        
        # Check if the dictionary has a "class" key
        if "class" not in value:
            errors.append(f"Key {key} does not contain 'class'.")
            continue
        
        # Check if the "class" value is a string
        if not isinstance(value["class"], str):
            errors.append(f"The 'class' value for key {key} is not a string.")
            continue
        if value["class"] not in syntatic_config["inst_seg_classes"]:
            errors.append(f"Invalid 'class' value for key :" +str(value["class"] ))
            continue
    
    return errors

def bbox_syntax(data):
    errors = []
    
    for key, value in data.items():
        if not isinstance(key, str):
            errors.append(f"Invalid class_name key: {key}")
            continue

        if not isinstance(value, list):
            errors.append(f"Value for key {key} is not a list.")
            continue
        

        if not all(isinstance(coord, (int, float)) for coord in value):
            errors.append(f"Value for key '{key}' contains non-numeric elements.")
            continue
        
        xmin, ymin, xmax, ymax = value
        if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0 or xmax>=1920 or ymax>=1080:
            errors.append(
                f"Invalid bounding box for key '{key}': [xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}]."
            )
    
    return errors

def grasp_syntax(data):
    errors = []
    
    for key, value in data.items():
        # Check if the key matches the RGBA pattern
        # import pdb; pdb.set_trace()
        if not isinstance(key, str) or key not in syntatic_config["grasp_classes"]:
            errors.append(f"Invalid class_name key: {key}")
            continue
        
        if len(value)==0:
            # errors.append(f"Value for key {key} is empty.")  ### empty grasp is allowed
            continue
        
        for grasp_data in value:
            for grasp_key, grasp_value in grasp_data.items():
                if not isinstance(grasp_key, str) or grasp_key not in syntatic_config["grasp_attr_classes"]:
                    errors.append(f"Invalid grasp key: {grasp_key}")
                    continue
            
            ###### key check
            if not isinstance(grasp_data["center"], list):
                errors.append(f"Value for key {key} is not a list.")
                continue
            if not isinstance(grasp_data["width"], (int, float)):
                errors.append(f"Value for key {key} is not in (int, float).")
                continue
            if not isinstance(grasp_data["height"], (int, float)):
                errors.append(f"Value for key {key} is not in (int, float).")
                continue
            if not isinstance(grasp_data["angle"], (int, float)):
                errors.append(f"Value for key {key} is not in (int, float).")
                continue
            
            ####### value check
            if grasp_data["center"][0] < 0 or grasp_data["center"][0] >= 1920 or grasp_data["center"][1] < 0 or grasp_data["center"][1] >= 1080:
                errors.append(f"Invalid center for key {key}.")
                continue
            if grasp_data["width"] <=0 or grasp_data["height"] <=0:
                errors.append(f"Invalid width or height for key {key}.")
                continue
            if grasp_data["angle"] < -np.pi or grasp_data["angle"] >np.pi:
                errors.append(f"Invalid angle for key {key}.")
                continue
            if grasp_data["gripper_model"] not in syntatic_config["gripper_models"]:
                errors.append(f"Invalid gripper_model for key {key}.")
                continue

    
    return errors


inst_seg_err_cnt = 0
bbox_err_cnt = 0
grasp_err_cnt = 0

sample_list = np.random.choice(range(ALL_DATA_NUM),size = data_sample_num, replace=False) #np.random.randint(0,ALL_DATA_NUM,data_sample_num)
for scene_num in sample_list:
    with open(os.path.join(root_path,inst_seg_path, f"semantics_mapping_{scene_num:04d}.json"),'r') as f:
        inst_seg_label = json.load(f)
    with open(os.path.join(root_path,bbox_path, f"bbox_{scene_num:04d}.json"),'r') as f:
        bbox_label = json.load(f)
    with open(os.path.join(root_path, grasp_path, f"grasp_{scene_num:04d}.json"),'r') as f:
        grasp_label = json.load(f)
    
    
    inst_seg_err = inst_seg_syntax(inst_seg_label)
    bbox_err = bbox_syntax(bbox_label)
    grasp_err = grasp_syntax(grasp_label)
    if len(inst_seg_err) != 0:
        inst_seg_err_cnt += 1
        print("inst_seg_err : ",inst_seg_err)
    if len(bbox_err) != 0:
        bbox_err_cnt += 1
        print("bbox_err",bbox_err)
    if len(grasp_err) != 0:
        grasp_err_cnt += 1
        print("grasp_err : ",grasp_err)
print("\n")
print("inst_seg acc : ",    (data_sample_num - inst_seg_err_cnt) / data_sample_num *100, "%")
print("bbox acc : ",        (data_sample_num - bbox_err_cnt) / data_sample_num *100, "%")
print("grasp acc : ",       (data_sample_num - grasp_err_cnt) / data_sample_num *100, "%")

    # grasp_err = grasp_syntax(grasp_label)   

