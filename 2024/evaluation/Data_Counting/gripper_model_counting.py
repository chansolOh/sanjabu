import fnmatch
import os
import json
import numpy as np

root_path = "../Data_samples/scene/Home"


grasp_path = "grasp"

grasp_all_list      = os.listdir(os.path.join(root_path, grasp_path))

grasp_list          = fnmatch.filter(grasp_all_list,    "grasp_????.json")

gripper_models = []
for grasp_file in grasp_list:
    with open(os.path.join(root_path,grasp_path,grasp_file), "r") as f:
        grasp_data = json.load(f)
    for class_name in grasp_data.keys():
        for grasp in grasp_data[class_name]:
            gripper_models.append(grasp["gripper_model"])

unique_gripper, counts = np.unique(gripper_models, return_counts=True)
for gripper, cnt in zip(unique_gripper, counts):
    print(f"{gripper} : {cnt}")
    
