import fnmatch
import os
import json
import numpy as np
from tqdm import tqdm



file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(file_path, "../../Data/dataset_v1")
# root_path = os.path.join(file_path, "../../Data/dataset_v1_real")
result_path = os.path.join(file_path, "../result/Gripper_model_counting")

gripper_models = []

grasp_path = "output_grasp"

sampling_active = True
sampling_num = 100

env_list = [i for i in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, i))]
for env in env_list:
    section_list = [i for i in os.listdir(os.path.join(root_path, env)) if os.path.isdir(os.path.join(root_path, env, i))]
    for section in section_list:
        platform_list = [i for i in os.listdir(os.path.join(root_path, env, section)) if os.path.isdir(os.path.join(root_path, env, section, i))]
        for platform in platform_list:
            print("################################################")
            print("Processing : ", os.path.join(env, section, platform))

            grasp_all_list      = os.listdir(os.path.join(root_path, env,section,platform, grasp_path))
            grasp_list          = fnmatch.filter(grasp_all_list,    "????.json")
            if sampling_active:
                grasp_list      = np.random.choice(grasp_list, size=sampling_num, replace=False)

            platform_gripper_models = []
            for grasp_file in tqdm(grasp_list):
                with open(os.path.join(root_path, env, section, platform, grasp_path,grasp_file), "r") as f:
                    grasp_data = json.load(f)
                temp_gripper_models = []
                for g_data in grasp_data:
                    temp_gripper_models.append(g_data["gripper_model"])
                
                unique_gripper, _ = np.unique(temp_gripper_models, return_counts=True)
                platform_gripper_models += unique_gripper.tolist()


            gripper_models+=platform_gripper_models
            print("unique gripper models : ", len(np.unique(platform_gripper_models)))

print("==============================================")
total_unique_gripper, counts = np.unique(gripper_models, return_counts=True)
print("Total unique gripper models: ", len(total_unique_gripper))
for gm, cnt  in zip(total_unique_gripper, counts):
    print(gm, ": ", cnt)

result = {
    "gripper_model_count": len(total_unique_gripper),
    "gripper_model_list": sorted(total_unique_gripper)
}
with open(os.path.join(result_path, "result.json"), 'w') as f:
    json.dump(result, f, indent=4)

# with open(os.path.join(result_path, "result_real.json"), 'w') as f:
#     json.dump(result, f, indent=4)