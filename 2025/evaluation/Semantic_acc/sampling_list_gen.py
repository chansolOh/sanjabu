import os
import numpy as np

file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(file_path, "../../Data/dataset_v1")
# root_path = os.path.join(file_path, "../Data_samples")

sampling_num = 1007

sample_list = []


env_list = [i for i in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, i))]
for env in env_list:
    section_list = [i for i in os.listdir(os.path.join(root_path, env)) if os.path.isdir(os.path.join(root_path, env, i))]
    for section in section_list:
        platform_list = [i for i in os.listdir(os.path.join(root_path, env, section)) if os.path.isdir(os.path.join(root_path, env, section, i))]
        for platform in platform_list:
            tmp_list = []
            scene_num_list = [i.strip(".png") for i in os.listdir(os.path.join(root_path, env, section, platform,"rgb","top_view_camera")) if i.endswith('.png') ]
            for scene_num in scene_num_list:
                path = os.path.join(env, section, platform, scene_num)
                tmp_list.append(path)
            tmp_list = sorted(tmp_list, key=lambda x: x.split("/")[-1])
            sample_list += tmp_list
    

random_sample_list = np.random.choice(sample_list, size=sampling_num, replace=False)
# random_sample_list = sample_list
with open(f"{file_path}/sampling_list.txt", "w") as f:
    for sample in random_sample_list:
        f.write(sample + "\n")
