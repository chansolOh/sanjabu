import numpy as np
import os
import tqdm

root_path  = "/nas/Dataset/Dataset_2026/dataset_v2"

for env_dir in os.listdir(root_path):
    env_dir_path = os.path.join(root_path, env_dir)
    if not os.path.isdir(env_dir_path):
        continue

    for section_dir in os.listdir(env_dir_path):
        section_dir_path = os.path.join(env_dir_path, section_dir)
        if not os.path.isdir(section_dir_path):
            continue

        for platform_dir in os.listdir(section_dir_path):
            platform_dir_path = os.path.join(section_dir_path, platform_dir)
            if not os.path.isdir(platform_dir_path):
                continue

            for cam_dir in os.listdir(os.path.join(platform_dir_path,"depth")):
                cam_dir_path = os.path.join(platform_dir_path,"depth", cam_dir)
                if not os.path.isdir(cam_dir_path):
                    continue
                for scene_num in tqdm.tqdm(os.listdir(cam_dir_path)):
                    scene_num_path = os.path.join(cam_dir_path, scene_num)
                    if scene_num_path.endswith(".npy"):
                        depth = np.load(scene_num_path)
                        if np.sum(depth) == np.inf:
                            print(env_dir, section_dir, platform_dir, cam_dir, scene_num)