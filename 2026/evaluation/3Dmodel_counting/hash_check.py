import os
import fnmatch
import hashlib
from tqdm import tqdm


file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(file_path, "../../Data/3d_model")




def sha256_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()




hash_dict = {}
y_dir_list = [os.path.join(root_path,i) for i in ["peel3_scan_data_2024", "peel3_scan_data_2025"]]
for y_dir in y_dir_list:
    obj_dir_list = [i for i in os.listdir(os.path.join(root_path, y_dir)) if os.path.isdir(os.path.join(root_path, y_dir, i)) and not i.startswith('.')  ]
    for obj_dir in tqdm(obj_dir_list):
        obj_list = fnmatch.filter(os.listdir(os.path.join(root_path, y_dir, obj_dir, "edited")), "*.obj")
        hash_dict[os.path.join(root_path,y_dir,obj_dir)] = sha256_hash(os.path.join(root_path, y_dir, obj_dir, "edited", obj_list[0]))




hash_map = {}
dup_list = []

for file_path, hash in hash_dict.items():
    if hash in hash_map:
        dup_list.append((hash_map[hash], file_path))
    else:
        hash_map[hash] = file_path


print("Duplicate 3D model count : ", len(dup_list))