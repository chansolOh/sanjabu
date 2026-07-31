from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World

import numpy as np
import omni
import os
import json

import carb.settings
settings = carb.settings.get_settings()

import omni.isaac.core.utils.bounds as bounds_utils
from omni.isaac.core.utils.stage import open_stage

import sys


############ scale check
import json
conf_path = "/nas/ochansol/3d_model/peel3_scan_data_2026/objects_conf.json"
with open(conf_path, "r") as f:
    conf = json.load(f)

conf_list = []
for file in conf:
    usd_path = file["path"]
    open_stage(usd_path)
    print("open stage : ", file["name"])
    stage = omni.usd.get_context().get_stage()

    prim = [i for i in stage.GetPrimAtPath("/World").GetChildren() if i.GetName() != "Looks"][0]  # Assuming the first child is the one we want


    cache = bounds_utils.create_bbox_cache()
    obb = np.array(bounds_utils.compute_obb_corners(cache,prim.GetPath()))
    max_vec = np.max(obb, axis=0)
    min_vec = np.min(obb, axis=0)
    xy_len = np.linalg.norm((max_vec - min_vec)[:2])
    xz_len = np.linalg.norm((max_vec - min_vec)[[0,2]])
    yz_len = np.linalg.norm((max_vec - min_vec)[2:])
    max_len = max(xy_len, xz_len, yz_len)
    orthogonal_len = np.linalg.norm((max_vec - min_vec))
    if max_len > 10:
        raise ValueError(f"Large object detected: {file['name']} with size {max_len}")
    file["size"] = orthogonal_len#max_len

    conf_list.append(file)

with open(conf_path, "w") as f:
    json.dump(conf_list, f, indent=4)


################ viz 
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
# size_list = []
# for file in conf :
#     size_list.append(file["size"])
# # import pdb;pdb.set_trace()
# plt.hist(size_list, bins=50)
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.xlabel('Size')
# plt.ylabel('Frequency')
# plt.title('Histogram of Object Sizes')
# plt.show()


# ################## size rank maiking
# conf_list2 = [] 
# for file in conf_list:
#     if file["size"]<1.5:
#         rank=0
#     elif file["size"]<2.5:
#         rank=1
#     else:
#         rank=2
#     file["size_rank"] = rank
#     conf_list2.append(file)
# with open(conf_path, "w") as f:
#     json.dump(conf_list2, f, indent=4)


simulation_app.close()



