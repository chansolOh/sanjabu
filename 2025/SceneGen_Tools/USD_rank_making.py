import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json

conf_paths = [#"/nas/ochansol/3d_model/peel3_scan_data_2025/objects_conf.json",
              "/nas/ochansol/3d_model/peel3_scan_data_2026/objects_conf.json"]

conf = []
for path in conf_paths:
    with open(path, "r") as f:
        conf += json.load(f)

# ################ viz 
# size_list = []
# for file in conf :
#     size_list.append(file["size"])
#     if file["size"] > 10:
#         print(f"Large object detected: {file['name']} with size {file['size']}")
#         import pdb;pdb.set_trace()
# plt.hist(size_list, bins=50)
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.xlabel('Size')
# plt.ylabel('Frequency')
# plt.title('Histogram of Object Sizes')
# plt.show()


######## ############## size rank making
conf_list2 = [] 
for file in conf:
    if file["size"]<1.5:
        rank=0
    elif file["size"]<2.5:
        rank=1
    else:
        rank=2
    file["size_rank"] = rank
    conf_list2.append(file)
with open(conf_paths[0], "w") as f:
    json.dump(conf_list2, f, indent=4)
