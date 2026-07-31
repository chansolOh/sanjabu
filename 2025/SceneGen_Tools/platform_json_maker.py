import json
import os

env_name = "Logistic_site"
section_name = "FOODnamoo"  # "LivingRoom_Kitchen" #FOODnamoo # FOODnamoo_poultry_plant
root_path = "/nas/ochansol/isaac/sanjabu/envs"
platform_dir_path = f"{root_path}/{env_name}/platform_usd/{section_name}"

platform_list = [i for i in os.listdir(platform_dir_path) if i.endswith(".usd") or i.endswith(".usda") or i.endswith(".usdc")]



platform_json = {
    env_name :{
        section_name: {
            "platforms": platform_list
        }
    }
}
