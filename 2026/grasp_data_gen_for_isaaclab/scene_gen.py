
import sys
from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": True})
import carb
print("SceneGen > App_start")
sys.stdout.flush()
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage




import omni

import omni.replicator.core as rep
import omni.graph.core as og
import omni.kit.commands


import os
import json
import numpy as np

import carb.settings
settings = carb.settings.get_settings()
settings.set("/rtx/useTextureStreaming", False)
settings.set("/rtx/useAsyncTextureUpload", False)
settings.set("/rtx/textureCacheSize", 0)


sys.path.append("/home/uon/ochansol/isaac_code/isaac_chansol")
from Utils.isaac_utils_51 import scan_rep
from Utils.isaac_utils_51 import rep_utils as csr
from Utils.isaac_utils_51.debug_tools import debug_draw_lines, debug_draw_obb, debug_draw_points, debug_draw_clear
from Utils.general_utils import mat_utils




############# set params
scene_num = 0
render_set = False
object_path_list = ["/nas/ochansol/3d_model/peel3_scan_data_2024", "/nas/ochansol/3d_model/peel3_scan_data_2025"] #"/scan_data_2204/objects_conf.json"
# object_path_list = [ "/nas/ochansol/3d_model/peel3_scan_data_2025"]

######object set
model_list = []
for path in object_path_list:
    with open(os.path.join(path, "objects_conf.json"),'r'  ) as f:
        model_list += json.load(f)
target_model = model_list[-2]

cam_model_conf_path = "/nas/ochansol/camera_params/azure_kinect_conf_new.json"
output_path = f"/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen/{target_model['name']}" 

grasp_point_ann_path = os.path.join(output_path,"pre_grasp")
conf_path = os.path.join(output_path,"conf")
os.makedirs(grasp_point_ann_path, exist_ok=True)
os.makedirs(conf_path, exist_ok=True)



# output_path =  f"{output_root_path}/{env_conf['env_name']}/{env_conf['section_name']}" 

gripper_info_path = "/nas/ochansol/gripper_info/gripper_info.json" 

with open(gripper_info_path, 'r') as f:
        gripper_info = json.load(f)





cam_conf = {
    "name":"",
    "pixel_size" : 0.003,
    "output_size" : (1920,1080),# min object 1920*1280 = 96*54( 5% )
    "clipping_range" : (0.0001, 100000),
    "focus_distance" : 0,
    "f_stop" : 0,
    "cam_poses" : [],
}



######################





my_world = World(stage_units_in_meters=1.0,
                physics_dt  = 0.001,
                rendering_dt = 0.005)
stage = omni.usd.get_context().get_stage()

my_world.reset()












######## cam set

with open(cam_model_conf_path, 'r') as f:
    cam_model_conf = json.load(f)

((fx,_,cx),(_,fy,cy),(_,_,_))= cam_model_conf["intrinsic_matrix"]

cam_conf["focal_length_isaac"] = (fx+fy)/2*cam_conf["pixel_size"]
cam_conf["horizontal_aperture"] = cam_conf["output_size"][0]*cam_conf["pixel_size"]
cam_conf["intrinsic_isaac"] = [[(fx+fy)/2, 0,cam_conf["output_size"][0]/2],
                            [0, (fx+fy)/2, cam_conf["output_size"][1]/2],
                            [0,0,1]]

top_view_camera = rep.create.camera(
    position = [0,0,1],
    rotation = [0,-90,0],
    # look_at =obj_rep_list[0].node,
    focal_length = cam_conf["focal_length_isaac"], 
    focus_distance =cam_conf["focus_distance"], 
    f_stop = cam_conf["f_stop"], 
    horizontal_aperture = cam_conf["horizontal_aperture"],
    clipping_range = cam_conf["clipping_range"])


cam_conf1 = cam_conf.copy()
cam_conf1["name"] = "top_view_camera"



print("cam set complete : ", cam_conf1["name"])


######## render set

render_product_top = rep.create.render_product(top_view_camera, cam_conf["output_size"])
pcd_ann = rep.AnnotatorRegistry.get_annotator("pointcloud")
pcd_ann.attach([render_product_top])


# # Attach render_product to the writer

# instance_seg_annotator = rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast")
# instance_seg_annotator.attach([render_product_top])
# depth_cam_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
# depth_cam_annotator.attach([render_product])
# depth_plane_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
# depth_plane_annotator.attach([render_product])


rep.orchestrator.pause()
rep.orchestrator.set_capture_on_play(False)

print("render set complete ")


##################################################################################33
my_world.reset()
my_world.stop()

physics_scene_conf={
    "physics:gravityMagnitude":0,
}

for key in physics_scene_conf.keys():
    stage.GetPrimAtPath("/physicsScene").GetAttribute(key).Set(physics_scene_conf[key])
    
    


my_world.reset()



sdg_pipe_prim = stage.GetPrimAtPath("/Replicator/SDGPipeline")
sdg_pipe_children = sdg_pipe_prim.GetChildren()

def remove_all_objects(obj_rep_all_list, sdg_pipe_prim, sdg_pipe_children):
    for OBJ in obj_rep_all_list:
        og.GraphController.delete_node(OBJ.node.node.get_prim_path())
        stage.RemovePrim(OBJ.prim.GetPath())
    
    for prim in sdg_pipe_prim.GetChildren():
        if prim not in sdg_pipe_children:
            stage.RemovePrim(prim.GetPath())


obj_rep_list = []

print("model_attr : ", target_model["name"])
scan_obj = scan_rep.Scan_Rep(usd_path =  target_model["path"],
                        class_name = target_model["name"],
                        size = target_model["size_rank"],)

obj_rep_list.append(scan_obj)


scene_end=100

my_world.reset()

while scene_num<=scene_end:


    output_json_list = []
    print("####################    scene_num : ",scene_num)
    scene_name = f"{scene_num:04d}"
    for obj_rep in obj_rep_list:
        csr.set_random_tf(obj_rep.prim, center_position=[0,0,0], scale=0)

    rep.orchestrator.step()

    pcd = pcd_ann.get_data()['data']

    sample_num = 1500
    pcd_sample_idx = np.random.choice(np.arange(len(pcd)), sample_num if len(pcd)>sample_num else len(pcd))
    pcd_sample = pcd[pcd_sample_idx].T
    pcd_dist_sample_3d = mat_utils.dist_based_sampling_3d(pcd_sample, dist_th = 0.01).T

    # debug_draw_points(pcd, color=[0,255,0], size=1)
    # debug_draw_points(pcd_dist_sample_3d, color=[255,0,0], size=3)


            

    # side_view_obj_count = writer.get_data()["annotators"]["instance_segmentation_fast"]["Replicator_01"]["idToSemantics"].keys().__len__()
    # while True:
    #     my_world.step(render=True)
    #     if my_world.is_stopped():
    #         my_world.reset()
    #         for obj_rep in obj_rep_list:
    #             csr.set_random_tf(obj_rep.prim, center_position=[0,0,0], scale=0)


    gripper_type_list =["finger2", "finger3"]
    for gripper_type in gripper_type_list:

        while True:
            random_gripper_name = np.random.choice(list(gripper_info.keys()))
            if gripper_type in gripper_info[random_gripper_name]["type"]:
                break
        # random_gripper_name = "Custom_schunk_parallel"
        # random_gripper_name = "Custom_GEP2016IO"
        # random_gripper_name = "Custom_onrobot"
        # random_gripper_name = "Hitbot_z_efg_100"
        # random_gripper_name = "OnRobot_RG2FTv2"
        # random_gripper_name = "OnRobot_RG6_v1_2"
        # random_gripper_name = "UON_Robotics_Jamin_Gripper"
        # random_gripper_name = "DH_Robotics_DH3"
        # random_gripper_name = "Robotiq_3Finger_Adaptive_Gripper"
        random_gripper_name = "Robotiq_2f140"
        gripper = gripper_info[random_gripper_name]
        # if gripper["type"] not in ["finger3", "finger3_par+allel"]:continue
        output_json = {
            "gripper_model": random_gripper_name,
            "data": [],
        }
    gripper_height = gripper["height"]
    gripper_depth = gripper["depth"]
    finger_thickness = gripper["finger_thickness"]
    gripper_width_list = np.round(np.array([1.0,0.7,0.4])*gripper["width"], 2).tolist()


    for obj in obj_rep_list:
        print(obj.class_name)
        for gripper_width in gripper_width_list:

                if gripper["type"] in ["finger2", "finger2_parallel"]:
                    gripper_finger_bbox = np.array([[ gripper_height/2, -gripper_width/2 -  finger_thickness],
                                                    [ gripper_height/2, -gripper_width/2 ],
                                                    [-gripper_height/2, -gripper_width/2 ],
                                                    [-gripper_height/2, -gripper_width/2 -  finger_thickness],

                                                    [ gripper_height/2,  gripper_width/2 ],
                                                    [ gripper_height/2,  gripper_width/2 +  finger_thickness],
                                                    [-gripper_height/2,  gripper_width/2 +  finger_thickness],
                                                    [-gripper_height/2,  gripper_width/2 ]])
                    ## 우측 하단 꼭지에서 반시계 방향
                    gripper_width_bbox = np.array([[ gripper_height/2, -gripper_width/2],
                                                [ gripper_height/2,  gripper_width/2],
                                                [-gripper_height/2,  gripper_width/2],
                                                [-gripper_height/2, -gripper_width/2]]) 
                    
                elif gripper["type"] in ["finger3", "finger3_parallel"]:

                    def tf_bbox(bbox,gripper):
                        shifted_bbox = []
                        for info in gripper["finger_bbox_info"]:
                            trans = mat_utils.trans([info["pos"][0], info["pos"][1], 0])
                            rot = mat_utils.rot_z(info["rot"])
                            shifted_bbox.append( mat_utils.mat_dot([trans,rot, np.vstack((bbox,np.ones_like(bbox[0])))])[:3] )
                        return np.concatenate(shifted_bbox, axis=1)[:2].T
                    

                    tmp_bbox = np.array([[ gripper_height/2, -gripper_width/2 -  finger_thickness],
                                            [ gripper_height/2, -gripper_width/2 ],
                                            [-gripper_height/2, -gripper_width/2 ],
                                            [-gripper_height/2, -gripper_width/2 -  finger_thickness]])
                    tmp_bbox = tmp_bbox.T
                    tmp_bbox = np.vstack((tmp_bbox,np.zeros_like(tmp_bbox[0])))
                    gripper_finger_bbox = tf_bbox(tmp_bbox,gripper)


                    tmp_bbox = np.array([[ gripper_height/2, -gripper_width/2],
                                            [ gripper_height/2,  0],
                                            [-gripper_height/2,  0],
                                            [-gripper_height/2, -gripper_width/2]]) 
                    tmp_bbox = tmp_bbox.T
                    tmp_bbox = np.vstack((tmp_bbox,np.zeros_like(tmp_bbox[0])))
                    gripper_width_bbox = tf_bbox(tmp_bbox,gripper)
            



                gripper_center_point = np.array([[ 0,0 ]])
                gripper_bbox = np.vstack((gripper_finger_bbox,gripper_width_bbox,gripper_center_point)).T
                gripper_bbox = np.vstack((gripper_bbox,np.zeros_like(gripper_bbox[0])))


                #### point in bbox

                rot_deg_range = range(gripper["yaw_max"],0,-10)
        

                for rot_deg in rot_deg_range:
                    # print("gripper_rot : ",rot_deg)
                    gripper_bbox_3d = mat_utils.rot_z(rot_deg).dot(np.vstack((gripper_bbox,np.ones_like(gripper_bbox[0]))))[:3]
                    gripper_bbox_3d = np.tile(pcd_dist_sample_3d[...,None], (1,1,gripper_bbox_3d.shape[1])) + np.tile(gripper_bbox_3d[None,...],(len(pcd_dist_sample_3d),1,1))
                    point_in_bboxes_idx = mat_utils.select_points(gripper_bbox_3d,pcd_sample)

                    for bbox_num in range(len(gripper_bbox_3d)):
                        if gripper["type"] in ["finger2", "finger2_parallel"]:
                            finger_max_depth = np.max([0 if len(point_in_bboxes_idx[bbox_num][0])==0 else pcd_sample[2][point_in_bboxes_idx[bbox_num][0]].max(),  
                                                        0 if len(point_in_bboxes_idx[bbox_num][1])==0 else pcd_sample[2][point_in_bboxes_idx[bbox_num][1]].max()])
                            palm_max_depth   = np.max( 0 if len(point_in_bboxes_idx[bbox_num][2])==0 else pcd_sample[2][point_in_bboxes_idx[bbox_num][2]].max())
                        elif gripper["type"] in ["finger3", "finger3_parallel"]:
                            finger_max_depth = np.max([0 if len(point_in_bboxes_idx[bbox_num][0])==0 else pcd_sample[2][point_in_bboxes_idx[bbox_num][0]].max(),  
                                                        0 if len(point_in_bboxes_idx[bbox_num][1])==0 else pcd_sample[2][point_in_bboxes_idx[bbox_num][1]].max(),
                                                        0 if len(point_in_bboxes_idx[bbox_num][2])==0 else pcd_sample[2][point_in_bboxes_idx[bbox_num][2]].max()])
                            
                            palm_max_depth   = np.max([ 0 if len(point_in_bboxes_idx[bbox_num][3])==0 else pcd_sample[2][point_in_bboxes_idx[bbox_num][3]].max(),
                                                        0 if len(point_in_bboxes_idx[bbox_num][4])==0 else pcd_sample[2][point_in_bboxes_idx[bbox_num][4]].max(),
                                                        0 if len(point_in_bboxes_idx[bbox_num][5])==0 else pcd_sample[2][point_in_bboxes_idx[bbox_num][5]].max()])
                        
                        ########### visualization
                        ##############
                        crit = palm_max_depth - finger_max_depth
                        margin = 0.002 # default = 0.005
                        if crit<0.01:
                            continue

                        if crit>gripper_depth/10*9:
                            target_z = palm_max_depth + margin - gripper_depth/10*9
                        else:
                            target_z = max(palm_max_depth + margin -0.04, finger_max_depth + margin)
                        
                        world_points = pcd_dist_sample_3d[bbox_num].copy()
                        world_points[2] = target_z
                        output_json["data"].append(
                            {
                                "gripper_model" : random_gripper_name,
                                "target_object": obj.class_name,
                                "target_points": world_points.tolist(),
                                "target_orientation": [0,0,rot_deg],
                                "target_width": gripper_width,
                            }
                        )
    output_json_list.append(output_json)


    obj_conf = []

    for OBJ in obj_rep_list:
        pose = OBJ.get_world_pose()
        scale = OBJ.get_scale()
        obj_conf.append({
            "class" : OBJ.class_name,
            "usd_path" : OBJ.usd_path,
            "translate" : pose["translation"],
            "orient" : pose["rotation"],
            "scale" : scale,
        })
        
    cam_conf1["cam_poses"] = np.array(csr.cal_cam_tf(top_view_camera.get_input_prims()["primsIn"][0].GetChildren()[0])).T.tolist()
    cam_conf_list = [ cam_conf1]

    save_conf = {
        "objects" : obj_conf,
        "cameras" : cam_conf_list,
        "physics_scene" : physics_scene_conf,
    }




    with open(os.path.join(conf_path, f"{scene_name}.json"), 'w') as f:
        json.dump(save_conf, f, indent=4)
    with open(os.path.join(grasp_point_ann_path,f"{scene_num:04d}"+".json"), 'w') as f:
        json.dump(output_json_list,f, indent=4)
    
    scene_num+=1

    print("scene save complete : ", scene_name)
    


remove_all_objects(obj_rep_list, sdg_pipe_prim, sdg_pipe_children)
