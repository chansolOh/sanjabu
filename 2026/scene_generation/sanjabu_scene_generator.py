
def main(output_root_path,
         env_name, 
         section_name, 
         platform_name, 
         scene_start, 
         scene_end, 
         object_num = 5, 
         ):
    debug = False
    import sys
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": not debug})
    import carb
    print("SceneGen > App_start")
    sys.stdout.flush()
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage


    import omni.isaac.core.prims as Prims
    from omni.isaac.core.utils.rotations import euler_angles_to_quat
    import omni.isaac.core.utils.rotations as rot_utils

    import omni

    import omni.replicator.core as rep
    import omni.graph.core as og
    import omni.kit.commands

    import numpy as np
    import os
    import json
    import numpy as np

    import carb.settings
    settings = carb.settings.get_settings()


    settings.set("/rtx/useTextureStreaming", False)
    settings.set("/rtx/useAsyncTextureUpload", False)
    settings.set("/rtx/textureCacheSize", 0)

    # settings.set_bool(
    #     "/rtx-transient/resourcemanager/texturestreaming/enabled",
    #     True,
    # )
    # settings.set_float(
    #     "/rtx-transient/resourcemanager/texturestreaming/memoryBudget",
    #     0.6,
    # )
    # settings.set_bool("/UJITSO/forceBuilds", True)
    # settings.set_bool(
    #     "/exts/isaacsim.core.throttling/enable_async",
    #     False,
    # )
    # settings.set_bool("/app/asyncRendering", False)
    # settings.set_bool("/omni/replicator/asyncRendering", False)

    import getpass
    sys.path.append(f"/home/{getpass.getuser()}/ochansol/isaac_code/isaac_chansol")
    from Utils.isaac_utils_51 import scan_rep
    from Utils.isaac_utils_51 import rep_utils as csr
    from Utils.isaac_utils_51 import light_set as light
    from Utils.isaac_utils_51 import sanjabu_Writer as SW

    required_writer_version = "2026-08-08-raw-id-labels-v7"
    installed_writer_version = getattr(SW, "SEMANTICS_WRITER_VERSION", None)
    if installed_writer_version != required_writer_version:
        raise RuntimeError(
            "SceneGen semantics writer version mismatch: "
            f"required={required_writer_version}, "
            f"installed={installed_writer_version}. "
            "Update sanjabu_scene_generator.py and sanjabu_Writer.py together."
        )
    print(f"SceneGen > SEMANTICS_FIX:{installed_writer_version}")
    sys.stdout.flush()




    ############# set params
    scene_num = scene_start
    render_set = debug
    object_path_list = [
        # "/nas/ochansol/3d_model/peel3_scan_data_2024",
        # "/nas/ochansol/3d_model/peel3_scan_data_2025",
        "/nas/ochansol/3d_model/peel3_scan_data_2026",
    ]
    root_path = "/nas/ochansol/isaac/sanjabu/envs"

    usd_path = f"{root_path}/{env_name}/{section_name}.usd"

    env_conf = {
        "env_name": env_name,
        "section_name": section_name,
        "platform_name" : "",
        "usd_path": usd_path,
        "position":[0,0,0],
        "orientation":[90,0,0],
        "scale":[0.01,0.01,0.01]
    }

    output_path =  f"{output_root_path}/{env_conf['env_name']}/{env_conf['section_name']}" 

    cam_model_conf_path = "/nas/ochansol/camera_params/percipio_FM855-E1_conf.json"
        
    cam_conf = {
        "name":"",
        "cam_model_conf_path" : cam_model_conf_path,
        "pixel_size" : 0.003,
        "output_size" : (1920,1080),# min object 1920*1280 = 96*54( 5% )
        "clipping_range" : (0.0001, 100000),
        "focus_distance" : 0,
        "f_stop" : 0,
        "cam_poses" : [],
    }

    writer_dict = {
        "rgb"                           : True,
        "bounding_box_2d_loose"         : False,
        "bounding_box_2d_tight"         : True,
        "bounding_box_3d"               : False,
        "distance_to_camera"            : False,
        "distance_to_image_plane"       : True,
        "instance_segmentation"         : True,
        "normals"                       : True,
        "semantic_segmentation"         : False,
        "use_common_output_dir"         : True,
        "pointcloud_include_unlabelled" : True,
        "pointcloud"                    : True
    }


    ######################





    my_world = World(stage_units_in_meters=1.0,
                    physics_dt  = 0.001,
                    rendering_dt = 0.005)
    stage = omni.usd.get_context().get_stage()

    my_world.reset()






    ######## env set


    env_usd = add_reference_to_stage(usd_path=env_conf["usd_path"], 
                                        prim_path="/World/"+env_conf["env_name"])

    env_prim = Prims.XFormPrim(name =env_conf["env_name"], prim_path="/World/"+env_conf["env_name"], 
                                position = env_conf["position"], 
                                orientation = rot_utils.euler_angles_to_quat( env_conf["orientation"], degrees = True), 
                                scale = env_conf["scale"] )
    light_list = csr.find_lights(env_usd)
    Lights = light.Light(light_list)
    Lights.random_trans(0.2, [1])
    Lights.random_exposure()
    Lights.random_intensity()


    ###### parent끼리 중복검사 해야됨





    ######object set
    model_list = []
    for path in object_path_list:
        with open(os.path.join(path, "objects_conf.json"),'r'  ) as f:
            model_list += json.load(f)
    



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

    side_view_camera = rep.create.camera(
        position = [0,0,0],
        # rotation = [],
        # look_at = obj_rep_list[0].node,
        focal_length = cam_conf["focal_length_isaac"], 
        focus_distance = cam_conf["focus_distance"], 
        f_stop = cam_conf["f_stop"], 
        horizontal_aperture = cam_conf["horizontal_aperture"],
        clipping_range = cam_conf["clipping_range"])

    cam_conf1 = cam_conf.copy()
    cam_conf2 = cam_conf.copy()
    cam_conf1["name"] = "top_view_camera"
    cam_conf2["name"] = "side_view_camera"


    print("cam set complete : ", cam_conf1["name"], cam_conf2["name"])


    ######## render set

    render_product_top = rep.create.render_product(top_view_camera, cam_conf["output_size"])
    render_product_side = rep.create.render_product(side_view_camera, cam_conf["output_size"])
    writer = rep.WriterRegistry.get("SanjabuWriter")
    writer.initialize(
        output_dir                      = output_path,
        rgb                             = writer_dict["rgb"],
        bounding_box_2d_loose           = writer_dict["bounding_box_2d_loose"],
        bounding_box_2d_tight           = writer_dict["bounding_box_2d_tight"],
        bounding_box_3d                 = writer_dict["bounding_box_3d"],
        distance_to_camera              = writer_dict["distance_to_camera"],
        distance_to_image_plane         = writer_dict["distance_to_image_plane"],
        instance_segmentation           = writer_dict["instance_segmentation"],
        colorize_instance_segmentation  = False,
        normals                         = writer_dict["normals"],
        semantic_segmentation           = writer_dict["semantic_segmentation"],
        use_common_output_dir           = writer_dict["use_common_output_dir"],
        pointcloud_include_unlabelled   = writer_dict["pointcloud_include_unlabelled"],
        pointcloud                      = writer_dict["pointcloud"]
    )
    # New SceneGen suppresses validation-frame disk writes explicitly. The
    # writer default remains enabled for compatibility with old SceneGen.
    writer.set_disk_writes_enabled(False)
    writer.set_path(output_path,
                    rgb_path = "rgb",
                    bounding_box_path = "bbox",
                    distance_to_image_plane_path = "depth",
                    instance_segmentation_path = "inst_seg",
                    pointcloud_path = "pointcloud",
                    normals_path = "normals",)
    writer.set_cam_name_list([cam_conf1["name"], cam_conf2["name"]])

    # # Attach render_product to the writer

    # instance_seg_annotator = rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast")
    # instance_seg_annotator.attach([render_product_top])
    # depth_cam_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    # depth_cam_annotator.attach([render_product])
    # depth_plane_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    # depth_plane_annotator.attach([render_product])

    # Keep annotator data refreshed on every camera-validation frame. The
    # custom writer suppresses disk I/O until the final frame below.
    writer.attach([render_product_top, render_product_side])
    rep.orchestrator.pause()
    rep.orchestrator.set_capture_on_play(False)

    print("render set complete ")


    ##################################################################################33
    my_world.reset()
    my_world.stop()
    # writer.set_frame(frame_id=0)
    os.makedirs(os.path.join(output_path, platform_name,"conf"), exist_ok=True)
    print("dir making complete : ")

    physics_scene_conf={
        # 'physxScene:enableGPUDynamics': 1, # True
        # 'physxScene:broadphaseType' : "GPU",
        # 'physxScene:collisionSystem' : "PCM",
        
        # 'physxScene:timeStepsPerSecond' : 1000,
        'physxScene:minPositionIterationCount' : 30,
        'physxScene:minVelocityIterationCount' : 1,
        "physics:gravityMagnitude":35,
        # "physxScene:updateType":"Asynchronous",
    }
    for key in physics_scene_conf.keys():
        stage.GetPrimAtPath("/physicsScene").GetAttribute(key).Set(physics_scene_conf[key])
        
        


    platform_area_prims = csr.find_target_name(env_prim.prim,["Mesh"],"platform_area")
    platform_area_prims = [i.GetParent() for i in platform_area_prims if i.GetParent().GetName() == platform_name][0]


    platform_path = platform_area_prims.GetPath().__str__()
    platform_rep = scan_rep.Scan_Rep_Platform(prim_path = platform_path,scale = [1,1,1], class_name = platform_path.split("/")[-1])
   

    my_world.reset()
  
    platform_tf = csr.find_parents_tf(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
    platform_scale = csr.find_parents_scale(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
    platform_rep.set_tf(platform_tf)
    platform_rep.set_scale(platform_scale)


    print("platform set complete")

    sdg_pipe_prim = stage.GetPrimAtPath("/Replicator/SDGPipeline")
    sdg_pipe_children = sdg_pipe_prim.GetChildren()

    def remove_all_objects(obj_rep_all_list, sdg_pipe_prim, sdg_pipe_children):
        rep.orchestrator.wait_until_complete()
        for OBJ in obj_rep_all_list:
            og.GraphController.delete_node(OBJ.node.node.get_prim_path())
            stage.RemovePrim(OBJ.prim.GetPath())
        
        for prim in sdg_pipe_prim.GetChildren():
            if prim not in sdg_pipe_children:
                stage.RemovePrim(prim.GetPath())

    model_class_names = {str(model["name"]).lower() for model in model_list}

    def canonical_object_class(class_name):
        if class_name is None:
            return None
        class_name = str(class_name).lower()
        if class_name in model_class_names:
            return class_name
        path_name = class_name.rstrip("/").rsplit("/", 1)[-1]
        if path_name in model_class_names:
            return path_name
        return None

    def instance_object_classes(writer_data, render_product_name):
        try:
            instance_data = writer_data["annotators"]["instance_segmentation_fast"][render_product_name]
            labels = instance_data["idToLabels"]
        except (KeyError, TypeError):
            return set()
        return {
            canonical_class
            for value in labels.values()
            for class_name in [value.get("class") if isinstance(value, dict) else value]
            for canonical_class in [canonical_object_class(class_name)]
            if canonical_class is not None
        }

    import time
    import select
    print("SceneGen > reset_complete")
    sys.stdout.flush()
    while scene_num<=scene_end:
        print("SceneGen > START")
        sys.stdout.flush()
        print(f"SceneGen > SCENE:{scene_num}")
        sys.stdout.flush()
        data_gen_time = time.time()
        print("####################    scene_num : ",scene_num)
        settings.set("/rtx/rendermode", "RayTraced")
        scene_name = f"{scene_num:04d}"
        

        print("platform_rep : ", platform_rep.prim)


        obj_rep_all_list = [platform_rep]
        size_list = []
        for model_attr in model_list:
            size_list.append(model_attr["size_rank"])

        size_rank = np.random.choice(size_list, replace=False) # 0: small, 1: medium, 2: large
        # size_rank=2
        print("size_rank : ", size_rank)

        size_sampled_model_list = []
        sampled_model_dict = {}

        for model_attr in model_list:
            if model_attr["size_rank"] == size_rank:
                size_sampled_model_list +=[model_attr["name"]]*model_attr["envs"][env_conf["env_name"]]
                sampled_model_dict[model_attr["name"]] = model_attr

        while True:
            sampled_model_list = np.random.choice(size_sampled_model_list, object_num, replace=False)
            if np.unique(sampled_model_list).__len__() < object_num:
                continue
            else:
                break

        for model_attr in sampled_model_list:
            model_attr = sampled_model_dict[model_attr]
            print("model_attr : ", model_attr["name"])
            scan_obj = scan_rep.Scan_Rep(usd_path =  model_attr["path"],
                                    class_name = model_attr["name"],
                                    size = model_attr["size_rank"],)

            obj_rep_all_list.append(scan_obj)
        
        # ##### @@@@@@@@@@@@@ debuging specific object
        # model_attr = [i for i in model_list if i["name"] == "whiteboard_eraser"][0]
        # scan_obj = scan_rep.Scan_Rep(usd_path =  model_attr["path"],
        #                 class_name = model_attr["name"],
        #                 size = model_attr["size_rank"],)
        # obj_rep_all_list.append(scan_obj)
        # ##########################

        for OBJ in obj_rep_all_list[1:]:
            print("set collider for : ", OBJ.class_name)
            OBJ.set_rigidbody_collider()
            # OBJ.set_contact_sensor()
            OBJ.set_physics_material(
                dynamic_friction=0.25,
                static_friction=0.4,
                restitution=0.0
            )



        my_world.reset()
        my_world.stop()


        Lights.random_exposure(val = 1)#, default_exposure = np.random.uniform(1,2.3) )
        Lights.random_temp(val = 300, default_temp = 5800)

        csr.scatter_in_platform_area(obj_rep_all_list[0],obj_rep_all_list)

        obj_rep_list = obj_rep_all_list[1:]
        expected_object_classes = {
            str(obj.class_name).lower() for obj in obj_rep_list
        }
        writer.set_canonical_class_names(
            [obj.class_name for obj in obj_rep_list]
        )
        
        my_world.play()
        obj_rotation_buf = []
        obj_location_buf = []

        for i in range(20):
            my_world.step(render = render_set)
            obj_rotation_buf.append([obj.get_local_pose()["rotation"]for obj in obj_rep_list])
            obj_location_buf.append([obj.get_local_pose()["translation"] for obj in obj_rep_list])

        while True:
            my_world.step(render = render_set)
            del(obj_rotation_buf[0])
            del(obj_location_buf[0])
            obj_rotation_buf.append([obj.get_local_pose()["rotation"]for obj in obj_rep_list])
            obj_location_buf.append([obj.get_local_pose()["translation"] for obj in obj_rep_list])
            # print(np.array(obj_rotation_buf).std(axis=0).max())
            # print(np.array(obj_location_buf).std(axis=0).max())
            if np.array(obj_rotation_buf).std(axis=0).max()<=0.00001 and np.array(obj_location_buf).std(axis=0).max()<=0.0001:
                break
            
            if my_world.current_time>6:
                break

        print("current_time : ",my_world.current_time)
        ########  
        obb_list = []
        for obj in obj_rep_list:
            obb_list.append(obj.get_obb())

        obb_arr = np.vstack(obb_list)
        obb_min = obb_arr.min(axis=0)
        obb_max = obb_arr.max(axis=0)
        center = (obb_min+obb_max)/2

        ########
        with top_view_camera:
            rep.modify.pose(position = [center[0],center[1],center[2]+1.2])
        top_object_classes = set()
        for _ in range(5):
            rep.orchestrator.step()
            top_object_classes = instance_object_classes(
                writer.get_data(), "Replicator"
            )
            if top_object_classes == expected_object_classes:
                break
        if top_object_classes != expected_object_classes:
            print(
                "scene_reset, top-view instance label mismatch: "
                f"expected={sorted(expected_object_classes)}, "
                f"actual={sorted(top_object_classes)}"
            )
            # import pdb; pdb.set_trace()
            remove_all_objects(obj_rep_list, sdg_pipe_prim, sdg_pipe_children)
            continue

        for _ in range(8):
            with side_view_camera:
                rad  = np.random.randint(0,360)/180*np.pi
                dist = 0.8
                x,y,z = dist*np.cos(rad)+center[0], dist*np.sin(rad)+center[1], center[2]+1
                rep.modify.pose(position=(x,y,z),
                                look_at = center,)
            rep.orchestrator.step()
    


            side_object_classes = instance_object_classes(
                writer.get_data(), "Replicator_01"
            )
            if side_object_classes != expected_object_classes:
                print(
                    "side-view instance label mismatch: "
                    f"expected={sorted(expected_object_classes)}, "
                    f"actual={sorted(side_object_classes)}"
                )
                continue
            else:
                break
        if side_object_classes != expected_object_classes:
            remove_all_objects(obj_rep_list, sdg_pipe_prim, sdg_pipe_children)
            continue

        # side_view_bboxes = np.array(writer.get_data()["annotators"]["bounding_box_2d_tight_fast"]["Replicator_01"]["data"].tolist())[:,1:5]
        # side_view_bboxes_xmax = np.max(side_view_bboxes[:,2])>=cam_conf["output_size"][0]
        # side_view_bboxes_ymax = np.max(side_view_bboxes[:,3])>=cam_conf["output_size"][1]
        # side_view_bboxes_min = np.min(side_view_bboxes)<=0
        # if side_view_bboxes_xmax or side_view_bboxes_ymax or side_view_bboxes_min:
        #     continue

        # if debug:
        #     print("SceneGen > GUI inspection mode. Press Enter to continue.")
        #     sys.stdout.flush()
        #     while True:
        #         my_world.step(render=True)
        #         if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        #             sys.stdin.readline()
        #             break

        my_world.pause()
        writer.set_frame(frame_id=scene_num)
            ####
        # rep.orchestrator.run()
        # rep.orchestrator.step()
        # rep.orchestrator.pause()

        # cam_persp = stage.GetPrimAtPath("/OmniverseKit_Persp")
        # cam_font = stage.GetPrimAtPath("/OmniverseKit_Front")
        # cam_top = stage.GetPrimAtPath("/OmniverseKit_Top")
        # cam_right = stage.GetPrimAtPath("/OmniverseKit_Right")

        # stage.RemovePrim(cam_persp.GetPath())
        # stage.RemovePrim(cam_font.GetPath())
        # stage.RemovePrim(cam_top.GetPath())
        # stage.RemovePrim(cam_right.GetPath())

        settings.set("/rtx/rendermode", "PathTracing")

        settings.set("/rtx/pathtracing/spp", 32) 
        settings.set("/rtx/pathtracing/totalSpp", 255)
        settings.set("/rtx/pathtracing/maxBounces", 12)
        settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 12)
        # settings.set("/rtx/pathtracing/optixDenoiser/enabled", False)


        writer.output_path = output_path +"/"+platform_name
        rep.orchestrator.step(
            delta_time=0.0,
            rt_subframes=255,
            wait_for_render=True,
        )
        rep.orchestrator.wait_until_complete()

        final_writer_data = writer.get_data()
        top_object_classes = instance_object_classes(
            final_writer_data, "Replicator"
        )
        side_object_classes = instance_object_classes(
            final_writer_data, "Replicator_01"
        )
        if (
            top_object_classes != expected_object_classes
            or side_object_classes != expected_object_classes
        ):
            print(
                "scene_reset, final instance label mismatch: "
                f"expected={sorted(expected_object_classes)}, "
                f"top={sorted(top_object_classes)}, "
                f"side={sorted(side_object_classes)}"
            )
            remove_all_objects(
                obj_rep_list, sdg_pipe_prim, sdg_pipe_children
            )
            continue

        writer.set_disk_writes_enabled(True)
        try:
            writer.write(final_writer_data)
            rep.BackendDispatch.wait_until_done()
        finally:
            writer.set_disk_writes_enabled(False)

        print("spp complete")
        

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
            
        cam_conf1["cam_poses"] = np.array(csr.cal_cam_node_tf(top_view_camera)).T.tolist()
        cam_conf2["cam_poses"] = np.array(csr.cal_cam_node_tf(side_view_camera)).T.tolist()
        cam_conf_list = [ cam_conf1, cam_conf2 ]

        Lights_conf = Lights.get_all_state()
        
        platform_rep.usd_path = f"{root_path}/{env_name}/platform_usd/{section_name}/{platform_rep.class_name}.usd"
        
        platform_conf = {
            "name": platform_rep.class_name,
            "usd_path": platform_rep.usd_path,
            "translate": env_conf["position"],
            "orient": euler_angles_to_quat(env_conf["orientation"], degrees=True).tolist(),
            "scale": env_conf["scale"],
            
        }


        save_conf = {
            "envs": env_conf,
            "objects" : obj_conf,
            "platform" :platform_conf,
            "cameras" : cam_conf_list,
            "lights" : Lights_conf,
            "physics_scene" : physics_scene_conf,
        }




        with open(writer.output_path+f"/conf/{scene_name}.json", 'w') as f:
            json.dump(save_conf, f, indent=4)
        
        
        scene_num+=1

        print("scene save complete : ", scene_name)
        


        remove_all_objects(obj_rep_list, sdg_pipe_prim, sdg_pipe_children)

        print("SceneGen > data_gen_time : ", time.time()-data_gen_time)
        sys.stdout.flush()
    print("SceneGen > END")
    sys.stdout.flush()
    simulation_app.close()
