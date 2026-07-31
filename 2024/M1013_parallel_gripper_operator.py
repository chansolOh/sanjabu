
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})
from omni.isaac.core import World
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.cortex.cortex_utils import get_assets_root_path
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
import Doosan_Robot_task as Doosan_Robot_task

import omni.isaac.core.prims as Prims
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import omni.isaac.core.utils.prims as prim_utils
import omni
from omni.physx.scripts import utils
from pxr import UsdPhysics
from omni.isaac.core.objects import GroundPlane
from pxr import PhysxSchema, Gf, Sdf, UsdPhysics
from omni.isaac.core import PhysicsContext
import omni.replicator.core as rep
import sys
import getpass
sys.path.append(f"/home/{getpass.getuser()}/ochansol/isaac_code/python/utils/my_rep.py")
import my_rep
# my_world = World(stage_units_in_meters=1.0,
#                  physics_prim_path="/physicsScne",
#                  device="cuda",backend="torch")
# omni.timeline.get_timeline_interface().play()
# physics_context = PhysicsContext(
#                 prim_path = "/physicsScene",
#                 device = "cuda",
#                 backend = "torch",
#                 )
# physics_context.enable_gpu_dynamics(True)

my_world = World(stage_units_in_meters=1.0)

stage = omni.usd.get_context().get_stage()

plane = GroundPlane(prim_path="/World/GroundPlane", z_position=0)
light_1 = prim_utils.create_prim(
    "/World/Light_1",
    "SphereLight",
    position=np.array([4.0, 4.0, 19.0]),
    attributes={
        "inputs:radius": 0.01,
        "inputs:intensity": 5e3,
        "inputs:color": (255, 250, 245),
        "inputs:exposure" : 10,
    }
)


my_robot_task = Doosan_Robot_task.My_Robot_Task(name="robot_task" )
my_world.add_task(my_robot_task)
my_world.reset()
robot_name = my_robot_task.get_robot_name
my_robot = my_world.scene.get_object(robot_name)
my_robot.set_world_pose(position = [0,0,0.056], orientation= euler_angles_to_quat([0,0,0]))


# box_rep = my_rep.rep_usd(usd_path ="/ochansol/isaac_sim/Box_model/box_green.usd",
#                                    prim_path = "box", count=1)
# with box_rep.node:
#     rep.modify.pose(position = (0.5,0,0.01),
#                     rotation = (90,0,90))

# gamja_rep = my_rep.rep_usd(usd_path = "/ochansol/isaac_sim/USD/etc_assets/snack_raw_data_cubox/USD/gamja_cloth_beta2.usd",
#                                 prim_path = "gamja_cloth",count =2,
#                                 rigidbody_collider=False,
#                                 particle_cloth = True)
# gamja_rep.scatter_3d(center_position = (0.5,0,0.5), scale =(0.2,0.3,0.4),prim_type = 'cube')
# gamja_rep.set_semantic("class","gamja")
# box_rep.set_semantic("class", "box")



# camera = rep.create.camera(position=[1,0,2.5], look_at=gamja_rep.node)
# render_product = rep.create.render_product(camera, (1280, 720))
# basic_writer = rep.WriterRegistry.get("BasicWriter")
# basic_writer.initialize(
#     output_dir=f"/ochansol/isaac_sim/render/gamja_data_code",
#     rgb                     =True,
#     bounding_box_2d_loose   =True,
#     bounding_box_2d_tight   =False,
#     bounding_box_3d         =True,
#     distance_to_camera      =True,
#     distance_to_image_plane =False,
#     instance_segmentation   =True,
#     normals                 =False,
#     semantic_segmentation   =True,
# )
        # # Attach render_product to the writer
# basic_writer.attach([render_product])
# rep.orchestrator.run_until_complete(num_frames=300)
# rep.orchestrator.stop()

i = 0
state = 0
target_idx = 0
ik_first_flag = True
stop_flag = True
gpu_dynamic_flag = 0
my_world.stop()
while simulation_app.is_running():
    my_world.step(render=True)
    if gpu_dynamic_flag<=110:
        gpu_dynamic_flag+=1
    if gpu_dynamic_flag >100:
        # import pdb;pdb.set_trace()
        stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:enableGPUDynamics').Set(True)
        stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:broadphaseType').Set("GPU")
        stage.GetPrimAtPath("/physicsScene").GetAttribute('physxScene:collisionSystem').Set("PCM")
        
        
    if my_world.is_stopped() and stop_flag:
        i=0
        state=0
        ik_first_flag=True
        stop_flag = False

    if my_world.is_playing():
        stop_flag=True
        if my_world.current_time_step_index <= 1:
            my_world.reset()
        i += 1

        if state==0:
            if ik_first_flag:
                #target_pos = gamja_rep.get_position()[target_idx]
                target_pos = np.array([0.5,0,0.5])
                ik,_ = my_robot_task.compute_ik(target_position = target_pos + np.array([0,0,0.4]),
                    target_orientation = [0,-180,0], # x,y,z 순서로 회전
                    frame_name = "J6",
                    )
                ik_first_flag =False
                print(target_pos)
            my_robot.apply_action(ArticulationAction(
                                    joint_indices=[0,1,2,3,4,5] ,
                                  joint_positions = ik) )


        if state==1:
            if ik_first_flag:
                # target_pos = gamja_rep.get_position()[target_idx]
                target_pos = np.array([0.5,0,0.5])
                ik,_ = my_robot_task.compute_ik(target_position = target_pos + np.array([0,0,0.19]),
                    target_orientation = [0,-180,0], # x,y,z 순서로 회전
                    frame_name = "J6",
                    )
                ik_first_flag = False
            my_robot.apply_action(ArticulationAction(
                                    joint_indices=[0,1,2,3,4,5] ,
                                  joint_positions = ik))

        if state==2:
            my_robot.apply_action(ArticulationAction(
                                    joint_indices=[6,7,8,9,10,11] ,
                                  joint_positions = [-0.5,0,0.35,0.35,-0.5,0]))
        # if state==3:
        #     my_robot.apply_action(ArticulationAction(
        #                             joint_indices=[0,1,2,3,4,5] ,
        #                           joint_positions = [0,0,1.5,0,0,0]) )
        if state==3:
            if ik_first_flag:
                # target_pos = get_usd_pose(gamja_prim)
                ik,_ = my_robot_task.compute_ik(target_position = target_pos + np.array([0,0,0.4]),
                    target_orientation = [0,-180,0], # x,y,z 순서로 회전
                    frame_name = "J6",
                    )
                ik_first_flag =False
            my_robot.apply_action(ArticulationAction(
                                    joint_indices=[0,1,2,3,4,5] ,
                                  joint_positions = ik) )
        if state==4:
            my_robot.apply_action(ArticulationAction(
                                    joint_indices=[6,7,8,9,10,11] ,
                                  joint_positions = [0,0,-1,-1,0,0]))
        # if state == 3:
        #     joint_pos = my_robot.get_joint_positions( joint_indices = [0,1,2,3,4,5])
        #     fk = my_robot_task.compute_fk(frame_name="J6", joint_positions=joint_pos)
        #     state=4
        #     i=0
        # if state == 4:
        #     ik,_ = my_robot_task.compute_ik(target_position = fk[0] + np.array([0,0,0.3]),
        #         target_orientation = [0,-180,0], # x,y,z 순서로 회전
        #         frame_name = "J6",
        #         )
        #     my_robot.apply_action(ArticulationAction(
        #                             joint_indices=[0,1,2,3,4,5] ,
        #                           joint_positions = ik))
        # if state==5:
        #     my_robot.apply_action(ArticulationAction(
        #                             joint_indices=[6,7,8, 9,10,11, 12,13,14] ,
        #                           joint_positions = [ 0, 0, 0, 
        #                                               0, 0, 0, 
        #                                               0, 0, 0]))
        if i >= 200:
            state+=1
            i=0
            ik_first_flag = True
        if state>=5:
            state=0
            target_idx +=1
        # if target_idx >= gamja_rep.count:
        #     target_idx =0

simulation_app.close()