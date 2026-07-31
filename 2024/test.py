from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})


from omni.isaac.core.objects import DynamicCuboid, DynamicSphere
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.world as world
import numpy as np

# 월드 초기화
sim_world = world.World()

# 큐브 생성
cube = DynamicCuboid(
    prim_path="/World/Cube",
    name="cube",
    position=np.array([0, 0, 0]),
    size=0.2,
    color=np.array([1, 0, 0])
)

# 구체 생성
sphere = DynamicSphere(
    prim_path="/World/Sphere",
    name="sphere",
    position=np.array([1, 1, 0]),
    radius=0.1,
    color=np.array([0, 0, 1])
)

sim_world.scene.add(cube)
sim_world.scene.add(sphere)


import omni.isaac.core.utils.physics as physics_utils
import numpy as np

def apply_gravitational_pull(step_size):  # Isaac Sim 콜백 함수
    cube_position = cube.get_world_pose()[0]
    sphere_position = sphere.get_world_pose()[0]

    direction = cube_position - sphere_position
    distance = np.linalg.norm(direction)
    
    if distance > 0:
        direction = direction / distance  # 정규화
        force_magnitude = 10 / (distance ** 2 + 0.1)  # 거리 제곱 반비례 법칙 적용
        force = force_magnitude * direction
        
        # 구체에 힘 적용
        sphere.apply_force_at_pos(force=force, position=sphere_position)

# 물리 업데이트 콜백 등록
sim_world.add_physics_callback("gravitational_pull", apply_gravitational_pull)

sim_world.reset()
sim_world.play()
