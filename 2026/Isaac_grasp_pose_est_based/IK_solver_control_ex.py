from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import ctypes
import importlib
import importlib.machinery
import importlib.util
import carb
import carb.input
import os
import sys
import sysconfig
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import omni.appwindow
import omni.usd
import yaml
from isaacsim.core.api import World
from isaacsim.core.api.objects import cuboid
from isaacsim.core.api.objects import VisualSphere
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.util.debug_draw import _debug_draw
from pxr import Gf, Vt
from scipy.spatial.transform import Rotation as R
from isaacsim.sensors.camera import Camera

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.physx.ui")
enable_extension("omni.physx")



THIS_DIR = Path(__file__).resolve().parent
ISAAC_CODE_ROOT = Path("/home/uon/ochansol/isaac_code")
ISAAC_CHANSOL_ROOT = ISAAC_CODE_ROOT / "isaac_chansol"
ROBOT_CONTROL_ROOT = ISAAC_CHANSOL_ROOT / "example" / "robot_control"

for path in (THIS_DIR, ISAAC_CHANSOL_ROOT, ROBOT_CONTROL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

IK_SOLVER_WRAPPER_DIR = Path("/home/uon/ochansol/ik_solver/python")
LOCAL_IK_SOLVER_DIR = THIS_DIR / "ik_solver_omy"
FALLBACK_IK_SOLVER_DIR = ROBOT_CONTROL_ROOT / "ik_solver_omy"
IK_SOLVER_DIR = LOCAL_IK_SOLVER_DIR if LOCAL_IK_SOLVER_DIR.exists() else FALLBACK_IK_SOLVER_DIR
IK_SOLVER_BINDING_DIR = IK_SOLVER_DIR / "py311_build"
IK_SOLVER_EXTENSION = IK_SOLVER_BINDING_DIR / f"ik_solver_py{sysconfig.get_config_var('EXT_SUFFIX')}"
ROS_URDFDOM_LIB_DIR = Path("/opt/ros/jazzy/lib/x86_64-linux-gnu")


def preload_ik_solver_module():
    supported_suffixes = importlib.machinery.EXTENSION_SUFFIXES
    if not IK_SOLVER_EXTENSION.exists():
        raise RuntimeError(
            "IK solver Python binding for this Isaac Python was not found.\n"
            f"  Python executable: {sys.executable}\n"
            f"  Python version: {sys.version}\n"
            f"  Expected binding file: {IK_SOLVER_EXTENSION}\n"
            "Build the binding for this Python version, or run the example with the matching Python."
        )
    if not any(str(IK_SOLVER_EXTENSION).endswith(suffix) for suffix in supported_suffixes):
        raise RuntimeError(
            "IK solver Python binding ABI mismatch.\n"
            f"  Python executable: {sys.executable}\n"
            f"  Python version: {sys.version}\n"
            f"  Binding file: {IK_SOLVER_EXTENSION}\n"
            f"  Supported extension suffixes: {supported_suffixes}\n"
            "Rebuild ik_solver_py for the Python used by Isaac Sim."
        )

    preload_libs = [
        ROS_URDFDOM_LIB_DIR / "liburdfdom_model.so.4.0",
    ]
    for lib_path in preload_libs:
        if not lib_path.exists():
            raise RuntimeError(f"Required IK solver dependency not found: {lib_path}")
        ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL | os.RTLD_NOW)

    try:
        importlib.import_module("ik_solver_py")
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import ik_solver_py after preloading dependencies.\n"
            f"  Python executable: {sys.executable}\n"
            f"  Python version: {sys.version}\n"
            f"  Binding file: {IK_SOLVER_EXTENSION}\n"
            f"  Binding dir: {IK_SOLVER_BINDING_DIR}\n"
            f"  ROS urdfdom lib dir: {ROS_URDFDOM_LIB_DIR}\n"
            f"  Original error: {exc}"
        ) from exc


if str(IK_SOLVER_WRAPPER_DIR) not in sys.path:
    sys.path.insert(0, str(IK_SOLVER_WRAPPER_DIR))
if str(IK_SOLVER_BINDING_DIR) not in sys.path:
    sys.path.insert(0, str(IK_SOLVER_BINDING_DIR))
preload_ik_solver_module()

from ik_solver import ik_solver
from Utils.general_utils import mat_utils
from Utils.Robot_45 import robot_policy
from Utils.isaac_utils_51 import scan_rep
import Utils.isaac_utils_51.rep_utils as csr


IK_JOINT_INDICES = np.arange(6, dtype=np.int32)
IDLE_JOINT = np.array([0, -32, 25, 43, 92, 0, 0, 0, 0, 0], dtype=float) / 180.0 * np.pi
TARGET_START_POSITION = np.array([0.5, 0.0, 0.5], dtype=float)
USE_TARGET_CUBE_ORIENTATION = True
OBSTACLE_BOX_HALF_EXTENTS = np.array([0.05, 0.05, 0.05], dtype=float)
COLLISION_CLEARANCE = 0.005
COLLISION_IMPROVEMENT_EPS = 0.001
COLLISION_LINE_SEARCH_ALPHAS = (1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.0)
SHOW_COLLISION_SPHERES = False
LINK_COLLISION_SPHERES = ()
SCATTER_OBJECTS_ON_RESET = True
GRASP_DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen")
GRASP_VIZ_SCRIPT_PATH = THIS_DIR.parent / "grasp_data_gen_for_isaaclab" / "merged_grasp_viz.py"
GRASP_MAX_DEBUG_DRAW = 50
GRASP_DEBUG_LINE_WIDTH = 2.0
GRASP_DEBUG_DRAW_COLOR = (1.0, 0.92, 0.0)
GRASP_SELECTED_COLOR = (0.0, 1.0, 0.1)
GRASP_UNSELECTED_COLOR = (0.0, 0.25, 1.0)
GRASP_USE_CAMERA_NORMAL_FILTER = True
SHOW_ONLY_SELECTED_GRASP_AFTER_V = True
GRASP_TARGET_POSITION_SOURCE = "box_center"  # "box_center", "target_points", or "mat"
GRASP_TARGET_ORIENTATION_SOURCE = "box"  # "box" or "mat"
GRASP_TARGET_APPROACH_AXIS = "-Y"  # TCP axis to align with grasp box 1->0 / 2->3.
GRASP_TARGET_YAW_AXIS = "X"  # TCP axis to align with grasp box 1->2.
GRASP_SELECTION_VECTOR_SOURCE = "gripper_to_object"  # "gripper_to_object" or "object_to_gripper"
GRASP_TARGET_TCP_CORRECTION_RPY = np.array([0.0, 0.0, 0.0], dtype=float)
SHOW_SELECTED_GRASP_TARGET_AXES = True
SELECTED_GRASP_TARGET_AXIS_LENGTH = 0.08
GRASP_CAMERA_VECTOR_LENGTH = 0.25
GRASP_CAMERA_VECTOR_COLOR = (0.0, 1.0, 1.0)
SHOW_GRASP_GRAVITY_VECTORS = True
GRASP_GRAVITY_VECTOR_LENGTH = 0.08
GRASP_GRAVITY_VECTOR_COLOR = (1.0, 0.0, 0.75)

# TODO: Replace these two vectors with your camera/gravity calculation.
GRASP_CAMERA_DIRECTION_WORLD = np.array([0.4, -0.7, 0.55], dtype=float)
GRASP_GRAVITY_DIRECTION_WORLD = np.array([0.0, 0.0, -1.0], dtype=float)

OBJECT_DICT = {
    "wireless_charging_stand": {
        "name": "wireless_charging_stand",
        "path": "/nas/ochansol/3d_model/peel3_scan_data_2025/wireless_charging_stand/edited/wireless_charging_stand.usd",
        "size_rank": 0,
        "scale": [0.1, 0.1, 0.1],
        "grasp_box_scale": [10.0, 10.0, 10.0],
        "position": [0.25, -0.015, 0.041],
    },
}
SELF_COLLISION_IGNORE_FRAME_PAIRS = {
    frozenset(("OMY_link1", "OMY_link2")),
    frozenset(("OMY_link2", "OMY_link3")),
    frozenset(("OMY_link3", "OMY_link4")),
    frozenset(("OMY_link4", "OMY_link5")),
    frozenset(("OMY_link5", "OMY_link6")),
    frozenset(("OMY_link6", "rh_p12_rn_l1")),
    frozenset(("OMY_link6", "rh_p12_rn_l2")),
    frozenset(("OMY_link6", "rh_p12_rn_r1")),
    frozenset(("OMY_link6", "rh_p12_rn_r2")),
    frozenset(("rh_p12_rn_l1", "rh_p12_rn_l2")),
    frozenset(("rh_p12_rn_l1", "rh_p12_rn_r1")),
    frozenset(("rh_p12_rn_l1", "rh_p12_rn_r2")),
    frozenset(("rh_p12_rn_l2", "rh_p12_rn_r1")),
    frozenset(("rh_p12_rn_l2", "rh_p12_rn_r2")),
    frozenset(("rh_p12_rn_r1", "rh_p12_rn_r2")),

}


def make_omy_ik_solver(urdf_path):
    solver = ik_solver(str(urdf_path))
    if not solver.init():
        raise RuntimeError(f"IK solver init failed: {urdf_path}")

    solver.set_strict_mode()
    solver.set_tcp_max_speed(2.0)
    solver.set_safety_scale(1.2)
    solver.set_workspace_limits(
        x_range=(-0.8, 0.8),
        y_range=(-0.8, 0.8),
        z_range=(0.0, 1.0),
    )
    configure_grasp_joint_as_tcp(solver, urdf_path)
    return solver


def load_fixed_joint_origin_from_urdf(urdf_path, joint_name):
    root = ET.parse(urdf_path).getroot()
    for joint in root.findall("joint"):
        if joint.attrib.get("name") != joint_name:
            continue
        origin = joint.find("origin")
        if origin is None:
            return np.zeros(3, dtype=float), np.zeros(3, dtype=float)
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ", dtype=float)
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ", dtype=float)
        if xyz.size != 3:
            xyz = np.zeros(3, dtype=float)
        if rpy.size != 3:
            rpy = np.zeros(3, dtype=float)
        return xyz, rpy
    return None


def configure_grasp_joint_as_tcp(solver, urdf_path):
    grasp_joint_origin = load_fixed_joint_origin_from_urdf(urdf_path, "OMY_grasp_joint_grasp_joint")
    if grasp_joint_origin is None:
        solver.remove_end_effector()
        return
    xyz, rpy = grasp_joint_origin
    solver.set_end_effector(float(xyz[0]), float(xyz[1]), float(xyz[2]), float(rpy[0]), float(rpy[1]), float(rpy[2]))


def load_collision_spheres_from_description(description_path):
    with open(description_path, "r", encoding="utf-8") as yaml_file:
        description = yaml.safe_load(yaml_file)

    specs = []
    for frame_entry in description.get("collision_spheres", []):
        for frame_name, spheres in frame_entry.items():
            if frame_name == "world":
                continue
            for sphere in spheres:
                specs.append(
                    {
                        "frame_name": frame_name,
                        "center": np.array(sphere["center"], dtype=float),
                        "radius": float(sphere["radius"]),
                    }
                )
    if not specs:
        raise RuntimeError(f"No robot collision spheres found in {description_path}")
    return tuple(specs)


def target_pose_to_ik_pose(position, orientation_wxyz, fallback_rpy):
    if USE_TARGET_CUBE_ORIENTATION:
        rpy = rotation_to_ik_solver_rpy(quat_wxyz_to_rotation(orientation_wxyz))
    else:
        rpy = np.asarray(fallback_rpy, dtype=float)
    return np.concatenate((np.asarray(position, dtype=float), rpy))


def quat_wxyz_to_rotation(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    return R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def rotation_to_quat_wxyz(rotation):
    quat_xyzw = rotation.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def ik_solver_rpy_to_rotation(rpy):
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    rot_x = R.from_rotvec(np.array([roll, 0.0, 0.0], dtype=float))
    rot_y = R.from_rotvec(np.array([0.0, pitch, 0.0], dtype=float))
    rot_z = R.from_rotvec(np.array([0.0, 0.0, yaw], dtype=float))
    return rot_x * rot_y * rot_z


def ik_solver_rpy_to_quat_wxyz(rpy):
    return rotation_to_quat_wxyz(ik_solver_rpy_to_rotation(rpy))


def rotation_to_ik_solver_rpy(rotation):
    # Match ik_solver/include/transform.hpp: Transform::make_tf uses R = Rx(roll) * Ry(pitch) * Rz(yaw).
    rot_matrix = rotation.as_matrix()
    pitch = np.arcsin(np.clip(rot_matrix[0, 2], -1.0, 1.0))
    if abs(rot_matrix[0, 2]) < 0.99999:
        roll = np.arctan2(-rot_matrix[1, 2], rot_matrix[2, 2])
        yaw = np.arctan2(-rot_matrix[0, 1], rot_matrix[0, 0])
    else:
        roll = np.arctan2(rot_matrix[2, 1], rot_matrix[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def get_robot_world_pose(robot_task):
    robot_articulation = getattr(robot_task, "_robot", None)
    if robot_articulation is not None:
        try:
            robot_position, robot_orientation = robot_articulation.get_world_pose()
            return np.asarray(robot_position, dtype=float), np.asarray(robot_orientation, dtype=float)
        except Exception:
            pass

    robot_position = np.asarray(getattr(robot_task, "robot_pos", np.zeros(3, dtype=float)), dtype=float)
    robot_orientation = np.asarray(
        getattr(robot_task, "robot_ori", np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
        dtype=float,
    )
    return robot_position, robot_orientation


def transform_world_pose_to_robot_local(position, orientation_wxyz, robot_task):
    robot_position, robot_orientation = get_robot_world_pose(robot_task)
    robot_rot = quat_wxyz_to_rotation(robot_orientation)
    world_rot = quat_wxyz_to_rotation(orientation_wxyz)
    local_position = robot_rot.inv().apply(np.asarray(position, dtype=float) - robot_position)
    local_orientation = rotation_to_quat_wxyz(robot_rot.inv() * world_rot)
    return local_position, local_orientation


def transform_robot_local_pose_to_world(position, orientation_wxyz, robot_task):
    robot_position, robot_orientation = get_robot_world_pose(robot_task)
    robot_rot = quat_wxyz_to_rotation(robot_orientation)
    local_rot = quat_wxyz_to_rotation(orientation_wxyz)
    world_position = robot_rot.apply(np.asarray(position, dtype=float)) + robot_position
    world_orientation = rotation_to_quat_wxyz(robot_rot * local_rot)
    return world_position, world_orientation


def get_full_joint_state_with_arm_q(robot_task, arm_q):
    full_q = np.asarray(robot_task.get_joint_positions(), dtype=float).copy()
    full_q[:6] = np.asarray(arm_q, dtype=float)
    return full_q


def compute_link_sphere_world_positions(robot_task, arm_q):
    full_q = get_full_joint_state_with_arm_q(robot_task, arm_q)
    fk_cache = {}
    sphere_entries = []

    for sphere in LINK_COLLISION_SPHERES:
        frame_name = sphere["frame_name"]
        if frame_name not in fk_cache:
            frame_pos, frame_euler_deg = robot_task.compute_fk(frame_name=frame_name, joint_positions=full_q)
            fk_cache[frame_name] = (
                np.asarray(frame_pos, dtype=float),
                R.from_euler("xyz", np.asarray(frame_euler_deg, dtype=float), degrees=True),
            )
        frame_pos, frame_rot = fk_cache[frame_name]
        sphere_entries.append(
            {
                "index": len(sphere_entries),
                "frame_name": frame_name,
                "center": frame_pos + frame_rot.apply(sphere["center"]),
                "radius": float(sphere["radius"]),
            }
        )
    return sphere_entries


def signed_sphere_box_margin(sphere_center, sphere_radius, box_position, box_orientation, box_half_extents):
    box_rot = quat_wxyz_to_rotation(box_orientation)
    local_center = box_rot.inv().apply(np.asarray(sphere_center, dtype=float) - np.asarray(box_position, dtype=float))
    outside = np.maximum(np.abs(local_center) - box_half_extents, 0.0)
    outside_distance = np.linalg.norm(outside)

    if outside_distance > 0.0:
        signed_point_distance = outside_distance
    else:
        signed_point_distance = -float(np.min(box_half_extents - np.abs(local_center)))
    return signed_point_distance - float(sphere_radius)


def evaluate_obstacle_collision_margin(robot_task, arm_q, obstacle_prim):
    obstacle_position, obstacle_orientation = obstacle_prim.get_world_pose()
    closest_margin = float("inf")
    closest_frame = None

    for sphere in compute_link_sphere_world_positions(robot_task, arm_q):
        margin = signed_sphere_box_margin(
            sphere_center=sphere["center"],
            sphere_radius=sphere["radius"] + COLLISION_CLEARANCE,
            box_position=obstacle_position,
            box_orientation=obstacle_orientation,
            box_half_extents=OBSTACLE_BOX_HALF_EXTENTS,
        )
        if margin < closest_margin:
            closest_margin = margin
            closest_frame = sphere["frame_name"]
    return closest_margin, closest_frame


def evaluate_self_collision_margin(robot_task, arm_q):
    spheres = compute_link_sphere_world_positions(robot_task, arm_q)
    closest_margin = float("inf")
    closest_pair = None
    colliding_indices = set()

    for i, sphere_a in enumerate(spheres):
        for j in range(i + 1, len(spheres)):
            sphere_b = spheres[j]
            if sphere_a["frame_name"] == sphere_b["frame_name"]:
                continue
            if frozenset((sphere_a["frame_name"], sphere_b["frame_name"])) in SELF_COLLISION_IGNORE_FRAME_PAIRS:
                continue

            center_distance = np.linalg.norm(sphere_a["center"] - sphere_b["center"])
            margin = center_distance - (sphere_a["radius"] + sphere_b["radius"] + COLLISION_CLEARANCE)
            if margin < closest_margin:
                closest_margin = margin
                closest_pair = (sphere_a["frame_name"], sphere_b["frame_name"])
            if margin < 0.0:
                colliding_indices.add(i)
                colliding_indices.add(j)

    return closest_margin, closest_pair, colliding_indices, spheres


def evaluate_collision_state(robot_task, arm_q, obstacle_prim):
    self_margin, self_pair, colliding_indices, spheres = evaluate_self_collision_margin(robot_task, arm_q)
    if obstacle_prim is None:
        return {
            "margin": self_margin,
            "kind": "self",
            "detail": self_pair,
            "colliding_indices": colliding_indices,
            "spheres": spheres,
            "self_margin": self_margin,
            "self_detail": self_pair,
            "obstacle_margin": float("inf"),
            "obstacle_detail": None,
        }

    obstacle_margin, obstacle_frame = evaluate_obstacle_collision_margin(robot_task, arm_q, obstacle_prim)

    if self_margin <= obstacle_margin:
        return {
            "margin": self_margin,
            "kind": "self",
            "detail": self_pair,
            "colliding_indices": colliding_indices,
            "spheres": spheres,
            "self_margin": self_margin,
            "self_detail": self_pair,
            "obstacle_margin": obstacle_margin,
            "obstacle_detail": obstacle_frame,
        }
    return {
        "margin": obstacle_margin,
        "kind": "obstacle",
        "detail": obstacle_frame,
        "colliding_indices": set(),
        "spheres": spheres,
        "self_margin": self_margin,
        "self_detail": self_pair,
        "obstacle_margin": obstacle_margin,
        "obstacle_detail": obstacle_frame,
    }


def filter_ik_q_with_collision_escape(robot_task, current_q, desired_q, obstacle_prim):
    current_q = np.asarray(current_q, dtype=float)
    desired_q = np.asarray(desired_q, dtype=float)
    current_state = evaluate_collision_state(robot_task, current_q, obstacle_prim)
    desired_state = evaluate_collision_state(robot_task, desired_q, obstacle_prim)

    if current_state["margin"] >= 0.0 and desired_state["margin"] >= 0.0:
        return desired_q, desired_state
    if current_state["margin"] < 0.0 and desired_state["margin"] > current_state["margin"] + COLLISION_IMPROVEMENT_EPS:
        return desired_q, desired_state

    best_q = current_q
    best_state = current_state

    for alpha in COLLISION_LINE_SEARCH_ALPHAS:
        candidate_q = current_q + alpha * (desired_q - current_q)
        candidate_state = evaluate_collision_state(robot_task, candidate_q, obstacle_prim)

        if current_state["margin"] >= 0.0 and candidate_state["margin"] >= 0.0:
            return candidate_q, candidate_state
        if current_state["margin"] < 0.0 and candidate_state["margin"] > current_state["margin"] + COLLISION_IMPROVEMENT_EPS:
            return candidate_q, candidate_state
        if candidate_state["margin"] > best_state["margin"]:
            best_q = candidate_q
            best_state = candidate_state

    return best_q, best_state


def create_collision_sphere_visuals():
    visuals = []
    for index, sphere in enumerate(LINK_COLLISION_SPHERES):
        visual = VisualSphere(
                prim_path=f"/World/ik_collision_sphere_{index:02d}",
                position=np.zeros(3, dtype=float),
                radius=float(sphere["radius"]),
                color=np.array([0.1, 0.7, 1.0], dtype=float),
                visible=SHOW_COLLISION_SPHERES,
            )
        set_visual_sphere_color(visual, np.array([0.1, 0.7, 1.0], dtype=float))
        visuals.append(visual)
    return visuals


def set_visual_sphere_color(visual_sphere, color):
    prim = getattr(visual_sphere, "prim", None)
    if prim is None:
        return
    display_color_attr = prim.GetAttribute("primvars:displayColor")
    if not display_color_attr or not display_color_attr.IsValid():
        display_color_attr = prim.CreateAttribute("primvars:displayColor", "color3f[]", False)
    display_color_attr.Set(Vt.Vec3fArray([Gf.Vec3f(*np.asarray(color, dtype=float))]))


def update_collision_sphere_visuals(sphere_visuals, collision_state):
    if not sphere_visuals:
        return
    colliding_indices = collision_state["colliding_indices"]
    for index, sphere in enumerate(collision_state["spheres"]):
        sphere_visuals[index].set_world_pose(position=sphere["center"])
        color = np.array([1.0, 0.2, 0.2], dtype=float) if index in colliding_indices else np.array([0.1, 0.7, 1.0], dtype=float)
        set_visual_sphere_color(sphere_visuals[index], color)


def load_merged_grasp_viz_module():
    if not GRASP_VIZ_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"merged_grasp_viz.py not found: {GRASP_VIZ_SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location("merged_grasp_viz_runtime", GRASP_VIZ_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized_or_default(vector, default):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-9 or not np.all(np.isfinite(vector)):
        if default is None:
            return None
        return np.asarray(default, dtype=float)
    return vector / norm


def grasp_npz_path_for_object(object_name):
    object_data_root = GRASP_DATASET_ROOT / object_name
    return object_data_root / f"{object_name}_merged_grasp_zero_pose_index.npz"


def load_grasp_database_for_object(object_name, merged_grasp_viz):
    npz_path = grasp_npz_path_for_object(object_name)
    if not npz_path.exists():
        raise FileNotFoundError(f"merged grasp npz not found: {npz_path}")

    with np.load(npz_path) as npz_data:
        grasp_items = merged_grasp_viz.npz_to_grasp_items(npz_data)

    centers = []
    for item in grasp_items:
        center = merged_grasp_viz.grasp_center(item)
        if center is not None:
            centers.append(center)
    object_center = np.mean(np.asarray(centers, dtype=float), axis=0) if centers else np.zeros(3, dtype=float)
    return {
        "object_name": object_name,
        "npz_path": npz_path,
        "grasp_items": grasp_items,
        "object_center": object_center,
    }


def select_debug_grasp_items(grasp_database, merged_grasp_viz, local_camera_direction):
    grasp_items = grasp_database["grasp_items"]
    if GRASP_USE_CAMERA_NORMAL_FILTER:
        selected = merged_grasp_viz.select_camera_normal_candidates(
            grasp_items,
            local_camera_direction,
            grasp_database["object_center"],
            top_k=GRASP_MAX_DEBUG_DRAW,
        )
        selected_items = [item for _, _, _, _, _, _, item in selected]
        if selected_items:
            return selected_items

    return sorted(grasp_items, key=merged_grasp_viz.final_quality_score, reverse=True)[:GRASP_MAX_DEBUG_DRAW]


def grasp_selection_score(item, merged_grasp_viz=None):
    score = safe_float(item.get("score"), default=None)
    if score is not None:
        return score
    if merged_grasp_viz is not None:
        return merged_grasp_viz.final_quality_score(item)
    return 0.0


def safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def grasp_box_approach_direction(item):
    grasp_box = np.asarray(item.get("grasp_box", []), dtype=float)
    if grasp_box.shape != (4, 3):
        return None
    return normalized_or_default(grasp_box[0] - grasp_box[1], None)


def grasp_center_from_item(item):
    grasp_box = np.asarray(item.get("grasp_box", []), dtype=float)
    if grasp_box.shape != (4, 3):
        return None
    return grasp_box.mean(axis=0)


def grasp_object_alignment_score(item, object_center):
    approach_direction = grasp_box_approach_direction(item)
    if approach_direction is None:
        return -float("inf")

    grasp_center = grasp_center_from_item(item)
    if grasp_center is None:
        return -float("inf")

    object_center = np.asarray(object_center, dtype=float)
    if GRASP_SELECTION_VECTOR_SOURCE == "object_to_gripper":
        object_gripper_vector = grasp_center - object_center
    else:
        object_gripper_vector = object_center - grasp_center
    object_gripper_vector = normalized_or_default(object_gripper_vector, None)
    if object_gripper_vector is None:
        return -float("inf")
    return float(np.dot(approach_direction, object_gripper_vector))


def object_world_matrix(object_rep):
    return np.asarray(csr.find_parents_tf(object_rep.prim, include_self=True), dtype=float).T


def transform_world_direction_to_object_local(direction, object_tf):
    direction = normalized_or_default(direction, [0.0, 0.0, -1.0])
    local_direction = np.linalg.inv(object_tf[:3, :3]) @ direction
    return normalized_or_default(local_direction, [0.0, 0.0, -1.0])


def grasp_box_scale_for_object(object_rep):
    scale = getattr(object_rep, "grasp_box_scale", 1.0)
    scale = np.asarray(scale, dtype=float)
    if scale.ndim == 0:
        scale = np.full(3, float(scale), dtype=float)
    if scale.size != 3 or not np.all(np.isfinite(scale)):
        return np.ones(3, dtype=float)
    return scale.reshape(3)


def transform_object_local_points_to_world(points, object_tf, grasp_box_scale):
    points = np.asarray(points, dtype=float) * grasp_box_scale
    points_h = np.concatenate((points, np.ones((len(points), 1), dtype=float)), axis=1)
    return (object_tf @ points_h.T).T[:, :3]


def orthonormalize_rotation_matrix(matrix):
    u, _, vh = np.linalg.svd(np.asarray(matrix, dtype=float))
    rotation_matrix = u @ vh
    if np.linalg.det(rotation_matrix) < 0.0:
        u[:, -1] *= -1.0
        rotation_matrix = u @ vh
    return rotation_matrix


def axis_name_to_index_sign(axis_name):
    axis_name = str(axis_name).strip().upper()
    sign = -1.0 if axis_name.startswith("-") else 1.0
    axis_char = axis_name[-1]
    if axis_char == "X":
        return 0, sign
    if axis_char == "Y":
        return 1, sign
    if axis_char == "Z":
        return 2, sign
    raise ValueError(f"Unsupported axis name: {axis_name}")


def rotation_matrix_from_axis_constraints(primary_axis_name, primary_direction, secondary_axis_name, secondary_direction):
    primary_index, primary_sign = axis_name_to_index_sign(primary_axis_name)
    secondary_index, secondary_sign = axis_name_to_index_sign(secondary_axis_name)
    if primary_index == secondary_index:
        raise ValueError("Primary and secondary axes must be different.")

    primary_direction = normalized_or_default(primary_direction, None)
    secondary_direction = normalized_or_default(secondary_direction, None)
    if primary_direction is None or secondary_direction is None:
        return None

    axes = [None, None, None]
    axes[primary_index] = primary_direction * primary_sign

    secondary_base = secondary_direction * secondary_sign
    secondary_base = secondary_base - np.dot(secondary_base, axes[primary_index]) * axes[primary_index]
    secondary_base = normalized_or_default(secondary_base, None)
    if secondary_base is None:
        return None
    axes[secondary_index] = secondary_base

    remaining_index = ({0, 1, 2} - {primary_index, secondary_index}).pop()
    if remaining_index == 0:
        axes[0] = normalized_or_default(np.cross(axes[1], axes[2]), None)
    elif remaining_index == 1:
        axes[1] = normalized_or_default(np.cross(axes[2], axes[0]), None)
    else:
        axes[2] = normalized_or_default(np.cross(axes[0], axes[1]), None)
    if axes[remaining_index] is None:
        return None

    # Recompute the secondary axis to remove residual numerical skew and keep a right-handed frame.
    if secondary_index == 0:
        axes[0] = normalized_or_default(np.cross(axes[1], axes[2]), None)
    elif secondary_index == 1:
        axes[1] = normalized_or_default(np.cross(axes[2], axes[0]), None)
    else:
        axes[2] = normalized_or_default(np.cross(axes[0], axes[1]), None)
    if axes[secondary_index] is None:
        return None

    return orthonormalize_rotation_matrix(np.column_stack(axes))


def grasp_box_rotation_matrix(grasp_box):
    grasp_box = np.asarray(grasp_box, dtype=float)
    if grasp_box.shape != (4, 3):
        return None

    approach_dir = normalized_or_default(grasp_box[0] - grasp_box[1], None)
    yaw_seed = normalized_or_default(grasp_box[2] - grasp_box[1], None)
    if approach_dir is None or yaw_seed is None:
        return None

    # Box semantics:
    #   1 -> 0 and 2 -> 3: gripper approach direction, not the box plane normal.
    #   1 -> 2: gripper yaw / horizontal jaw direction.
    return rotation_matrix_from_axis_constraints(
        GRASP_TARGET_APPROACH_AXIS,
        approach_dir,
        GRASP_TARGET_YAW_AXIS,
        yaw_seed,
    )


def target_local_rotation_matrix(item, grasp_mat):
    source = GRASP_TARGET_ORIENTATION_SOURCE.lower()
    if source == "box":
        box_rotation = grasp_box_rotation_matrix(item.get("grasp_box", []))
        if box_rotation is not None:
            return box_rotation
        print("[grasp debug] box frame invalid; falling back to grasp_mat rotation.")

    return orthonormalize_rotation_matrix(grasp_mat[:3, :3])


def target_local_position(item, grasp_mat, grasp_box_scale):
    source = GRASP_TARGET_POSITION_SOURCE.lower()
    if source == "box_center":
        grasp_box = np.asarray(item.get("grasp_box", []), dtype=float)
        if grasp_box.shape == (4, 3):
            return grasp_box.mean(axis=0) * grasp_box_scale
        print("[grasp debug] box center invalid; falling back to grasp_mat translation.")

    if source == "target_points":
        target_points = np.asarray(item.get("target_points", []), dtype=float)
        if target_points.shape == (3,):
            return target_points * grasp_box_scale
        print("[grasp debug] target_points invalid; falling back to grasp_mat translation.")

    return grasp_mat[:3, 3] * grasp_box_scale


def transform_object_local_grasp_tf_to_world(item, object_tf, grasp_box_scale):
    grasp_mat = np.asarray(item.get("grasp_mat"), dtype=float)
    grasp_mat = np.asarray(grasp_mat, dtype=float)
    if grasp_mat.shape != (4, 4):
        return None, None, None

    scaled_local_grasp_tf = grasp_mat.copy()
    scaled_local_grasp_tf[:3, 3] = target_local_position(item, grasp_mat, grasp_box_scale)
    scaled_local_grasp_tf[:3, :3] = target_local_rotation_matrix(item, grasp_mat)
    world_tf = object_tf @ scaled_local_grasp_tf

    world_position = world_tf[:3, 3].copy()
    world_rotation = orthonormalize_rotation_matrix(world_tf[:3, :3])
    tcp_correction = ik_solver_rpy_to_rotation(GRASP_TARGET_TCP_CORRECTION_RPY).as_matrix()
    world_rotation = orthonormalize_rotation_matrix(world_rotation @ tcp_correction)
    world_orientation = rotation_to_quat_wxyz(R.from_matrix(world_rotation))
    world_tf[:3, :3] = world_rotation
    return world_tf, world_position, world_orientation


def transform_object_local_direction_to_world(direction, object_tf):
    direction = normalized_or_default(direction, [0.0, 0.0, -1.0])
    world_direction = object_tf[:3, :3] @ direction
    return normalized_or_default(world_direction, [0.0, 0.0, -1.0])


def append_world_grasp_box_lines(state, starts, ends, colors, widths):
    world_boxes = state.get("grasp_world_boxes", [])
    selected_index = state.get("selected_grasp_index")
    show_selection = selected_index is not None
    edge_indices = ((0, 1), (1, 2), (2, 3))
    for rank, box in enumerate(world_boxes):
        box = np.asarray(box, dtype=float)
        if box.shape != (4, 3):
            continue
        if show_selection:
            color = GRASP_SELECTED_COLOR if rank == selected_index else GRASP_UNSELECTED_COLOR
        elif len(world_boxes) > 1:
            t = rank / float(len(world_boxes) - 1)
            color = tuple((1.0 - t) * np.asarray(GRASP_DEBUG_DRAW_COLOR) + t * np.asarray([0.0, 0.25, 1.0]))
        else:
            color = GRASP_DEBUG_DRAW_COLOR
        for start_idx, end_idx in edge_indices:
            starts.append(carb.Float3(box[start_idx]))
            ends.append(carb.Float3(box[end_idx]))
            colors.append(carb.ColorRgba(color[0], color[1], color[2], 1.0))
            widths.append(GRASP_DEBUG_LINE_WIDTH)


def append_grasp_gravity_lines(gravity_lines, starts, ends, colors, widths):
    for start, end in gravity_lines:
        starts.append(carb.Float3(start))
        ends.append(carb.Float3(end))
        colors.append(carb.ColorRgba(*GRASP_GRAVITY_VECTOR_COLOR, 1.0))
        widths.append(GRASP_DEBUG_LINE_WIDTH + 1.0)


def append_selected_target_axes(state, starts, ends, colors, widths):
    if not SHOW_SELECTED_GRASP_TARGET_AXES:
        return
    pose = state.get("best_grasp_world_pose")
    if pose is None:
        return
    tf = np.asarray(pose.get("tf"), dtype=float)
    if tf.shape != (4, 4):
        return

    origin = tf[:3, 3]
    axis_colors = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.35, 1.0),
    )
    for axis_idx, color in enumerate(axis_colors):
        starts.append(carb.Float3(origin))
        ends.append(carb.Float3(origin + tf[:3, axis_idx] * SELECTED_GRASP_TARGET_AXIS_LENGTH))
        colors.append(carb.ColorRgba(*color, 1.0))
        widths.append(GRASP_DEBUG_LINE_WIDTH + 2.0)


def draw_debug_overlay(state):
    draw = state["draw"]
    draw.clear_lines()
    starts = []
    ends = []
    colors = []
    widths = []

    append_world_grasp_box_lines(state, starts, ends, colors, widths)
    if SHOW_GRASP_GRAVITY_VECTORS:
        append_grasp_gravity_lines(
            state.get("grasp_gravity_lines", []),
            starts,
            ends,
            colors,
            widths,
        )
    append_selected_target_axes(state, starts, ends, colors, widths)

    camera_line = state.get("camera_line")
    if camera_line is not None:
        start, end = camera_line
        starts.append(carb.Float3(start))
        ends.append(carb.Float3(end))
        colors.append(carb.ColorRgba(*GRASP_CAMERA_VECTOR_COLOR, 1.0))
        widths.append(GRASP_DEBUG_LINE_WIDTH + 1.0)

    if starts:
        draw.draw_lines(starts, ends, colors, widths)


def update_camera_direction_debug(wrist_camera, object_rep, state):
    global GRASP_CAMERA_DIRECTION_WORLD

    if object_rep is None:
        draw_debug_overlay(state)
        return

    camera_position, _ = wrist_camera.get_world_pose()
    camera_position = np.asarray(camera_position, dtype=float)
    object_tf = object_world_matrix(object_rep)
    object_position = object_tf[:3, 3]
    object_to_camera = camera_position - object_position

    GRASP_CAMERA_DIRECTION_WORLD = normalized_or_default(object_to_camera, [0.0, 0.0, -1.0])
    state["camera_direction_world"] = GRASP_CAMERA_DIRECTION_WORLD
    state["camera_position_world"] = camera_position
    state["camera_target_position_world"] = object_position
    state["camera_line"] = (
        object_position,
        object_position + GRASP_CAMERA_DIRECTION_WORLD * GRASP_CAMERA_VECTOR_LENGTH,
    )
    draw_debug_overlay(state)


def draw_grasp_debug_once(object_rep, state):
    if object_rep is None:
        return

    object_name = getattr(object_rep, "class_name", None)
    if not object_name:
        return

    merged_grasp_viz = state.get("merged_grasp_viz")
    if merged_grasp_viz is None:
        merged_grasp_viz = load_merged_grasp_viz_module()
        state["merged_grasp_viz"] = merged_grasp_viz

    if state.get("object_name") != object_name:
        try:
            state["database"] = load_grasp_database_for_object(object_name, merged_grasp_viz)
            state["object_name"] = object_name
            print(
                f"[grasp debug] loaded {len(state['database']['grasp_items'])} grasps "
                f"for {object_name}: {state['database']['npz_path']}"
            )
        except Exception as exc:
            state["database"] = None
            state["object_name"] = object_name
            print(f"[grasp debug] disabled for {object_name}: {exc}")

    grasp_database = state.get("database")
    if grasp_database is None:
        state["grasp_world_boxes"] = []
        state["grasp_gravity_lines"] = []
        state["selected_grasp_items"] = []
        state["selected_grasp_index"] = None
        state["use_best_grasp_ik_target"] = False
        draw_debug_overlay(state)
        return

    object_tf = object_world_matrix(object_rep)
    local_camera_direction = transform_world_direction_to_object_local(
        GRASP_CAMERA_DIRECTION_WORLD,
        object_tf,
    )
    local_gravity_direction = transform_world_direction_to_object_local(
        GRASP_GRAVITY_DIRECTION_WORLD,
        object_tf,
    )
    state["local_camera_direction"] = local_camera_direction
    state["local_gravity_direction"] = local_gravity_direction

    selected_items = select_debug_grasp_items(grasp_database, merged_grasp_viz, local_camera_direction)
    if not selected_items:
        state["grasp_world_boxes"] = []
        state["grasp_gravity_lines"] = []
        state["selected_grasp_items"] = []
        state["selected_grasp_index"] = None
        state["use_best_grasp_ik_target"] = False
        draw_debug_overlay(state)
        return

    grasp_box_scale = grasp_box_scale_for_object(object_rep)
    world_boxes = []
    gravity_lines = []
    drawn_items = []
    for item in selected_items:
        grasp_box = np.asarray(item.get("grasp_box", []), dtype=float)
        if grasp_box.shape != (4, 3):
            continue
        world_box = transform_object_local_points_to_world(grasp_box, object_tf, grasp_box_scale)
        world_boxes.append(world_box)
        drawn_items.append(item)

        gravity_direction = item.get("normal")
        if gravity_direction is None:
            continue
        world_gravity_direction = transform_object_local_direction_to_world(gravity_direction, object_tf)
        center = world_box.mean(axis=0)
        gravity_lines.append(
            (
                center,
                center + world_gravity_direction * GRASP_GRAVITY_VECTOR_LENGTH,
            )
        )
    state["grasp_world_boxes"] = world_boxes
    state["grasp_gravity_lines"] = gravity_lines
    state["selected_grasp_items"] = drawn_items
    state["selected_grasp_index"] = None
    state["best_grasp_item"] = None
    state["best_grasp_world_pose"] = None
    state["use_best_grasp_ik_target"] = False
    draw_debug_overlay(state)
    print(
        f"[grasp debug] drew {len(world_boxes)} boxes for {object_name}, "
        f"scale={grasp_box_scale.tolist()}, "
        f"camera_local={local_camera_direction.tolist()}, "
        f"gravity_local={local_gravity_direction.tolist()}"
    )


def select_best_grasp_and_set_ik_target(object_rep, state):
    if object_rep is None:
        return False

    selected_items = state.get("selected_grasp_items", [])
    if not selected_items:
        print("[grasp debug] no C-key grasp candidates. Press C before V.")
        return False

    merged_grasp_viz = state.get("merged_grasp_viz")
    grasp_database = state.get("database")
    object_center = grasp_database["object_center"] if grasp_database is not None else np.zeros(3, dtype=float)
    best_index, best_item = max(
        enumerate(selected_items),
        key=lambda row: (
            grasp_object_alignment_score(row[1], object_center),
            grasp_selection_score(row[1], merged_grasp_viz),
        ),
    )
    grasp_mat = best_item.get("grasp_mat")
    if grasp_mat is None:
        print("[grasp debug] best grasp has no grasp_mat; cannot set IK target.")
        return False

    object_tf = object_world_matrix(object_rep)
    grasp_box_scale = grasp_box_scale_for_object(object_rep)
    best_world_tf, best_position, best_orientation = transform_object_local_grasp_tf_to_world(
        best_item,
        object_tf,
        grasp_box_scale,
    )
    if best_world_tf is None:
        print("[grasp debug] invalid grasp_mat shape; cannot set IK target.")
        return False

    if SHOW_ONLY_SELECTED_GRASP_AFTER_V:
        state["grasp_world_boxes"] = [state["grasp_world_boxes"][best_index]]
        if best_index < len(state.get("grasp_gravity_lines", [])):
            state["grasp_gravity_lines"] = [state["grasp_gravity_lines"][best_index]]
        else:
            state["grasp_gravity_lines"] = []
        state["selected_grasp_items"] = [best_item]
        best_index = 0

    state["selected_grasp_index"] = best_index
    state["best_grasp_item"] = best_item
    state["best_grasp_world_pose"] = {
        "tf": best_world_tf,
        "position": best_position,
        "orientation": best_orientation,
    }
    state["ik_target_position"] = best_position
    state["ik_target_orientation"] = best_orientation
    state["use_best_grasp_ik_target"] = True
    draw_debug_overlay(state)
    best_rpy_deg = np.degrees(rotation_to_ik_solver_rpy(R.from_matrix(best_world_tf[:3, :3])))
    print(
        f"[grasp debug] selected best grasp idx={best_index}, "
        f"object_alignment={grasp_object_alignment_score(best_item, object_center):.4f}, "
        f"score={grasp_selection_score(best_item, merged_grasp_viz):.4f}, "
        f"selection_vector={GRASP_SELECTION_VECTOR_SOURCE}, "
        f"position_source={GRASP_TARGET_POSITION_SOURCE}, "
        f"orientation_source={GRASP_TARGET_ORIENTATION_SOURCE}, "
        f"position={best_position.tolist()}, orientation_wxyz={best_orientation.tolist()}, "
        f"rpy_deg={best_rpy_deg.tolist()}"
    )
    return True


def make_keyboard_state():
    state = {
        "draw_grasp_once": False,
        "select_best_grasp_once": False,
        "input": carb.input.acquire_input_interface(),
        "keyboard": omni.appwindow.get_default_app_window().get_keyboard(),
        "subscription": None,
    }

    def on_keyboard_event(event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input == carb.input.KeyboardInput.C:
            state["draw_grasp_once"] = True
            return True
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input == carb.input.KeyboardInput.V:
            state["select_best_grasp_once"] = True
            return True
        return False

    state["subscription"] = state["input"].subscribe_to_keyboard_events(
        state["keyboard"],
        on_keyboard_event,
    )
    return state


def create_scan_reps(object_dict):
    obj_rep_all_list = []
    for key, model_attr in object_dict.items():
        print(f"load object: {model_attr['name']}")
        scan_obj = scan_rep.Scan_Rep(
            usd_path=model_attr["path"],
            class_name=model_attr["name"],
            size=model_attr["size_rank"],
            scale=model_attr.get("scale", [0.1, 0.1, 0.1]),
            position=model_attr.get("position", [0.0, 0.0, 0.0]),
        )
        scan_obj.grasp_box_scale = model_attr.get("grasp_box_scale", 1.0)
        object_dict[key]["rep"] = scan_obj
        obj_rep_all_list.append(scan_obj)

    for obj_rep in obj_rep_all_list:
        print(f"set collider for: {obj_rep.class_name}")
        obj_rep.set_rigidbody_collider()
        obj_rep.set_physics_material(
            dynamic_friction=0.25,
            static_friction=0.4,
            restitution=0.1,
        )

    return obj_rep_all_list


def make_platform_rep(stage, env_prim):
    platform_area_prims = csr.find_target_name(env_prim, ["Mesh"], "platform_area")
    platform_parents = [prim.GetParent() for prim in platform_area_prims if prim.GetParent().GetName() == "demo"]
    if not platform_parents:
        raise RuntimeError("Could not find platform_area under env demo prim.")

    platform_path = str(platform_parents[0].GetPath())
    platform_rep = scan_rep.Scan_Rep_Platform(
        prim_path=platform_path,
        scale=[1, 1, 1],
        class_name=platform_path.split("/")[-1],
    )
    platform_tf = csr.find_parents_tf(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
    platform_scale = csr.find_parents_scale(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
    platform_rep.set_tf(platform_tf)
    platform_rep.set_scale(platform_scale)
    return platform_rep


def scatter_objects(platform_rep, obj_rep_all_list):
    csr.scatter_in_platform_area(
        platform_rep,
        obj_rep_all_list,
        fixed_first=True,
        rotation=False,
    )


def sync_solver_to_robot(solver, robot_task):
    current_q = np.asarray(robot_task.get_joint_positions()[:6], dtype=float)
    solver.movej(current_q)
    return rotation_to_ik_solver_rpy(R.from_matrix(solver.get_tcp_tf()[:3, :3]))


def apply_ik_to_robot(
    solver,
    robot_task,
    target_position,
    target_orientation,
    fallback_rpy,
    obstacle_prim=None,
    sphere_visuals=None,
):
    local_target_position, local_target_orientation = transform_world_pose_to_robot_local(
        target_position,
        target_orientation,
        robot_task,
    )
    current_q = np.asarray(robot_task.get_joint_positions()[:6], dtype=float)
    solver.movel(target_pose_to_ik_pose(local_target_position, local_target_orientation, fallback_rpy))
    desired_q = np.asarray(solver.get_current_joints_rad(), dtype=float)

    safe_q, collision_state = filter_ik_q_with_collision_escape(
        robot_task,
        current_q=current_q,
        desired_q=desired_q,
        obstacle_prim=obstacle_prim,
    )
    if np.linalg.norm(safe_q - desired_q) > 1e-5:
        apply_ik_to_robot.filter_print_count = getattr(apply_ik_to_robot, "filter_print_count", 0) + 1
        if apply_ik_to_robot.filter_print_count % 30 == 1:
            print(
                f"[ik collision filter] adjusted command: kind={collision_state['kind']}, "
                f"margin={collision_state['margin']:.4f}, detail={collision_state['detail']}, "
                f"self_margin={collision_state['self_margin']:.4f}, obstacle_margin={collision_state['obstacle_margin']:.4f}"
            )
    solver.movej(safe_q)

    robot_task.apply_action(joint_indices=IK_JOINT_INDICES, joint_positions=safe_q)
    if sphere_visuals is not None:
        update_collision_sphere_visuals(sphere_visuals, collision_state)
    return safe_q


my_world = World(
    stage_units_in_meters=1.0,
    physics_dt=0.01,
    rendering_dt=0.01,
)

my_robot_task = robot_policy.My_Robot_Task(
    name="Robotis_OMY_self_collision",
    idle_joint=IDLE_JOINT,
)
LINK_COLLISION_SPHERES = load_collision_spheres_from_description(my_robot_task.cfg.description_path)

my_world.add_task(my_robot_task)
my_world.reset()

ik = make_omy_ik_solver(my_robot_task.cfg.urdf_path)
ik_target_rpy = sync_solver_to_robot(ik, my_robot_task)
target_start_local_orientation = ik_solver_rpy_to_quat_wxyz(ik_target_rpy)
_, target_start_orientation = transform_robot_local_pose_to_world(
    np.zeros(3, dtype=float),
    target_start_local_orientation,
    my_robot_task,
)

stage = omni.usd.get_context().get_stage()
robot_name = my_robot_task.get_robot_name
my_robot = my_robot_task._robot
my_robot_prim = my_robot_task.robot_prim
my_robot_task.set_semantic_labels()

env_prim = add_reference_to_stage(
    prim_path="/World/env",
    usd_path="/nas/ochansol/isaac/sim2real/uon_vla_demo_robotis_env.usd",
)
obj_rep_all_list = create_scan_reps(OBJECT_DICT)

stage = omni.usd.get_context().get_stage()
platform_rep = make_platform_rep(stage, env_prim)
my_world.reset()
scatter_objects(platform_rep, obj_rep_all_list)

wrist_cam_path = f"{my_robot_task.prim_path}/OMY/link6/wrist_camera"


wrist_res=(848,480)

wrist_camera = Camera(
    prim_path=wrist_cam_path,
    name="cam_wrist",
    frequency=30,
    resolution=wrist_res,)







target_cube = cuboid.VisualCuboid(
    "/World/target",
    position=TARGET_START_POSITION,
    orientation=target_start_orientation,
    color=np.array([1.0, 0.0, 0.0]),
    size=0.1,
)

obstacle = cuboid.VisualCuboid(
    "/World/obstacle",
    position=np.array([0.8, 0.0, 0.5]),
    color=np.array([0.0, 1.0, 0.0]),
    size=0.1,
)
sphere_visuals = create_collision_sphere_visuals()
grasp_debug_state = {
    "draw": _debug_draw.acquire_debug_draw_interface(),
    "grasp_world_boxes": [],
    "grasp_gravity_lines": [],
    "selected_grasp_items": [],
    "selected_grasp_index": None,
    "use_best_grasp_ik_target": False,
}
grasp_target_object = obj_rep_all_list[0] if obj_rep_all_list else None
keyboard_state = make_keyboard_state()

my_world.reset()
ik_target_rpy = sync_solver_to_robot(ik, my_robot_task)
initial_collision_state = evaluate_collision_state(
    my_robot_task,
    np.asarray(my_robot_task.get_joint_positions()[:6], dtype=float),
    obstacle,
)
update_collision_sphere_visuals(
    sphere_visuals,
    initial_collision_state,
)

reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True

    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            if SCATTER_OBJECTS_ON_RESET:
                scatter_objects(platform_rep, obj_rep_all_list)
            ik_target_rpy = sync_solver_to_robot(ik, my_robot_task)
            reset_collision_state = evaluate_collision_state(
                my_robot_task,
                np.asarray(my_robot_task.get_joint_positions()[:6], dtype=float),
                obstacle,
            )
            update_collision_sphere_visuals(
                sphere_visuals,
                reset_collision_state,
            )
            grasp_debug_state["grasp_world_boxes"] = []
            grasp_debug_state["grasp_gravity_lines"] = []
            grasp_debug_state["selected_grasp_items"] = []
            grasp_debug_state["selected_grasp_index"] = None
            grasp_debug_state["best_grasp_item"] = None
            grasp_debug_state["best_grasp_world_pose"] = None
            grasp_debug_state["use_best_grasp_ik_target"] = False
            target_cube.set_world_pose(
                position=TARGET_START_POSITION,
                orientation=target_start_orientation,
            )
            grasp_debug_state["draw"].clear_lines()
            reset_needed = False
            grasp_debug_state["ik_target_position"] = TARGET_START_POSITION
            grasp_debug_state["ik_target_orientation"] = target_start_orientation
        update_camera_direction_debug(wrist_camera, grasp_target_object, grasp_debug_state)

        if keyboard_state["draw_grasp_once"]:
            keyboard_state["draw_grasp_once"] = False
            draw_grasp_debug_once(grasp_target_object, grasp_debug_state)

        if keyboard_state["select_best_grasp_once"]:
            keyboard_state["select_best_grasp_once"] = False
            select_best_grasp_and_set_ik_target(grasp_target_object, grasp_debug_state)

        if grasp_debug_state.get("use_best_grasp_ik_target"):
            target_position = grasp_debug_state["ik_target_position"]
            target_orientation = grasp_debug_state["ik_target_orientation"]
            target_cube.set_world_pose(position=target_position, orientation=target_orientation)
        else:
            target_position, target_orientation = target_cube.get_world_pose()
        apply_ik_to_robot(
            ik,
            my_robot_task,
            target_position=target_position,
            target_orientation=target_orientation,
            fallback_rpy=ik_target_rpy,
            obstacle_prim=obstacle,
            sphere_visuals=sphere_visuals,
        )

simulation_app.close()
