import importlib.util
import json
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
VIZ_SCRIPT_PATH = THIS_DIR / "grasp_data_viz.py"

# =========================
# Pose visualization params
# =========================
DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen")
OBJECT_FOLDER_NAME = "aircon_remote"
OBJECT_DATA_ROOT = DATASET_ROOT / OBJECT_FOLDER_NAME

CONF_DIR = OBJECT_DATA_ROOT / "conf"
SCENE_START = 0
SCENE_END = None
SCENE_LIST = []
OBJECT_NAME = None

MESH_PATH = None
MESH_UNIT_SCALE = 0.01
SHOW_FRAME = True


KEY_RIGHT = 262
KEY_LEFT = 263


def load_viz_module():
    spec = importlib.util.spec_from_file_location("grasp_data_viz", VIZ_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path):
    with path.open("r") as f:
        return json.load(f)


def scene_ids_for_conf(conf_dir):
    return sorted(path.stem for path in conf_dir.glob("*.json"))


def select_scene_ids(viz, scene_ids, scene_start=None, scene_end=None, scene_list=None):
    return viz.select_scene_ids(
        scene_ids,
        scene_start=scene_start,
        scene_end=scene_end,
        scene_list=scene_list,
    )


def get_object_conf(viz, scene_id):
    conf_path = CONF_DIR / f"{scene_id}.json"
    conf = load_json(conf_path)
    obj_conf = viz.find_object_conf(conf, OBJECT_NAME)
    if obj_conf is None:
        raise RuntimeError(f"No object matched. scene={scene_id}, object={OBJECT_NAME}")
    return obj_conf


def make_scene_geometry(o3d, viz, scene_id):
    obj_conf = get_object_conf(viz, scene_id)
    mesh, mesh_path = viz.load_object_geometry(
        o3d,
        obj_conf,
        mesh_override=MESH_PATH,
        mesh_unit_scale=MESH_UNIT_SCALE,
        zero_pose=False,
    )
    return mesh, mesh_path, obj_conf


def set_camera_to_geometry(vis, geometries):
    min_bounds = []
    max_bounds = []
    for geom in geometries:
        bbox = geom.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        if np.all(np.isfinite(extent)) and np.linalg.norm(extent) > 0:
            min_bounds.append(bbox.get_min_bound())
            max_bounds.append(bbox.get_max_bound())

    if not min_bounds:
        return

    min_bound = np.min(np.asarray(min_bounds), axis=0)
    max_bound = np.max(np.asarray(max_bounds), axis=0)
    center = (min_bound + max_bound) / 2.0

    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_front([0.4, -0.7, 0.55])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.75)


def print_scene_info(scene_id, scene_idx, total, mesh_path, obj_conf):
    print("")
    print(f"scene: {scene_id} ({scene_idx + 1}/{total})")
    print(f"mesh: {mesh_path}")
    print(f"translate: {obj_conf.get('translate')}")
    print(f"orient: {obj_conf.get('orient')}")
    print(f"scale: {obj_conf.get('scale')}")


def main():
    import open3d as o3d

    if not CONF_DIR.exists():
        raise FileNotFoundError(f"conf dir not found: {CONF_DIR}")

    viz = load_viz_module()
    scene_ids = select_scene_ids(
        viz,
        scene_ids_for_conf(CONF_DIR),
        scene_start=SCENE_START,
        scene_end=SCENE_END,
        scene_list=SCENE_LIST,
    )
    if not scene_ids:
        raise RuntimeError(f"No conf json files selected: {CONF_DIR}")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"object pose viz - {OBJECT_FOLDER_NAME}",
        width=1280,
        height=900,
    )

    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([0.03, 0.03, 0.035])

    state = {
        "index": 0,
        "mesh": None,
        "frame": None,
    }

    def load_scene(index, reset_camera=False):
        index = max(0, min(index, len(scene_ids) - 1))
        scene_id = scene_ids[index]
        mesh, mesh_path, obj_conf = make_scene_geometry(o3d, viz, scene_id)

        if state["mesh"] is not None:
            vis.remove_geometry(state["mesh"], reset_bounding_box=False)
        if state["frame"] is not None:
            vis.remove_geometry(state["frame"], reset_bounding_box=False)

        vis.add_geometry(mesh, reset_bounding_box=reset_camera)
        geometries = [mesh]

        frame = None
        if SHOW_FRAME:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
            vis.add_geometry(frame, reset_bounding_box=False)
            geometries.append(frame)

        state["index"] = index
        state["mesh"] = mesh
        state["frame"] = frame
        print_scene_info(scene_id, index, len(scene_ids), mesh_path, obj_conf)

        if reset_camera:
            set_camera_to_geometry(vis, geometries)

        vis.update_renderer()
        return False

    def next_scene(vis):
        return load_scene((state["index"] + 1) % len(scene_ids))

    def prev_scene(vis):
        return load_scene((state["index"] - 1) % len(scene_ids))

    vis.register_key_callback(KEY_RIGHT, next_scene)
    vis.register_key_callback(KEY_LEFT, prev_scene)
    vis.register_key_callback(ord("D"), next_scene)
    vis.register_key_callback(ord("A"), prev_scene)

    print("keyboard: Right/D=next scene, Left/A=previous scene")
    load_scene(0, reset_camera=True)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
