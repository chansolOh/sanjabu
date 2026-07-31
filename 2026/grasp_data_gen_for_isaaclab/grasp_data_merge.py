import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


THIS_DIR = Path(__file__).resolve().parent
VIZ_SCRIPT_PATH = THIS_DIR / "grasp_data_viz.py"

# =================
# Merge parameters
# =================
DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen")
OBJECT_FOLDER_NAME = "wireless_charging_stand"
OBJECT_DATA_ROOT = DATASET_ROOT / OBJECT_FOLDER_NAME

CONF_DIR = OBJECT_DATA_ROOT / "conf"
GRASP_DIR = OBJECT_DATA_ROOT / "output_grasp"

SCENE_START = 0
SCENE_END = 100
SCENE_LIST = []
OBJECT_NAME = None

MESH_PATH = None
MESH_UNIT_SCALE = 0.01
STRIDE = 1

# Apply all filters once, then save merged grasps.
APPLY_EMPTY_BOX_FILTER = True
APPLY_SCORE_FILTER = True
APPLY_OUTLIER_FILTER = True
APPLY_NMS = True

BOX_THICKNESS = 0.02
BOX_MARGIN = 0.001
INCLUDE_TRIANGLE_CENTERS = True
FILTER_ENDPOINT_INSIDE_MESH = True
ENDPOINT_INSIDE_INDICES = (0, 3)

SCORE_TOP_RATIO = 0.5

OUTLIER_FILTER_BY_SCENE = True
OUTLIER_PCA_COMPONENTS = 3
OUTLIER_Z_THRESHOLD = 3.5
OUTLIER_MIN_KEEP_RATIO = 0.2

NMS_CENTER_THRESHOLD = 0.015
NMS_ROTATION_THRESHOLD_DEG = 20.0

# For view-based grasp lookup, compare the camera/view direction with this
# object-frame gripper axis from grasp_mat[:3, :3].
APPROACH_AXIS_INDEX = 2
APPROACH_AXIS_SIGN = 1.0

VISUALIZE = False
SHOW_FRAME = False

OUTPUT_JSON_PATH = Path(
    OBJECT_DATA_ROOT / f"{OBJECT_FOLDER_NAME}_merged_grasp_zero_pose.json"
)
OUTPUT_NPZ_PATH = Path(
    OBJECT_DATA_ROOT / f"{OBJECT_FOLDER_NAME}_merged_grasp_zero_pose_index.npz"
)


def load_viz_module():
    spec = importlib.util.spec_from_file_location("grasp_data_viz", VIZ_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_viz_module(viz):
    viz.MESH_PATH = MESH_PATH
    viz.MESH_UNIT_SCALE = MESH_UNIT_SCALE
    viz.OBJECT_NAME = OBJECT_NAME
    viz.STRIDE = STRIDE

    viz.FILTER_EMPTY_BOX = APPLY_EMPTY_BOX_FILTER
    viz.BOX_THICKNESS = BOX_THICKNESS
    viz.BOX_MARGIN = BOX_MARGIN
    viz.INCLUDE_TRIANGLE_CENTERS = INCLUDE_TRIANGLE_CENTERS
    viz.FILTER_ENDPOINT_INSIDE_MESH = FILTER_ENDPOINT_INSIDE_MESH
    viz.ENDPOINT_INSIDE_INDICES = ENDPOINT_INSIDE_INDICES

    viz.FILTER_SCORE_TOP = APPLY_SCORE_FILTER
    viz.SCORE_TOP_RATIO = SCORE_TOP_RATIO

    viz.USE_OUTLIER_FILTER = APPLY_OUTLIER_FILTER
    viz.OUTLIER_FILTER_BY_SCENE = OUTLIER_FILTER_BY_SCENE
    viz.OUTLIER_PCA_COMPONENTS = OUTLIER_PCA_COMPONENTS
    viz.OUTLIER_Z_THRESHOLD = OUTLIER_Z_THRESHOLD
    viz.OUTLIER_MIN_KEEP_RATIO = OUTLIER_MIN_KEEP_RATIO

    viz.USE_NMS = APPLY_NMS
    viz.NMS_CENTER_THRESHOLD = NMS_CENTER_THRESHOLD
    viz.NMS_ROTATION_THRESHOLD_DEG = NMS_ROTATION_THRESHOLD_DEG


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def normalized(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)


def build_merged_data(viz):
    import open3d as o3d

    if not CONF_DIR.exists():
        raise FileNotFoundError(f"conf dir not found: {CONF_DIR}")
    if not GRASP_DIR.exists():
        raise FileNotFoundError(f"grasp dir not found: {GRASP_DIR}")

    all_scene_ids = viz.scene_ids_for_aggregation(CONF_DIR, GRASP_DIR)
    scene_ids = viz.select_scene_ids(
        all_scene_ids,
        scene_start=SCENE_START,
        scene_end=SCENE_END,
        scene_list=SCENE_LIST,
    )
    if not scene_ids:
        raise RuntimeError(f"No matching scenes selected: {CONF_DIR}, {GRASP_DIR}")

    grasp_items, obj_conf, used_scene_ids = viz.load_scene_grasps_in_zero_pose(
        CONF_DIR,
        GRASP_DIR,
        scene_ids,
        object_name=OBJECT_NAME,
    )
    if obj_conf is None:
        raise RuntimeError(f"No object matched. object={OBJECT_NAME}")

    mesh, mesh_path = viz.load_object_geometry(
        o3d,
        obj_conf,
        mesh_override=MESH_PATH,
        mesh_unit_scale=MESH_UNIT_SCALE,
        zero_pose=True,
    )
    occupancy_points = viz.mesh_occupancy_points(
        [mesh],
        include_triangle_centers=INCLUDE_TRIANGLE_CENTERS,
    )
    mesh_inside_checker = viz.make_mesh_inside_checker(o3d, [mesh])

    object_filter = OBJECT_NAME if OBJECT_NAME is not None else obj_conf.get("class")
    raw_candidates = viz.filter_grasp_candidates(
        grasp_items,
        object_name=object_filter,
        stride=STRIDE,
    )
    (
        filtered_grasps,
        empty_filtered_grasps,
        score_filtered_grasps,
        outlier_filtered_grasps,
    ) = viz.build_filtered_grasps(
        raw_candidates,
        occupancy_points,
        mesh_inside_checker,
        use_empty_filter=APPLY_EMPTY_BOX_FILTER,
        use_score_filter=APPLY_SCORE_FILTER,
        use_nms=APPLY_NMS,
    )

    metadata = {
        "object": obj_conf.get("class"),
        "object_folder_name": OBJECT_FOLDER_NAME,
        "object_data_root": str(OBJECT_DATA_ROOT),
        "mesh_path": str(mesh_path),
        "conf_dir": str(CONF_DIR),
        "grasp_dir": str(GRASP_DIR),
        "scene_start": SCENE_START,
        "scene_end": SCENE_END,
        "scene_list": SCENE_LIST,
        "used_scene_ids": used_scene_ids,
        "coordinate": "object_zero_pose",
        "filters": {
            "empty_box": {
                "enabled": APPLY_EMPTY_BOX_FILTER,
                "box_thickness": BOX_THICKNESS,
                "box_margin": BOX_MARGIN,
                "include_triangle_centers": INCLUDE_TRIANGLE_CENTERS,
                "endpoint_inside_mesh_enabled": FILTER_ENDPOINT_INSIDE_MESH,
                "endpoint_indices": list(ENDPOINT_INSIDE_INDICES),
            },
            "score_top": {
                "enabled": APPLY_SCORE_FILTER,
                "top_ratio": SCORE_TOP_RATIO,
                "grouping": "scene_id",
            },
            "pca_outlier": {
                "enabled": APPLY_OUTLIER_FILTER and APPLY_NMS,
                "by_scene": OUTLIER_FILTER_BY_SCENE,
                "components": OUTLIER_PCA_COMPONENTS,
                "z_threshold": OUTLIER_Z_THRESHOLD,
                "min_keep_ratio": OUTLIER_MIN_KEEP_RATIO,
            },
            "nms": {
                "enabled": APPLY_NMS,
                "center_threshold": NMS_CENTER_THRESHOLD,
                "rotation_threshold_deg": NMS_ROTATION_THRESHOLD_DEG,
            },
            "lookup": {
                "approach_axis_index": APPROACH_AXIS_INDEX,
                "approach_axis_sign": APPROACH_AXIS_SIGN,
                "recommended_query": (
                    "Transform camera/view direction into object_zero_pose frame, "
                    "normalize it, then maximize approach_vector @ view_direction. "
                    "Use rotation_matrix or quat_xyzw if full orientation matching is needed."
                ),
            },
        },
        "counts": {
            "raw": len(raw_candidates),
            "after_empty_box": len(empty_filtered_grasps),
            "after_score": len(score_filtered_grasps),
            "after_outlier": len(outlier_filtered_grasps),
            "final": len(filtered_grasps),
        },
    }

    return metadata, filtered_grasps, mesh


def grasp_items_to_index_arrays(viz, metadata, grasp_items):
    num_grasps = len(grasp_items)
    grasp_box = np.zeros((num_grasps, 4, 3), dtype=np.float32)
    grasp_mat = np.zeros((num_grasps, 4, 4), dtype=np.float32)
    rotation_matrix = np.zeros((num_grasps, 3, 3), dtype=np.float32)
    grasp_center = np.zeros((num_grasps, 3), dtype=np.float32)
    target_points = np.full((num_grasps, 3), np.nan, dtype=np.float32)
    target_width = np.full((num_grasps,), np.nan, dtype=np.float32)
    score = np.full((num_grasps,), np.nan, dtype=np.float32)
    rotation_score = np.full((num_grasps,), np.nan, dtype=np.float32)
    normal = np.full((num_grasps, 3), np.nan, dtype=np.float32)

    scene_id = []
    target_object = []
    gripper_model = []
    gripper_type = []

    for idx, item in enumerate(grasp_items):
        box = viz.get_grasp_box(item)
        if box is None:
            continue

        grasp_box[idx] = box.astype(np.float32)
        grasp_center[idx] = box.mean(axis=0).astype(np.float32)

        if "grasp_mat" in item:
            mat = np.asarray(item["grasp_mat"], dtype=np.float32)
            if mat.shape == (4, 4):
                grasp_mat[idx] = mat
            else:
                grasp_mat[idx] = np.eye(4, dtype=np.float32)
        else:
            center, rot = viz.grasp_frame(item)
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = rot.astype(np.float32)
            mat[:3, 3] = center.astype(np.float32)
            grasp_mat[idx] = mat

        rot = grasp_mat[idx, :3, :3].astype(np.float64)
        u, _, vh = np.linalg.svd(rot)
        rot = u @ vh
        rotation_matrix[idx] = rot.astype(np.float32)

        if "target_points" in item:
            points = np.asarray(item["target_points"], dtype=np.float32)
            if points.shape == (3,):
                target_points[idx] = points
        if "target_width" in item:
            target_width[idx] = float(item["target_width"])
        if "score" in item:
            score[idx] = float(item["score"])
        if "rotation_score" in item:
            rotation_score[idx] = float(item["rotation_score"])
        if "normal" in item:
            normal_vec = np.asarray(item["normal"], dtype=np.float32)
            if normal_vec.shape == (3,):
                normal[idx] = normal_vec

        scene_id.append(str(item.get("scene_id", "")))
        target_object.append(str(item.get("target_object", "")))
        gripper_model.append(str(item.get("gripper_model", "")))
        gripper_type.append(str(item.get("gripper_type", "")))

    quat_xyzw = R.from_matrix(rotation_matrix.astype(np.float64)).as_quat().astype(np.float32)
    gripper_axes = rotation_matrix.copy()
    approach_vector = normalized(
        rotation_matrix[:, :, APPROACH_AXIS_INDEX] * float(APPROACH_AXIS_SIGN)
    )
    approach_vector_opposite = -approach_vector

    return {
        "metadata_json": np.asarray(json.dumps(metadata)),
        "grasp_box": grasp_box,
        "grasp_mat": grasp_mat,
        "grasp_center": grasp_center,
        "rotation_matrix": rotation_matrix,
        "quat_xyzw": quat_xyzw,
        "gripper_axes": gripper_axes,
        "approach_vector": approach_vector,
        "approach_vector_opposite": approach_vector_opposite,
        "target_points": target_points,
        "target_width": target_width,
        "score": score,
        "rotation_score": rotation_score,
        "normal": normal,
        "scene_id": np.asarray(scene_id),
        "target_object": np.asarray(target_object),
        "gripper_model": np.asarray(gripper_model),
        "gripper_type": np.asarray(gripper_type),
    }


def visualize(viz, metadata, filtered_grasps, mesh):
    import open3d as o3d

    geometries = [mesh]
    grasp_lines, grasp_count = viz.make_grasp_lineset(
        o3d,
        filtered_grasps,
        max_grasps=0,
    )
    geometries.append(grasp_lines)
    if SHOW_FRAME:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"merged grasp - {metadata['object']} ({grasp_count})",
        width=1280,
        height=900,
    )
    for geom in geometries:
        vis.add_geometry(geom)

    render_opt = vis.get_render_option()
    render_opt.background_color = [0.03, 0.03, 0.035]
    render_opt.line_width = 3.0

    vis.run()
    vis.destroy_window()


def main():
    viz = load_viz_module()
    configure_viz_module(viz)

    metadata, filtered_grasps, mesh = build_merged_data(viz)
    index_arrays = grasp_items_to_index_arrays(viz, metadata, filtered_grasps)
    save_json(
        OUTPUT_JSON_PATH,
        {
            "metadata": metadata,
            "data": filtered_grasps,
        },
    )
    save_npz(OUTPUT_NPZ_PATH, index_arrays)

    print(f"saved json: {OUTPUT_JSON_PATH}")
    print(f"saved npz: {OUTPUT_NPZ_PATH}")
    print(f"object: {metadata['object']}")
    print(f"used scenes: {len(metadata['used_scene_ids'])}")
    print(f"raw: {metadata['counts']['raw']}")
    print(f"after empty-box: {metadata['counts']['after_empty_box']}")
    print(f"after score: {metadata['counts']['after_score']}")
    print(f"after outlier: {metadata['counts']['after_outlier']}")
    print(f"final: {metadata['counts']['final']}")

    if VISUALIZE:
        visualize(viz, metadata, filtered_grasps, mesh)


if __name__ == "__main__":
    main()
