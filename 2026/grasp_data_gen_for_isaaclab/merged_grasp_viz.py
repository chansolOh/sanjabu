import importlib.util
import json
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
VIZ_SCRIPT_PATH = THIS_DIR / "grasp_data_viz.py"

# ===========================
# Merged grasp visualization
# ===========================
DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen")
OBJECT_FOLDER_NAME = "wireless_charging_stand"
OBJECT_DATA_ROOT = DATASET_ROOT / OBJECT_FOLDER_NAME

MERGED_NPZ_PATH = OBJECT_DATA_ROOT / f"{OBJECT_FOLDER_NAME}_merged_grasp_zero_pose_index.npz"
MERGED_JSON_PATH = OBJECT_DATA_ROOT / f"{OBJECT_FOLDER_NAME}_merged_grasp_zero_pose.json"

CONF_DIR = OBJECT_DATA_ROOT / "conf"
MESH_UNIT_SCALE = 0.01
MAX_GRASPS = 0
SHOW_FRAME = False
CAMERA_CANDIDATE_COUNT = 100
CAMERA_DIRECTION_PREFILTER_COUNT = 200
CAMERA_NORMAL_PREFILTER_COUNT = 100
CAMERA_DIRECTION_MIN_SIMILARITY = 0.0
FINAL_SCORE_WEIGHT = 0.5
FINAL_ROTATION_SCORE_WEIGHT = 0.5

# The saved gravity normal is currently opposite to the direction we want to use.
INVERT_NORMAL_VECTOR = True

# The grasp box facing direction is opposite to the center vector in this dataset.
INVERT_GRASP_BOX_FACING = True


def load_viz_module():
    spec = importlib.util.spec_from_file_location("grasp_data_viz", VIZ_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path):
    with path.open("r") as f:
        return json.load(f)


def load_metadata_from_npz(npz_data):
    if "metadata_json" not in npz_data.files:
        return {}
    return json.loads(str(npz_data["metadata_json"]))


def reference_object_conf(viz, metadata):
    mesh_path = metadata.get("mesh_path")
    object_name = metadata.get("object")
    conf_dir = Path(metadata.get("conf_dir", CONF_DIR))

    for conf_path in sorted(conf_dir.glob("*.json")):
        conf = load_json(conf_path)
        obj_conf = viz.find_object_conf(conf, object_name)
        if obj_conf is None:
            continue
        if mesh_path is None or str(obj_conf.get("usd_path")) == str(mesh_path).replace(".obj", ".usd"):
            return obj_conf
        return obj_conf

    return {
        "class": object_name or OBJECT_FOLDER_NAME,
        "usd_path": mesh_path,
        "translate": [0.0, 0.0, 0.0],
        "orient": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    }


def npz_to_grasp_items(npz_data):
    boxes = np.asarray(npz_data["grasp_box"])
    rotation_scores = np.asarray(npz_data["rotation_score"]) if "rotation_score" in npz_data.files else None
    normals = np.asarray(npz_data["normal"]) if "normal" in npz_data.files else None
    scores = np.asarray(npz_data["score"]) if "score" in npz_data.files else None
    grasp_mats = np.asarray(npz_data["grasp_mat"]) if "grasp_mat" in npz_data.files else None
    scene_ids = np.asarray(npz_data["scene_id"]) if "scene_id" in npz_data.files else None

    items = []
    for idx, box in enumerate(boxes):
        item = {
            "grasp_box": box.tolist(),
        }
        if rotation_scores is not None and np.isfinite(rotation_scores[idx]):
            item["rotation_score"] = float(rotation_scores[idx])
        if normals is not None and np.all(np.isfinite(normals[idx])):
            normal = np.asarray(normals[idx], dtype=float)
            if INVERT_NORMAL_VECTOR:
                normal = -normal
            item["normal"] = normal.tolist()
        if scores is not None and np.isfinite(scores[idx]):
            item["score"] = float(scores[idx])
        if grasp_mats is not None:
            item["grasp_mat"] = grasp_mats[idx].tolist()
        if scene_ids is not None:
            item["scene_id"] = str(scene_ids[idx])
        items.append(item)

    return items


def normalized(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-9 or not np.isfinite(norm):
        return None
    return vector / norm


def camera_view_direction(vis):
    ctr = vis.get_view_control()
    if hasattr(ctr, "get_front"):
        front = normalized(ctr.get_front())
        if front is not None:
            return front

    params = ctr.convert_to_pinhole_camera_parameters()
    rotation_world_to_camera = params.extrinsic[:3, :3]
    camera_to_world = rotation_world_to_camera.T
    return normalized(camera_to_world @ np.asarray([0.0, 0.0, -1.0]))


def grasp_center(item):
    box = np.asarray(item.get("grasp_box", []), dtype=float)
    if box.shape != (4, 3):
        return None
    return box.mean(axis=0)


def safe_score(item, key, default=0.0):
    value = item.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return value


def final_quality_score(item):
    score = safe_score(item, "score")
    rotation_score = safe_score(item, "rotation_score")
    return FINAL_SCORE_WEIGHT * score + FINAL_ROTATION_SCORE_WEIGHT * rotation_score


def select_camera_normal_candidates(
    grasp_items,
    view_direction,
    object_center,
    top_k=10,
):
    view_direction = normalized(view_direction)
    if view_direction is None:
        return []

    camera_facing_direction = -view_direction
    object_center = np.asarray(object_center, dtype=float)
    candidates = []
    for idx, item in enumerate(grasp_items):
        center = grasp_center(item)
        normal = normalized(item.get("normal"))
        if center is None or normal is None:
            continue

        center_direction = normalized(center - object_center)
        if center_direction is None:
            continue
        if INVERT_GRASP_BOX_FACING:
            center_direction = -center_direction

        center_similarity = float(np.dot(camera_facing_direction, center_direction))
        if center_similarity < CAMERA_DIRECTION_MIN_SIMILARITY:
            continue

        normal_similarity = float(np.dot(camera_facing_direction, normal))
        score = safe_score(item, "score")
        rotation_score = safe_score(item, "rotation_score")
        quality_score = final_quality_score(item)
        candidates.append(
            (
                center_similarity,
                normal_similarity,
                quality_score,
                score,
                rotation_score,
                idx,
                item,
            )
        )

    if not candidates:
        return []

    direction_count = max(top_k, min(CAMERA_DIRECTION_PREFILTER_COUNT, len(candidates)))
    candidates.sort(key=lambda row: (row[0], row[2]), reverse=True)
    direction_filtered = candidates[:direction_count]

    normal_count = max(top_k, min(CAMERA_NORMAL_PREFILTER_COUNT, len(direction_filtered)))
    direction_filtered.sort(key=lambda row: (row[1], row[2]), reverse=True)
    normal_filtered = direction_filtered[:normal_count]

    normal_filtered.sort(key=lambda row: (row[2], row[3], row[4], row[1], row[0]), reverse=True)
    return normal_filtered[:top_k]


def rank_gradient_color(rank, count):
    if count <= 1:
        return [1.0, 0.92, 0.0]

    t = rank / float(count - 1)
    yellow = np.asarray([1.0, 0.92, 0.0])
    green = np.asarray([0.0, 0.9, 0.2])
    blue = np.asarray([0.0, 0.25, 1.0])

    if t < 0.5:
        local_t = t / 0.5
        color = (1.0 - local_t) * yellow + local_t * green
    else:
        local_t = (t - 0.5) / 0.5
        color = (1.0 - local_t) * green + local_t * blue
    return color.tolist()


def make_ranked_grasp_lineset(o3d, grasp_items, max_grasps=0):
    points = []
    lines = []
    colors = []
    selected_items = grasp_items[: max_grasps if max_grasps > 0 else len(grasp_items)]

    for rank, item in enumerate(selected_items):
        grasp_box = np.asarray(item.get("grasp_box", []), dtype=float)
        if grasp_box.shape != (4, 3):
            continue

        base_idx = len(points)
        color = rank_gradient_color(rank, len(selected_items))
        points.extend(grasp_box.tolist())
        for start, end in ((0, 1), (1, 2), (2, 3)):
            lines.append([base_idx + start, base_idx + end])
            colors.append(color)

    line_set = o3d.geometry.LineSet()
    if points:
        line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))

    return line_set, len(selected_items)


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


def main():
    import open3d as o3d

    if not MERGED_NPZ_PATH.exists():
        raise FileNotFoundError(f"merged npz not found: {MERGED_NPZ_PATH}")

    viz = load_viz_module()
    npz_data = np.load(MERGED_NPZ_PATH)
    metadata = load_metadata_from_npz(npz_data)
    grasp_items = npz_to_grasp_items(npz_data)

    obj_conf = reference_object_conf(viz, metadata)
    mesh_override = Path(metadata["mesh_path"]) if metadata.get("mesh_path") else None
    mesh, mesh_path = viz.load_object_geometry(
        o3d,
        obj_conf,
        mesh_override=mesh_override,
        mesh_unit_scale=MESH_UNIT_SCALE,
        zero_pose=True,
    )
    object_center = mesh.get_axis_aligned_bounding_box().get_center()

    geometries = [mesh]
    grasp_lines, grasp_count = viz.make_grasp_lineset(
        o3d,
        grasp_items,
        max_grasps=MAX_GRASPS,
    )
    normal_lines = viz.make_grasp_normal_lineset(
        o3d,
        grasp_items,
        max_grasps=MAX_GRASPS,
    )
    geometries.extend([grasp_lines, normal_lines])

    if SHOW_FRAME:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12))

    print(f"npz: {MERGED_NPZ_PATH}")
    print(f"json: {MERGED_JSON_PATH if MERGED_JSON_PATH.exists() else 'not found'}")
    print(f"object: {metadata.get('object', OBJECT_FOLDER_NAME)}")
    print(f"mesh: {mesh_path}")
    print(f"grasps: {grasp_count}")

    dynamic_geometries = {
        "grasp_lines": grasp_lines,
        "normal_lines": normal_lines,
        "mode": "all",
    }

    def redraw_grasps(vis, items, mode_name, ranked_colors=False):
        for key in ("grasp_lines", "normal_lines"):
            geom = dynamic_geometries.get(key)
            if geom is not None:
                vis.remove_geometry(geom, reset_bounding_box=False)

        if ranked_colors:
            new_grasp_lines, new_grasp_count = make_ranked_grasp_lineset(
                o3d,
                items,
                max_grasps=MAX_GRASPS,
            )
        else:
            new_grasp_lines, new_grasp_count = viz.make_grasp_lineset(
                o3d,
                items,
                max_grasps=MAX_GRASPS,
            )
        new_normal_lines = viz.make_grasp_normal_lineset(
            o3d,
            items,
            max_grasps=MAX_GRASPS,
        )
        vis.add_geometry(new_grasp_lines, reset_bounding_box=False)
        vis.add_geometry(new_normal_lines, reset_bounding_box=False)

        dynamic_geometries["grasp_lines"] = new_grasp_lines
        dynamic_geometries["normal_lines"] = new_normal_lines
        dynamic_geometries["mode"] = mode_name
        vis.update_renderer()
        return new_grasp_count

    def show_camera_candidates(vis):
        view_direction = camera_view_direction(vis)
        selected = select_camera_normal_candidates(
            grasp_items,
            view_direction,
            object_center,
            top_k=CAMERA_CANDIDATE_COUNT,
        )
        selected_items = [item for _, _, _, _, _, _, item in selected]
        shown_count = redraw_grasps(
            vis,
            selected_items,
            "camera_candidates",
            ranked_colors=True,
        )

        print("\n[C] camera-facing grasp candidates")
        print(f"view direction: {view_direction.tolist()}")
        print(f"camera-facing direction: {(-view_direction).tolist()}")
        print(f"direction prefilter: {CAMERA_DIRECTION_PREFILTER_COUNT}")
        print(f"normal prefilter: {CAMERA_NORMAL_PREFILTER_COUNT}")
        print(f"shown grasps: {shown_count}/{len(grasp_items)}")
        for rank, (
            center_similarity,
            normal_similarity,
            quality_score,
            score,
            rotation_score,
            idx,
            item,
        ) in enumerate(
            selected,
            start=1,
        ):
            scene_id = item.get("scene_id", "?")
            rotation_score = item.get("rotation_score", "?")
            print(
                f"{rank:02d}. idx={idx}, scene={scene_id}, "
                f"center_sim={center_similarity:.4f}, "
                f"normal_sim={normal_similarity:.4f}, "
                f"quality={quality_score:.4f}, "
                f"score={score:.4f}, "
                f"rotation_score={rotation_score:.4f}"
            )
        return False

    def show_all_grasps(vis):
        shown_count = redraw_grasps(vis, grasp_items, "all")
        print(f"\n[A] show all grasps: {shown_count}")
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"merged grasp viz - {metadata.get('object', OBJECT_FOLDER_NAME)}",
        width=1280,
        height=900,
    )
    for geom in geometries:
        vis.add_geometry(geom)

    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([0.03, 0.03, 0.035])
    render_opt.line_width = 3.0

    set_camera_to_geometry(vis, geometries)
    vis.register_key_callback(ord("C"), show_camera_candidates)
    vis.register_key_callback(ord("A"), show_all_grasps)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
