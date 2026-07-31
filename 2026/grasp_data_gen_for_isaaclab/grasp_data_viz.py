import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# object_name = "dslr_camera"
object_name = "wireless_charging_stand"
# object_name = "barcode_scanner"
DEFAULT_DATA_ROOT = Path(f"/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen/{object_name}")
DEFAULT_CONF_DIR = DEFAULT_DATA_ROOT / "conf"
DEFAULT_GRASP_DIR = DEFAULT_DATA_ROOT / "output_grasp"

# =========================
# Visualization parameters
# =========================
SCENE = "0000"
CONF_DIR = DEFAULT_CONF_DIR
GRASP_DIR = DEFAULT_GRASP_DIR
AGGREGATE_ALL_SCENES = True
SCENE_START =15
SCENE_END = 20
SCENE_LIST = []

# Set MESH_PATH to a specific .obj/.glb/.ply path to override the conf usd_path.
MESH_PATH = None

# OBJ scan vertices are usually centimeters. 0.01 converts cm to meters.
MESH_UNIT_SCALE = 0.01
OBJECT_NAME = None

MAX_GRASPS = 0
STRIDE = 1
CLOSE_BOX = False
SHOW_FRAME = False
DRY_RUN = False
ENABLE_KEY_TOGGLE = True

# If True, filters only remove from the same initial visible candidate pool.
# This makes keyboard toggling intuitive: NMS cannot reveal later unseen boxes.
LIMIT_CANDIDATES_BEFORE_FILTER = False

# Keep only grasp boxes whose thin 3D slab contains mesh occupancy points.
FILTER_EMPTY_BOX = False
BOX_THICKNESS = 0.02
BOX_MARGIN = 0.001
INCLUDE_TRIANGLE_CENTERS = True
FILTER_ENDPOINT_INSIDE_MESH = True
ENDPOINT_INSIDE_INDICES = (0, 3)

# Keep top score ratio for each scene before outlier/NMS filtering.
FILTER_SCORE_TOP = False
SCORE_TOP_RATIO = 0.5

# Remove duplicate-like grasps only when both center and rotation are close.
USE_NMS = False
USE_OUTLIER_FILTER = True
OUTLIER_FILTER_BY_SCENE = True
OUTLIER_PCA_COMPONENTS = 3
OUTLIER_Z_THRESHOLD = 3.5
OUTLIER_MIN_KEEP_RATIO = 0.2
NMS_CENTER_THRESHOLD = 0.015
NMS_ROTATION_THRESHOLD_DEG = 20.0

NORMAL_LINE_SCALE = 0.08
NORMAL_LINE_COLOR = [1.0, 0.0, 0.75]


def normalize_scene_id(scene):
    scene = str(scene)
    return f"{int(scene):04d}" if scene.isdigit() else scene


def load_json(path):
    with path.open("r") as f:
        return json.load(f)


def resolve_mesh_path(usd_or_mesh_path):
    """Open3D cannot read USD reliably, so prefer same-folder mesh files."""
    path = Path(usd_or_mesh_path)
    if path.suffix.lower() in {".obj", ".ply", ".stl", ".glb", ".gltf"}:
        return path

    candidates = []
    for suffix in (".obj", ".glb", ".gltf", ".ply", ".stl"):
        candidates.append(path.with_suffix(suffix))
    candidates.extend(path.parent.glob("*.obj"))
    candidates.extend(path.parent.glob("*.glb"))
    candidates.extend(path.parent.glob("*.gltf"))
    candidates.extend(path.parent.glob("*.ply"))
    candidates.extend(path.parent.glob("*.stl"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path


def make_object_transform(obj_conf, mesh_unit_scale=1.0):
    translate = np.asarray(obj_conf.get("translate", [0.0, 0.0, 0.0]), dtype=float)
    orient = np.asarray(obj_conf.get("orient", [0.0, 0.0, 0.0]), dtype=float)
    scale = np.asarray(obj_conf.get("scale", [1.0, 1.0, 1.0]), dtype=float)

    tf = np.eye(4)
    # scene_gen.py stores scipy Rotation.as_euler("xyz", degrees=True) values.
    tf[:3, :3] = (
        R.from_euler("xyz", orient, degrees=True).as_matrix()
        @ np.diag(scale * mesh_unit_scale)
    )
    tf[:3, 3] = translate
    return tf


def make_object_zero_pose_transform(obj_conf, mesh_unit_scale=1.0):
    scale = np.asarray(obj_conf.get("scale", [1.0, 1.0, 1.0]), dtype=float)

    tf = np.eye(4)
    tf[:3, :3] = np.diag(scale * mesh_unit_scale)
    return tf


def make_object_pose_transform(obj_conf):
    translate = np.asarray(obj_conf.get("translate", [0.0, 0.0, 0.0]), dtype=float)
    orient = np.asarray(obj_conf.get("orient", [0.0, 0.0, 0.0]), dtype=float)

    tf = np.eye(4)
    tf[:3, :3] = R.from_euler("xyz", orient, degrees=True).as_matrix()
    tf[:3, 3] = translate
    return tf


def load_object_geometry(o3d, obj_conf, mesh_override=None, mesh_unit_scale=1.0, zero_pose=False):
    mesh_path = mesh_override or resolve_mesh_path(obj_conf["usd_path"])
    if mesh_path.suffix.lower() == ".obj":
        mesh = load_mesh_with_trimesh(o3d, mesh_path)
    else:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
        if mesh.is_empty():
            mesh = load_mesh_with_trimesh(o3d, mesh_path)

    mesh.compute_vertex_normals()
    if zero_pose:
        mesh.transform(make_object_zero_pose_transform(obj_conf, mesh_unit_scale))
    else:
        mesh.transform(make_object_transform(obj_conf, mesh_unit_scale))

    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.72, 0.72, 0.72])

    return mesh, mesh_path


def load_mesh_with_trimesh(o3d, mesh_path):
    try:
        import trimesh
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Open3D failed to load mesh and trimesh is not installed: {mesh_path}"
        ) from exc

    trimesh_obj = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(trimesh_obj, trimesh.Scene):
        dumped = trimesh_obj.dump(concatenate=True)
        trimesh_obj = dumped if not isinstance(dumped, list) else trimesh.util.concatenate(dumped)

    if trimesh_obj.vertices is None or trimesh_obj.faces is None:
        raise RuntimeError(f"trimesh failed to load mesh: {mesh_path}")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(trimesh_obj.vertices, dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(trimesh_obj.faces, dtype=np.int32))
    return mesh


def get_grasp_box(item):
    grasp_box = np.asarray(item.get("grasp_box", []), dtype=float)
    if grasp_box.shape != (4, 3):
        return None
    return grasp_box


def normalize_grasp_items(grasp_json):
    if isinstance(grasp_json, list):
        return grasp_json
    if isinstance(grasp_json, dict):
        if isinstance(grasp_json.get("data"), list):
            return grasp_json["data"]
        if isinstance(grasp_json.get("grasps"), list):
            return grasp_json["grasps"]
    return []


def find_object_conf(conf, object_name=None):
    objects = conf.get("objects", [])
    if object_name is not None:
        objects = [obj for obj in objects if obj.get("class") == object_name]
    if not objects:
        return None
    return objects[0]


def transformed_grasp_item(item, inverse_pose_tf, scene_id):
    grasp_box = get_grasp_box(item)
    if grasp_box is None:
        return None

    tf = inverse_pose_tf
    grasp_box_h = np.column_stack([grasp_box, np.ones(len(grasp_box))])

    out = dict(item)
    out["grasp_box"] = (grasp_box_h @ tf.T)[:, :3].tolist()
    out["scene_id"] = scene_id

    if "grasp_mat" in item:
        grasp_mat = np.asarray(item["grasp_mat"], dtype=float)
        if grasp_mat.shape == (4, 4):
            new_grasp_mat = tf @ grasp_mat
            out["grasp_mat"] = new_grasp_mat.tolist()

    if "target_points" in item:
        target_points = np.asarray(item["target_points"], dtype=float)
        if target_points.shape == (3,):
            target_h = np.append(target_points, 1.0)
            out["target_points"] = (tf @ target_h)[:3].tolist()

    return out


def scene_ids_for_aggregation(conf_dir, grasp_dir):
    conf_ids = {path.stem for path in conf_dir.glob("*.json")}
    grasp_ids = {path.stem for path in grasp_dir.glob("*.json")}
    return sorted(conf_ids & grasp_ids)


def select_scene_ids(scene_ids, scene_start=None, scene_end=None, scene_list=None):
    if scene_list:
        wanted = {normalize_scene_id(scene_id) for scene_id in scene_list}
        return [scene_id for scene_id in scene_ids if scene_id in wanted]

    selected = scene_ids
    if scene_start is not None:
        start_id = normalize_scene_id(scene_start)
        selected = [scene_id for scene_id in selected if scene_id >= start_id]
    if scene_end is not None:
        end_id = normalize_scene_id(scene_end)
        selected = [scene_id for scene_id in selected if scene_id <= end_id]
    return selected


def load_scene_grasps_in_zero_pose(conf_dir, grasp_dir, scene_ids, object_name=None):
    aggregated = []
    reference_obj_conf = None
    used_scene_ids = []

    for scene_id in scene_ids:
        conf_path = conf_dir / f"{scene_id}.json"
        grasp_path = grasp_dir / f"{scene_id}.json"
        conf = load_json(conf_path)
        obj_conf = find_object_conf(conf, object_name)
        if obj_conf is None:
            continue

        if reference_obj_conf is None:
            reference_obj_conf = obj_conf

        pose_tf = make_object_pose_transform(obj_conf)
        inverse_pose_tf = np.linalg.inv(pose_tf)
        grasp_items = normalize_grasp_items(load_json(grasp_path))

        scene_count = 0
        for item in grasp_items:
            if object_name is not None and item.get("target_object") != object_name:
                continue
            transformed = transformed_grasp_item(item, inverse_pose_tf, scene_id)
            if transformed is None:
                continue
            aggregated.append(transformed)
            scene_count += 1

        if scene_count > 0:
            used_scene_ids.append(scene_id)

    return aggregated, reference_obj_conf, used_scene_ids


def filter_grasp_candidates(grasp_items, object_name=None, stride=1):
    candidates = []
    for item in grasp_items[:: max(1, stride)]:
        if object_name is not None and item.get("target_object") != object_name:
            continue
        if get_grasp_box(item) is None:
            continue
        candidates.append(item)
    return candidates


def grasp_frame(item):
    grasp_box = get_grasp_box(item)
    if grasp_box is None:
        return None, None

    center = grasp_box.mean(axis=0)

    if "grasp_mat" in item:
        rot = np.asarray(item["grasp_mat"], dtype=float)[:3, :3]
        u, _, vh = np.linalg.svd(rot)
        return center, u @ vh

    axis_x = grasp_box[1] - grasp_box[0]
    axis_y = grasp_box[3] - grasp_box[0]
    norm_x = np.linalg.norm(axis_x)
    norm_y = np.linalg.norm(axis_y)
    if norm_x < 1e-9 or norm_y < 1e-9:
        return center, np.eye(3)

    x_dir = axis_x / norm_x
    y_dir = axis_y - np.dot(axis_y, x_dir) * x_dir
    norm_y_dir = np.linalg.norm(y_dir)
    if norm_y_dir < 1e-9:
        return center, np.eye(3)

    y_dir = y_dir / norm_y_dir
    z_dir = np.cross(x_dir, y_dir)
    z_dir = z_dir / np.linalg.norm(z_dir)
    return center, np.column_stack([x_dir, y_dir, z_dir])


def mesh_vertices_in_grasp_slab(mesh_vertices, grasp_box, thickness=0.02, margin=0.002):
    center = grasp_box.mean(axis=0)
    edge_x = grasp_box[1] - grasp_box[0]
    edge_y = grasp_box[3] - grasp_box[0]
    len_x = np.linalg.norm(edge_x)
    len_y = np.linalg.norm(edge_y)
    if len_x < 1e-9 or len_y < 1e-9:
        return False

    x_dir = edge_x / len_x
    y_dir = edge_y - np.dot(edge_y, x_dir) * x_dir
    len_y_orth = np.linalg.norm(y_dir)
    if len_y_orth < 1e-9:
        return False

    y_dir = y_dir / len_y_orth
    z_dir = np.cross(x_dir, y_dir)
    z_dir = z_dir / np.linalg.norm(z_dir)

    rel = mesh_vertices - center
    local_x = rel @ x_dir
    local_y = rel @ y_dir
    local_z = rel @ z_dir

    return bool(
        np.any(
            (np.abs(local_x) <= len_x / 2.0 + margin)
            & (np.abs(local_y) <= len_y_orth / 2.0 + margin)
            & (np.abs(local_z) <= thickness / 2.0 + margin)
        )
    )


def filter_empty_grasp_boxes(grasp_items, mesh_vertices, thickness=0.02, margin=0.002):
    kept = []
    for item in grasp_items:
        grasp_box = get_grasp_box(item)
        if grasp_box is None:
            continue
        if mesh_vertices_in_grasp_slab(mesh_vertices, grasp_box, thickness=thickness, margin=margin):
            kept.append(item)
    return kept


def make_mesh_inside_checker(o3d, meshes):
    scene = o3d.t.geometry.RaycastingScene()
    mesh_count = 0
    for mesh in meshes:
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            continue
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(tensor_mesh)
        mesh_count += 1

    if mesh_count == 0:
        return None

    def is_inside(points):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(points) == 0:
            return np.zeros((0,), dtype=bool)
        query = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
        return np.asarray(scene.compute_occupancy(query).numpy()) > 0.5

    return is_inside


def filter_grasp_endpoints_inside_mesh(grasp_items, mesh_inside_checker):
    if mesh_inside_checker is None:
        return grasp_items

    kept = []
    endpoint_indices = list(ENDPOINT_INSIDE_INDICES)
    for item in grasp_items:
        grasp_box = get_grasp_box(item)
        if grasp_box is None:
            continue

        endpoints = grasp_box[endpoint_indices]
        if not np.any(mesh_inside_checker(endpoints)):
            kept.append(item)

    return kept


def mesh_occupancy_points(meshes, include_triangle_centers=True):
    points = []
    for mesh in meshes:
        vertices = np.asarray(mesh.vertices)
        if len(vertices) > 0:
            points.append(vertices)

        if include_triangle_centers:
            triangles = np.asarray(mesh.triangles)
            if len(vertices) > 0 and len(triangles) > 0:
                points.append(vertices[triangles].mean(axis=1))

    if not points:
        return np.empty((0, 3), dtype=float)
    return np.concatenate(points, axis=0)


def rotation_distance_deg(rot_a, rot_b):
    rel = rot_a.T @ rot_b
    cos_theta = (np.trace(rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def nms_grasps(grasp_items, center_threshold=0.01, rotation_threshold_deg=10.0):
    kept = []
    kept_frames = []

    for item in grasp_items:
        center, rot = grasp_frame(item)
        if center is None:
            continue

        is_duplicate = False
        for kept_center, kept_rot in kept_frames:
            center_dist = np.linalg.norm(center - kept_center)
            if center_dist > center_threshold:
                continue
            if rotation_distance_deg(rot, kept_rot) <= rotation_threshold_deg:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(item)
            kept_frames.append((center, rot))

    return kept


def pca_outlier_filter(
    grasp_items,
    pca_components=3,
    z_threshold=3.5,
    min_keep_ratio=0.2,
):
    frames = []
    for item in grasp_items:
        center, rot = grasp_frame(item)
        if center is None:
            continue
        frames.append((item, center))

    if len(frames) < 4:
        return [item for item, _ in frames]

    items = [item for item, _ in frames]
    centers = np.asarray([center for _, center in frames], dtype=float)

    robust_center = np.median(centers, axis=0)
    centered = centers - robust_center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    num_components = max(1, min(pca_components, vh.shape[0]))
    pca_basis = vh[:num_components].T
    projected = centered @ pca_basis

    projected_med = np.median(projected, axis=0)
    mad = np.median(np.abs(projected - projected_med), axis=0)
    mad = np.maximum(mad, 1e-9)
    robust_z = np.abs(projected - projected_med) / (1.4826 * mad)
    scores = robust_z.max(axis=1)
    keep_mask = scores <= z_threshold

    min_keep = int(np.ceil(len(items) * min_keep_ratio))
    if keep_mask.sum() < min_keep:
        keep_indices = np.argsort(scores)[:min_keep]
    else:
        keep_indices = np.flatnonzero(keep_mask)

    return [items[idx] for idx in keep_indices]


def pca_outlier_filter_by_scene(
    grasp_items,
    pca_components=3,
    z_threshold=3.5,
    min_keep_ratio=0.2,
):
    grouped = {}
    for item in grasp_items:
        scene_id = item.get("scene_id", "__single_scene__")
        grouped.setdefault(scene_id, []).append(item)

    filtered = []
    for scene_id in sorted(grouped):
        filtered.extend(
            pca_outlier_filter(
                grouped[scene_id],
                pca_components=pca_components,
                z_threshold=z_threshold,
                min_keep_ratio=min_keep_ratio,
            )
        )
    return filtered


def score_top_filter_by_scene(grasp_items, top_ratio=0.5):
    grouped = {}
    for item in grasp_items:
        scene_id = item.get("scene_id", "__single_scene__")
        grouped.setdefault(scene_id, []).append(item)

    filtered = []
    for scene_id in sorted(grouped):
        scene_items = grouped[scene_id]
        if not scene_items:
            continue

        scores = np.asarray(
            [float(item.get("score", -np.inf)) for item in scene_items],
            dtype=float,
        )
        if not np.any(np.isfinite(scores)):
            filtered.extend(scene_items)
            continue

        keep_count = int(np.ceil(len(scene_items) * top_ratio))
        keep_count = max(1, min(keep_count, len(scene_items)))
        order = np.argsort(scores)[::-1]
        filtered.extend([scene_items[idx] for idx in order[:keep_count]])

    return filtered


def build_filtered_grasps(
    grasp_candidates,
    occupancy_points,
    mesh_inside_checker,
    use_empty_filter,
    use_score_filter,
    use_nms,
):
    after_empty_filter = grasp_candidates
    if use_empty_filter:
        after_empty_filter = filter_empty_grasp_boxes(
            grasp_candidates,
            occupancy_points,
            thickness=BOX_THICKNESS,
            margin=BOX_MARGIN,
        )
        if FILTER_ENDPOINT_INSIDE_MESH:
            after_empty_filter = filter_grasp_endpoints_inside_mesh(
                after_empty_filter,
                mesh_inside_checker,
            )

    after_score_filter = after_empty_filter
    if use_score_filter:
        after_score_filter = score_top_filter_by_scene(
            after_empty_filter,
            top_ratio=SCORE_TOP_RATIO,
        )

    after_outlier_filter = after_score_filter
    after_nms = after_outlier_filter
    if use_nms:
        if USE_OUTLIER_FILTER:
            if OUTLIER_FILTER_BY_SCENE:
                after_outlier_filter = pca_outlier_filter_by_scene(
                    after_score_filter,
                    pca_components=OUTLIER_PCA_COMPONENTS,
                    z_threshold=OUTLIER_Z_THRESHOLD,
                    min_keep_ratio=OUTLIER_MIN_KEEP_RATIO,
                )
            else:
                after_outlier_filter = pca_outlier_filter(
                    after_score_filter,
                    pca_components=OUTLIER_PCA_COMPONENTS,
                    z_threshold=OUTLIER_Z_THRESHOLD,
                    min_keep_ratio=OUTLIER_MIN_KEEP_RATIO,
                )
        after_nms = nms_grasps(
            after_outlier_filter,
            center_threshold=NMS_CENTER_THRESHOLD,
            rotation_threshold_deg=NMS_ROTATION_THRESHOLD_DEG,
        )

    return after_nms, after_empty_filter, after_score_filter, after_outlier_filter


def limit_candidates_for_display(grasp_candidates, max_grasps, limit_before_filter=True):
    if not limit_before_filter or max_grasps <= 0:
        return grasp_candidates
    return grasp_candidates[:max_grasps]


def print_grasp_status(
    grasp_candidates,
    raw_candidate_count,
    grasp_after_empty_filter,
    grasp_after_score_filter,
    grasp_after_outlier_filter,
    grasp_after_nms,
    grasp_count,
    occupancy_point_count,
    use_empty_filter,
    use_score_filter,
    use_nms,
):
    print(
        "filters: "
        f"empty_box={'ON' if use_empty_filter else 'OFF'}, "
        f"score_top={'ON' if use_score_filter else 'OFF'}, "
        f"nms={'ON' if use_nms else 'OFF'}"
    )
    if raw_candidate_count == len(grasp_candidates):
        print(f"candidate grasps: {len(grasp_candidates)}")
    else:
        print(f"candidate grasps: {len(grasp_candidates)} shown-pool / {raw_candidate_count} total")
    if use_empty_filter:
        print(
            "after empty-box filter: "
            f"{len(grasp_after_empty_filter)} "
            f"(thickness={BOX_THICKNESS}, margin={BOX_MARGIN}, "
            f"points={occupancy_point_count}, "
            f"endpoint_inside={FILTER_ENDPOINT_INSIDE_MESH})"
        )
    if use_score_filter:
        print(
            "after score top filter: "
            f"{len(grasp_after_score_filter)} "
            f"(top_ratio={SCORE_TOP_RATIO})"
        )
    if use_nms:
        if USE_OUTLIER_FILTER:
            print(
                "after pca outlier filter: "
                f"{len(grasp_after_outlier_filter)} "
                f"(components={OUTLIER_PCA_COMPONENTS}, "
                f"z={OUTLIER_Z_THRESHOLD}, "
                f"min_keep={OUTLIER_MIN_KEEP_RATIO}, "
                f"by_scene={OUTLIER_FILTER_BY_SCENE})"
            )
        print(
            "after nms: "
            f"{len(grasp_after_nms)} "
            f"(center={NMS_CENTER_THRESHOLD}, rot={NMS_ROTATION_THRESHOLD_DEG} deg)"
        )
    print(f"draw grasps: {grasp_count}")


def make_grasp_lineset(o3d, grasp_items, max_grasps=300):
    points = []
    lines = []
    colors = []
    selected_count = 0

    for item in grasp_items:
        grasp_box = get_grasp_box(item)
        if grasp_box is None:
            continue

        base_idx = len(points)
        points.extend(grasp_box.tolist())

        edge_pairs = [(0, 1), (1, 2), (2, 3)]
        for start, end in edge_pairs:
            lines.append([base_idx + start, base_idx + end])
            if (start, end) == (1, 2):
                colors.append(rotation_score_color(item.get("rotation_score", 1.0)))
            else:
                colors.append([1.0, 0.12, 0.05])

        selected_count += 1
        if max_grasps > 0 and selected_count >= max_grasps:
            break

    line_set = o3d.geometry.LineSet()
    if points:
        line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))

    return line_set, selected_count


def rotation_score_color(rotation_score):
    score = float(np.clip(rotation_score, 0.0, 1.0))
    blue = np.asarray([0.0, 0.25, 1.0])
    green = np.asarray([0.0, 0.95, 0.2])
    yellow = np.asarray([1.0, 0.9, 0.0])

    if score < 0.5:
        t = score / 0.5
        color = (1.0 - t) * blue + t * green
    else:
        t = (score - 0.5) / 0.5
        color = (1.0 - t) * green + t * yellow
    return color.tolist()


def make_grasp_normal_lineset(o3d, grasp_items, max_grasps=300):
    points = []
    lines = []
    colors = []
    selected_count = 0

    for item in grasp_items:
        grasp_box = get_grasp_box(item)
        normal = item.get("normal")
        if grasp_box is None or normal is None:
            continue

        normal = np.asarray(normal, dtype=float)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm

        center = grasp_box.mean(axis=0)
        base_idx = len(points)
        points.append(center.tolist())
        points.append((center + normal * NORMAL_LINE_SCALE).tolist())
        lines.append([base_idx, base_idx + 1])
        colors.append(NORMAL_LINE_COLOR)

        selected_count += 1
        if max_grasps > 0 and selected_count >= max_grasps:
            break

    line_set = o3d.geometry.LineSet()
    if points:
        line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))

    return line_set


def main():
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "open3d is not installed in this uv environment.\n"
            "Install it with:\n"
            "  cd /home/uon/ochansol/isaac_code/isaac_chansol && uv add open3d"
        ) from exc

    geometries = []
    loaded_meshes = []
    object_meshes = []

    if AGGREGATE_ALL_SCENES:
        scene_ids = scene_ids_for_aggregation(CONF_DIR, GRASP_DIR)
        scene_ids = select_scene_ids(
            scene_ids,
            scene_start=SCENE_START,
            scene_end=SCENE_END,
            scene_list=SCENE_LIST,
        )
        if not scene_ids:
            raise RuntimeError(f"No matching conf/grasp json files: {CONF_DIR}, {GRASP_DIR}")

        grasp_items, obj_conf, used_scene_ids = load_scene_grasps_in_zero_pose(
            CONF_DIR,
            GRASP_DIR,
            scene_ids,
            object_name=OBJECT_NAME,
        )
        if obj_conf is None:
            raise RuntimeError(f"No object matched in aggregated scenes. object={OBJECT_NAME}")

        objects = [obj_conf]
        object_filter = OBJECT_NAME if OBJECT_NAME is not None else obj_conf.get("class")
        scene_label = f"aggregate {len(used_scene_ids)} scenes"
        if SCENE_LIST:
            scene_label += f" list={SCENE_LIST}"
        elif SCENE_START is not None or SCENE_END is not None:
            scene_label += f" range={SCENE_START}~{SCENE_END}"
        conf_label = str(CONF_DIR)
        grasp_label = str(GRASP_DIR)
    else:
        scene_id = normalize_scene_id(SCENE)
        conf_path = CONF_DIR / f"{scene_id}.json"
        grasp_path = GRASP_DIR / f"{scene_id}.json"
        conf = load_json(conf_path)
        grasp_items = normalize_grasp_items(load_json(grasp_path))

        objects = conf.get("objects", [])
        if OBJECT_NAME is not None:
            objects = [obj for obj in objects if obj.get("class") == OBJECT_NAME]
        if not objects:
            raise RuntimeError(f"No object matched. scene={scene_id}, object={OBJECT_NAME}")

        object_filter = OBJECT_NAME if OBJECT_NAME is not None else None
        if object_filter is None and len(objects) == 1:
            object_filter = objects[0].get("class")
        scene_label = scene_id
        conf_label = str(conf_path)
        grasp_label = str(grasp_path)

    for obj_conf in objects:
        mesh, mesh_path = load_object_geometry(
            o3d,
            obj_conf,
            mesh_override=MESH_PATH,
            mesh_unit_scale=MESH_UNIT_SCALE,
            zero_pose=AGGREGATE_ALL_SCENES,
        )
        geometries.append(mesh)
        object_meshes.append(mesh)
        loaded_meshes.append((obj_conf.get("class", "object"), mesh_path))

    occupancy_points = mesh_occupancy_points(
        object_meshes,
        include_triangle_centers=INCLUDE_TRIANGLE_CENTERS,
    )
    mesh_inside_checker = make_mesh_inside_checker(o3d, object_meshes)
    raw_grasp_candidates = filter_grasp_candidates(
        grasp_items,
        object_name=object_filter,
        stride=STRIDE,
    )
    grasp_candidates = limit_candidates_for_display(
        raw_grasp_candidates,
        MAX_GRASPS,
        limit_before_filter=LIMIT_CANDIDATES_BEFORE_FILTER,
    )
    (
        grasp_after_nms,
        grasp_after_empty_filter,
        grasp_after_score_filter,
        grasp_after_outlier_filter,
    ) = build_filtered_grasps(
        grasp_candidates,
        occupancy_points,
        mesh_inside_checker,
        use_empty_filter=FILTER_EMPTY_BOX,
        use_score_filter=FILTER_SCORE_TOP,
        use_nms=USE_NMS,
    )

    grasp_lines, grasp_count = make_grasp_lineset(
        o3d,
        grasp_after_nms,
        max_grasps=MAX_GRASPS,
    )
    geometries.append(grasp_lines)
    normal_lines = make_grasp_normal_lineset(o3d, grasp_after_nms, max_grasps=MAX_GRASPS)
    geometries.append(normal_lines)

    if SHOW_FRAME:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

    print(f"scene: {scene_label}")
    print(f"conf: {conf_label}")
    print(f"grasp: {grasp_label}")
    if AGGREGATE_ALL_SCENES:
        print(f"used scenes: {len(used_scene_ids)}")
    for obj_name, mesh_path in loaded_meshes:
        print(f"mesh[{obj_name}]: {mesh_path}")
    print_grasp_status(
        grasp_candidates=grasp_candidates,
        raw_candidate_count=len(raw_grasp_candidates),
        grasp_after_empty_filter=grasp_after_empty_filter,
        grasp_after_score_filter=grasp_after_score_filter,
        grasp_after_outlier_filter=grasp_after_outlier_filter,
        grasp_after_nms=grasp_after_nms,
        grasp_count=grasp_count,
        occupancy_point_count=len(occupancy_points),
        use_empty_filter=FILTER_EMPTY_BOX,
        use_score_filter=FILTER_SCORE_TOP,
        use_nms=USE_NMS,
    )

    if DRY_RUN:
        print("dry-run: geometry load complete")
        return

    vis = (
        o3d.visualization.VisualizerWithKeyCallback()
        if ENABLE_KEY_TOGGLE
        else o3d.visualization.Visualizer()
    )
    vis.create_window(window_name=f"grasp_data_viz - {scene_label}", width=1280, height=900)
    for geom in geometries:
        vis.add_geometry(geom)

    state = {
        "filter_empty_box": FILTER_EMPTY_BOX,
        "filter_score_top": FILTER_SCORE_TOP,
        "use_nms": USE_NMS,
        "grasp_lines": grasp_lines,
        "normal_lines": normal_lines,
    }

    def refresh_grasp_geometry(vis):
        (
            filtered_grasps,
            empty_filtered_grasps,
            score_filtered_grasps,
            outlier_filtered_grasps,
        ) = build_filtered_grasps(
            grasp_candidates,
            occupancy_points,
            mesh_inside_checker,
            use_empty_filter=state["filter_empty_box"],
            use_score_filter=state["filter_score_top"],
            use_nms=state["use_nms"],
        )
        new_lines, new_count = make_grasp_lineset(
            o3d,
            filtered_grasps,
            max_grasps=MAX_GRASPS,
        )
        new_normal_lines = make_grasp_normal_lineset(
            o3d,
            filtered_grasps,
            max_grasps=MAX_GRASPS,
        )

        vis.remove_geometry(state["grasp_lines"], reset_bounding_box=False)
        vis.remove_geometry(state["normal_lines"], reset_bounding_box=False)
        vis.add_geometry(new_lines, reset_bounding_box=False)
        vis.add_geometry(new_normal_lines, reset_bounding_box=False)
        state["grasp_lines"] = new_lines
        state["normal_lines"] = new_normal_lines

        print("")
        print_grasp_status(
            grasp_candidates=grasp_candidates,
            raw_candidate_count=len(raw_grasp_candidates),
            grasp_after_empty_filter=empty_filtered_grasps,
            grasp_after_score_filter=score_filtered_grasps,
            grasp_after_outlier_filter=outlier_filtered_grasps,
            grasp_after_nms=filtered_grasps,
            grasp_count=new_count,
            occupancy_point_count=len(occupancy_points),
            use_empty_filter=state["filter_empty_box"],
            use_score_filter=state["filter_score_top"],
            use_nms=state["use_nms"],
        )
        vis.update_renderer()
        return False

    def toggle_empty_filter(vis):
        state["filter_empty_box"] = not state["filter_empty_box"]
        return refresh_grasp_geometry(vis)

    def toggle_score_filter(vis):
        state["filter_score_top"] = not state["filter_score_top"]
        return refresh_grasp_geometry(vis)

    def toggle_nms_filter(vis):
        state["use_nms"] = not state["use_nms"]
        return refresh_grasp_geometry(vis)

    def reset_filters(vis):
        state["filter_empty_box"] = False
        state["filter_score_top"] = False
        state["use_nms"] = False
        return refresh_grasp_geometry(vis)

    if ENABLE_KEY_TOGGLE:
        vis.register_key_callback(ord("E"), toggle_empty_filter)
        vis.register_key_callback(ord("S"), toggle_score_filter)
        vis.register_key_callback(ord("N"), toggle_nms_filter)
        vis.register_key_callback(ord("R"), reset_filters)
        print("")
        print("keyboard: E=toggle empty-box filter, S=toggle score top filter, N=toggle NMS, R=turn all off")

    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([0.03, 0.03, 0.035])
    render_opt.line_width = 3.0
    render_opt.point_size = 5.0

    if geometries:
        min_bounds = []
        max_bounds = []
        for geom in geometries:
            bbox = geom.get_axis_aligned_bounding_box()
            extent_vec = bbox.get_extent()
            if np.all(np.isfinite(extent_vec)) and np.linalg.norm(extent_vec) > 0:
                min_bounds.append(bbox.get_min_bound())
                max_bounds.append(bbox.get_max_bound())

        if min_bounds:
            min_bound = np.min(np.asarray(min_bounds), axis=0)
            max_bound = np.max(np.asarray(max_bounds), axis=0)
            center = (min_bound + max_bound) / 2.0
            extent = np.linalg.norm(max_bound - min_bound)
        else:
            center = np.zeros(3)
            extent = 0.0

        ctr = vis.get_view_control()
        ctr.set_lookat(center)
        ctr.set_front([0.4, -0.7, 0.55])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.7 if extent > 0 else 1.0)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
