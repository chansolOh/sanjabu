import json
import gc
from pathlib import Path

from isaacsim import SimulationApp


ROOT_DIR = Path("/nas/ochansol/3d_model/peel3_scan_data_2026")
HEADLESS = True
IMAGE_SIZE = (640, 480)
OBJECT_SCALE = 0.01
MARGIN = 0.10
CAMERA_CLEARANCE_RATIO = 0.35
MIN_CAMERA_CLEARANCE = 0.03
SPP = 256
SPP_BATCH = 32
PATH_TRACING_MAX_BOUNCES = 12
MAX_FIT_ITER = 16
SCENE_WARMUP_FRAMES = 3
READBACK_DELAY_FRAMES = 2
OVERWRITE = False


simulation_app = SimulationApp({"headless": HEADLESS})

import carb.settings
import numpy as np
import omni
import omni.graph.core as og
import omni.kit.app
import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.utils import bounds as bounds_utils
from PIL import Image
from pxr import Gf, Sdf, Semantics, UsdGeom, UsdLux


settings = carb.settings.get_settings()
settings.set("/rtx/useTextureStreaming", False)
settings.set("/rtx/useAsyncTextureUpload", False)
settings.set("/rtx/textureCacheSize", 0)


CAMERA_VIEWS = {
    "front": np.array([0.0, -1.0, 0.0]),
    "side": np.array([1.0, 0.0, 0.0]),
    "rear": np.array([0.0, 1.0, 0.0]),
    "top": np.array([0.0, 0.0, 1.0]),
    "front_quarter": np.array([1.0, -1.0, 1.0]),
    "rear_quarter": np.array([1.0, 1.0, 1.0]),
}
EXPECTED_RENDER_FILES = {f"{view_name}.png" for view_name in CAMERA_VIEWS}
RENDER_LIGHT_PATHS = [
    Sdf.Path("/World/RenderDomeLight"),
    Sdf.Path("/World/RenderKeyLight"),
]
REPLICATOR_OBJECT_ROOT = Sdf.Path("/Replicator/Objects")


def find_object_folders(root_dir):
    if not root_dir.exists():
        raise FileNotFoundError(f"root dir not found: {root_dir}")
    return sorted(path for path in root_dir.iterdir() if path.is_dir())


def find_usd_file(object_dir):
    edited_dir = object_dir / "edited"
    if not edited_dir.exists():
        return None
    usd_files = sorted(list(edited_dir.glob("*.usd")) + list(edited_dir.glob("*.usda")) + list(edited_dir.glob("*.usdc")))
    if not usd_files:
        return None
    preferred = edited_dir / f"{object_dir.name}.usd"
    return preferred if preferred.exists() else usd_files[0]


def safe_prim_name(name):
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def clear_prim(path):
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)


def child_path_snapshot(stage, parent_path):
    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        return set()
    return {child.GetPath().pathString for child in parent.GetChildren()}


def remove_new_children(stage, parent_path, baseline_child_paths):
    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        return 0
    removed = 0
    for child in list(parent.GetChildren()):
        child_path = child.GetPath().pathString
        if child_path not in baseline_child_paths:
            stage.RemovePrim(child.GetPath())
            removed += 1
    return removed


def delete_replicator_item(item):
    for attr_path in (("node", "node"), ("node",)):
        try:
            node = item
            for attr in attr_path:
                node = getattr(node, attr)
            og.GraphController.delete_node(node.get_prim_path())
            return True
        except Exception:
            pass
    return False


def cleanup_replicator_state(stage, baseline_paths):
    removed = 0
    for parent_path, child_paths in baseline_paths.items():
        removed += remove_new_children(stage, parent_path, child_paths)
    return removed


def ensure_replicator_object_root(stage):
    if not stage.GetPrimAtPath("/Replicator").IsValid():
        stage.DefinePrim("/Replicator", "Scope")
    if not stage.GetPrimAtPath(REPLICATOR_OBJECT_ROOT).IsValid():
        stage.DefinePrim(REPLICATOR_OBJECT_ROOT, "Scope")
    return stage.GetPrimAtPath(REPLICATOR_OBJECT_ROOT)


def ensure_render_lights(stage):
    world_prim = stage.DefinePrim("/World", "Xform")
    UsdGeom.Xformable(world_prim)

    dome = UsdLux.DomeLight.Define(stage, RENDER_LIGHT_PATHS[0])
    dome.CreateIntensityAttr(1300.0)
    dome.CreateColorAttr(Gf.Vec3f(0.45, 0.48, 0.52))

    key = UsdLux.DistantLight.Define(stage, RENDER_LIGHT_PATHS[1])
    key.CreateIntensityAttr(4000.0)
    key.CreateAngleAttr(0.8)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
    UsdGeom.XformCommonAPI(key.GetPrim()).SetRotate((45.0, 0.0, -35.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def load_replicator_object(stage, usd_path, label):
    ensure_replicator_object_root(stage)
    object_path = omni.usd.get_stage_next_free_path(
        stage,
        f"{REPLICATOR_OBJECT_ROOT.pathString}/{safe_prim_name(label)}",
        False,
    )
    object_prim = stage.DefinePrim(object_path, "Xform")
    object_prim.CreateAttribute("replicatorXform", Sdf.ValueTypeNames.Bool).Set(True)
    set_prim_scale(object_prim, OBJECT_SCALE)

    ref_path = f"{object_path}/Ref"
    ref_prim = stage.DefinePrim(ref_path)
    ref_prim.GetReferences().AddReference(str(usd_path))

    object_node = rep.create.group([object_path])
    object_prim = stage.GetPrimAtPath(object_path)
    set_semantics_recursive(object_prim, label)
    return object_node, object_prim


def remove_replicator_object(stage, object_node, object_prim, baseline_paths):
    flush_render_state(update_frames=1)
    node_deleted = delete_replicator_item(object_node)
    prim_path = object_prim.GetPath() if object_prim and object_prim.IsValid() else None
    if prim_path and stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)
    removed_children = cleanup_replicator_state(stage, baseline_paths)
    flush_render_state(update_frames=2)
    return node_deleted, removed_children


def configure_raytraced_rendering():
    settings.set("/rtx/rendermode", "RayTraced")


def configure_pathtracing(total_spp):
    spp_per_frame = max(1, min(SPP_BATCH, total_spp))
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", spp_per_frame)
    settings.set("/rtx/pathtracing/totalSpp", max(1, total_spp))
    settings.set("/rtx/pathtracing/maxBounces", PATH_TRACING_MAX_BOUNCES)
    settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", PATH_TRACING_MAX_BOUNCES)
    settings.set("/rtx/ecoMode/maxFramesWithoutChange", 500)
    return int(np.ceil(max(1, total_spp) / spp_per_frame))


def set_prim_scale(prim, scale):
    UsdGeom.XformCommonAPI(prim).SetScale((scale, scale, scale))


def set_semantics_recursive(prim, label):
    stack = [prim]
    while stack:
        current = stack.pop()
        if current.IsValid():
            sem = Semantics.SemanticsAPI.Apply(current, "Semantics")
            sem.CreateSemanticTypeAttr().Set("class")
            sem.CreateSemanticDataAttr().Set(label)
            stack.extend(current.GetChildren())


def compute_world_bbox(prim_path):
    cache = bounds_utils.create_bbox_cache()
    corners = np.asarray(bounds_utils.compute_obb_corners(cache, prim_path), dtype=float)
    if corners.size == 0 or not np.all(np.isfinite(corners)):
        raise RuntimeError(f"could not compute bbox for {prim_path}")
    bbox_min = corners.min(axis=0)
    bbox_max = corners.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = bbox_max - bbox_min
    radius = max(float(np.linalg.norm(extent) / 2.0), 0.01)
    return center, extent, radius, corners


def minimum_camera_distance(center, direction, radius, corners):
    direction = direction / np.linalg.norm(direction)
    rel_corners = np.asarray(corners, dtype=float) - center
    front_support = float(np.max(rel_corners @ direction))
    clearance = max(radius * CAMERA_CLEARANCE_RATIO, MIN_CAMERA_CLEARANCE)
    return max(front_support + clearance, clearance)


def add_camera(name, center, direction, radius, image_size):
    direction = direction / np.linalg.norm(direction)
    distance = radius * 3.0
    position = center + direction * distance
    camera = rep.create.camera(
        name=name,
        position=position.tolist(),
        look_at=center.tolist(),
        focal_length=35.0,
        focus_distance=max(distance, 0.01),
        f_stop=0.0,
        horizontal_aperture=20.955,
        clipping_range=(0.0001, 100000.0),
    )
    render_product = rep.create.render_product(camera, image_size)
    return camera, render_product, distance


def set_camera_pose(camera, center, direction, distance, radius):
    position = center + direction * distance
    with camera:
        rep.modify.pose(position=position.tolist(), look_at=center.tolist())


def step_render(world, frame_count, capture=False):
    for _ in range(max(frame_count, 0)):
        world.step(render=False)
        if capture:
            rep.orchestrator.step()


def rgba_key_to_uint32(key):
    if isinstance(key, str):
        stripped = key.strip("()[] ")
        if "," in stripped:
            key = [int(part.strip()) for part in stripped.split(",")]
        else:
            return int(stripped)
    if isinstance(key, (list, tuple, np.ndarray)):
        if len(key) == 4:
            return int((int(key[3]) << 24) + (int(key[2]) << 16) + (int(key[1]) << 8) + int(key[0]))
        if len(key) == 1:
            return int(key[0])
    return int(key)


def label_matches(semantics, label):
    if isinstance(semantics, dict):
        return label in semantics.values() or semantics.get("class") == label
    if isinstance(semantics, (list, tuple)):
        return label in semantics
    return semantics == label


def mask_bbox(seg_data, id_to_semantics, label):
    seg = np.asarray(seg_data)
    if seg.ndim == 3 and seg.shape[-1] == 4:
        seg = (
            seg[..., 0].astype(np.uint32)
            + (seg[..., 1].astype(np.uint32) << 8)
            + (seg[..., 2].astype(np.uint32) << 16)
            + (seg[..., 3].astype(np.uint32) << 24)
        )
    else:
        seg = seg.astype(np.uint32)

    target_ids = []
    for key, semantics in id_to_semantics.items():
        if label_matches(semantics, label):
            target_ids.append(rgba_key_to_uint32(key))

    if target_ids:
        mask = np.isin(seg, np.asarray(target_ids, dtype=np.uint32))
    else:
        values, counts = np.unique(seg, return_counts=True)
        valid = values != 0
        if not np.any(valid):
            return None
        mask = seg == values[valid][np.argmax(counts[valid])]

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_margin_error(bbox, width, height, margin):
    xmin, ymin, xmax, ymax = bbox
    left = xmin / width
    right = (width - 1 - xmax) / width
    top = ymin / height
    bottom = (height - 1 - ymax) / height
    min_margin = min(left, right, top, bottom)
    fill = max((xmax - xmin + 1) / width, (ymax - ymin + 1) / height)
    target_fill = 1.0 - 2.0 * margin
    return min_margin, fill, target_fill


def fit_camera(camera, annotator, world, center, direction, radius, corners, image_size, label, margin, max_iter):
    width, height = image_size
    min_distance = minimum_camera_distance(center, direction, radius, corners)
    distance = max(radius * 3.0, min_distance)
    last_bbox = None
    configure_raytraced_rendering()

    for _ in range(max_iter):
        distance = max(distance, min_distance)
        set_camera_pose(camera, center, direction, distance, radius)
        step_render(world, READBACK_DELAY_FRAMES)
        for _ in range(READBACK_DELAY_FRAMES):
            rep.orchestrator.step()
        data = annotator.get_data()
        bbox = mask_bbox(data["data"], data.get("idToSemantics", {}), label)
        if bbox is None:
            distance = max(distance * 1.4, min_distance)
            continue

        last_bbox = bbox
        min_margin, fill, target_fill = bbox_margin_error(bbox, width, height, margin)
        if min_margin >= margin * 0.65 and abs(fill - target_fill) < 0.06:
            break
        if min_margin < margin * 0.65:
            distance = max(distance * 1.18, min_distance)
        else:
            distance = max(distance * max(0.82, min(0.98, fill / target_fill)), min_distance)

    distance = max(distance, min_distance)
    set_camera_pose(camera, center, direction, distance, radius)
    step_render(world, READBACK_DELAY_FRAMES)
    return distance, last_bbox, min_distance


def rgba_to_rgb(image):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[-1] == 4:
        return image[..., :3]
    return image


def render_rgb(world, rgb_annotator, spp):
    frame_count = configure_pathtracing(max(spp, 1))
    for _ in range(frame_count):
        world.step(render=False)
    for _ in range(READBACK_DELAY_FRAMES):
        rep.orchestrator.step()
        world.step(render=False)
    rep.orchestrator.step()
    return rgba_to_rgb(rgb_annotator.get_data())


def write_metadata(path, metadata):
    with path.open("w") as f:
        json.dump(metadata, f, indent=4)


def flush_render_state(update_frames=2):
    try:
        if hasattr(rep.orchestrator, "wait_until_complete"):
            rep.orchestrator.wait_until_complete()
    except Exception:
        pass
    app = omni.kit.app.get_app()
    for _ in range(update_frames):
        app.update()
    gc.collect()


def main():
    image_size = IMAGE_SIZE
    world = World(stage_units_in_meters=1.0, physics_dt=0.001, rendering_dt=0.005)
    stage = omni.usd.get_context().get_stage()
    world.reset()
    ensure_render_lights(stage)
    rep.orchestrator.pause()
    rep.orchestrator.set_capture_on_play(False)
    shared_camera, shared_render_product, _ = add_camera(
        "render_camera",
        np.zeros(3),
        CAMERA_VIEWS["front"],
        1.0,
        image_size,
    )
    inst_ann = rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast")
    rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    inst_ann.attach([shared_render_product])
    rgb_ann.attach([shared_render_product])
    ensure_replicator_object_root(stage)
    baseline_paths = {
        "/Replicator/SDGPipeline": child_path_snapshot(stage, "/Replicator/SDGPipeline"),
        REPLICATOR_OBJECT_ROOT.pathString: child_path_snapshot(stage, REPLICATOR_OBJECT_ROOT.pathString),
    }

    object_folders = find_object_folders(ROOT_DIR)
    print(f"found object folders: {len(object_folders)}")

    for object_dir in object_folders:
        usd_path = find_usd_file(object_dir)
        if usd_path is None:
            print(f"[skip] no edited usd: {object_dir}")
            continue

        output_dir = object_dir / "rendering"
        existing_renders = {path.name for path in output_dir.glob("*.png")} if output_dir.exists() else set()
        if output_dir.exists() and not OVERWRITE and EXPECTED_RENDER_FILES.issubset(existing_renders):
            print(f"[skip] rendering exists: {output_dir}")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)

        label = object_dir.name
        object_node = None
        object_prim = None
        try:
            object_node, object_prim = load_replicator_object(stage, usd_path, label)
            prim_path = object_prim.GetPath()

            world.reset()
            ensure_render_lights(stage)
            step_render(world, SCENE_WARMUP_FRAMES)

            center, extent, radius, corners = compute_world_bbox(prim_path)
            print(f"[render] {label} | usd={usd_path} | extent={np.round(extent, 4).tolist()}")

            metadata = {
                "object": label,
                "usd_path": str(usd_path),
                "object_scale": OBJECT_SCALE,
                "prim_path": prim_path.pathString,
                "bbox_center": center.tolist(),
                "bbox_extent": extent.tolist(),
                "views": {},
            }

            for view_name, direction in CAMERA_VIEWS.items():
                direction = direction / np.linalg.norm(direction)
                distance, seg_bbox, min_distance = fit_camera(
                    shared_camera,
                    inst_ann,
                    world,
                    center,
                    direction,
                    radius,
                    corners,
                    image_size,
                    label,
                    MARGIN,
                    MAX_FIT_ITER,
                )
                image = render_rgb(world, rgb_ann, SPP)
                image_path = output_dir / f"{view_name}.png"
                Image.fromarray(image).save(image_path)

                metadata["views"][view_name] = {
                    "camera_position": (center + direction * distance).tolist(),
                    "look_at": center.tolist(),
                    "distance": distance,
                    "minimum_camera_distance": min_distance,
                    "segmentation_bbox_xyxy": list(seg_bbox) if seg_bbox is not None else None,
                }
                print(f"  saved {image_path.name} bbox={seg_bbox}")

            write_metadata(output_dir / "rendering_meta.json", metadata)
        finally:
            if object_node is not None or object_prim is not None:
                node_deleted, removed_children = remove_replicator_object(
                    stage,
                    object_node,
                    object_prim,
                    baseline_paths,
                )
                print(f"  cleanup node_deleted={node_deleted} sdg_removed={removed_children}")

    simulation_app.close()


if __name__ == "__main__":
    main()
