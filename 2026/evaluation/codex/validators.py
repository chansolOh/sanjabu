"""Schema validators for Dataset 2026 JSON and binary payloads."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import config
from evaluation_core import parse_rgba_key


OBJECT_ID_RE = re.compile(config.OBJECT_ID_PATTERN)
NUMBER = (int, float)


def issue(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def is_number(value: Any) -> bool:
    return (
        isinstance(value, NUMBER)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(
        is_number(item) for item in value
    )


def validate_matrix(value: Any, rows: int, columns: int) -> bool:
    return isinstance(value, list) and len(value) == rows and all(
        validate_vector(row, columns) for row in value
    )


def require_keys(data: Any, keys: set[str], path: str) -> list[dict[str, str]]:
    if not isinstance(data, dict):
        return [issue("type", path, "must be a JSON object")]
    return [
        issue("missing_key", f"{path}.{key}", "required key is missing")
        for key in sorted(keys - set(data))
    ]


def valid_object_id(value: Any, allowed_objects: set[str]) -> bool:
    return (
        isinstance(value, str)
        and OBJECT_ID_RE.fullmatch(value) is not None
        and value in allowed_objects
    )


def validate_transform_record(
    data: Any, path: str, allowed_objects: set[str], class_key: str = "class"
) -> list[dict[str, str]]:
    errors = require_keys(
        data, {class_key, "usd_path", "translate", "orient", "scale"}, path
    )
    if not isinstance(data, dict):
        return errors
    if class_key in data and not valid_object_id(data[class_key], allowed_objects):
        errors.append(issue("object_id", f"{path}.{class_key}", "invalid object ID"))
    if "usd_path" in data and not (
        isinstance(data["usd_path"], str) and data["usd_path"].endswith(".usd")
    ):
        errors.append(issue("usd_path", f"{path}.usd_path", "must end in .usd"))
    for key, length in (("translate", 3), ("orient", 4), ("scale", 3)):
        if key in data and not validate_vector(data[key], length):
            errors.append(
                issue("shape", f"{path}.{key}", f"must contain {length} finite numbers")
            )
    return errors


def validate_conf(data: Any, allowed_objects: set[str]) -> list[dict[str, str]]:
    errors = require_keys(
        data, {"envs", "objects", "platform", "cameras", "lights", "physics_scene"}, "root"
    )
    if not isinstance(data, dict):
        return errors

    envs = data.get("envs")
    errors.extend(
        require_keys(
            envs,
            {
                "env_name",
                "section_name",
                "platform_name",
                "usd_path",
                "position",
                "orientation",
                "scale",
            },
            "envs",
        )
    )
    if isinstance(envs, dict):
        for key in ("env_name", "section_name", "platform_name", "usd_path"):
            if key in envs and not isinstance(envs[key], str):
                errors.append(issue("type", f"envs.{key}", "must be a string"))
        for key in ("position", "orientation", "scale"):
            if key in envs and not validate_vector(envs[key], 3):
                errors.append(issue("shape", f"envs.{key}", "must contain 3 finite numbers"))

    objects = data.get("objects")
    if not isinstance(objects, list):
        errors.append(issue("type", "objects", "must be an array"))
    else:
        if len(objects) != config.EXPECTED_OBJECTS_PER_SCENE:
            errors.append(
                issue(
                    "object_count",
                    "objects",
                    "expected "
                    f"{config.EXPECTED_OBJECTS_PER_SCENE} objects, got {len(objects)}",
                )
            )
        seen: set[str] = set()
        for index, item in enumerate(objects):
            errors.extend(
                validate_transform_record(item, f"objects[{index}]", allowed_objects)
            )
            if isinstance(item, dict) and isinstance(item.get("class"), str):
                if item["class"] in seen:
                    errors.append(
                        issue("duplicate", f"objects[{index}].class", "duplicate object ID")
                    )
                seen.add(item["class"])

    platform = data.get("platform")
    errors.extend(
        require_keys(platform, {"name", "usd_path", "translate", "orient", "scale"}, "platform")
    )
    if isinstance(platform, dict):
        for key in ("name", "usd_path"):
            if key in platform and not isinstance(platform[key], str):
                errors.append(issue("type", f"platform.{key}", "must be a string"))
        for key, length in (("translate", 3), ("orient", 4), ("scale", 3)):
            if key in platform and not validate_vector(platform[key], length):
                errors.append(
                    issue("shape", f"platform.{key}", f"must contain {length} finite numbers")
                )

    cameras = data.get("cameras")
    if not isinstance(cameras, list):
        errors.append(issue("type", "cameras", "must be an array"))
    else:
        camera_names: list[str] = []
        camera_keys = {
            "name",
            "cam_model_conf_path",
            "pixel_size",
            "output_size",
            "clipping_range",
            "focus_distance",
            "f_stop",
            "cam_poses",
            "focal_length_isaac",
            "horizontal_aperture",
            "intrinsic_isaac",
        }
        for index, camera in enumerate(cameras):
            camera_path = f"cameras[{index}]"
            errors.extend(require_keys(camera, camera_keys, camera_path))
            if not isinstance(camera, dict):
                continue
            if isinstance(camera.get("name"), str):
                camera_names.append(camera["name"])
            else:
                errors.append(issue("type", f"{camera_path}.name", "must be a string"))
            if "output_size" in camera and not (
                isinstance(camera["output_size"], list)
                and len(camera["output_size"]) == 2
                and all(is_int(number) and number > 0 for number in camera["output_size"])
            ):
                errors.append(issue("shape", f"{camera_path}.output_size", "must be 2 positive integers"))
            if "cam_poses" in camera and not validate_matrix(camera["cam_poses"], 4, 4):
                errors.append(issue("shape", f"{camera_path}.cam_poses", "must be a finite 4x4 matrix"))
            if "intrinsic_isaac" in camera and not validate_matrix(camera["intrinsic_isaac"], 3, 3):
                errors.append(issue("shape", f"{camera_path}.intrinsic_isaac", "must be a finite 3x3 matrix"))
        if set(camera_names) != set(config.CAMERAS):
            errors.append(
                issue(
                    "camera_names",
                    "cameras",
                    f"expected {sorted(config.CAMERAS)}, got {sorted(camera_names)}",
                )
            )

    if not isinstance(data.get("lights"), list):
        errors.append(issue("type", "lights", "must be an array"))
    if not isinstance(data.get("physics_scene"), dict):
        errors.append(issue("type", "physics_scene", "must be an object"))
    return errors


def validate_inst_seg_mapping(data: Any, allowed_objects: set[str]) -> list[dict[str, str]]:
    if not isinstance(data, dict):
        return [issue("type", "root", "must be an RGBA-to-label object")]
    errors: list[dict[str, str]] = []
    object_classes: list[str] = []
    background_found = False
    for rgba_key, value in data.items():
        if parse_rgba_key(rgba_key) is None:
            errors.append(issue("rgba_key", rgba_key, "must be a 4-byte tuple string"))
        if not isinstance(value, dict) or not isinstance(value.get("class"), str):
            errors.append(issue("class", rgba_key, "value must contain string class"))
            continue
        class_name = value["class"]
        if class_name in {"BACKGROUND", "UNLABELLED"}:
            background_found = True
        elif not valid_object_id(class_name, allowed_objects):
            errors.append(issue("object_id", f"{rgba_key}.class", "invalid object ID"))
        else:
            object_classes.append(class_name)
    if len(object_classes) != len(set(object_classes)):
        errors.append(issue("duplicate", "root", "an object class is assigned to multiple colors"))
    if not background_found:
        errors.append(issue("background", "root", "BACKGROUND or UNLABELLED entry is required"))
    return errors


def validate_bbox(data: Any, allowed_objects: set[str]) -> list[dict[str, str]]:
    if not isinstance(data, dict):
        return [issue("type", "root", "must be an object-to-bbox object")]
    width, height = config.EXPECTED_IMAGE_SIZE
    errors: list[dict[str, str]] = []
    for object_id, bbox in data.items():
        if not valid_object_id(object_id, allowed_objects):
            errors.append(issue("object_id", object_id, "invalid object ID"))
        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(is_int(value) for value in bbox)
        ):
            errors.append(issue("bbox_shape", object_id, "must be [x1,y1,x2,y2] integers"))
            continue
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
            errors.append(issue("bbox_range", object_id, "coordinates are unordered or out of image bounds"))
    return errors


def validate_scene_meta(data: Any, allowed_objects: set[str]) -> list[dict[str, str]]:
    errors = require_keys(data, {"objects", "Description"}, "root")
    if not isinstance(data, dict):
        return errors
    if "Description" in data and not isinstance(data["Description"], str):
        errors.append(issue("type", "Description", "must be a string"))
    objects = data.get("objects")
    if not isinstance(objects, dict):
        errors.append(issue("type", "objects", "must be an object"))
        return errors
    fields = {
        "level_1",
        "level_2",
        "level_3",
        "object_name",
        "color",
        "packaging",
        "features",
        "description",
    }
    for object_id, attributes in objects.items():
        if not valid_object_id(object_id, allowed_objects):
            errors.append(issue("object_id", f"objects.{object_id}", "invalid object ID"))
        errors.extend(require_keys(attributes, fields, f"objects.{object_id}"))
        if not isinstance(attributes, dict):
            continue
        for field in fields:
            if field in attributes and not isinstance(attributes[field], str):
                errors.append(issue("type", f"objects.{object_id}.{field}", "must be a string"))
        for field in fields - {"packaging"}:
            if field in attributes and isinstance(attributes[field], str) and not attributes[field].strip():
                errors.append(issue("empty", f"objects.{object_id}.{field}", "must not be empty"))
    return errors


def validate_grasp_item(
    item: Any,
    path: str,
    allowed_objects: set[str],
    allowed_grippers: set[str],
    output: bool,
) -> list[dict[str, str]]:
    required = {
        "target_points",
        "target_orientation",
        "target_width",
        "target_object",
        "gripper_model",
        "gripper_type",
    }
    if output:
        required |= {
            "bbox_2d",
            "target_base_tf",
            "target_joint_pos",
            "joint_unit",
            "disturbed_object_count",
        }
    errors = require_keys(item, required, path)
    if not isinstance(item, dict):
        return errors
    if "target_points" in item and not validate_vector(item["target_points"], 3):
        errors.append(issue("shape", f"{path}.target_points", "must contain 3 finite numbers"))
    if "target_orientation" in item and not (
        validate_vector(item["target_orientation"], 3)
        and all(-360 <= float(number) <= 360 for number in item["target_orientation"])
    ):
        # The pre-grasp generator intentionally emits 360 degrees as the
        # endpoint of its angular sweep; it is equivalent to 0 degrees.
        errors.append(issue("range", f"{path}.target_orientation", "must be 3 angles in [-360,360]"))
    if "target_width" in item and not (
        is_number(item["target_width"]) and 0 <= float(item["target_width"]) <= 1
    ):
        errors.append(issue("range", f"{path}.target_width", "must be in [0,1]"))
    if "target_object" in item and not valid_object_id(item["target_object"], allowed_objects):
        errors.append(issue("object_id", f"{path}.target_object", "invalid object ID"))
    if "gripper_model" in item and not (
        isinstance(item["gripper_model"], str)
        and item["gripper_model"]
        and (not allowed_grippers or item["gripper_model"] in allowed_grippers)
    ):
        errors.append(issue("gripper_model", f"{path}.gripper_model", "unknown gripper model"))
    if "gripper_type" in item and not (
        isinstance(item["gripper_type"], str) and item["gripper_type"]
    ):
        errors.append(issue("type", f"{path}.gripper_type", "must be a non-empty string"))
    if not output:
        return errors

    bbox_2d = item.get("bbox_2d")
    errors.extend(require_keys(bbox_2d, {"bbox", "center", "width", "height", "angle"}, f"{path}.bbox_2d"))
    if isinstance(bbox_2d, dict):
        polygons = bbox_2d.get("bbox")
        valid_polygons = isinstance(polygons, list) and bool(polygons)
        if valid_polygons:
            width, height = config.EXPECTED_IMAGE_SIZE
            for polygon in polygons:
                if not (
                    isinstance(polygon, list)
                    and len(polygon) == 4
                    and all(validate_vector(point, 2) for point in polygon)
                    and all(0 <= point[0] < width and 0 <= point[1] < height for point in polygon)
                ):
                    valid_polygons = False
                    break
        if not valid_polygons:
            errors.append(issue("bbox_polygon", f"{path}.bbox_2d.bbox", "must contain in-bounds 4-point polygons"))
        if "center" in bbox_2d and not validate_vector(bbox_2d["center"], 2):
            errors.append(issue("shape", f"{path}.bbox_2d.center", "must contain 2 finite numbers"))
        for field in ("width", "height"):
            if field in bbox_2d and not (is_number(bbox_2d[field]) and float(bbox_2d[field]) > 0):
                errors.append(issue("range", f"{path}.bbox_2d.{field}", "must be positive"))
        if "angle" in bbox_2d and not (
            is_number(bbox_2d["angle"])
            and -2 * math.pi - 1e-5
            <= float(bbox_2d["angle"])
            <= 2 * math.pi + 1e-5
        ):
            # Legacy and current grasp generators preserve the complete
            # 0..360-degree sweep, so equivalent angles can reach +/-2*pi.
            errors.append(issue("range", f"{path}.bbox_2d.angle", "must be radians in [-2*pi,2*pi]"))

    target_base_tf = item.get("target_base_tf")
    errors.extend(require_keys(target_base_tf, {"start", "end"}, f"{path}.target_base_tf"))
    if isinstance(target_base_tf, dict):
        for phase in ("start", "end"):
            pose = target_base_tf.get(phase)
            errors.extend(
                require_keys(
                    pose,
                    {"frame", "position", "orientation_wxyz", "rpy_deg"},
                    f"{path}.target_base_tf.{phase}",
                )
            )
            if isinstance(pose, dict):
                for field, length in (("position", 3), ("orientation_wxyz", 4), ("rpy_deg", 3)):
                    if field in pose and not validate_vector(pose[field], length):
                        errors.append(issue("shape", f"{path}.target_base_tf.{phase}.{field}", f"must contain {length} finite numbers"))

    target_joint_pos = item.get("target_joint_pos")
    errors.extend(require_keys(target_joint_pos, {"start", "end"}, f"{path}.target_joint_pos"))
    if isinstance(target_joint_pos, dict):
        for phase in ("start", "end"):
            joints = target_joint_pos.get(phase)
            if not isinstance(joints, dict) or not joints or not all(
                isinstance(name, str) and name and is_number(value)
                for name, value in joints.items()
            ):
                errors.append(issue("joint_positions", f"{path}.target_joint_pos.{phase}", "must be a non-empty joint-to-number object"))
    if "joint_unit" in item and not (
        isinstance(item["joint_unit"], str) and item["joint_unit"]
    ):
        errors.append(issue("type", f"{path}.joint_unit", "must be a non-empty string"))
    if "disturbed_object_count" in item and not (
        is_int(item["disturbed_object_count"]) and item["disturbed_object_count"] >= 0
    ):
        errors.append(issue("range", f"{path}.disturbed_object_count", "must be an integer >= 0"))
    return errors


def validate_pre_grasp(data: Any, allowed_objects: set[str], allowed_grippers: set[str]) -> list[dict[str, str]]:
    errors = require_keys(data, {"gripper_model", "data"}, "root")
    if not isinstance(data, dict):
        return errors
    model = data.get("gripper_model")
    if not (
        isinstance(model, str)
        and model
        and (not allowed_grippers or model in allowed_grippers)
    ):
        errors.append(issue("gripper_model", "gripper_model", "unknown gripper model"))
    items = data.get("data")
    if not isinstance(items, list):
        errors.append(issue("type", "data", "must be an array"))
    else:
        for index, item in enumerate(items):
            errors.extend(
                validate_grasp_item(
                    item,
                    f"data[{index}]",
                    allowed_objects,
                    allowed_grippers,
                    output=False,
                )
            )
            if isinstance(item, dict) and model and item.get("gripper_model") != model:
                errors.append(issue("inconsistent", f"data[{index}].gripper_model", "does not match root gripper_model"))
    return errors


def validate_output_grasp(data: Any, allowed_objects: set[str], allowed_grippers: set[str]) -> list[dict[str, str]]:
    if not isinstance(data, list):
        return [issue("type", "root", "must be an array")]
    errors: list[dict[str, str]] = []
    for index, item in enumerate(data):
        errors.extend(
            validate_grasp_item(
                item,
                f"[{index}]",
                allowed_objects,
                allowed_grippers,
                output=True,
            )
        )
    return errors


def validate_pointcloud_mapping(data: Any, allowed_objects: set[str]) -> list[dict[str, str]]:
    errors = require_keys(
        data,
        {"scene_id", "camera", "instance_source", "inst_seg_source", "semantics_source", "verification", "instances"},
        "root",
    )
    if not isinstance(data, dict):
        return errors
    if not isinstance(data.get("scene_id"), str) or not data.get("scene_id", "").isdigit():
        errors.append(issue("scene_id", "scene_id", "must be a numeric string"))
    if data.get("camera") not in config.CAMERAS:
        errors.append(issue("camera", "camera", "unknown camera"))
    verification = data.get("verification")
    errors.extend(require_keys(verification, {"point_count", "pixel_count", "all_instance_counts_match"}, "verification"))
    if isinstance(verification, dict):
        for field in ("point_count", "pixel_count"):
            if field in verification and not (is_int(verification[field]) and verification[field] >= 0):
                errors.append(issue("type", f"verification.{field}", "must be an integer >= 0"))
        if "all_instance_counts_match" in verification and not isinstance(verification["all_instance_counts_match"], bool):
            errors.append(issue("type", "verification.all_instance_counts_match", "must be boolean"))
    instances = data.get("instances")
    if not isinstance(instances, dict):
        errors.append(issue("type", "instances", "must be an object"))
    else:
        for instance_id, value in instances.items():
            instance_path = f"instances.{instance_id}"
            if not str(instance_id).isdigit():
                errors.append(issue("instance_id", instance_path, "key must be numeric"))
            errors.extend(require_keys(value, {"class", "rgba", "point_count", "pixel_count"}, instance_path))
            if not isinstance(value, dict):
                continue
            class_name = value.get("class")
            if class_name not in {"BACKGROUND", "UNLABELLED"} and not valid_object_id(class_name, allowed_objects):
                errors.append(issue("object_id", f"{instance_path}.class", "invalid object ID"))
            rgba = value.get("rgba")
            if not (
                isinstance(rgba, list)
                and len(rgba) == 4
                and all(is_int(number) and 0 <= number <= 255 for number in rgba)
            ):
                errors.append(issue("rgba", f"{instance_path}.rgba", "must contain four bytes"))
    return errors


def validate_image(path: Path, expected_mode: set[str] | None = None) -> list[dict[str, str]]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            size = image.size
            mode = image.mode
    except Exception as error:
        return [issue("image_decode", str(path), str(error))]
    errors: list[dict[str, str]] = []
    if size != config.EXPECTED_IMAGE_SIZE:
        errors.append(issue("image_size", str(path), f"expected {config.EXPECTED_IMAGE_SIZE}, got {size}"))
    if expected_mode and mode not in expected_mode:
        errors.append(issue("image_mode", str(path), f"expected {sorted(expected_mode)}, got {mode}"))
    return errors


def validate_npy(path: Path, expected_shape: tuple[int, ...], expected_kinds: set[str]) -> list[dict[str, str]]:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        return [issue("npy_decode", str(path), str(error))]
    errors: list[dict[str, str]] = []
    if array.shape != expected_shape:
        errors.append(issue("array_shape", str(path), f"expected {expected_shape}, got {array.shape}"))
    if array.dtype.kind not in expected_kinds:
        errors.append(issue("array_dtype", str(path), f"unexpected dtype {array.dtype}"))
    return errors
