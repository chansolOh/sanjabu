"""Validate JSON schemas and sampled binary payloads for all three splits."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import config
from evaluation_core import (
    SceneRef,
    atomic_write_json,
    load_allowed_grippers,
    load_object_catalog,
    run_metadata,
    scene_refs_for_split,
    stratified_sample,
    strict_json_load,
    write_csv,
)
from validators import (
    validate_bbox,
    validate_conf,
    validate_image,
    validate_inst_seg_mapping,
    validate_npy,
    validate_output_grasp,
    validate_pointcloud_mapping,
    validate_pre_grasp,
    validate_scene_meta,
)


Validator = Callable[[Any], list[dict[str, str]]]


def scene_json_checks(
    scene: SceneRef,
    allowed_objects: set[str],
    allowed_grippers: set[str],
) -> list[tuple[str, Path, Validator]]:
    platform = scene.platform_path
    name = scene.scene_name
    checks: list[tuple[str, Path, Validator]] = [
        (
            "conf",
            platform / "conf" / f"{name}.json",
            lambda data: validate_conf(data, allowed_objects),
        ),
        (
            "scene_meta",
            platform / "scene_meta" / f"{name}.json",
            lambda data: validate_scene_meta(data, allowed_objects),
        ),
    ]
    settings = config.DATASETS[scene.split]
    if settings["require_pre_grasp"]:
        checks.append(
            (
                "pre_grasp",
                platform / "pre_grasp" / f"{name}.json",
                lambda data: validate_pre_grasp(
                    data, allowed_objects, allowed_grippers
                ),
            )
        )
    if settings["require_output_grasp"]:
        checks.append(
            (
                "output_grasp",
                platform / "output_grasp" / f"{name}.json",
                lambda data: validate_output_grasp(
                    data, allowed_objects, allowed_grippers
                ),
            )
        )
    for camera in config.CAMERAS:
        checks.extend(
            [
                (
                    f"bbox/{camera}",
                    platform / "bbox" / camera / f"{name}.json",
                    lambda data: validate_bbox(data, allowed_objects),
                ),
                (
                    f"inst_seg_mapping/{camera}",
                    platform
                    / "inst_seg"
                    / camera
                    / f"semantics_mapping_{name}.json",
                    lambda data: validate_inst_seg_mapping(data, allowed_objects),
                ),
                (
                    f"pointcloud_mapping/{camera}",
                    platform
                    / "pointcloud"
                    / camera
                    / f"pointcloud_inst_seg_mapping_{name}.json",
                    lambda data: validate_pointcloud_mapping(
                        data, allowed_objects
                    ),
                ),
            ]
        )
    return checks


def scene_binary_checks(scene: SceneRef) -> list[tuple[str, Path, Callable[[Path], list[dict[str, str]]]]]:
    platform = scene.platform_path
    name = scene.scene_name
    width, height = config.EXPECTED_IMAGE_SIZE
    point_count = width * height
    checks: list[tuple[str, Path, Callable[[Path], list[dict[str, str]]]]] = []
    for camera in config.CAMERAS:
        checks.extend(
            [
                (
                    f"rgb/{camera}",
                    platform / "rgb" / camera / f"{name}.png",
                    lambda path: validate_image(path, {"RGB", "RGBA"}),
                ),
                (
                    f"normals/{camera}",
                    platform / "normals" / camera / f"{name}.png",
                    lambda path: validate_image(path, {"RGB", "RGBA"}),
                ),
                (
                    f"inst_seg_image/{camera}",
                    platform / "inst_seg" / camera / f"{name}.png",
                    lambda path: validate_image(path, {"RGBA"}),
                ),
                (
                    f"depth/{camera}",
                    platform / "depth" / camera / f"{name}.npy",
                    lambda path, shape=(height, width): validate_npy(
                        path, shape, {"f"}
                    ),
                ),
                (
                    f"pointcloud_xyz/{camera}",
                    platform
                    / "pointcloud"
                    / camera
                    / f"pointcloud_{name}.npy",
                    lambda path, shape=(point_count, 3): validate_npy(
                        path, shape, {"f"}
                    ),
                ),
                (
                    f"pointcloud_rgb/{camera}",
                    platform
                    / "pointcloud"
                    / camera
                    / f"pointcloud_rgb_{name}.npy",
                    lambda path, shape=(point_count, 4): validate_npy(
                        path, shape, {"u", "i"}
                    ),
                ),
                (
                    f"pointcloud_inst_seg/{camera}",
                    platform
                    / "pointcloud"
                    / camera
                    / f"pointcloud_inst_seg_{name}.npy",
                    lambda path, shape=(point_count,): validate_npy(
                        path, shape, {"u", "i"}
                    ),
                ),
            ]
        )
    return checks


def new_stats() -> dict[str, int]:
    return {"checked": 0, "valid": 0, "invalid": 0, "missing": 0, "issue_count": 0}


def validate_syntax() -> dict[str, Any]:
    catalog = load_object_catalog()
    allowed_objects = set(catalog)
    allowed_grippers = load_allowed_grippers()
    result: dict[str, Any] = {
        "metadata": run_metadata("validate_syntax.py"),
        "scope": {
            "json_max_scenes_per_split": config.SYNTAX_MAX_SCENES_PER_SPLIT,
            "binary_payload_samples_per_split": config.BINARY_PAYLOAD_SAMPLES_PER_SPLIT,
            "allowed_object_count": len(allowed_objects),
            "allowed_object_min": min(allowed_objects),
            "allowed_object_max": max(allowed_objects),
            "allowed_grippers": sorted(allowed_grippers),
        },
        "splits": {},
        "issue_examples": [],
    }
    issue_code_counts: Counter[str] = Counter()
    summary_rows: list[dict[str, Any]] = []

    def record_issues(
        scene: SceneRef,
        unit: str,
        path: Path,
        errors: list[dict[str, str]],
    ) -> None:
        for error in errors:
            issue_code_counts[error["code"]] += 1
            if len(result["issue_examples"]) < config.MAX_ISSUE_EXAMPLES:
                result["issue_examples"].append(
                    {
                        "split": scene.split,
                        "scene": scene.relative_scene,
                        "unit": unit,
                        "file": str(path),
                        **error,
                    }
                )

    for split_number, split in enumerate(config.DATASETS):
        all_scenes = scene_refs_for_split(split)
        scenes = stratified_sample(
            all_scenes,
            config.SYNTAX_MAX_SCENES_PER_SPLIT,
            lambda ref: (ref.env, ref.section, ref.platform),
            config.RANDOM_SEED + split_number,
        )
        binary_scenes = stratified_sample(
            scenes,
            config.BINARY_PAYLOAD_SAMPLES_PER_SPLIT,
            lambda ref: (ref.env, ref.section, ref.platform),
            config.RANDOM_SEED + 100 + split_number,
        )
        unit_stats: dict[str, dict[str, int]] = defaultdict(new_stats)
        record_totals = {"pre_grasp": 0, "output_grasp": 0}
        invalid_record_indices = {"pre_grasp": set(), "output_grasp": set()}

        for scene_number, scene in enumerate(scenes, start=1):
            if scene_number == 1 or scene_number % 250 == 0 or scene_number == len(scenes):
                print(f"[{split}] JSON {scene_number:,}/{len(scenes):,} {scene.relative_scene}")
            for unit, path, validator in scene_json_checks(
                scene, allowed_objects, allowed_grippers
            ):
                stats = unit_stats[unit]
                stats["checked"] += 1
                if not path.is_file():
                    stats["missing"] += 1
                    stats["invalid"] += 1
                    stats["issue_count"] += 1
                    record_issues(
                        scene,
                        unit,
                        path,
                        [{"code": "missing_file", "path": "file", "message": "required file is missing"}],
                    )
                    continue
                try:
                    data = strict_json_load(path)
                except Exception as error:
                    errors = [{"code": "json_parse", "path": "root", "message": str(error)}]
                    data = None
                else:
                    errors = validator(data)

                if unit in record_totals and data is not None:
                    items = data.get("data", []) if unit == "pre_grasp" and isinstance(data, dict) else data
                    if isinstance(items, list):
                        record_totals[unit] += len(items)
                        for error in errors:
                            error_path = error["path"]
                            if error_path.startswith("data["):
                                index_text = error_path[5:].split("]", 1)[0]
                            elif error_path.startswith("["):
                                index_text = error_path[1:].split("]", 1)[0]
                            else:
                                continue
                            if index_text.isdigit():
                                invalid_record_indices[unit].add(
                                    f"{scene.uid}|{index_text}"
                                )

                if errors:
                    stats["invalid"] += 1
                    stats["issue_count"] += len(errors)
                    record_issues(scene, unit, path, errors)
                else:
                    stats["valid"] += 1

        binary_scene_uids = {scene.uid for scene in binary_scenes}
        for scene_number, scene in enumerate(binary_scenes, start=1):
            if scene_number == 1 or scene_number % 50 == 0 or scene_number == len(binary_scenes):
                print(f"[{split}] binary {scene_number:,}/{len(binary_scenes):,} {scene.relative_scene}")
            for unit, path, validator in scene_binary_checks(scene):
                stats = unit_stats[unit]
                stats["checked"] += 1
                if not path.is_file():
                    errors = [{"code": "missing_file", "path": "file", "message": "sampled binary file is missing"}]
                    stats["missing"] += 1
                else:
                    errors = validator(path)
                if errors:
                    stats["invalid"] += 1
                    stats["issue_count"] += len(errors)
                    record_issues(scene, unit, path, errors)
                else:
                    stats["valid"] += 1

        serialized_stats: dict[str, Any] = {}
        total_checked = total_valid = total_invalid = 0
        for unit, stats in sorted(unit_stats.items()):
            checked = stats["checked"]
            stats["accuracy"] = stats["valid"] / checked if checked else 0.0
            serialized_stats[unit] = dict(stats)
            total_checked += checked
            total_valid += stats["valid"]
            total_invalid += stats["invalid"]
            summary_rows.append({"split": split, "unit": unit, **stats})

        record_summary = {}
        for unit in record_totals:
            total = record_totals[unit]
            invalid = len(invalid_record_indices[unit])
            record_summary[unit] = {
                "record_count": total,
                "invalid_record_count": invalid,
                "record_accuracy": (total - invalid) / total if total else None,
            }
        result["splits"][split] = {
            "available_conf_scenes": len(all_scenes),
            "json_checked_scenes": len(scenes),
            "binary_checked_scenes": len(binary_scene_uids),
            "file_accuracy": total_valid / total_checked if total_checked else 0.0,
            "checked_file_units": total_checked,
            "valid_file_units": total_valid,
            "invalid_file_units": total_invalid,
            "units": serialized_stats,
            "records": record_summary,
        }

    result["issue_code_counts"] = dict(sorted(issue_code_counts.items()))
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config.RESULT_DIR / "syntax_validation.json", result)
    write_csv(config.RESULT_DIR / "syntax_validation_summary.csv", summary_rows)
    return result


def main() -> None:
    result = validate_syntax()
    print("\n=== SYNTAX ACCURACY ===")
    for split, data in result["splits"].items():
        print(
            f"{split:10s} {data['valid_file_units']:,}/"
            f"{data['checked_file_units']:,} "
            f"({data['file_accuracy'] * 100:.6f}%)"
        )
    print(f"saved: {config.RESULT_DIR / 'syntax_validation.json'}")


if __name__ == "__main__":
    main()
