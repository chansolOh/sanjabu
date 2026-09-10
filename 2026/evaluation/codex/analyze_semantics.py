"""Automatic cross-modal semantic consistency analysis.

Instance segmentation also requires human review. This script measures the
parts that can be established objectively: class-set consistency, color-map
coverage, exact mask-derived boxes, pointcloud mapping, grasp targets, and CSV
metadata agreement.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import config
from evaluation_core import (
    SceneRef,
    atomic_write_json,
    load_object_catalog,
    normalized_text,
    parse_rgba_key,
    run_metadata,
    scene_refs_for_split,
    stratified_sample,
    strict_json_load,
    write_csv,
)


META_FIELD_MAP = {
    "level_1": "Level_1",
    "level_2": "Level_2",
    "level_3": "Level_3",
    "object_name": "Object_name",
    "color": "Color",
    "packaging": "Packaging",
    "features": "Features",
    "description": "Description",
}
SPECIAL_CLASSES = {"BACKGROUND", "UNLABELLED"}


def class_set_from_conf(conf: Any) -> set[str]:
    if not isinstance(conf, dict) or not isinstance(conf.get("objects"), list):
        return set()
    return {
        item.get("class")
        for item in conf["objects"]
        if isinstance(item, dict) and isinstance(item.get("class"), str)
    }


def pack_rgba(rgba: tuple[int, int, int, int]) -> int:
    r, g, b, a = rgba
    return r | (g << 8) | (b << 16) | (a << 24)


def image_color_counts(path: Path) -> tuple[dict[int, int], np.ndarray]:
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    packed = (
        rgba[..., 0].astype(np.uint32)
        | (rgba[..., 1].astype(np.uint32) << 8)
        | (rgba[..., 2].astype(np.uint32) << 16)
        | (rgba[..., 3].astype(np.uint32) << 24)
    )
    colors, counts = np.unique(packed, return_counts=True)
    return {int(color): int(count) for color, count in zip(colors, counts)}, packed


def mapping_by_class(mapping: Any) -> dict[str, tuple[int, int, int, int]]:
    if not isinstance(mapping, dict):
        return {}
    result: dict[str, tuple[int, int, int, int]] = {}
    for rgba_key, value in mapping.items():
        rgba = parse_rgba_key(rgba_key)
        if rgba is None or not isinstance(value, dict):
            continue
        class_name = value.get("class")
        if isinstance(class_name, str):
            result[class_name] = rgba
    return result


def new_metric(unit: str) -> dict[str, Any]:
    return {"unit": unit, "checked": 0, "passed": 0, "failed": 0, "accuracy": None}


def analyze_semantics() -> dict[str, Any]:
    catalog = load_object_catalog()
    result: dict[str, Any] = {
        "metadata": run_metadata("analyze_semantics.py"),
        "scope": {
            "samples_per_split": config.SEMANTIC_SAMPLES_PER_SPLIT,
            "cameras": list(config.CAMERAS),
            "bbox_pixel_tolerance": config.BBOX_PIXEL_TOLERANCE,
            "manual_inst_seg_required": True,
        },
        "metric_definitions": {
            "conf_object_catalog": "conf object ID and USD filename agree with the 2026 CSV",
            "scene_meta_object_set": "scene_meta object keys equal conf object IDs",
            "scene_meta_csv_fields": "each scene_meta object field equals the 2026 CSV",
            "scene_description_coverage": "combined description contains every object description",
            "pre_grasp_target": "pre_grasp target_object exists in the same scene",
            "output_grasp_target": "output_grasp target_object exists in the same scene",
            "inst_seg_class_set": "foreground mapping classes equal conf object IDs",
            "inst_seg_color_coverage": "image colors are mapped and every mapped object has pixels",
            "bbox_class_set": "bbox keys equal conf object IDs",
            "bbox_tightness": "bbox equals the min/max pixels of its mapped instance mask",
            "pointcloud_mapping": "pointcloud mapping agrees with scene/camera/classes/pixel counts",
        },
        "splits": {},
        "issue_examples": [],
    }
    issue_codes: Counter[str] = Counter()
    summary_rows: list[dict[str, Any]] = []

    def add(
        metrics: dict[str, dict[str, Any]],
        metric: str,
        unit: str,
        passed: bool,
        scene: SceneRef,
        code: str,
        message: str = "",
        camera: str | None = None,
        object_id: str | None = None,
    ) -> None:
        if metric not in metrics:
            metrics[metric] = new_metric(unit)
        entry = metrics[metric]
        entry["checked"] += 1
        entry["passed" if passed else "failed"] += 1
        if not passed:
            issue_codes[code] += 1
            if len(result["issue_examples"]) < config.MAX_ISSUE_EXAMPLES:
                result["issue_examples"].append(
                    {
                        "split": scene.split,
                        "scene": scene.relative_scene,
                        "camera": camera,
                        "object_id": object_id,
                        "metric": metric,
                        "code": code,
                        "message": message,
                    }
                )

    for split_number, split in enumerate(config.DATASETS):
        all_scenes = scene_refs_for_split(split)
        scenes = stratified_sample(
            all_scenes,
            config.SEMANTIC_SAMPLES_PER_SPLIT,
            lambda ref: (ref.env, ref.section, ref.platform),
            config.RANDOM_SEED + 200 + split_number,
        )
        metrics: dict[str, dict[str, Any]] = {}
        for scene_number, scene in enumerate(scenes, start=1):
            if scene_number == 1 or scene_number % 50 == 0 or scene_number == len(scenes):
                print(f"[{split}] semantic {scene_number:,}/{len(scenes):,} {scene.relative_scene}")
            platform = scene.platform_path
            name = scene.scene_name
            try:
                conf = strict_json_load(platform / "conf" / f"{name}.json")
            except Exception as error:
                add(metrics, "conf_object_catalog", "object", False, scene, "conf_read", str(error))
                continue
            conf_classes = class_set_from_conf(conf)
            conf_objects = conf.get("objects", []) if isinstance(conf, dict) else []

            for item in conf_objects:
                if not isinstance(item, dict):
                    add(metrics, "conf_object_catalog", "object", False, scene, "conf_object_type")
                    continue
                object_id = item.get("class")
                catalog_item = catalog.get(object_id)
                usd_stem = Path(str(item.get("usd_path", ""))).stem
                passed = (
                    catalog_item is not None
                    and usd_stem == catalog_item["Object_name"]
                )
                add(
                    metrics,
                    "conf_object_catalog",
                    "object",
                    passed,
                    scene,
                    "conf_catalog_mismatch",
                    f"USD stem={usd_stem!r}, CSV={catalog_item['Object_name']!r}" if catalog_item else "object ID absent from CSV",
                    object_id=object_id,
                )

            meta_path = platform / "scene_meta" / f"{name}.json"
            try:
                scene_meta = strict_json_load(meta_path)
                meta_objects = scene_meta.get("objects", {}) if isinstance(scene_meta, dict) else {}
            except Exception as error:
                scene_meta = {}
                meta_objects = {}
                meta_error = str(error)
            else:
                meta_error = ""
            meta_classes = set(meta_objects) if isinstance(meta_objects, dict) else set()
            add(
                metrics,
                "scene_meta_object_set",
                "scene",
                meta_classes == conf_classes,
                scene,
                "scene_meta_object_set",
                meta_error or f"conf={sorted(conf_classes)}, scene_meta={sorted(meta_classes)}",
            )
            if isinstance(meta_objects, dict):
                for object_id in sorted(conf_classes):
                    actual = meta_objects.get(object_id)
                    expected = catalog.get(object_id)
                    fields_ok = isinstance(actual, dict) and expected is not None
                    differences: list[str] = []
                    if fields_ok:
                        for meta_field, csv_field in META_FIELD_MAP.items():
                            actual_value = normalized_text(actual.get(meta_field, ""))
                            expected_value = normalized_text(expected[csv_field])
                            if actual_value != expected_value:
                                fields_ok = False
                                differences.append(
                                    f"{meta_field}: {actual_value!r} != {expected_value!r}"
                                )
                    add(
                        metrics,
                        "scene_meta_csv_fields",
                        "object",
                        fields_ok,
                        scene,
                        "scene_meta_csv_mismatch",
                        "; ".join(differences[:4]) or "missing object metadata",
                        object_id=object_id,
                    )

            combined_description = normalized_text(
                scene_meta.get("Description", "") if isinstance(scene_meta, dict) else ""
            )
            missing_descriptions = [
                object_id
                for object_id in conf_classes
                if object_id in catalog
                and normalized_text(catalog[object_id]["Description"])
                not in combined_description
            ]
            add(
                metrics,
                "scene_description_coverage",
                "scene",
                not missing_descriptions,
                scene,
                "description_missing",
                f"missing descriptions for {sorted(missing_descriptions)}",
            )

            grasp_sources = []
            if config.DATASETS[split]["require_pre_grasp"]:
                grasp_sources.append(("pre_grasp", platform / "pre_grasp" / f"{name}.json"))
            if config.DATASETS[split]["require_output_grasp"]:
                grasp_sources.append(("output_grasp", platform / "output_grasp" / f"{name}.json"))
            for grasp_kind, grasp_path in grasp_sources:
                try:
                    grasp_data = strict_json_load(grasp_path)
                    items = grasp_data.get("data", []) if grasp_kind == "pre_grasp" else grasp_data
                    if not isinstance(items, list):
                        raise TypeError("grasp records are not an array")
                except Exception as error:
                    add(metrics, f"{grasp_kind}_target", "grasp", False, scene, f"{grasp_kind}_read", str(error))
                    continue
                for record_index, record in enumerate(items):
                    target = record.get("target_object") if isinstance(record, dict) else None
                    add(
                        metrics,
                        f"{grasp_kind}_target",
                        "grasp",
                        target in conf_classes,
                        scene,
                        f"{grasp_kind}_target",
                        f"record {record_index}: target={target!r}, scene={sorted(conf_classes)}",
                        object_id=target,
                    )

            for camera in config.CAMERAS:
                mapping_path = platform / "inst_seg" / camera / f"semantics_mapping_{name}.json"
                image_path = platform / "inst_seg" / camera / f"{name}.png"
                bbox_path = platform / "bbox" / camera / f"{name}.json"
                pcd_mapping_path = platform / "pointcloud" / camera / f"pointcloud_inst_seg_mapping_{name}.json"
                try:
                    mapping = strict_json_load(mapping_path)
                    class_to_rgba = mapping_by_class(mapping)
                    mapped_classes = set(class_to_rgba) - SPECIAL_CLASSES
                except Exception as error:
                    mapping = {}
                    class_to_rgba = {}
                    mapped_classes = set()
                    mapping_error = str(error)
                else:
                    mapping_error = ""
                add(
                    metrics,
                    "inst_seg_class_set",
                    "camera scene",
                    mapped_classes == conf_classes,
                    scene,
                    "inst_seg_class_set",
                    mapping_error or f"conf={sorted(conf_classes)}, mapping={sorted(mapped_classes)}",
                    camera=camera,
                )

                try:
                    color_counts, packed_image = image_color_counts(image_path)
                    mapped_colors = {pack_rgba(rgba) for rgba in class_to_rgba.values()}
                    observed_colors = set(color_counts)
                    missing_color_labels = observed_colors - mapped_colors
                    empty_objects = [
                        object_id
                        for object_id in mapped_classes
                        if color_counts.get(pack_rgba(class_to_rgba[object_id]), 0) == 0
                    ]
                    color_ok = not missing_color_labels and not empty_objects
                    color_message = (
                        f"unmapped colors={len(missing_color_labels)}, empty objects={empty_objects}"
                    )
                except Exception as error:
                    color_counts = {}
                    packed_image = None
                    color_ok = False
                    color_message = str(error)
                add(
                    metrics,
                    "inst_seg_color_coverage",
                    "camera scene",
                    color_ok,
                    scene,
                    "inst_seg_color_coverage",
                    color_message,
                    camera=camera,
                )

                try:
                    bbox = strict_json_load(bbox_path)
                    bbox_classes = set(bbox) if isinstance(bbox, dict) else set()
                except Exception as error:
                    bbox = {}
                    bbox_classes = set()
                    bbox_error = str(error)
                else:
                    bbox_error = ""
                add(
                    metrics,
                    "bbox_class_set",
                    "camera scene",
                    bbox_classes == conf_classes,
                    scene,
                    "bbox_class_set",
                    bbox_error or f"conf={sorted(conf_classes)}, bbox={sorted(bbox_classes)}",
                    camera=camera,
                )

                for object_id in sorted(conf_classes):
                    expected_bbox = bbox.get(object_id) if isinstance(bbox, dict) else None
                    rgba = class_to_rgba.get(object_id)
                    mask_ok = packed_image is not None and rgba is not None
                    if mask_ok:
                        y_indices, x_indices = np.where(packed_image == pack_rgba(rgba))
                        mask_ok = bool(x_indices.size)
                    if mask_ok:
                        actual_bbox = [
                            int(x_indices.min()),
                            int(y_indices.min()),
                            int(x_indices.max()),
                            int(y_indices.max()),
                        ]
                        bbox_ok = (
                            isinstance(expected_bbox, list)
                            and len(expected_bbox) == 4
                            and max(abs(int(a) - int(b)) for a, b in zip(actual_bbox, expected_bbox))
                            <= config.BBOX_PIXEL_TOLERANCE
                        )
                    else:
                        actual_bbox = None
                        bbox_ok = False
                    add(
                        metrics,
                        "bbox_tightness",
                        "object camera",
                        bbox_ok,
                        scene,
                        "bbox_tightness",
                        f"stored={expected_bbox}, mask-derived={actual_bbox}",
                        camera=camera,
                        object_id=object_id,
                    )

                try:
                    pcd_mapping = strict_json_load(pcd_mapping_path)
                    instances = pcd_mapping.get("instances", {})
                    pcd_classes = {
                        value.get("class")
                        for value in instances.values()
                        if isinstance(value, dict)
                    }
                    pcd_counts = {
                        value.get("class"): value.get("pixel_count")
                        for value in instances.values()
                        if isinstance(value, dict)
                    }
                    verification = pcd_mapping.get("verification", {})
                    # Transparent BACKGROUND pixels do not produce pointcloud
                    # instances. Opaque UNLABELLED pixels do and must remain.
                    expected_pcd_classes = set(class_to_rgba) - {"BACKGROUND"}
                    pcd_ok = (
                        pcd_mapping.get("scene_id") == name
                        and pcd_mapping.get("camera") == camera
                        and pcd_classes == expected_pcd_classes
                        and verification.get("all_instance_counts_match") is True
                        and all(
                            pcd_counts.get(class_name)
                            == color_counts.get(pack_rgba(rgba), 0)
                            for class_name, rgba in class_to_rgba.items()
                            if class_name != "BACKGROUND"
                        )
                    )
                    pcd_message = (
                        f"mapping classes={sorted(class_to_rgba)}, pcd classes={sorted(str(v) for v in pcd_classes)}"
                    )
                except Exception as error:
                    pcd_ok = False
                    pcd_message = str(error)
                add(
                    metrics,
                    "pointcloud_mapping",
                    "camera scene",
                    pcd_ok,
                    scene,
                    "pointcloud_mapping",
                    pcd_message,
                    camera=camera,
                )

        for metric in metrics.values():
            metric["accuracy"] = (
                metric["passed"] / metric["checked"] if metric["checked"] else None
            )
            summary_rows.append({"split": split, **metric})
        result["splits"][split] = {
            "available_conf_scenes": len(all_scenes),
            "sampled_scenes": len(scenes),
            "metrics": dict(sorted(metrics.items())),
        }

    result["issue_code_counts"] = dict(sorted(issue_codes.items()))
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config.RESULT_DIR / "semantic_analysis.json", result)
    write_csv(config.RESULT_DIR / "semantic_analysis_summary.csv", summary_rows)
    return result


def main() -> None:
    result = analyze_semantics()
    print("\n=== AUTOMATIC SEMANTIC ACCURACY ===")
    for split, split_data in result["splits"].items():
        print(f"[{split}] sampled scenes={split_data['sampled_scenes']:,}")
        for name, metric in split_data["metrics"].items():
            accuracy = metric["accuracy"]
            text = "N/A" if accuracy is None else f"{accuracy * 100:.6f}%"
            print(f"  {name:30s} {metric['passed']:,}/{metric['checked']:,} {text}")
    print(f"saved: {config.RESULT_DIR / 'semantic_analysis.json'}")


if __name__ == "__main__":
    main()
