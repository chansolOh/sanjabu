"""Repair stale SceneGen semantics_mapping JSON files without rerendering.

Isaac Sim 5.1 may leave ``instance_segmentation_fast.idToSemantics`` from a
previous scene even though the instance-segmentation PNG and 2-D BBox output
belong to the current scene.  The visible mask BBox of every RGBA color is the
same geometry used by the tight BBox annotator, so the correct color-to-class
table can be recovered from the existing PNG and BBox JSON.

Only ``inst_seg/<camera>/semantics_mapping_XXXX.json`` files are changed.
RGB, depth, point cloud, segmentation PNG, BBox and conf files are read-only.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/dataset_v2")
CAMERA_NAMES = ("top_view_camera", "side_view_camera")

# False: scan/validate everything and only write reports.
# True: back up and atomically replace only mismatched semantics_mapping JSONs.
APPLY_CHANGES = True

# False makes the first dry-run a fast JSON-only conf/mapping class comparison.
# True additionally reads every PNG to catch a wrong color permutation even
# when the mapping happens to contain the same class set; this is much slower.
DEEP_VALIDATE_ALL_MAPPINGS = False

# A matched instance mask and tight BBox normally have IoU around 0.98-0.99.
MIN_MASK_BBOX_IOU = 0.90

# Optional narrowing for a test run. None means no filtering.
PLATFORM_PATH_CONTAINS = None  # Example: "conveyor_track_01"
SCENE_IDS = None  # Example: {"0000", "0001"}

# tqdm always shows the live progress. This interval controls additional
# persistent status lines printed through tqdm.write().
PROGRESS_PRINT_INTERVAL = 1000

BACKUP_ROOT = DATASET_ROOT.with_name(
    f"{DATASET_ROOT.name}_before_semantics_mapping_repair"
)
REPORT_DIR = Path(__file__).resolve().parent / "semantics_mapping_repair_reports"
REPORT_JSON = REPORT_DIR / "semantics_mapping_repair_report.json"
MISMATCH_PATH_LIST = REPORT_DIR / "semantics_mapping_mismatch_paths.txt"
UNRESOLVED_PATH_LIST = REPORT_DIR / "semantics_mapping_unresolved_paths.txt"

NUMERIC_SCENE_ID = re.compile(r"^\d+$")


def normalize_class_name(value: object) -> str:
    return str(value).strip().lower()


def parse_rgba(text: str) -> tuple[int, int, int, int]:
    value = ast.literal_eval(text)
    if not isinstance(value, tuple) or len(value) != 4:
        raise ValueError(f"Invalid RGBA mapping key: {text!r}")
    return tuple(int(channel) for channel in value)


def rgba_key(color: tuple[int, int, int, int]) -> str:
    return str(tuple(int(channel) for channel in color))


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def conf_class_table(conf_path: Path) -> dict[str, str]:
    data = load_json(conf_path)
    table: dict[str, str] = {}
    for obj in data.get("objects", []):
        class_name = obj.get("class") if isinstance(obj, dict) else None
        if not class_name:
            continue
        normalized = normalize_class_name(class_name)
        if normalized in table:
            raise ValueError(f"Duplicate conf class after normalization: {class_name!r}")
        table[normalized] = str(class_name)
    if not table:
        raise ValueError(f"No objects[].class entries in {conf_path}")
    return table


def canonical_bbox_table(
    bbox_path: Path,
    canonical_classes: dict[str, str],
) -> dict[str, list[float]]:
    raw = load_json(bbox_path)
    if not isinstance(raw, dict):
        raise ValueError(f"BBox JSON is not an object: {bbox_path}")

    result: dict[str, list[float]] = {}
    for bbox_class, bbox in raw.items():
        normalized = normalize_class_name(bbox_class)
        if normalized not in canonical_classes:
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            raise ValueError(f"Invalid BBox for {bbox_class!r} in {bbox_path}")
        result[canonical_classes[normalized]] = [float(value) for value in bbox[:4]]

    expected = set(canonical_classes.values())
    if set(result) != expected:
        missing = sorted(expected - set(result))
        raise ValueError(f"BBox is missing conf objects {missing}: {bbox_path}")
    return result


def mask_bbox(image: np.ndarray, color: tuple[int, int, int, int]) -> list[float] | None:
    rows, columns = np.where(np.all(image == np.asarray(color, dtype=np.uint8), axis=-1))
    if columns.size == 0:
        return None
    # max + 1 follows the half-open image rectangle convention. Existing BBox
    # JSON xmax/ymax differs by one pixel, which still gives IoU > 0.98.
    return [
        float(columns.min()),
        float(rows.min()),
        float(columns.max() + 1),
        float(rows.max() + 1),
    ]


def bbox_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def image_colors(image_path: Path) -> list[tuple[int, int, int, int]]:
    with Image.open(image_path) as source:
        rgba = source.convert("RGBA")
        colors_and_counts = rgba.getcolors(maxcolors=4096)
    if colors_and_counts is None:
        raise ValueError(f"Too many colors in instance segmentation: {image_path}")
    return [
        tuple(int(channel) for channel in color)
        for count, color in colors_and_counts
        if count > 0
    ]


def recover_object_color_mapping(
    image_path: Path,
    bbox_path: Path,
    canonical_classes: dict[str, str],
) -> tuple[dict[tuple[int, int, int, int], str], dict[str, float]]:
    bbox_by_class = canonical_bbox_table(bbox_path, canonical_classes)
    colors = [
        color for color in image_colors(image_path)
        if color[:3] != (0, 0, 0)
    ]
    if len(colors) < len(bbox_by_class):
        raise ValueError(
            f"Only {len(colors)} non-background colors for "
            f"{len(bbox_by_class)} objects: {image_path}"
        )

    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGBA"), dtype=np.uint8)

    color_bboxes = []
    visible_colors = []
    for color in colors:
        bbox = mask_bbox(image, color)
        if bbox is not None:
            visible_colors.append(color)
            color_bboxes.append(bbox)

    classes = list(bbox_by_class)
    costs = np.empty((len(visible_colors), len(classes)), dtype=float)
    ious = np.empty_like(costs)
    for color_index, color_bbox in enumerate(color_bboxes):
        for class_index, class_name in enumerate(classes):
            score = bbox_iou(color_bbox, bbox_by_class[class_name])
            ious[color_index, class_index] = score
            costs[color_index, class_index] = 1.0 - score

    color_indices, class_indices = linear_sum_assignment(costs)
    recovered: dict[tuple[int, int, int, int], str] = {}
    match_ious: dict[str, float] = {}
    for color_index, class_index in zip(color_indices, class_indices):
        color = visible_colors[int(color_index)]
        class_name = classes[int(class_index)]
        score = float(ious[color_index, class_index])
        if score < MIN_MASK_BBOX_IOU:
            raise ValueError(
                f"Low mask/BBox IoU {score:.4f} for {class_name!r} in {image_path}"
            )
        recovered[color] = class_name
        match_ious[class_name] = score

    if set(recovered.values()) != set(classes):
        raise ValueError(f"Could not assign every conf object in {image_path}")
    return recovered, match_ious


def repaired_mapping(
    mapping: dict,
    recovered: dict[tuple[int, int, int, int], str],
) -> dict:
    result = {str(key): value for key, value in mapping.items()}
    for color, class_name in recovered.items():
        key = rgba_key(color)
        old_value = result.get(key)
        if isinstance(old_value, dict):
            new_value = dict(old_value)
            new_value["class"] = class_name
        else:
            new_value = {"class": class_name}
        result[key] = new_value
    return result


def object_mapping_is_equal(
    existing: dict,
    recovered: dict[tuple[int, int, int, int], str],
) -> bool:
    for color, expected_class in recovered.items():
        labels = existing.get(rgba_key(color))
        if not isinstance(labels, dict) or labels.get("class") != expected_class:
            return False
    return True


def fast_mapping_may_be_wrong(mapping: dict, canonical_classes: dict[str, str]) -> bool:
    existing_classes = {
        labels.get("class")
        for labels in mapping.values()
        if isinstance(labels, dict) and labels.get("class") is not None
    }
    return not set(canonical_classes.values()).issubset(existing_classes)


def atomic_write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.repair.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=4, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)


def backup_mapping(mapping_path: Path) -> Path:
    backup_path = BACKUP_ROOT / mapping_path.relative_to(DATASET_ROOT)
    if not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mapping_path, backup_path)
    return backup_path


def discover_conf_paths() -> list[Path]:
    paths = []
    for path in DATASET_ROOT.rglob("conf/*.json"):
        if not NUMERIC_SCENE_ID.fullmatch(path.stem):
            continue
        platform_root = path.parent.parent
        if PLATFORM_PATH_CONTAINS and PLATFORM_PATH_CONTAINS not in str(platform_root):
            continue
        if SCENE_IDS is not None and path.stem not in {str(value).zfill(4) for value in SCENE_IDS}:
            continue
        paths.append(path)
    return sorted(paths)


def mapping_paths_for_scene(platform_root: Path, scene_id: str):
    for camera_name in CAMERA_NAMES:
        camera_inst_seg = platform_root / "inst_seg" / camera_name
        yield (
            camera_name,
            camera_inst_seg / f"{scene_id}.png",
            platform_root / "bbox" / camera_name / f"{scene_id}.json",
            camera_inst_seg / f"semantics_mapping_{scene_id}.json",
        )


def write_reports(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(REPORT_JSON, report)
    MISMATCH_PATH_LIST.write_text(
        "".join(f"{path}\n" for path in report["mismatched_paths"]),
        encoding="utf-8",
    )
    UNRESOLVED_PATH_LIST.write_text(
        "".join(f"{item['mapping_path']}\n" for item in report["unresolved"]),
        encoding="utf-8",
    )


def main() -> int:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {DATASET_ROOT}")

    report = {
        "dataset_root": str(DATASET_ROOT),
        "apply_changes": APPLY_CHANGES,
        "deep_validate_all_mappings": DEEP_VALIDATE_ALL_MAPPINGS,
        "minimum_mask_bbox_iou": MIN_MASK_BBOX_IOU,
        "mismatched_paths": [],
        "repaired_paths": [],
        "unresolved": [],
        "checked_mapping_count": 0,
        "checked_scene_count": 0,
    }

    conf_paths = discover_conf_paths()
    print(f"Dataset root: {DATASET_ROOT}", flush=True)
    print(f"Scenes to check: {len(conf_paths)}", flush=True)
    print(f"Apply changes: {APPLY_CHANGES}", flush=True)
    progress = tqdm(
        conf_paths,
        desc="semantics mapping",
        unit="scene",
        dynamic_ncols=True,
        mininterval=0.2,
    )
    for scene_index, conf_path in enumerate(progress, start=1):
        scene_id = conf_path.stem
        platform_root = conf_path.parent.parent
        try:
            canonical_classes = conf_class_table(conf_path)
        except Exception as error:
            report["unresolved"].append(
                {
                    "mapping_path": str(conf_path),
                    "reason": f"conf: {type(error).__name__}: {error}",
                }
            )
            continue

        report["checked_scene_count"] += 1
        for camera_name, image_path, bbox_path, mapping_path in mapping_paths_for_scene(
            platform_root, scene_id
        ):
            report["checked_mapping_count"] += 1
            try:
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                if not bbox_path.is_file():
                    raise FileNotFoundError(bbox_path)
                if not mapping_path.is_file():
                    raise FileNotFoundError(mapping_path)

                existing = load_json(mapping_path)
                fast_mismatch = fast_mapping_may_be_wrong(existing, canonical_classes)
                if not fast_mismatch and not DEEP_VALIDATE_ALL_MAPPINGS:
                    continue
                if fast_mismatch and not APPLY_CHANGES and not DEEP_VALIDATE_ALL_MAPPINGS:
                    report["mismatched_paths"].append(str(mapping_path))
                    continue

                recovered, match_ious = recover_object_color_mapping(
                    image_path,
                    bbox_path,
                    canonical_classes,
                )
                if object_mapping_is_equal(existing, recovered):
                    continue

                report["mismatched_paths"].append(str(mapping_path))
                if APPLY_CHANGES:
                    updated = repaired_mapping(existing, recovered)
                    backup_path = backup_mapping(mapping_path)
                    atomic_write_json(mapping_path, updated)
                    report["repaired_paths"].append(
                        {
                            "mapping_path": str(mapping_path),
                            "backup_path": str(backup_path),
                            "camera": camera_name,
                            "minimum_iou": min(match_ious.values()),
                        }
                    )
            except Exception as error:
                report["unresolved"].append(
                    {
                        "mapping_path": str(mapping_path),
                        "camera": camera_name,
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )

        progress.set_postfix(
            mismatch=len(report["mismatched_paths"]),
            repaired=len(report["repaired_paths"]),
            unresolved=len(report["unresolved"]),
            refresh=False,
        )
        if (
            PROGRESS_PRINT_INTERVAL > 0
            and (
                scene_index % PROGRESS_PRINT_INTERVAL == 0
                or scene_index == len(conf_paths)
            )
        ):
            tqdm.write(
                f"[{scene_index}/{len(conf_paths)}] "
                f"mismatch={len(report['mismatched_paths'])}, "
                f"repaired={len(report['repaired_paths'])}, "
                f"unresolved={len(report['unresolved'])}"
            )

    progress.close()

    write_reports(report)
    print(f"Mismatch path list: {MISMATCH_PATH_LIST}")
    print(f"Unresolved path list: {UNRESOLVED_PATH_LIST}")
    print(f"Full report: {REPORT_JSON}")
    if APPLY_CHANGES:
        print(f"Backup root: {BACKUP_ROOT}")
    return 1 if report["unresolved"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
