#!/usr/bin/env python3
"""Create explicit point-instance-ID mappings for Dataset_2026 point clouds.

Dataset_2026 contains two encodings: packed-RGBA uint32 IDs and sequential
``1..N`` IDs colored with Isaac Sim's deterministic palette. This script
detects both without relying on JSON key order. Before accepting a pairing, it
verifies every point-ID count against the matching PNG RGBA pixel count. Extra
semantics entries that do not occur in the PNG are recorded and ignored.

Object class strings are copied verbatim from semantics_mapping; this script
does not translate names to obj_###. The default mode is a read-only dry run.
Set ``APPLY_CHANGES = True`` below to create files in each pointcloud camera
directory.
"""

from __future__ import annotations

import ast
import colorsys
import json
import os
import re
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


# =============================================================================
# 사용자 설정
# =============================================================================
DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/dataset_v2")

# 비어 있으면 environment 기준 필터를 사용하지 않습니다.
# 3대의 PC에서 각각 아래처럼 하나씩 지정할 수 있습니다.
#   PC 1: ("Home",)
#   PC 2: ("Logistic_site",)
#   PC 3: ("Manufactory",)
SELECTED_ENVIRONMENTS: tuple[str, ...] = ()

# environment 전체가 아니라 특정 platform만 처리할 때 사용합니다.
# SELECTED_ENVIRONMENTS와 함께 지정하면 두 선택 범위를 합쳐서 처리합니다.
# 예: ("Home/MasterBedroom/bed_01", "Home/MasterBedroom/vanity_01")
SELECTED_PLATFORMS: tuple[str, ...] = ("Manufactory/Seongju_Melon_Processing_Facility/stainless_steel_work_table_01", "Logistic_site/General_LogisticSite/rack_small_A5_02")

SELECTED_CAMERAS = ("top_view_camera", "side_view_camera")
WORKERS = 8

# 빠른 테스트 시 정수로 지정하고, 전체 처리 시 None으로 둡니다.
LIMIT_SCENES: int | None = None

# False: 검사만 수행, True: 매핑 JSON 생성
APPLY_CHANGES = True

# 기존 파일의 내용이 다를 때 덮어쓸지 여부
OVERWRITE_EXISTING = False

# True이면 오류가 하나라도 있을 때 정상 scene의 파일도 생성하지 않습니다.
FAIL_ON_ERROR = False

# 실행 보고서를 파일로 남기려면 PC마다 서로 다른 Path를 지정합니다.
# 예: Path("/home/uon/ochansol/isaac_code/python/sanjabu/2026/etc/report_home.json")
REPORT_JSON: Path | None = None


CAMERA_NAMES = ("top_view_camera", "side_view_camera")
INSTANCE_RE = re.compile(r"^pointcloud_inst_seg_(\d+)\.npy$")


@dataclass(frozen=True)
class Candidate:
    platform: Path
    camera: str
    scene_id: int
    instance_path: Path
    image_path: Path
    semantics_path: Path
    output_path: Path


@dataclass
class Inspection:
    candidate: Candidate
    output_data: dict[str, Any] | None = None
    action: str = ""
    error: str | None = None


def discover_platforms(
    dataset_root: Path,
    selected_platforms: tuple[str, ...],
    selected_environments: tuple[str, ...],
) -> list[Path]:
    root = dataset_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"dataset root not found: {root}")

    result: set[Path] = set()
    for value_text in selected_platforms:
        value = Path(value_text)
        platform = (value if value.is_absolute() else root / value).resolve()
        if not platform.is_dir():
            raise NotADirectoryError(f"platform not found: {platform}")
        if root != platform and root not in platform.parents:
            raise ValueError(f"platform is outside dataset root: {platform}")
        result.add(platform)

    for environment_name in selected_environments:
        if not environment_name or Path(environment_name).name != environment_name:
            raise ValueError(
                "SELECTED_ENVIRONMENTS에는 environment 폴더 이름만 넣으십시오: "
                f"{environment_name!r}"
            )
        environment = (root / environment_name).resolve()
        if not environment.is_dir() or environment.parent != root:
            raise NotADirectoryError(f"environment directory not found: {environment}")
        if environment.name == "pregrasp_statistics":
            raise ValueError("pregrasp_statistics is not a dataset environment")

        found_platforms = 0
        for section in environment.iterdir():
            if not section.is_dir():
                continue
            for platform in section.iterdir():
                if not platform.is_dir():
                    continue
                result.add(platform.resolve())
                found_platforms += 1
        if found_platforms == 0:
            raise ValueError(f"no platforms found under environment: {environment}")

    if selected_platforms or selected_environments:
        return sorted(result)

    all_platforms = []
    for environment in root.iterdir():
        if not environment.is_dir() or environment.name == "pregrasp_statistics":
            continue
        for section in environment.iterdir():
            if not section.is_dir():
                continue
            all_platforms.extend(path for path in section.iterdir() if path.is_dir())
    return sorted(set(all_platforms))


def discover_candidates(
    platforms: Iterable[Path], cameras: Iterable[str], limit: int | None
) -> list[Candidate]:
    candidates = []
    for platform in platforms:
        for camera in cameras:
            pointcloud_dir = platform / "pointcloud" / camera
            inst_seg_dir = platform / "inst_seg" / camera
            if not pointcloud_dir.is_dir():
                continue
            for path in pointcloud_dir.iterdir():
                match = INSTANCE_RE.fullmatch(path.name)
                if not match or not path.is_file():
                    continue
                scene_id = int(match.group(1))
                stem = f"{scene_id:04d}"
                candidates.append(
                    Candidate(
                        platform=platform,
                        camera=camera,
                        scene_id=scene_id,
                        instance_path=path,
                        image_path=inst_seg_dir / f"{stem}.png",
                        semantics_path=inst_seg_dir / f"semantics_mapping_{stem}.json",
                        output_path=pointcloud_dir / f"pointcloud_inst_seg_mapping_{stem}.json",
                    )
                )
    candidates.sort(key=lambda item: (str(item.platform), item.camera, item.scene_id))
    if limit is not None:
        if limit < 0:
            raise ValueError("LIMIT_SCENES must be >= 0")
        candidates = candidates[:limit]
    return candidates


def parse_rgba(value: str) -> tuple[int, int, int, int]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"invalid RGBA mapping key {value!r}") from error
    if not isinstance(parsed, (tuple, list)) or len(parsed) != 4:
        raise ValueError(f"RGBA key must contain four channels: {value!r}")
    rgba = tuple(int(channel) for channel in parsed)
    if any(channel < 0 or channel > 255 for channel in rgba):
        raise ValueError(f"RGBA channel outside 0..255: {value!r}")
    return rgba


def rgba_to_uint32(rgba: tuple[int, int, int, int]) -> int:
    red, green, blue, alpha = rgba
    return red + (green << 8) + (blue << 16) + (alpha << 24)


def load_png_counts(path: Path) -> tuple[Counter[int], tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"inst_seg PNG not found: {path}")
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    packed = (
        rgba[..., 0].astype(np.uint32)
        + (rgba[..., 1].astype(np.uint32) << 8)
        + (rgba[..., 2].astype(np.uint32) << 16)
        + (rgba[..., 3].astype(np.uint32) << 24)
    )
    values, counts = np.unique(packed, return_counts=True)
    color_counts = Counter(
        {int(value): int(count) for value, count in zip(values, counts)}
    )
    return color_counts, rgba.shape[:2]


def legacy_palette(size: int) -> list[tuple[int, int, int, int]]:
    """Reproduce the deterministic Isaac Sim 5.0 segmentation palette."""
    colors = []
    for index in range(size):
        red, green, blue = colorsys.hsv_to_rgb(index / size, 0.9, 1.0)
        colors.append((int(red * 255), int(green * 255), int(blue * 255), 255))
    if colors:
        # The writer makes the first (unlabelled) entry black. Dataset_2026's
        # transparency-normalization pass changes its alpha from 0 to 255.
        colors[0] = (0, 0, 0, 255)
    return colors


def resolve_id_color_pairs(
    unique_ids: np.ndarray,
    point_counts: np.ndarray,
    semantics: dict[str, Any],
    png_counts: Counter[int],
) -> tuple[
    list[tuple[int, int, tuple[int, int, int, int], dict[str, Any]]],
    str,
    list[dict[str, Any]],
]:
    """Resolve mixed Dataset_2026 encodings without trusting JSON key order.

    Dataset_2026 contains both packed-RGBA point IDs and sequential point IDs.
    Some semantics JSON files also contain colors that do not occur in the PNG.
    Resolution therefore uses, in order: direct packed-ID matching, the known
    deterministic Isaac 5.0 palette, then unique per-instance counts.
    """
    active: dict[tuple[int, int, int, int], tuple[dict[str, Any], int]] = {}
    ignored = []
    for color_key, raw_labels in semantics.items():
        rgba = parse_rgba(color_key)
        labels = raw_labels if isinstance(raw_labels, dict) else {"class": str(raw_labels)}
        pixel_count = int(png_counts.get(rgba_to_uint32(rgba), 0))
        if pixel_count == 0:
            ignored.append({"rgba": list(rgba), "labels": dict(labels), "reason": "absent_from_png"})
            continue
        if rgba in active:
            raise ValueError(f"duplicate active RGBA mapping: {rgba}")
        active[rgba] = (dict(labels), pixel_count)

    point_data = [(int(instance_id), int(count)) for instance_id, count in zip(unique_ids, point_counts)]
    if len(point_data) != len(active):
        raise ValueError(
            f"unique point IDs ({len(point_data)}) != active PNG/semantics colors "
            f"({len(active)}); ignored semantics entries={len(ignored)}"
        )

    packed_to_color = {rgba_to_uint32(rgba): rgba for rgba in active}
    if all(instance_id in packed_to_color for instance_id, _count in point_data):
        pairs = []
        for instance_id, point_count in point_data:
            rgba = packed_to_color[instance_id]
            labels, pixel_count = active[rgba]
            if point_count != pixel_count:
                raise ValueError(
                    f"packed-ID count mismatch for {instance_id} / {rgba}: "
                    f"points={point_count}, pixels={pixel_count}"
                )
            pairs.append((instance_id, point_count, rgba, labels))
        return pairs, "packed_rgba_uint32", ignored

    expected_palette = legacy_palette(len(point_data))
    if set(expected_palette) == set(active):
        pairs = []
        palette_valid = True
        for (instance_id, point_count), rgba in zip(point_data, expected_palette):
            labels, pixel_count = active[rgba]
            if point_count != pixel_count:
                palette_valid = False
                break
            pairs.append((instance_id, point_count, rgba, labels))
        if palette_valid:
            return pairs, "isaac_sim_5_0_deterministic_palette", ignored

    ids_by_count: dict[int, list[int]] = {}
    colors_by_count: dict[int, list[tuple[int, int, int, int]]] = {}
    for instance_id, point_count in point_data:
        ids_by_count.setdefault(point_count, []).append(instance_id)
    for rgba, (_labels, pixel_count) in active.items():
        colors_by_count.setdefault(pixel_count, []).append(rgba)
    if set(ids_by_count) != set(colors_by_count):
        raise ValueError(
            "point-ID counts and PNG color counts differ; no safe mapping can be inferred"
        )

    pairs = []
    for count, instance_ids in ids_by_count.items():
        colors = colors_by_count[count]
        if len(instance_ids) != 1 or len(colors) != 1:
            raise ValueError(
                f"ambiguous count {count}: IDs={instance_ids}, RGBA colors={colors}"
            )
        instance_id = instance_ids[0]
        rgba = colors[0]
        labels, _pixel_count = active[rgba]
        pairs.append((instance_id, count, rgba, labels))
    pairs.sort(key=lambda item: item[0])
    return pairs, "unique_point_and_pixel_counts", ignored


def build_mapping(candidate: Candidate) -> dict[str, Any]:
    if not candidate.semantics_path.is_file():
        raise FileNotFoundError(f"semantics mapping not found: {candidate.semantics_path}")
    with candidate.semantics_path.open("r", encoding="utf-8") as stream:
        semantics = json.load(stream)
    if not isinstance(semantics, dict) or not semantics:
        raise ValueError("semantics mapping root must be a non-empty JSON object")

    instance_ids = np.load(candidate.instance_path, mmap_mode="r", allow_pickle=False)
    if instance_ids.ndim != 1:
        raise ValueError(f"point instance array must be 1-D, got {instance_ids.shape}")
    unique_ids, point_counts = np.unique(instance_ids, return_counts=True)
    if not np.issubdtype(unique_ids.dtype, np.integer):
        raise ValueError(f"point instance dtype must be integer, got {instance_ids.dtype}")

    png_counts, image_shape = load_png_counts(candidate.image_path)
    if instance_ids.size != image_shape[0] * image_shape[1]:
        raise ValueError(
            f"point count {instance_ids.size} != PNG pixel count "
            f"{image_shape[0] * image_shape[1]}"
        )

    resolved, method, ignored = resolve_id_color_pairs(
        unique_ids, point_counts, semantics, png_counts
    )
    instances: dict[str, Any] = {}
    for raw_id, point_count, rgba, labels in resolved:
        pixel_count = int(png_counts[rgba_to_uint32(rgba)])
        entry = dict(labels)
        entry.update(
            {
                "rgba": list(rgba),
                "point_count": point_count,
                "pixel_count": pixel_count,
            }
        )
        instances[str(raw_id)] = entry

    return {
        "scene_id": f"{candidate.scene_id:04d}",
        "camera": candidate.camera,
        "instance_source": candidate.instance_path.name,
        "inst_seg_source": candidate.image_path.name,
        "semantics_source": candidate.semantics_path.name,
        "verification": {
            "method": method,
            "point_count": int(instance_ids.size),
            "pixel_count": int(image_shape[0] * image_shape[1]),
            "all_instance_counts_match": True,
            "ignored_semantics_entries": ignored,
        },
        "instances": instances,
    }


def inspect_candidate(candidate: Candidate, overwrite: bool) -> Inspection:
    try:
        output_data = build_mapping(candidate)
        if candidate.output_path.exists():
            with candidate.output_path.open("r", encoding="utf-8") as stream:
                existing = json.load(stream)
            comparison_keys = (
                "scene_id",
                "camera",
                "instance_source",
                "inst_seg_source",
                "semantics_source",
                "instances",
            )
            if all(existing.get(key) == output_data.get(key) for key in comparison_keys):
                return Inspection(candidate, output_data, action="unchanged")
            if not overwrite:
                return Inspection(
                    candidate,
                    output_data,
                    action="conflict",
                    error=(
                        "mapping already exists with different instance content; "
                        "set OVERWRITE_EXISTING = True"
                    ),
                )
            return Inspection(candidate, output_data, action="overwrite")
        return Inspection(candidate, output_data, action="create")
    except Exception as error:
        return Inspection(
            candidate,
            action="error",
            error=f"{type(error).__name__}: {error}",
        )


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=4)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_report(
    platforms: list[Path],
    results: list[Inspection],
    written: int,
) -> dict[str, Any]:
    actions = Counter(result.action for result in results)
    errors = [
        {"path": str(result.candidate.instance_path), "error": result.error}
        for result in results
        if result.error
    ]
    return {
        "mode": "apply" if APPLY_CHANGES else "dry-run",
        "dataset_root": str(DATASET_ROOT.resolve()),
        "selected_environments": list(SELECTED_ENVIRONMENTS),
        "selected_platforms": list(SELECTED_PLATFORMS),
        "selected_cameras": list(SELECTED_CAMERAS),
        "platform_count": len(platforms),
        "candidate_scenes": len(results),
        "actions": dict(sorted(actions.items())),
        "written_files": written,
        "errors": errors,
        "note": "Class strings are copied unchanged from inst_seg semantics_mapping JSON.",
    }


def main() -> int:
    if WORKERS < 1:
        raise ValueError("WORKERS must be >= 1")
    if OVERWRITE_EXISTING and not APPLY_CHANGES:
        raise ValueError("OVERWRITE_EXISTING=True requires APPLY_CHANGES=True")
    invalid_cameras = set(SELECTED_CAMERAS) - set(CAMERA_NAMES)
    if invalid_cameras:
        raise ValueError(f"invalid SELECTED_CAMERAS: {sorted(invalid_cameras)}")

    platforms = discover_platforms(
        DATASET_ROOT,
        SELECTED_PLATFORMS,
        SELECTED_ENVIRONMENTS,
    )
    candidates = discover_candidates(platforms, SELECTED_CAMERAS, LIMIT_SCENES)
    print(
        f"mode={'APPLY' if APPLY_CHANGES else 'DRY-RUN'} "
        f"platforms={len(platforms)} camera_scenes={len(candidates)}"
    )

    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for index, result in enumerate(
            executor.map(
                lambda item: inspect_candidate(item, OVERWRITE_EXISTING), candidates
            ),
            1,
        ):
            results.append(result)
            if index % 500 == 0 or index == len(candidates):
                errors = sum(item.error is not None for item in results)
                print(f"inspected {index}/{len(candidates)} errors={errors}", flush=True)

    errors = [result for result in results if result.error]
    written = 0
    can_write = APPLY_CHANGES and (not errors or not FAIL_ON_ERROR)
    if can_write:
        for result in results:
            if result.action not in {"create", "overwrite"}:
                continue
            assert result.output_data is not None
            atomic_write_json(result.candidate.output_path, result.output_data)
            written += 1
            if written % 500 == 0:
                print(f"written {written}", flush=True)
    if errors and APPLY_CHANGES and not FAIL_ON_ERROR:
        print(
            f"WARNING: skipped {len(errors)} invalid scenes; valid mappings were written.",
            file=sys.stderr,
        )

    report = build_report(platforms, results, written)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if REPORT_JSON:
        REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
        REPORT_JSON.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 2 if errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; no files written before validation completes", file=sys.stderr)
        raise SystemExit(130)
