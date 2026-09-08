#!/usr/bin/env python3
"""Generate Dataset_2026 scene_meta JSON files from conf and the 2026 CSV.

The output schema follows the Dataset_2025 ``scene_meta_gen.ipynb`` result.
Object keys are always the CSV ``Class_name`` (obj_###), regardless of whether
the source conf currently contains obj_### or the original object name.

The default is a read-only dry run. Change the variables in the user-settings
section below; this script intentionally has no command-line arguments.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OBJ_ID_RE = re.compile(r"obj_\d{3,}")


# =============================================================================
# 사용자 설정
# =============================================================================
DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/dataset_v2")
MAPPING_CSV = Path(
    "/nas/ochansol/3d_model/peel3_scan_data_2026/2026_objects_cat_attr.csv"
)

# 비어 있으면 environment 기준 필터를 사용하지 않습니다.
# 예: ("Home",), ("Logistic_site",), ("Manufactory",)
SELECTED_ENVIRONMENTS: tuple[str, ...] = ()

# 특정 platform만 생성할 때 사용합니다. environment 선택과 함께 쓰면 합집합입니다.
# 예: ("Home/MasterBedroom/bed_01",)
SELECTED_PLATFORMS: tuple[str, ...] = ()

WORKERS = 8
LIMIT_FILES: int | None = None

# False: 검사만 하며 scene_meta 파일과 폴더를 만들지 않습니다.
# True : 전체 사전 검사가 성공한 경우에만 scene_meta를 생성합니다.
APPLY_CHANGES = True

# False이면 기존 scene_meta JSON은 보존하고 건너뜁니다.
# True이면 기존 파일도 새 CSV와 conf 기준으로 덮어씁니다.
OVERWRITE_EXISTING = False

REPORT_JSON: Path | None = None


@dataclass(frozen=True)
class ObjectMetadata:
    class_name: str
    object_name: str
    level_1: str
    level_2: str
    level_3: str
    color: str
    packaging: str
    features: str
    description: str

    def scene_meta_fields(self) -> dict[str, str]:
        return {
            "level_1": self.level_1,
            "level_2": self.level_2,
            "level_3": self.level_3,
            "object_name": self.object_name,
            "color": self.color,
            "packaging": self.packaging,
            "features": self.features,
            "description": self.description,
        }


@dataclass(frozen=True)
class Candidate:
    conf_path: Path
    output_path: Path
    platform: Path


@dataclass
class Inspection:
    candidate: Candidate
    object_count: int = 0
    duplicate_class_count: int = 0
    unmapped: set[str] | None = None
    error: str | None = None


def _clean(row: dict[str, Any], key: str) -> str:
    return str(row.get(key) or "").strip()


def load_metadata(
    csv_path: Path,
) -> tuple[dict[str, ObjectMetadata], dict[str, ObjectMetadata]]:
    """Return metadata indexed by Class_name and case-folded Object_name."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"metadata CSV not found: {csv_path}")

    by_class: dict[str, ObjectMetadata] = {}
    by_object_name: dict[str, ObjectMetadata] = {}
    required = {
        "Object_name",
        "Level_1",
        "Level_2",
        "Level_3",
        "Class_name",
        "Color",
        "Packaging",
        "Features",
        "Description",
    }

    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"CSV is missing required columns {missing}: {csv_path}")

        for line_number, row in enumerate(reader, 2):
            object_name = _clean(row, "Object_name")
            class_name = _clean(row, "Class_name")

            # Blank/footer statistics rows in the supplied CSV are not object
            # records. Some contain values in columns other than Object_name.
            if not object_name:
                continue
            if not OBJ_ID_RE.fullmatch(class_name):
                raise ValueError(
                    f"invalid Class_name at CSV line {line_number}: {class_name!r}"
                )

            metadata = ObjectMetadata(
                class_name=class_name,
                object_name=object_name,
                level_1=_clean(row, "Level_1"),
                level_2=_clean(row, "Level_2"),
                level_3=_clean(row, "Level_3"),
                color=_clean(row, "Color"),
                packaging=_clean(row, "Packaging"),
                features=_clean(row, "Features"),
                description=_clean(row, "Description"),
            )

            existing_class = by_class.get(class_name)
            if existing_class and existing_class != metadata:
                raise ValueError(f"duplicate Class_name in CSV: {class_name}")
            normalized_name = object_name.casefold()
            existing_name = by_object_name.get(normalized_name)
            if existing_name and existing_name.class_name != class_name:
                raise ValueError(
                    "case-insensitive duplicate Object_name with different IDs: "
                    f"{object_name}"
                )
            by_class[class_name] = metadata
            by_object_name[normalized_name] = metadata

    if not by_class:
        raise ValueError(f"no object metadata found in CSV: {csv_path}")
    return by_class, by_object_name


def discover_all_platforms(dataset_root: Path) -> list[Path]:
    platforms: list[Path] = []
    for environment in dataset_root.iterdir():
        if not environment.is_dir() or environment.name == "pregrasp_statistics":
            continue
        for section in environment.iterdir():
            if not section.is_dir():
                continue
            platforms.extend(path for path in section.iterdir() if path.is_dir())
    return sorted(set(platforms))


def resolve_platforms(dataset_root: Path) -> list[Path]:
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"dataset root not found: {dataset_root}")

    selected: set[Path] = set()
    for relative in SELECTED_PLATFORMS:
        platform = (dataset_root / relative).resolve()
        if not platform.is_dir() or dataset_root not in platform.parents:
            raise NotADirectoryError(f"platform directory not found: {platform}")
        selected.add(platform)

    for environment_name in SELECTED_ENVIRONMENTS:
        if not environment_name or Path(environment_name).name != environment_name:
            raise ValueError(
                "SELECTED_ENVIRONMENTS에는 environment 폴더 이름만 넣으십시오: "
                f"{environment_name!r}"
            )
        environment = (dataset_root / environment_name).resolve()
        if not environment.is_dir() or environment.parent != dataset_root:
            raise NotADirectoryError(f"environment directory not found: {environment}")
        if environment.name == "pregrasp_statistics":
            raise ValueError("pregrasp_statistics is not a dataset environment")

        found_platforms = 0
        for section in environment.iterdir():
            if not section.is_dir():
                continue
            for path in section.iterdir():
                if path.is_dir():
                    selected.add(path.resolve())
                    found_platforms += 1
        if found_platforms == 0:
            raise ValueError(f"no platforms found under environment: {environment}")

    if SELECTED_PLATFORMS or SELECTED_ENVIRONMENTS:
        return sorted(selected)
    return discover_all_platforms(dataset_root)


def discover_candidates(platforms: list[Path]) -> list[Candidate]:
    candidates = [
        Candidate(
            conf_path=conf_path,
            output_path=platform / "scene_meta" / conf_path.name,
            platform=platform,
        )
        for platform in platforms
        for conf_path in platform.glob("conf/*.json")
        if conf_path.is_file()
    ]
    candidates.sort(key=lambda item: str(item.conf_path))
    if LIMIT_FILES is not None:
        if LIMIT_FILES < 0:
            raise ValueError("LIMIT_FILES must be >= 0")
        candidates = candidates[:LIMIT_FILES]
    return candidates


def build_scene_meta(
    conf_data: Any,
    by_class: dict[str, ObjectMetadata],
    by_object_name: dict[str, ObjectMetadata],
) -> tuple[dict[str, Any], set[str], int, int]:
    if not isinstance(conf_data, dict) or not isinstance(conf_data.get("objects"), list):
        raise ValueError("conf JSON must contain an objects list")

    output_objects: dict[str, dict[str, str]] = {}
    descriptions: list[str] = []
    unmapped: set[str] = set()
    object_count = 0
    duplicate_count = 0

    for item in conf_data["objects"]:
        if not isinstance(item, dict):
            raise ValueError("conf objects entries must be objects")
        identifier = item.get("class")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"invalid conf object class: {identifier!r}")
        identifier = identifier.strip()

        metadata = by_class.get(identifier)
        if metadata is None:
            metadata = by_object_name.get(identifier.casefold())
        if metadata is None:
            unmapped.add(identifier)
            continue

        object_count += 1
        if metadata.class_name in output_objects:
            duplicate_count += 1
        output_objects[metadata.class_name] = metadata.scene_meta_fields()
        if metadata.description:
            descriptions.append(metadata.description)

    # Keep the leading space produced by the 2025 notebook for schema/content
    # compatibility with the existing Dataset_2025 scene_meta files.
    combined_description = f" {' '.join(descriptions)}" if descriptions else ""
    return (
        {"objects": output_objects, "Description": combined_description},
        unmapped,
        object_count,
        duplicate_count,
    )


def inspect_candidate(
    candidate: Candidate,
    by_class: dict[str, ObjectMetadata],
    by_object_name: dict[str, ObjectMetadata],
) -> Inspection:
    try:
        with candidate.conf_path.open("r", encoding="utf-8") as stream:
            conf_data = json.load(stream)
        _, unmapped, object_count, duplicate_count = build_scene_meta(
            conf_data, by_class, by_object_name
        )
        return Inspection(
            candidate=candidate,
            object_count=object_count,
            duplicate_class_count=duplicate_count,
            unmapped=unmapped,
        )
    except Exception as error:
        return Inspection(
            candidate=candidate,
            unmapped=set(),
            error=f"{type(error).__name__}: {error}",
        )


def atomic_write_json(path: Path, data: Any, source_mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = path.stat().st_mode if path.exists() else source_mode
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
        os.chmod(temporary, output_mode)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def print_banner() -> None:
    border = "=" * 78
    print(border)
    if APPLY_CHANGES:
        print("APPLY MODE — SCENE_META JSON FILES WILL BE CREATED")
        print(f"overwrite existing: {OVERWRITE_EXISTING}")
    else:
        print("DRY-RUN MODE — NO FOLDERS OR FILES WILL BE CREATED")
        print("생성하려면 파일 상단의 APPLY_CHANGES = True 로 설정하십시오.")
    print(border, flush=True)


def main() -> int:
    if WORKERS < 1:
        raise ValueError("WORKERS must be >= 1")

    print_banner()
    by_class, by_object_name = load_metadata(MAPPING_CSV)
    platforms = resolve_platforms(DATASET_ROOT)
    candidates = discover_candidates(platforms)
    print(
        f"metadata={len(by_class)} platforms={len(platforms)} "
        f"conf_files={len(candidates)}",
        flush=True,
    )

    inspections: list[Inspection] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        iterator = executor.map(
            lambda item: inspect_candidate(item, by_class, by_object_name),
            candidates,
        )
        for index, inspection in enumerate(iterator, 1):
            inspections.append(inspection)
            if index % 1000 == 0 or index == len(candidates):
                print(f"inspected {index}/{len(candidates)}", flush=True)

    errors = [item for item in inspections if item.error]
    unmapped_locations: dict[str, list[str]] = defaultdict(list)
    for item in inspections:
        for identifier in sorted(item.unmapped or ()):
            if len(unmapped_locations[identifier]) < 20:
                unmapped_locations[identifier].append(str(item.candidate.conf_path))

    if errors:
        print(
            f"ERROR: {len(errors)} invalid/unreadable conf files; nothing written",
            file=sys.stderr,
        )
    if unmapped_locations:
        print(
            f"ERROR: {len(unmapped_locations)} unmapped identifiers; nothing written",
            file=sys.stderr,
        )

    existing_count = sum(item.candidate.output_path.exists() for item in inspections)
    writable = [
        item
        for item in inspections
        if OVERWRITE_EXISTING or not item.candidate.output_path.exists()
    ]
    written = 0
    if APPLY_CHANGES and not errors and not unmapped_locations:
        for item in writable:
            with item.candidate.conf_path.open("r", encoding="utf-8") as stream:
                conf_data = json.load(stream)
            scene_meta, unmapped, object_count, duplicate_count = build_scene_meta(
                conf_data, by_class, by_object_name
            )
            if unmapped or object_count != item.object_count:
                raise RuntimeError(
                    f"conf changed during preflight: {item.candidate.conf_path}"
                )
            if duplicate_count != item.duplicate_class_count:
                raise RuntimeError(
                    f"conf changed during preflight: {item.candidate.conf_path}"
                )
            atomic_write_json(
                item.candidate.output_path,
                scene_meta,
                item.candidate.conf_path.stat().st_mode,
            )
            written += 1
            if written % 1000 == 0:
                print(f"written {written}", flush=True)

    error_samples = [
        {"path": str(item.candidate.conf_path), "error": item.error}
        for item in errors[:100]
    ]
    report = {
        "mode": "APPLY" if APPLY_CHANGES else "DRY-RUN (DATASET NOT MODIFIED)",
        "dataset_root": str(DATASET_ROOT.resolve()),
        "mapping_csv": str(MAPPING_CSV.resolve()),
        "selected_environments": list(SELECTED_ENVIRONMENTS),
        "selected_platforms": list(SELECTED_PLATFORMS),
        "metadata_count": len(by_class),
        "platform_count": len(platforms),
        "conf_files": len(inspections),
        "object_references": sum(item.object_count for item in inspections),
        "duplicate_class_references": sum(
            item.duplicate_class_count for item in inspections
        ),
        "existing_scene_meta_files": existing_count,
        "would_write_files": len(writable),
        "written_files": written,
        "unmapped": dict(sorted(unmapped_locations.items())),
        "errors": error_samples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if REPORT_JSON is not None:
        REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
        REPORT_JSON.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    border = "=" * 78
    print(border)
    if APPLY_CHANGES:
        print(f"APPLY COMPLETE — written={written}")
    else:
        print(f"DRY-RUN COMPLETE — would_write={len(writable)}, written=0")
        print("DATASET WAS NOT MODIFIED")
    print(border, flush=True)

    return 2 if errors or unmapped_locations else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; no write occurs before preflight completes", file=sys.stderr)
        raise SystemExit(130)
