#!/usr/bin/env python3
"""Convert completed Dataset_2026 labels to obj_### identifiers.

This main converter intentionally processes only metadata that is ready now:

* conf/objects[].class (never objects[].usd_path)
* bbox JSON object keys
* inst_seg semantics_mapping JSON class values

pre_grasp, output_grasp, and scene_meta have separate entry-point scripts
because they are converted at different times. Images and numeric arrays are
never modified. Pointcloud instance IDs are numeric and contain no object-name
label.

The default mode is a read-only dry run. Set ``APPLY_CHANGES = True`` in the
user-settings section to write changes.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/dataset_v2_test")
DEFAULT_MAPPING_CSV = Path(
    "/nas/ochansol/3d_model/peel3_scan_data_2026/2026_objects_cat_attr.csv"
)
OBJ_ID_RE = re.compile(r"obj_\d{3,}")
RESERVED_SEMANTIC_LABELS = {"BACKGROUND", "UNLABELLED"}

FOLDER_PATTERNS = {
    "conf": ("conf/*.json",),
    "bbox": ("bbox/*/*.json",),
    "inst_seg": ("inst_seg/*/semantics_mapping_*.json",),
    "pre_grasp": ("pre_grasp/*.json",),
    "output_grasp": ("output_grasp/*.json",),
    "scene_meta": ("scene_meta/*.json",),
}
CORE_FOLDERS = ("conf", "bbox", "inst_seg")


# =============================================================================
# 사용자 설정: 기본 변환(conf.class, bbox key, inst_seg class)
# =============================================================================
DATASET_ROOT = DEFAULT_DATASET_ROOT
MAPPING_CSV = DEFAULT_MAPPING_CSV

# 비어 있으면 전체 platform을 처리합니다.
# 예: ("Home/MasterBedroom/bed_01", "Home/MasterBedroom/vanity_01")
SELECTED_PLATFORMS: tuple[str, ...] = ()
SELECTED_FOLDERS = CORE_FOLDERS
WORKERS = 8

# 빠른 테스트 시 정수로 지정하고, 전체 검사 시 None으로 둡니다.
LIMIT_FILES: int | None = None

# 반드시 dry-run 결과를 확인한 다음 True로 변경하십시오.
APPLY_CHANGES = True

# 실제 적용 전 원본 JSON을 백업하려면 경로를 지정합니다.
BACKUP_ROOT: Path | None = None

# False이면 CSV에 없는 물체명이 하나라도 있을 때 실제 적용하지 않습니다.
ALLOW_UNMAPPED = False

# 실행 보고서를 별도 파일로 남기려면 경로를 지정합니다.
REPORT_JSON: Path | None = None


@dataclass(frozen=True)
class ConversionConfig:
    name: str
    dataset_root: Path
    mapping_csv: Path
    selected_platforms: tuple[str, ...]
    folders: tuple[str, ...]
    workers: int
    limit_files: int | None
    apply_changes: bool
    backup_root: Path | None
    allow_unmapped: bool
    report_json: Path | None


@dataclass(frozen=True)
class Candidate:
    path: Path
    platform: Path
    folder: str


@dataclass
class Inspection:
    candidate: Candidate
    replacements: Counter
    unmapped: set[str]
    error: str | None = None

    @property
    def changed(self) -> bool:
        return bool(self.replacements) and self.error is None


def load_mapping(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"mapping CSV not found: {path}")

    mapping: dict[str, str] = {}
    target_to_source: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Object_name", "Class_name"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns {sorted(required)}: {path}")

        for line_number, row in enumerate(reader, 2):
            source = (row.get("Object_name") or "").strip()
            target = (row.get("Class_name") or "").strip()

            # The supplied CSV contains blank/footer statistics rows. They are
            # not object records and are intentionally ignored.
            if not source:
                continue
            if not OBJ_ID_RE.fullmatch(target):
                raise ValueError(
                    f"invalid Class_name at CSV line {line_number}: {source!r} -> {target!r}"
                )
            # Dataset generators did not use consistent case for units in two
            # names (30kg/30KG and 20g/20G). Object identifiers are therefore
            # matched case-insensitively while all other characters must still
            # match exactly.
            normalized_source = source.casefold()
            if normalized_source in mapping and mapping[normalized_source] != target:
                raise ValueError(
                    "case-insensitive duplicate Object_name with different IDs: "
                    f"{source}"
                )
            if target in target_to_source and target_to_source[target] != source:
                raise ValueError(
                    f"duplicate Class_name {target}: {target_to_source[target]} and {source}"
                )
            mapping[normalized_source] = target
            target_to_source[target] = source

    if not mapping:
        raise ValueError(f"no valid mappings found in {path}")
    return mapping


def discover_platforms(dataset_root: Path, selected: list[Path] | None) -> list[Path]:
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"dataset root not found: {dataset_root}")

    if selected:
        platforms = []
        for value in selected:
            path = value if value.is_absolute() else dataset_root / value
            path = path.resolve()
            if not path.is_dir():
                raise NotADirectoryError(f"platform directory not found: {path}")
            if dataset_root != path and dataset_root not in path.parents:
                raise ValueError(f"platform is outside dataset root: {path}")
            platforms.append(path)
        return sorted(set(platforms))

    platforms = []
    for environment in dataset_root.iterdir():
        if not environment.is_dir() or environment.name == "pregrasp_statistics":
            continue
        for section in environment.iterdir():
            if not section.is_dir():
                continue
            platforms.extend(path for path in section.iterdir() if path.is_dir())
    return sorted(platforms)


def discover_candidates(
    platforms: Iterable[Path], folders: Iterable[str], limit_files: int | None
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for platform in platforms:
        for folder in folders:
            for pattern in FOLDER_PATTERNS[folder]:
                candidates.extend(
                    Candidate(path, platform, folder)
                    for path in platform.glob(pattern)
                    if path.is_file()
                )
    candidates.sort(key=lambda item: str(item.path))

    # Defensive de-duplication if future patterns overlap.
    unique: dict[Path, Candidate] = {candidate.path: candidate for candidate in candidates}
    candidates = [unique[path] for path in sorted(unique, key=str)]
    if limit_files is not None:
        if limit_files < 0:
            raise ValueError("LIMIT_FILES must be >= 0")
        candidates = candidates[:limit_files]
    return candidates


def mapped_identifier(
    value: Any,
    mapping: dict[str, str],
    unmapped: set[str],
    *,
    allow_reserved: bool = False,
) -> Any:
    if not isinstance(value, str):
        return value
    normalized_value = value.casefold()
    if normalized_value in mapping:
        return mapping[normalized_value]
    if OBJ_ID_RE.fullmatch(value):
        return value
    if allow_reserved and value in RESERVED_SEMANTIC_LABELS:
        return value
    if value:
        unmapped.add(value)
    return value


def replace_target_objects(node: Any, mapping: dict[str, str], unmapped: set[str]) -> int:
    replacements = 0
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "target_object":
                converted = mapped_identifier(value, mapping, unmapped)
                replacements += converted != value
                node[key] = converted
            else:
                replacements += replace_target_objects(value, mapping, unmapped)
    elif isinstance(node, list):
        for value in node:
            replacements += replace_target_objects(value, mapping, unmapped)
    return replacements


def transform_document(
    data: Any,
    folder: str,
    mapping: dict[str, str],
) -> tuple[Any, Counter, set[str]]:
    replacements: Counter = Counter()
    unmapped: set[str] = set()

    def record(kind: str, before: Any, after: Any) -> Any:
        if before != after:
            replacements[kind] += 1
        return after

    if folder == "conf":
        if not isinstance(data, dict) or not isinstance(data.get("objects"), list):
            raise ValueError("conf JSON must contain an objects list")
        for item in data["objects"]:
            if not isinstance(item, dict):
                raise ValueError("conf objects entries must be objects")
            before = item.get("class")
            after = mapped_identifier(before, mapping, unmapped)
            item["class"] = record("conf.class", before, after)

    elif folder == "bbox":
        if not isinstance(data, dict):
            raise ValueError("bbox JSON root must be an object")
        converted = {}
        for key, value in data.items():
            new_key = mapped_identifier(key, mapping, unmapped)
            if new_key in converted:
                raise ValueError(f"bbox key collision after mapping: {key} -> {new_key}")
            converted[new_key] = value
            record("bbox.key", key, new_key)
        data = converted

    elif folder == "inst_seg":
        if not isinstance(data, dict):
            raise ValueError("semantics mapping JSON root must be an object")
        for color, semantic in data.items():
            if not isinstance(semantic, dict) or "class" not in semantic:
                continue
            before = semantic["class"]
            after = mapped_identifier(before, mapping, unmapped, allow_reserved=True)
            semantic["class"] = record("inst_seg.class", before, after)

    elif folder in {"pre_grasp", "output_grasp"}:
        count = replace_target_objects(data, mapping, unmapped)
        if count:
            replacements[f"{folder}.target_object"] += count

    elif folder == "scene_meta":
        if not isinstance(data, dict):
            raise ValueError("scene_meta JSON root must be an object")
        objects = data.get("objects")
        if objects is not None:
            if not isinstance(objects, dict):
                raise ValueError("scene_meta objects must be an object")
            converted = {}
            for key, value in objects.items():
                new_key = mapped_identifier(key, mapping, unmapped)
                if new_key in converted:
                    raise ValueError(f"scene_meta key collision after mapping: {key} -> {new_key}")
                converted[new_key] = value
                record("scene_meta.objects_key", key, new_key)
            data["objects"] = converted
            # Deliberately retain objects.<id>.object_name as human-readable
            # metadata, matching Dataset_2025.
    else:
        raise ValueError(f"unsupported metadata folder: {folder}")

    return data, replacements, unmapped


def read_and_transform(
    candidate: Candidate,
    mapping: dict[str, str],
) -> tuple[Any, Counter, set[str]]:
    with candidate.path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    return transform_document(data, candidate.folder, mapping)


def inspect_candidate(
    candidate: Candidate,
    mapping: dict[str, str],
) -> Inspection:
    try:
        _data, replacements, unmapped = read_and_transform(candidate, mapping)
        return Inspection(candidate, replacements, unmapped)
    except Exception as error:
        return Inspection(candidate, Counter(), set(), f"{type(error).__name__}: {error}")


def atomic_write_json(path: Path, data: Any) -> None:
    mode = path.stat().st_mode
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=4)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def backup_file(path: Path, dataset_root: Path, backup_root: Path) -> None:
    relative = path.resolve().relative_to(dataset_root.resolve())
    destination = backup_root.resolve() / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"backup already exists: {destination}")
    shutil.copy2(path, destination)


def build_report(
    config: ConversionConfig,
    mapping: dict[str, str],
    platforms: list[Path],
    inspections: list[Inspection],
    written: int,
) -> dict[str, Any]:
    replacement_totals: Counter = Counter()
    folder_files: Counter = Counter()
    changed_by_folder: Counter = Counter()
    unmapped_locations: dict[str, list[str]] = defaultdict(list)
    errors = []
    for result in inspections:
        folder_files[result.candidate.folder] += 1
        if result.changed:
            changed_by_folder[result.candidate.folder] += 1
        replacement_totals.update(result.replacements)
        for name in sorted(result.unmapped):
            if len(unmapped_locations[name]) < 20:
                unmapped_locations[name].append(str(result.candidate.path))
        if result.error:
            errors.append({"path": str(result.candidate.path), "error": result.error})

    return {
        "mode": "APPLY" if config.apply_changes else "DRY-RUN (DATASET NOT MODIFIED)",
        "conversion": config.name,
        "dataset_root": str(config.dataset_root.resolve()),
        "mapping_csv": str(config.mapping_csv.resolve()),
        "selected_platforms": list(config.selected_platforms),
        "folders": list(config.folders),
        "mapping_count": len(mapping),
        "platform_count": len(platforms),
        "candidate_files": len(inspections),
        "changed_files": sum(result.changed for result in inspections),
        "written_files": written,
        "files_by_folder": dict(sorted(folder_files.items())),
        "changed_files_by_folder": dict(sorted(changed_by_folder.items())),
        "replacements": dict(sorted(replacement_totals.items())),
        "unmapped": dict(sorted(unmapped_locations.items())),
        "errors": errors,
        "notes": [
            "conf objects[].usd_path is never modified.",
            (
                "pre_grasp target_object values are included in this conversion."
                if "pre_grasp" in config.folders
                else "pre_grasp files are not included in this conversion."
            ),
            "Pointcloud, PNG, and NPY files are never modified.",
            "scene_meta objects.<id>.object_name is descriptive metadata and was retained.",
        ],
}


def print_mode_banner(config: ConversionConfig) -> None:
    border = "=" * 78
    print(border)
    if config.apply_changes:
        print("APPLY MODE — DATASET JSON FILES WILL BE MODIFIED")
    else:
        print("DRY-RUN MODE — DATASET JSON FILES WILL NOT BE MODIFIED")
        print("변경하려면 파일 상단의 APPLY_CHANGES = True 로 설정하십시오.")
    print(f"conversion: {config.name}")
    print(f"folders: {', '.join(config.folders)}")
    print(border, flush=True)


def run_conversion(config: ConversionConfig) -> int:
    if config.workers < 1:
        raise ValueError("WORKERS must be >= 1")
    if config.backup_root and not config.apply_changes:
        raise ValueError("BACKUP_ROOT requires APPLY_CHANGES = True")
    invalid_folders = set(config.folders) - set(FOLDER_PATTERNS)
    if invalid_folders:
        raise ValueError(f"invalid folders: {sorted(invalid_folders)}")

    mapping = load_mapping(config.mapping_csv)
    selected = [Path(value) for value in config.selected_platforms] or None
    platforms = discover_platforms(config.dataset_root, selected)
    candidates = discover_candidates(platforms, config.folders, config.limit_files)
    print_mode_banner(config)
    print(
        f"mappings={len(mapping)} platforms={len(platforms)} "
        f"candidates={len(candidates)}"
    )

    inspections: list[Inspection] = []
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        for index, result in enumerate(
            executor.map(lambda item: inspect_candidate(item, mapping), candidates),
            1,
        ):
            inspections.append(result)
            if index % 1000 == 0 or index == len(candidates):
                changed = sum(item.changed for item in inspections)
                print(f"inspected {index}/{len(candidates)} changed={changed}", flush=True)

    errors = [result for result in inspections if result.error]
    unmapped = {name for result in inspections for name in result.unmapped}
    if errors:
        print(f"ERROR: {len(errors)} invalid/unreadable JSON files; no files written", file=sys.stderr)
    if unmapped and not config.allow_unmapped:
        print(
            f"ERROR: {len(unmapped)} unmapped identifiers; no dataset files written. "
            "Fix the CSV or set REPORT_JSON to inspect locations.",
            file=sys.stderr,
        )

    can_apply = (
        config.apply_changes
        and not errors
        and (not unmapped or config.allow_unmapped)
    )
    written = 0
    if can_apply:
        for index, result in enumerate(inspections, 1):
            if not result.changed:
                continue
            data, replacements, _unmapped = read_and_transform(result.candidate, mapping)
            if replacements != result.replacements:
                raise RuntimeError(f"file changed during preflight: {result.candidate.path}")
            if config.backup_root:
                backup_file(
                    result.candidate.path,
                    config.dataset_root,
                    config.backup_root,
                )
            atomic_write_json(result.candidate.path, data)
            written += 1
            if written % 1000 == 0:
                print(f"written {written}", flush=True)

    report = build_report(config, mapping, platforms, inspections, written)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if config.report_json:
        config.report_json.parent.mkdir(parents=True, exist_ok=True)
        config.report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    changed = report["changed_files"]
    border = "=" * 78
    print(border)
    if config.apply_changes:
        print(f"APPLY COMPLETE — changed={changed}, written={written}")
    else:
        print(f"DRY-RUN COMPLETE — would_change={changed}, written=0")
        print("DATASET JSON FILES WERE NOT MODIFIED")
    print(border, flush=True)

    if errors or (unmapped and not config.allow_unmapped):
        return 2
    return 0


def main() -> int:
    config = ConversionConfig(
        name="core object labels",
        dataset_root=DATASET_ROOT,
        mapping_csv=MAPPING_CSV,
        selected_platforms=SELECTED_PLATFORMS,
        folders=SELECTED_FOLDERS,
        workers=WORKERS,
        limit_files=LIMIT_FILES,
        apply_changes=APPLY_CHANGES,
        backup_root=BACKUP_ROOT,
        allow_unmapped=ALLOW_UNMAPPED,
        report_json=REPORT_JSON,
    )
    return run_conversion(config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; no write occurs before preflight completes", file=sys.stderr)
        raise SystemExit(130)
