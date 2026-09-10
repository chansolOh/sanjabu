"""Count every dataset modality and report missing/orphan scene identifiers."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import config
from evaluation_core import (
    atomic_write_json,
    discover_platforms,
    run_metadata,
    write_csv,
)
from file_specs import FILE_SPECS, FileSpec


def scan_platform(platform_path: Path) -> tuple[dict[str, set[int]], list[str], list[str]]:
    by_directory: dict[Path, list[FileSpec]] = defaultdict(list)
    for spec in FILE_SPECS:
        by_directory[spec.relative_dir].append(spec)

    identifiers = {spec.key: set() for spec in FILE_SPECS}
    invalid_files: list[str] = []
    missing_directories: list[str] = []
    for relative_dir, directory_specs in by_directory.items():
        directory = platform_path / relative_dir
        if not directory.is_dir():
            missing_directories.append(str(relative_dir))
            continue
        for entry in os.scandir(directory):
            if not entry.is_file():
                continue
            matched = False
            for spec in directory_specs:
                scene_id = spec.scene_id(entry.name)
                if scene_id is not None:
                    identifiers[spec.key].add(scene_id)
                    matched = True
            if not matched and not entry.name.startswith("."):
                invalid_files.append(str(relative_dir / entry.name))
    return identifiers, invalid_files, missing_directories


def evaluate_counts() -> dict[str, Any]:
    result: dict[str, Any] = {
        "metadata": run_metadata("count_data.py"),
        "definition": {
            "anchor": "conf scene identifiers",
            "missing": "conf id exists but modality id does not",
            "orphan": "modality id exists but conf id does not",
            "complete_scene": "all split-required modalities exist for a conf id",
        },
        "splits": {},
    }
    platform_rows: list[dict[str, Any]] = []
    modality_rows: list[dict[str, Any]] = []

    for split, settings in config.DATASETS.items():
        root = Path(settings["root"])
        split_modality_counts: dict[str, int] = defaultdict(int)
        split_missing_counts: dict[str, int] = defaultdict(int)
        split_orphan_counts: dict[str, int] = defaultdict(int)
        split_conf_count = 0
        split_complete_count = 0
        split_platforms: list[dict[str, Any]] = []

        platforms = discover_platforms(split, root)
        for platform_number, platform in enumerate(platforms, start=1):
            print(
                f"[{split}] {platform_number}/{len(platforms)} "
                f"{platform.relative_path}"
            )
            ids, invalid_files, missing_directories = scan_platform(platform.path)
            conf_ids = ids["conf"]
            required_specs = [spec for spec in FILE_SPECS if split in spec.required_in]
            complete_ids = set(conf_ids)
            modalities: dict[str, Any] = {}

            for spec in FILE_SPECS:
                modality_ids = ids[spec.key]
                missing_ids = sorted(conf_ids - modality_ids)
                orphan_ids = sorted(modality_ids - conf_ids)
                if spec in required_specs:
                    complete_ids &= modality_ids
                split_modality_counts[spec.key] += len(modality_ids)
                split_missing_counts[spec.key] += len(missing_ids)
                split_orphan_counts[spec.key] += len(orphan_ids)
                modalities[spec.key] = {
                    "required": split in spec.required_in,
                    "count": len(modality_ids),
                    "missing_count": len(missing_ids),
                    "missing_examples": missing_ids[: config.MAX_ISSUE_EXAMPLES],
                    "orphan_count": len(orphan_ids),
                    "orphan_examples": orphan_ids[: config.MAX_ISSUE_EXAMPLES],
                }
                modality_rows.append(
                    {
                        "split": split,
                        "env": platform.env,
                        "section": platform.section,
                        "platform": platform.platform,
                        "modality": spec.key,
                        "required": split in spec.required_in,
                        "count": len(modality_ids),
                        "missing_count": len(missing_ids),
                        "orphan_count": len(orphan_ids),
                    }
                )

            if conf_ids:
                scene_min = min(conf_ids)
                scene_max = max(conf_ids)
                range_gaps = sorted(set(range(scene_min, scene_max + 1)) - conf_ids)
            else:
                scene_min = scene_max = None
                range_gaps = []

            required_directories = {str(spec.relative_dir) for spec in required_specs}
            missing_required_directories = sorted(
                directory
                for directory in set(missing_directories)
                if directory in required_directories
            )
            optional_absent_directories = sorted(
                directory
                for directory in set(missing_directories)
                if directory not in required_directories
            )
            platform_result = {
                "env": platform.env,
                "section": platform.section,
                "platform": platform.platform,
                "conf_count": len(conf_ids),
                "scene_min": scene_min,
                "scene_max": scene_max,
                "range_gap_count": len(range_gaps),
                "range_gap_examples": range_gaps[: config.MAX_ISSUE_EXAMPLES],
                "complete_scene_count": len(complete_ids),
                "complete_scene_rate": (
                    len(complete_ids) / len(conf_ids) if conf_ids else 0.0
                ),
                "invalid_file_count": len(invalid_files),
                "invalid_file_examples": invalid_files[: config.MAX_ISSUE_EXAMPLES],
                "missing_required_directories": missing_required_directories,
                "optional_absent_directories": optional_absent_directories,
                "modalities": modalities,
            }
            split_platforms.append(platform_result)
            split_conf_count += len(conf_ids)
            split_complete_count += len(complete_ids)
            platform_rows.append(
                {
                    "split": split,
                    "env": platform.env,
                    "section": platform.section,
                    "platform": platform.platform,
                    "conf_count": len(conf_ids),
                    "scene_min": scene_min,
                    "scene_max": scene_max,
                    "range_gap_count": len(range_gaps),
                    "complete_scene_count": len(complete_ids),
                    "complete_scene_rate": (
                        len(complete_ids) / len(conf_ids) if conf_ids else 0.0
                    ),
                    "invalid_file_count": len(invalid_files),
                }
            )

        result["splits"][split] = {
            "root": str(root),
            "platform_count": len(platforms),
            "conf_scene_count": split_conf_count,
            "complete_scene_count": split_complete_count,
            "complete_scene_rate": (
                split_complete_count / split_conf_count if split_conf_count else 0.0
            ),
            "modality_counts": dict(sorted(split_modality_counts.items())),
            "missing_counts": dict(sorted(split_missing_counts.items())),
            "orphan_counts": dict(sorted(split_orphan_counts.items())),
            "platforms": split_platforms,
        }

    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config.RESULT_DIR / "data_count.json", result)
    write_csv(config.RESULT_DIR / "data_count_platforms.csv", platform_rows)
    write_csv(config.RESULT_DIR / "data_count_modalities.csv", modality_rows)
    return result


def main() -> None:
    result = evaluate_counts()
    print("\n=== DATA COUNT SUMMARY ===")
    for split, data in result["splits"].items():
        print(
            f"{split:10s} conf={data['conf_scene_count']:,} "
            f"complete={data['complete_scene_count']:,} "
            f"({data['complete_scene_rate'] * 100:.4f}%)"
        )
    print(f"saved: {config.RESULT_DIR}")


if __name__ == "__main__":
    main()
