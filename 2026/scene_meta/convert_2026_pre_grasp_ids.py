#!/usr/bin/env python3
"""Convert only Dataset_2026 pre_grasp target_object labels.

Pose, joint, gripper, and grasp geometry fields are left unchanged. The
default is a read-only dry run; set ``APPLY_CHANGES = True`` below only after
checking the complete preflight report.
"""

from __future__ import annotations

import sys
from pathlib import Path

from convert_2026_object_ids import (
    ConversionConfig,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MAPPING_CSV,
    run_conversion,
)


# =============================================================================
# 사용자 설정: pre_grasp 전용
# =============================================================================
DATASET_ROOT = DEFAULT_DATASET_ROOT
MAPPING_CSV = DEFAULT_MAPPING_CSV

# 비어 있으면 environment 기준 필터를 사용하지 않습니다.
# 3대의 PC에서 각각 아래처럼 하나씩 지정해서 동시에 나눠 처리할 수 있습니다.
#   PC 1: ("Home",)
#   PC 2: ("Logistic_site",)
#   PC 3: ("Manufactory",)
SELECTED_ENVIRONMENTS: tuple[str, ...] = ("Home", "Logistic_site", "Manufactory")

# environment 전체가 아니라 특정 platform만 처리할 때 사용합니다.
# SELECTED_ENVIRONMENTS와 함께 지정하면 두 선택 범위를 합쳐서 처리합니다.
# 예: ("Home/MasterBedroom/bed_01",)
SELECTED_PLATFORMS: tuple[str, ...] = ()
WORKERS = 8
LIMIT_FILES: int | None = None

# False: 검사만 수행하며 파일을 절대 수정하지 않습니다.
# True : 전체 사전 검사가 성공한 경우에만 target_object를 실제로 수정합니다.
APPLY_CHANGES = True

# 실제 적용 전에 원본 JSON을 별도 위치에 보관하려면 경로를 지정합니다.
BACKUP_ROOT: Path | None = None
ALLOW_UNMAPPED = False
REPORT_JSON: Path | None = None


def resolve_selected_platforms() -> tuple[str, ...]:
    """Expand selected environment names to dataset-relative platform paths."""
    selected = set(SELECTED_PLATFORMS)
    dataset_root = DATASET_ROOT.resolve()

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

        platform_count = 0
        for section in environment.iterdir():
            if not section.is_dir():
                continue
            for platform in section.iterdir():
                if not platform.is_dir():
                    continue
                selected.add(str(platform.resolve().relative_to(dataset_root)))
                platform_count += 1
        if platform_count == 0:
            raise ValueError(f"no platforms found under environment: {environment}")

    return tuple(sorted(selected))


def main() -> int:
    selected_platforms = resolve_selected_platforms()
    return run_conversion(
        ConversionConfig(
            name="pre_grasp target_object",
            dataset_root=DATASET_ROOT,
            mapping_csv=MAPPING_CSV,
            selected_platforms=selected_platforms,
            folders=("pre_grasp",),
            workers=WORKERS,
            limit_files=LIMIT_FILES,
            apply_changes=APPLY_CHANGES,
            backup_root=BACKUP_ROOT,
            allow_unmapped=ALLOW_UNMAPPED,
            report_json=REPORT_JSON,
        )
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; no write occurs before preflight completes", file=sys.stderr)
        raise SystemExit(130)
