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


def main() -> int:
    return run_conversion(
        ConversionConfig(
            name="pre_grasp target_object",
            dataset_root=DATASET_ROOT,
            mapping_csv=MAPPING_CSV,
            selected_platforms=SELECTED_PLATFORMS,
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
