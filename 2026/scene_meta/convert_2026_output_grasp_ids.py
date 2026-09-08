#!/usr/bin/env python3
"""Convert only Dataset_2026 output_grasp target_object labels.

Run this after output_grasp collection has finished. It is a read-only dry run
unless ``APPLY_CHANGES = True`` is set below.
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
# 사용자 설정: output_grasp 전용
# =============================================================================
DATASET_ROOT = DEFAULT_DATASET_ROOT
MAPPING_CSV = DEFAULT_MAPPING_CSV
SELECTED_PLATFORMS: tuple[str, ...] = ()
WORKERS = 8
LIMIT_FILES: int | None = None

# 반드시 dry-run 결과를 확인한 다음 True로 변경하십시오.
APPLY_CHANGES = True
BACKUP_ROOT: Path | None = None
ALLOW_UNMAPPED = False
REPORT_JSON: Path | None = None


def main() -> int:
    return run_conversion(
        ConversionConfig(
            name="output_grasp target_object",
            dataset_root=DATASET_ROOT,
            mapping_csv=MAPPING_CSV,
            selected_platforms=SELECTED_PLATFORMS,
            folders=("output_grasp",),
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
