#!/usr/bin/env python3
"""Move selected Dataset 2026 data folders into separate dataset roots.

Example:
    dataset_v2/Home/MasterBedroom/bed_01/depth
        -> dataset_v2_depth/Home/MasterBedroom/bed_01/depth

There are no command-line arguments. Edit only the settings below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


# =============================================================================
# 사용자 설정
# =============================================================================

SOURCE_DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/dataset_v2_val")

# 한 종류만 옮길 때: ("depth",)
# 여러 종류를 옮길 때: ("rgb", "pcd", "inst_seg")
# pcd는 실제 폴더명인 pointcloud를 뜻합니다.
SELECTED_DATA_TYPES: tuple[str, ...] = ("pointcloud",)

# True: 이동할 경로만 출력하고 실제로 이동하지 않습니다.
# False: 실제로 폴더를 이동합니다.
DRY_RUN = False


# 선택 이름: (원본 platform 안의 폴더명, 새 dataset 폴더의 접미사)
# "pcd" 선택 시 pointcloud 폴더를 dataset_v2_pcd로 이동합니다.
# "pointcloud"를 선택하면 목적지 이름은 dataset_v2_pointcloud가 됩니다.
DATA_TYPE_CONFIG: dict[str, tuple[str, str]] = {
    "depth": ("depth", "depth"),
    "rgb": ("rgb", "rgb"),
    "inst_seg": ("inst_seg", "inst_seg"),
    "pcd": ("pointcloud", "pcd"),
    "pointcloud": ("pointcloud", "pointcloud"),
    "bbox": ("bbox", "bbox"),
    "normals": ("normals", "normals"),
    "conf": ("conf", "conf"),
    "scene_meta": ("scene_meta", "scene_meta"),
    "pre_grasp": ("pre_grasp", "pre_grasp"),
    "output_grasp": ("output_grasp", "output_grasp"),
}


@dataclass(frozen=True)
class MoveItem:
    data_type: str
    source: Path
    destination: Path


def validate_settings() -> Path:
    root = SOURCE_DATASET_ROOT.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"원본 dataset 폴더가 없습니다: {root}")
    if not SELECTED_DATA_TYPES:
        raise ValueError("SELECTED_DATA_TYPES가 비어 있습니다.")

    unknown = sorted(set(SELECTED_DATA_TYPES) - set(DATA_TYPE_CONFIG))
    if unknown:
        raise ValueError(
            f"지원하지 않는 데이터 종류입니다: {unknown}\n"
            f"사용 가능: {sorted(DATA_TYPE_CONFIG)}"
        )

    if len(SELECTED_DATA_TYPES) != len(set(SELECTED_DATA_TYPES)):
        raise ValueError("SELECTED_DATA_TYPES에 같은 항목이 중복되었습니다.")

    source_folders = [DATA_TYPE_CONFIG[name][0] for name in SELECTED_DATA_TYPES]
    if len(source_folders) != len(set(source_folders)):
        raise ValueError(
            "pcd와 pointcloud처럼 같은 원본 폴더를 가리키는 항목을 "
            "동시에 선택할 수 없습니다."
        )
    return root


def discover_moves(root: Path) -> list[MoveItem]:
    moves: list[MoveItem] = []
    for data_type in SELECTED_DATA_TYPES:
        source_folder, destination_suffix = DATA_TYPE_CONFIG[data_type]
        destination_root = root.parent / f"{root.name}_{destination_suffix}"

        # Dataset 구조는 root/environment/section/platform/data_folder 입니다.
        for source in sorted(root.glob(f"*/*/*/{source_folder}")):
            if not source.is_dir() or source.is_symlink():
                continue
            relative = source.relative_to(root)
            if len(relative.parts) != 4:
                continue
            moves.append(
                MoveItem(
                    data_type=data_type,
                    source=source,
                    destination=destination_root / relative,
                )
            )
    return moves


def preflight(root: Path, moves: list[MoveItem]) -> None:
    if not moves:
        selected = ", ".join(SELECTED_DATA_TYPES)
        raise RuntimeError(f"이동할 폴더를 찾지 못했습니다: {root} ({selected})")

    destinations = [item.destination for item in moves]
    if len(destinations) != len(set(destinations)):
        raise RuntimeError("동일한 목적지 경로가 중복되었습니다.")

    conflicts = [item.destination for item in moves if item.destination.exists()]
    if conflicts:
        preview = "\n".join(str(path) for path in conflicts[:20])
        remainder = len(conflicts) - 20
        if remainder > 0:
            preview += f"\n... 외 {remainder}개"
        raise FileExistsError(
            "목적지에 같은 데이터 폴더가 이미 있어 아무것도 이동하지 않습니다:\n"
            + preview
        )


def print_plan(root: Path, moves: list[MoveItem]) -> None:
    counts: dict[str, int] = {}
    for item in moves:
        counts[item.data_type] = counts.get(item.data_type, 0) + 1

    print("=" * 72)
    print("미리보기 모드 - 실제 이동 없음" if DRY_RUN else "실제 이동 모드")
    print(f"원본: {root}")
    for data_type, count in counts.items():
        _source_folder, suffix = DATA_TYPE_CONFIG[data_type]
        print(
            f"- {data_type}: {count}개 platform -> "
            f"{root.parent / f'{root.name}_{suffix}'}"
        )
    print(f"총 이동 대상: {len(moves)}개 폴더")
    print("=" * 72)
    for item in moves:
        print(f"{item.source}  ->  {item.destination}")


def move_all(moves: list[MoveItem]) -> None:
    completed: list[MoveItem] = []
    try:
        for index, item in enumerate(moves, start=1):
            item.destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item.source), str(item.destination))
            completed.append(item)
            print(f"[{index}/{len(moves)}] 이동 완료: {item.destination}")
    except Exception as error:
        print(f"이동 중 오류 발생: {type(error).__name__}: {error}")
        print("이번 실행에서 이미 이동한 폴더를 원래 위치로 되돌립니다.")
        rollback_errors: list[str] = []
        for item in reversed(completed):
            try:
                item.source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item.destination), str(item.source))
            except Exception as rollback_error:
                rollback_errors.append(
                    f"{item.destination} -> {item.source}: {rollback_error}"
                )
        if rollback_errors:
            raise RuntimeError(
                "이동과 자동 복구가 모두 실패했습니다:\n"
                + "\n".join(rollback_errors)
            ) from error
        raise RuntimeError("오류가 발생하여 이동한 폴더를 모두 복구했습니다.") from error


def main() -> None:
    root = validate_settings()
    moves = discover_moves(root)
    preflight(root, moves)
    print_plan(root, moves)

    if DRY_RUN:
        print("\nDRY_RUN=True이므로 실제 파일은 이동하지 않았습니다.")
        print("경로를 확인한 다음 DRY_RUN=False로 바꾸고 다시 실행하십시오.")
        return

    move_all(moves)
    print(f"\n완료: {len(moves)}개 폴더를 이동했습니다.")


if __name__ == "__main__":
    main()
