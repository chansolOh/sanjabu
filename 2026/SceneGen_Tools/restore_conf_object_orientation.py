"""dataset_v2의 모든 SceneGen conf에 객체 USD root의 X축 회전을 합성한다.

Scene generator가 저장한 ``objects[].orient``는 Replicator wrapper의
quaternion(wxyz)이고, 객체 USD root의 고정 X +90도 회전은 포함하지 않는다.
최종 mesh pose를 직접 복원할 때는 아래 순서가 되어야 한다.

    mesh --X +90 deg--> asset frame --saved orientation--> world frame

따라서 합성 quaternion은 ``q_corrected = q_saved * q_x90`` 이다.
``envs``와 ``platform`` 항목은 수정하지 않는다.
"""

import json
import math
import os
import shutil
from pathlib import Path


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/dataset_v2")

# conf 폴더를 오염시키지 않도록 데이터셋 밖에 동일한 디렉터리 구조로 백업한다.
BACKUP_ROOT = DATASET_ROOT.with_name(f"{DATASET_ROOT.name}_before_orientation_restore")

# 이전 단일 파일 실험에서 사용한 백업 이름. 발견하면 최초 원본으로 재사용한다.
LEGACY_BACKUP_SUFFIX = "_before_orientation_restore"

ASSET_ROOT_ROTATE_X_DEG = 90.0


def normalize_quaternion_wxyz(quaternion):
    if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
        raise ValueError(f"wxyz quaternion 4개 값이 필요합니다: {quaternion!r}")

    values = [float(value) for value in quaternion]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("길이가 0인 quaternion은 보정할 수 없습니다.")
    return [value / norm for value in values]


def multiply_quaternion_wxyz(left, right):
    """Hamilton product: left * right (right 회전이 먼저 적용됨)."""
    lw, lx, ly, lz = normalize_quaternion_wxyz(left)
    rw, rx, ry, rz = normalize_quaternion_wxyz(right)
    result = [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]
    return normalize_quaternion_wxyz(result)


def x_axis_quaternion_wxyz(degrees):
    half_angle = math.radians(float(degrees)) * 0.5
    return [math.cos(half_angle), math.sin(half_angle), 0.0, 0.0]


def correct_object_orientations(data):
    objects = data.get("objects")
    if not isinstance(objects, list):
        raise ValueError("conf JSON에 objects 리스트가 없습니다.")

    asset_rotation = x_axis_quaternion_wxyz(ASSET_ROOT_ROTATE_X_DEG)
    changes = []
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict) or "orient" not in obj:
            raise ValueError(f"objects[{index}]에 orient가 없습니다.")

        saved_orientation = normalize_quaternion_wxyz(obj["orient"])
        corrected_orientation = multiply_quaternion_wxyz(
            saved_orientation,
            asset_rotation,
        )
        obj["orient"] = corrected_orientation
        changes.append(
            {
                "class": obj.get("class", f"objects[{index}]"),
                "before": saved_orientation,
                "after": corrected_orientation,
            }
        )
    return changes


def find_conf_paths():
    """모든 <경로>/conf/*.json을 찾고 기존 백업 파일은 제외한다."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"데이터셋 폴더가 없습니다: {DATASET_ROOT}")

    return sorted(
        path
        for path in DATASET_ROOT.rglob("*.json")
        if path.parent.name == "conf"
        and not path.stem.endswith(LEGACY_BACKUP_SUFFIX)
    )


def backup_path_for(conf_path):
    return BACKUP_ROOT / conf_path.relative_to(DATASET_ROOT)


def legacy_backup_path_for(conf_path):
    return conf_path.with_name(f"{conf_path.stem}{LEGACY_BACKUP_SUFFIX}.json")


def prepare_backup(conf_path):
    """최초 원본을 별도 backup root에 한 번만 보관한다."""
    backup_path = backup_path_for(conf_path)
    if backup_path.exists():
        return backup_path, False

    backup_path.parent.mkdir(parents=True, exist_ok=True)

    # 기존 단일 파일 실험본은 이미 수정되었으므로 옆에 둔 원본 백업을 우선한다.
    legacy_backup_path = legacy_backup_path_for(conf_path)
    source_path = legacy_backup_path if legacy_backup_path.is_file() else conf_path
    shutil.copy2(source_path, backup_path)
    return backup_path, True


def process_conf(conf_path):
    backup_path, backup_created = prepare_backup(conf_path)
    if backup_created:
        print(f"BACKUP: {backup_path}")

    # 항상 최초 백업에서 읽기 때문에 반복 실행해도 +90도가 누적되지 않는다.
    with backup_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    changes = correct_object_orientations(data)

    # 백업에서 계산한 기대 결과와 현재 파일이 같으면 이미 변환된 파일이다.
    # 백업만 생성되고 변환에 실패했던 파일은 일치하지 않으므로 다시 시도한다.
    try:
        with conf_path.open("r", encoding="utf-8") as stream:
            current_data = json.load(stream)
    except (OSError, json.JSONDecodeError):
        current_data = None

    if current_data == data:
        return "skipped", len(changes)

    temporary_path = conf_path.with_name(f".{conf_path.name}.orientation_restore.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=4, ensure_ascii=False)
        stream.write("\n")
    shutil.copymode(backup_path, temporary_path)
    os.replace(temporary_path, conf_path)

    print(f"UPDATED: {conf_path} (objects={len(changes)})")
    return "updated", len(changes)


def main():
    conf_paths = find_conf_paths()
    if not conf_paths:
        print(f"처리할 conf JSON이 없습니다: {DATASET_ROOT}")
        return 1

    updated_files = 0
    updated_objects = 0
    skipped_files = 0
    failed_files = []

    print(f"CONF FILES: {len(conf_paths)}")
    print(f"BACKUP ROOT: {BACKUP_ROOT}")
    for conf_path in conf_paths:
        try:
            status, object_count = process_conf(conf_path)
            if status == "skipped":
                skipped_files += 1
            else:
                updated_files += 1
                updated_objects += object_count
        except Exception as exc:
            failed_files.append(
                (conf_path, f"{type(exc).__name__}: {exc}")
            )

    print(
        f"DONE: updated_files={updated_files}, "
        f"updated_objects={updated_objects}, skipped_files={skipped_files}, "
        f"failed_files={len(failed_files)}"
    )
    if failed_files:
        print("FAILED FILES:")
        for conf_path, reason in failed_files:
            print(f"  - {conf_path}")
            print(f"    reason: {reason}")

    return 1 if failed_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
