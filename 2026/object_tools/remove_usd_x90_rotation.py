"""Peel3 edited USD의 object root X +90도 회전을 제거한다.

처리 대상:
    <ROOT_PATH>/<object_name>/edited/*.usd|*.usda|*.usdc

각 USD의 ``/World`` 직계 자식 prim에서 다음을 수행한다.

1. 유효 ``xformOp:orient``가 X +90도이면 identity quaternion으로 변경
2. ``xformOp:rotateXYZ``의 X 값이 +90도이면 X만 0도로 변경
3. ``xformOp:rotateX``가 +90도이면 0도로 변경
4. Rotate ``unitResolve``/``unitsResolve`` 속성과 xformOpOrder 토큰 제거

Isaac Sim 5.1 uv 환경에서 실행한다. CLI argument는 사용하지 않으며 아래
User settings 변수로 경로와 동작을 설정한다.
"""

import math
import os
import shutil
from pathlib import Path


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
ROOT_PATH = Path("/nas/ochansol/3d_model/peel3_scan_data_2026")

TARGET_X_DEG = 90.0
ANGLE_TOLERANCE_DEG = 0.01

USD_SUFFIXES = {".usd", ".usda", ".usdc"}
UNITS_RESOLVE_SUFFIXES = (":unitResolve", ":unitsResolve")


def find_usd_paths():
    if not ROOT_PATH.is_dir():
        raise FileNotFoundError(f"데이터셋 폴더가 없습니다: {ROOT_PATH}")

    return sorted(
        path
        for path in ROOT_PATH.glob("*/edited/*")
        if path.is_file() and path.suffix.lower() in USD_SUFFIXES
        and ".x90_cleanup.tmp" not in path.name
    )


def angle_is_target(value, target=TARGET_X_DEG):
    """360도 주기를 고려해 value가 target 회전인지 확인한다."""
    delta = (float(value) - float(target) + 180.0) % 360.0 - 180.0
    return abs(delta) <= ANGLE_TOLERANCE_DEG


def quaternion_is_x90(value):
    """부호가 반대인 동일 quaternion까지 X +90도로 판정한다."""
    if value is None:
        return False

    imaginary = value.GetImaginary()
    quaternion = [
        float(value.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    ]
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm <= 1e-12:
        return False
    quaternion = [component / norm for component in quaternion]

    half_angle = math.radians(TARGET_X_DEG) * 0.5
    target = [math.cos(half_angle), math.sin(half_angle), 0.0, 0.0]
    dot = abs(sum(a * b for a, b in zip(quaternion, target)))
    rotation_error_deg = math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))
    return rotation_error_deg <= ANGLE_TOLERANCE_DEG


def remove_ordered_property(prim, property_name):
    """xformOpOrder에서 토큰을 제거한 뒤 해당 USD property를 제거한다."""
    order_attr = prim.GetAttribute("xformOpOrder")
    if order_attr.IsValid():
        old_order = list(order_attr.Get() or [])
        invert_name = f"!invert!{property_name}"
        new_order = [
            token
            for token in old_order
            if str(token) not in {property_name, invert_name}
        ]
        if new_order != old_order:
            order_attr.Set(new_order)

    return prim.RemoveProperty(property_name)


def find_rotate_units_resolve_attrs(prim):
    return [
        attr.GetName()
        for attr in prim.GetAttributes()
        if attr.GetName().startswith("xformOp:rotate")
        and attr.GetName().endswith(UNITS_RESOLVE_SUFFIXES)
    ]


def clean_world_child(prim):
    """하나의 /World 직계 자식 prim을 정리하고 변경 내역을 반환한다."""
    changes = []

    orient_attr = prim.GetAttribute("xformOp:orient")
    if orient_attr.IsValid() and quaternion_is_x90(orient_attr.Get()):
        current = orient_attr.Get()
        orient_attr.Set(type(current).GetIdentity())
        changes.append("xformOp:orient X+90 -> identity")

    rotate_xyz_attr = prim.GetAttribute("xformOp:rotateXYZ")
    if rotate_xyz_attr.IsValid():
        rotation = rotate_xyz_attr.Get()
        if rotation is not None and angle_is_target(rotation[0]):
            rotate_xyz_attr.Set(type(rotation)(0.0, float(rotation[1]), float(rotation[2])))
            changes.append("xformOp:rotateXYZ.x +90 -> 0")

    rotate_x_attr = prim.GetAttribute("xformOp:rotateX")
    if rotate_x_attr.IsValid():
        rotation_x = rotate_x_attr.Get()
        if rotation_x is not None and angle_is_target(rotation_x):
            rotate_x_attr.Set(0.0)
            changes.append("xformOp:rotateX +90 -> 0")

    for attr_name in find_rotate_units_resolve_attrs(prim):
        remove_ordered_property(prim, attr_name)
        if prim.HasAttribute(attr_name):
            raise RuntimeError(
                f"{prim.GetPath()}: {attr_name}가 현재 root layer에서 제거되지 않았습니다."
            )
        changes.append(f"removed {attr_name}")

    return changes


def validate_world_children(stage):
    world = stage.GetPrimAtPath("/World")
    if not world.IsValid():
        raise RuntimeError("/World prim이 없습니다.")

    for prim in world.GetChildren():
        orient_attr = prim.GetAttribute("xformOp:orient")
        if orient_attr.IsValid() and quaternion_is_x90(orient_attr.Get()):
            raise RuntimeError(f"{prim.GetPath()}: X +90 orient가 남아 있습니다.")

        rotate_xyz_attr = prim.GetAttribute("xformOp:rotateXYZ")
        if rotate_xyz_attr.IsValid() and rotate_xyz_attr.Get() is not None:
            if angle_is_target(rotate_xyz_attr.Get()[0]):
                raise RuntimeError(f"{prim.GetPath()}: rotateXYZ.x +90이 남아 있습니다.")

        rotate_x_attr = prim.GetAttribute("xformOp:rotateX")
        if rotate_x_attr.IsValid() and rotate_x_attr.Get() is not None:
            if angle_is_target(rotate_x_attr.Get()):
                raise RuntimeError(f"{prim.GetPath()}: rotateX +90이 남아 있습니다.")

        remaining = find_rotate_units_resolve_attrs(prim)
        if remaining:
            raise RuntimeError(f"{prim.GetPath()}: unitResolve가 남아 있습니다: {remaining}")


def export_atomically(stage, usd_path, mode_source_path):
    """같은 폴더의 임시 USD로 검증한 뒤 원본을 교체한다."""
    from pxr import Usd

    temporary_path = usd_path.with_name(
        f".{usd_path.stem}.x90_cleanup.tmp{usd_path.suffix}"
    )
    try:
        if not stage.GetRootLayer().Export(str(temporary_path)):
            raise RuntimeError(f"임시 USD export 실패: {temporary_path}")

        exported_stage = Usd.Stage.Open(str(temporary_path))
        if exported_stage is None:
            raise RuntimeError(f"임시 USD 재검증 open 실패: {temporary_path}")
        validate_world_children(exported_stage)

        shutil.copymode(mode_source_path, temporary_path)
        os.replace(temporary_path, usd_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def process_usd(usd_path):
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError("USD를 열지 못했습니다.")
    stage.SetEditTarget(stage.GetRootLayer())

    world = stage.GetPrimAtPath("/World")
    if not world.IsValid():
        raise RuntimeError("/World prim이 없습니다.")

    changes = []
    for prim in world.GetChildren():
        prim_changes = clean_world_child(prim)
        if prim_changes:
            changes.append((prim.GetPath().pathString, prim_changes))

    if not changes:
        return "skipped", 0

    validate_world_children(stage)
    export_atomically(stage, usd_path, usd_path)

    print(f"UPDATED: {usd_path} (prims={len(changes)})")
    return "updated", len(changes)


def run():
    usd_paths = find_usd_paths()
    if not usd_paths:
        print(f"처리할 USD가 없습니다: {ROOT_PATH}")
        return 1

    updated_files = 0
    updated_prims = 0
    skipped_files = 0
    failed_files = []

    print(f"USD FILES: {len(usd_paths)}")

    for usd_path in usd_paths:
        try:
            status, prim_count = process_usd(usd_path)
            if status == "skipped":
                skipped_files += 1
            else:
                updated_files += 1
                updated_prims += prim_count
        except Exception as exc:
            failed_files.append((usd_path, f"{type(exc).__name__}: {exc}"))

    print(
        f"DONE: updated_files={updated_files}, updated_prims={updated_prims}, "
        f"skipped_files={skipped_files}, failed_files={len(failed_files)}"
    )
    if failed_files:
        print("FAILED FILES:")
        for usd_path, reason in failed_files:
            print(f"  - {usd_path}")
            print(f"    reason: {reason}")

    return 1 if failed_files else 0


def main():
    # Physx/USD schema plugin이 준비된 뒤 pxr USD를 사용한다.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        return run()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
