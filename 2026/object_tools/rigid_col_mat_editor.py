"""기존 ``*_rigid.usd``의 rigid body, collider와 physics material을 갱신한다.

CLI 인자는 사용하지 않는다. 아래 ``TARGET_FOLDER``만 수정한 뒤 Isaac Sim
5.1이 설치된 uv 환경에서 실행한다. 원본 USD를 직접 교체하며 별도 백업은
생성하지 않는다. 저장은 같은 폴더의 임시 USD를 검증한 뒤 원자적으로
교체한다.

실행 예시::

    cd /home/uon/ochansol/isaac_code/isaac_chansol
    uv run python \
      /home/uon/ochansol/isaac_code/python/sanjabu/2026/object_tools/rigid_col_mat_editor.py

``TARGET_FOLDER``에는 다음 중 하나를 지정할 수 있다.

* 물체 폴더: ``.../peel3_scan_data_2026/air_plunger``
* edited 폴더: ``.../air_plunger/edited``
* 단일 파일: ``.../edited/air_plunger_rigid.usd``
* 데이터셋 루트: ``.../peel3_scan_data_2026`` (하위 물체 전체 처리)
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
TARGET_FOLDER = Path(
    "/nas/ochansol/3d_model/peel3_scan_data_2026"
)

# True이면 대상과 변경 예정 값만 검사하고 USD는 저장하지 않는다.
DRY_RUN = False

# Convex decomposition collider
MAX_CONVEX_HULLS = 32
HULL_VERTEX_LIMIT = 64
VOXEL_RESOLUTION = 200_000
ERROR_PERCENTAGE = 10.0

# Scan_Rep.set_rigidbody_collider()가 함께 사용하던 collision 설정
CONTACT_OFFSET = 0.000001
REST_OFFSET = 0.0
SHRINK_WRAP = True

# Rigid body CCD 설정
ENABLE_CCD = True
ENABLE_SPECULATIVE_CCD = True

# Physics material
DYNAMIC_FRICTION = 0.25
STATIC_FRICTION = 0.4
RESTITUTION = 0.0


def _rigid_usds_in_edited_dir(edited_dir: Path, object_name: str) -> list[Path]:
    """물체명 기반 파일을 우선하고, 없으면 해당 edited의 rigid USD를 찾는다."""
    expected = edited_dir / f"{object_name}_rigid.usd"
    if expected.is_file():
        return [expected]
    return sorted(
        path
        for path in edited_dir.glob("*_rigid.usd")
        if path.is_file() and ".rigid_col_mat.tmp" not in path.name
    )


def find_target_usd_paths(target: Path) -> list[Path]:
    """파일·물체 폴더·edited 폴더·데이터셋 루트를 자동 판별한다."""
    target = target.expanduser().resolve()
    if target.is_file():
        if target.suffix.lower() != ".usd" or not target.stem.endswith("_rigid"):
            raise ValueError(f"*_rigid.usd 파일이 아닙니다: {target}")
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"대상 경로가 없습니다: {target}")

    if target.name == "edited":
        paths = _rigid_usds_in_edited_dir(target, target.parent.name)
    elif (target / "edited").is_dir():
        paths = _rigid_usds_in_edited_dir(target / "edited", target.name)
    else:
        # 데이터셋 루트를 지정한 경우 <object>/edited/<object>_rigid.usd 탐색
        paths = []
        for object_dir in sorted(target.iterdir()):
            if not object_dir.is_dir() or object_dir.name.startswith("."):
                continue
            edited_dir = object_dir / "edited"
            if edited_dir.is_dir():
                paths.extend(
                    _rigid_usds_in_edited_dir(edited_dir, object_dir.name)
                )

    unique_paths = list(dict.fromkeys(paths))
    if not unique_paths:
        raise FileNotFoundError(f"대상 아래에서 *_rigid.usd를 찾지 못했습니다: {target}")
    return unique_paths


def find_meshes(stage):
    from pxr import UsdGeom

    return [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]


def find_rigid_bodies(stage):
    """UsdPhysics.RigidBodyAPI가 적용된 모든 prim을 반환한다."""
    from pxr import UsdPhysics

    return [
        prim for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]


def enable_rigid_body_ccd(rigid_bodies) -> list[str]:
    """각 rigid body의 CCD 값을 상단 User settings에 맞춰 설정한다."""
    from pxr import PhysxSchema

    paths = []
    for rigid_body in rigid_bodies:
        physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_body)
        physx_rigid_body.CreateEnableCCDAttr().Set(ENABLE_CCD)
        physx_rigid_body.CreateEnableSpeculativeCCDAttr().Set(
            ENABLE_SPECULATIVE_CCD
        )
        paths.append(str(rigid_body.GetPath()))
    return paths


def set_convex_decomposition(mesh) -> None:
    """Isaac Sim 5.1 schema로 collider와 VHACD 값을 생성 또는 덮어쓴다."""
    from pxr import PhysxSchema, UsdPhysics

    collision_api = UsdPhysics.CollisionAPI.Apply(mesh)
    collision_api.CreateCollisionEnabledAttr().Set(True)

    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh)
    mesh_collision_api.CreateApproximationAttr().Set("convexDecomposition")

    physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
    physx_collision_api.CreateContactOffsetAttr().Set(CONTACT_OFFSET)
    physx_collision_api.CreateRestOffsetAttr().Set(REST_OFFSET)

    decomposition_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh)
    decomposition_api.CreateShrinkWrapAttr().Set(SHRINK_WRAP)
    decomposition_api.CreateMaxConvexHullsAttr().Set(MAX_CONVEX_HULLS)
    decomposition_api.CreateHullVertexLimitAttr().Set(HULL_VERTEX_LIMIT)
    decomposition_api.CreateVoxelResolutionAttr().Set(VOXEL_RESOLUTION)
    decomposition_api.CreateErrorPercentageAttr().Set(ERROR_PERCENTAGE)


def _physics_materials(stage):
    """현재 stage에 이미 존재하는 physics-purpose material을 반환한다."""
    from pxr import UsdPhysics, UsdShade

    return [
        UsdShade.Material(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdShade.Material) and prim.HasAPI(UsdPhysics.MaterialAPI)
    ]


def _new_physics_material(stage, first_mesh):
    """Scan_Rep과 같은 위치인 첫 mesh의 parent 아래에 material을 만든다."""
    from pxr import UsdShade

    parent_path = first_mesh.GetParent().GetPath()
    material_path = parent_path.AppendChild("PhysicsMaterial")
    suffix = 2
    while stage.GetPrimAtPath(material_path).IsValid():
        existing = stage.GetPrimAtPath(material_path)
        if existing.IsA(UsdShade.Material):
            return UsdShade.Material(existing)
        material_path = parent_path.AppendChild(f"PhysicsMaterial_{suffix}")
        suffix += 1
    return UsdShade.Material.Define(stage, material_path)


def set_physics_material(stage, meshes):
    """기존 physics material을 갱신하고 unbound mesh에도 적용한다."""
    from pxr import Sdf, UsdPhysics, UsdShade

    materials = _physics_materials(stage)
    if not materials:
        materials = [_new_physics_material(stage, meshes[0])]

    # USD에 여러 physics material이 있더라도 모든 collider가 요청한 동일
    # 물성을 갖도록 기존 값을 전부 갱신한다.
    for material in materials:
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        material_api.CreateDynamicFrictionAttr().Set(DYNAMIC_FRICTION)
        material_api.CreateStaticFrictionAttr().Set(STATIC_FRICTION)
        material_api.CreateRestitutionAttr().Set(RESTITUTION)

    default_material = materials[0]
    for mesh in meshes:
        binding_api = UsdShade.MaterialBindingAPI.Apply(mesh)
        bound_path = binding_api.GetDirectBinding("physics").GetMaterialPath()
        bound_prim = (
            stage.GetPrimAtPath(bound_path)
            if bound_path != Sdf.Path.emptyPath
            else None
        )
        if (
            bound_prim is None
            or not bound_prim.IsValid()
            or not bound_prim.HasAPI(UsdPhysics.MaterialAPI)
        ):
            binding_api.Bind(
                default_material,
                UsdShade.Tokens.weakerThanDescendants,
                "physics",
            )

    return [str(material.GetPath()) for material in materials]


def _assert_close(actual, expected, label: str) -> None:
    if actual is None or not math.isclose(
        float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-9
    ):
        raise RuntimeError(f"저장 검증 실패: {label}, expected={expected}, actual={actual}")


def verify_stage(
    stage, expected_mesh_count: int, expected_rigid_body_count: int
) -> None:
    """저장한 collider/material 값과 binding을 다시 검사한다."""
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    meshes = find_meshes(stage)
    if len(meshes) != expected_mesh_count:
        raise RuntimeError(
            f"저장 검증 실패: mesh count {expected_mesh_count} -> {len(meshes)}"
        )

    rigid_bodies = find_rigid_bodies(stage)
    if len(rigid_bodies) != expected_rigid_body_count:
        raise RuntimeError(
            "저장 검증 실패: rigid body count "
            f"{expected_rigid_body_count} -> {len(rigid_bodies)}"
        )
    for rigid_body in rigid_bodies:
        rigid_body_path = str(rigid_body.GetPath())
        if not rigid_body.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            raise RuntimeError(
                f"저장 검증 실패: PhysxRigidBodyAPI 없음: {rigid_body_path}"
            )
        physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI(rigid_body)
        enable_ccd = physx_rigid_body.GetEnableCCDAttr().Get()
        if enable_ccd is not ENABLE_CCD:
            raise RuntimeError(
                "저장 검증 실패: CCD 설정 불일치: "
                f"{rigid_body_path}, expected={ENABLE_CCD}, actual={enable_ccd}"
            )
        enable_speculative_ccd = (
            physx_rigid_body.GetEnableSpeculativeCCDAttr().Get()
        )
        if enable_speculative_ccd is not ENABLE_SPECULATIVE_CCD:
            raise RuntimeError(
                "저장 검증 실패: speculative CCD 설정 불일치: "
                f"{rigid_body_path}, expected={ENABLE_SPECULATIVE_CCD}, "
                f"actual={enable_speculative_ccd}"
            )

    for mesh in meshes:
        mesh_path = str(mesh.GetPath())
        if not mesh.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"저장 검증 실패: CollisionAPI 없음: {mesh_path}")
        if not mesh.HasAPI(UsdPhysics.MeshCollisionAPI):
            raise RuntimeError(f"저장 검증 실패: MeshCollisionAPI 없음: {mesh_path}")
        if not mesh.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
            raise RuntimeError(f"저장 검증 실패: ConvexDecompositionAPI 없음: {mesh_path}")

        mesh_collision = UsdPhysics.MeshCollisionAPI(mesh)
        if mesh_collision.GetApproximationAttr().Get() != "convexDecomposition":
            raise RuntimeError(f"저장 검증 실패: approximation: {mesh_path}")
        decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI(mesh)
        if decomposition.GetMaxConvexHullsAttr().Get() != MAX_CONVEX_HULLS:
            raise RuntimeError(f"저장 검증 실패: maxConvexHulls: {mesh_path}")
        if decomposition.GetHullVertexLimitAttr().Get() != HULL_VERTEX_LIMIT:
            raise RuntimeError(f"저장 검증 실패: hullVertexLimit: {mesh_path}")
        if decomposition.GetVoxelResolutionAttr().Get() != VOXEL_RESOLUTION:
            raise RuntimeError(f"저장 검증 실패: voxelResolution: {mesh_path}")
        _assert_close(
            decomposition.GetErrorPercentageAttr().Get(),
            ERROR_PERCENTAGE,
            f"errorPercentage: {mesh_path}",
        )

        material_path = (
            UsdShade.MaterialBindingAPI(mesh)
            .GetDirectBinding("physics")
            .GetMaterialPath()
        )
        material_prim = stage.GetPrimAtPath(material_path)
        if not material_prim.IsValid() or not material_prim.HasAPI(UsdPhysics.MaterialAPI):
            raise RuntimeError(
                f"저장 검증 실패: physics material binding 없음: {mesh_path}"
            )
        material_api = UsdPhysics.MaterialAPI(material_prim)
        _assert_close(
            material_api.GetDynamicFrictionAttr().Get(),
            DYNAMIC_FRICTION,
            f"dynamic friction: {material_path}",
        )
        _assert_close(
            material_api.GetStaticFrictionAttr().Get(),
            STATIC_FRICTION,
            f"static friction: {material_path}",
        )
        _assert_close(
            material_api.GetRestitutionAttr().Get(),
            RESTITUTION,
            f"restitution: {material_path}",
        )


def export_atomically(
    stage, usd_path: Path, mesh_count: int, rigid_body_count: int
) -> None:
    """백업 없이 임시 USD 검증 후 원본 파일을 교체한다."""
    from pxr import Usd

    temporary_path = usd_path.with_name(
        f".{usd_path.stem}.rigid_col_mat.tmp{usd_path.suffix}"
    )
    try:
        if not stage.GetRootLayer().Export(str(temporary_path)):
            raise RuntimeError(f"임시 USD export 실패: {temporary_path}")
        verify = Usd.Stage.Open(str(temporary_path))
        if verify is None:
            raise RuntimeError(f"임시 USD를 다시 열지 못했습니다: {temporary_path}")
        verify_stage(verify, mesh_count, rigid_body_count)
        shutil.copymode(usd_path, temporary_path)
        os.replace(temporary_path, usd_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def process_usd(usd_path: Path) -> dict:
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"USD를 열지 못했습니다: {usd_path}")
    stage.SetEditTarget(stage.GetRootLayer())

    meshes = find_meshes(stage)
    if not meshes:
        raise RuntimeError("Mesh prim이 없습니다.")
    for mesh in meshes:
        set_convex_decomposition(mesh)
    material_paths = set_physics_material(stage, meshes)

    rigid_bodies = find_rigid_bodies(stage)
    if not rigid_bodies:
        raise RuntimeError("UsdPhysics.RigidBodyAPI가 적용된 prim이 없습니다.")
    rigid_body_paths = enable_rigid_body_ccd(rigid_bodies)

    if not DRY_RUN:
        export_atomically(stage, usd_path, len(meshes), len(rigid_bodies))

    return {
        "usd_path": str(usd_path),
        "mesh_count": len(meshes),
        "rigid_body_paths": rigid_body_paths,
        "material_paths": material_paths,
        "status": "dry_run" if DRY_RUN else "updated",
    }


def run() -> int:
    usd_paths = find_target_usd_paths(TARGET_FOLDER)
    print(f"TARGET: {TARGET_FOLDER}")
    print(f"MODE: {'DRY RUN' if DRY_RUN else 'APPLY'}")
    print(f"FILES: {len(usd_paths)}")
    print(
        f"CCD: enableCCD={ENABLE_CCD}, "
        f"enableSpeculativeCCD={ENABLE_SPECULATIVE_CCD}"
    )

    failures = []
    for index, usd_path in enumerate(usd_paths, start=1):
        try:
            result = process_usd(usd_path)
            print(
                f"[{index}/{len(usd_paths)}] {result['status'].upper()}: "
                f"{usd_path} (meshes={result['mesh_count']}, "
                f"ccd_rigid_bodies={len(result['rigid_body_paths'])}, "
                f"materials={result['material_paths']})"
            )
        except Exception as error:
            failures.append((usd_path, error))
            print(
                f"[{index}/{len(usd_paths)}] ERROR: {usd_path}: "
                f"{type(error).__name__}: {error}"
            )

    print("\nDONE")
    print(f"SUCCESS: {len(usd_paths) - len(failures)}")
    print(f"FAILURES: {len(failures)}")
    if failures:
        print("\nFAILED FILES")
        for usd_path, error in failures:
            print(f"- {usd_path}: {type(error).__name__}: {error}")
    return 1 if failures else 0


def main() -> int:
    # PhysxSchema를 사용하기 전에 Isaac/Kit을 초기화해야 한다.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        return run()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
