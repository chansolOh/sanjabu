"""USD에 rigid body와 convex-decomposition collider를 추가한다.

아래 User settings의 변수만 수정한 뒤 Isaac Sim 5.1 uv 환경에서 실행한다.
"""

from pathlib import Path


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
ROOT_PATH = Path("/nas/ochansol/3d_model/peel3_scan_data_2026")

# 빈 리스트: ROOT_PATH/<object>/edited/<object>.usd 전체 처리
# 특정 파일만 처리: [Path("/path/to/object.usd")]
USD_PATHS: list[Path] = []

# 기존 <이름>_rigid.usd 파일을 덮어쓸지 여부
OVERWRITE = True

# 보통 비워두면 /World 아래에서 자동으로 찾는다.
# 자동 탐색이 틀린 USD만 예: ["/World/Object"]
RIGID_ROOT_PATHS: list[str] = []

# Convex decomposition(VHACD) 설정
MAX_CONVEX_HULLS = 50
HULL_VERTEX_LIMIT = 64
VOXEL_RESOLUTION = 200_000
ERROR_PERCENTAGE = 10.0
CONTACT_OFFSET = 0.000001
REST_OFFSET = 0.0

# Physics material 설정
DYNAMIC_FRICTION = 0.3
STATIC_FRICTION = 0.4
RESTITUTION = 0.5


def find_usd_paths() -> list[Path]:
    """설정된 단일/복수 파일 또는 데이터셋 전체 입력을 반환한다."""
    if USD_PATHS:
        return [Path(path) for path in USD_PATHS]

    paths = []
    for object_dir in sorted(ROOT_PATH.iterdir()):
        if not object_dir.is_dir() or object_dir.name.startswith("."):
            continue
        usd_path = object_dir / "edited" / f"{object_dir.name}.usd"
        if usd_path.is_file():
            paths.append(usd_path)
    return paths


def find_meshes(root):
    from pxr import Usd, UsdGeom

    return [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]


def find_rigid_roots(stage):
    """RigidBodyAPI를 적용할 물체 최상위 prim을 찾는다."""
    from pxr import UsdGeom

    if RIGID_ROOT_PATHS:
        roots = [stage.GetPrimAtPath(path) for path in RIGID_ROOT_PATHS]
        invalid = [path for path, prim in zip(RIGID_ROOT_PATHS, roots) if not prim.IsValid()]
        if invalid:
            raise RuntimeError(f"Rigid root prim을 찾지 못함: {invalid}")
        return roots

    world = stage.GetPrimAtPath("/World")
    if not world.IsValid():
        raise RuntimeError("/World prim을 찾지 못했습니다. RIGID_ROOT_PATHS를 직접 지정하세요.")

    roots = []
    for child in world.GetChildren():
        if child.GetName().lower() in {"looks", "materials", "physicsmaterial"}:
            continue
        if UsdGeom.Xformable(child) and find_meshes(child):
            roots.append(child)

    # Mesh가 /World prim 자체에 직접 들어 있는 예외 구조
    if not roots and UsdGeom.Xformable(world) and find_meshes(world):
        roots.append(world)
    if not roots:
        raise RuntimeError("/World 아래에서 Mesh를 포함한 물체 prim을 찾지 못했습니다.")
    return roots


def add_rigid_body(root) -> None:
    """물체 공통 transform에 rigid body를 한 번만 적용한다."""
    from pxr import PhysxSchema, UsdPhysics

    rigid_api = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_api.CreateRigidBodyEnabledAttr().Set(True)

    physx_rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    physx_rigid_api.CreateEnableCCDAttr().Set(True)
    # physx_rigid_api.CreateMaxAngularVelocityAttr().Set(720.0)
    # physx_rigid_api.CreateMaxLinearVelocityAttr().Set(2.5)
    # physx_rigid_api.CreateLinearDampingAttr().Set(0.7)


def add_collider(mesh) -> None:
    """각 Mesh에 Isaac Sim 5.1 convex-decomposition collider를 적용한다."""
    from pxr import PhysxSchema, UsdPhysics

    collision_api = UsdPhysics.CollisionAPI.Apply(mesh)
    collision_api.CreateCollisionEnabledAttr().Set(True)

    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh)
    mesh_collision_api.CreateApproximationAttr().Set("convexDecomposition")

    physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
    physx_collision_api.CreateContactOffsetAttr().Set(CONTACT_OFFSET)
    physx_collision_api.CreateRestOffsetAttr().Set(REST_OFFSET)

    decomposition_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh)
    decomposition_api.CreateShrinkWrapAttr().Set(True)
    decomposition_api.CreateMaxConvexHullsAttr().Set(MAX_CONVEX_HULLS)
    decomposition_api.CreateHullVertexLimitAttr().Set(HULL_VERTEX_LIMIT)
    decomposition_api.CreateVoxelResolutionAttr().Set(VOXEL_RESOLUTION)
    decomposition_api.CreateErrorPercentageAttr().Set(ERROR_PERCENTAGE)


def set_physics_material(stage, root, meshes) -> None:
    """시각 재질은 유지하고 physics purpose 재질을 collider mesh에 적용한다."""
    from pxr import UsdPhysics, UsdShade

    material_path = root.GetPath().AppendChild("PhysicsMaterial")
    physics_material = UsdShade.Material.Define(stage, material_path)

    material_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    material_api.CreateDynamicFrictionAttr().Set(DYNAMIC_FRICTION)
    material_api.CreateStaticFrictionAttr().Set(STATIC_FRICTION)
    material_api.CreateRestitutionAttr().Set(RESTITUTION)

    for mesh in meshes:
        binding_api = UsdShade.MaterialBindingAPI.Apply(mesh)
        binding_api.Bind(
            physics_material,
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )
        # 기존 set_physics_material()과 동일하게 질량 계산 API도 적용한다.
        UsdPhysics.MassAPI.Apply(mesh)


def process_usd(usd_path: Path) -> bool:
    from pxr import Usd

    usd_path = usd_path.expanduser().resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(usd_path)
    if usd_path.stem.endswith("_rigid"):
        raise RuntimeError(f"_rigid.usd는 입력으로 사용할 수 없습니다: {usd_path}")

    output_path = usd_path.with_name(f"{usd_path.stem}_rigid.usd")
    if output_path.exists() and not OVERWRITE:
        print(f"SKIP: {output_path}")
        return False

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"USD를 열지 못했습니다: {usd_path}")
    stage.SetEditTarget(stage.GetRootLayer())

    rigid_roots = find_rigid_roots(stage)
    collider_count = 0
    for root in rigid_roots:
        add_rigid_body(root)
        meshes = find_meshes(root)
        for mesh in meshes:
            add_collider(mesh)
            collider_count += 1
        set_physics_material(stage, root, meshes)

    if not stage.GetRootLayer().Export(str(output_path)):
        raise RuntimeError(f"USD 저장 실패: {output_path}")

    print(
        f"OK: {output_path} "
        f"(rigid roots={len(rigid_roots)}, colliders={collider_count})"
    )
    return True


def main() -> int:
    # PhysxSchema를 사용하기 전에 Isaac/Kit을 먼저 초기화해야 한다.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    failed = 0
    try:
        usd_paths = find_usd_paths()
        if not usd_paths:
            print("처리할 USD 파일이 없습니다.")
            return 1

        for usd_path in usd_paths:
            try:
                process_usd(usd_path)
            except Exception as exc:
                failed += 1
                print(f"ERROR: {usd_path}: {exc}")
        return 1 if failed else 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
