"""CSV의 Weight(g)를 edited USD rigid body의 physics:mass(kg)에 적용한다.

Isaac Sim 5.1 uv 환경에서 실행한다. CLI 인자는 사용하지 않으며 아래
User settings 변수로 경로와 동작을 설정한다.

처리 방식:
    1. CSV의 Object_name / Weight(g)를 읽는다.
    2. <ROOT_PATH>/<Object_name>/edited/*.usd를 모두 검사한다.
    3. 실제 RigidBodyAPI가 존재하는 USD만 처리한다.
    4. rigid-body prim에 MassAPI를 적용하고 g -> kg 변환값을 저장한다.

원본 USD를 직접 갱신한다. 미리 검사만 하려면 DRY_RUN=True로 변경한다.
"""

import csv
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
ROOT_PATH = Path("/nas/ochansol/3d_model/peel3_scan_data_2026")
CSV_PATH = ROOT_PATH / "2026_objects_cat_attr.csv"

OBJECT_NAME_COLUMN = "Object_name"
WEIGHT_COLUMN = "Weight(g)"

USD_SUFFIXES = {".usd"}

# True: 변경 예정 대상만 확인, False: 실제 USD 수정
DRY_RUN = False

# 기존 physics:mass가 있어도 CSV 값으로 갱신할지 여부
OVERWRITE_EXISTING_MASS = True

# 하나의 USD에 rigid body가 여러 개면 전체 무게를 어떻게 나눌지 알 수 없다.
# 기본값 False에서는 해당 USD를 실패 처리한다.
ALLOW_MULTIPLE_RIGID_BODIES = False

REPORT_DIR = Path(__file__).resolve().parent / "mass_update_reports"


def parse_weight_grams(raw_value, row_number):
    value = str(raw_value or "").strip().replace(",", "")
    if not value:
        raise ValueError(f"row {row_number}: {WEIGHT_COLUMN} 값이 비어 있습니다.")
    try:
        weight_g = float(value)
    except ValueError as error:
        raise ValueError(
            f"row {row_number}: 잘못된 {WEIGHT_COLUMN} 값: {raw_value!r}"
        ) from error
    if not math.isfinite(weight_g) or weight_g <= 0.0:
        raise ValueError(f"row {row_number}: 무게는 0보다 커야 합니다: {raw_value!r}")
    return weight_g


def load_object_weights():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV 파일이 없습니다: {CSV_PATH}")

    weights = {}
    errors = []
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        required = {OBJECT_NAME_COLUMN, WEIGHT_COLUMN}
        missing_columns = sorted(required - fieldnames)
        if missing_columns:
            raise KeyError(f"CSV 필수 컬럼이 없습니다: {missing_columns}")

        for row_number, row in enumerate(reader, start=2):
            object_name = str(row.get(OBJECT_NAME_COLUMN) or "").strip()
            if not object_name:
                # CSV 하단의 빈 행과 분류별 집계 행은 물체 데이터가 아니다.
                continue
            try:
                weight_g = parse_weight_grams(row.get(WEIGHT_COLUMN), row_number)
            except ValueError as error:
                errors.append(str(error))
                continue

            if object_name in weights:
                errors.append(f"row {row_number}: Object_name 중복: {object_name}")
                continue
            weights[object_name] = weight_g

    return weights, errors


def find_object_usd_paths(object_name):
    edited_dir = ROOT_PATH / object_name / "edited"
    if not edited_dir.is_dir():
        return []
    return sorted(
        path
        for path in edited_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in USD_SUFFIXES
        and ".mass_update.tmp" not in path.name
    )


def find_rigid_body_prims(stage):
    from pxr import UsdPhysics

    return [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]


def get_authored_mass(prim):
    from pxr import UsdPhysics

    if not prim.HasAPI(UsdPhysics.MassAPI):
        return None
    mass_attr = UsdPhysics.MassAPI(prim).GetMassAttr()
    return mass_attr.Get() if mass_attr.IsValid() and mass_attr.HasAuthoredValueOpinion() else None


def set_mass_kg(prim, mass_kg):
    from pxr import UsdPhysics

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(float(mass_kg))


def export_atomically(stage, usd_path, expected_masses):
    """같은 폴더의 임시 USD로 export·검증한 후 원본을 교체한다."""
    from pxr import Usd, UsdPhysics

    temporary_path = usd_path.with_name(
        f".{usd_path.stem}.mass_update.tmp{usd_path.suffix}"
    )
    try:
        if not stage.GetRootLayer().Export(str(temporary_path)):
            raise RuntimeError(f"임시 USD export 실패: {temporary_path}")

        verify_stage = Usd.Stage.Open(str(temporary_path), load=Usd.Stage.LoadNone)
        if verify_stage is None:
            raise RuntimeError(f"저장된 임시 USD를 다시 열지 못했습니다: {temporary_path}")

        for prim_path, expected_mass in expected_masses.items():
            prim = verify_stage.GetPrimAtPath(prim_path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.MassAPI):
                raise RuntimeError(f"저장 검증 실패, MassAPI 없음: {prim_path}")
            actual_mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if actual_mass is None or not math.isclose(
                float(actual_mass), float(expected_mass), rel_tol=1e-6, abs_tol=1e-9
            ):
                raise RuntimeError(
                    f"저장 검증 실패: {prim_path}, expected={expected_mass}, actual={actual_mass}"
                )

        shutil.copymode(usd_path, temporary_path)
        os.replace(temporary_path, usd_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def process_usd(usd_path, weight_g):
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError("USD를 열지 못했습니다.")
    stage.SetEditTarget(stage.GetRootLayer())

    rigid_prims = find_rigid_body_prims(stage)
    if not rigid_prims:
        return {
            "status": "no_rigid_body",
            "usd_path": str(usd_path),
            "weight_g": weight_g,
        }
    if len(rigid_prims) > 1 and not ALLOW_MULTIPLE_RIGID_BODIES:
        raise RuntimeError(
            "RigidBodyAPI prim이 여러 개입니다. 질량 분배를 임의로 적용하지 않습니다: "
            f"{[str(prim.GetPath()) for prim in rigid_prims]}"
        )

    # ALLOW_MULTIPLE_RIGID_BODIES=True인 경우에만 총 무게를 균등 분배한다.
    mass_per_body_kg = (weight_g / 1000.0) / len(rigid_prims)
    changes = []
    expected_masses = {}
    for prim in rigid_prims:
        old_mass = get_authored_mass(prim)
        if old_mass is not None and not OVERWRITE_EXISTING_MASS:
            continue
        prim_path = str(prim.GetPath())
        changes.append(
            {
                "prim_path": prim_path,
                "old_mass_kg": None if old_mass is None else float(old_mass),
                "new_mass_kg": mass_per_body_kg,
            }
        )
        expected_masses[prim_path] = mass_per_body_kg
        if not DRY_RUN:
            set_mass_kg(prim, mass_per_body_kg)

    if not changes:
        return {
            "status": "existing_mass_kept",
            "usd_path": str(usd_path),
            "weight_g": weight_g,
        }

    if not DRY_RUN:
        export_atomically(stage, usd_path, expected_masses)

    return {
        "status": "dry_run" if DRY_RUN else "updated",
        "usd_path": str(usd_path),
        "weight_g": weight_g,
        "changes": changes,
    }


def save_report(results, failures, csv_errors, missing_object_dirs):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"mass_update_{timestamp}.json"
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": DRY_RUN,
        "root_path": str(ROOT_PATH),
        "csv_path": str(CSV_PATH),
        "results": results,
        "failures": failures,
        "csv_errors": csv_errors,
        "missing_object_dirs": missing_object_dirs,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def run():
    if not ROOT_PATH.is_dir():
        raise FileNotFoundError(f"데이터셋 폴더가 없습니다: {ROOT_PATH}")

    weights, csv_errors = load_object_weights()
    results = []
    failures = []
    missing_object_dirs = []

    print(f"CSV OBJECTS: {len(weights)}")
    print(f"MODE: {'DRY RUN (USD 수정 안 함)' if DRY_RUN else 'APPLY (USD 수정)'}")

    for index, (object_name, weight_g) in enumerate(weights.items(), start=1):
        usd_paths = find_object_usd_paths(object_name)
        if not usd_paths:
            missing_object_dirs.append(object_name)
            print(f"[{index}/{len(weights)}] MISSING USD: {object_name}")
            continue

        for usd_path in usd_paths:
            try:
                result = process_usd(usd_path, weight_g)
                result["object_name"] = object_name
                results.append(result)
                print(
                    f"[{index}/{len(weights)}] {result['status'].upper()}: "
                    f"{usd_path} ({weight_g:g} g -> {weight_g / 1000.0:g} kg)"
                )
            except Exception as error:
                failure = {
                    "object_name": object_name,
                    "usd_path": str(usd_path),
                    "error": f"{type(error).__name__}: {error}",
                }
                failures.append(failure)
                print(f"[{index}/{len(weights)}] ERROR: {usd_path}: {failure['error']}")

    report_path = save_report(results, failures, csv_errors, missing_object_dirs)
    status_counts = {}
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    print("\nDONE")
    print(f"STATUS: {status_counts}")
    print(f"FAILURES: {len(failures)}")
    print(f"CSV ERRORS: {len(csv_errors)}")
    print(f"MISSING OBJECT USD: {len(missing_object_dirs)}")
    print(f"REPORT: {report_path}")

    if failures:
        print("\nFAILED FILES")
        for failure in failures:
            print(f"- {failure['usd_path']}: {failure['error']}")

    return 1 if failures or csv_errors else 0


def main():
    # Physx/UsdPhysics schema 사용 전에 Kit을 초기화한다.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        return run()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
