from pathlib import Path

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Sdf, Usd


ROOT_PATH = Path("/nas/ochansol/3d_model/peel3_scan_data_2024")
REQUIRE_TARGET_EXISTS = True


def is_target_texture(asset_path, obj_name):
    if not asset_path:
        return False
    return Path(asset_path).name == f"{obj_name}.bmp"


def iter_texture_asset_attrs(stage):
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath):
                yield prim, attr, value


def update_usd_texture_paths(usd_path, obj_name):
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        print(f"[skip] failed to open usd: {usd_path}", flush=True)
        return 0

    changed = 0
    target_path = usd_path.parents[1] / f"{obj_name}_edited.bmp"
    new_asset_path = f"../{obj_name}_edited.bmp"

    if REQUIRE_TARGET_EXISTS and not target_path.exists():
        print(f"[missing target] {obj_name}: {target_path}", flush=True)
        return 0

    for prim, attr, value in iter_texture_asset_attrs(stage):
        if not is_target_texture(value.path, obj_name):
            continue
        attr.Set(Sdf.AssetPath(new_asset_path))
        changed += 1
        print(f"[update] {usd_path.name}: {prim.GetPath()} {attr.GetName()} {value.path} -> {new_asset_path}", flush=True)

    if changed:
        stage.GetRootLayer().Save()
    return changed


def main():
    total_changed = 0
    total_usd = 0

    for obj_dir in sorted(ROOT_PATH.iterdir()):
        if not obj_dir.is_dir() or obj_dir.name.startswith("."):
            continue

        edited_dir = obj_dir / "edited"
        if not edited_dir.is_dir():
            print(f"[skip] no edited dir: {obj_dir}", flush=True)
            continue

        usd_paths = sorted(edited_dir.glob("*.usd"))
        if not usd_paths:
            print(f"[skip] no usd: {edited_dir}", flush=True)
            continue

        for usd_path in usd_paths:
            total_usd += 1
            total_changed += update_usd_texture_paths(usd_path, obj_dir.name)

    print(f"complete: usd={total_usd}, changed_attrs={total_changed}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
