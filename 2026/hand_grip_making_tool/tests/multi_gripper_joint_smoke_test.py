"""Verify dynamic joint/root discovery against representative hand structures."""

import asyncio
import math
import traceback
from pathlib import Path

import omni.kit.app
import omni.usd
from isaacsim.core.prims import SingleArticulation
from omni.ext._impl import _internal


HAND_ROOT = Path("/nas/ochansol/isaac/USD/robots/gripper/Hand")
CASES = {
    "Inspire-F1_right": (6, 0),
    "Agibot-Omnihand-Pro_right": (12, 0),
    "Allegro-V5-Plus_right": (16, 0),
    "Brainco-Revo2_right": (6, 0),
    "Brainco-Revo3_right": (21, 0),
    "DG-5F_right": (20, 0),
    "DG-5F-S_right": (20, 0),
    "Leap-Hand-V1_right": (16, 0),
    "Linkerbot-L20_right": (16, 0),
    "OYMotion-ROHand": (6, 19),
    "Orca-V1_right": (17, 0),
    "Psyonic-Ability-Hand_right": (6, 0),
    "Robotis-HX5-D20_right": (20, 0),
    "SharpaWave_right": (22, 0),
    "Wuji-V1-1_right": (20, 0),
    "xHand-1_right": (12, 0),
}


async def _run():
    return_code = 0
    try:
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()

        extension = None
        for ext_id, modules in _internal._extensions.items():
            if ext_id.startswith("hand_grip_making_tool-") and modules._started_extensions:
                extension = modules._started_extensions[0][0]
                break
        assert extension is not None, "Extension instance was not found"

        for folder_name, (expected_joints, expected_unbounded) in CASES.items():
            folder = HAND_ROOT / folder_name
            usd_path = folder / f"{folder_name}.usd"
            extension._usd_model.set_value(str(usd_path))
            extension._gripper_key_model.set_value(folder_name)
            extension._load_gripper()
            await omni.kit.app.get_app().next_update_async()

            assert set(extension._joint_infos) == set(extension._joint_models), (
                folder_name,
                sorted(extension._joint_infos),
                sorted(extension._joint_models),
            )
            assert len(extension._joint_infos) == expected_joints, (
                folder_name,
                extension._joint_infos,
            )
            assert len(extension._skipped_unbounded_joints) == expected_unbounded, (
                folder_name,
                extension._skipped_unbounded_joints,
            )
            assert extension._articulation_root_path
            assert all(
                math.isfinite(info.lower_rad) and math.isfinite(info.upper_rad)
                for info in extension._joint_infos.values()
            )

            # Regression: loading Agibot after another hand used to retain the
            # old UI models, then Open/Close raised KeyError: R_1_abd_joint.
            if folder_name == "Agibot-Omnihand-Pro_right":
                assert "R_1_abd_joint" in extension._joint_infos
                extension._set_all_joints(close=False)
                extension._set_all_joints(close=True)

            extension._timeline.play()
            await omni.kit.app.get_app().next_update_async()
            articulation = SingleArticulation(
                prim_path=extension._articulation_root_path,
                name=f"joint_discovery_{folder_name}",
                reset_xform_properties=False,
            )
            articulation.initialize()
            missing = sorted(set(extension._joint_infos) - set(articulation.dof_names))
            assert not missing, (folder_name, missing, list(articulation.dof_names))
            extension._timeline.stop()
            await omni.kit.app.get_app().next_update_async()

            print(
                "MULTI_HAND_JOINT_CASE_PASS",
                folder_name,
                len(extension._joint_infos),
                extension._articulation_root_path,
            )

        print("MULTI_HAND_JOINT_SMOKE_TEST: PASS")
    except Exception:
        return_code = 2
        traceback.print_exc()
        print("MULTI_HAND_JOINT_SMOKE_TEST: FAIL")
    finally:
        omni.kit.app.get_app().post_quit(return_code)


asyncio.ensure_future(_run())
