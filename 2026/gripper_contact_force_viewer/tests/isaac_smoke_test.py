"""Headless smoke test for the Isaac Sim extension.

Run with::

    isaacsim --no-window \
      --ext-folder /home/uon/ochansol/isaac_code/python/sanjabu/2026 \
      --enable gripper_contact_force_viewer \
      --exec /home/uon/ochansol/isaac_code/python/sanjabu/2026/gripper_contact_force_viewer/tests/isaac_smoke_test.py
"""

import asyncio
import traceback

import omni.kit.app
import omni.usd
from omni.ext._impl import _internal
from pxr import PhysxSchema, UsdGeom, UsdPhysics


async def _run():
    return_code = 0
    try:
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()

        extension = None
        for ext_id, modules in _internal._extensions.items():
            if ext_id.startswith("gripper_contact_force_viewer-") and modules._started_extensions:
                extension = modules._started_extensions[0][0]
                break
        assert extension is not None, "Extension instance was not found"

        extension._load_gripper()
        await omni.kit.app.get_app().next_update_async()
        assert extension._stage().GetPrimAtPath(extension._gripper_prim().GetPath()).IsValid()
        assert extension._joint_infos, "Default Robotiq asset should expose driven joints"

        collision_path = "/World/GripperContactForceViewer/Gripper/Asset/SmokeContactBody"
        collision = UsdGeom.Cube.Define(extension._stage(), collision_path).GetPrim()
        UsdPhysics.CollisionAPI.Apply(collision)
        UsdPhysics.RigidBodyAPI.Apply(collision)
        PhysxSchema.PhysxContactReportAPI.Apply(collision).CreateThresholdAttr().Set(0.0)
        extension._refresh_sensors()

        sensor_path = f"{collision_path}/ContactForceViewerSensor"
        assert extension._stage().GetPrimAtPath(sensor_path).IsValid(), sensor_path
        assert sensor_path in extension._sensor_states
        assert extension._sensor_states[sensor_path].body_path == collision_path

        extension._set_gripper(False)
        extension._set_gripper(True)
        extension._play()
        await omni.kit.app.get_app().next_update_async()
        extension._update_forces()
        extension._timeline.stop()
        await omni.kit.app.get_app().next_update_async()
        print("GRIPPER_CONTACT_FORCE_VIEWER_SMOKE_TEST_PASS")
    except Exception:
        return_code = 1
        traceback.print_exc()
    finally:
        omni.kit.app.get_app().post_quit(return_code)


asyncio.ensure_future(_run())
