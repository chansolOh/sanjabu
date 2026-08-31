"""Headless smoke test for the Isaac Sim extension.

Run with::

    isaacsim --no-window \
      --ext-folder /home/uon/ochansol/isaac_code/python/sanjabu/2026 \
      --enable gripper_contact_force_viewer \
      --exec /home/uon/ochansol/isaac_code/python/sanjabu/2026/gripper_contact_force_viewer/tests/isaac_smoke_test.py
"""

import asyncio
import math
import traceback

import omni.kit.app
import omni.usd
from omni.ext._impl import _internal
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics


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
        assert set(extension._joint_infos) == {"left_finger_joint"}, extension._joint_infos

        extension._test_scale_x_model.set_value(0.05)
        extension._test_scale_y_model.set_value(0.05)
        extension._test_scale_z_model.set_value(0.05)
        extension._create_test_object()
        test_object = extension._stage().GetPrimAtPath(
            "/World/GripperContactForceViewer/TestObject"
        )
        assert test_object.IsValid()
        assert test_object.IsA(UsdGeom.Mesh)
        assert not test_object.HasAPI(UsdPhysics.RigidBodyAPI)
        assert test_object.HasAPI(UsdPhysics.CollisionAPI)
        assert (
            UsdPhysics.MeshCollisionAPI(test_object).GetApproximationAttr().Get()
            == "convexHull"
        )
        cube_point_count = len(UsdGeom.Mesh(test_object).GetPointsAttr().Get())
        actual_scale = tuple(
            Gf.Transform(UsdGeom.Xformable(test_object).GetLocalTransformation()).GetScale()
        )
        assert max(abs(value - 0.05) for value in actual_scale) < 1.0e-6

        extension._test_shape_combo.model.get_item_value_model().set_value(1)
        extension._create_test_object()
        sphere = UsdGeom.Mesh.Get(
            extension._stage(), "/World/GripperContactForceViewer/TestObject"
        )
        assert sphere.GetPrim().IsValid()
        assert not sphere.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI)
        assert len(sphere.GetPointsAttr().Get()) > cube_point_count
        assert (
            UsdPhysics.MeshCollisionAPI(sphere).GetApproximationAttr().Get()
            == "convexHull"
        )

        body_path = "/World/GripperContactForceViewer/Gripper/Asset/SmokeContactBody"
        body = UsdGeom.Cube.Define(extension._stage(), body_path).GetPrim()
        UsdPhysics.CollisionAPI.Apply(body)
        UsdPhysics.RigidBodyAPI.Apply(body)
        PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr().Set(0.0)
        extension._refresh_sensors()

        assert body_path in extension._sensor_states
        assert extension._sensor_states[body_path].body_path == body_path

        extension._set_gripper(False)
        extension._set_gripper(True)

        # Hand assets are matched to gripper_info_hand_2026.json by USD path.
        # Their START/END values, rather than joint-limit endpoints, must drive
        # OPEN/CLOSE. Agibot also verifies generic DOF-name discovery.
        hand_usd = (
            "/nas/ochansol/isaac/USD/robots/gripper/Hand/"
            "Agibot-Omnihand-Pro_right/Agibot-Omnihand-Pro_right.usd"
        )
        extension._usd_model.set_value(hand_usd)
        extension._load_gripper()
        await omni.kit.app.get_app().next_update_async()
        assert extension._is_hand_gripper()
        assert extension._matched_hand_db_key == "Agibot-Omnihand-Pro_right"
        assert extension._hand_presets
        assert "R_1_abd_joint" in extension._joint_infos

        extension._hand_preset_combo_model.get_item_value_model().set_value(0)
        preset_name, open_targets, _missing = extension._hand_joint_targets(False)
        assert preset_name == extension._hand_preset_names[0]
        assert open_targets
        extension._set_gripper(False)
        for name, value in open_targets.items():
            info = extension._joint_infos[name]
            drive = UsdPhysics.DriveAPI.Get(
                extension._stage().GetPrimAtPath(info.prim_path), info.drive_name
            )
            expected = max(info.lower, min(info.upper, value))
            if info.revolute:
                expected = math.degrees(expected)
            assert abs(float(drive.GetTargetPositionAttr().Get()) - expected) < 1.0e-4

        _preset_name, close_targets, _missing = extension._hand_joint_targets(True)
        extension._set_gripper(True)
        assert close_targets
        extension._play()
        await omni.kit.app.get_app().next_update_async()
        extension._on_physics_step(1.0 / 60.0)
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
