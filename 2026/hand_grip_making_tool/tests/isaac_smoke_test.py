"""Headless smoke test.

Run from an Isaac Sim install, for example::

    ./isaac-sim.sh --no-window \
      --ext-folder /path/to/python/sanjabu/2026 \
      --enable hand_grip_making_tool \
      --exec /path/to/hand_grip_making_tool/tests/isaac_smoke_test.py

This does not modify the real gripper database.
"""

import asyncio
import json
import shutil
import tempfile
import traceback
from pathlib import Path

import omni.kit.app
import omni.usd
from omni.ext._impl import _internal
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


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

        extension._load_gripper()
        await omni.kit.app.get_app().next_update_async()
        assert len(extension._joint_infos) == 6, extension._joint_infos
        assert len(extension._joint_models) == 6, extension._joint_models
        assert len(extension._disabled_dangling_joints) == 10, extension._disabled_dangling_joints
        assert extension._matched_db_key == "Inspire-F1"
        assert "3f_grip" in extension._saved_preset_names
        assert extension._start_pose is not None and extension._end_pose is not None
        assert extension._preset_name_model.get_value_as_string() in extension._saved_preset_names
        assert "right_hand_index_2_joint" in extension._read_urdf_mimic_joint_names(
            extension._urdf_model.get_value_as_string()
        )

        extension._apply_drive_gains()
        # Use a deliberately non-zero START so the historical 0 -> START
        # preview bug cannot pass this test unnoticed.
        extension._suppress_callbacks = True
        try:
            for name, info in extension._joint_infos.items():
                value = info.lower_rad + 0.35 * (info.upper_rad - info.lower_rad)
                extension._joint_models[name].set_value(value)
                extension._set_joint_target(name, value, set_initial_state=True)
        finally:
            extension._suppress_callbacks = False
        extension._capture_pose("start")
        assert any(abs(value) > 1e-4 for value in extension._start_pose.joints_rad.values())
        extension._set_all_joints(True)
        extension._base_models["x"].set_value(
            extension._base_models["x"].get_value_as_float() + 0.05
        )
        extension._apply_base_fields()
        extension._capture_pose("end")
        halfway = extension._interpolate_pose(extension._start_pose, extension._end_pose, 0.5)
        assert len(halfway.joints_rad) == 6

        index_tip = extension._stage().GetPrimAtPath(
            "/World/HandGripPresetTool/Hand/Asset/right_hand_index_2/mesh"
        )
        middle_tip = extension._stage().GetPrimAtPath(
            "/World/HandGripPresetTool/Hand/Asset/right_hand_middle_2/mesh"
        )
        extension._register_tip_prim(index_tip)
        extension._register_tip_prim(middle_tip)
        assert any("R_2_mcp_joint" in joints for joints in extension._tip_links.values())
        assert any("R_3_mcp_joint" in joints for joints in extension._tip_links.values())
        index_path = "/right_hand_index_2/mesh"
        middle_path = "/right_hand_middle_2/mesh"
        # Keep the middle finger stationary. Its debug BBox and persisted
        # fingertip_points entry must both be omitted.
        for joint_name in extension._tip_links[middle_path]:
            extension._end_pose.joints_rad[joint_name] = (
                extension._start_pose.joints_rad[joint_name]
            )
        await extension._generate_tip_bboxes_async()
        assert "Generated grasp BBoxes" in extension._status_label.text
        grasp_bbox_root = extension._stage().GetPrimAtPath(
            "/World/HandGripPresetTool/FingertipBBoxes/GraspBBoxes"
        )
        end_grasp_center = extension._stage().GetPrimAtPath(
            "/World/HandGripPresetTool/FingertipBBoxes/EndGraspCenter"
        )
        assert grasp_bbox_root.IsValid() and end_grasp_center.IsA(UsdGeom.Sphere)
        changed_joints = {
            name
            for name, value in extension._start_pose.joints_rad.items()
            if abs(extension._end_pose.joints_rad[name] - value)
            > extension._joint_delta_model.get_value_as_float()
        }
        active_tip_count = sum(
            bool(changed_joints.intersection(joints)) for joints in extension._tip_links.values()
        )
        assert active_tip_count == 1
        grasp_bboxes = list(grasp_bbox_root.GetChildren())
        assert len(grasp_bboxes) == active_tip_count
        assert all(prim.IsA(UsdGeom.BasisCurves) for prim in grasp_bboxes)
        for prim in grasp_bboxes:
            curve = UsdGeom.BasisCurves(prim)
            assert list(curve.GetCurveVertexCountsAttr().Get()) == [5]
            points = list(curve.GetPointsAttr().Get())
            assert len(points) == 5 and points[0] == points[-1]
            assert max(float(point[2]) for point in points) == min(
                float(point[2]) for point in points
            )

        # Verify lowest-layer filtering in world coordinates without
        # comparing against a live articulation that can drift after a step.
        lowest_test = UsdGeom.Mesh.Define(
            extension._stage(), "/World/HandGripPresetTool/LowestLayerTest"
        )
        lowest_test.CreatePointsAttr().Set(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(2.0, 1.0, 0.0004),
                Gf.Vec3f(50.0, 60.0, 0.001),
                Gf.Vec3f(-70.0, -80.0, 1.0),
            ]
        )
        UsdGeom.Xformable(lowest_test).AddTranslateOp().Set(Gf.Vec3d(10.0, 20.0, 30.0))
        lowest_bound, lowest_center = extension._lowest_mesh_top_geometry(lowest_test.GetPrim())
        expected_lowest_bound = (10.0, 20.0, 60.0, 80.0, 30.0)
        expected_lowest_center = (
            10.0 + 52.0 / 3.0,
            20.0 + 61.0 / 3.0,
            30.0 + 0.0014 / 3.0,
        )
        assert max(
            abs(value - reference)
            for value, reference in zip(lowest_bound, expected_lowest_bound)
        ) < 1e-6, (lowest_bound, expected_lowest_bound)
        assert max(
            abs(value - reference)
            for value, reference in zip(lowest_center, expected_lowest_center)
        ) < 1e-6, (lowest_center, expected_lowest_center)
        extension._stage().RemovePrim(lowest_test.GetPath())

        # Verify the physical START synchronization independently of the drive
        # interpolation that follows it.
        extension._timeline.play()
        await omni.kit.app.get_app().next_update_async()
        extension._sync_articulation_to_pose(extension._start_pose)
        names = list(extension._start_pose.joints_rad)
        indices = [extension._articulation.get_dof_index(name) for name in names]
        actual = extension._articulation.get_joint_positions(joint_indices=indices)
        expected = [extension._start_pose.joints_rad[name] for name in names]
        assert max(abs(float(a) - b) for a, b in zip(actual, expected)) < 1e-4, (actual, expected)
        extension._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        # Run the complete preview path as well; its first visible frame is
        # shown only after the direct START synchronization above.
        extension._duration_model.set_value(0.1)
        await extension._preview_async()
        extension._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        candidates = extension._discover_object_usds(
            Path("/nas/ochansol/3d_model/peel3_scan_data_2026")
        )
        assert len(candidates) >= 250, len(candidates)
        extension._object_scale_model.set_value(0.1)
        extension._load_random_object()
        object_prim = extension._stage().GetPrimAtPath("/World/HandGripPresetTool/TestObject")
        assert object_prim.IsValid()
        assert object_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        meshes = [prim for prim in Usd.PrimRange(object_prim) if prim.IsA(UsdGeom.Mesh)]
        assert meshes
        assert all(
            UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == "convexDecomposition"
            for mesh in meshes
        )
        assert all(mesh.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI) for mesh in meshes)
        assert extension._stage().GetPrimAtPath("/World/HandGripPresetTool/GroundPlane").IsValid()
        extension._remove_test_object()

        with tempfile.TemporaryDirectory(prefix="hand_grip_smoke_") as directory:
            db_path = Path(directory) / "gripper_info_hand.json"
            shutil.copy2(
                "/nas/ochansol/gripper_info/gripper_info_hand.json",
                db_path,
            )
            extension._db_path_model.set_value(str(db_path))
            extension._preset_name_model.set_value("__smoke_test__")
            await extension._save_preset_async()
            saved = json.loads(db_path.read_text(encoding="utf-8"))
            presets = saved["Inspire-F1"]["preset"]
            result = next(item for item in presets if item["name"] == "__smoke_test__")
            assert len(result["start_joint_pos"]) == 6
            assert len(result["end_joint_pos"]) == 6
            assert result["joint_unit"] == "rad"
            assert result["start_base_tf"]["frame"] == "world"
            assert "tip_links" not in saved["Inspire-F1"]
            assert "tip_link_names" not in saved["Inspire-F1"]
            assert len(result["fingertip_points"]) == 1
            assert "right_hand_middle_2" not in result["fingertip_points"]
            contact = result["fingertip_points"]["right_hand_index_2"]
            assert contact["mesh_path"] == index_path
            assert contact["frame"] == "gripper_base"
            assert contact["base_prim_path"] == "/World/HandGripPresetTool/Hand"
            assert len(contact["grasp_bbox"]["points"]) == 4
            assert all(len(point) == 3 for point in contact["grasp_bbox"]["points"])
            assert contact["grasp_bbox"]["projection_axis"] == "world_z"
            assert contact["grasp_bbox"]["lowest_z_band"] == 0.005
            assert "end_center" not in contact["grasp_bbox"]
            grasp_center = result["grasp_center"]
            assert len(grasp_center["point"]) == 3
            assert "calculation" not in grasp_center
            assert set(grasp_center["source_mesh_paths"]) == {
                value["mesh_path"] for value in result["fingertip_points"].values()
            }
            assert all(
                len(preset.get("grasp_center", {}).get("point", [])) == 3
                and "calculation" not in preset["grasp_center"]
                for preset in presets
            )
            start_tf = result["start_base_tf"]
            w, x, y, z = start_tf["orientation_wxyz"]
            start_rotation = Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
            start_translation = Gf.Vec3d(*start_tf["position"])
            reconstructed = [
                start_rotation.TransformDir(Gf.Vec3d(*point)) + start_translation
                for point in contact["grasp_bbox"]["points"]
            ]
            assert max(
                abs(float(point[2]) - contact["grasp_bbox"]["world_min_z"])
                for point in reconstructed
            ) < 1e-6

            extension._refresh_db_presets(
                auto_load=False, report=False, preferred_name="__smoke_test__"
            )
            loaded_name = extension._load_saved_preset(extension._saved_preset_index, report=False)
            assert loaded_name == "__smoke_test__"
            assert len(extension._start_pose.joints_rad) == 6
            assert len(extension._end_pose.joints_rad) == 6
            assert set(extension._tip_links) == {
                value["mesh_path"] for value in result["fingertip_points"].values()
            }

        print("HAND_GRIP_SMOKE_TEST: PASS")
    except Exception:
        return_code = 2
        traceback.print_exc()
        print("HAND_GRIP_SMOKE_TEST: FAIL")
    finally:
        omni.kit.app.get_app().post_quit(return_code)


asyncio.ensure_future(_run())
