"""Isaac Sim extension for authoring hand-gripper start/end pose presets.

The extension intentionally uses USD/PhysX schemas instead of a version-specific
Isaac Core articulation wrapper.  It therefore works across Isaac Sim 4.5, 5.x,
and newer Kit builds as long as the dependencies in ``extension.toml`` exist.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import os
import random
import shutil
import tempfile
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import carb
import carb.settings
import omni.ext
import omni.kit.app
import omni.timeline
import omni.ui as ui
import omni.usd
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade


WINDOW_TITLE = "Hand Grip Preset Maker"
TOOL_ROOT_PATH = "/World/HandGripPresetTool"
HAND_PATH = f"{TOOL_ROOT_PATH}/Hand"
HAND_ASSET_PATH = f"{HAND_PATH}/Asset"
OBJECT_PATH = f"{TOOL_ROOT_PATH}/TestObject"
OBJECT_ASSET_PATH = f"{OBJECT_PATH}/Asset"
PLANE_PATH = f"{TOOL_ROOT_PATH}/GroundPlane"
DEBUG_BBOX_ROOT_PATH = f"{TOOL_ROOT_PATH}/FingertipBBoxes"
LEGACY_DEBUG_BBOX_ROOT_PATH = f"{HAND_PATH}/FingertipBBoxes"
GRASP_BBOX_CURVE_PATH = f"{DEBUG_BBOX_ROOT_PATH}/GraspBBoxes"
END_GRASP_CENTER_PATH = f"{DEBUG_BBOX_ROOT_PATH}/EndGraspCenter"
PHYSICS_SCENE_PATH = "/World/physicsScene"

DEFAULT_USD_PATH = "/nas/ochansol/isaac/USD/robots/gripper/Hand/Inspire-F1/Inspire-F1.usd"
DEFAULT_DB_PATH = "/nas/ochansol/gripper_info/gripper_info_hand.json"
DEFAULT_OBJECT_FOLDER = "/nas/ochansol/3d_model/peel3_scan_data_2026"
DEFAULT_GRIPPER_KEY = "Inspire-F1"
GRASP_LOWEST_Z_BAND = 0.005


@dataclass
class PoseSnapshot:
    """World base transform and controlled joint targets (radians)."""

    position: Tuple[float, float, float]
    orientation_wxyz: Tuple[float, float, float, float]
    rpy_rad: Tuple[float, float, float]
    joints_rad: Dict[str, float]

    def base_tf_json(self) -> dict:
        return {
            "frame": "world",
            "position": [round(v, 9) for v in self.position],
            "orientation_wxyz": [round(v, 9) for v in self.orientation_wxyz],
            "rpy_rad": [round(v, 9) for v in self.rpy_rad],
        }


@dataclass
class JointInfo:
    name: str
    prim_path: str
    lower_rad: float
    upper_rad: float
    kind: str


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _rpy_from_quat(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi * 0.5, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return roll, pitch, math.atan2(siny_cosp, cosy_cosp)


def _normalize_quat(q: Iterable[float]) -> Tuple[float, float, float, float]:
    values = tuple(float(v) for v in q)
    norm = math.sqrt(sum(v * v for v in values))
    if norm < 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    return tuple(v / norm for v in values)  # type: ignore[return-value]


def _slerp(q0: Iterable[float], q1: Iterable[float], t: float) -> Tuple[float, float, float, float]:
    a = _normalize_quat(q0)
    b = _normalize_quat(q1)
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b = tuple(-v for v in b)
        dot = -dot
    if dot > 0.9995:
        return _normalize_quat(x + t * (y - x) for x, y in zip(a, b))
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    s0 = math.sin((1.0 - t) * theta_0) / sin_theta_0
    s1 = math.sin(t * theta_0) / sin_theta_0
    return tuple(s0 * x + s1 * y for x, y in zip(a, b))  # type: ignore[return-value]


class HandGripPresetMakerExtension(omni.ext.IExt):
    """Interactive extension entry point."""

    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._window: Optional[ui.Window] = None
        self._menu_items = [MenuItemDescription(name=WINDOW_TITLE, onclick_fn=self._toggle_window)]
        add_menu_items(self._menu_items, "Tools")

        self._timeline = omni.timeline.get_timeline_interface()
        self._joint_infos: Dict[str, JointInfo] = {}
        self._joint_models: Dict[str, ui.AbstractValueModel] = {}
        self._base_models: Dict[str, ui.AbstractValueModel] = {}
        self._start_pose: Optional[PoseSnapshot] = None
        self._end_pose: Optional[PoseSnapshot] = None
        self._saved_presets: List[dict] = []
        self._saved_preset_names: List[str] = []
        self._saved_preset_index = 0
        self._saved_preset_combo_model = None
        self._matched_db_key: Optional[str] = None
        self._tip_links: Dict[str, Tuple[str, ...]] = {}
        self._disabled_dangling_joints: List[str] = []
        self._articulation = None
        self._preview_task: Optional[asyncio.Task] = None
        self._bbox_task: Optional[asyncio.Task] = None
        self._save_task: Optional[asyncio.Task] = None
        self._suppress_callbacks = False

        self._window = ui.Window(WINDOW_TITLE, width=520, height=840, visible=True)
        self._window.set_visibility_changed_fn(self._on_visibility_changed)
        self._build_ui()
        carb.log_info(f"[{WINDOW_TITLE}] Extension started")

    def on_shutdown(self) -> None:
        self._cancel_preview(stop_timeline=False)
        self._cancel_bbox_task(stop_timeline=False)
        self._cancel_save_task()
        if getattr(self, "_menu_items", None):
            remove_menu_items(self._menu_items, "Tools")
        self._window = None
        self._joint_models.clear()
        carb.log_info(f"[{WINDOW_TITLE}] Extension stopped")

    # ------------------------------------------------------------------ UI
    def _toggle_window(self) -> None:
        if self._window:
            self._window.visible = not self._window.visible

    def _on_visibility_changed(self, visible: bool) -> None:
        if visible and self._window:
            self._window.focus()

    def _build_ui(self) -> None:
        assert self._window is not None
        with self._window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=8, height=0):
                    ui.Label(
                        "USD를 불러온 뒤 base TF와 6개 drive joint를 조절하고 Start/End를 저장하세요.",
                        word_wrap=True,
                        height=42,
                    )

                    with ui.CollapsableFrame("1. Gripper load", collapsed=False):
                        with ui.VStack(spacing=5, height=0):
                            self._usd_model = self._string_row("USD", DEFAULT_USD_PATH)
                            self._urdf_model = self._string_row("URDF", str(Path(DEFAULT_USD_PATH).with_suffix(".urdf")))
                            self._gripper_key_model = self._string_row("DB key", DEFAULT_GRIPPER_KEY)
                            with ui.HStack(height=28, spacing=5):
                                ui.Button("Load / Reload", clicked_fn=lambda: self._guard(self._load_gripper))
                                ui.Button("Select base", clicked_fn=lambda: self._guard(self._select_base))
                                ui.Button("Live control", clicked_fn=lambda: self._guard(self._start_live_control))
                                ui.Button("Pause", clicked_fn=self._timeline.pause)

                    with ui.CollapsableFrame("2. Base TF (world, m / deg)", collapsed=False):
                        with ui.VStack(spacing=4, height=0):
                            for key, label in (
                                ("x", "X"), ("y", "Y"), ("z", "Z"),
                                ("roll", "Roll"), ("pitch", "Pitch"), ("yaw", "Yaw"),
                            ):
                                self._base_models[key] = self._float_row(
                                    label, 0.0, -1000.0, 1000.0,
                                    lambda _m, k=key: self._on_base_changed(k),
                                )
                            with ui.HStack(height=28, spacing=5):
                                ui.Button("Apply fields", clicked_fn=lambda: self._guard(self._apply_base_fields))
                                ui.Button("Read viewport TF", clicked_fn=lambda: self._guard(self._refresh_base_fields))

                    with ui.CollapsableFrame("3. Joint positions (rad)", collapsed=False):
                        with ui.VStack(spacing=4, height=0):
                            self._joint_frame = ui.Frame(height=0)
                            self._joint_frame.set_build_fn(self._build_joint_rows)
                            with ui.HStack(height=28, spacing=5):
                                ui.Button("Open all", clicked_fn=lambda: self._guard(lambda: self._set_all_joints(False)))
                                ui.Button("Close all", clicked_fn=lambda: self._guard(lambda: self._set_all_joints(True)))
                                ui.Button("Apply gains", clicked_fn=lambda: self._guard(self._apply_drive_gains))
                            self._stiffness_model = self._float_row("Stiffness", 500.0, 0.0, 1_000_000.0)
                            self._damping_model = self._float_row("Damping", 10.0, 0.0, 1_000_000.0)
                            self._max_force_model = self._float_row("Max force", 15.0, 0.0, 1_000_000.0)

                    with ui.CollapsableFrame("4. Start / End and preview", collapsed=False):
                        with ui.VStack(spacing=5, height=0):
                            with ui.HStack(height=30, spacing=5):
                                ui.Button("Save START", clicked_fn=lambda: self._guard(lambda: self._capture_pose("start")))
                                ui.Button("Restore START", clicked_fn=lambda: self._guard(lambda: self._restore_pose("start")))
                            self._start_label = ui.Label("START: not captured", word_wrap=True, height=34)
                            with ui.HStack(height=30, spacing=5):
                                ui.Button("Save END", clicked_fn=lambda: self._guard(lambda: self._capture_pose("end")))
                                ui.Button("Restore END", clicked_fn=lambda: self._guard(lambda: self._restore_pose("end")))
                            self._end_label = ui.Label("END: not captured", word_wrap=True, height=34)
                            self._saved_preset_frame = ui.Frame(height=26)
                            self._saved_preset_frame.set_build_fn(self._build_saved_preset_row)
                            with ui.HStack(height=30, spacing=5):
                                ui.Button(
                                    "Load saved preset",
                                    clicked_fn=lambda: self._guard(self._load_selected_saved_preset),
                                )
                                ui.Button(
                                    "Refresh DB presets",
                                    clicked_fn=lambda: self._guard(lambda: self._refresh_db_presets(auto_load=False)),
                                )
                            self._duration_model = self._float_row("Duration (sec)", 2.0, 0.1, 60.0)
                            with ui.HStack(height=30, spacing=5):
                                ui.Button("Preview START -> END", clicked_fn=self._start_preview)
                                ui.Button("Stop", clicked_fn=lambda: self._cancel_preview(stop_timeline=True))

                    with ui.CollapsableFrame("5. Fingertip grasp BBox", collapsed=False):
                        with ui.VStack(spacing=5, height=0):
                            ui.Label(
                                "Viewport에서 fingertip의 mesh 또는 link를 선택한 뒤 등록하세요. START/END에서 값이 변한 joint의 finger만 표시합니다.",
                                word_wrap=True,
                                height=40,
                            )
                            with ui.HStack(height=30, spacing=5):
                                ui.Button(
                                    "Add selected tip",
                                    clicked_fn=lambda: self._guard(self._add_selected_tip_links),
                                )
                                ui.Button(
                                    "Remove selected",
                                    clicked_fn=lambda: self._guard(self._remove_selected_tip_links),
                                )
                                ui.Button("Clear tips", clicked_fn=lambda: self._guard(self._clear_tip_links))
                            self._tip_links_frame = ui.Frame(height=0)
                            self._tip_links_frame.set_build_fn(self._build_tip_link_rows)
                            self._joint_delta_model = self._float_row("Joint delta tol", 0.0001, 0.0, 10.0)
                            self._bbox_line_width_model = self._float_row("Line width", 0.002, 0.00001, 1.0)
                            self._center_radius_model = self._float_row("Center radius", 0.004, 0.00001, 1.0)
                            self._bbox_z_offset_model = self._float_row("Z offset from min", 0.0, -1.0, 1.0)
                            with ui.HStack(height=30, spacing=5):
                                ui.Button("Generate grasp BBoxes", clicked_fn=self._start_bbox_generation)
                                ui.Button(
                                    "Clear debug lines",
                                    clicked_fn=lambda: self._guard(self._clear_tip_bboxes),
                                )
                            ui.Label(
                                "BBox들과 END 시점 모든 선택 손가락 사이의 전체 grasp center를 표시합니다.",
                                word_wrap=True,
                                height=34,
                            )

                    with ui.CollapsableFrame("6. Random grasp object", collapsed=True):
                        with ui.VStack(spacing=5, height=0):
                            self._object_folder_model = self._string_row("Object folder", DEFAULT_OBJECT_FOLDER)
                            self._object_scale_model = self._float_row("Uniform scale", 1.0, 0.0001, 1000.0)
                            self._object_x_model = self._float_row("Spawn X", 0.0, -100.0, 100.0)
                            self._object_y_model = self._float_row("Spawn Y", 0.0, -100.0, 100.0)
                            self._object_z_model = self._float_row("Spawn Z", 0.10, -100.0, 100.0)
                            self._plane_z_model = self._float_row("Plane Z", 0.0, -100.0, 100.0)
                            with ui.HStack(height=30, spacing=5):
                                ui.Button("Load random + plane", clicked_fn=lambda: self._guard(self._load_random_object))
                                ui.Button("Remove object", clicked_fn=lambda: self._guard(self._remove_test_object))
                            ui.Label(
                                "objects_conf.json의 edited USD를 우선 사용하며 convex decomposition, rigid body, 물리 재질을 자동 적용합니다.",
                                word_wrap=True,
                                height=34,
                            )

                    with ui.CollapsableFrame("7. Save database preset", collapsed=False):
                        with ui.VStack(spacing=5, height=0):
                            self._preset_name_model = self._string_row("Preset name", "new_grip")
                            self._db_path_model = self._string_row("JSON DB", DEFAULT_DB_PATH)
                            ui.Button("Save + recompute all presets", height=32, clicked_fn=self._start_preset_save)
                            ui.Label("현재 preset 저장 후 모든 preset의 BBox/center를 순차 재계산합니다.", height=24)

                    self._status_label = ui.Label("Ready", word_wrap=True, height=58)

    def _string_row(self, label: str, value: str) -> ui.AbstractValueModel:
        with ui.HStack(height=24, spacing=5):
            ui.Label(label, width=110)
            field = ui.StringField()
            field.model.set_value(value)
        return field.model

    def _float_row(
        self,
        label: str,
        value: float,
        minimum: float,
        maximum: float,
        callback=None,
    ) -> ui.AbstractValueModel:
        with ui.HStack(height=24, spacing=5):
            ui.Label(label, width=110)
            drag = ui.FloatDrag(min=minimum, max=maximum, step=0.001)
            drag.model.set_value(value)
            if callback:
                drag.model.add_value_changed_fn(callback)
        return drag.model

    def _build_joint_rows(self) -> None:
        with ui.VStack(spacing=4, height=0):
            if not self._joint_infos:
                ui.Label("Load a gripper to discover driven joints.", height=26)
                return
            for name, info in self._joint_infos.items():
                model = self._joint_models.setdefault(name, ui.SimpleFloatModel(info.lower_rad))
                with ui.HStack(height=25, spacing=5):
                    ui.Label(name, width=170)
                    ui.FloatSlider(model, min=info.lower_rad, max=info.upper_rad, step=0.001)
                    ui.FloatDrag(model, min=info.lower_rad, max=info.upper_rad, step=0.001, width=85)
                    model.add_value_changed_fn(lambda changed_model, n=name: self._on_joint_changed(n, changed_model))

    def _build_saved_preset_row(self) -> None:
        items = self._saved_preset_names or ["(no matching preset)"]
        index = min(self._saved_preset_index, len(items) - 1)
        with ui.HStack(height=26, spacing=5):
            ui.Label("Saved preset", width=110)
            combo = ui.ComboBox(index, *items)
            self._saved_preset_combo_model = combo.model

    def _build_tip_link_rows(self) -> None:
        with ui.VStack(spacing=4, height=0):
            if not self._tip_links:
                ui.Label("Fingertips: none", height=26)
                return
            for relative_path, joints in self._tip_links.items():
                mesh_name = self._default_tip_name(relative_path)
                joint_text = ", ".join(joints)
                with ui.HStack(height=25, spacing=5):
                    ui.Label(f"{mesh_name} <- {joint_text}", tooltip=relative_path)

    def _set_status(self, message: str, error: bool = False) -> None:
        if getattr(self, "_status_label", None):
            prefix = "ERROR: " if error else ""
            self._status_label.text = prefix + message
        (carb.log_error if error else carb.log_info)(f"[{WINDOW_TITLE}] {message}")

    def _guard(self, fn) -> None:
        try:
            fn()
        except Exception as exc:
            self._set_status(str(exc), error=True)
            carb.log_error(traceback.format_exc())

    # -------------------------------------------------------------- stage/USD
    def _stage(self) -> Usd.Stage:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Open or create a USD stage first.")
        return stage

    def _hand_prim(self) -> Usd.Prim:
        prim = self._stage().GetPrimAtPath(HAND_PATH)
        if not prim.IsValid():
            raise RuntimeError("Load the gripper first.")
        return prim

    def _load_gripper(self) -> None:
        usd_path = self._usd_model.get_value_as_string().strip()
        if not usd_path or not os.path.isfile(usd_path):
            raise FileNotFoundError(f"USD file not found: {usd_path}")
        guessed_urdf = str(Path(usd_path).with_suffix(".urdf"))
        urdf_path = self._urdf_model.get_value_as_string().strip()
        if not urdf_path or not os.path.isfile(urdf_path):
            urdf_path = guessed_urdf
            self._urdf_model.set_value(urdf_path)

        self._cancel_preview(stop_timeline=True)
        self._cancel_bbox_task(stop_timeline=True)
        self._cancel_save_task()
        self._articulation = None
        stage = self._stage()
        for debug_path in (DEBUG_BBOX_ROOT_PATH, LEGACY_DEBUG_BBOX_ROOT_PATH):
            if stage.GetPrimAtPath(debug_path).IsValid():
                stage.RemovePrim(debug_path)
        self._tip_links = {}
        self._refresh_tip_links_label()
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, TOOL_ROOT_PATH)
        if stage.GetPrimAtPath(HAND_PATH).IsValid():
            stage.RemovePrim(HAND_PATH)
        UsdGeom.Xform.Define(stage, HAND_PATH)
        asset = UsdGeom.Xform.Define(stage, HAND_ASSET_PATH).GetPrim()
        if not asset.GetReferences().AddReference(usd_path):
            raise RuntimeError(f"Failed to add USD reference: {usd_path}")
        self._set_local_transform(HAND_PATH, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

        self._disabled_dangling_joints = self._disable_dangling_joints(asset)
        self._joint_infos = self._discover_driven_joints(asset)
        urdf_mimic_names = self._read_urdf_mimic_joint_names(urdf_path) if os.path.isfile(urdf_path) else set()
        self._joint_infos = {
            name: info for name, info in self._joint_infos.items() if name not in urdf_mimic_names
        }
        if not self._joint_infos:
            raise RuntimeError("No revolute/prismatic joint with PhysicsDriveAPI was found in the USD.")
        self._joint_models = {
            name: ui.SimpleFloatModel(info.lower_rad) for name, info in self._joint_infos.items()
        }
        self._joint_frame.rebuild()
        self._refresh_base_fields()
        if self._has_app_window():
            self._select_base()
        self._ensure_physics_scene()
        self._start_pose = None
        self._end_pose = None
        self._start_label.text = "START: not captured"
        self._end_label.text = "END: not captured"

        urdf_summary = self._read_urdf_summary(urdf_path) if os.path.isfile(urdf_path) else None
        message = f"Loaded {Path(usd_path).name}; {len(self._joint_infos)} driven joints"
        if urdf_summary:
            overlap = sorted(set(self._joint_infos) & set(urdf_summary))
            message += f"; URDF active joints={len(urdf_summary)}, matching names={len(overlap)}"
            if not overlap:
                message += " (USD uses coupled/renamed drive joints, so USD limits are authoritative)"
        if self._disabled_dangling_joints:
            message += f"; disabled dangling joints={len(self._disabled_dangling_joints)}"
        loaded_preset = self._refresh_db_presets(auto_load=True, report=False)
        if self._matched_db_key:
            message += f"; DB matched={self._matched_db_key}"
        if loaded_preset:
            message += f"; loaded preset={loaded_preset}"
        self._set_status(message)

    def _disable_dangling_joints(self, root: Usd.Prim) -> List[str]:
        """Disable joints whose body relationships target missing prims.

        PhysX 5.1 reports these as stage errors when simulation starts.  The
        Inspire-F1 source asset contains ten such fixed joints for omitted
        fingertip/force-sensor bodies; disabling them does not affect its six
        valid drive joints.
        """
        stage = root.GetStage()
        disabled: List[str] = []
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            targets = list(joint.GetBody0Rel().GetTargets()) + list(joint.GetBody1Rel().GetTargets())
            if any(not stage.GetPrimAtPath(target).IsValid() for target in targets):
                # ``physics:jointEnabled = false`` still makes the 5.1 parser
                # validate dangling body relationships.  Deactivating the
                # broken prim in this reference instance prevents that parse
                # error while leaving the source USD untouched.
                prim.SetActive(False)
                disabled.append(str(prim.GetPath()))
        return disabled

    def _has_app_window(self) -> bool:
        """Return false for ``--no-window`` and headless SimulationApp runs."""
        return carb.settings.get_settings().get_as_bool("/app/window/enabled")

    def _discover_driven_joints(self, root: Usd.Prim) -> Dict[str, JointInfo]:
        result: Dict[str, JointInfo] = {}
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(PhysxSchema.PhysxMimicJointAPI):
                continue
            if prim.IsA(UsdPhysics.RevoluteJoint):
                kind, drive_name, scale = "revolute", "angular", math.pi / 180.0
                joint_with_limits = UsdPhysics.RevoluteJoint(prim)
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                kind, drive_name, scale = "prismatic", "linear", 1.0
                joint_with_limits = UsdPhysics.PrismaticJoint(prim)
            else:
                continue
            drive = UsdPhysics.DriveAPI.Get(prim, drive_name)
            if not drive or not drive.GetTargetPositionAttr().IsValid():
                continue
            lower = joint_with_limits.GetLowerLimitAttr().Get()
            upper = joint_with_limits.GetUpperLimitAttr().Get()
            lower_value = float(lower) * scale if lower is not None else -math.pi
            upper_value = float(upper) * scale if upper is not None else math.pi
            result[prim.GetName()] = JointInfo(
                name=prim.GetName(),
                prim_path=str(prim.GetPath()),
                lower_rad=lower_value,
                upper_rad=upper_value,
                kind=kind,
            )
        return dict(sorted(result.items()))

    def _read_urdf_summary(self, path: str) -> Dict[str, Tuple[float, float]]:
        result: Dict[str, Tuple[float, float]] = {}
        root = ET.parse(path).getroot()
        for joint in root.findall("joint"):
            if joint.get("type") not in ("revolute", "continuous", "prismatic"):
                continue
            if joint.find("mimic") is not None:
                continue
            limit = joint.find("limit")
            lower = float(limit.get("lower", "-3.141592653589793")) if limit is not None else -math.pi
            upper = float(limit.get("upper", "3.141592653589793")) if limit is not None else math.pi
            result[joint.get("name", "unnamed")] = (lower, upper)
        return result

    def _read_urdf_mimic_joint_names(self, path: str) -> set[str]:
        root = ET.parse(path).getroot()
        return {
            joint.get("name", "")
            for joint in root.findall("joint")
            if joint.find("mimic") is not None and joint.get("name")
        }

    def _select_base(self) -> None:
        self._hand_prim()
        omni.usd.get_context().get_selection().set_selected_prim_paths([HAND_PATH], True)
        self._set_status(f"Selected {HAND_PATH}. The viewport transform gizmo can now edit the base TF.")

    def _ensure_physics_scene(self) -> None:
        stage = self._stage()
        if not stage.GetPrimAtPath(PHYSICS_SCENE_PATH).IsValid():
            scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH)
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)

    # ------------------------------------------------------------ transforms
    def _world_pose(self, prim: Usd.Prim) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        matrix = omni.usd.get_world_transform_matrix(prim)
        translation = matrix.ExtractTranslation()
        quat = matrix.ExtractRotationQuat()
        imaginary = quat.GetImaginary()
        return (
            tuple(float(v) for v in translation),
            _normalize_quat((quat.GetReal(), imaginary[0], imaginary[1], imaginary[2])),
        )

    def _set_local_transform(
        self,
        prim_path: str,
        position: Iterable[float],
        orientation_wxyz: Iterable[float],
        scale: Optional[Iterable[float]] = None,
    ) -> None:
        prim = self._stage().GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid prim: {prim_path}")
        q = _normalize_quat(orientation_wxyz)
        api = UsdGeom.XformCommonAPI(prim)
        api.SetTranslate(Gf.Vec3d(*[float(v) for v in position]))
        api.SetRotate(Gf.Vec3f(*[math.degrees(v) for v in _rpy_from_quat(q)]), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        if scale is not None:
            api.SetScale(Gf.Vec3f(*[float(v) for v in scale]))

    def _refresh_base_fields(self, report: bool = True) -> None:
        position, quat = self._world_pose(self._hand_prim())
        rpy_deg = tuple(math.degrees(v) for v in _rpy_from_quat(quat))
        self._suppress_callbacks = True
        try:
            for key, value in zip(("x", "y", "z"), position):
                self._base_models[key].set_value(value)
            for key, value in zip(("roll", "pitch", "yaw"), rpy_deg):
                self._base_models[key].set_value(value)
        finally:
            self._suppress_callbacks = False
        if report:
            self._set_status("Base TF fields refreshed from the viewport/world transform.")

    def _apply_base_fields(self) -> None:
        position = tuple(self._base_models[k].get_value_as_float() for k in ("x", "y", "z"))
        rpy = tuple(math.radians(self._base_models[k].get_value_as_float()) for k in ("roll", "pitch", "yaw"))
        self._set_local_transform(HAND_PATH, position, _quat_from_rpy(*rpy))

    def _on_base_changed(self, _key: str) -> None:
        if self._suppress_callbacks:
            return
        try:
            self._apply_base_fields()
        except Exception:
            carb.log_warn(f"[{WINDOW_TITLE}] Base edit ignored until a hand is loaded")

    # --------------------------------------------------------------- joints
    def _on_joint_changed(self, name: str, model: ui.AbstractValueModel) -> None:
        if self._suppress_callbacks:
            return
        try:
            self._set_joint_target(name, model.get_value_as_float(), set_initial_state=self._timeline.is_stopped())
        except Exception as exc:
            self._set_status(str(exc), error=True)

    def _set_joint_target(self, name: str, value_rad: float, set_initial_state: bool = False) -> None:
        info = self._joint_infos[name]
        value_rad = max(info.lower_rad, min(info.upper_rad, float(value_rad)))
        prim = self._stage().GetPrimAtPath(info.prim_path)
        drive_name = "angular" if info.kind == "revolute" else "linear"
        drive = UsdPhysics.DriveAPI.Get(prim, drive_name)
        usd_value = math.degrees(value_rad) if info.kind == "revolute" else value_rad
        drive.GetTargetPositionAttr().Set(usd_value)
        if set_initial_state:
            state = PhysxSchema.JointStateAPI.Apply(prim, drive_name)
            state.CreatePositionAttr().Set(usd_value)

    def _set_all_joints(self, close: bool) -> None:
        self._suppress_callbacks = True
        try:
            for name, info in self._joint_infos.items():
                value = info.upper_rad if close else info.lower_rad
                self._joint_models[name].set_value(value)
                self._set_joint_target(name, value, set_initial_state=self._timeline.is_stopped())
        finally:
            self._suppress_callbacks = False
        self._set_status("Applied closed joint limits." if close else "Applied open joint limits.")

    def _apply_drive_gains(self) -> None:
        stiffness = self._stiffness_model.get_value_as_float()
        damping = self._damping_model.get_value_as_float()
        max_force = self._max_force_model.get_value_as_float()
        stage = self._stage()
        for info in self._joint_infos.values():
            drive_name = "angular" if info.kind == "revolute" else "linear"
            drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(info.prim_path), drive_name)
            drive.CreateStiffnessAttr().Set(stiffness)
            drive.CreateDampingAttr().Set(damping)
            drive.CreateMaxForceAttr().Set(max_force)
        self._set_status(f"Drive gains applied to {len(self._joint_infos)} joints.")

    def _start_live_control(self) -> None:
        self._hand_prim()
        self._ensure_physics_scene()
        self._apply_drive_gains()
        self._timeline.play()
        self._set_status("Timeline is playing. Joint sliders now command PhysX drive targets in real time.")

    # ---------------------------------------------------------- pose/preview
    def _snapshot(self) -> PoseSnapshot:
        position, quat = self._world_pose(self._hand_prim())
        joints = {name: model.get_value_as_float() for name, model in self._joint_models.items()}
        if set(joints) != set(self._joint_infos):
            raise RuntimeError("Joint UI is not initialized. Reload the gripper.")
        return PoseSnapshot(position, quat, _rpy_from_quat(quat), joints)

    def _capture_pose(self, which: str) -> None:
        pose = self._snapshot()
        if which == "start":
            self._start_pose = pose
            self._start_label.text = self._pose_label("START", pose)
        else:
            self._end_pose = pose
            self._end_label.text = self._pose_label("END", pose)
        self._set_status(f"Captured {which.upper()} pose ({len(pose.joints_rad)} joints).")

    def _pose_label(self, name: str, pose: PoseSnapshot) -> str:
        p = ", ".join(f"{v:.3f}" for v in pose.position)
        return f"{name}: base=({p}), joints={len(pose.joints_rad)}"

    def _restore_pose(self, which: str) -> None:
        pose = self._start_pose if which == "start" else self._end_pose
        if pose is None:
            raise RuntimeError(f"{which.upper()} pose has not been captured.")
        self._apply_pose(pose, set_initial_state=self._timeline.is_stopped())
        self._set_status(f"Restored {which.upper()} pose.")

    def _apply_pose(self, pose: PoseSnapshot, set_initial_state: bool = False) -> None:
        self._set_local_transform(HAND_PATH, pose.position, pose.orientation_wxyz)
        self._suppress_callbacks = True
        try:
            for name, value in pose.joints_rad.items():
                if name not in self._joint_infos:
                    continue
                self._joint_models[name].set_value(value)
                self._set_joint_target(name, value, set_initial_state=set_initial_state)
        finally:
            self._suppress_callbacks = False
        self._refresh_base_fields(report=False)

    def _start_preview(self) -> None:
        try:
            if self._start_pose is None or self._end_pose is None:
                raise RuntimeError("Capture both START and END before preview.")
            self._cancel_bbox_task(stop_timeline=True)
            self._cancel_preview(stop_timeline=True)
            self._preview_task = asyncio.ensure_future(self._preview_async())
        except Exception as exc:
            self._set_status(str(exc), error=True)

    async def _preview_async(self) -> None:
        assert self._start_pose is not None and self._end_pose is not None
        visibility_attr = UsdGeom.Imageable(self._hand_prim()).GetVisibilityAttr()
        original_visibility = visibility_attr.Get() or UsdGeom.Tokens.inherited
        try:
            self._ensure_physics_scene()
            self._apply_drive_gains()
            self._apply_pose(self._start_pose, set_initial_state=True)
            await omni.kit.app.get_app().next_update_async()

            # Hide only during PhysX handle creation and the one-time teleport,
            # so a zero-valued initialization frame can never be rendered.
            visibility_attr.Set(UsdGeom.Tokens.invisible)
            self._timeline.play()

            # USD JointState values do not overwrite an already-created PhysX
            # articulation state.  Once Play creates the handles, teleport the
            # controlled DOFs to START so preview contains only START -> END.
            await omni.kit.app.get_app().next_update_async()
            self._sync_articulation_to_pose(self._start_pose)
            self._apply_pose(self._start_pose, set_initial_state=False)
            await omni.kit.app.get_app().next_update_async()
            visibility_attr.Set(original_visibility)

            duration = max(0.1, self._duration_model.get_value_as_float())
            steps = max(2, int(duration * 60.0))
            for index in range(1, steps + 1):
                linear_t = index / steps
                t = linear_t * linear_t * (3.0 - 2.0 * linear_t)  # smoothstep
                pose = self._interpolate_pose(self._start_pose, self._end_pose, t)
                self._apply_pose(pose, set_initial_state=False)
                await omni.kit.app.get_app().next_update_async()
            self._timeline.pause()
            self._set_status("Preview finished at END pose; timeline paused for inspection.")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._set_status(f"Preview failed: {exc}", error=True)
            carb.log_error(traceback.format_exc())
        finally:
            visibility_attr.Set(original_visibility)
            self._preview_task = None

    def _sync_articulation_to_pose(self, pose: PoseSnapshot) -> None:
        """Immediately set PhysX DOFs to ``pose`` after the timeline starts."""
        import numpy as np
        from isaacsim.core.prims import SingleArticulation

        articulation = SingleArticulation(
            prim_path=HAND_ASSET_PATH,
            name="hand_grip_preset_preview",
            reset_xform_properties=False,
        )
        articulation.initialize()

        missing = [name for name in pose.joints_rad if name not in articulation.dof_names]
        if missing:
            raise RuntimeError(f"Preview articulation is missing DOFs: {', '.join(missing)}")
        names = [name for name in pose.joints_rad if name in self._joint_infos]
        indices = np.asarray([articulation.get_dof_index(name) for name in names], dtype=np.int32)
        positions = np.asarray([pose.joints_rad[name] for name in names], dtype=np.float32)
        articulation.set_joint_positions(positions, joint_indices=indices)
        articulation.set_joint_velocities(np.zeros_like(positions), joint_indices=indices)
        self._articulation = articulation

    def _interpolate_pose(self, start: PoseSnapshot, end: PoseSnapshot, t: float) -> PoseSnapshot:
        position = tuple(a + (b - a) * t for a, b in zip(start.position, end.position))
        quat = _slerp(start.orientation_wxyz, end.orientation_wxyz, t)
        joints = {
            name: start.joints_rad[name] + (end.joints_rad[name] - start.joints_rad[name]) * t
            for name in start.joints_rad
            if name in end.joints_rad
        }
        return PoseSnapshot(position, quat, _rpy_from_quat(quat), joints)

    def _cancel_preview(self, stop_timeline: bool) -> None:
        task = getattr(self, "_preview_task", None)
        if task and not task.done():
            task.cancel()
        self._preview_task = None
        if stop_timeline and getattr(self, "_timeline", None):
            self._timeline.stop()
            self._articulation = None

    # ------------------------------------------------ fingertip top-view BBox
    def _refresh_tip_links_label(self) -> None:
        frame = getattr(self, "_tip_links_frame", None)
        if frame is not None:
            frame.rebuild()

    @staticmethod
    def _default_tip_name(relative_path: str) -> str:
        path = Path(relative_path)
        return path.parent.name if path.name == "mesh" else path.name

    def _add_selected_tip_links(self) -> None:
        selected_paths = omni.usd.get_context().get_selection().get_selected_prim_paths()
        if not selected_paths:
            raise RuntimeError("Select a fingertip mesh or rigid link in the viewport first.")
        added = 0
        errors = []
        for selected_path in selected_paths:
            prim = self._stage().GetPrimAtPath(selected_path)
            if not prim.IsValid():
                continue
            # A GeomSubset/material child selection is promoted to its mesh.
            candidate = prim
            while candidate.IsValid() and not candidate.IsA(UsdGeom.Mesh) and not candidate.HasAPI(UsdPhysics.RigidBodyAPI):
                if str(candidate.GetPath()) == HAND_ASSET_PATH:
                    break
                candidate = candidate.GetParent()
            try:
                self._register_tip_prim(candidate)
                added += 1
            except (RuntimeError, ValueError) as exc:
                errors.append(str(exc))
        if not added:
            raise RuntimeError(errors[0] if errors else "No selectable fingertip geometry was found.")
        self._refresh_tip_links_label()
        suffix = f" ({errors[0]})" if errors else ""
        self._set_status(f"Registered {added} fingertip link(s).{suffix}")

    def _register_tip_prim(self, prim: Usd.Prim) -> None:
        if not prim.IsValid():
            raise ValueError("Selected prim is invalid.")
        path = str(prim.GetPath())
        if path != HAND_ASSET_PATH and not path.startswith(HAND_ASSET_PATH + "/"):
            raise ValueError(f"Selection is outside the loaded hand: {path}")
        has_mesh = prim.IsA(UsdGeom.Mesh) or any(child.IsA(UsdGeom.Mesh) for child in Usd.PrimRange(prim))
        if not has_mesh:
            raise ValueError(f"Selected link contains no mesh: {path}")
        joints = self._controlled_joints_for_tip(prim)
        if not joints:
            raise ValueError(f"Selected link is not downstream of a controlled joint: {path}")
        relative_path = path[len(HAND_ASSET_PATH):]
        self._tip_links[relative_path] = joints

    def _controlled_joints_for_tip(self, tip_prim: Usd.Prim) -> Tuple[str, ...]:
        """Find controlled joints on the articulation-tree path to ``tip_prim``."""
        root = self._stage().GetPrimAtPath(HAND_ASSET_PATH)
        adjacency: Dict[str, List[Tuple[str, str]]] = {}
        body_paths = set()
        root_bodies = set()
        body1_paths = set()
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            body0 = [str(path) for path in joint.GetBody0Rel().GetTargets()]
            body1 = [str(path) for path in joint.GetBody1Rel().GetTargets()]
            body_paths.update(body0)
            body_paths.update(body1)
            body1_paths.update(body1)
            if not body0 and len(body1) == 1:
                root_bodies.add(body1[0])
            elif not body1 and len(body0) == 1:
                root_bodies.add(body0[0])
            for left in body0:
                for right in body1:
                    adjacency.setdefault(left, []).append((right, prim.GetName()))
                    adjacency.setdefault(right, []).append((left, prim.GetName()))

        tip_path = str(tip_prim.GetPath())
        candidates = [
            body for body in body_paths
            if tip_path == body or tip_path.startswith(body + "/")
        ]
        if not candidates:
            return ()
        tip_body = max(candidates, key=len)
        if not root_bodies:
            root_bodies = body_paths - body1_paths
        if not root_bodies and body_paths:
            root_bodies = {min(body_paths, key=len)}

        queue: List[Tuple[str, Tuple[str, ...]]] = [(body, ()) for body in sorted(root_bodies)]
        visited = set(root_bodies)
        while queue:
            body, path_joints = queue.pop(0)
            if body == tip_body:
                return tuple(name for name in path_joints if name in self._joint_infos)
            for neighbor, joint_name in adjacency.get(body, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path_joints + (joint_name,)))
        return ()

    def _remove_selected_tip_links(self) -> None:
        selected = omni.usd.get_context().get_selection().get_selected_prim_paths()
        if not selected:
            raise RuntimeError("Select a registered fingertip link to remove.")
        removed = []
        for relative_path in list(self._tip_links):
            full_path = HAND_ASSET_PATH + relative_path
            if any(
                path == full_path or path.startswith(full_path + "/") or full_path.startswith(path + "/")
                for path in selected
            ):
                removed.append(relative_path)
                self._tip_links.pop(relative_path, None)
        if not removed:
            raise RuntimeError("The selection does not match a registered fingertip link.")
        self._clear_tip_bboxes(report=False)
        self._refresh_tip_links_label()
        self._set_status(f"Removed {len(removed)} fingertip link(s).")

    def _clear_tip_links(self) -> None:
        self._tip_links.clear()
        self._clear_tip_bboxes(report=False)
        self._refresh_tip_links_label()
        self._set_status("Cleared all registered fingertip links and grasp BBoxes.")

    def _start_bbox_generation(self) -> None:
        try:
            if self._start_pose is None or self._end_pose is None:
                raise RuntimeError("Capture or load both START and END first.")
            if not self._tip_links:
                raise RuntimeError("Register at least one fingertip mesh/link first.")
            self._cancel_preview(stop_timeline=True)
            self._cancel_bbox_task(stop_timeline=True)
            self._bbox_task = asyncio.ensure_future(self._generate_tip_bboxes_async())
        except Exception as exc:
            self._set_status(str(exc), error=True)

    def _changed_joint_names(
        self,
        start_pose: PoseSnapshot,
        end_pose: PoseSnapshot,
    ) -> set:
        tolerance = max(0.0, self._joint_delta_model.get_value_as_float())
        return {
            name for name, start_value in start_pose.joints_rad.items()
            if name in end_pose.joints_rad
            and abs(end_pose.joints_rad[name] - start_value) > tolerance
        }

    def _moving_tip_paths(
        self,
        relative_paths: List[str],
        start_pose: PoseSnapshot,
        end_pose: PoseSnapshot,
    ) -> Tuple[set, List[str]]:
        """Return only tips downstream of a joint that changes in this preset."""
        changed_joints = self._changed_joint_names(start_pose, end_pose)
        moving_paths: List[str] = []
        for relative_path in relative_paths:
            joints = self._tip_links.get(relative_path)
            if joints is None:
                tip_prim = self._stage().GetPrimAtPath(HAND_ASSET_PATH + relative_path)
                joints = self._controlled_joints_for_tip(tip_prim) if tip_prim.IsValid() else ()
            if changed_joints.intersection(joints):
                moving_paths.append(relative_path)
        return changed_joints, moving_paths

    async def _generate_tip_bboxes_async(self) -> None:
        assert self._start_pose is not None and self._end_pose is not None
        try:
            tolerance = max(0.0, self._joint_delta_model.get_value_as_float())
            changed_joints, active_tip_paths = self._moving_tip_paths(
                list(self._tip_links), self._start_pose, self._end_pose
            )
            if not changed_joints:
                raise RuntimeError(f"No joint changes exceed tolerance {tolerance:g} rad.")
            if not active_tip_paths:
                raise RuntimeError("No registered fingertip belongs to a joint that changes from START to END.")

            self._clear_tip_bboxes(report=False)
            start_bounds, end_bounds, _start_centers, end_centers = (
                await self._sample_tip_bounds_async(active_tip_paths)
            )

            self._create_grasp_bboxes(
                active_tip_paths, start_bounds, end_bounds, end_centers
            )
            self._set_status(
                f"Generated grasp BBoxes for "
                f"{len(active_tip_paths)} moving fingertip(s); "
                f"changed joints={', '.join(sorted(changed_joints))}."
            )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._set_status(f"Grasp BBox generation failed: {exc}", error=True)
            carb.log_error(traceback.format_exc())
        finally:
            self._timeline.pause()
            self._bbox_task = None

    async def _sample_tip_bounds_async(
        self,
        relative_paths: List[str],
        z_offset: Optional[float] = None,
    ) -> Tuple[
        List[Tuple[float, float, float, float, float]],
        List[Tuple[float, float, float, float, float]],
        List[Tuple[float, float, float]],
        List[Tuple[float, float, float]],
    ]:
        """Evaluate fingertip bounds at the exact START and END articulation poses."""
        assert self._start_pose is not None and self._end_pose is not None
        current_pose = self._snapshot()
        visibility_attr = UsdGeom.Imageable(self._hand_prim()).GetVisibilityAttr()
        original_visibility = visibility_attr.Get() or UsdGeom.Tokens.inherited
        restored = False
        try:
            self._ensure_physics_scene()
            self._apply_drive_gains()
            self._apply_pose(self._start_pose, set_initial_state=True)
            await omni.kit.app.get_app().next_update_async()
            visibility_attr.Set(UsdGeom.Tokens.invisible)
            self._timeline.play()
            await omni.kit.app.get_app().next_update_async()

            self._sync_articulation_to_pose(self._start_pose)
            self._apply_pose(self._start_pose, set_initial_state=False)
            await omni.kit.app.get_app().next_update_async()
            start_bounds, start_centers = self._compute_tip_top_geometry(
                relative_paths, z_offset=z_offset
            )

            self._sync_articulation_to_pose(self._end_pose)
            self._apply_pose(self._end_pose, set_initial_state=False)
            await omni.kit.app.get_app().next_update_async()
            end_bounds, end_centers = self._compute_tip_top_geometry(
                relative_paths, z_offset=z_offset
            )

            self._sync_articulation_to_pose(current_pose)
            self._apply_pose(current_pose, set_initial_state=False)
            await omni.kit.app.get_app().next_update_async()
            restored = True
            return start_bounds, end_bounds, start_centers, end_centers
        finally:
            if not restored:
                try:
                    self._sync_articulation_to_pose(current_pose)
                    self._apply_pose(current_pose, set_initial_state=False)
                except Exception:
                    carb.log_warn(f"[{WINDOW_TITLE}] Could not restore pose after point sampling.")
            self._timeline.pause()
            visibility_attr.Set(original_visibility)

    def _compute_tip_top_bounds(
        self, relative_paths: List[str], z_offset: Optional[float] = None
    ) -> List[Tuple[float, float, float, float, float]]:
        bounds, _centers = self._compute_tip_top_geometry(relative_paths, z_offset)
        return bounds

    def _compute_tip_top_geometry(
        self, relative_paths: List[str], z_offset: Optional[float] = None
    ) -> Tuple[
        List[Tuple[float, float, float, float, float]],
        List[Tuple[float, float, float]],
    ]:
        if z_offset is None:
            z_offset = self._bbox_z_offset_model.get_value_as_float()
        bounds = []
        centers = []
        for relative_path in relative_paths:
            prim = self._stage().GetPrimAtPath(HAND_ASSET_PATH + relative_path)
            if not prim.IsValid():
                raise RuntimeError(f"Registered fingertip no longer exists: {relative_path}")
            bound, center = self._lowest_mesh_top_geometry(prim)
            min_x, min_y, max_x, max_y, min_z = bound
            bounds.append((min_x, min_y, max_x, max_y, min_z + z_offset))
            centers.append((center[0], center[1], center[2] + z_offset))
        return bounds, centers

    def _lowest_mesh_top_bound(
        self, tip_prim: Usd.Prim
    ) -> Tuple[float, float, float, float, float]:
        bound, _center = self._lowest_mesh_top_geometry(tip_prim)
        return bound

    def _lowest_mesh_top_geometry(
        self, tip_prim: Usd.Prim
    ) -> Tuple[
        Tuple[float, float, float, float, float],
        Tuple[float, float, float],
    ]:
        """Return the XY extent of a tip's lowest world-Z vertex layer.

        A selected link may contain several descendant meshes, so all of their
        points participate. Mesh-local points are transformed to world before
        the world-Z contact layer and top-view XY extent are calculated.
        """
        world_points: List[Tuple[float, float, float]] = []
        for prim in Usd.PrimRange(tip_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get(Usd.TimeCode.Default())
            if not points:
                continue
            world_transform = omni.usd.get_world_transform_matrix(prim)
            for point in points:
                world_point = world_transform.Transform(
                    Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                )
                values = (float(world_point[0]), float(world_point[1]), float(world_point[2]))
                if all(math.isfinite(value) for value in values):
                    world_points.append(values)

        if not world_points:
            raise RuntimeError(f"Selected tip contains no readable mesh vertices: {tip_prim.GetPath()}")

        min_z = min(point[2] for point in world_points)
        z_band = GRASP_LOWEST_Z_BAND
        # Include the fixed 5 mm contact layer and absorb only insignificant
        # floating-point transform noise at its upper boundary.
        epsilon = max(1.0e-9, abs(min_z) * 1.0e-9)
        lowest_points = [
            point for point in world_points if point[2] <= min_z + z_band + epsilon
        ]
        min_x = min(point[0] for point in lowest_points)
        min_y = min(point[1] for point in lowest_points)
        max_x = max(point[0] for point in lowest_points)
        max_y = max(point[1] for point in lowest_points)
        count = float(len(lowest_points))
        center = tuple(
            sum(point[axis] for point in lowest_points) / count for axis in range(3)
        )
        return (min_x, min_y, max_x, max_y, min_z), center  # type: ignore[return-value]

    @staticmethod
    def _combined_grasp_bound(
        start_bound: Tuple[float, float, float, float, float],
        end_bound: Tuple[float, float, float, float, float],
    ) -> Tuple[float, float, float, float, float]:
        return (
            min(start_bound[0], end_bound[0]),
            min(start_bound[1], end_bound[1]),
            max(start_bound[2], end_bound[2]),
            max(start_bound[3], end_bound[3]),
            min(start_bound[4], end_bound[4]),
        )

    @staticmethod
    def _bbox_corners(
        bound: Tuple[float, float, float, float, float]
    ) -> List[Tuple[float, float, float]]:
        min_x, min_y, max_x, max_y, z = bound
        return [
            (min_x, min_y, z),
            (max_x, min_y, z),
            (max_x, max_y, z),
            (min_x, max_y, z),
        ]

    @staticmethod
    def _world_point_to_pose_local(
        point: Tuple[float, float, float], pose: PoseSnapshot
    ) -> Tuple[float, float, float]:
        w, x, y, z = _normalize_quat(pose.orientation_wxyz)
        rotation = Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        delta = Gf.Vec3d(
            point[0] - pose.position[0],
            point[1] - pose.position[1],
            point[2] - pose.position[2],
        )
        local = rotation.GetInverse().TransformDir(delta)
        return float(local[0]), float(local[1]), float(local[2])

    def _create_grasp_bboxes(
        self,
        relative_paths: List[str],
        start_bounds: List[Tuple[float, float, float, float, float]],
        end_bounds: List[Tuple[float, float, float, float, float]],
        end_centers: List[Tuple[float, float, float]],
    ) -> None:
        stage = self._stage()
        UsdGeom.Xform.Define(stage, DEBUG_BBOX_ROOT_PATH)
        UsdGeom.Xform.Define(stage, GRASP_BBOX_CURVE_PATH)
        color = Gf.Vec3f(0.2, 0.8, 1.0)
        center_color = Gf.Vec3f(1.0, 0.15, 0.8)
        center_radius = max(0.00001, self._center_radius_model.get_value_as_float())

        for index, (relative_path, start_bound, end_bound) in enumerate(
            zip(relative_paths, start_bounds, end_bounds), start=1
        ):
            min_x, min_y, max_x, max_y, z = self._combined_grasp_bound(
                start_bound, end_bound
            )
            corners = self._bbox_corners((min_x, min_y, max_x, max_y, z))
            curve = UsdGeom.BasisCurves.Define(stage, f"{GRASP_BBOX_CURVE_PATH}/Tip_{index}")
            curve.GetPrim().SetDisplayName(self._default_tip_name(relative_path))
            curve.GetPrim().SetCustomDataByKey("mesh_path", relative_path)
            curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
            curve.CreateWrapAttr().Set(UsdGeom.Tokens.nonperiodic)
            curve.CreateCurveVertexCountsAttr().Set([5])
            curve.CreatePointsAttr().Set(
                [Gf.Vec3f(*point) for point in corners + corners[:1]]
            )
            curve.CreateWidthsAttr().Set(
                [max(0.00001, self._bbox_line_width_model.get_value_as_float())]
            )
            curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            curve.CreateDisplayColorAttr().Set([color])

        if not end_centers:
            raise RuntimeError("No END fingertip centers were computed.")
        count = float(len(end_centers))
        grasp_center = tuple(
            sum(point[axis] for point in end_centers) / count for axis in range(3)
        )
        sphere = UsdGeom.Sphere.Define(stage, END_GRASP_CENTER_PATH)
        sphere.CreateRadiusAttr().Set(center_radius)
        sphere.CreateDisplayColorAttr().Set([center_color])
        sphere.GetPrim().SetDisplayName("END grasp center")
        UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*grasp_center))

    def _clear_tip_bboxes(self, report: bool = True) -> None:
        stage = self._stage()
        for debug_path in (DEBUG_BBOX_ROOT_PATH, LEGACY_DEBUG_BBOX_ROOT_PATH):
            if stage.GetPrimAtPath(debug_path).IsValid():
                stage.RemovePrim(debug_path)
        if report:
            self._set_status("Cleared fingertip grasp BBoxes and END grasp center.")

    def _cancel_bbox_task(self, stop_timeline: bool) -> None:
        task = getattr(self, "_bbox_task", None)
        if task and not task.done():
            task.cancel()
        self._bbox_task = None
        if stop_timeline and getattr(self, "_timeline", None):
            self._timeline.stop()
            self._articulation = None

    def _cancel_save_task(self) -> None:
        task = getattr(self, "_save_task", None)
        if task and not task.done():
            task.cancel()
        self._save_task = None

    # -------------------------------------------------------- object/ground
    def _load_random_object(self) -> None:
        folder = Path(self._object_folder_model.get_value_as_string().strip())
        if not folder.is_dir():
            raise NotADirectoryError(f"Object folder not found: {folder}")
        candidates = self._discover_object_usds(folder)
        if not candidates:
            raise FileNotFoundError(f"No USD files under: {folder}")
        chosen = random.choice(candidates)

        self._cancel_preview(stop_timeline=True)
        self._cancel_bbox_task(stop_timeline=True)
        stage = self._stage()
        self._remove_test_object(report=False)
        obj = UsdGeom.Xform.Define(stage, OBJECT_PATH).GetPrim()
        asset = UsdGeom.Xform.Define(stage, OBJECT_ASSET_PATH).GetPrim()
        if not asset.GetReferences().AddReference(str(chosen)):
            raise RuntimeError(f"Could not reference object: {chosen}")
        scale = self._object_scale_model.get_value_as_float()
        position = (
            self._object_x_model.get_value_as_float(),
            self._object_y_model.get_value_as_float(),
            self._object_z_model.get_value_as_float(),
        )
        self._set_local_transform(OBJECT_PATH, position, (1.0, 0.0, 0.0, 0.0), (scale, scale, scale))
        self._make_object_physics_ready(obj)
        self._create_ground_plane(self._plane_z_model.get_value_as_float())
        self._ensure_physics_scene()
        if self._has_app_window():
            omni.usd.get_context().get_selection().set_selected_prim_paths([OBJECT_PATH], True)
        self._set_status(
            f"Loaded random object: {chosen.parent.parent.name} (scale={scale:g}); "
            "convex decomposition + rigid body + ground plane ready."
        )

    def _discover_object_usds(self, folder: Path) -> List[Path]:
        """Return scan-object USDs, honoring the dataset manifest when present."""
        manifest = folder / "objects_conf.json"
        candidates: List[Path] = []
        if manifest.is_file():
            try:
                with manifest.open("r", encoding="utf-8") as stream:
                    records = json.load(stream)
                if not isinstance(records, list):
                    raise ValueError("manifest root is not a list")
                for record in records:
                    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                        continue
                    path = Path(record["path"])
                    if not path.is_absolute():
                        path = folder / path
                    if path.is_file() and path.suffix.lower() in (".usd", ".usda", ".usdc"):
                        candidates.append(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                carb.log_warn(f"[{WINDOW_TITLE}] Could not use {manifest}: {exc}; scanning edited folders")
        if not candidates:
            # Dataset layout: <root>/<object_name>/edited/<object_name>.usd.
            # Restricting the fallback avoids accidentally selecting helper USDs.
            candidates = [
                path for path in folder.glob("*/edited/*")
                if path.is_file() and path.suffix.lower() in (".usd", ".usda", ".usdc")
            ]
        return sorted(set(candidates))

    def _make_object_physics_ready(self, root: Usd.Prim) -> None:
        """Mirror ``Scan_Rep.set_rigidbody_collider`` for a referenced scan."""
        from omni.physx.scripts import utils as physx_utils

        # Apply physics to the wrapper, keeping the referenced source USD
        # read-only while scale and spawn transform share one rigid frame.
        physx_utils.setRigidBody(root, "convexDecomposition", False)
        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        rigid_api.CreateMaxAngularVelocityAttr().Set(720.0)
        rigid_api.CreateMaxLinearVelocityAttr().Set(2.5)
        rigid_api.CreateLinearDampingAttr().Set(0.7)
        rigid_api.CreateEnableCCDAttr().Set(True)

        material = UsdShade.Material.Define(self._stage(), f"{OBJECT_PATH}/PhysicsMaterial")
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        material_api.CreateDynamicFrictionAttr().Set(0.25)
        material_api.CreateStaticFrictionAttr().Set(0.4)
        material_api.CreateRestitutionAttr().Set(0.0)

        meshes = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise RuntimeError("The selected object USD contains no mesh prims.")
        for mesh in meshes:
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
            collision_api.CreateContactOffsetAttr().Set(0.000001)
            collision_api.CreateRestOffsetAttr().Set(0.0)
            decomposition_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh)
            decomposition_api.CreateShrinkWrapAttr().Set(True)
            decomposition_api.CreateMaxConvexHullsAttr().Set(240)
            decomposition_api.CreateHullVertexLimitAttr().Set(64)
            decomposition_api.CreateVoxelResolutionAttr().Set(700000)
            decomposition_api.CreateErrorPercentageAttr().Set(8.0)
            UsdPhysics.MassAPI.Apply(mesh)
            UsdShade.MaterialBindingAPI.Apply(mesh).Bind(
                material,
                UsdShade.Tokens.weakerThanDescendants,
                "physics",
            )

    def _create_ground_plane(self, z: float) -> None:
        stage = self._stage()
        if stage.GetPrimAtPath(PLANE_PATH).IsValid():
            stage.RemovePrim(PLANE_PATH)
        plane = UsdGeom.Cube.Define(stage, PLANE_PATH)
        plane.CreateSizeAttr().Set(1.0)
        self._set_local_transform(
            PLANE_PATH,
            (0.0, 0.0, z - 0.01),
            (1.0, 0.0, 0.0, 0.0),
            (4.0, 4.0, 0.02),
        )
        UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
        display = plane.CreateDisplayColorAttr()
        display.Set([Gf.Vec3f(0.18, 0.18, 0.20)])

    def _remove_test_object(self, report: bool = True) -> None:
        stage = self._stage()
        if stage.GetPrimAtPath(OBJECT_PATH).IsValid():
            stage.RemovePrim(OBJECT_PATH)
        if report:
            self._set_status("Removed the test object. Ground plane was kept.")

    # -------------------------------------------------------------- database
    def _normalized_asset_path(self, path: str) -> str:
        value = os.path.expanduser(str(path).strip())
        if "://" in value:
            return value.rstrip("/")
        return os.path.normcase(os.path.realpath(value))

    def _find_db_entry(self, database: dict, usd_path: str) -> Tuple[Optional[str], Optional[dict]]:
        wanted = self._normalized_asset_path(usd_path)
        for key, entry in database.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("usd_path"), str):
                continue
            if self._normalized_asset_path(entry["usd_path"]) == wanted:
                return str(key), entry
        return None, None

    def _refresh_db_presets(
        self,
        auto_load: bool,
        report: bool = True,
        preferred_name: Optional[str] = None,
    ) -> Optional[str]:
        """Match the current gripper by USD path and refresh its preset list."""
        db_path = Path(self._db_path_model.get_value_as_string().strip())
        self._saved_presets = []
        self._saved_preset_names = []
        self._saved_preset_index = 0
        self._matched_db_key = None
        if not db_path.is_file():
            self._saved_preset_frame.rebuild()
            if report:
                raise FileNotFoundError(f"Database JSON not found: {db_path}")
            return None

        with db_path.open("r", encoding="utf-8") as stream:
            database = json.load(stream)
        if not isinstance(database, dict):
            raise ValueError("Database root must be a JSON object.")
        key, entry = self._find_db_entry(database, self._usd_model.get_value_as_string())
        if entry is None or key is None:
            self._saved_preset_frame.rebuild()
            if report:
                self._set_status("No database gripper entry matches the current USD path.")
            return None

        self._matched_db_key = key
        self._gripper_key_model.set_value(key)
        db_urdf = entry.get("urdf_path")
        if isinstance(db_urdf, str) and db_urdf:
            self._urdf_model.set_value(db_urdf)
        presets = entry.get("preset", [])
        if isinstance(presets, list):
            self._saved_presets = [preset for preset in presets if isinstance(preset, dict)]
        self._saved_preset_names = [
            str(preset.get("name") or f"preset_{index + 1}")
            for index, preset in enumerate(self._saved_presets)
        ]
        if self._saved_presets:
            if preferred_name in self._saved_preset_names:
                self._saved_preset_index = self._saved_preset_names.index(preferred_name)
            else:
                # Prefer the most recently saved preset. Legacy entries without
                # updated_at remain selectable but do not override newer data.
                self._saved_preset_index = max(
                    range(len(self._saved_presets)),
                    key=lambda index: str(self._saved_presets[index].get("updated_at") or ""),
                )
        self._saved_preset_frame.rebuild()

        loaded_name = None
        if auto_load and self._saved_presets:
            loaded_name = self._load_saved_preset(self._saved_preset_index, report=False)
        elif report:
            self._set_status(
                f"Matched DB gripper {key!r}; found {len(self._saved_presets)} saved presets."
            )
        return loaded_name

    def _load_tip_links_from_preset(self, preset: dict) -> None:
        self._tip_links.clear()
        stored = preset.get("fingertip_points", {})
        if isinstance(stored, dict):
            for value in stored.values():
                if not isinstance(value, dict):
                    continue
                raw_path = value.get("mesh_path")
                if not isinstance(raw_path, str) or not raw_path:
                    continue
                relative_path = raw_path
                if raw_path.startswith(HAND_ASSET_PATH + "/"):
                    relative_path = raw_path[len(HAND_ASSET_PATH):]
                prim = self._stage().GetPrimAtPath(HAND_ASSET_PATH + relative_path)
                if not prim.IsValid():
                    carb.log_warn(f"[{WINDOW_TITLE}] Stored fingertip does not exist: {raw_path}")
                    continue
                try:
                    self._register_tip_prim(prim)
                except (RuntimeError, ValueError) as exc:
                    carb.log_warn(f"[{WINDOW_TITLE}] Ignored preset fingertip {raw_path}: {exc}")
        self._clear_tip_bboxes(report=False)
        self._refresh_tip_links_label()

    def _load_selected_saved_preset(self) -> None:
        if not self._saved_presets:
            # This also handles a DB path that was edited after gripper load.
            self._refresh_db_presets(auto_load=False, report=False)
        if not self._saved_presets:
            raise RuntimeError("No saved preset matches the current gripper USD path.")
        index = self._saved_preset_index
        if self._saved_preset_combo_model is not None:
            index = self._saved_preset_combo_model.get_item_value_model().get_value_as_int()
        self._load_saved_preset(index, report=True)

    def _load_saved_preset(self, index: int, report: bool) -> str:
        if index < 0 or index >= len(self._saved_presets):
            raise IndexError(f"Saved preset index out of range: {index}")
        self._cancel_preview(stop_timeline=True)
        self._cancel_bbox_task(stop_timeline=True)
        preset = self._saved_presets[index]
        self._load_tip_links_from_preset(preset)
        current_position, current_quat = self._world_pose(self._hand_prim())
        current_joints = {
            name: model.get_value_as_float() for name, model in self._joint_models.items()
        }
        start = self._pose_from_saved_preset(
            preset, "start", current_position, current_quat, current_joints
        )
        end = self._pose_from_saved_preset(
            preset, "end", current_position, current_quat, current_joints
        )
        self._start_pose = start
        self._end_pose = end
        self._start_label.text = self._pose_label("START", start)
        self._end_label.text = self._pose_label("END", end)

        name = str(preset.get("name") or self._saved_preset_names[index])
        self._preset_name_model.set_value(name)
        transition = preset.get("transition")
        if isinstance(transition, dict) and isinstance(transition.get("duration_sec"), (int, float)):
            self._duration_model.set_value(max(0.1, float(transition["duration_sec"])))
        self._load_saved_joint_gains()
        self._apply_pose(start, set_initial_state=True)
        self._saved_preset_index = index
        if report:
            self._set_status(
                f"Loaded saved preset {name!r}: START/END base TF, joint values, duration and gains restored."
            )
        return name

    def _pose_from_saved_preset(
        self,
        preset: dict,
        prefix: str,
        default_position: Tuple[float, float, float],
        default_quat: Tuple[float, float, float, float],
        default_joints: Dict[str, float],
    ) -> PoseSnapshot:
        raw_joints = preset.get(f"{prefix}_joint_pos")
        if not isinstance(raw_joints, dict):
            raise ValueError(f"Preset has no valid {prefix}_joint_pos object.")
        joints = dict(default_joints)
        matched = 0
        unit = str(preset.get("joint_unit") or "rad").lower()
        for name, raw_value in raw_joints.items():
            if name not in self._joint_infos or not isinstance(raw_value, (int, float)):
                continue
            value = float(raw_value)
            info = self._joint_infos[name]
            if unit in ("deg", "degree", "degrees") and info.kind == "revolute":
                value = math.radians(value)
            joints[name] = max(info.lower_rad, min(info.upper_rad, value))
            matched += 1
        if matched == 0:
            raise ValueError(f"Preset {prefix}_joint_pos has no joints used by the loaded gripper.")

        position = default_position
        quat = default_quat
        raw_tf = preset.get(f"{prefix}_base_tf")
        if isinstance(raw_tf, dict):
            raw_position = raw_tf.get("position")
            raw_quat = raw_tf.get("orientation_wxyz")
            raw_rpy = raw_tf.get("rpy_rad")
            if isinstance(raw_position, list) and len(raw_position) == 3:
                position = tuple(float(value) for value in raw_position)
            if isinstance(raw_quat, list) and len(raw_quat) == 4:
                quat = _normalize_quat(raw_quat)
            elif isinstance(raw_rpy, list) and len(raw_rpy) == 3:
                quat = _quat_from_rpy(*[float(value) for value in raw_rpy])
        return PoseSnapshot(position, quat, _rpy_from_quat(quat), joints)

    def _load_saved_joint_gains(self) -> None:
        db_path = Path(self._db_path_model.get_value_as_string().strip())
        with db_path.open("r", encoding="utf-8") as stream:
            database = json.load(stream)
        entry = database.get(self._matched_db_key, {}) if self._matched_db_key else {}
        joint_cfg = entry.get("joint_cfg", {}) if isinstance(entry, dict) else {}
        if not isinstance(joint_cfg, dict):
            return
        for name in self._joint_infos:
            cfg = joint_cfg.get(name)
            if not isinstance(cfg, dict):
                continue
            if isinstance(cfg.get("stiffness"), (int, float)):
                self._stiffness_model.set_value(float(cfg["stiffness"]))
            if isinstance(cfg.get("damping"), (int, float)):
                self._damping_model.set_value(float(cfg["damping"]))
            if isinstance(cfg.get("effort_limit"), (int, float)):
                self._max_force_model.set_value(float(cfg["effort_limit"]))
            break

    def _start_preset_save(self) -> None:
        try:
            if self._start_pose is None or self._end_pose is None:
                raise RuntimeError("Capture both START and END before saving.")
            if not self._tip_links:
                raise RuntimeError("Register at least one fingertip mesh/link before saving.")
            self._cancel_preview(stop_timeline=True)
            self._cancel_bbox_task(stop_timeline=True)
            self._cancel_save_task()
            self._save_task = asyncio.ensure_future(self._save_preset_async())
        except Exception as exc:
            self._set_status(str(exc), error=True)

    def _tip_storage_names(self, relative_paths: Optional[List[str]] = None) -> Dict[str, str]:
        names: Dict[str, str] = {}
        used = set()
        for relative_path in relative_paths or list(self._tip_links):
            base_name = self._default_tip_name(relative_path) or "fingertip"
            storage_name = base_name
            suffix = 2
            while storage_name in used:
                storage_name = f"{base_name}_{suffix}"
                suffix += 1
            used.add(storage_name)
            names[relative_path] = storage_name
        return names

    def _build_preset_geometry(
        self,
        paths: List[str],
        start_bounds: List[Tuple[float, float, float, float, float]],
        end_bounds: List[Tuple[float, float, float, float, float]],
        end_centers: List[Tuple[float, float, float]],
    ) -> Tuple[dict, dict]:
        assert self._start_pose is not None
        tip_names = self._tip_storage_names(paths)
        fingertip_points = {}
        for relative_path, start_bound, end_bound in zip(paths, start_bounds, end_bounds):
            min_x, min_y, max_x, max_y, z = self._combined_grasp_bound(
                start_bound, end_bound
            )
            world_corners = self._bbox_corners((min_x, min_y, max_x, max_y, z))
            local_corners = [
                self._world_point_to_pose_local(point, self._start_pose)
                for point in world_corners
            ]
            fingertip_points[tip_names[relative_path]] = {
                "mesh_path": relative_path,
                "frame": "gripper_base",
                "base_prim_path": HAND_PATH,
                "base_pose": "start",
                "grasp_bbox": {
                    "points": [
                        [round(value, 9) for value in point] for point in local_corners
                    ],
                    "projection_axis": "world_z",
                    "world_min_z": round(z, 9),
                    "lowest_z_band": GRASP_LOWEST_Z_BAND,
                },
            }

        center_count = float(len(end_centers))
        world_grasp_center = tuple(
            sum(point[axis] for point in end_centers) / center_count
            for axis in range(3)
        )
        local_grasp_center = self._world_point_to_pose_local(
            world_grasp_center, self._start_pose
        )
        grasp_center = {
            "frame": "gripper_base",
            "base_prim_path": HAND_PATH,
            "base_pose": "start",
            "point": [round(value, 9) for value in local_grasp_center],
            "source_mesh_paths": paths,
            "lowest_z_band": GRASP_LOWEST_Z_BAND,
        }
        return fingertip_points, grasp_center

    async def _save_preset_async(self) -> None:
        try:
            if self._start_pose is None or self._end_pose is None:
                raise RuntimeError("Capture both START and END before saving.")
            if not self._tip_links:
                raise RuntimeError("Register at least one fingertip mesh/link before saving.")
            changed_joints, paths = self._moving_tip_paths(
                list(self._tip_links), self._start_pose, self._end_pose
            )
            tolerance = max(0.0, self._joint_delta_model.get_value_as_float())
            if not changed_joints:
                raise RuntimeError(f"No joint changes exceed tolerance {tolerance:g} rad.")
            if not paths:
                raise RuntimeError(
                    "No registered fingertip belongs to a joint that changes from START to END."
                )
            # Stored grasp boxes use the true minimum Z, never the optional
            # display-only debug line offset.
            start_bounds, end_bounds, _start_centers, end_centers = (
                await self._sample_tip_bounds_async(paths, z_offset=0.0)
            )
            fingertip_points, grasp_center = self._build_preset_geometry(
                paths, start_bounds, end_bounds, end_centers
            )
            saved_name = self._preset_name_model.get_value_as_string().strip()
            self._save_preset(fingertip_points, grasp_center)
            updated, skipped = await self._recompute_all_preset_geometry_async(saved_name)
            suffix = f"; skipped={', '.join(skipped)}" if skipped else ""
            self._set_status(
                f"Saved {saved_name!r}; recomputed geometry for {updated} preset(s){suffix}."
            )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._set_status(f"Preset save failed: {exc}", error=True)
            carb.log_error(traceback.format_exc())
        finally:
            self._save_task = None

    async def _recompute_all_preset_geometry_async(
        self, preferred_name: str
    ) -> Tuple[int, List[str]]:
        """Re-evaluate every stored preset without requiring manual loading."""
        db_path = Path(self._db_path_model.get_value_as_string().strip())
        with db_path.open("r", encoding="utf-8") as stream:
            database = json.load(stream)
        if not isinstance(database, dict):
            raise ValueError("Database root must be a JSON object.")
        key, entry = self._find_db_entry(database, self._usd_model.get_value_as_string())
        if key is None or entry is None:
            raise RuntimeError("Current gripper USD has no matching database entry.")
        presets = entry.get("preset", [])
        if not isinstance(presets, list):
            raise ValueError(f"{key}.preset must be a list.")

        original_start = self._start_pose
        original_end = self._end_pose
        current_position, current_quat = self._world_pose(self._hand_prim())
        current_joints = {
            name: model.get_value_as_float() for name, model in self._joint_models.items()
        }
        updated = 0
        skipped: List[str] = []
        try:
            for preset in presets:
                if not isinstance(preset, dict):
                    continue
                preset_name = str(preset.get("name") or f"preset_{updated + 1}")
                stored_tips = preset.get("fingertip_points", {})
                paths: List[str] = []
                if isinstance(stored_tips, dict):
                    for value in stored_tips.values():
                        raw_path = value.get("mesh_path") if isinstance(value, dict) else None
                        if not isinstance(raw_path, str) or not raw_path:
                            continue
                        relative_path = raw_path
                        if raw_path.startswith(HAND_ASSET_PATH + "/"):
                            relative_path = raw_path[len(HAND_ASSET_PATH):]
                        if relative_path not in paths and self._stage().GetPrimAtPath(
                            HAND_ASSET_PATH + relative_path
                        ).IsValid():
                            paths.append(relative_path)
                if not paths:
                    skipped.append(preset_name)
                    continue
                try:
                    self._start_pose = self._pose_from_saved_preset(
                        preset, "start", current_position, current_quat, current_joints
                    )
                    self._end_pose = self._pose_from_saved_preset(
                        preset, "end", current_position, current_quat, current_joints
                    )
                    _changed_joints, paths = self._moving_tip_paths(
                        paths, self._start_pose, self._end_pose
                    )
                    if not paths:
                        # Remove stale geometry instead of keeping BBoxes for
                        # fingertips whose joints do not move in this preset.
                        preset["fingertip_points"] = {}
                        preset["grasp_center"] = {}
                        updated += 1
                        continue
                    self._set_status(f"Recomputing preset geometry: {preset_name}")
                    start_bounds, end_bounds, _start_centers, end_centers = (
                        await self._sample_tip_bounds_async(paths, z_offset=0.0)
                    )
                    fingertip_points, grasp_center = self._build_preset_geometry(
                        paths, start_bounds, end_bounds, end_centers
                    )
                    preset["fingertip_points"] = fingertip_points
                    preset["grasp_center"] = grasp_center
                    updated += 1
                except Exception as exc:
                    skipped.append(preset_name)
                    carb.log_warn(
                        f"[{WINDOW_TITLE}] Could not recompute preset {preset_name!r}: {exc}"
                    )
        finally:
            self._start_pose = original_start
            self._end_pose = original_end
            if original_start is not None:
                self._start_label.text = self._pose_label("START", original_start)
            if original_end is not None:
                self._end_label.text = self._pose_label("END", original_end)

        if updated:
            self._atomic_json_save(db_path, database)
            self._refresh_db_presets(
                auto_load=False, report=False, preferred_name=preferred_name
            )
        return updated, skipped

    def _save_preset(
        self,
        fingertip_points: Optional[dict] = None,
        grasp_center: Optional[dict] = None,
    ) -> None:
        if self._start_pose is None or self._end_pose is None:
            raise RuntimeError("Capture both START and END before saving.")
        name = self._preset_name_model.get_value_as_string().strip()
        if not name:
            raise ValueError("Preset name cannot be empty.")
        db_path = Path(self._db_path_model.get_value_as_string().strip())
        if not db_path.is_file():
            raise FileNotFoundError(f"Database JSON not found: {db_path}")

        with db_path.open("r", encoding="utf-8") as stream:
            database = json.load(stream)
        if not isinstance(database, dict):
            raise ValueError("Database root must be a JSON object.")

        key = self._gripper_key_model.get_value_as_string().strip() or DEFAULT_GRIPPER_KEY
        usd_path = self._usd_model.get_value_as_string().strip()
        urdf_path = self._urdf_model.get_value_as_string().strip()
        entry = database.setdefault(
            key,
            {"gripper_name": key, "usd_path": usd_path, "type": "Hand", "preset": []},
        )
        if not isinstance(entry, dict):
            raise ValueError(f"Database entry {key!r} must be an object.")
        entry["gripper_name"] = entry.get("gripper_name") or key
        entry["usd_path"] = usd_path
        entry["urdf_path"] = urdf_path
        entry["type"] = "Hand"
        # Fingertips are preset-specific. Remove legacy gripper-wide fields
        # when this entry is next saved.
        entry.pop("tip_links", None)
        entry.pop("tip_link_names", None)

        preset = {
            "name": name,
            "joint_unit": "rad",
            "base_tf_frame": "world",
            "start_base_tf": self._start_pose.base_tf_json(),
            "end_base_tf": self._end_pose.base_tf_json(),
            "start_joint_pos": {k: round(v, 9) for k, v in self._start_pose.joints_rad.items()},
            "end_joint_pos": {k: round(v, 9) for k, v in self._end_pose.joints_rad.items()},
            "fingertip_points": fingertip_points or {},
            "grasp_center": grasp_center or {},
            "transition": {
                "duration_sec": round(max(0.1, self._duration_model.get_value_as_float()), 4),
                "interpolation": "smoothstep",
            },
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        presets = entry.setdefault("preset", [])
        if not isinstance(presets, list):
            raise ValueError(f"{key}.preset must be a list.")
        replaced = False
        for index, old in enumerate(presets):
            if isinstance(old, dict) and old.get("name") == name:
                presets[index] = preset
                replaced = True
                break
        if not replaced:
            presets.append(preset)

        joint_cfg = entry.setdefault("joint_cfg", {})
        if isinstance(joint_cfg, dict):
            for joint_name in self._joint_infos:
                joint_cfg.setdefault(
                    joint_name,
                    {
                        "effort_limit": self._max_force_model.get_value_as_float(),
                        "velocity_limit": "None",
                        "stiffness": self._stiffness_model.get_value_as_float(),
                        "damping": self._damping_model.get_value_as_float(),
                    },
                )

        self._atomic_json_save(db_path, database)
        self._refresh_db_presets(auto_load=False, report=False, preferred_name=name)
        action = "Replaced" if replaced else "Added"
        self._set_status(f"{action} preset {name!r} in {db_path}. Backup: {db_path.name}.bak")

    def _atomic_json_save(self, path: Path, data: dict) -> None:
        backup = path.with_name(path.name + ".bak")
        shutil.copy2(path, backup)
        temp_name: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as stream:
                temp_name = stream.name
                json.dump(data, stream, indent=4, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        except Exception:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
            raise
