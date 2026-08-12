"""Isaac Sim 5.1 extension for visualizing gripper contact forces.

The viewer reads each Isaac contact sensor's raw PhysX impulses and converts
them to world-space force using ``force = impulse / physics_dt``.  Existing
sensor prims are reused.  A gripper link that only carries
``PhysxContactReportAPI`` receives a small viewer-owned contact sensor child so
the same raw-data interface can be used without modifying the source USD.
"""

from __future__ import annotations

import math
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import carb
import omni.ext
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.ui as ui
import omni.usd
import omni.isaac.IsaacSensorSchema as IsaacSensorSchema
import numpy as np
from isaacsim.sensors.physics import _sensor
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


WINDOW_TITLE = "Gripper Contact Force Viewer"
TOOL_ROOT_PATH = "/World/GripperContactForceViewer"
GRIPPER_PATH = f"{TOOL_ROOT_PATH}/Gripper"
GRIPPER_ASSET_PATH = f"{GRIPPER_PATH}/Asset"
PHYSICS_SCENE_PATH = "/World/physicsScene"
VIEWER_SENSOR_NAME = "ContactForceViewerSensor"

DEFAULT_USD_PATH = (
    "/nas/ochansol/isaac/USD/robots/gripper/Robotiq_2f140/"
    "Robotiq_2f140.usd"
)


@dataclass(frozen=True)
class JointInfo:
    name: str
    prim_path: str
    lower: float
    upper: float
    drive_name: str
    revolute: bool


@dataclass
class SensorState:
    sensor_path: str
    body_path: str
    raw_force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    display_force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    zero_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    peak: float = 0.0
    contacts: int = 0
    valid: bool = False


@dataclass
class SensorWidgets:
    value_labels: Dict[str, ui.Label]
    bar_models: Dict[str, ui.SimpleFloatModel]
    contact_label: ui.Label


def _component(value, axis: str, index: int) -> float:
    """Read x/y/z from dict-like, structured numpy, Gf, or sequence values."""
    try:
        return float(value[axis])
    except Exception:
        pass
    try:
        return float(getattr(value, axis))
    except Exception:
        pass
    return float(value[index])


def _vector(raw_item, field_name: str) -> np.ndarray:
    value = raw_item[field_name]
    return np.asarray(
        [_component(value, "x", 0), _component(value, "y", 1), _component(value, "z", 2)],
        dtype=float,
    )


def _scalar(raw_item, field_name: str, default: float) -> float:
    try:
        return float(raw_item[field_name])
    except Exception:
        return float(default)


def _path_matches(actor_path: str, body_path: str) -> bool:
    actor = actor_path.rstrip("/")
    body = body_path.rstrip("/")
    return actor == body or actor.startswith(body + "/") or body.startswith(actor + "/")


class GripperContactForceViewerExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._window: Optional[ui.Window] = None
        self._menu_items = [MenuItemDescription(name=WINDOW_TITLE, onclick_fn=self._toggle_window)]
        add_menu_items(self._menu_items, "Tools")

        self._timeline = omni.timeline.get_timeline_interface()
        self._contact_interface = _sensor.acquire_contact_sensor_interface()
        self._update_subscription = None
        self._joint_infos: Dict[str, JointInfo] = {}
        self._skipped_joint_limits: list[str] = []
        self._disabled_dangling_joints: list[str] = []
        self._sensor_states: Dict[str, SensorState] = {}
        self._sensor_widgets: Dict[str, SensorWidgets] = {}
        self._last_update_time = 0.0
        self._monitor_enabled = True
        self._unresolved_reports: list[str] = []

        self._window = ui.Window(WINDOW_TITLE, width=620, height=820, visible=True)
        self._window.set_visibility_changed_fn(self._on_visibility_changed)
        self._build_ui()
        self._update_subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_app_update, name="gripper_contact_force_viewer")
        )
        self._set_status("Ready. Load a gripper USD.")

    def on_shutdown(self) -> None:
        self._update_subscription = None
        if getattr(self, "_menu_items", None):
            remove_menu_items(self._menu_items, "Tools")
        self._sensor_widgets.clear()
        self._sensor_states.clear()
        self._window = None
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
                        "Gripper의 contact report/sensor를 찾아 world XYZ force와 resultant를 실시간 표시합니다.",
                        word_wrap=True,
                        height=38,
                    )

                    with ui.CollapsableFrame("1. Gripper", collapsed=False):
                        with ui.VStack(spacing=5, height=0):
                            self._usd_model = self._string_row("USD", DEFAULT_USD_PATH)
                            with ui.HStack(height=30, spacing=5):
                                ui.Button("Load / Reload", clicked_fn=lambda: self._guard(self._load_gripper))
                                ui.Button("Select base", clicked_fn=lambda: self._guard(self._select_gripper))
                                ui.Button("Refresh sensors", clicked_fn=lambda: self._guard(self._refresh_sensors))
                            self._auto_sensor_model = ui.SimpleBoolModel(True)
                            with ui.HStack(height=24, spacing=5):
                                ui.CheckBox(model=self._auto_sensor_model, width=20)
                                ui.Label(
                                    "ContactReportAPI 링크에 viewer sensor 자동 연결",
                                    tooltip="Source USD는 수정하지 않고 현재 stage reference instance에만 추가합니다.",
                                )
                            ui.Button(
                                "Add sensor to selected collider",
                                height=28,
                                clicked_fn=lambda: self._guard(self._add_sensor_to_selected),
                            )

                    with ui.CollapsableFrame("2. Simulation / Joint control", collapsed=False):
                        with ui.VStack(spacing=5, height=0):
                            with ui.HStack(height=30, spacing=5):
                                ui.Button("Play", clicked_fn=lambda: self._guard(self._play))
                                ui.Button("Pause", clicked_fn=self._timeline.pause)
                                ui.Button("Stop", clicked_fn=self._timeline.stop)
                                ui.Button("OPEN", clicked_fn=lambda: self._guard(lambda: self._set_gripper(False)))
                                ui.Button("CLOSE", clicked_fn=lambda: self._guard(lambda: self._set_gripper(True)))
                            self._reverse_model = ui.SimpleBoolModel(False)
                            with ui.HStack(height=24, spacing=5):
                                ui.CheckBox(model=self._reverse_model, width=20)
                                ui.Label("Open/Close limits reverse (기본: lower=open, upper=close)")
                            self._stiffness_model = self._float_row("Stiffness", 500.0, 0.0, 1_000_000.0)
                            self._damping_model = self._float_row("Damping", 10.0, 0.0, 1_000_000.0)
                            self._max_force_model = self._float_row("Drive max force", 100.0, 0.0, 1_000_000.0)
                            self._joint_summary_label = ui.Label("Driven joints: none", word_wrap=True, height=34)

                    with ui.CollapsableFrame("3. Force display settings", collapsed=False):
                        with ui.VStack(spacing=5, height=0):
                            self._refresh_hz_model = self._float_row("UI refresh (Hz)", 20.0, 1.0, 120.0)
                            self._force_scale_model = self._float_row("Bar max (N)", 50.0, 0.001, 1_000_000.0)
                            self._smoothing_model = self._float_row("Smoothing alpha", 0.35, 0.0, 1.0)
                            with ui.HStack(height=30, spacing=5):
                                self._monitor_button = ui.Button("Pause monitor", clicked_fn=self._toggle_monitor)
                                ui.Button("Zero current", clicked_fn=lambda: self._guard(self._zero_current))
                                ui.Button("Reset zero", clicked_fn=lambda: self._guard(self._reset_zero))
                                ui.Button("Clear peaks", clicked_fn=lambda: self._guard(self._clear_peaks))
                            ui.Label(
                                "단위: N. XYZ는 PhysX world 좌표계이며 |F|는 세 축 합력 벡터의 크기입니다.",
                                word_wrap=True,
                                height=34,
                            )

                    with ui.CollapsableFrame("4. Contact forces", collapsed=False):
                        with ui.VStack(spacing=6, height=0):
                            self._global_force_label = ui.Label(
                                "All sensors: Fx 0.000  Fy 0.000  Fz 0.000  |ΣF| 0.000 N  Σ|F| 0.000 N",
                                word_wrap=True,
                                height=36,
                            )
                            self._sensor_count_label = ui.Label("Sensors: 0", height=24)
                            self._sensor_frame = ui.Frame(height=0)
                            self._sensor_frame.set_build_fn(self._build_sensor_cards)

                    self._status_label = ui.Label("Ready", word_wrap=True, height=62)

    def _string_row(self, label: str, value: str) -> ui.AbstractValueModel:
        with ui.HStack(height=25, spacing=5):
            ui.Label(label, width=115)
            field = ui.StringField()
            field.model.set_value(value)
        return field.model

    def _float_row(self, label: str, value: float, minimum: float, maximum: float) -> ui.AbstractValueModel:
        model = ui.SimpleFloatModel(value)
        with ui.HStack(height=25, spacing=5):
            ui.Label(label, width=115)
            ui.FloatDrag(model=model, min=minimum, max=maximum, step=0.01)
        return model

    def _build_sensor_cards(self) -> None:
        self._sensor_widgets = {}
        with ui.VStack(spacing=7, height=0):
            if not self._sensor_states:
                ui.Label("No contact sensors found.", height=28)
                return
            for sensor_path, state in sorted(self._sensor_states.items()):
                relative_name = sensor_path.replace(GRIPPER_ASSET_PATH, "") or sensor_path
                with ui.CollapsableFrame(relative_name, collapsed=False):
                    with ui.VStack(spacing=3, height=0):
                        ui.Label(f"Body: {state.body_path}", word_wrap=True, height=28)
                        labels: Dict[str, ui.Label] = {}
                        bars: Dict[str, ui.SimpleFloatModel] = {}
                        for axis in ("x", "y", "z", "total"):
                            bars[axis] = ui.SimpleFloatModel(0.0)
                            with ui.HStack(height=22, spacing=5):
                                title = "|F|" if axis == "total" else f"F{axis}"
                                ui.Label(title, width=38)
                                labels[axis] = ui.Label("0.000 N", width=105)
                                ui.ProgressBar(model=bars[axis], height=8)
                        contact_label = ui.Label("contacts=0  peak=0.000 N  waiting for physics data", height=24)
                        self._sensor_widgets[sensor_path] = SensorWidgets(labels, bars, contact_label)

    # -------------------------------------------------------------- helpers
    def _set_status(self, message: str, error: bool = False) -> None:
        if getattr(self, "_status_label", None):
            self._status_label.text = ("ERROR: " if error else "") + message
        (carb.log_error if error else carb.log_info)(f"[{WINDOW_TITLE}] {message}")

    def _guard(self, function) -> None:
        try:
            function()
        except Exception as error:
            self._set_status(str(error), error=True)
            carb.log_error(traceback.format_exc())

    def _stage(self) -> Usd.Stage:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Open or create a USD stage first.")
        return stage

    def _gripper_prim(self) -> Usd.Prim:
        prim = self._stage().GetPrimAtPath(GRIPPER_PATH)
        if not prim.IsValid():
            raise RuntimeError("Load the gripper first.")
        return prim

    def _physics_dt(self) -> float:
        stage = self._stage()
        for prim in stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                frequency = PhysxSchema.PhysxSceneAPI(prim).GetTimeStepsPerSecondAttr().Get()
                if frequency and float(frequency) > 0.0:
                    return 1.0 / float(frequency)
        return 1.0 / 60.0

    def _ensure_physics_scene(self) -> None:
        stage = self._stage()
        if not stage.GetPrimAtPath(PHYSICS_SCENE_PATH).IsValid():
            scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH)
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)

    # ---------------------------------------------------------- load/sensors
    def _load_gripper(self) -> None:
        usd_path = self._usd_model.get_value_as_string().strip()
        if not usd_path or not os.path.isfile(usd_path):
            raise FileNotFoundError(f"USD file not found: {usd_path}")

        self._timeline.stop()
        stage = self._stage()
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, TOOL_ROOT_PATH)
        if stage.GetPrimAtPath(GRIPPER_PATH).IsValid():
            stage.RemovePrim(GRIPPER_PATH)
        UsdGeom.Xform.Define(stage, GRIPPER_PATH)
        asset = UsdGeom.Xform.Define(stage, GRIPPER_ASSET_PATH).GetPrim()
        if not asset.GetReferences().AddReference(usd_path):
            raise RuntimeError(f"Failed to reference USD: {usd_path}")

        self._ensure_physics_scene()
        self._disabled_dangling_joints = self._disable_dangling_joints(asset)
        self._joint_infos = self._discover_driven_joints(asset)
        self._joint_summary_label.text = self._joint_summary()
        self._refresh_sensors()
        self._select_gripper()
        self._set_status(
            f"Loaded {Path(usd_path).name}; driven joints={len(self._joint_infos)}; "
            f"contact sensors={len(self._sensor_states)}. Press Play to receive force data."
        )

    def _select_gripper(self) -> None:
        self._gripper_prim()
        omni.usd.get_context().get_selection().set_selected_prim_paths([GRIPPER_PATH], True)

    def _is_contact_sensor(self, prim: Usd.Prim) -> bool:
        try:
            return prim.IsA(IsaacSensorSchema.IsaacContactSensor)
        except Exception:
            return prim.GetTypeName() == "IsaacContactSensor"

    def _sensor_body_path(self, sensor_prim: Usd.Prim) -> str:
        return str(sensor_prim.GetParent().GetPath())

    def _collision_target(self, report_prim: Usd.Prim) -> Optional[Usd.Prim]:
        if report_prim.HasAPI(UsdPhysics.CollisionAPI):
            return report_prim
        for prim in Usd.PrimRange(report_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                return prim
        return None

    def _create_sensor_on_collision(self, collision_prim: Usd.Prim) -> str:
        parent_path = str(collision_prim.GetPath())
        sensor_path = f"{parent_path}/{VIEWER_SENSOR_NAME}"
        if self._stage().GetPrimAtPath(sensor_path).IsValid():
            return sensor_path
        PhysxSchema.PhysxContactReportAPI.Apply(collision_prim).CreateThresholdAttr().Set(0.0)
        success, _ = omni.kit.commands.execute(
            "IsaacSensorCreateContactSensor",
            path=f"/{VIEWER_SENSOR_NAME}",
            parent=parent_path,
            sensor_period=self._physics_dt(),
            min_threshold=0.0,
            max_threshold=1.0e9,
            radius=-1.0,
            translation=Gf.Vec3d(0.0, 0.0, 0.0),
        )
        if not success:
            raise RuntimeError(f"Failed to create contact sensor under {parent_path}")
        return sensor_path

    def _refresh_sensors(self) -> None:
        root = self._stage().GetPrimAtPath(GRIPPER_ASSET_PATH)
        if not root.IsValid():
            raise RuntimeError("Load the gripper first.")

        sensors: Dict[str, str] = {}
        reports: list[Usd.Prim] = []
        for prim in Usd.PrimRange(root):
            if self._is_contact_sensor(prim):
                sensors[str(prim.GetPath())] = self._sensor_body_path(prim)
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                reports.append(prim)

        self._unresolved_reports = []
        if self._auto_sensor_model.get_value_as_bool():
            existing_bodies = set(sensors.values())
            for report_prim in reports:
                target = self._collision_target(report_prim)
                if target is None:
                    self._unresolved_reports.append(str(report_prim.GetPath()))
                    continue
                target_path = str(target.GetPath())
                if target_path in existing_bodies:
                    continue
                try:
                    sensor_path = self._create_sensor_on_collision(target)
                    sensors[sensor_path] = target_path
                    existing_bodies.add(target_path)
                except Exception as error:
                    self._unresolved_reports.append(f"{report_prim.GetPath()}: {error}")

        previous = self._sensor_states
        self._sensor_states = {}
        for sensor_path, body_path in sensors.items():
            state = previous.get(sensor_path, SensorState(sensor_path, body_path))
            state.body_path = body_path
            self._sensor_states[sensor_path] = state

        self._sensor_count_label.text = (
            f"Sensors: {len(self._sensor_states)}  | ContactReport prims: {len(reports)}"
            f"  | Unresolved: {len(self._unresolved_reports)}"
        )
        self._sensor_frame.rebuild()

    def _add_sensor_to_selected(self) -> None:
        paths = omni.usd.get_context().get_selection().get_selected_prim_paths()
        if not paths:
            raise RuntimeError("Select a collision prim in the viewport first.")
        prim = self._stage().GetPrimAtPath(paths[-1])
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"Selected prim has no CollisionAPI: {prim.GetPath()}")
        sensor_path = self._create_sensor_on_collision(prim)
        self._refresh_sensors()
        self._set_status(f"Contact report/sensor ready: {sensor_path}")

    # --------------------------------------------------------------- joints
    def _disable_dangling_joints(self, root: Usd.Prim) -> list[str]:
        """Deactivate joints whose referenced bodies do not exist in this asset instance."""
        stage = root.GetStage()
        disabled: list[str] = []
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            targets = list(joint.GetBody0Rel().GetTargets()) + list(joint.GetBody1Rel().GetTargets())
            if any(not stage.GetPrimAtPath(target).IsValid() for target in targets):
                prim.SetActive(False)
                disabled.append(str(prim.GetPath()))
        return disabled

    def _discover_driven_joints(self, root: Usd.Prim) -> Dict[str, JointInfo]:
        result: Dict[str, JointInfo] = {}
        self._skipped_joint_limits = []
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(PhysxSchema.PhysxMimicJointAPI):
                continue
            if prim.IsA(UsdPhysics.RevoluteJoint):
                joint, drive_name, scale, revolute = UsdPhysics.RevoluteJoint(prim), "angular", math.pi / 180.0, True
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                joint, drive_name, scale, revolute = UsdPhysics.PrismaticJoint(prim), "linear", 1.0, False
            else:
                continue
            drive = UsdPhysics.DriveAPI.Get(prim, drive_name)
            if not drive or not drive.GetTargetPositionAttr().IsValid():
                continue
            lower = joint.GetLowerLimitAttr().Get()
            upper = joint.GetUpperLimitAttr().Get()
            lower_value = float(lower) * scale if lower is not None else math.nan
            upper_value = float(upper) * scale if upper is not None else math.nan

            # USD uses float extrema for an unbounded joint.  Sending those
            # values to PhysX as OPEN/CLOSE targets causes invalid articulation
            # poses (revolute targets must also stay inside [-2*pi, 2*pi]).
            # An unbounded joint has no meaningful endpoint, so leave it out of
            # automatic endpoint control instead of guessing a motion range.
            if not math.isfinite(lower_value) or not math.isfinite(upper_value):
                self._skipped_joint_limits.append(prim.GetName())
                continue
            if revolute:
                lower_value = max(-2.0 * math.pi, lower_value)
                upper_value = min(2.0 * math.pi, upper_value)
            elif abs(lower_value) > 10.0 or abs(upper_value) > 10.0:
                self._skipped_joint_limits.append(prim.GetName())
                continue
            if lower_value >= upper_value:
                self._skipped_joint_limits.append(prim.GetName())
                continue
            result[prim.GetName()] = JointInfo(
                prim.GetName(), str(prim.GetPath()), lower_value, upper_value, drive_name, revolute
            )
        return dict(sorted(result.items()))

    def _joint_summary(self) -> str:
        if not self._joint_infos:
            summary = "Driven joints: none"
            if self._skipped_joint_limits:
                summary += f" | skipped unbounded/invalid: {', '.join(self._skipped_joint_limits)}"
            return summary
        values = [f"{name}[{info.lower:.3f}, {info.upper:.3f}]" for name, info in self._joint_infos.items()]
        summary = "Driven joints (rad/m): " + ", ".join(values)
        if self._skipped_joint_limits:
            summary += f" | skipped unbounded/invalid: {', '.join(self._skipped_joint_limits)}"
        return summary

    def _apply_drive_gains(self) -> None:
        stage = self._stage()
        stiffness = self._stiffness_model.get_value_as_float()
        damping = self._damping_model.get_value_as_float()
        max_force = self._max_force_model.get_value_as_float()
        for info in self._joint_infos.values():
            drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(info.prim_path), info.drive_name)
            drive.CreateStiffnessAttr().Set(stiffness)
            drive.CreateDampingAttr().Set(damping)
            drive.CreateMaxForceAttr().Set(max_force)

    def _set_joint_target(self, info: JointInfo, value: float) -> None:
        value = max(info.lower, min(info.upper, float(value)))
        prim = self._stage().GetPrimAtPath(info.prim_path)
        drive = UsdPhysics.DriveAPI.Get(prim, info.drive_name)
        usd_value = math.degrees(value) if info.revolute else value
        drive.GetTargetPositionAttr().Set(usd_value)
        if self._timeline.is_stopped():
            PhysxSchema.JointStateAPI.Apply(prim, info.drive_name).CreatePositionAttr().Set(usd_value)

    def _set_gripper(self, close: bool) -> None:
        if not self._joint_infos:
            raise RuntimeError("No driven joints were found in the loaded gripper.")
        reverse = self._reverse_model.get_value_as_bool()
        use_upper = close != reverse
        for info in self._joint_infos.values():
            self._set_joint_target(info, info.upper if use_upper else info.lower)
        self._set_status("CLOSE target applied." if close else "OPEN target applied.")

    def _play(self) -> None:
        self._gripper_prim()
        self._ensure_physics_scene()
        self._apply_drive_gains()
        self._timeline.play()
        self._set_status("Timeline playing; monitoring contact forces.")

    # --------------------------------------------------------- force update
    def _decode_body(self, encoded) -> str:
        try:
            return str(self._contact_interface.decode_body_name(int(encoded)))
        except Exception:
            return ""

    def _read_sensor_force(self, state: SensorState) -> Tuple[np.ndarray, int, bool]:
        reading = self._contact_interface.get_sensor_reading(state.sensor_path, use_latest_data=True)
        raw_data = self._contact_interface.get_contact_sensor_raw_data(state.sensor_path)
        fallback_dt = self._physics_dt()
        force = np.zeros(3, dtype=float)
        for contact in raw_data:
            dt = max(_scalar(contact, "dt", fallback_dt), 1.0e-9)
            impulse = _vector(contact, "impulse")
            body0 = self._decode_body(contact["body0"])
            body1 = self._decode_body(contact["body1"])
            # PhysX raw impulse points toward body1. Convert it to the force
            # acting on the body that owns this sensor.
            if _path_matches(body0, state.body_path) and not _path_matches(body1, state.body_path):
                impulse = -impulse
            force += impulse / dt
        return force, len(raw_data), bool(getattr(reading, "is_valid", False))

    def _on_app_update(self, _event) -> None:
        if not self._monitor_enabled or not self._timeline.is_playing() or not self._sensor_states:
            return
        now = time.monotonic()
        refresh_hz = max(1.0, self._refresh_hz_model.get_value_as_float())
        if now - self._last_update_time < 1.0 / refresh_hz:
            return
        self._last_update_time = now
        try:
            self._update_forces()
        except Exception as error:
            carb.log_warn(f"[{WINDOW_TITLE}] Force update failed: {error}")

    def _update_forces(self) -> None:
        alpha = min(1.0, max(0.0, self._smoothing_model.get_value_as_float()))
        scale = max(1.0e-6, self._force_scale_model.get_value_as_float())
        total_vector = np.zeros(3, dtype=float)
        magnitude_sum = 0.0

        for sensor_path, state in self._sensor_states.items():
            raw_force, contacts, valid = self._read_sensor_force(state)
            state.raw_force = raw_force
            corrected = raw_force - state.zero_offset
            state.display_force = alpha * corrected + (1.0 - alpha) * state.display_force
            state.contacts = contacts
            state.valid = valid
            magnitude = float(np.linalg.norm(state.display_force))
            state.peak = max(state.peak, magnitude)
            total_vector += state.display_force
            magnitude_sum += magnitude

            widgets = self._sensor_widgets.get(sensor_path)
            if widgets:
                for axis, value in zip(("x", "y", "z"), state.display_force):
                    widgets.value_labels[axis].text = f"{value:+.3f} N"
                    widgets.bar_models[axis].set_value(min(1.0, abs(float(value)) / scale))
                widgets.value_labels["total"].text = f"{magnitude:.3f} N"
                widgets.bar_models["total"].set_value(min(1.0, magnitude / scale))
                validity = "valid" if valid else "waiting for physics data"
                widgets.contact_label.text = (
                    f"contacts={contacts}  peak={state.peak:.3f} N  {validity}"
                )

        resultant = float(np.linalg.norm(total_vector))
        self._global_force_label.text = (
            f"All sensors: Fx {total_vector[0]:+.3f}  Fy {total_vector[1]:+.3f}  "
            f"Fz {total_vector[2]:+.3f}  |ΣF| {resultant:.3f} N  Σ|F| {magnitude_sum:.3f} N"
        )

    def _toggle_monitor(self) -> None:
        self._monitor_enabled = not self._monitor_enabled
        self._monitor_button.text = "Pause monitor" if self._monitor_enabled else "Resume monitor"

    def _zero_current(self) -> None:
        for state in self._sensor_states.values():
            force, contacts, valid = self._read_sensor_force(state)
            state.zero_offset = force
            state.display_force[:] = 0.0
            state.contacts = contacts
            state.valid = valid
        self._set_status("Current raw XYZ forces stored as zero offsets.")

    def _reset_zero(self) -> None:
        for state in self._sensor_states.values():
            state.zero_offset[:] = 0.0
        self._set_status("All zero offsets reset.")

    def _clear_peaks(self) -> None:
        for state in self._sensor_states.values():
            state.peak = 0.0
        self._set_status("Peak forces cleared.")
