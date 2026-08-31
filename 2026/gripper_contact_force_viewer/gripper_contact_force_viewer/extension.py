"""Isaac Sim 5.1 extension for visualizing gripper contact forces.

The viewer reads each Isaac contact sensor's raw PhysX impulses and converts
them to world-space force using ``force = impulse / physics_dt``.  Existing
sensor prims are reused.  A gripper link that only carries
``PhysxContactReportAPI`` is read directly from its rigid-body prim. Raw
impulses are accumulated on every physics step and converted to average force.
"""

from __future__ import annotations

import math
import json
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
import omni.physx
import omni.timeline
import omni.ui as ui
import omni.usd
import numpy as np
from isaacsim.sensors.physics import _sensor
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


WINDOW_TITLE = "Gripper Contact Force Viewer"
TOOL_ROOT_PATH = "/World/GripperContactForceViewer"
GRIPPER_PATH = f"{TOOL_ROOT_PATH}/Gripper"
GRIPPER_ASSET_PATH = f"{GRIPPER_PATH}/Asset"
PHYSICS_SCENE_PATH = "/World/physicsScene"
TEST_OBJECT_PATH = f"{TOOL_ROOT_PATH}/TestObject"

DEFAULT_USD_PATH = (
    "/nas/ochansol/isaac/USD/robots/gripper/Robotiq_2f140/"
    "Robotiq_2f140.usd"
)
DEFAULT_HAND_DB_PATH = "/nas/ochansol/gripper_info/gripper_info_hand_2026.json"


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
    accumulated_impulse: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    accumulated_time: float = 0.0


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
        self._physics_subscription = None
        self._joint_infos: Dict[str, JointInfo] = {}
        self._skipped_joint_limits: list[str] = []
        self._disabled_dangling_joints: list[str] = []
        self._sensor_states: Dict[str, SensorState] = {}
        self._sensor_widgets: Dict[str, SensorWidgets] = {}
        self._last_update_time = 0.0
        self._monitor_enabled = True
        self._unresolved_reports: list[str] = []
        self._gripper_type = ""
        self._matched_hand_db_key: Optional[str] = None
        self._hand_presets: list[dict] = []
        self._hand_preset_names: list[str] = []
        self._hand_preset_index = 0
        self._hand_preset_combo_model = None

        self._window = ui.Window(WINDOW_TITLE, width=620, height=820, visible=True)
        self._window.set_visibility_changed_fn(self._on_visibility_changed)
        self._build_ui()
        self._update_subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_app_update, name="gripper_contact_force_viewer")
        )
        self._physics_subscription = (
            omni.physx.get_physx_interface()
            .subscribe_physics_step_events(self._on_physics_step)
        )
        self._set_status("Ready. Load a gripper USD.")

    def on_shutdown(self) -> None:
        self._update_subscription = None
        self._physics_subscription = None
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
                        "Gripper의 contact report body를 찾아 world XYZ force와 resultant를 실시간 표시합니다.",
                        word_wrap=True,
                        height=38,
                    )

                    with ui.CollapsableFrame("1. Gripper", collapsed=False):
                        with ui.VStack(spacing=5, height=0):
                            self._usd_model = self._string_row("USD", DEFAULT_USD_PATH)
                            self._hand_db_model = self._string_row("Hand preset DB", DEFAULT_HAND_DB_PATH)
                            with ui.HStack(height=30, spacing=5):
                                ui.Button("Load / Reload", clicked_fn=lambda: self._guard(self._load_gripper))
                                ui.Button("Select base", clicked_fn=lambda: self._guard(self._select_gripper))
                                ui.Button("Refresh sensors", clicked_fn=lambda: self._guard(self._refresh_sensors))
                            with ui.HStack(height=24, spacing=5):
                                ui.Label(
                                    "현재 stage의 ContactReportAPI를 직접 사용",
                                    tooltip="IsaacContactSensor prim을 만들지 않고 rigid-body raw contact를 읽습니다.",
                                )
                            ui.Button(
                                "Add report to selected body/collider",
                                height=28,
                                clicked_fn=lambda: self._guard(self._add_report_to_selected),
                            )
                            self._gripper_type_label = ui.Label(
                                "Type: standard gripper (limit control)", word_wrap=True, height=24
                            )
                            self._hand_preset_frame = ui.Frame(height=26)
                            self._hand_preset_frame.set_build_fn(self._build_hand_preset_row)

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
                                ui.Label("Reverse OPEN/CLOSE (standard: lower/upper, Hand: start/end)")
                            self._stiffness_model = self._float_row("Stiffness", 500.0, 0.0, 1_000_000.0)
                            self._damping_model = self._float_row("Damping", 10.0, 0.0, 1_000_000.0)
                            self._max_force_model = self._float_row("Max force/torque", 100.0, 0.0, 1_000_000.0)
                            ui.Button(
                                "Apply drive settings",
                                height=27,
                                clicked_fn=lambda: self._guard(self._apply_drive_gains_with_status),
                            )
                            self._joint_summary_label = ui.Label("Driven joints: none", word_wrap=True, height=34)
                            ui.Label(
                                "Angular drive max = torque (N·m), linear drive max = force (N).",
                                word_wrap=True,
                                height=30,
                            )

                    with ui.CollapsableFrame("Test object", collapsed=False):
                        with ui.VStack(spacing=5, height=0):
                            with ui.HStack(height=25, spacing=5):
                                ui.Label("Shape", width=115)
                                self._test_shape_combo = ui.ComboBox(0, "Cube", "Sphere")
                            with ui.HStack(height=25, spacing=5):
                                ui.Label("Scale XYZ", width=115)
                                self._test_scale_x_model = ui.SimpleFloatModel(0.05)
                                self._test_scale_y_model = ui.SimpleFloatModel(0.05)
                                self._test_scale_z_model = ui.SimpleFloatModel(0.05)
                                for model in (
                                    self._test_scale_x_model,
                                    self._test_scale_y_model,
                                    self._test_scale_z_model,
                                ):
                                    ui.FloatDrag(model=model, min=0.0001, max=1000.0, step=0.005)
                            ui.Button(
                                "Create test object",
                                height=28,
                                clicked_fn=lambda: self._guard(self._create_test_object),
                                tooltip=(
                                    "원점 (0, 0, 0)에 convex-hull collider를 가진 정적 "
                                    "Cube 또는 Sphere를 생성합니다."
                                ),
                            )

                    with ui.CollapsableFrame("3. Force display settings", collapsed=False):
                        with ui.VStack(spacing=5, height=0):
                            self._refresh_hz_model = self._float_row("UI refresh (Hz)", 20.0, 1.0, 120.0)
                            self._force_scale_model = self._float_row("Bar max (N)", 50.0, 0.001, 1_000_000.0)
                            self._smoothing_model = self._float_row("Smoothing alpha", 0.35, 0.0, 1.0)
                            self._exclude_self_model = ui.SimpleBoolModel(True)
                            with ui.HStack(height=24, spacing=5):
                                ui.CheckBox(model=self._exclude_self_model, width=20)
                                ui.Label("Exclude gripper self-contact")
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
                                "All reports: Fx 0.000  Fy 0.000  Fz 0.000  |ΣF| 0.000 N  Σ|F| 0.000 N",
                                word_wrap=True,
                                height=36,
                            )
                            self._sensor_count_label = ui.Label("Report bodies: 0", height=24)
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
                ui.Label("No PhysxContactReportAPI rigid bodies found.", height=28)
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

    def _build_hand_preset_row(self) -> None:
        items = self._hand_preset_names or ["(Hand preset not matched)"]
        index = min(self._hand_preset_index, len(items) - 1)
        with ui.HStack(height=26, spacing=5):
            ui.Label("Hand preset", width=115)
            combo = ui.ComboBox(index, *items)
            self._hand_preset_combo_model = combo.model

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

    # ---------------------------------------------------------- test object
    def _create_test_object(self) -> None:
        if not self._timeline.is_stopped():
            raise RuntimeError("Stop the timeline before creating a test object.")

        scale = tuple(
            model.get_value_as_float()
            for model in (
                self._test_scale_x_model,
                self._test_scale_y_model,
                self._test_scale_z_model,
            )
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in scale):
            raise ValueError(f"Scale XYZ must be finite and greater than zero: {scale}")

        shape_index = (
            self._test_shape_combo.model.get_item_value_model().get_value_as_int()
        )
        shape_name = "Sphere" if shape_index == 1 else "Cube"

        stage = self._stage()
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, TOOL_ROOT_PATH)
        if stage.GetPrimAtPath(TEST_OBJECT_PATH).IsValid():
            stage.RemovePrim(TEST_OBJECT_PATH)

        success, created_path = omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform",
            prim_type=shape_name,
            prim_path=TEST_OBJECT_PATH,
            prepend_default_prim=False,
            select_new_prim=False,
            object_origin=Gf.Vec3f(0.0, 0.0, 0.0),
            # A one-stage-unit half extent makes the UI values map directly
            # to the authored xform scale (0.05 -> 0.05 stage units radius).
            half_scale=100.0,
        )
        if not success or created_path != TEST_OBJECT_PATH:
            raise RuntimeError(
                f"Isaac mesh creation failed: success={success}, path={created_path}"
            )

        prim = stage.GetPrimAtPath(created_path)
        if not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"Isaac mesh command did not create a Mesh: {created_path}")

        xformable = UsdGeom.Xformable(prim)
        ordered_ops = xformable.GetOrderedXformOps()
        translate_op = next(
            (op for op in ordered_ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
            None,
        )
        scale_op = next(
            (op for op in ordered_ops if op.GetOpType() == UsdGeom.XformOp.TypeScale),
            None,
        )
        if translate_op is not None and scale_op is not None:
            translate_vector = (
                Gf.Vec3d(0.0, 0.0, 0.0)
                if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
                else Gf.Vec3f(0.0, 0.0, 0.0)
            )
            scale_vector = (
                Gf.Vec3d(*scale)
                if scale_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
                else Gf.Vec3f(*scale)
            )
            translate_op.Set(translate_vector)
            scale_op.Set(scale_vector)
        else:
            # This handles a user preference that creates one matrix xform op.
            xformable.ClearXformOpOrder()
            matrix = Gf.Matrix4d(1.0)
            matrix.SetScale(Gf.Vec3d(*scale))
            matrix.SetTranslateOnly(Gf.Vec3d(0.0, 0.0, 0.0))
            xformable.AddTransformOp(opSuffix="contactViewer").Set(matrix)

        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexHull")
        self._ensure_physics_scene()

        omni.usd.get_context().get_selection().set_selected_prim_paths(
            [TEST_OBJECT_PATH], True
        )
        self._set_status(
            f"Created static {shape_name} at (0, 0, 0), scale={scale}, "
            "collider=convexHull (no rigid body)."
        )

    # ---------------------------------------------------------- load/sensors
    @staticmethod
    def _normalized_asset_path(path: str) -> str:
        value = os.path.expanduser(str(path).strip())
        if "://" in value:
            return value.rstrip("/")
        return os.path.normcase(os.path.realpath(value))

    def _load_gripper_profile(self, usd_path: str) -> None:
        """Match a Hand entry by USD path and expose its saved pose presets."""
        previous_name = None
        if self._hand_preset_names:
            previous_name = self._hand_preset_names[
                min(self._hand_preset_index, len(self._hand_preset_names) - 1)
            ]
        self._gripper_type = ""
        self._matched_hand_db_key = None
        self._hand_presets = []
        self._hand_preset_names = []
        self._hand_preset_index = 0

        db_path = Path(self._hand_db_model.get_value_as_string().strip())
        if db_path.is_file():
            with db_path.open("r", encoding="utf-8") as stream:
                database = json.load(stream)
            if not isinstance(database, dict):
                raise ValueError(f"Hand preset DB root must be a JSON object: {db_path}")

            wanted = self._normalized_asset_path(usd_path)
            for key, entry in database.items():
                if not isinstance(entry, dict) or not isinstance(entry.get("usd_path"), str):
                    continue
                if self._normalized_asset_path(entry["usd_path"]) != wanted:
                    continue
                self._gripper_type = str(entry.get("type") or "")
                if self._gripper_type.casefold() == "hand":
                    self._matched_hand_db_key = str(key)
                    raw_presets = entry.get("preset", [])
                    if isinstance(raw_presets, list):
                        self._hand_presets = [item for item in raw_presets if isinstance(item, dict)]
                    self._hand_preset_names = [
                        str(item.get("name") or f"preset_{index + 1}")
                        for index, item in enumerate(self._hand_presets)
                    ]
                    if previous_name in self._hand_preset_names:
                        self._hand_preset_index = self._hand_preset_names.index(previous_name)
                    elif self._hand_presets:
                        self._hand_preset_index = max(
                            range(len(self._hand_presets)),
                            key=lambda index: str(self._hand_presets[index].get("updated_at") or ""),
                        )
                break

        self._hand_preset_frame.rebuild()
        if self._matched_hand_db_key:
            self._gripper_type_label.text = (
                f"Type: Hand | DB: {self._matched_hand_db_key} | "
                f"presets: {len(self._hand_presets)}"
            )
        else:
            self._gripper_type_label.text = "Type: standard gripper (limit control)"

    def _is_hand_gripper(self) -> bool:
        return self._gripper_type.casefold() == "hand" and self._matched_hand_db_key is not None

    def _load_gripper(self) -> None:
        usd_path = self._usd_model.get_value_as_string().strip()
        if not usd_path or not os.path.isfile(usd_path):
            raise FileNotFoundError(f"USD file not found: {usd_path}")

        self._load_gripper_profile(usd_path)

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
        self._joint_infos = self._discover_driven_joints(asset, drive_all_non_mimic=self._is_hand_gripper())
        self._joint_summary_label.text = self._joint_summary()
        self._refresh_sensors()
        self._select_gripper()
        profile = ""
        if self._is_hand_gripper():
            selected = self._selected_hand_preset_name() if self._hand_presets else "none"
            profile = f"; Hand DB={self._matched_hand_db_key}; preset={selected}"
        self._set_status(
            f"Loaded {Path(usd_path).name}; driven joints={len(self._joint_infos)}; "
            f"contact report bodies={len(self._sensor_states)}{profile}. "
            "Press Play to receive force data."
        )

    def _select_gripper(self) -> None:
        self._gripper_prim()
        omni.usd.get_context().get_selection().set_selected_prim_paths([GRIPPER_PATH], True)

    def _rigid_body_ancestor(self, prim: Usd.Prim) -> Optional[Usd.Prim]:
        root_path = GRIPPER_ASSET_PATH.rstrip("/")
        current = prim
        while current.IsValid() and str(current.GetPath()).startswith(root_path):
            if current.HasAPI(UsdPhysics.RigidBodyAPI):
                return current
            current = current.GetParent()
        return None

    def _refresh_sensors(self) -> None:
        root = self._stage().GetPrimAtPath(GRIPPER_ASSET_PATH)
        if not root.IsValid():
            raise RuntimeError("Load the gripper first.")

        reports: list[Usd.Prim] = []
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                rigid_body = self._rigid_body_ancestor(prim)
                if rigid_body is not None:
                    reports.append(rigid_body)

        self._unresolved_reports = []
        report_paths = sorted({str(prim.GetPath()) for prim in reports})
        for body_path in report_paths:
            body_prim = self._stage().GetPrimAtPath(body_path)
            PhysxSchema.PhysxContactReportAPI.Apply(body_prim).CreateThresholdAttr().Set(0.0)

        previous = self._sensor_states
        self._sensor_states = {}
        for body_path in report_paths:
            state = previous.get(body_path, SensorState(body_path, body_path))
            state.body_path = body_path
            self._sensor_states[body_path] = state

        self._sensor_count_label.text = (
            f"Report bodies: {len(self._sensor_states)}  | ContactReport prims: {len(reports)}"
            f"  | Unresolved: {len(self._unresolved_reports)}"
        )
        self._sensor_frame.rebuild()

    def _add_report_to_selected(self) -> None:
        if not self._timeline.is_stopped():
            raise RuntimeError("Stop the timeline before adding a ContactReportAPI.")
        paths = omni.usd.get_context().get_selection().get_selected_prim_paths()
        if not paths:
            raise RuntimeError("Select one or more gripper rigid bodies/colliders first.")
        added = []
        for path in paths:
            prim = self._stage().GetPrimAtPath(path)
            rigid_body = self._rigid_body_ancestor(prim)
            if rigid_body is None:
                raise RuntimeError(f"No gripper rigid-body ancestor: {path}")
            PhysxSchema.PhysxContactReportAPI.Apply(rigid_body).CreateThresholdAttr().Set(0.0)
            added.append(str(rigid_body.GetPath()))
        self._refresh_sensors()
        self._set_status(f"Contact reports ready: {', '.join(sorted(set(added)))}")

    # --------------------------------------------------------------- joints
    def _mimic_master_joint_paths(self, root: Usd.Prim) -> set[str]:
        """Return joints referenced as masters by PhysX mimic relationships."""
        masters: set[str] = set()
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(PhysxSchema.PhysxMimicJointAPI):
                continue
            for relationship in prim.GetRelationships():
                name = relationship.GetName()
                if name.startswith("physxMimicJoint:") and name.endswith(":referenceJoint"):
                    masters.update(str(target) for target in relationship.GetTargets())
        return masters

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

    def _discover_driven_joints(
        self, root: Usd.Prim, drive_all_non_mimic: bool = False
    ) -> Dict[str, JointInfo]:
        result: Dict[str, JointInfo] = {}
        self._skipped_joint_limits = []
        mimic_masters = self._mimic_master_joint_paths(root)
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(PhysxSchema.PhysxMimicJointAPI):
                continue
            # In a mimic gripper, drive only the explicitly referenced master.
            # Internal linkage drives keep their USD-authored gains and targets.
            if not drive_all_non_mimic and mimic_masters and str(prim.GetPath()) not in mimic_masters:
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
            name_override = prim.GetAttribute("isaac:nameOverride")
            joint_name = (
                str(name_override.Get())
                if name_override and name_override.Get()
                else prim.GetName()
            )
            if joint_name in result:
                raise RuntimeError(f"Duplicate articulation DOF name: {joint_name}")
            result[joint_name] = JointInfo(
                joint_name, str(prim.GetPath()), lower_value, upper_value, drive_name, revolute
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

    def _apply_drive_gains_with_status(self) -> None:
        if not self._joint_infos:
            raise RuntimeError("No controller joints were found in the loaded gripper.")
        self._apply_drive_gains()
        self._set_status(
            f"Drive settings applied to {len(self._joint_infos)} controller joint(s): "
            f"stiffness={self._stiffness_model.get_value_as_float():g}, "
            f"damping={self._damping_model.get_value_as_float():g}, "
            f"max={self._max_force_model.get_value_as_float():g}."
        )

    def _set_joint_target(self, info: JointInfo, value: float) -> None:
        value = max(info.lower, min(info.upper, float(value)))
        prim = self._stage().GetPrimAtPath(info.prim_path)
        drive = UsdPhysics.DriveAPI.Get(prim, info.drive_name)
        usd_value = math.degrees(value) if info.revolute else value
        drive.GetTargetPositionAttr().Set(usd_value)
        if self._timeline.is_stopped():
            PhysxSchema.JointStateAPI.Apply(prim, info.drive_name).CreatePositionAttr().Set(usd_value)

    def _selected_hand_preset(self) -> Tuple[str, dict]:
        if not self._hand_presets:
            raise RuntimeError(
                "This USD is type=Hand, but its database entry has no pose preset. "
                f"Check {self._hand_db_model.get_value_as_string().strip()}."
            )
        index = self._hand_preset_index
        if self._hand_preset_combo_model is not None:
            index = self._hand_preset_combo_model.get_item_value_model().get_value_as_int()
        if index < 0 or index >= len(self._hand_presets):
            raise IndexError(f"Hand preset index out of range: {index}")
        self._hand_preset_index = index
        return self._hand_preset_names[index], self._hand_presets[index]

    def _selected_hand_preset_name(self) -> str:
        return self._selected_hand_preset()[0]

    def _hand_joint_targets(self, close: bool) -> Tuple[str, Dict[str, float], list[str]]:
        preset_name, preset = self._selected_hand_preset()
        field_name = "end_joint_pos" if close else "start_joint_pos"
        raw_targets = preset.get(field_name)
        if not isinstance(raw_targets, dict):
            raise ValueError(f"Hand preset {preset_name!r} has no valid {field_name} object.")

        unit = str(preset.get("joint_unit") or "rad").casefold()
        targets: Dict[str, float] = {}
        for name, raw_value in raw_targets.items():
            info = self._joint_infos.get(str(name))
            if info is None or not isinstance(raw_value, (int, float)):
                continue
            value = float(raw_value)
            if not math.isfinite(value):
                continue
            if info.revolute and unit in ("deg", "degree", "degrees"):
                value = math.radians(value)
            targets[info.name] = value
        if not targets:
            raise ValueError(
                f"Hand preset {preset_name!r} {field_name} has no joint names matching the loaded USD."
            )
        missing = sorted(set(self._joint_infos) - set(targets))
        return preset_name, targets, missing

    def _set_gripper(self, close: bool) -> None:
        if not self._joint_infos:
            raise RuntimeError("No driven joints were found in the loaded gripper.")
        reverse = self._reverse_model.get_value_as_bool()
        use_upper = close != reverse
        self._apply_drive_gains()
        if self._is_hand_gripper():
            preset_name, targets, missing = self._hand_joint_targets(use_upper)
            for name, value in targets.items():
                self._set_joint_target(self._joint_infos[name], value)
            action = "CLOSE/end" if use_upper else "OPEN/start"
            message = (
                f"Hand preset {preset_name!r}: {action} applied to {len(targets)} joint(s)."
            )
            if missing:
                message += f" Preset has no value for: {', '.join(missing)}"
            self._set_status(message)
            return
        for info in self._joint_infos.values():
            self._set_joint_target(info, info.upper if use_upper else info.lower)
        self._set_status("CLOSE target applied." if close else "OPEN target applied.")

    def _play(self) -> None:
        self._gripper_prim()
        self._ensure_physics_scene()
        self._apply_drive_gains()
        for state in self._sensor_states.values():
            state.accumulated_impulse[:] = 0.0
            state.accumulated_time = 0.0
        self._timeline.play()
        self._set_status("Timeline playing; monitoring contact forces.")

    # --------------------------------------------------------- force update
    def _decode_body(self, encoded) -> str:
        try:
            return str(self._contact_interface.decode_body_name(int(encoded)))
        except Exception:
            return ""

    def _contact_impulse(self, state: SensorState) -> Tuple[np.ndarray, int]:
        """Read this report body's current PhysX contact impulses directly."""
        raw_data = self._contact_interface.get_rigid_body_raw_data(state.body_path)
        impulse_sum = np.zeros(3, dtype=float)
        contacts = 0
        exclude_self = self._exclude_self_model.get_value_as_bool()
        for contact in raw_data:
            impulse = _vector(contact, "impulse")
            body0 = self._decode_body(contact["body0"])
            body1 = self._decode_body(contact["body1"])
            body_is_0 = _path_matches(body0, state.body_path)
            body_is_1 = _path_matches(body1, state.body_path)
            if not body_is_0 and not body_is_1:
                continue
            other_body = body1 if body_is_0 else body0
            if exclude_self and other_body.startswith(GRIPPER_ASSET_PATH.rstrip("/") + "/"):
                continue
            # PhysX raw impulse points toward body1. Convert it to the force
            # acting on the report body.
            if body_is_0 and not body_is_1:
                impulse = -impulse
            impulse_sum += impulse
            contacts += 1
        return impulse_sum, contacts

    def _on_physics_step(self, step_dt: float) -> None:
        """Accumulate impulses at physics rate so the slower UI misses no steps."""
        if not self._monitor_enabled or not self._sensor_states:
            return
        dt = max(float(step_dt), 1.0e-9)
        try:
            for state in self._sensor_states.values():
                impulse, contacts = self._contact_impulse(state)
                state.accumulated_impulse += impulse
                state.accumulated_time += dt
                state.contacts = contacts
                state.valid = True
        except Exception as error:
            carb.log_warn(f"[{WINDOW_TITLE}] Physics contact accumulation failed: {error}")

    def _consume_sensor_force(self, state: SensorState) -> Tuple[np.ndarray, int, bool]:
        if state.accumulated_time <= 0.0:
            return state.raw_force.copy(), state.contacts, state.valid
        force = state.accumulated_impulse / state.accumulated_time
        state.accumulated_impulse[:] = 0.0
        state.accumulated_time = 0.0
        return force, state.contacts, state.valid

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
            raw_force, contacts, valid = self._consume_sensor_force(state)
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
            f"All reports: Fx {total_vector[0]:+.3f}  Fy {total_vector[1]:+.3f}  "
            f"Fz {total_vector[2]:+.3f}  |ΣF| {resultant:.3f} N  Σ|F| {magnitude_sum:.3f} N"
        )

    def _toggle_monitor(self) -> None:
        self._monitor_enabled = not self._monitor_enabled
        self._monitor_button.text = "Pause monitor" if self._monitor_enabled else "Resume monitor"

    def _zero_current(self) -> None:
        for state in self._sensor_states.values():
            force, contacts, valid = self._consume_sensor_force(state)
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
