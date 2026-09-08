#!/usr/bin/env python3
"""Visually compare Dataset_2026 pointcloud instance mappings with inst_seg.

The UI shows the inst_seg RGBA image, a sampled 3-D point cloud colored with
the exact same RGBA mapping, and per-instance point/pixel counts. It also
re-validates ID, color, class, and counts against the current source files.
"""

from __future__ import annotations

import json
import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import customtkinter as ctk
import matplotlib
import numpy as np
from PIL import Image

from create_pointcloud_instance_mappings import load_png_counts, parse_rgba, rgba_to_uint32


matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# =============================================================================
# 사용자 설정
# =============================================================================
DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/dataset_v2")
INITIAL_SELECTION = ("Home", "MasterBedroom", "bed_01", "side_view_camera")
MAX_DISPLAY_POINTS = 100_000


CAMERA_NAMES = ("top_view_camera", "side_view_camera")
XYZ_RE = re.compile(r"^pointcloud_(\d+)\.npy$")
INSTANCE_RE = re.compile(r"^pointcloud_inst_seg_(\d+)\.npy$")
MAPPING_RE = re.compile(r"^pointcloud_inst_seg_mapping_(\d+)\.json$")


@dataclass(frozen=True)
class InstanceEntry:
    instance_id: int
    class_name: str
    rgba: tuple[int, int, int, int]
    point_count: int
    pixel_count: int


@dataclass
class SceneData:
    scene_id: int
    points: np.ndarray
    point_colors: np.ndarray
    image: np.ndarray
    entries: list[InstanceEntry]
    source_count: int
    displayed_count: int
    file_summary: str


def directory_names(path: Path) -> list[str]:
    try:
        return sorted(
            child.name
            for child in path.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        )
    except OSError:
        return []


def set_combo_values(
    combo: ctk.CTkComboBox, values: Iterable[str], preferred: str = ""
) -> str:
    values = list(values)
    combo.configure(values=values or [""])
    selected = preferred if preferred in values else (values[0] if values else "")
    combo.set(selected)
    return selected


def file_scene_ids(directory: Path, pattern: re.Pattern[str]) -> set[int]:
    result = set()
    try:
        for path in directory.iterdir():
            match = pattern.fullmatch(path.name)
            if match and path.is_file():
                result.add(int(match.group(1)))
    except OSError:
        pass
    return result


def load_scene(
    platform: Path,
    camera: str,
    scene_id: int,
    max_points: int,
) -> SceneData:
    stem = f"{scene_id:04d}"
    pointcloud_dir = platform / "pointcloud" / camera
    inst_seg_dir = platform / "inst_seg" / camera
    paths = {
        "xyz": pointcloud_dir / f"pointcloud_{stem}.npy",
        "instance": pointcloud_dir / f"pointcloud_inst_seg_{stem}.npy",
        "mapping": pointcloud_dir / f"pointcloud_inst_seg_mapping_{stem}.json",
        "png": inst_seg_dir / f"{stem}.png",
        "semantics": inst_seg_dir / f"semantics_mapping_{stem}.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} file not found: {path}")

    xyz = np.load(paths["xyz"], mmap_mode="r", allow_pickle=False)
    instance_ids = np.load(paths["instance"], mmap_mode="r", allow_pickle=False)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"unexpected XYZ shape: {xyz.shape}")
    if instance_ids.ndim != 1 or len(instance_ids) != len(xyz):
        raise ValueError(
            f"XYZ/instance shape mismatch: XYZ={xyz.shape}, instance={instance_ids.shape}"
        )

    with paths["mapping"].open("r", encoding="utf-8") as stream:
        mapping_document = json.load(stream)
    with paths["semantics"].open("r", encoding="utf-8") as stream:
        semantics_document = json.load(stream)
    raw_instances = mapping_document.get("instances")
    if not isinstance(raw_instances, dict) or not raw_instances:
        raise ValueError("pointcloud mapping must contain a non-empty instances object")
    if not isinstance(semantics_document, dict) or not semantics_document:
        raise ValueError("semantics mapping must be a non-empty object")

    semantics_by_rgba = {}
    for color_key, labels in semantics_document.items():
        rgba = parse_rgba(color_key)
        if rgba in semantics_by_rgba:
            raise ValueError(f"duplicate RGBA in semantics mapping: {rgba}")
        semantics_by_rgba[rgba] = labels if isinstance(labels, dict) else {"class": str(labels)}

    png_counts, image_shape = load_png_counts(paths["png"])
    unique_ids, current_counts = np.unique(instance_ids, return_counts=True)
    current_count_by_id = {
        int(instance_id): int(count)
        for instance_id, count in zip(unique_ids, current_counts)
    }
    mapped_ids = {int(value) for value in raw_instances}
    if mapped_ids != set(current_count_by_id):
        missing = sorted(set(current_count_by_id) - mapped_ids)
        extra = sorted(mapped_ids - set(current_count_by_id))
        raise ValueError(f"mapping ID mismatch: missing={missing}, extra={extra}")

    entries = []
    for key, raw_entry in raw_instances.items():
        if not isinstance(raw_entry, dict):
            raise ValueError(f"instance {key} mapping entry must be an object")
        instance_id = int(key)
        rgba_value = raw_entry.get("rgba")
        if not isinstance(rgba_value, list) or len(rgba_value) != 4:
            raise ValueError(f"instance {key} has invalid rgba: {rgba_value!r}")
        rgba = tuple(int(channel) for channel in rgba_value)
        class_name = str(raw_entry.get("class", ""))
        semantics_entry = semantics_by_rgba.get(rgba)
        if semantics_entry is None:
            raise ValueError(f"instance {key} RGBA {rgba} is absent from semantics mapping")
        if str(semantics_entry.get("class", "")) != class_name:
            raise ValueError(
                f"instance {key} class mismatch: pointcloud={class_name!r}, "
                f"inst_seg={semantics_entry.get('class')!r}"
            )

        point_count = current_count_by_id[instance_id]
        pixel_count = int(png_counts.get(rgba_to_uint32(rgba), 0))
        if point_count != pixel_count:
            raise ValueError(
                f"instance {key} count mismatch: points={point_count}, pixels={pixel_count}"
            )
        if int(raw_entry.get("point_count", -1)) != point_count:
            raise ValueError(f"instance {key} stored point_count is stale")
        if int(raw_entry.get("pixel_count", -1)) != pixel_count:
            raise ValueError(f"instance {key} stored pixel_count is stale")
        entries.append(InstanceEntry(instance_id, class_name, rgba, point_count, pixel_count))

    if len(xyz) != image_shape[0] * image_shape[1]:
        raise ValueError(
            f"point count {len(xyz)} != PNG pixel count {image_shape[0] * image_shape[1]}"
        )

    sample_count = min(len(xyz), max_points)
    indices = np.linspace(0, len(xyz) - 1, sample_count, dtype=np.int64)
    points = np.asarray(xyz[indices, :3], dtype=np.float32)
    sampled_ids = np.asarray(instance_ids[indices], dtype=np.int64)
    color_by_id = {
        entry.instance_id: np.asarray(entry.rgba[:3], dtype=np.float32) / 255.0
        for entry in entries
    }
    point_colors = np.empty((len(sampled_ids), 3), dtype=np.float32)
    for instance_id, color in color_by_id.items():
        point_colors[sampled_ids == instance_id] = color

    valid = np.isfinite(points).all(axis=1)
    valid &= np.linalg.norm(points, axis=1) > 1e-10
    points = points[valid]
    point_colors = point_colors[valid]
    with Image.open(paths["png"]) as source_image:
        image = np.asarray(source_image.convert("RGBA"), dtype=np.uint8).copy()

    summary = "\n".join(f"✓ {path.name}" for path in paths.values())
    return SceneData(
        scene_id=scene_id,
        points=points,
        point_colors=point_colors,
        image=image,
        entries=entries,
        source_count=len(xyz),
        displayed_count=int(valid.sum()),
        file_summary=summary,
    )


class MappingViewer:
    def __init__(self, root: ctk.CTk, dataset_root: Path, max_points: int) -> None:
        self.root = root
        self.dataset_root = dataset_root
        self.max_points = max_points
        self.scene_ids: list[int] = []
        self.scene_index = 0
        self.load_generation = 0
        self.load_queue: queue.Queue[tuple[int, SceneData | None, str | None]] = queue.Queue()

        root.title("PCD ↔ Instance Segmentation Mapping Validator")
        root.geometry("1840x980")
        root.minsize(1250, 720)
        self._build_ui()
        self._bind_keys()
        self._load_initial_selection()
        root.after(60, self._poll_load_queue)

    def _build_ui(self) -> None:
        outer = ctk.CTkFrame(self.root, corner_radius=0)
        outer.pack(fill="both", expand=True)
        controls = ctk.CTkFrame(outer, width=300, corner_radius=0)
        controls.pack(side="left", fill="y")
        controls.pack_propagate(False)
        viewer = ctk.CTkFrame(outer, corner_radius=0)
        viewer.pack(side="right", fill="both", expand=True)

        ctk.CTkLabel(
            controls,
            text="PCD ↔ inst_seg\nMapping Validator",
            font=ctk.CTkFont(size=21, weight="bold"),
        ).pack(padx=16, pady=(20, 16))

        root_box = ctk.CTkFrame(controls)
        root_box.pack(fill="x", padx=14, pady=(0, 10))
        ctk.CTkLabel(root_box, text="Dataset root", anchor="w").pack(
            fill="x", padx=10, pady=(8, 2)
        )
        self.root_entry = ctk.CTkEntry(root_box)
        self.root_entry.insert(0, str(self.dataset_root))
        self.root_entry.pack(fill="x", padx=10, pady=(0, 7))
        ctk.CTkButton(root_box, text="Rescan", command=self.rescan_root).pack(
            fill="x", padx=10, pady=(0, 9)
        )

        self.env_combo = self._combo(controls, "Environment", self.on_env_changed)
        self.section_combo = self._combo(controls, "Section", self.on_section_changed)
        self.platform_combo = self._combo(controls, "Platform", self.on_platform_changed)
        self.camera_combo = self._combo(controls, "Camera", self.on_camera_changed)

        sample_box = ctk.CTkFrame(controls)
        sample_box.pack(fill="x", padx=14, pady=(13, 0))
        ctk.CTkLabel(sample_box, text="3D display samples", anchor="w").pack(
            fill="x", padx=10, pady=(8, 3)
        )
        sample_row = ctk.CTkFrame(sample_box, fg_color="transparent")
        sample_row.pack(fill="x", padx=10, pady=(0, 8))
        self.sample_entry = ctk.CTkEntry(sample_row)
        self.sample_entry.insert(0, str(self.max_points))
        self.sample_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(sample_row, text="Apply", width=62, command=self.apply_sample_count).pack(
            side="right"
        )

        nav = ctk.CTkFrame(controls)
        nav.pack(fill="x", padx=14, pady=(14, 8))
        ctk.CTkLabel(nav, text="Scene", font=ctk.CTkFont(weight="bold")).pack(
            pady=(8, 3)
        )
        self.scene_label = ctk.CTkLabel(nav, text="- / -", font=ctk.CTkFont(size=17))
        self.scene_label.pack(pady=3)
        jump = ctk.CTkFrame(nav, fg_color="transparent")
        jump.pack(fill="x", padx=9, pady=4)
        self.scene_entry = ctk.CTkEntry(jump, placeholder_text="scene number")
        self.scene_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(jump, text="Go", width=45, command=self.jump_to_scene).pack(side="right")
        buttons = ctk.CTkFrame(nav, fg_color="transparent")
        buttons.pack(fill="x", padx=9, pady=(3, 9))
        self.prev_button = ctk.CTkButton(buttons, text="◀ Prev", command=self.previous_scene)
        self.prev_button.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.next_button = ctk.CTkButton(buttons, text="Next ▶", command=self.next_scene)
        self.next_button.pack(side="right", fill="x", expand=True, padx=(3, 0))

        self.status_label = ctk.CTkLabel(
            controls, text="", justify="left", anchor="nw", wraplength=270
        )
        self.status_label.pack(fill="x", padx=17, pady=(8, 4))
        ctk.CTkLabel(
            controls,
            text="←/A previous   →/D next\nT top   S side   R reload",
            justify="left",
            text_color=("gray35", "gray70"),
        ).pack(side="bottom", fill="x", padx=18, pady=16)

        self.figure = Figure(figsize=(15, 8), dpi=100, facecolor="#202124")
        grid = self.figure.add_gridspec(1, 3, width_ratios=(1.25, 1.0, 1.05))
        self.pcd_axis = self.figure.add_subplot(grid[0, 0], projection="3d")
        self.image_axis = self.figure.add_subplot(grid[0, 1])
        self.count_axis = self.figure.add_subplot(grid[0, 2])
        self.figure.subplots_adjust(left=0.025, right=0.985, bottom=0.08, top=0.92, wspace=0.2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=viewer)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, viewer, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        self._draw_empty("Generate pointcloud mapping files, then select a scene")

    def _combo(self, parent: ctk.CTkFrame, label: str, command) -> ctk.CTkComboBox:
        ctk.CTkLabel(parent, text=label, anchor="w").pack(
            fill="x", padx=18, pady=(7, 2)
        )
        combo = ctk.CTkComboBox(parent, values=[""], command=command, state="readonly")
        combo.pack(fill="x", padx=18)
        return combo

    def _bind_keys(self) -> None:
        for key in ("<Left>", "<Key-a>", "<Key-A>"):
            self.root.bind(key, lambda _event: self.previous_scene())
        for key in ("<Right>", "<Key-d>", "<Key-D>"):
            self.root.bind(key, lambda _event: self.next_scene())
        self.root.bind("<Key-t>", lambda _event: self.select_camera("top_view_camera"))
        self.root.bind("<Key-T>", lambda _event: self.select_camera("top_view_camera"))
        self.root.bind("<Key-s>", lambda _event: self.select_camera("side_view_camera"))
        self.root.bind("<Key-S>", lambda _event: self.select_camera("side_view_camera"))
        self.root.bind("<Key-r>", lambda _event: self.load_current_scene())
        self.root.bind("<Key-R>", lambda _event: self.load_current_scene())
        self.scene_entry.bind("<Return>", lambda _event: self.jump_to_scene())
        self.sample_entry.bind("<Return>", lambda _event: self.apply_sample_count())

    def _load_initial_selection(self) -> None:
        env, section, platform, camera = INITIAL_SELECTION
        self._refresh_envs(env)
        self._refresh_sections(section)
        self._refresh_platforms(platform)
        self._refresh_cameras(camera)
        self.scan_scenes()

    def _platform_path(self) -> Path:
        return (
            self.dataset_root
            / self.env_combo.get()
            / self.section_combo.get()
            / self.platform_combo.get()
        )

    def _pointcloud_path(self) -> Path:
        return self._platform_path() / "pointcloud" / self.camera_combo.get()

    def _inst_seg_path(self) -> Path:
        return self._platform_path() / "inst_seg" / self.camera_combo.get()

    def _refresh_envs(self, preferred: str = "") -> None:
        set_combo_values(self.env_combo, directory_names(self.dataset_root), preferred)

    def _refresh_sections(self, preferred: str = "") -> None:
        set_combo_values(
            self.section_combo,
            directory_names(self.dataset_root / self.env_combo.get()),
            preferred,
        )

    def _refresh_platforms(self, preferred: str = "") -> None:
        base = self.dataset_root / self.env_combo.get() / self.section_combo.get()
        values = [name for name in directory_names(base) if (base / name / "pointcloud").is_dir()]
        set_combo_values(self.platform_combo, values, preferred)

    def _refresh_cameras(self, preferred: str = "") -> None:
        base = self._platform_path() / "pointcloud"
        values = [name for name in CAMERA_NAMES if (base / name).is_dir()]
        set_combo_values(self.camera_combo, values, preferred)

    def rescan_root(self) -> None:
        candidate = Path(self.root_entry.get().strip()).expanduser()
        if not candidate.is_dir():
            self._set_status(f"Dataset root not found:\n{candidate}", error=True)
            return
        self.dataset_root = candidate
        self._refresh_envs()
        self._refresh_sections()
        self._refresh_platforms()
        self._refresh_cameras()
        self.scan_scenes()

    def on_env_changed(self, _choice: str) -> None:
        self._refresh_sections()
        self._refresh_platforms()
        self._refresh_cameras()
        self.scan_scenes()

    def on_section_changed(self, _choice: str) -> None:
        self._refresh_platforms()
        self._refresh_cameras()
        self.scan_scenes()

    def on_platform_changed(self, _choice: str) -> None:
        self._refresh_cameras()
        self.scan_scenes()

    def on_camera_changed(self, _choice: str) -> None:
        self.scan_scenes()

    def select_camera(self, camera: str) -> None:
        if camera in self.camera_combo.cget("values"):
            self.camera_combo.set(camera)
            self.scan_scenes()

    def scan_scenes(self) -> None:
        pointcloud_dir = self._pointcloud_path()
        inst_seg_dir = self._inst_seg_path()
        id_sets = [
            file_scene_ids(pointcloud_dir, XYZ_RE),
            file_scene_ids(pointcloud_dir, INSTANCE_RE),
            file_scene_ids(pointcloud_dir, MAPPING_RE),
            file_scene_ids(inst_seg_dir, re.compile(r"^(\d+)\.png$")),
            file_scene_ids(inst_seg_dir, re.compile(r"^semantics_mapping_(\d+)\.json$")),
        ]
        complete = set.intersection(*id_sets) if id_sets else set()
        all_ids = set.union(*id_sets) if id_sets else set()
        self.scene_ids = sorted(complete)
        self.scene_index = 0
        if not self.scene_ids:
            self._draw_empty("No complete mapping comparison scenes")
            self._set_status(
                f"Complete scenes: 0\nIncomplete scenes: {len(all_ids)}", error=True
            )
            self._update_controls()
            return
        incomplete = len(all_ids - complete)
        self._set_status(
            f"Complete scenes: {len(self.scene_ids):,}\nIncomplete scenes: {incomplete}",
            error=bool(incomplete),
        )
        self.load_current_scene()

    def previous_scene(self) -> None:
        if self.scene_index > 0:
            self.scene_index -= 1
            self.load_current_scene()

    def next_scene(self) -> None:
        if self.scene_ids and self.scene_index < len(self.scene_ids) - 1:
            self.scene_index += 1
            self.load_current_scene()

    def jump_to_scene(self) -> None:
        try:
            self.scene_index = self.scene_ids.index(int(self.scene_entry.get().strip()))
        except ValueError:
            self._set_status("Scene number is not available.", error=True)
            return
        self.load_current_scene()

    def apply_sample_count(self) -> None:
        try:
            value = int(self.sample_entry.get().strip().replace(",", "").replace("_", ""))
        except ValueError:
            self._set_status("Sample points must be an integer.", error=True)
            return
        if not 1_000 <= value <= 2_000_000:
            self._set_status("Samples must be between 1,000 and 2,000,000.", error=True)
            return
        self.max_points = value
        self.load_current_scene()

    def load_current_scene(self) -> None:
        if not self.scene_ids:
            return
        self.load_generation += 1
        generation = self.load_generation
        scene_id = self.scene_ids[self.scene_index]
        self._set_status(f"Loading and validating scene {scene_id:04d} ...")
        self._update_controls(loading=True)
        threading.Thread(
            target=self._load_worker,
            args=(generation, self._platform_path(), self.camera_combo.get(), scene_id),
            daemon=True,
        ).start()

    def _load_worker(
        self, generation: int, platform: Path, camera: str, scene_id: int
    ) -> None:
        try:
            data = load_scene(platform, camera, scene_id, self.max_points)
            self.load_queue.put((generation, data, None))
        except Exception as error:
            self.load_queue.put(
                (generation, None, f"{type(error).__name__}: {error}")
            )

    def _poll_load_queue(self) -> None:
        try:
            while True:
                generation, data, error = self.load_queue.get_nowait()
                if generation != self.load_generation:
                    continue
                if error:
                    self._draw_empty(f"Validation failed\n{error}")
                    self._set_status(error, error=True)
                    self._update_controls()
                elif data is not None:
                    self._render(data)
        except queue.Empty:
            pass
        self.root.after(60, self._poll_load_queue)

    def _render(self, data: SceneData) -> None:
        if not len(data.points):
            self._draw_empty("No valid 3-D points")
            self._set_status("No valid 3-D points", error=True)
            self._update_controls()
            return

        for axis in (self.pcd_axis, self.image_axis, self.count_axis):
            axis.clear()
            axis.set_facecolor("#202124")

        mins = data.points.min(axis=0)
        maxs = data.points.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = max(float(np.max(maxs - mins)) / 2.0, 1e-3)
        self.pcd_axis.scatter(
            data.points[:, 0],
            data.points[:, 1],
            data.points[:, 2],
            c=data.point_colors,
            s=0.4,
            marker=".",
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )
        self.pcd_axis.set_xlim(center[0] - radius, center[0] + radius)
        self.pcd_axis.set_ylim(center[1] - radius, center[1] + radius)
        self.pcd_axis.set_zlim(center[2] - radius, center[2] + radius)
        self.pcd_axis.set_box_aspect((1, 1, 1))
        self.pcd_axis.view_init(elev=28, azim=-60)
        self.pcd_axis.set_title("PCD colored by instance mapping", color="white")
        self.pcd_axis.tick_params(colors="white", labelsize=7)

        self.image_axis.imshow(data.image)
        self.image_axis.set_title("inst_seg PNG", color="white")
        self.image_axis.set_axis_off()

        y = np.arange(len(data.entries))
        counts = np.asarray([entry.point_count for entry in data.entries])
        colors = np.asarray([entry.rgba[:3] for entry in data.entries], dtype=float) / 255.0
        labels = [f"{entry.instance_id}: {entry.class_name}" for entry in data.entries]
        self.count_axis.barh(y, counts, color=colors, alpha=0.82, label="PCD points")
        self.count_axis.scatter(
            [entry.pixel_count for entry in data.entries],
            y,
            marker="|",
            s=180,
            color="white",
            label="PNG pixels",
        )
        self.count_axis.set_yticks(y, labels=labels, color="white", fontsize=8)
        self.count_axis.invert_yaxis()
        self.count_axis.set_xscale("log")
        self.count_axis.tick_params(axis="x", colors="white", labelsize=8)
        self.count_axis.set_xlabel("count", color="white")
        self.count_axis.set_title("ID / class / count verification", color="white")
        self.count_axis.grid(axis="x", alpha=0.18)
        legend = self.count_axis.legend(fontsize=8)
        legend.get_frame().set_alpha(0.25)

        self.figure.suptitle(
            f"{self.env_combo.get()} / {self.section_combo.get()} / "
            f"{self.platform_combo.get()}  •  {self.camera_combo.get()}  •  "
            f"scene {data.scene_id:04d}  •  VERIFIED",
            color="#70e080",
            fontsize=13,
        )
        self.canvas.draw_idle()
        self._set_status(
            f"✓ VERIFIED scene {data.scene_id:04d}\n"
            f"Instances: {len(data.entries)}\n"
            f"Displayed: {data.displayed_count:,} / {data.source_count:,}\n\n"
            f"{data.file_summary}"
        )
        self._update_controls()

    def _draw_empty(self, message: str) -> None:
        for axis in (self.pcd_axis, self.image_axis, self.count_axis):
            axis.clear()
            axis.set_facecolor("#202124")
            axis.set_axis_off()
        self.image_axis.text(
            0.5,
            0.5,
            message,
            transform=self.image_axis.transAxes,
            color="white",
            ha="center",
            va="center",
            fontsize=12,
            wrap=True,
        )
        self.figure.suptitle("")
        self.canvas.draw_idle()

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.configure(
            text=text,
            text_color=("#b00020", "#ff7777") if error else ("gray20", "gray85"),
        )

    def _update_controls(self, loading: bool = False) -> None:
        if self.scene_ids:
            scene_id = self.scene_ids[self.scene_index]
            self.scene_label.configure(
                text=f"{scene_id:04d} ({self.scene_index + 1:,}/{len(self.scene_ids):,})"
            )
        else:
            self.scene_label.configure(text="- / -")
        state = "disabled" if loading else "normal"
        self.prev_button.configure(state=state if self.scene_index > 0 else "disabled")
        self.next_button.configure(
            state=(
                state
                if self.scene_ids and self.scene_index < len(self.scene_ids) - 1
                else "disabled"
            )
        )


def main() -> None:
    if MAX_DISPLAY_POINTS < 1:
        raise SystemExit("MAX_DISPLAY_POINTS must be at least 1")
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    MappingViewer(root, DATASET_ROOT, MAX_DISPLAY_POINTS)
    root.mainloop()


if __name__ == "__main__":
    main()
