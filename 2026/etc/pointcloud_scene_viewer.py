#!/usr/bin/env python3
"""Dataset 2026 point-cloud triplet viewer.

Each scene is stored as three aligned arrays:

* pointcloud_####.npy          : XYZ positions
* pointcloud_rgb_####.npy      : RGB(A) colors
* pointcloud_inst_seg_####.npy : instance IDs

The viewer renders the same sampled XYZ points as height, RGB, and instance
views.  The source arrays are never modified.
"""

from __future__ import annotations

import argparse
import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import customtkinter as ctk
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


DEFAULT_DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/dataset_v2")
DEFAULT_SELECTION = ("Home", "MasterBedroom", "bed_01", "side_view_camera")
VIEW_NAMES = ("top_view_camera", "side_view_camera")
FILE_PATTERNS = {
    "xyz": re.compile(r"^pointcloud_(\d+)\.npy$"),
    "rgb": re.compile(r"^pointcloud_rgb_(\d+)\.npy$"),
    "instance": re.compile(r"^pointcloud_inst_seg_(\d+)\.npy$"),
}


@dataclass
class SceneData:
    scene_id: int
    points: np.ndarray
    rgb: np.ndarray
    instance_ids: np.ndarray
    source_count: int
    valid_count: int
    file_summary: str


def directory_names(path: Path) -> list[str]:
    """Return sorted, non-hidden child directory names."""
    try:
        return sorted(p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        return []


def set_combo_values(combo: ctk.CTkComboBox, values: Iterable[str], preferred: str = "") -> str:
    values = list(values)
    combo.configure(values=values or [""])
    selected = preferred if preferred in values else (values[0] if values else "")
    combo.set(selected)
    return selected


class PointCloudSceneViewer:
    def __init__(self, root: ctk.CTk, dataset_root: Path, max_points: int) -> None:
        self.root = root
        self.dataset_root = dataset_root
        self.max_points = max_points
        self.scene_ids: list[int] = []
        self.scene_index = 0
        self.load_generation = 0
        self.load_queue: queue.Queue[tuple[int, SceneData | None, str | None]] = queue.Queue()

        self.root.title("Dataset Point-cloud Scene Viewer")
        self.root.geometry("1760x980")
        self.root.minsize(1150, 700)

        self._build_ui()
        self._bind_keys()
        self._load_initial_selection()
        self.root.after(60, self._poll_load_queue)

    def _build_ui(self) -> None:
        outer = ctk.CTkFrame(self.root, corner_radius=0)
        outer.pack(fill="both", expand=True)

        controls = ctk.CTkFrame(outer, width=285, corner_radius=0)
        controls.pack(side="left", fill="y")
        controls.pack_propagate(False)

        viewer = ctk.CTkFrame(outer, corner_radius=0)
        viewer.pack(side="right", fill="both", expand=True)

        ctk.CTkLabel(
            controls,
            text="Point-cloud\nScene Viewer",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(padx=18, pady=(22, 18))

        root_box = ctk.CTkFrame(controls)
        root_box.pack(fill="x", padx=14, pady=(0, 12))
        ctk.CTkLabel(root_box, text="Dataset root", anchor="w").pack(fill="x", padx=10, pady=(8, 2))
        self.root_entry = ctk.CTkEntry(root_box)
        self.root_entry.insert(0, str(self.dataset_root))
        self.root_entry.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkButton(root_box, text="Rescan", command=self.rescan_root).pack(fill="x", padx=10, pady=(0, 10))

        self.env_combo = self._combo(controls, "Environment", self.on_env_changed)
        self.section_combo = self._combo(controls, "Section", self.on_section_changed)
        self.platform_combo = self._combo(controls, "Platform", self.on_platform_changed)
        self.view_combo = self._combo(controls, "Camera view", self.on_view_changed)

        sample_box = ctk.CTkFrame(controls)
        sample_box.pack(fill="x", padx=14, pady=(14, 0))
        ctk.CTkLabel(
            sample_box,
            text="Display sample points",
            anchor="w",
            font=ctk.CTkFont(weight="bold"),
        ).pack(fill="x", padx=10, pady=(8, 3))
        sample_row = ctk.CTkFrame(sample_box, fg_color="transparent")
        sample_row.pack(fill="x", padx=10, pady=(0, 5))
        self.sample_entry = ctk.CTkEntry(sample_row, placeholder_text="e.g. 100000")
        self.sample_entry.insert(0, str(self.max_points))
        self.sample_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(
            sample_row, text="Apply", width=62, command=self.apply_sample_count
        ).pack(side="right")
        self.sample_hint_label = ctk.CTkLabel(
            sample_box,
            text=f"Current: {self.max_points:,}  (1,000–2,000,000)",
            anchor="w",
            text_color=("gray40", "gray70"),
        )
        self.sample_hint_label.pack(fill="x", padx=10, pady=(0, 8))

        nav = ctk.CTkFrame(controls)
        nav.pack(fill="x", padx=14, pady=(16, 8))
        ctk.CTkLabel(nav, text="Scene", font=ctk.CTkFont(weight="bold")).pack(pady=(9, 4))
        self.scene_label = ctk.CTkLabel(nav, text="- / -", font=ctk.CTkFont(size=18, weight="bold"))
        self.scene_label.pack(pady=3)

        jump = ctk.CTkFrame(nav, fg_color="transparent")
        jump.pack(fill="x", padx=9, pady=5)
        self.scene_entry = ctk.CTkEntry(jump, placeholder_text="scene number")
        self.scene_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(jump, text="Go", width=48, command=self.jump_to_scene).pack(side="right")

        buttons = ctk.CTkFrame(nav, fg_color="transparent")
        buttons.pack(fill="x", padx=9, pady=(3, 10))
        self.prev_button = ctk.CTkButton(buttons, text="◀ Prev", command=self.previous_scene)
        self.prev_button.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.next_button = ctk.CTkButton(buttons, text="Next ▶", command=self.next_scene)
        self.next_button.pack(side="right", fill="x", expand=True, padx=(3, 0))

        self.status_label = ctk.CTkLabel(
            controls,
            text="Select a dataset",
            justify="left",
            anchor="nw",
            wraplength=250,
        )
        self.status_label.pack(fill="x", padx=18, pady=(10, 4))

        ctk.CTkLabel(
            controls,
            text="Keys\n← / A  previous scene\n→ / D  next scene\nT  top view   S  side view\nR  reload scene",
            justify="left",
            anchor="sw",
            text_color=("gray35", "gray70"),
        ).pack(side="bottom", fill="x", padx=18, pady=18)

        self.figure = Figure(figsize=(14, 8), dpi=100, facecolor="#202124")
        self.axes = [self.figure.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
        self.figure.subplots_adjust(left=0.015, right=0.99, bottom=0.02, top=0.94, wspace=0.03)
        self.canvas = FigureCanvasTkAgg(self.figure, master=viewer)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, viewer, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        self._draw_empty("Choose Environment / Section / Platform / Camera view")

    def _combo(self, parent: ctk.CTkFrame, label: str, command) -> ctk.CTkComboBox:
        ctk.CTkLabel(parent, text=label, anchor="w", font=ctk.CTkFont(weight="bold")).pack(
            fill="x", padx=18, pady=(8, 3)
        )
        combo = ctk.CTkComboBox(parent, values=[""], command=command, state="readonly")
        combo.pack(fill="x", padx=18)
        return combo

    def _bind_keys(self) -> None:
        for key in ("<Left>", "<Key-a>", "<Key-A>"):
            self.root.bind(key, lambda _event: self.previous_scene())
        for key in ("<Right>", "<Key-d>", "<Key-D>"):
            self.root.bind(key, lambda _event: self.next_scene())
        self.root.bind("<Key-t>", lambda _event: self.select_view("top_view_camera"))
        self.root.bind("<Key-T>", lambda _event: self.select_view("top_view_camera"))
        self.root.bind("<Key-s>", lambda _event: self.select_view("side_view_camera"))
        self.root.bind("<Key-S>", lambda _event: self.select_view("side_view_camera"))
        self.root.bind("<Key-r>", lambda _event: self.load_current_scene())
        self.root.bind("<Key-R>", lambda _event: self.load_current_scene())
        self.scene_entry.bind("<Return>", lambda _event: self.jump_to_scene())
        self.sample_entry.bind("<Return>", lambda _event: self.apply_sample_count())

    def _load_initial_selection(self) -> None:
        env, section, platform, view = DEFAULT_SELECTION
        self._refresh_envs(env)
        self._refresh_sections(section)
        self._refresh_platforms(platform)
        self._refresh_views(view)
        self.scan_scenes()

    def rescan_root(self) -> None:
        candidate = Path(self.root_entry.get().strip()).expanduser()
        if not candidate.is_dir():
            self._set_status(f"Dataset root not found:\n{candidate}", error=True)
            return
        self.dataset_root = candidate
        self._refresh_envs()
        self._refresh_sections()
        self._refresh_platforms()
        self._refresh_views()
        self.scan_scenes()

    def _refresh_envs(self, preferred: str = "") -> None:
        set_combo_values(self.env_combo, directory_names(self.dataset_root), preferred)

    def _refresh_sections(self, preferred: str = "") -> None:
        path = self.dataset_root / self.env_combo.get()
        set_combo_values(self.section_combo, directory_names(path), preferred)

    def _refresh_platforms(self, preferred: str = "") -> None:
        path = self.dataset_root / self.env_combo.get() / self.section_combo.get()
        platforms = [name for name in directory_names(path) if (path / name / "pointcloud").is_dir()]
        set_combo_values(self.platform_combo, platforms, preferred)

    def _refresh_views(self, preferred: str = "") -> None:
        base = self._platform_path() / "pointcloud"
        views = [name for name in VIEW_NAMES if (base / name).is_dir()]
        set_combo_values(self.view_combo, views, preferred)

    def on_env_changed(self, _choice: str) -> None:
        self._refresh_sections()
        self._refresh_platforms()
        self._refresh_views()
        self.scan_scenes()

    def on_section_changed(self, _choice: str) -> None:
        self._refresh_platforms()
        self._refresh_views()
        self.scan_scenes()

    def on_platform_changed(self, _choice: str) -> None:
        self._refresh_views()
        self.scan_scenes()

    def on_view_changed(self, _choice: str) -> None:
        self.scan_scenes()

    def select_view(self, view: str) -> None:
        if view in self.view_combo.cget("values"):
            self.view_combo.set(view)
            self.scan_scenes()

    def _platform_path(self) -> Path:
        return self.dataset_root / self.env_combo.get() / self.section_combo.get() / self.platform_combo.get()

    def _pointcloud_path(self) -> Path:
        return self._platform_path() / "pointcloud" / self.view_combo.get()

    @staticmethod
    def _ids_for(directory: Path, kind: str) -> set[int]:
        pattern = FILE_PATTERNS[kind]
        ids: set[int] = set()
        try:
            for path in directory.iterdir():
                match = pattern.match(path.name)
                if match:
                    ids.add(int(match.group(1)))
        except OSError:
            pass
        return ids

    def scan_scenes(self) -> None:
        directory = self._pointcloud_path()
        id_sets = {kind: self._ids_for(directory, kind) for kind in FILE_PATTERNS}
        complete = set.intersection(*id_sets.values()) if id_sets else set()
        all_ids = set.union(*id_sets.values()) if id_sets else set()
        self.scene_ids = sorted(complete)
        self.scene_index = 0

        if not directory.is_dir():
            self._draw_empty(f"Folder not found\n{directory}")
            self._set_status(f"Folder not found:\n{directory}", error=True)
            self._update_scene_controls()
            return

        missing_triplets = len(all_ids - complete)
        if not self.scene_ids:
            self._draw_empty("No complete point-cloud triplets")
            self._set_status(f"No complete triplets\nIncomplete scenes: {missing_triplets}", error=True)
            self._update_scene_controls()
            return

        suffix = f"\nIncomplete scenes: {missing_triplets}" if missing_triplets else ""
        self._set_status(f"Found {len(self.scene_ids):,} scenes{suffix}", error=bool(missing_triplets))
        self.load_current_scene()

    def previous_scene(self) -> None:
        if self.scene_ids and self.scene_index > 0:
            self.scene_index -= 1
            self.load_current_scene()

    def next_scene(self) -> None:
        if self.scene_ids and self.scene_index < len(self.scene_ids) - 1:
            self.scene_index += 1
            self.load_current_scene()

    def jump_to_scene(self) -> None:
        try:
            scene_id = int(self.scene_entry.get().strip())
            self.scene_index = self.scene_ids.index(scene_id)
        except ValueError:
            self._set_status("Scene number is not available.", error=True)
            return
        self.load_current_scene()

    def apply_sample_count(self) -> None:
        raw_value = self.sample_entry.get().strip().replace(",", "").replace("_", "")
        try:
            sample_count = int(raw_value)
        except ValueError:
            self._set_status("Sample points must be an integer.", error=True)
            return
        if not 1_000 <= sample_count <= 2_000_000:
            self._set_status("Sample points must be between 1,000 and 2,000,000.", error=True)
            return

        self.max_points = sample_count
        self.sample_entry.delete(0, "end")
        self.sample_entry.insert(0, str(sample_count))
        self.sample_hint_label.configure(text=f"Current: {sample_count:,}  (1,000–2,000,000)")
        self.load_current_scene()

    def load_current_scene(self) -> None:
        if not self.scene_ids:
            return
        self.load_generation += 1
        generation = self.load_generation
        scene_id = self.scene_ids[self.scene_index]
        directory = self._pointcloud_path()
        self._set_status(f"Loading scene {scene_id:04d} ...")
        self._update_scene_controls(loading=True)
        threading.Thread(
            target=self._load_scene_worker,
            args=(generation, directory, scene_id),
            daemon=True,
        ).start()

    def _load_scene_worker(self, generation: int, directory: Path, scene_id: int) -> None:
        try:
            paths = {
                "xyz": directory / f"pointcloud_{scene_id:04d}.npy",
                "rgb": directory / f"pointcloud_rgb_{scene_id:04d}.npy",
                "instance": directory / f"pointcloud_inst_seg_{scene_id:04d}.npy",
            }
            xyz = np.load(paths["xyz"], mmap_mode="r", allow_pickle=False)
            rgb = np.load(paths["rgb"], mmap_mode="r", allow_pickle=False)
            instance = np.load(paths["instance"], mmap_mode="r", allow_pickle=False)

            if xyz.ndim != 2 or xyz.shape[1] < 3:
                raise ValueError(f"Unexpected XYZ shape: {xyz.shape}")
            if len(rgb) != len(xyz) or len(instance) != len(xyz):
                raise ValueError(
                    f"Array length mismatch: XYZ={len(xyz):,}, RGB={len(rgb):,}, "
                    f"instance={len(instance):,}"
                )

            source_count = len(xyz)
            sample_count = min(source_count, self.max_points)
            indices = np.linspace(0, source_count - 1, sample_count, dtype=np.int64)
            points = np.asarray(xyz[indices, :3], dtype=np.float32)
            colors = np.asarray(rgb[indices, :3], dtype=np.float32)
            if colors.size and colors.max() > 1.0:
                colors /= 255.0
            ids = np.asarray(instance[indices]).reshape(-1)

            valid = np.isfinite(points).all(axis=1)
            # Isaac depth-to-pointcloud output sometimes contains an all-zero
            # sentinel for invalid depth. Do not render those points.
            valid &= np.linalg.norm(points, axis=1) > 1e-10
            points, colors, ids = points[valid], colors[valid], ids[valid]
            file_summary = "\n".join(f"✓ {path.name}" for path in paths.values())
            result = SceneData(scene_id, points, colors, ids, source_count, int(valid.sum()), file_summary)
            self.load_queue.put((generation, result, None))
        except Exception as exc:
            self.load_queue.put((generation, None, f"{type(exc).__name__}: {exc}"))

    def _poll_load_queue(self) -> None:
        try:
            while True:
                generation, data, error = self.load_queue.get_nowait()
                if generation != self.load_generation:
                    continue
                if error:
                    self._draw_empty(f"Could not load scene\n{error}")
                    self._set_status(error, error=True)
                elif data is not None:
                    self._render(data)
        except queue.Empty:
            pass
        self.root.after(60, self._poll_load_queue)

    def _render(self, data: SceneData) -> None:
        if not len(data.points):
            self._draw_empty("No valid points in this scene")
            self._set_status(f"Scene {data.scene_id:04d}: no valid points", error=True)
            self._update_scene_controls()
            return

        titles = ("XYZ / height", "RGB", "Instance segmentation")
        z = data.points[:, 2]
        instance_colors = self._instance_colors(data.instance_ids)
        color_values = (z, np.clip(data.rgb, 0.0, 1.0), instance_colors)
        cmaps = ("turbo", None, None)

        mins = data.points.min(axis=0)
        maxs = data.points.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = max(float(np.max(maxs - mins)) / 2.0, 1e-3)

        for axis, title, colors, cmap in zip(self.axes, titles, color_values, cmaps):
            axis.clear()
            axis.set_facecolor("#202124")
            axis.scatter(
                data.points[:, 0], data.points[:, 1], data.points[:, 2],
                c=colors, cmap=cmap, s=0.35, marker=".", linewidths=0,
                depthshade=False, rasterized=True,
            )
            axis.set_title(title, color="white", pad=8)
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
            axis.set_box_aspect((1, 1, 1))
            axis.set_xlabel("X", color="white", labelpad=0)
            axis.set_ylabel("Y", color="white", labelpad=0)
            axis.set_zlabel("Z", color="white", labelpad=0)
            axis.tick_params(colors="white", labelsize=7, pad=0)
            axis.grid(False)
            axis.view_init(elev=28, azim=-60)

        self.figure.suptitle(
            f"{self.env_combo.get()} / {self.section_combo.get()} / {self.platform_combo.get()}"
            f"  •  {self.view_combo.get()}  •  scene {data.scene_id:04d}",
            color="white", fontsize=13,
        )
        self.canvas.draw_idle()
        self._set_status(
            f"Scene {data.scene_id:04d}\n"
            f"Displayed: {data.valid_count:,} / {data.source_count:,} points\n"
            f"Instance IDs: {len(np.unique(data.instance_ids)):,}\n\n{data.file_summary}"
        )
        self._update_scene_controls()

    @staticmethod
    def _instance_colors(instance_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(instance_ids, dtype=np.uint64)
        # Stable integer hash gives nearby instance IDs visibly different colors.
        hashed = ids * np.uint64(11400714819323198485)
        colors = np.column_stack(
            (
                ((hashed >> np.uint64(16)) & np.uint64(255)),
                ((hashed >> np.uint64(32)) & np.uint64(255)),
                ((hashed >> np.uint64(48)) & np.uint64(255)),
            )
        ).astype(np.float32) / 255.0
        colors = 0.25 + colors * 0.75
        colors[ids == 0] = (0.05, 0.05, 0.05)
        return colors

    def _draw_empty(self, message: str) -> None:
        for axis in self.axes:
            axis.clear()
            axis.set_facecolor("#202124")
            axis.set_axis_off()
        self.axes[1].text2D(
            0.5, 0.5, message, transform=self.axes[1].transAxes,
            color="white", ha="center", va="center", fontsize=13,
        )
        self.figure.suptitle("")
        self.canvas.draw_idle()

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.configure(text=text, text_color=("#b00020", "#ff7777") if error else ("gray20", "gray85"))

    def _update_scene_controls(self, loading: bool = False) -> None:
        if self.scene_ids:
            scene_id = self.scene_ids[self.scene_index]
            self.scene_label.configure(
                text=f"{scene_id:04d}  ({self.scene_index + 1:,} / {len(self.scene_ids):,})"
            )
        else:
            self.scene_label.configure(text="- / -")
        state = "disabled" if loading else "normal"
        self.prev_button.configure(state=state if self.scene_index > 0 else "disabled")
        self.next_button.configure(
            state=state if self.scene_ids and self.scene_index < len(self.scene_ids) - 1 else "disabled"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse Dataset 2026 point-cloud scene triplets")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT, help="dataset_v2 directory")
    parser.add_argument(
        "--max-points", type=int, default=100_000,
        help="maximum displayed points per panel (default: 100000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_points < 1:
        raise SystemExit("--max-points must be at least 1")
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    PointCloudSceneViewer(root, args.root, args.max_points)
    root.mainloop()


if __name__ == "__main__":
    main()
