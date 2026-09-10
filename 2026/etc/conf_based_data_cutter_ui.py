"""Delete only data whose scene number exceeds the last conf scene.

The numeric JSON files in ``conf/`` are the authoritative scene set. The UI
shows only modality files whose scene number is greater than the maximum conf
scene number.

Missing intermediate conf numbers and unknown filenames are ignored. Only exact
files above the conf maximum can be cut.

There are no command-line arguments. Edit the variables in the section below.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import shutil
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Iterable


# =============================================================================
# User settings
# =============================================================================
DATASET_ROOTS = {
    "train": Path("/nas/Dataset/Dataset_2026/dataset_v2"),
    "validation": Path("/nas/Dataset/Dataset_2026/dataset_v2_val"),
    "test": Path("/nas/Dataset/Dataset_2026/dataset_v2_test"),
}

# True means the Cut button only opens an exact file preview. No file is moved
# or deleted. Change to False after inspecting the scan result.
DRY_RUN =False

# "quarantine" moves files to QUARANTINE_ROOT and can be restored manually.
# The UI also offers permanent deletion, which requires typed confirmation.
DEFAULT_ACTION_MODE = "quarantine"
QUARANTINE_ROOT = Path(
    "/nas/Dataset/Dataset_2026/.conf_based_data_cutter_quarantine"
)
AUDIT_LOG_PATH = Path(__file__).with_name("conf_based_data_cutter_audit.jsonl")

CAMERAS = ("top_view_camera", "side_view_camera")
WINDOW_SIZE = "1700x930"
MAX_DETAIL_ITEMS = 1_000


@dataclass(frozen=True)
class FileSpec:
    key: str
    label: str
    relative_directory: Path
    filename_pattern: re.Pattern[str]
    camera: str = "-"
    optional_directory: bool = False

    def scene_id(self, filename: str) -> int | None:
        match = self.filename_pattern.fullmatch(filename)
        return int(match.group("scene")) if match else None


def make_file_specs() -> tuple[FileSpec, ...]:
    specs = [
        FileSpec(
            "scene_meta",
            "scene_meta",
            Path("scene_meta"),
            re.compile(r"(?P<scene>\d{4,})\.json"),
        ),
        FileSpec(
            "pre_grasp",
            "pre_grasp",
            Path("pre_grasp"),
            re.compile(r"(?P<scene>\d{4,})\.json"),
            optional_directory=True,
        ),
        FileSpec(
            "output_grasp",
            "output_grasp",
            Path("output_grasp"),
            re.compile(r"(?P<scene>\d{4,})\.json"),
            optional_directory=True,
        ),
    ]
    for camera in CAMERAS:
        specs.extend(
            [
                FileSpec(
                    f"rgb/{camera}",
                    "rgb",
                    Path("rgb") / camera,
                    re.compile(r"(?P<scene>\d{4,})\.png"),
                    camera,
                ),
                FileSpec(
                    f"depth/{camera}",
                    "depth",
                    Path("depth") / camera,
                    re.compile(r"(?P<scene>\d{4,})\.npy"),
                    camera,
                ),
                FileSpec(
                    f"normals/{camera}",
                    "normals",
                    Path("normals") / camera,
                    re.compile(r"(?P<scene>\d{4,})\.png"),
                    camera,
                ),
                FileSpec(
                    f"bbox/{camera}",
                    "bbox",
                    Path("bbox") / camera,
                    re.compile(r"(?P<scene>\d{4,})\.json"),
                    camera,
                ),
                FileSpec(
                    f"inst_seg_image/{camera}",
                    "inst_seg image",
                    Path("inst_seg") / camera,
                    re.compile(r"(?P<scene>\d{4,})\.png"),
                    camera,
                ),
                FileSpec(
                    f"inst_seg_mapping/{camera}",
                    "inst_seg mapping",
                    Path("inst_seg") / camera,
                    re.compile(r"semantics_mapping_(?P<scene>\d{4,})\.json"),
                    camera,
                ),
                FileSpec(
                    f"pointcloud_xyz/{camera}",
                    "pointcloud XYZ",
                    Path("pointcloud") / camera,
                    re.compile(r"pointcloud_(?P<scene>\d{4,})\.npy"),
                    camera,
                ),
                FileSpec(
                    f"pointcloud_rgb/{camera}",
                    "pointcloud RGB",
                    Path("pointcloud") / camera,
                    re.compile(r"pointcloud_rgb_(?P<scene>\d{4,})\.npy"),
                    camera,
                ),
                FileSpec(
                    f"pointcloud_inst_seg/{camera}",
                    "pointcloud inst_seg",
                    Path("pointcloud") / camera,
                    re.compile(r"pointcloud_inst_seg_(?P<scene>\d{4,})\.npy"),
                    camera,
                ),
                FileSpec(
                    f"pointcloud_mapping/{camera}",
                    "pointcloud mapping",
                    Path("pointcloud") / camera,
                    re.compile(
                        r"pointcloud_inst_seg_mapping_(?P<scene>\d{4,})\.json"
                    ),
                    camera,
                ),
            ]
        )
    return tuple(specs)


FILE_SPECS = make_file_specs()
CONF_PATTERN = re.compile(r"(?P<scene>\d{4,})\.json")


@dataclass(frozen=True)
class PlatformRef:
    split: str
    root: Path
    env: str
    section: str
    platform: str

    @property
    def path(self) -> Path:
        return self.root / self.env / self.section / self.platform

    @property
    def relative_path(self) -> str:
        return f"{self.env}/{self.section}/{self.platform}"


@dataclass
class RowResult:
    platform: PlatformRef
    spec: FileSpec
    conf_ids: frozenset[int]
    conf_invalid_files: tuple[Path, ...]
    actual_scene_ids: frozenset[int]
    matching_file_count: int
    # Only paths above the maximum conf scene are retained.
    actual_files: dict[int, tuple[Path, ...]]
    unexpected_files: tuple[Path, ...]
    directory_exists: bool

    @property
    def actual_ids(self) -> frozenset[int]:
        return self.actual_scene_ids

    @property
    def extra_ids(self) -> list[int]:
        return sorted(self.actual_files)

    def files_for_ids(self, scene_ids: Iterable[int]) -> list[Path]:
        return [
            path
            for scene_id in scene_ids
            for path in self.actual_files.get(scene_id, ())
        ]


@dataclass
class ScanResult:
    split: str
    scope: str
    created_at_utc: str
    platforms: tuple[PlatformRef, ...]
    rows: list[RowResult]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def numeric_files(
    directory: Path, pattern: re.Pattern[str]
) -> tuple[dict[int, tuple[Path, ...]], tuple[Path, ...]]:
    matched: dict[int, list[Path]] = defaultdict(list)
    unexpected: list[Path] = []
    if not directory.is_dir():
        return {}, ()
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue
            if entry.name.startswith("."):
                continue
            match = pattern.fullmatch(entry.name)
            path = Path(entry.path)
            if match:
                matched[int(match.group("scene"))].append(path)
            else:
                unexpected.append(path)
    return (
        {key: tuple(sorted(paths)) for key, paths in matched.items()},
        tuple(sorted(unexpected)),
    )


def scan_platform(platform: PlatformRef) -> list[RowResult]:
    conf_files, conf_invalid = numeric_files(platform.path / "conf", CONF_PATTERN)
    conf_ids = frozenset(conf_files)
    conf_max = max(conf_ids) if conf_ids else None
    rows: list[RowResult] = []

    specs_by_directory: dict[Path, list[FileSpec]] = defaultdict(list)
    for spec in FILE_SPECS:
        specs_by_directory[spec.relative_directory].append(spec)

    for relative_directory, directory_specs in specs_by_directory.items():
        directory = platform.path / relative_directory
        if not directory.is_dir() and all(
            spec.optional_directory for spec in directory_specs
        ):
            continue

        matched_ids: dict[str, set[int]] = {
            spec.key: set() for spec in directory_specs
        }
        matched_counts: dict[str, int] = {spec.key: 0 for spec in directory_specs}
        excess_files: dict[str, dict[int, list[Path]]] = {
            spec.key: defaultdict(list) for spec in directory_specs
        }
        unexpected: list[Path] = []
        if directory.is_dir():
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if entry.name.startswith("."):
                        continue
                    entry_matched = False
                    for spec in directory_specs:
                        scene_id = spec.scene_id(entry.name)
                        if scene_id is not None:
                            matched_ids[spec.key].add(scene_id)
                            matched_counts[spec.key] += 1
                            if conf_max is not None and scene_id > conf_max:
                                excess_files[spec.key][scene_id].append(Path(entry.path))
                            entry_matched = True
                    if not entry_matched:
                        unexpected.append(Path(entry.path))

        for spec in directory_specs:
            actual_files = {
                scene_id: tuple(sorted(paths))
                for scene_id, paths in excess_files[spec.key].items()
            }
            rows.append(
                RowResult(
                    platform=platform,
                    spec=spec,
                    conf_ids=conf_ids,
                    conf_invalid_files=conf_invalid,
                    actual_scene_ids=frozenset(matched_ids[spec.key]),
                    matching_file_count=matched_counts[spec.key],
                    actual_files=actual_files,
                    unexpected_files=tuple(sorted(unexpected)),
                    directory_exists=directory.is_dir(),
                )
            )
    return rows


def discover_platforms(split: str) -> list[PlatformRef]:
    root = DATASET_ROOTS[split]
    if not root.is_dir():
        raise NotADirectoryError(root)
    result = []
    for conf_directory in sorted(root.glob("*/*/*/conf")):
        if not conf_directory.is_dir():
            continue
        relative = conf_directory.parent.relative_to(root)
        if len(relative.parts) != 3:
            continue
        result.append(PlatformRef(split, root, *relative.parts))
    return result


def scan_platforms(
    split: str,
    scope: str,
    platforms: list[PlatformRef],
    progress_callback=None,
) -> ScanResult:
    rows: list[RowResult] = []
    for index, platform in enumerate(platforms, start=1):
        if progress_callback:
            progress_callback(index, len(platforms), platform.relative_path)
        rows.extend(scan_platform(platform))
    return ScanResult(split, scope, utc_now(), tuple(platforms), rows)


def compact_ids(values: Iterable[int], maximum_characters: int = 90) -> str:
    numbers = sorted(set(values))
    if not numbers:
        return "-"
    groups: list[str] = []
    start = previous = numbers[0]
    for number in numbers[1:] + [None]:
        if number is not None and number == previous + 1:
            previous = number
            continue
        groups.append(
            f"{start:04d}" if start == previous else f"{start:04d}-{previous:04d}"
        )
        if number is not None:
            start = previous = number
    text = ", ".join(groups)
    if len(text) <= maximum_characters:
        return text
    return text[: maximum_characters - 3] + "..."


def row_to_dict(row: RowResult) -> dict:
    return {
        "split": row.platform.split,
        "platform": row.platform.relative_path,
        "modality": row.spec.label,
        "key": row.spec.key,
        "camera": row.spec.camera,
        "directory": str(row.platform.path / row.spec.relative_directory),
        "directory_exists": row.directory_exists,
        "conf_count": len(row.conf_ids),
        "conf_min": min(row.conf_ids) if row.conf_ids else None,
        "conf_max": max(row.conf_ids) if row.conf_ids else None,
        "matching_file_count": row.matching_file_count,
        "actual_scene_count": len(row.actual_ids),
        "scene_ids_above_conf_max": row.extra_ids,
        "files_above_conf_max": [
            str(path) for path in row.files_for_ids(row.extra_ids)
        ],
        "invalid_conf_files": [str(path) for path in row.conf_invalid_files],
    }


def scan_summary(result: ScanResult) -> dict[str, int]:
    platform_conf_counts = {}
    invalid_conf_paths = set()
    for row in result.rows:
        platform_conf_counts[row.platform.relative_path] = len(row.conf_ids)
        invalid_conf_paths.update(row.conf_invalid_files)
    return {
        "platforms": len(result.platforms),
        "conf_scenes": sum(platform_conf_counts.values()),
        "rows": len(result.rows),
        "rows_with_excess": sum(bool(row.extra_ids) for row in result.rows),
        "excess_files": sum(
            len(row.files_for_ids(row.extra_ids)) for row in result.rows
        ),
        "invalid_conf_files": len(invalid_conf_paths),
    }


class DataCutterUI:
    def __init__(self, window: tk.Tk) -> None:
        self.window = window
        self.window.title("Dataset 2026 — Conf-based Data Cutter")
        self.window.geometry(WINDOW_SIZE)
        self.window.minsize(1250, 720)

        self.scan_result: ScanResult | None = None
        self.row_by_tree_id: dict[str, RowResult] = {}
        self.last_scan_platforms: list[PlatformRef] = []
        self.last_scan_scope = ""
        self.worker: threading.Thread | None = None
        self.events: queue.Queue = queue.Queue()

        self.split_var = tk.StringVar(value=next(iter(DATASET_ROOTS)))
        self.env_var = tk.StringVar()
        self.section_var = tk.StringVar()
        self.platform_var = tk.StringVar()
        self.action_mode_var = tk.StringVar(value=DEFAULT_ACTION_MODE)
        self.status_var = tk.StringVar(value="스캔 대기 중")
        self.summary_var = tk.StringVar(value="결과 없음")

        self.build_ui()
        self.refresh_environments()
        self.window.after(100, self.poll_events)

    def build_ui(self) -> None:
        style = ttk.Style(self.window)
        style.configure(".", font=("Arial", 11))
        style.configure("TButton", font=("Arial", 11), padding=(10, 7))
        style.configure("Treeview", font=("Arial", 11), rowheight=32)
        style.configure("Treeview.Heading", font=("Arial", 11, "bold"))
        style.configure("TLabelframe.Label", font=("Arial", 11, "bold"))

        outer = ttk.Frame(self.window, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        banner_color = "#9a3412" if DRY_RUN else "#991b1b"
        banner_text = (
            "현재 미리보기 모드입니다. 정리 버튼을 눌러도 파일은 변경되지 않습니다."
            if DRY_RUN
            else "실제 정리 모드입니다. 데이터 생성 프로세스를 중단한 뒤 사용하세요."
        )
        tk.Label(
            outer,
            text=banner_text,
            background="#ffedd5" if DRY_RUN else "#fee2e2",
            foreground=banner_color,
            font=("Arial", 13, "bold"),
            anchor=tk.W,
            padx=14,
            pady=10,
        ).pack(fill=tk.X, pady=(0, 12))

        ttk.Label(
            outer,
            text="1. 데이터 선택   →   2. 스캔   →   3. 초과 행 선택 후 정리",
            font=("Arial", 12, "bold"),
        ).pack(anchor=tk.W, pady=(0, 8))

        selectors = ttk.LabelFrame(outer, text="1. 데이터 선택 및 스캔", padding=12)
        selectors.pack(fill=tk.X)
        ttk.Label(selectors, text="데이터셋").grid(row=0, column=0, sticky=tk.W)
        split_combo = ttk.Combobox(
            selectors,
            textvariable=self.split_var,
            values=list(DATASET_ROOTS),
            state="readonly",
            width=12,
        )
        split_combo.grid(row=0, column=1, padx=(6, 18))
        split_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_environments())

        ttk.Label(selectors, text="환경").grid(row=0, column=2, sticky=tk.W)
        self.env_combo = ttk.Combobox(
            selectors, textvariable=self.env_var, state="readonly", width=22
        )
        self.env_combo.grid(row=0, column=3, padx=(6, 18))
        self.env_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_sections())

        ttk.Label(selectors, text="구역").grid(row=0, column=4, sticky=tk.W)
        self.section_combo = ttk.Combobox(
            selectors, textvariable=self.section_var, state="readonly", width=28
        )
        self.section_combo.grid(row=0, column=5, padx=(6, 18))
        self.section_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_platforms())

        ttk.Label(selectors, text="플랫폼").grid(row=0, column=6, sticky=tk.W)
        self.platform_combo = ttk.Combobox(
            selectors, textvariable=self.platform_var, state="readonly", width=30
        )
        self.platform_combo.grid(row=0, column=7, padx=(6, 18), sticky=tk.EW)
        selectors.columnconfigure(7, weight=1)

        self.scan_platform_button = ttk.Button(
            selectors, text="선택한 플랫폼 스캔", command=self.scan_current_platform
        )
        self.scan_platform_button.grid(row=0, column=8, padx=4)
        self.scan_split_button = ttk.Button(
            selectors, text="이 데이터셋 전체 스캔", command=self.scan_entire_split
        )
        self.scan_split_button.grid(row=0, column=9, padx=4)

        status_frame = ttk.Frame(outer)
        status_frame.pack(fill=tk.X, pady=10)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(status_frame, mode="determinate", length=300)
        self.progress.pack(side=tk.LEFT, padx=14)
        ttk.Label(
            status_frame,
            textvariable=self.summary_var,
            font=("Arial", 12, "bold"),
        ).pack(side=tk.RIGHT)

        table_box = ttk.LabelFrame(
            outer,
            text="2. conf 마지막 번호를 초과한 파일",
            padding=8,
        )
        table_box.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            table_box,
            text="conf의 가장 큰 scene 번호보다 번호가 큰 파일만 표시됩니다.",
            foreground="#991b1b",
        ).pack(anchor=tk.W, pady=(0, 7))
        table_frame = ttk.Frame(table_box)
        table_frame.pack(fill=tk.BOTH, expand=True)
        columns = (
            "platform",
            "data_type",
            "conf",
            "scenes",
            "extra",
            "extra_ids",
        )
        self.tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", selectmode="extended"
        )
        headings = {
            "platform": "환경 / 구역 / 플랫폼",
            "data_type": "데이터 종류 / 카메라",
            "conf": "conf 마지막 번호",
            "scenes": "데이터 마지막 번호",
            "extra": "초과 파일 수",
            "extra_ids": "초과 scene 번호",
        }
        widths = {
            "platform": 390,
            "data_type": 300,
            "conf": 100,
            "scenes": 110,
            "extra": 120,
            "extra_ids": 330,
        }
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(
                column,
                width=widths[column],
                minwidth=50,
                anchor=tk.W if column in {"platform", "data_type", "extra_ids"} else tk.CENTER,
            )
        self.tree.tag_configure("excess", background="#fee2e2")
        self.tree.bind("<<TreeviewSelect>>", self.show_selected_detail)
        vertical = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        horizontal = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        selection_controls = ttk.LabelFrame(outer, text="3. 선택 및 정리", padding=10)
        selection_controls.pack(fill=tk.X, pady=(10, 7))
        ttk.Button(
            selection_controls,
            text="전부 선택",
            command=lambda: self.select_rows("all"),
        ).pack(side=tk.LEFT)
        ttk.Button(
            selection_controls,
            text="RGB만 선택",
            command=lambda: self.select_rows("rgb"),
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            selection_controls, text="선택 해제", command=lambda: self.tree.selection_set(())
        ).pack(side=tk.LEFT)
        ttk.Button(
            selection_controls, text="결과 저장", command=self.export_scan
        ).pack(side=tk.LEFT, padx=(18, 4))

        ttk.Label(selection_controls, text="정리 방법:").pack(side=tk.LEFT, padx=(28, 6))
        ttk.Radiobutton(
            selection_controls,
            text="백업 폴더로 이동 (추천)",
            variable=self.action_mode_var,
            value="quarantine",
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            selection_controls,
            text="완전히 삭제",
            variable=self.action_mode_var,
            value="delete",
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.cut_button = ttk.Button(
            selection_controls,
            text="정리할 파일 미리보기" if DRY_RUN else "선택한 초과 파일 정리",
            command=self.cut_selected,
        )
        self.cut_button.pack(side=tk.RIGHT)

        detail_frame = ttk.LabelFrame(outer, text="선택한 파일", padding=7)
        detail_frame.pack(fill=tk.X)
        self.detail = tk.Text(
            detail_frame, height=8, wrap=tk.NONE, font=("DejaVu Sans Mono", 10)
        )
        detail_scroll_y = ttk.Scrollbar(
            detail_frame, orient=tk.VERTICAL, command=self.detail.yview
        )
        detail_scroll_x = ttk.Scrollbar(
            detail_frame, orient=tk.HORIZONTAL, command=self.detail.xview
        )
        self.detail.configure(
            yscrollcommand=detail_scroll_y.set, xscrollcommand=detail_scroll_x.set
        )
        self.detail.grid(row=0, column=0, sticky="nsew")
        detail_scroll_y.grid(row=0, column=1, sticky="ns")
        detail_scroll_x.grid(row=1, column=0, sticky="ew")
        detail_frame.columnconfigure(0, weight=1)

    def selected_root(self) -> Path:
        return DATASET_ROOTS[self.split_var.get()]

    @staticmethod
    def directory_names(directory: Path) -> list[str]:
        if not directory.is_dir():
            return []
        return sorted(
            entry.name
            for entry in directory.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        )

    @staticmethod
    def set_combo(combo: ttk.Combobox, variable: tk.StringVar, values: list[str]) -> None:
        combo["values"] = values
        variable.set(values[0] if values else "")

    def refresh_environments(self) -> None:
        self.set_combo(
            self.env_combo,
            self.env_var,
            self.directory_names(self.selected_root()),
        )
        self.refresh_sections()

    def refresh_sections(self) -> None:
        self.set_combo(
            self.section_combo,
            self.section_var,
            self.directory_names(self.selected_root() / self.env_var.get()),
        )
        self.refresh_platforms()

    def refresh_platforms(self) -> None:
        self.set_combo(
            self.platform_combo,
            self.platform_var,
            self.directory_names(
                self.selected_root() / self.env_var.get() / self.section_var.get()
            ),
        )

    def current_platform_ref(self) -> PlatformRef | None:
        values = (
            self.env_var.get(),
            self.section_var.get(),
            self.platform_var.get(),
        )
        if not all(values):
            return None
        reference = PlatformRef(
            self.split_var.get(), self.selected_root(), *values
        )
        return reference if (reference.path / "conf").is_dir() else None

    def scan_current_platform(self) -> None:
        platform = self.current_platform_ref()
        if platform is None:
            messagebox.showerror("스캔 오류", "유효한 conf 폴더가 있는 platform을 선택하세요.")
            return
        self.start_scan([platform], f"platform: {platform.relative_path}")

    def scan_entire_split(self) -> None:
        try:
            platforms = discover_platforms(self.split_var.get())
        except Exception as error:
            messagebox.showerror("스캔 오류", str(error))
            return
        if not platforms:
            messagebox.showwarning("스캔", "conf를 가진 platform이 없습니다.")
            return
        self.start_scan(platforms, f"entire split: {self.split_var.get()}")

    def set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        self.scan_platform_button.configure(state=state)
        self.scan_split_button.configure(state=state)
        self.cut_button.configure(state=state)

    def start_scan(self, platforms: list[PlatformRef], scope: str) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.last_scan_platforms = list(platforms)
        self.last_scan_scope = scope
        self.set_busy(True)
        self.progress.configure(maximum=len(platforms), value=0)
        self.status_var.set("스캔 시작...")
        split = self.split_var.get()

        def worker() -> None:
            try:
                result = scan_platforms(
                    split,
                    scope,
                    platforms,
                    lambda index, total, name: self.events.put(
                        ("scan_progress", index, total, name)
                    ),
                )
                self.events.put(("scan_done", result))
            except Exception as error:
                self.events.put(("error", "스캔 실패", str(error)))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "scan_progress":
                    _, index, total, name = event
                    self.progress.configure(maximum=total, value=index)
                    self.status_var.set(f"스캔 {index}/{total}: {name}")
                elif kind == "scan_done":
                    self.scan_result = event[1]
                    self.populate_scan_result()
                    self.set_busy(False)
                elif kind == "cut_done":
                    _, message = event
                    messagebox.showinfo("정리 완료", message)
                    self.set_busy(False)
                    self.start_scan(self.last_scan_platforms, self.last_scan_scope)
                elif kind == "error":
                    _, title, message = event
                    self.set_busy(False)
                    self.status_var.set(message)
                    messagebox.showerror(title, message)
        except queue.Empty:
            pass
        self.window.after(100, self.poll_events)

    def populate_scan_result(self) -> None:
        assert self.scan_result is not None
        self.tree.delete(*self.tree.get_children())
        self.row_by_tree_id.clear()
        # Show only files whose scene number is above the maximum conf number.
        display_rows = sorted(
            (row for row in self.scan_result.rows if row.extra_ids),
            key=lambda row: (row.platform.relative_path, row.spec.key),
        )
        for row in display_rows:
            extra_files = len(row.files_for_ids(row.extra_ids))
            data_type = row.spec.label
            if row.spec.camera != "-":
                camera_name = row.spec.camera.replace("_view_camera", "")
                data_type = f"{data_type} / {camera_name}"
            item_id = self.tree.insert(
                "",
                tk.END,
                values=(
                    row.platform.relative_path,
                    data_type,
                    max(row.conf_ids) if row.conf_ids else "-",
                    max(row.actual_ids) if row.actual_ids else "-",
                    extra_files,
                    compact_ids(row.extra_ids),
                ),
                tags=("excess",),
            )
            self.row_by_tree_id[item_id] = row

        summary = scan_summary(self.scan_result)
        self.summary_var.set(
            f"플랫폼 {summary['platforms']:,}개  |  conf {summary['conf_scenes']:,}개  |  "
            f"초과 파일 {summary['excess_files']:,}개"
        )
        self.status_var.set(
            f"스캔 완료 — 초과 항목 {summary['rows_with_excess']:,}개"
        )
        self.show_scan_overview(summary)

    def show_scan_overview(self, summary: dict[str, int]) -> None:
        lines = [
            f"스캔 범위        : {self.scan_result.scope if self.scan_result else '-'}",
            f"플랫폼 수        : {summary['platforms']:,}",
            f"conf scene 수    : {summary['conf_scenes']:,}",
            f"conf 번호 초과 파일 수: {summary['excess_files']:,}",
        ]
        if summary["excess_files"]:
            lines.extend(["", "표에서 행을 선택하면 실제 파일 경로가 표시됩니다."])
        else:
            lines.extend(["", "conf 마지막 번호를 초과한 파일이 없습니다."])
        self.set_detail("\n".join(lines))

    def set_detail(self, text: str) -> None:
        self.detail.configure(state=tk.NORMAL)
        self.detail.delete("1.0", tk.END)
        self.detail.insert("1.0", text)
        self.detail.configure(state=tk.DISABLED)

    def show_selected_detail(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            if self.scan_result:
                self.show_scan_overview(scan_summary(self.scan_result))
            return
        if len(selected) > 1:
            rows = [self.row_by_tree_id[item] for item in selected]
            self.set_detail(
                f"선택한 항목 : {len(rows):,}개\n"
                f"정리할 파일 : {sum(len(row.files_for_ids(row.extra_ids)) for row in rows):,}개"
            )
            return

        row = self.row_by_tree_id[selected[0]]
        extra_paths = row.files_for_ids(row.extra_ids)
        lines = [
            f"플랫폼            : {row.platform.relative_path}",
            f"데이터 종류        : {row.spec.label}",
            f"카메라             : {row.spec.camera}",
            f"conf 마지막 번호    : {max(row.conf_ids):04d}",
            f"데이터 마지막 번호  : {max(row.actual_ids):04d}",
            f"초과 scene 번호     : {compact_ids(row.extra_ids, 500)}",
            "",
            "정리 대상 파일:",
        ]
        lines.extend(str(path) for path in extra_paths[:MAX_DETAIL_ITEMS])
        if len(extra_paths) > MAX_DETAIL_ITEMS:
            lines.append(
                f"... 나머지 {len(extra_paths) - MAX_DETAIL_ITEMS:,}개는 결과 JSON에서 확인"
            )
        self.set_detail("\n".join(lines))

    def select_rows(self, selection_type: str) -> None:
        selected = []
        for item_id, row in self.row_by_tree_id.items():
            if not row.extra_ids:
                continue
            if selection_type == "rgb" and row.spec.label != "rgb":
                continue
            selected.append(item_id)
        self.tree.selection_set(selected)
        if selected:
            self.tree.see(selected[0])

    def export_scan(self) -> None:
        if self.scan_result is None:
            messagebox.showwarning("저장", "먼저 스캔하세요.")
            return
        default_name = (
            f"conf_based_cut_scan_{self.scan_result.split}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        destination_text = filedialog.asksaveasfilename(
            title="스캔 결과 저장",
            initialdir=str(Path(__file__).parent),
            initialfile=default_name,
            defaultextension=".json",
            filetypes=(("JSON", "*.json"),),
        )
        if not destination_text:
            return
        destination = Path(destination_text)
        payload = {
            "schema_version": "conf-based-data-cutter-scan-v1",
            "created_at_utc": self.scan_result.created_at_utc,
            "split": self.scan_result.split,
            "root": str(DATASET_ROOTS[self.scan_result.split]),
            "scope": self.scan_result.scope,
            "summary": scan_summary(self.scan_result),
            "rows": [row_to_dict(row) for row in self.scan_result.rows],
        }
        atomic_write_json(destination, payload)
        messagebox.showinfo("저장 완료", str(destination))

    def selected_targets(self) -> list[tuple[RowResult, int, Path]]:
        targets = []
        seen_paths = set()
        for item_id in self.tree.selection():
            row = self.row_by_tree_id[item_id]
            for scene_id in row.extra_ids:
                for path in row.actual_files.get(scene_id, ()):
                    if path not in seen_paths:
                        seen_paths.add(path)
                        targets.append((row, scene_id, path))
        return sorted(targets, key=lambda item: str(item[2]))

    @staticmethod
    def validate_targets(targets: list[tuple[RowResult, int, Path]]) -> None:
        conf_cache: dict[Path, set[int]] = {}
        for row, scene_id, path in targets:
            platform_path = row.platform.path
            if not row.conf_ids:
                raise RuntimeError(
                    "conf anchor is empty; refusing to cut any modality file: "
                    f"{platform_path / 'conf'}"
                )
            if row.conf_invalid_files:
                raise RuntimeError(
                    "invalid conf filenames exist; resolve them before cutting:\n"
                    + "\n".join(str(item) for item in row.conf_invalid_files)
                )
            if platform_path not in conf_cache:
                conf_files, _invalid = numeric_files(platform_path / "conf", CONF_PATTERN)
                conf_cache[platform_path] = set(conf_files)
                if conf_cache[platform_path] != set(row.conf_ids):
                    raise RuntimeError(
                        "conf changed after the scan; rescan before cutting: "
                        f"{platform_path}"
                    )
            current_conf_max = max(conf_cache[platform_path])
            if scene_id <= current_conf_max:
                raise RuntimeError(
                    "conf 마지막 번호 이하의 파일은 정리하지 않습니다: "
                    f"{path}"
                )
            expected_parent = platform_path / row.spec.relative_directory
            if path.parent != expected_parent:
                raise RuntimeError(f"path escaped expected modality directory: {path}")
            if row.spec.scene_id(path.name) != scene_id:
                raise RuntimeError(f"filename no longer matches the scanned target: {path}")
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"target is missing or is a symlink: {path}")

    def show_target_preview(
        self, targets: list[tuple[RowResult, int, Path]], title: str
    ) -> None:
        preview = tk.Toplevel(self.window)
        preview.title(title)
        preview.geometry("1250x720")
        text_widget = tk.Text(preview, wrap=tk.NONE, font=("DejaVu Sans Mono", 9))
        scroll_y = ttk.Scrollbar(preview, orient=tk.VERTICAL, command=text_widget.yview)
        scroll_x = ttk.Scrollbar(preview, orient=tk.HORIZONTAL, command=text_widget.xview)
        text_widget.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        text_widget.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)
        lines = [
            title,
            f"files: {len(targets):,}",
            f"action mode: {self.action_mode_var.get()}",
            "",
        ]
        lines.extend(str(path) for _row, _scene_id, path in targets)
        text_widget.insert("1.0", "\n".join(lines))
        text_widget.configure(state=tk.DISABLED)

    def cut_selected(self) -> None:
        targets = self.selected_targets()
        if not targets:
            messagebox.showwarning(
                "정리 대상 없음",
                "표에서 정리할 행을 먼저 선택하세요.",
            )
            return
        try:
            self.validate_targets(targets)
        except Exception as error:
            messagebox.showerror("안전 검사 실패", str(error))
            return

        if DRY_RUN:
            self.show_target_preview(targets, "DRY RUN — 실제 변경 없음")
            return

        mode = self.action_mode_var.get()
        total_bytes = 0
        for _row, _scene_id, path in targets:
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
        action_text = "격리 폴더로 이동" if mode == "quarantine" else "영구 삭제"
        if not messagebox.askyesno(
            "초과 데이터 정리 확인",
            f"{len(targets):,}개 파일 ({total_bytes / 1024**3:.3f} GiB)을 "
            f"{action_text}합니다.\n\n"
                "conf 마지막 번호를 초과한 파일만 처리됩니다. 계속하시겠습니까?",
        ):
            return
        if mode == "delete":
            required = f"DELETE {len(targets)}"
            entered = simpledialog.askstring(
                "영구 삭제 재확인",
                f"복구할 수 없습니다. 아래 문구를 입력하세요.\n\n{required}",
                parent=self.window,
            )
            if entered != required:
                messagebox.showwarning("취소", "확인 문구가 일치하지 않아 취소했습니다.")
                return

        self.set_busy(True)
        self.status_var.set(f"{len(targets):,}개 파일 처리 중...")

        def worker() -> None:
            try:
                message = execute_targets(targets, mode)
                self.events.put(("cut_done", message))
            except Exception as error:
                self.events.put(("error", "정리 실패", str(error)))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def append_audit_record(record: dict) -> None:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def execute_targets(
    targets: list[tuple[RowResult, int, Path]], mode: str
) -> str:
    if mode not in {"quarantine", "delete"}:
        raise ValueError(f"unknown action mode: {mode}")
    DataCutterUI.validate_targets(targets)
    operation_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    moved: list[tuple[Path, Path]] = []
    deleted: list[Path] = []
    quarantine_session = QUARANTINE_ROOT / operation_id

    if mode == "quarantine":
        destinations = []
        for row, _scene_id, source in targets:
            relative = source.relative_to(row.platform.root)
            destination = quarantine_session / row.platform.split / relative
            if destination.exists():
                raise FileExistsError(f"quarantine destination already exists: {destination}")
            destinations.append((source, destination))
        try:
            for source, destination in destinations:
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    source.rename(destination)
                except OSError:
                    # Normally source and quarantine are on the same NAS. This
                    # fallback also supports a separately configured filesystem.
                    shutil.move(str(source), str(destination))
                moved.append((source, destination))
        except Exception:
            rollback_errors = []
            for source, destination in reversed(moved):
                try:
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.rename(source)
                except Exception as error:
                    rollback_errors.append(f"{destination} -> {source}: {error}")
            if rollback_errors:
                raise RuntimeError(
                    "quarantine failed and rollback was incomplete:\n"
                    + "\n".join(rollback_errors)
                )
            raise
    else:
        for _row, _scene_id, path in targets:
            path.unlink()
            deleted.append(path)

    record = {
        "schema_version": "conf-based-data-cutter-audit-v1",
        "operation_id": operation_id,
        "created_at_utc": utc_now(),
        "mode": mode,
        "file_count": len(targets),
        "quarantine_session": str(quarantine_session) if mode == "quarantine" else None,
        "files": [
            {
                "split": row.platform.split,
                "platform": row.platform.relative_path,
                "modality": row.spec.key,
                "scene_id": scene_id,
                "source": str(path),
                "destination": (
                    str(
                        quarantine_session
                        / row.platform.split
                        / path.relative_to(row.platform.root)
                    )
                    if mode == "quarantine"
                    else None
                ),
            }
            for row, scene_id, path in targets
        ],
    }
    append_audit_record(record)
    if mode == "quarantine":
        return (
            f"{len(moved):,}개 파일을 격리했습니다.\n\n"
            f"복구 위치:\n{quarantine_session}\n\n"
            f"감사 로그:\n{AUDIT_LOG_PATH}"
        )
    return (
        f"{len(deleted):,}개 파일을 영구 삭제했습니다.\n\n"
        f"감사 로그:\n{AUDIT_LOG_PATH}"
    )


def main() -> None:
    window = tk.Tk()
    DataCutterUI(window)
    window.mainloop()


if __name__ == "__main__":
    main()
