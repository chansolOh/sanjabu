"""Human instance-segmentation semantic-accuracy review GUI.

The reviewer judges one deterministic manifest item at a time. Every decision
is atomically saved with an audit history; dataset files are read-only.
"""

from __future__ import annotations

import getpass
import json
import tkinter as tk
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageTk

import config
from evaluation_core import (
    atomic_write_json,
    file_sha256,
    parse_rgba_key,
    strict_json_load,
    wilson_interval,
    write_csv,
)
from make_inst_seg_samples import create_manifest


DECISIONS = ("correct", "incorrect", "skip")
ISSUE_TYPES = (
    "",
    "wrong_class",
    "missing_object_mask",
    "extra_object_mask",
    "boundary_error",
    "color_mapping_error",
    "identity_ambiguous",
    "unreadable_or_missing_file",
    "other",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def blank_store(manifest_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "manual-inst-seg-review-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "reviewer": config.REVIEWER_NAME.strip() or getpass.getuser(),
        "created_at_utc": now_utc(),
        "updated_at_utc": now_utc(),
        "records": {},
        "history": [],
    }


def summarize(manifest: dict[str, Any], store: dict[str, Any]) -> dict[str, Any]:
    items = manifest["items"]
    records = store.get("records", {})

    def group_summary(group_items: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(
            records.get(item["uid"], {}).get("decision", "pending")
            for item in group_items
        )
        decided = counts["correct"] + counts["incorrect"]
        lower, upper = wilson_interval(counts["correct"], decided)
        return {
            "total": len(group_items),
            "correct": counts["correct"],
            "incorrect": counts["incorrect"],
            "skip": counts["skip"],
            "pending": counts["pending"],
            "decided": decided,
            "accuracy": counts["correct"] / decided if decided else None,
            "wilson_95_lower": lower if decided else None,
            "wilson_95_upper": upper if decided else None,
        }

    by_split: dict[str, Any] = {}
    by_camera: dict[str, Any] = {}
    by_split_camera: dict[str, Any] = {}
    for split in config.DATASETS:
        split_items = [item for item in items if item["split"] == split]
        by_split[split] = group_summary(split_items)
    for camera in config.MANUAL_CAMERAS:
        camera_items = [item for item in items if item["camera"] == camera]
        by_camera[camera] = group_summary(camera_items)
    for split in config.DATASETS:
        for camera in config.MANUAL_CAMERAS:
            key = f"{split}/{camera}"
            group_items = [
                item
                for item in items
                if item["split"] == split and item["camera"] == camera
            ]
            by_split_camera[key] = group_summary(group_items)

    issue_counts = Counter(
        record.get("issue_type") or "unspecified"
        for record in records.values()
        if record.get("decision") == "incorrect"
    )
    return {
        "schema_version": "manual-inst-seg-summary-v1",
        "updated_at_utc": now_utc(),
        "reviewer": store.get("reviewer"),
        "manifest": store.get("manifest"),
        "manifest_sha256": store.get("manifest_sha256"),
        "overall": group_summary(items),
        "by_split": by_split,
        "by_camera": by_camera,
        "by_split_camera": by_split_camera,
        "incorrect_issue_counts": dict(sorted(issue_counts.items())),
    }


class InstanceSegmentationReviewer:
    def __init__(self) -> None:
        self.manifest = create_manifest()
        self.items: list[dict[str, Any]] = self.manifest["items"]
        if not self.items:
            raise RuntimeError("manual review manifest contains no items")
        if config.MANUAL_RECORD_PATH.is_file():
            self.store = strict_json_load(config.MANUAL_RECORD_PATH)
            current_hash = file_sha256(config.MANUAL_MANIFEST_PATH)
            if self.store.get("manifest_sha256") != current_hash:
                raise RuntimeError(
                    "manual record belongs to a different manifest; move the old "
                    "record or restore the matching manifest"
                )
        else:
            self.store = blank_store(config.MANUAL_MANIFEST_PATH)

        self.index = self.first_pending_index()
        self.photo_images: list[ImageTk.PhotoImage] = []
        self.current_load_error = ""

        self.window = tk.Tk()
        self.window.title("Dataset 2026 — Manual Instance Segmentation Review")
        self.window.geometry(config.REVIEW_WINDOW_SIZE)
        self.window.minsize(1280, 760)
        self.build_ui()
        self.bind_keys()
        self.load_current_item()
        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def first_pending_index(self) -> int:
        records = self.store.get("records", {})
        for index, item in enumerate(self.items):
            if item["uid"] not in records:
                return index
        return 0

    def build_ui(self) -> None:
        outer = ttk.Frame(self.window, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X)
        self.identity_label = ttk.Label(header, font=("Arial", 12, "bold"))
        self.identity_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress_label = ttk.Label(header, font=("Arial", 12, "bold"))
        self.progress_label.pack(side=tk.RIGHT)

        panels = ttk.Frame(outer)
        panels.pack(fill=tk.BOTH, expand=True, pady=6)
        self.panel_labels: list[ttk.Label] = []
        for column, title in enumerate(("RGB", "RGB + Instance Overlay", "Raw Instance Segmentation")):
            panel = ttk.LabelFrame(panels, text=title, padding=4)
            panel.grid(row=0, column=column, sticky="nsew", padx=3)
            label = ttk.Label(panel, anchor=tk.CENTER)
            label.pack(fill=tk.BOTH, expand=True)
            self.panel_labels.append(label)
            panels.columnconfigure(column, weight=1)
        panels.rowconfigure(0, weight=1)

        details = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        details.pack(fill=tk.X, pady=4)
        legend_frame = ttk.LabelFrame(details, text="Class / pixel evidence", padding=4)
        note_frame = ttk.LabelFrame(details, text="Review note", padding=4)
        details.add(legend_frame, weight=3)
        details.add(note_frame, weight=2)
        self.legend_text = tk.Text(legend_frame, height=8, wrap=tk.NONE, font=("DejaVu Sans Mono", 10))
        self.legend_text.pack(fill=tk.BOTH, expand=True)
        self.note_text = tk.Text(note_frame, height=8, wrap=tk.WORD)
        self.note_text.pack(fill=tk.BOTH, expand=True)

        options = ttk.Frame(outer)
        options.pack(fill=tk.X, pady=4)
        ttk.Label(options, text="Overlay alpha").pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=config.DEFAULT_OVERLAY_ALPHA)
        self.alpha_scale = ttk.Scale(
            options,
            from_=0.0,
            to=0.9,
            variable=self.alpha_var,
            command=lambda _value: self.render_current_item(),
        )
        self.alpha_scale.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        self.show_bbox_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options,
            text="Show bbox",
            variable=self.show_bbox_var,
            command=self.render_current_item,
        ).pack(side=tk.LEFT, padx=10)
        ttk.Label(options, text="Incorrect type").pack(side=tk.LEFT)
        self.issue_var = tk.StringVar(value="")
        self.issue_combo = ttk.Combobox(
            options,
            textvariable=self.issue_var,
            values=ISSUE_TYPES,
            state="readonly",
            width=28,
        )
        self.issue_combo.pack(side=tk.LEFT, padx=6)

        actions = ttk.Frame(outer)
        actions.pack(fill=tk.X, pady=6)
        ttk.Button(actions, text="◀ Previous", command=self.previous).pack(side=tk.LEFT)
        ttk.Button(actions, text="Next ▶", command=self.next).pack(side=tk.LEFT, padx=4)
        ttk.Label(actions, text="Go to item").pack(side=tk.LEFT, padx=(14, 3))
        self.goto_var = tk.StringVar()
        goto_entry = ttk.Entry(actions, textvariable=self.goto_var, width=8)
        goto_entry.pack(side=tk.LEFT)
        goto_entry.bind("<Return>", lambda _event: self.go_to_item())
        ttk.Button(actions, text="Go", command=self.go_to_item).pack(side=tk.LEFT, padx=3)

        ttk.Button(
            actions, text="1  Correct", command=lambda: self.save_decision("correct")
        ).pack(side=tk.RIGHT, padx=4)
        ttk.Button(
            actions, text="2  Incorrect", command=lambda: self.save_decision("incorrect")
        ).pack(side=tk.RIGHT, padx=4)
        ttk.Button(
            actions, text="3  Skip", command=lambda: self.save_decision("skip")
        ).pack(side=tk.RIGHT, padx=4)
        ttk.Button(actions, text="Clear decision", command=self.clear_decision).pack(side=tk.RIGHT, padx=12)

        self.status_label = ttk.Label(
            outer,
            text=(
                "Criteria: every visible object has the correct class mask; no missing, "
                "extra, swapped, or materially incorrect boundary. Keys: 1/O, 2/X, 3/S, arrows."
            ),
        )
        self.status_label.pack(fill=tk.X)

    def bind_keys(self) -> None:
        self.window.bind("<Left>", lambda event: self.handle_navigation_key(event, -1))
        self.window.bind("<Right>", lambda event: self.handle_navigation_key(event, 1))
        self.window.bind("1", lambda event: self.handle_decision_key(event, "correct"))
        self.window.bind("o", lambda event: self.handle_decision_key(event, "correct"))
        self.window.bind("2", lambda event: self.handle_decision_key(event, "incorrect"))
        self.window.bind("x", lambda event: self.handle_decision_key(event, "incorrect"))
        self.window.bind("3", lambda event: self.handle_decision_key(event, "skip"))
        self.window.bind("s", lambda event: self.handle_decision_key(event, "skip"))

    @staticmethod
    def is_text_input(widget: Any) -> bool:
        return isinstance(widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Combobox))

    def handle_navigation_key(self, event: Any, direction: int) -> str | None:
        if self.is_text_input(event.widget):
            return None
        self.previous() if direction < 0 else self.next()
        return "break"

    def handle_decision_key(self, event: Any, decision: str) -> str | None:
        if self.is_text_input(event.widget):
            return None
        self.save_decision(decision)
        return "break"

    @property
    def current_item(self) -> dict[str, Any]:
        return self.items[self.index]

    def load_current_item(self) -> None:
        item = self.current_item
        self.goto_var.set(str(self.index + 1))
        self.identity_label.config(
            text=(
                f"{item['split']} | {item['relative_scene']} | {item['camera']}"
            )
        )
        record = self.store.get("records", {}).get(item["uid"], {})
        self.issue_var.set(record.get("issue_type", ""))
        self.note_text.delete("1.0", tk.END)
        self.note_text.insert("1.0", record.get("note", ""))
        self.render_current_item()
        self.update_progress()

    def render_current_item(self) -> None:
        item = self.current_item
        paths = {key: Path(value) for key, value in item["paths"].items()}
        try:
            rgb = Image.open(paths["rgb"]).convert("RGB")
            seg = Image.open(paths["inst_seg"]).convert("RGBA")
            mapping = strict_json_load(paths["mapping"])
            conf = strict_json_load(paths["conf"])
            bbox = strict_json_load(paths["bbox"])
            self.current_load_error = ""
        except Exception as error:
            self.current_load_error = str(error)
            placeholder = Image.new("RGB", config.REVIEW_PANEL_SIZE, "#202020")
            draw = ImageDraw.Draw(placeholder)
            draw.multiline_text((20, 20), f"LOAD ERROR\n{error}", fill="red")
            self.set_panel_images([placeholder, placeholder, placeholder])
            self.legend_text.delete("1.0", tk.END)
            self.legend_text.insert("1.0", f"LOAD ERROR\n{error}")
            return

        seg_array = np.asarray(seg, dtype=np.uint8)
        class_rows: list[tuple[str, tuple[int, int, int, int], int, list[int] | None]] = []
        for rgba_key, label in mapping.items():
            rgba = parse_rgba_key(rgba_key)
            class_name = label.get("class") if isinstance(label, dict) else None
            if rgba is None or not isinstance(class_name, str):
                continue
            mask = np.all(seg_array == np.asarray(rgba, dtype=np.uint8), axis=2)
            ys, xs = np.where(mask)
            bounds = (
                [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                if xs.size
                else None
            )
            class_rows.append((class_name, rgba, int(mask.sum()), bounds))

        seg_rgb = seg.convert("RGB")
        overlay = Image.blend(rgb, seg_rgb, float(self.alpha_var.get()))
        if self.show_bbox_var.get():
            draw_rgb = ImageDraw.Draw(rgb)
            draw_overlay = ImageDraw.Draw(overlay)
            for class_name, rgba, _pixel_count, bounds in class_rows:
                if class_name in {"BACKGROUND", "UNLABELLED"}:
                    continue
                stored_bbox = bbox.get(class_name) if isinstance(bbox, dict) else None
                color = tuple(rgba[:3])
                if isinstance(stored_bbox, list) and len(stored_bbox) == 4:
                    draw_rgb.rectangle(stored_bbox, outline=color, width=3)
                    draw_overlay.rectangle(stored_bbox, outline=color, width=3)
                    draw_overlay.text(
                        (stored_bbox[0] + 4, stored_bbox[1] + 4),
                        class_name,
                        fill=color,
                        stroke_width=2,
                        stroke_fill="black",
                    )
                elif bounds:
                    draw_overlay.text(
                        (bounds[0], bounds[1]),
                        f"{class_name} (no bbox)",
                        fill=color,
                        stroke_width=2,
                        stroke_fill="black",
                    )

        self.set_panel_images([rgb, overlay, seg_rgb])
        conf_classes = sorted(
            item.get("class")
            for item in conf.get("objects", [])
            if isinstance(item, dict) and isinstance(item.get("class"), str)
        )
        mapped_classes = sorted(
            class_name
            for class_name, _rgba, _count, _bounds in class_rows
            if class_name not in {"BACKGROUND", "UNLABELLED"}
        )
        lines = [
            f"conf classes   : {conf_classes}",
            f"mapping classes: {mapped_classes}",
            f"class-set match: {conf_classes == mapped_classes}",
            "",
            "class          RGBA                    pixels       mask bbox            stored bbox",
        ]
        for class_name, rgba, pixel_count, bounds in class_rows:
            stored = bbox.get(class_name) if isinstance(bbox, dict) else None
            lines.append(
                f"{class_name:<14} {str(rgba):<23} {pixel_count:>10,}  "
                f"{str(bounds):<20} {stored}"
            )
        self.legend_text.delete("1.0", tk.END)
        self.legend_text.insert("1.0", "\n".join(lines))

    def set_panel_images(self, images: list[Image.Image]) -> None:
        self.photo_images = []
        target_width, target_height = config.REVIEW_PANEL_SIZE
        for label, image in zip(self.panel_labels, images):
            rendered = image.copy()
            rendered.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(rendered)
            label.configure(image=photo)
            self.photo_images.append(photo)

    def update_progress(self) -> None:
        summary = summarize(self.manifest, self.store)
        overall = summary["overall"]
        accuracy = overall["accuracy"]
        accuracy_text = "N/A" if accuracy is None else f"{accuracy * 100:.3f}%"
        decision = self.store.get("records", {}).get(
            self.current_item["uid"], {}
        ).get("decision", "pending")
        self.progress_label.config(
            text=(
                f"Item {self.index + 1:,}/{len(self.items):,} | current={decision} | "
                f"O={overall['correct']:,} X={overall['incorrect']:,} "
                f"S={overall['skip']:,} pending={overall['pending']:,} | acc={accuracy_text}"
            )
        )

    def persist(self) -> None:
        self.store["updated_at_utc"] = now_utc()
        atomic_write_json(config.MANUAL_RECORD_PATH, self.store)
        summary = summarize(self.manifest, self.store)
        atomic_write_json(config.RESULT_DIR / "manual_inst_seg_summary.json", summary)
        rows = []
        for item in self.items:
            record = self.store.get("records", {}).get(item["uid"], {})
            rows.append(
                {
                    "uid": item["uid"],
                    "split": item["split"],
                    "env": item["env"],
                    "section": item["section"],
                    "platform": item["platform"],
                    "scene_name": item["scene_name"],
                    "camera": item["camera"],
                    "decision": record.get("decision", "pending"),
                    "issue_type": record.get("issue_type", ""),
                    "note": record.get("note", ""),
                    "reviewer": record.get("reviewer", ""),
                    "updated_at_utc": record.get("updated_at_utc", ""),
                }
            )
        write_csv(config.RESULT_DIR / "manual_inst_seg_records.csv", rows)

    def save_decision(self, decision: str) -> None:
        if decision not in DECISIONS:
            raise ValueError(decision)
        issue_type = self.issue_var.get().strip()
        if decision == "incorrect" and not issue_type:
            issue_type = "unspecified"
        note = self.note_text.get("1.0", tk.END).strip()
        uid = self.current_item["uid"]
        record = {
            "decision": decision,
            "issue_type": issue_type if decision == "incorrect" else "",
            "note": note,
            "reviewer": self.store["reviewer"],
            "updated_at_utc": now_utc(),
            "load_error_at_review": self.current_load_error,
        }
        previous = self.store.setdefault("records", {}).get(uid)
        self.store["records"][uid] = record
        self.store.setdefault("history", []).append(
            {
                "uid": uid,
                "previous_decision": previous.get("decision") if previous else None,
                **record,
            }
        )
        self.persist()
        self.update_progress()
        if self.index < len(self.items) - 1:
            self.index += 1
            self.load_current_item()

    def clear_decision(self) -> None:
        uid = self.current_item["uid"]
        previous = self.store.setdefault("records", {}).pop(uid, None)
        if previous is not None:
            self.store.setdefault("history", []).append(
                {
                    "uid": uid,
                    "previous_decision": previous.get("decision"),
                    "decision": "cleared",
                    "reviewer": self.store["reviewer"],
                    "updated_at_utc": now_utc(),
                }
            )
            self.persist()
            self.load_current_item()

    def previous(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.load_current_item()

    def next(self) -> None:
        if self.index < len(self.items) - 1:
            self.index += 1
            self.load_current_item()

    def go_to_item(self) -> None:
        try:
            target = int(self.goto_var.get()) - 1
        except ValueError:
            messagebox.showwarning("Invalid item", "Enter a numeric item position.")
            return
        if not 0 <= target < len(self.items):
            messagebox.showwarning(
                "Invalid item", f"Valid range: 1 through {len(self.items):,}."
            )
            return
        self.index = target
        self.load_current_item()

    def close(self) -> None:
        self.persist()
        self.window.destroy()

    def run(self) -> None:
        self.window.mainloop()


def main() -> None:
    InstanceSegmentationReviewer().run()


if __name__ == "__main__":
    main()
