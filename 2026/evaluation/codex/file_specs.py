"""Dataset file inventory specification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import config


@dataclass(frozen=True)
class FileSpec:
    key: str
    relative_dir: Path
    pattern: re.Pattern[str]
    required_in: frozenset[str]

    def scene_id(self, filename: str) -> int | None:
        match = self.pattern.fullmatch(filename)
        return int(match.group("scene")) if match else None


ALL_SPLITS = frozenset(config.DATASETS)
TRAIN_ONLY = frozenset({"train"})


def build_file_specs() -> list[FileSpec]:
    specs = [
        FileSpec("conf", Path("conf"), re.compile(r"(?P<scene>\d{4,})\.json"), ALL_SPLITS),
        FileSpec(
            "scene_meta",
            Path("scene_meta"),
            re.compile(r"(?P<scene>\d{4,})\.json"),
            ALL_SPLITS,
        ),
        FileSpec(
            "pre_grasp",
            Path("pre_grasp"),
            re.compile(r"(?P<scene>\d{4,})\.json"),
            TRAIN_ONLY,
        ),
        FileSpec(
            "output_grasp",
            Path("output_grasp"),
            re.compile(r"(?P<scene>\d{4,})\.json"),
            TRAIN_ONLY,
        ),
    ]
    for camera in config.CAMERAS:
        camera_suffix = f"/{camera}"
        specs.extend(
            [
                FileSpec(
                    f"rgb{camera_suffix}",
                    Path("rgb") / camera,
                    re.compile(r"(?P<scene>\d{4,})\.png"),
                    ALL_SPLITS,
                ),
                FileSpec(
                    f"depth{camera_suffix}",
                    Path("depth") / camera,
                    re.compile(r"(?P<scene>\d{4,})\.npy"),
                    ALL_SPLITS,
                ),
                FileSpec(
                    f"normals{camera_suffix}",
                    Path("normals") / camera,
                    re.compile(r"(?P<scene>\d{4,})\.png"),
                    ALL_SPLITS,
                ),
                FileSpec(
                    f"bbox{camera_suffix}",
                    Path("bbox") / camera,
                    re.compile(r"(?P<scene>\d{4,})\.json"),
                    ALL_SPLITS,
                ),
                FileSpec(
                    f"inst_seg_image{camera_suffix}",
                    Path("inst_seg") / camera,
                    re.compile(r"(?P<scene>\d{4,})\.png"),
                    ALL_SPLITS,
                ),
                FileSpec(
                    f"inst_seg_mapping{camera_suffix}",
                    Path("inst_seg") / camera,
                    re.compile(r"semantics_mapping_(?P<scene>\d{4,})\.json"),
                    ALL_SPLITS,
                ),
                FileSpec(
                    f"pointcloud_xyz{camera_suffix}",
                    Path("pointcloud") / camera,
                    re.compile(r"pointcloud_(?P<scene>\d{4,})\.npy"),
                    ALL_SPLITS,
                ),
                FileSpec(
                    f"pointcloud_rgb{camera_suffix}",
                    Path("pointcloud") / camera,
                    re.compile(r"pointcloud_rgb_(?P<scene>\d{4,})\.npy"),
                    ALL_SPLITS,
                ),
                FileSpec(
                    f"pointcloud_inst_seg{camera_suffix}",
                    Path("pointcloud") / camera,
                    re.compile(r"pointcloud_inst_seg_(?P<scene>\d{4,})\.npy"),
                    ALL_SPLITS,
                ),
                FileSpec(
                    f"pointcloud_inst_seg_mapping{camera_suffix}",
                    Path("pointcloud") / camera,
                    re.compile(
                        r"pointcloud_inst_seg_mapping_(?P<scene>\d{4,})\.json"
                    ),
                    ALL_SPLITS,
                ),
            ]
        )
    return specs


FILE_SPECS = build_file_specs()

