"""Shared discovery, sampling, serialization, and statistics utilities."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import random
import socket
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import config


@dataclass(frozen=True, order=True)
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


@dataclass(frozen=True, order=True)
class SceneRef:
    split: str
    root: Path
    env: str
    section: str
    platform: str
    scene_id: int

    @property
    def platform_path(self) -> Path:
        return self.root / self.env / self.section / self.platform

    @property
    def scene_name(self) -> str:
        return f"{self.scene_id:04d}"

    @property
    def relative_scene(self) -> str:
        return (
            f"{self.env}/{self.section}/{self.platform}/{self.scene_name}"
        )

    @property
    def uid(self) -> str:
        return f"{self.split}|{self.relative_scene}"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["root"] = str(self.root)
        result["scene_name"] = self.scene_name
        result["relative_scene"] = self.relative_scene
        result["uid"] = self.uid
        return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant)


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_metadata(script_name: str) -> dict[str, Any]:
    datasets = {
        split: {
            "root": str(settings["root"]),
            "require_pre_grasp": settings["require_pre_grasp"],
            "require_output_grasp": settings["require_output_grasp"],
        }
        for split, settings in config.DATASETS.items()
    }
    return {
        "schema_version": "2026-certification-eval-v1",
        "script": script_name,
        "created_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "datasets": datasets,
        "object_csv": str(config.OBJECT_CSV),
        "object_csv_sha256": (
            file_sha256(config.OBJECT_CSV)
            if config.OBJECT_CSV.is_file()
            else None
        ),
        "random_seed": config.RANDOM_SEED,
    }


@lru_cache(maxsize=None)
def discover_platforms(split: str, root: Path) -> list[PlatformRef]:
    if not root.is_dir():
        raise NotADirectoryError(f"dataset root does not exist: {root}")
    result: list[PlatformRef] = []
    for env_path in sorted(path for path in root.iterdir() if path.is_dir()):
        for section_path in sorted(
            path for path in env_path.iterdir() if path.is_dir()
        ):
            for platform_path in sorted(
                path for path in section_path.iterdir() if path.is_dir()
            ):
                if not (platform_path / "conf").is_dir():
                    continue
                result.append(
                    PlatformRef(
                        split=split,
                        root=root,
                        env=env_path.name,
                        section=section_path.name,
                        platform=platform_path.name,
                    )
                )
    return result


def numeric_file_ids(directory: Path, prefix: str = "", suffix: str = "") -> set[int]:
    if not directory.is_dir():
        return set()
    result: set[int] = set()
    for entry in os.scandir(directory):
        if not entry.is_file():
            continue
        name = entry.name
        if prefix and not name.startswith(prefix):
            continue
        if suffix and not name.endswith(suffix):
            continue
        start = len(prefix)
        end = len(name) - len(suffix) if suffix else len(name)
        number = name[start:end]
        if number.isdigit() and len(number) >= 4:
            result.add(int(number))
    return result


def platform_scene_ids(platform: PlatformRef) -> set[int]:
    return numeric_file_ids(platform.path / "conf", suffix=".json")


@lru_cache(maxsize=None)
def scene_refs_for_split(split: str) -> list[SceneRef]:
    root = Path(config.DATASETS[split]["root"])
    refs: list[SceneRef] = []
    for platform in discover_platforms(split, root):
        refs.extend(
            SceneRef(
                split=split,
                root=root,
                env=platform.env,
                section=platform.section,
                platform=platform.platform,
                scene_id=scene_id,
            )
            for scene_id in sorted(platform_scene_ids(platform))
        )
    return refs


def stratified_sample(
    items: Sequence[Any],
    sample_count: int | None,
    stratum_key: Callable[[Any], Any],
    seed: int,
) -> list[Any]:
    """Deterministically sample while distributing items across strata."""
    if sample_count is None or sample_count >= len(items):
        return list(items)
    if sample_count <= 0:
        return []

    grouped: dict[Any, list[Any]] = defaultdict(list)
    for item in items:
        grouped[stratum_key(item)].append(item)

    rng = random.Random(seed)
    keys = sorted(grouped, key=str)
    for key in keys:
        rng.shuffle(grouped[key])

    selected: list[Any] = []
    cursor = {key: 0 for key in keys}
    while len(selected) < sample_count:
        progressed = False
        for key in keys:
            index = cursor[key]
            if index < len(grouped[key]):
                selected.append(grouped[key][index])
                cursor[key] += 1
                progressed = True
                if len(selected) == sample_count:
                    break
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def load_object_catalog() -> dict[str, dict[str, str]]:
    required_columns = {
        "Object_name",
        "Level_1",
        "Level_2",
        "Level_3",
        "Class_name",
        "Color",
        "Packaging",
        "Features",
        "Description",
    }
    with config.OBJECT_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"object CSV missing columns: {sorted(missing)}")
        result: dict[str, dict[str, str]] = {}
        for row_number, row in enumerate(reader, start=2):
            object_id = (row.get("Class_name") or "").strip()
            if not object_id:
                raise ValueError(f"empty Class_name at CSV row {row_number}")
            if object_id in result:
                raise ValueError(f"duplicate Class_name in CSV: {object_id}")
            result[object_id] = {
                key: (row.get(key) or "").strip() for key in required_columns
            }
    return result


def load_allowed_grippers() -> set[str]:
    result: set[str] = set()
    for path in (config.FINGER_GRIPPER_CATALOG, config.HAND_GRIPPER_CATALOG):
        if not path.is_file():
            continue
        data = strict_json_load(path)
        if isinstance(data, dict):
            result.update(str(key) for key in data)
    return result


def parse_rgba_key(value: str) -> tuple[int, int, int, int] | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, tuple) or len(parsed) != 4:
        return None
    if not all(isinstance(number, int) and 0 <= number <= 255 for number in parsed):
        return None
    return parsed


def normalized_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


def bounded_examples(values: Iterable[Any]) -> tuple[int, list[Any]]:
    examples: list[Any] = []
    count = 0
    for value in values:
        count += 1
        if len(examples) < config.MAX_ISSUE_EXAMPLES:
            examples.append(value)
    return count, examples


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
