import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


ROOT_PATH = Path("/nas/Dataset/Dataset_2026/dataset_v2")
RESULT_ROOT = Path(__file__).resolve().parent / "conf_mismatch_results"
EXPECTED_OBJECT_COUNT = 5
IGNORED_CLASSES = {
    "background",
    "unlabel",
    "unlabeled",
    "unlabelled",
}


def normalize_class_name(value):
    if value is None:
        return None
    name = str(value).strip().lower()
    return name if name and name not in IGNORED_CLASSES else None


def get_conf_classes(conf_data):
    classes = []
    for obj in conf_data.get("objects", []):
        if not isinstance(obj, dict):
            continue
        class_name = normalize_class_name(obj.get("class"))
        if class_name is not None:
            classes.append(class_name)
    return classes


def get_inst_seg_classes(inst_seg_data):
    classes = []
    values = inst_seg_data.values() if isinstance(inst_seg_data, dict) else []
    for item in values:
        class_value = item.get("class") if isinstance(item, dict) else item
        class_name = normalize_class_name(class_value)
        if class_name is not None:
            classes.append(class_name)
    return classes


class MismatchCollector:
    """Collect unique conf paths by mismatch type and save deletion-ready lists."""

    def __init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = RESULT_ROOT / timestamp
        self.groups = defaultdict(set)
        self.details = []

    def add(self, mismatch_type, conf_path, camera_name, reason, conf_classes=None, inst_classes=None):
        conf_path = conf_path.resolve()
        self.groups[mismatch_type].add(str(conf_path))
        self.details.append(
            {
                "type": mismatch_type,
                "conf_path": str(conf_path),
                "camera": camera_name,
                "reason": reason,
                "conf_classes": sorted(set(conf_classes or [])),
                "inst_classes": sorted(set(inst_classes or [])),
            }
        )
        tqdm.write(
            f"[{mismatch_type}] {conf_path.relative_to(ROOT_PATH)} "
            f"camera={camera_name}: {reason}"
        )

    def save(self, comparison_count):
        self.result_dir.mkdir(parents=True, exist_ok=False)
        all_paths = set()

        for mismatch_type, paths in sorted(self.groups.items()):
            sorted_paths = sorted(paths)
            all_paths.update(sorted_paths)
            list_path = self.result_dir / f"{mismatch_type}.txt"
            list_path.write_text("".join(f"{path}\n" for path in sorted_paths), encoding="utf-8")

        all_paths = sorted(all_paths)
        (self.result_dir / "all_mismatches.txt").write_text(
            "".join(f"{path}\n" for path in all_paths),
            encoding="utf-8",
        )

        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_root": str(ROOT_PATH),
            "expected_object_count": EXPECTED_OBJECT_COUNT,
            "comparisons": comparison_count,
            "unique_mismatch_conf_count": len(all_paths),
            "groups": {
                mismatch_type: {
                    "count": len(paths),
                    "list_file": f"{mismatch_type}.txt",
                }
                for mismatch_type, paths in sorted(self.groups.items())
            },
            "details": self.details,
        }
        (self.result_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return len(all_paths)


def check_platform(platform_dir, collector):
    conf_dir = platform_dir / "conf"
    inst_seg_dir = platform_dir / "inst_seg"
    if not conf_dir.is_dir() or not inst_seg_dir.is_dir():
        return 0

    conf_paths = sorted(conf_dir.glob("*.json"))
    camera_dirs = sorted(path for path in inst_seg_dir.iterdir() if path.is_dir())
    comparison_count = 0

    if not camera_dirs:
        for conf_path in conf_paths:
            collector.add(
                "missing_camera_directory",
                conf_path,
                "-",
                "inst_seg contains no camera directory",
            )
        return len(conf_paths)

    for camera_dir in camera_dirs:
        description = str(camera_dir.relative_to(ROOT_PATH))
        for conf_path in tqdm(conf_paths, desc=description, leave=False):
            comparison_count += 1
            scene_num = conf_path.stem
            mapping_path = camera_dir / f"semantics_mapping_{scene_num}.json"

            try:
                with conf_path.open("r", encoding="utf-8") as file:
                    conf_data = json.load(file)
                conf_classes = get_conf_classes(conf_data)
            except (OSError, json.JSONDecodeError) as error:
                collector.add("conf_read_error", conf_path, camera_dir.name, str(error))
                continue

            if not mapping_path.is_file():
                collector.add(
                    "missing_mapping",
                    conf_path,
                    camera_dir.name,
                    f"missing {mapping_path.name}",
                    conf_classes,
                )
                continue

            try:
                with mapping_path.open("r", encoding="utf-8") as file:
                    inst_seg_data = json.load(file)
                inst_classes = get_inst_seg_classes(inst_seg_data)
            except (OSError, json.JSONDecodeError) as error:
                collector.add(
                    "mapping_read_error",
                    conf_path,
                    camera_dir.name,
                    str(error),
                    conf_classes,
                )
                continue

            conf_set = set(conf_classes)
            inst_set = set(inst_classes)

            if len(conf_classes) != EXPECTED_OBJECT_COUNT or len(conf_set) != EXPECTED_OBJECT_COUNT:
                collector.add(
                    "conf_object_count_mismatch",
                    conf_path,
                    camera_dir.name,
                    f"conf list={len(conf_classes)}, unique={len(conf_set)}, expected={EXPECTED_OBJECT_COUNT}",
                    conf_classes,
                    inst_classes,
                )
            if len(inst_set) != EXPECTED_OBJECT_COUNT:
                collector.add(
                    "inst_class_count_mismatch",
                    conf_path,
                    camera_dir.name,
                    f"inst unique={len(inst_set)}, expected={EXPECTED_OBJECT_COUNT}",
                    conf_classes,
                    inst_classes,
                )
            if conf_set != inst_set:
                collector.add(
                    "class_set_mismatch",
                    conf_path,
                    camera_dir.name,
                    f"conf_only={sorted(conf_set - inst_set)}, inst_only={sorted(inst_set - conf_set)}",
                    conf_classes,
                    inst_classes,
                )

    return comparison_count


def main():
    if not ROOT_PATH.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {ROOT_PATH}")

    platform_dirs = sorted(
        conf_dir.parent
        for conf_dir in ROOT_PATH.rglob("conf")
        if conf_dir.is_dir() and (conf_dir.parent / "inst_seg").is_dir()
    )

    collector = MismatchCollector()
    total_comparisons = 0
    for platform_dir in tqdm(platform_dirs, desc="Platforms"):
        total_comparisons += check_platform(platform_dir, collector)

    mismatch_count = collector.save(total_comparisons)
    print(f"\nDone: comparisons={total_comparisons}, unique mismatch conf={mismatch_count}")
    print(f"Result directory: {collector.result_dir}")
    for mismatch_type, paths in sorted(collector.groups.items()):
        print(f"  {mismatch_type}: {len(paths)}")


if __name__ == "__main__":
    main()
