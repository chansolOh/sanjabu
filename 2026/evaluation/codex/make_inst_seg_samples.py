"""Create a reproducible, platform/camera-balanced manual review manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import config
from evaluation_core import (
    atomic_write_json,
    run_metadata,
    scene_refs_for_split,
    stratified_sample,
    strict_json_load,
    write_csv,
)


def review_item(scene, camera: str) -> dict[str, Any]:
    platform = scene.platform_path
    name = scene.scene_name
    paths = {
        "rgb": platform / "rgb" / camera / f"{name}.png",
        "inst_seg": platform / "inst_seg" / camera / f"{name}.png",
        "mapping": platform
        / "inst_seg"
        / camera
        / f"semantics_mapping_{name}.json",
        "bbox": platform / "bbox" / camera / f"{name}.json",
        "conf": platform / "conf" / f"{name}.json",
    }
    return {
        "uid": f"{scene.uid}|{camera}",
        "split": scene.split,
        "root": str(scene.root),
        "env": scene.env,
        "section": scene.section,
        "platform": scene.platform,
        "scene_id": scene.scene_id,
        "scene_name": name,
        "camera": camera,
        "relative_scene": scene.relative_scene,
        "files_complete_at_sampling": all(path.is_file() for path in paths.values()),
        "paths": {key: str(path) for key, path in paths.items()},
    }


def create_manifest() -> dict[str, Any]:
    if (
        config.MANUAL_MANIFEST_PATH.is_file()
        and not config.REBUILD_MANUAL_SAMPLE_MANIFEST
    ):
        manifest = strict_json_load(config.MANUAL_MANIFEST_PATH)
        print(f"Keeping existing manifest: {config.MANUAL_MANIFEST_PATH}")
        print(
            "Set REBUILD_MANUAL_SAMPLE_MANIFEST=True in config.py to replace it."
        )
        return manifest

    items: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    for split_number, split in enumerate(config.DATASETS):
        scenes = scene_refs_for_split(split)
        pool = [
            (scene, camera)
            for scene in scenes
            for camera in config.MANUAL_CAMERAS
        ]
        requested = config.MANUAL_SAMPLES_PER_SPLIT.get(split, 0)
        selected_pairs = stratified_sample(
            pool,
            requested,
            lambda pair: (
                pair[0].env,
                pair[0].section,
                pair[0].platform,
                pair[1],
            ),
            config.RANDOM_SEED + 300 + split_number,
        )
        # Existence checks are intentionally performed only for the selected
        # review items, not for hundreds of thousands of candidates on NAS.
        selected = [review_item(scene, camera) for scene, camera in selected_pairs]
        items.extend(selected)
        selection[split] = {
            "available_items": len(pool),
            "requested_items": requested,
            "selected_items": len(selected),
            "incomplete_selected_items": sum(
                not item["files_complete_at_sampling"] for item in selected
            ),
        }

    manifest = {
        "metadata": run_metadata("make_inst_seg_samples.py"),
        "selection": selection,
        "cameras": list(config.MANUAL_CAMERAS),
        "item_count": len(items),
        "items": items,
    }
    atomic_write_json(config.MANUAL_MANIFEST_PATH, manifest)
    write_csv(
        config.MANUAL_MANIFEST_PATH.with_suffix(".csv"),
        [
            {
                key: value
                for key, value in item.items()
                if key not in {"paths"}
            }
            for item in items
        ],
    )
    print(f"Created {len(items):,} manual-review items")
    print(f"saved: {config.MANUAL_MANIFEST_PATH}")
    return manifest


def main() -> None:
    create_manifest()


if __name__ == "__main__":
    main()
