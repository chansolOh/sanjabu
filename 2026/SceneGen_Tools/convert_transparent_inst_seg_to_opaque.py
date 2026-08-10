"""Convert transparent-black SceneGen instance masks to opaque black.

Only files below ``*/inst_seg/<camera>/`` are considered.  For each RGBA PNG,
pixels equal to ``(0, 0, 0, 0)`` become ``(0, 0, 0, 255)``.  The matching
``semantics_mapping_XXXX.json`` color key is updated as well.

The operation is idempotent: PNGs without transparent-black pixels are skipped.
Writes use a temporary file followed by ``os.replace`` so a partially written
PNG or JSON is never exposed.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/dataset_v2")

# False performs a read-only scan and prints what would be changed.
# Set True to update the PNG and corresponding semantics mapping JSON files.
APPLY_CHANGES =True

# None scans the entire dataset.  Set a substring to restrict a test run.
PATH_CONTAINS: str | None = None

TRANSPARENT_BLACK = np.array((0, 0, 0, 0), dtype=np.uint8)
OPAQUE_BLACK = np.array((0, 0, 0, 255), dtype=np.uint8)
TRANSPARENT_KEY = "(0, 0, 0, 0)"
OPAQUE_KEY = "(0, 0, 0, 255)"


def atomic_write_png(path: Path, image: np.ndarray) -> None:
    """Write an RGBA PNG beside the original and atomically replace it."""
    original_mode = stat.S_IMODE(path.stat().st_mode)
    temporary_path = path.with_name(f".{path.name}.opaque_tmp")
    try:
        Image.fromarray(image, mode="RGBA").save(temporary_path, format="PNG")
        temporary_path.chmod(original_mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, data: dict) -> None:
    """Write mapping JSON beside the original and atomically replace it."""
    original_mode = stat.S_IMODE(path.stat().st_mode)
    temporary_path = path.with_name(f".{path.name}.opaque_tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=4)
            stream.write("\n")
        temporary_path.chmod(original_mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def mapping_path_for(image_path: Path) -> Path:
    return image_path.with_name(f"semantics_mapping_{image_path.stem}.json")


def converted_mapping(mapping_path: Path) -> tuple[dict | None, bool]:
    """Return updated mapping and whether its transparent key was changed."""
    if not mapping_path.is_file():
        return None, False

    with mapping_path.open("r", encoding="utf-8") as stream:
        mapping = json.load(stream)
    if not isinstance(mapping, dict):
        raise ValueError("mapping root is not a JSON object")
    if TRANSPARENT_KEY not in mapping:
        return mapping, False

    transparent_label = mapping.pop(TRANSPARENT_KEY)
    if OPAQUE_KEY not in mapping:
        mapping[OPAQUE_KEY] = transparent_label
    # If both keys existed, opaque black already represents UNLABELLED.  Once
    # the PNG pixels are opaque there is no distinct transparent color left,
    # so preserve the existing opaque entry and remove the obsolete key.
    return mapping, True


def iter_instance_segmentation_pngs(dataset_root: Path):
    """Yield only PNGs with the expected */inst_seg/<camera>/XXXX.png layout."""
    for image_path in dataset_root.glob("**/inst_seg/*/*.png"):
        if not image_path.stem.isdigit():
            continue
        if PATH_CONTAINS and PATH_CONTAINS not in str(image_path):
            continue
        yield image_path


def collect_instance_segmentation_pngs(dataset_root: Path) -> list[Path]:
    """Collect target paths while showing NAS/glob discovery progress."""
    image_paths: list[Path] = []
    discovery_progress = tqdm(
        desc="Discovering InstSeg PNGs",
        unit="image",
        dynamic_ncols=True,
    )
    try:
        for image_path in iter_instance_segmentation_pngs(dataset_root):
            image_paths.append(image_path)
            discovery_progress.update()
            discovery_progress.set_postfix(found=len(image_paths), refresh=False)
    finally:
        discovery_progress.close()

    return image_paths


def inspect_image(image_path: Path) -> tuple[np.ndarray, int]:
    """Load an RGBA mask and count its transparent-black pixels."""
    with Image.open(image_path) as source_image:
        image = np.asarray(source_image)

    if image.ndim != 3 or image.shape[2] != 4 or image.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 RGBA PNG, got dtype={image.dtype}, shape={image.shape}"
        )

    transparent_mask = np.all(image == TRANSPARENT_BLACK, axis=-1)
    return image, int(np.count_nonzero(transparent_mask))


def main() -> int:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    scanned = 0
    changed = 0
    skipped = 0
    changed_pixels = 0
    mapping_updated = 0
    mapping_missing = []
    failures: list[tuple[Path, str]] = []

    mode = "APPLY" if APPLY_CHANGES else "DRY RUN"
    print(f"[{mode}] Dataset root: {DATASET_ROOT}")

    image_paths = collect_instance_segmentation_pngs(DATASET_ROOT)
    print(f"Discovered InstSeg PNGs: {len(image_paths):,}")
    progress = tqdm(
        image_paths,
        desc=f"InstSeg {mode}",
        unit="image",
        dynamic_ncols=True,
    )

    for image_path in progress:
        scanned += 1
        try:
            image, transparent_pixel_count = inspect_image(image_path)
            if transparent_pixel_count == 0:
                skipped += 1
                continue

            current_mapping_path = mapping_path_for(image_path)
            updated_mapping, mapping_changed = converted_mapping(current_mapping_path)
            if not current_mapping_path.is_file():
                mapping_missing.append(current_mapping_path)

            if APPLY_CHANGES:
                converted_image = image.copy()
                transparent_mask = np.all(converted_image == TRANSPARENT_BLACK, axis=-1)
                converted_image[transparent_mask] = OPAQUE_BLACK
                # Mapping first keeps a failed run recoverable: if the PNG
                # replacement then fails, the still-transparent PNG is found
                # and retried on the next run.  A mapping failure leaves the
                # original PNG untouched.
                if mapping_changed and updated_mapping is not None:
                    atomic_write_json(current_mapping_path, updated_mapping)
                atomic_write_png(image_path, converted_image)

            changed += 1
            changed_pixels += transparent_pixel_count
            mapping_updated += int(mapping_changed)

            if changed <= 10:
                tqdm.write(
                    f"  {'updated' if APPLY_CHANGES else 'would update'}: "
                    f"{image_path} ({transparent_pixel_count:,} pixels)"
                )
        except Exception as error:
            failures.append((image_path, str(error)))
        finally:
            progress.set_postfix(
                target=changed,
                skipped=skipped,
                failures=len(failures),
                refresh=False,
            )

    print("\nSummary")
    print(f"  mode: {mode}")
    print(f"  scanned PNGs: {scanned:,}")
    print(f"  already opaque/skipped: {skipped:,}")
    print(f"  {'updated' if APPLY_CHANGES else 'would update'} PNGs: {changed:,}")
    print(f"  transparent pixels found: {changed_pixels:,}")
    print(f"  {'updated' if APPLY_CHANGES else 'would update'} mappings: {mapping_updated:,}")
    print(f"  missing mappings: {len(mapping_missing):,}")
    print(f"  failures: {len(failures):,}")

    if mapping_missing:
        print("\nMissing mapping files")
        for path in mapping_missing:
            print(f"  - {path}")

    if failures:
        print("\nFailed files")
        for path, error in failures:
            print(f"  - {path}: {error}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
