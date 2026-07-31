#!/usr/bin/env python3
import csv
import json
from pathlib import Path


BASE_DIR = "/nas/ochansol/3d_model/peel3_scan_data_2026"
CSV_PATH = BASE_DIR / "2026_objects_cat_attr.csv"
OUTPUT_PATH = BASE_DIR / "objects_conf.json"
YEAR = 2026

DEFAULT_ENVS = {
    "Home": 0,
    "Manufactory": 0,
    "Logistic_site": 0,
}


def clean(value):
    return (value or "").strip()


def load_rows(csv_path):
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if clean(row.get("Class_name"))]


def row_to_object(row):
    class_name = clean(row["Class_name"])
    object_name = clean(row.get("Object_name")) or class_name

    return {
        "path": str(BASE_DIR / class_name / "edited" / f"{class_name}.usd"),
        "name": class_name,
        "envs": dict(DEFAULT_ENVS),
        "attributes": {
            "color": clean(row.get("Color")),
            "packaging": clean(row.get("Packaging")),
            "features": clean(row.get("Features")),
            "description": clean(row.get("Description")),
        },
        "size": 0,
        "size_rank": 0,
        "year": YEAR,
        "category": {
            "level_1": clean(row.get("Level_1")),
            "level_2": clean(row.get("Level_2")),
            "level_3": clean(row.get("Level_3")),
            "object_name": object_name,
        },
    }


def main():
    rows = load_rows(CSV_PATH)
    objects_conf = [row_to_object(row) for row in rows]

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(objects_conf, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(objects_conf)} objects to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
