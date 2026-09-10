"""User-editable settings for the Dataset 2026 certification evaluation.

This project intentionally uses variables instead of required CLI arguments.
Edit this file, then run the individual scripts directly.
"""

from pathlib import Path


HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Dataset and reference-data paths
# ---------------------------------------------------------------------------
DATASETS = {
    "train": {
        "root": Path("/nas/Dataset/Dataset_2026/dataset_v2"),
        "require_pre_grasp": True,
        "require_output_grasp": True,
    },
    "validation": {
        "root": Path("/nas/Dataset/Dataset_2026/dataset_v2_val"),
        "require_pre_grasp": False,
        "require_output_grasp": False,
    },
    "test": {
        "root": Path("/nas/Dataset/Dataset_2026/dataset_v2_test"),
        "require_pre_grasp": False,
        "require_output_grasp": False,
    },
}

OBJECT_CSV = Path(
    "/nas/ochansol/3d_model/peel3_scan_data_2026/"
    "2026_objects_cat_attr.csv"
)
FINGER_GRIPPER_CATALOG = Path(
    "/nas/ochansol/gripper_info/gripper_info_new_2026.json"
)
HAND_GRIPPER_CATALOG = Path(
    "/nas/ochansol/gripper_info/gripper_info_hand_2026.json"
)

CAMERAS = ("top_view_camera", "side_view_camera")
EXPECTED_IMAGE_SIZE = (1920, 1080)  # width, height
EXPECTED_OBJECTS_PER_SCENE = 5
OBJECT_ID_PATTERN = r"^obj_\d{3}$"

# Results from every script are collected beneath this directory.
RUN_NAME = "certification_2026"
RESULT_DIR = HERE / "results" / RUN_NAME
MAX_ISSUE_EXAMPLES = 10_000
RANDOM_SEED = 20260908

# ---------------------------------------------------------------------------
# Automatic syntax validation
# ---------------------------------------------------------------------------
# None means every conf scene. Set an integer only for a quick development run.
SYNTAX_MAX_SCENES_PER_SPLIT = None

# JSON is checked for every selected scene. Large binary arrays/images are
# opened only for this deterministic stratified sample per split.
BINARY_PAYLOAD_SAMPLES_PER_SPLIT = 200

# ---------------------------------------------------------------------------
# Automatic semantic/cross-modal validation
# ---------------------------------------------------------------------------
# Per split, sampled evenly across platforms. Each selected scene is checked
# for both cameras. Set None for a full semantic image scan.
SEMANTIC_SAMPLES_PER_SPLIT = 1_007
BBOX_PIXEL_TOLERANCE = 0

# ---------------------------------------------------------------------------
# Manual instance-segmentation review
# ---------------------------------------------------------------------------
# Counts are review items, not scenes. Items are balanced across platform and
# camera, and selection is reproducible with RANDOM_SEED.
MANUAL_SAMPLES_PER_SPLIT = {
    "train": 1_007,
    "validation": 1_007,
    "test": 1_007,
}
MANUAL_CAMERAS = CAMERAS
REBUILD_MANUAL_SAMPLE_MANIFEST = False
REVIEWER_NAME = ""
MANUAL_MANIFEST_PATH = RESULT_DIR / "manual_inst_seg_manifest.json"
MANUAL_RECORD_PATH = RESULT_DIR / "manual_inst_seg_records.json"

# GUI presentation.
REVIEW_WINDOW_SIZE = "1760x980"
REVIEW_PANEL_SIZE = (540, 420)
DEFAULT_OVERLAY_ALPHA = 0.45
