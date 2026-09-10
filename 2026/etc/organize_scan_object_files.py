"""Organize the original Peel3 files in each 2026 object directory.

For an object directory named ``object_name`` this script performs:

    object_name.mtl  -> org/object_name.mtl
    object_name.bmp  -> org/object_name.bmp
    object_name.obj  -> org/object_name.obj
    object_name_org.bmp -> delete

``object_name_edited.bmp`` and every other file/directory remain untouched.
There are intentionally no command-line arguments; edit the variables below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


# =============================================================================
# User settings
# =============================================================================
ROOT_DIR ="" #Path("/nas/ochansol/3d_model/2026_real_world_test_objects")

# True: only print the complete plan. No directory is created, no file is moved,
# and no file is deleted. Run this mode first and inspect the summary.
DRY_RUN = False

# A real run requires BOTH DRY_RUN=False and this exact confirmation text.
# This prevents an accidental run after changing only one variable.
EXECUTION_CONFIRMATION = "ORGANIZE_2026_OBJECT_FILES"
REQUIRED_CONFIRMATION = "ORGANIZE_2026_OBJECT_FILES"

# Empty means every object directory. To test selected objects first, for example:
# ONLY_OBJECT_NAMES = ["air_plunger", "bike_bell"]
ONLY_OBJECT_NAMES: list[str] = []

# The requested deletion can be disabled independently if necessary.
DELETE_OBJECT_NAME_ORG_BMP = True

# A directory missing both the source file and its org/ destination is considered
# malformed. Keeping this True prevents a partial dataset-wide reorganization.
FAIL_ON_MISSING_REQUIRED_FILES = True

SKIP_DIRECTORY_NAMES = {"__pycache__"}
MOVE_EXTENSIONS = (".mtl", ".bmp", ".obj")


@dataclass(frozen=True)
class MoveAction:
    object_name: str
    source: Path
    destination: Path


@dataclass(frozen=True)
class DeleteAction:
    object_name: str
    path: Path


@dataclass
class Plan:
    object_directories: list[Path]
    moves: list[MoveAction]
    deletes: list[DeleteAction]
    already_moved: list[Path]
    already_deleted: list[Path]
    warnings: list[str]
    errors: list[str]


def discover_object_directories() -> tuple[list[Path], list[str]]:
    if not ROOT_DIR.is_dir():
        raise NotADirectoryError(f"ROOT_DIR does not exist: {ROOT_DIR}")

    directories = sorted(
        path
        for path in ROOT_DIR.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and not path.name.startswith(".")
        and path.name not in SKIP_DIRECTORY_NAMES
    )

    if not ONLY_OBJECT_NAMES:
        return directories, []

    requested = set(ONLY_OBJECT_NAMES)
    duplicate_names = sorted(
        name for name in requested if ONLY_OBJECT_NAMES.count(name) > 1
    )
    available = {path.name for path in directories}
    unknown = sorted(requested - available)
    errors = []
    if duplicate_names:
        errors.append(f"duplicate ONLY_OBJECT_NAMES: {duplicate_names}")
    if unknown:
        errors.append(f"unknown ONLY_OBJECT_NAMES: {unknown}")
    return [path for path in directories if path.name in requested], errors


def build_plan() -> Plan:
    object_directories, discovery_errors = discover_object_directories()
    moves: list[MoveAction] = []
    deletes: list[DeleteAction] = []
    already_moved: list[Path] = []
    already_deleted: list[Path] = []
    warnings: list[str] = []
    errors = list(discovery_errors)

    for object_directory in object_directories:
        object_name = object_directory.name
        org_directory = object_directory / "org"

        if org_directory.exists() and not org_directory.is_dir():
            errors.append(f"org path is not a directory: {org_directory}")
            continue

        edited_bmp = object_directory / f"{object_name}_edited.bmp"
        if not edited_bmp.is_file():
            warnings.append(f"edited BMP is missing (no action taken): {edited_bmp}")

        for extension in MOVE_EXTENSIONS:
            source = object_directory / f"{object_name}{extension}"
            destination = org_directory / source.name
            source_exists = source.is_file()
            destination_exists = destination.is_file()

            if source_exists and destination_exists:
                errors.append(
                    "move conflict; source and destination both exist: "
                    f"{source} <-> {destination}"
                )
            elif source_exists:
                moves.append(MoveAction(object_name, source, destination))
            elif destination_exists:
                already_moved.append(destination)
            else:
                message = (
                    "required original is missing from both locations: "
                    f"{source} / {destination}"
                )
                if FAIL_ON_MISSING_REQUIRED_FILES:
                    errors.append(message)
                else:
                    warnings.append(message)

        original_copy = object_directory / f"{object_name}_org.bmp"
        if original_copy.exists() and not original_copy.is_file():
            errors.append(f"*_org.bmp path is not a file: {original_copy}")
        elif DELETE_OBJECT_NAME_ORG_BMP and original_copy.is_file():
            deletes.append(DeleteAction(object_name, original_copy))
        elif not DELETE_OBJECT_NAME_ORG_BMP and original_copy.is_file():
            warnings.append(f"deletion is disabled; keeping: {original_copy}")
        else:
            already_deleted.append(original_copy)

    return Plan(
        object_directories=object_directories,
        moves=moves,
        deletes=deletes,
        already_moved=already_moved,
        already_deleted=already_deleted,
        warnings=warnings,
        errors=errors,
    )


def print_plan(plan: Plan) -> None:
    mode = "DRY RUN — NO FILES WILL BE CHANGED" if DRY_RUN else "EXECUTION MODE"
    border = "=" * 88
    print(border)
    print(mode)
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(border)

    for action in plan.moves:
        print(f"[MOVE]   {action.source} -> {action.destination}")
    for action in plan.deletes:
        print(f"[DELETE] {action.path}")
    for warning in plan.warnings:
        print(f"[WARNING] {warning}")
    for error in plan.errors:
        print(f"[ERROR]   {error}")

    print(border)
    print(f"object directories : {len(plan.object_directories):,}")
    print(f"files to move       : {len(plan.moves):,}")
    print(f"files to delete     : {len(plan.deletes):,}")
    print(f"already moved       : {len(plan.already_moved):,}")
    print(f"already deleted     : {len(plan.already_deleted):,}")
    print(f"warnings            : {len(plan.warnings):,}")
    print(f"errors              : {len(plan.errors):,}")
    print(border)


def execute_plan(plan: Plan) -> None:
    if plan.errors:
        raise RuntimeError("preflight errors found; nothing was changed")
    if DRY_RUN:
        print("DRY RUN complete. Nothing was changed.")
        return
    if EXECUTION_CONFIRMATION != REQUIRED_CONFIRMATION:
        raise RuntimeError(
            "execution confirmation is missing. Set EXECUTION_CONFIRMATION to "
            f"{REQUIRED_CONFIRMATION!r}. Nothing was changed."
        )

    # Complete and verify all moves before performing irreversible deletions.
    completed_moves: list[MoveAction] = []
    try:
        for action in plan.moves:
            action.destination.parent.mkdir(parents=True, exist_ok=True)
            action.source.rename(action.destination)
            completed_moves.append(action)
    except Exception:
        # Moving within an object directory is reversible. Roll back moves from
        # this run if any later move fails; pre-existing org files are untouched.
        rollback_errors = []
        for action in reversed(completed_moves):
            try:
                action.destination.rename(action.source)
            except Exception as error:
                rollback_errors.append(f"{action.destination}: {error}")
        if rollback_errors:
            print("ROLLBACK ERRORS:", file=sys.stderr)
            print("\n".join(rollback_errors), file=sys.stderr)
        raise

    for action in plan.moves:
        if action.source.exists() or not action.destination.is_file():
            raise RuntimeError(f"move verification failed: {action}")

    for action in plan.deletes:
        action.path.unlink()

    for action in plan.deletes:
        if action.path.exists():
            raise RuntimeError(f"delete verification failed: {action.path}")

    # The edited texture must not be moved or deleted by this script.
    missing_edited = [
        directory / f"{directory.name}_edited.bmp"
        for directory in plan.object_directories
        if not (directory / f"{directory.name}_edited.bmp").is_file()
    ]
    if missing_edited:
        print(
            "WARNING: edited BMP was already missing in these directories:\n"
            + "\n".join(str(path) for path in missing_edited),
            file=sys.stderr,
        )

    print(
        f"Completed: moved {len(plan.moves):,} files and "
        f"deleted {len(plan.deletes):,} *_org.bmp files."
    )


def main() -> None:
    plan = build_plan()
    print_plan(plan)
    execute_plan(plan)


if __name__ == "__main__":
    main()
