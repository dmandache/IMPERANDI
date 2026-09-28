"""Pre-write and staged-dataset validation."""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from .metadata import DISALLOWED_OUTPUT_FRAGMENTS
from .models import ExportPlan, MidsExportError

PATH_COLUMNS = {"filename", "mask_path", "source_image"}
IMAGE_PATH = re.compile(
    r"^sub-[A-Za-z0-9]+/ses-[A-Za-z0-9]+/mim-(ct|mr)/"
    r"sub-[A-Za-z0-9]+_ses-[A-Za-z0-9]+(?:_(?:bp|acq|pc|rec)-[a-z0-9]+)*(?:_run-[0-9]+)?_(?:ct|mr)\.nii(?:\.gz)?$"
)


def _safe_relative(path: Path) -> bool:
    pure = PurePosixPath(path.as_posix())
    return not pure.is_absolute() and ".." not in pure.parts and bool(pure.parts)


def validate_plan(plan: ExportPlan) -> None:
    """Validate identifiers, sources, destinations, associations, and tables."""
    errors: list[str] = []
    all_destinations = [item.relative_path for item in plan.files]
    all_destinations += (
        list(plan.tables) + list(plan.json_files) + list(plan.text_files)
    )
    duplicate_paths = sorted(
        {
            path.as_posix()
            for path in all_destinations
            if all_destinations.count(path) > 1
        }
    )
    if duplicate_paths:
        errors.append("duplicate output paths: " + ", ".join(duplicate_paths))
    if plan.collisions:
        errors.append("unresolved filename collisions: " + ", ".join(plan.collisions))
    for item in plan.files:
        if not item.source.is_file():
            errors.append(f"missing source file for {item.image_id}")
        try:
            item.source.resolve().relative_to(plan.output_dir.resolve())
        except ValueError:
            pass
        else:
            errors.append(
                f"source file is inside output_dir and would not remain immutable: {item.image_id}"
            )
        if not _safe_relative(item.relative_path):
            errors.append(f"unsafe destination: {item.relative_path}")
        if item.kind == "image" and not IMAGE_PATH.fullmatch(
            item.relative_path.as_posix()
        ):
            errors.append(f"unsupported image path/label: {item.relative_path}")

    image_paths = {
        item.relative_path.as_posix() for item in plan.files if item.kind == "image"
    }
    if not image_paths:
        errors.append("no exportable CT/MR NIfTI images")
    for item in plan.files:
        if item.kind.endswith("mask") and item.source_image not in image_paths:
            errors.append(
                f"mask has no planned source-image association: {item.relative_path}"
            )

    for path, rows in plan.tables.items():
        if not _safe_relative(path):
            errors.append(f"unsafe table destination: {path}")
        for row in rows:
            for column, value in row.items():
                folded = str(column).casefold()
                if any(fragment in folded for fragment in DISALLOWED_OUTPUT_FRAGMENTS):
                    errors.append(f"disallowed identifying column in {path}: {column}")
                if column not in PATH_COLUMNS and isinstance(value, str):
                    if os.path.isabs(value) or value.startswith("file:"):
                        errors.append(f"raw source path found in {path}:{column}")
            if "filename" in row:
                candidate = path.parent / str(row["filename"])
                if candidate.as_posix() not in image_paths:
                    errors.append(f"scan table reference is not planned: {candidate}")
            if "source_image" in row and str(row["source_image"]) not in image_paths:
                errors.append(
                    f"derivative source_image is not planned: {row['source_image']}"
                )

    if errors:
        raise MidsExportError(
            "Export validation failed: " + "; ".join(dict.fromkeys(errors))
        )


def validate_written_dataset(root: Path, plan: ExportPlan) -> None:
    """Confirm that every planned output and every TSV reference exists."""
    missing = [
        path.as_posix()
        for path in [
            *(item.relative_path for item in plan.files),
            *plan.tables,
            *plan.json_files,
            *plan.text_files,
        ]
        if not (root / path).is_file()
    ]
    if missing:
        raise MidsExportError("Staged export is missing files: " + ", ".join(missing))
    for path, rows in plan.tables.items():
        for row in rows:
            if (
                "filename" in row
                and not (root / path.parent / str(row["filename"])).is_file()
            ):
                raise MidsExportError(
                    f"Broken scan-table reference in {path}: {row['filename']}"
                )
            for column in ("source_image", "mask_path"):
                if column in row and not (root / str(row[column])).is_file():
                    raise MidsExportError(
                        f"Broken {column} reference in {path}: {row[column]}"
                    )
