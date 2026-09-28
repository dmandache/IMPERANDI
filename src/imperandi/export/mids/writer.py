"""Self-contained, staged file writer for MIDS-style exports."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import pandas as pd

from .metadata import tsv_value
from .models import ExportPlan, MidsExportError
from .validation import validate_written_dataset


def _validate_target(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if resolved == Path(resolved.anchor) or len(resolved.parts) < 3:
        raise MidsExportError(f"Refusing unsafe export target: {resolved}")


def write_plan(plan: ExportPlan, *, overwrite: str = "never") -> None:
    """Write to a sibling staging directory, validate, then atomically publish."""
    if plan.output_dir.is_symlink():
        raise MidsExportError("output_dir must not be a symbolic link.")
    output = plan.output_dir.resolve()
    _validate_target(output)
    if overwrite not in {"never", "replace"}:
        raise MidsExportError("overwrite must be 'never' or 'replace'.")
    if output.exists() and overwrite == "never":
        raise MidsExportError(
            f"Output already exists: {output}. Use --overwrite replace to rebuild it safely."
        )
    if output.exists() and not output.is_dir():
        raise MidsExportError(f"Output exists but is not a directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staging = output.parent / f".{output.name}.imperandi-stage-{token}"
    backup = output.parent / f".{output.name}.imperandi-backup-{token}"
    staging.mkdir()
    try:
        for item in plan.files:
            destination = staging / item.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.source, destination)
        for relative, rows in plan.tables.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            normalized = [
                {key: tsv_value(value) for key, value in row.items()} for row in rows
            ]
            pd.DataFrame(normalized).to_csv(
                destination, sep="\t", index=False, lineterminator="\n"
            )
        for relative, content in plan.json_files.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        for relative, content in plan.text_files.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        (staging / "export_report.json").write_text(
            json.dumps(plan.report(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_written_dataset(staging, plan)
        if output.exists():
            os.replace(output, backup)
        try:
            os.replace(staging, output)
        except Exception:
            if backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
