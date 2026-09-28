"""Data models shared by the MIDS-style exporter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class MidsExportError(ValueError):
    """Raised for an invalid or unsafe export request."""


@dataclass(frozen=True)
class ExportFile:
    """One source file and its destination relative to the dataset root."""

    source: Path
    relative_path: Path
    kind: str
    image_id: str
    source_image: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportPlan:
    """Complete, write-free export plan."""

    output_dir: Path
    files: list[ExportFile] = field(default_factory=list)
    tables: dict[Path, list[dict[str, Any]]] = field(default_factory=dict)
    json_files: dict[Path, dict[str, Any]] = field(default_factory=dict)
    text_files: dict[Path, str] = field(default_factory=dict)
    excluded_rows: list[dict[str, Any]] = field(default_factory=list)
    missing_fields: dict[str, list[str]] = field(default_factory=dict)
    collisions: dict[str, list[str]] = field(default_factory=dict)
    resolved_collisions: dict[str, list[str]] = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def report(self) -> dict[str, Any]:
        image_count = sum(item.kind == "image" for item in self.files)
        mask_count = sum(
            item.kind in {"source_mask", "consensus_mask"} for item in self.files
        )
        radiomics_rows = sum(
            len(rows)
            for path, rows in self.tables.items()
            if path.name == "radiomics.tsv"
        )
        invalid = (
            bool(self.collisions) or image_count == 0 or bool(self.validation_errors)
        )
        return {
            "profile": "IMPERANDI strict CT/MR volumetric MIDS-style profile",
            "status": (
                "invalid"
                if invalid
                else ("ready_with_exclusions" if self.excluded_rows else "ready")
            ),
            "counts": {
                "images": image_count,
                "masks": mask_count,
                "radiomics_rows": radiomics_rows,
                "excluded_rows": len(self.excluded_rows),
                "planned_dataset_files": (
                    len(self.files)
                    + len(self.tables)
                    + len(self.json_files)
                    + len(self.text_files)
                    + 1
                ),
            },
            "planned_files": sorted(
                {
                    *(item.relative_path.as_posix() for item in self.files),
                    *(path.as_posix() for path in self.tables),
                    *(path.as_posix() for path in self.json_files),
                    *(path.as_posix() for path in self.text_files),
                    "export_report.json",
                }
            ),
            "missing_fields": self.missing_fields,
            "collisions": self.collisions,
            "resolved_collisions": self.resolved_collisions,
            "validation_errors": self.validation_errors,
            "excluded_rows": self.excluded_rows,
            "warnings": self.warnings,
        }
