from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imperandi.process import _registration_common as reg_common


@dataclass(frozen=True)
class TransformArtifact:
    transform_path: str
    metadata_path: str
    matrix_3x4: dict[str, float]


def save_transform_artifacts(
    *,
    row_dir: str | Path,
    transform: Any,
    sitk_module: Any,
    prefix: str,
    metadata: dict[str, Any] | None = None,
) -> TransformArtifact:
    """Persist a transform and deterministic metadata under the row directory."""
    transforms_dir = Path(row_dir) / "transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)

    matrix_3x4 = reg_common.transform_to_flat_3x4(transform)
    transform_path = transforms_dir / f"{prefix}.tfm"
    metadata_path = transforms_dir / f"{prefix}.json"

    sitk_module.WriteTransform(transform, str(transform_path))
    payload: dict[str, Any] = {
        "format": "sitk_transform",
        "path": str(transform_path),
        "matrix_3x4": matrix_3x4,
    }
    if metadata:
        payload.update(metadata)
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    return TransformArtifact(
        transform_path=str(transform_path),
        metadata_path=str(metadata_path),
        matrix_3x4=matrix_3x4,
    )
