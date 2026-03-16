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
    rigid_pose: dict[str, Any] | None


def _rigid_pose_to_population_pose(rigid_pose: dict[str, Any]) -> dict[str, Any]:
    quaternion = rigid_pose.get("quaternion_xyzw")
    translation = rigid_pose.get("translation_xyz")
    if not isinstance(quaternion, list) or len(quaternion) != 4:
        raise ValueError("rigid_pose.quaternion_xyzw must be a 4-item list")
    if not isinstance(translation, list) or len(translation) != 3:
        raise ValueError("rigid_pose.translation_xyz must be a 3-item list")
    return {
        "population_tx_qx": quaternion[0],
        "population_tx_qy": quaternion[1],
        "population_tx_qz": quaternion[2],
        "population_tx_qw": quaternion[3],
        "population_tx_t0": translation[0],
        "population_tx_t1": translation[1],
        "population_tx_t2": translation[2],
    }


def _rigid_pose_payload(transform: Any) -> dict[str, Any] | None:
    try:
        flat_pose = reg_common.transform_to_flat_quaternion_translation(transform)
    except Exception:
        return None
    return {
        "quaternion_xyzw": [
            float(flat_pose[column]) for column in reg_common.POPULATION_QUATERNION_COLUMNS
        ],
        "translation_xyz": [
            float(flat_pose[column]) for column in reg_common.POPULATION_TRANSLATION_COLUMNS
        ],
    }


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

    rigid_pose = _rigid_pose_payload(transform)
    transform_path = transforms_dir / f"{prefix}.tfm"
    metadata_path = transforms_dir / f"{prefix}.json"

    sitk_module.WriteTransform(transform, str(transform_path))
    payload: dict[str, Any] = {
        "format": "sitk_transform",
        "path": str(transform_path),
    }
    if rigid_pose is not None:
        payload["rigid_pose"] = rigid_pose
    if metadata:
        payload.update(metadata)
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    return TransformArtifact(
        transform_path=str(transform_path),
        metadata_path=str(metadata_path),
        rigid_pose=rigid_pose,
    )


def load_rigid_transform_from_metadata(
    metadata_path: str | Path,
    *,
    sitk_module: Any,
):
    payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    rigid_pose = payload.get("rigid_pose")
    if isinstance(rigid_pose, dict):
        return reg_common.rigid_transform_from_population_pose(
            _rigid_pose_to_population_pose(rigid_pose),
            sitk_module=sitk_module,
        )
    matrix_3x4 = payload.get("matrix_3x4")
    if isinstance(matrix_3x4, dict):
        return reg_common.rigid_transform_from_population_pose(
            matrix_3x4,
            sitk_module=sitk_module,
        )
    raise ValueError(
        f"transform metadata does not contain a loadable rigid pose: {metadata_path}"
    )
