"""Privacy-safe participant, session, and image identity mapping."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .models import MidsExportError

SAFE_LABEL = re.compile(r"^[A-Za-z0-9]+$")


def _present(value: Any) -> bool:
    return not pd.isna(value) and bool(str(value).strip())


def validate_label(
    value: Any, *, field: str, max_length: int = 32, lowercase: bool = True
) -> str:
    """Validate a label before it can become part of an exported path."""
    label = str(value).strip()
    if not label or len(label) > max_length or not SAFE_LABEL.fullmatch(label):
        raise MidsExportError(
            f"Unsafe {field} {value!r}; labels must contain 1-{max_length} "
            "ASCII letters or digits only (without the sub-/ses- prefix)."
        )
    return label.lower() if lowercase else label


def canonical_values(row: pd.Series, columns: list[str], *, field: str) -> list[str]:
    missing = [
        column for column in columns if column not in row or not _present(row[column])
    ]
    if missing:
        raise MidsExportError(f"Missing {field} source field(s): {', '.join(missing)}")
    return [str(row[column]).strip() for column in columns]


def _digest(key: bytes, namespace: str, values: list[str], length: int) -> str:
    payload = json.dumps(
        [namespace, *values], ensure_ascii=False, separators=(",", ":")
    )
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()[:length]


class IdentityMapper:
    """Map cohort identities either through an external map or keyed HMACs."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        id_map_path: str | None = None,
        id_map_frame: pd.DataFrame | None = None,
        key_file: str | None = None,
    ) -> None:
        self.config = config
        self.subject_column = str(config.get("subject_column", "patient_key"))
        session = config.get("session", {})
        session_columns = session.get("source_columns", ["study_id"])
        if not isinstance(session_columns, list) or not all(
            isinstance(value, str) and value.strip() for value in session_columns
        ):
            raise MidsExportError(
                "identity.session.source_columns must be a non-empty list of column names."
            )
        self.session_columns = list(session_columns)
        if not self.session_columns:
            raise MidsExportError("identity.session.source_columns must not be empty.")
        if self.session_columns == ["exam_stage"]:
            raise MidsExportError(
                "exam_stage cannot be the sole session identity because a clinical "
                "stage can contain repeated examinations."
            )
        image_columns = config.get("image_source_columns", ["volume_id"])
        if (
            not isinstance(image_columns, list)
            or not image_columns
            or not all(
                isinstance(value, str) and value.strip() for value in image_columns
            )
        ):
            raise MidsExportError(
                "identity.image_source_columns must be a non-empty list of column names."
            )
        self.image_columns = list(image_columns)
        self.length = int(config.get("digest_length", 12))
        if self.length < 8 or self.length > 32:
            raise MidsExportError("identity.digest_length must be between 8 and 32.")

        configured_map = (
            None if id_map_frame is not None else config.get("external_map")
        )
        self.map_path = (
            Path(id_map_path or configured_map).expanduser()
            if (id_map_path or configured_map)
            else None
        )
        if id_map_frame is not None and self.map_path is not None:
            raise MidsExportError(
                "Provide either an external ID-map path or an in-memory ID map, not both."
            )
        self.map: dict[tuple[str, ...], tuple[str, str, str]] = {}
        self.key: bytes | None = None
        self.uses_external_map = id_map_frame is not None or self.map_path is not None
        if id_map_frame is not None:
            self._load_map_frame(id_map_frame)
        elif self.map_path is not None:
            self._load_map(self.map_path)
        else:
            self.key = self._load_key(config, key_file)

    def _load_key(self, config: dict[str, Any], key_file: str | None) -> bytes:
        if key_file:
            path = Path(key_file).expanduser()
            if not path.is_file():
                raise MidsExportError(f"Identity key file not found: {path}")
            key = path.read_bytes().strip()
        else:
            env_name = str(config.get("key_env", "IMPERANDI_MIDS_KEY"))
            key = os.environ.get(env_name, "").encode("utf-8")
            if not key:
                raise MidsExportError(
                    "No privacy-safe identity mapping is available. Supply --id-map, "
                    "--key-file, or set the manifest identity.key_env variable "
                    f"({env_name})."
                )
        if len(key) < 16:
            raise MidsExportError("The identity key must contain at least 16 bytes.")
        return key

    def _load_map(self, path: Path) -> None:
        if not path.is_file():
            raise MidsExportError(f"External ID map not found: {path}")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        self._load_map_frame(frame)

    def _load_map_frame(self, frame: pd.DataFrame) -> None:
        required = [
            self.subject_column,
            *self.session_columns,
            *self.image_columns,
            "participant_label",
            "session_label",
            "image_label",
        ]
        missing = [column for column in required if column not in frame]
        if missing:
            raise MidsExportError(
                "External ID map is missing column(s): " + ", ".join(missing)
            )
        exam_labels: dict[tuple[str, ...], tuple[str, str]] = {}
        participant_sources: dict[str, str] = {}
        image_labels: dict[str, tuple[str, ...]] = {}
        for _, row in frame.iterrows():
            key = tuple(
                str(row[column]).strip()
                for column in [
                    self.subject_column,
                    *self.session_columns,
                    *self.image_columns,
                ]
            )
            if not all(key):
                raise MidsExportError(
                    "External ID map contains an empty identity value."
                )
            labels = (
                validate_label(
                    row["participant_label"],
                    field="participant_label",
                    lowercase=False,
                ),
                validate_label(
                    row["session_label"], field="session_label", lowercase=False
                ),
                validate_label(
                    row["image_label"], field="image_label", lowercase=False
                ),
            )
            previous = self.map.get(key)
            if previous is not None and previous != labels:
                raise MidsExportError(
                    f"External ID map has conflicting rows for {key!r}."
                )
            exam_key = tuple(
                str(row[column]).strip()
                for column in [self.subject_column, *self.session_columns]
            )
            if exam_key in exam_labels and exam_labels[exam_key] != labels[:2]:
                raise MidsExportError(
                    "External ID map assigns inconsistent labels to one examination."
                )
            exam_labels[exam_key] = labels[:2]
            source_subject = str(row[self.subject_column]).strip()
            if (
                labels[0] in participant_sources
                and participant_sources[labels[0]] != source_subject
            ):
                raise MidsExportError(
                    "External ID map reuses a participant label for different participants."
                )
            participant_sources[labels[0]] = source_subject
            safe_image_key = labels[2]
            if safe_image_key in image_labels and image_labels[safe_image_key] != key:
                raise MidsExportError(
                    "External ID map reuses an image label for different source volumes."
                )
            image_labels[safe_image_key] = key
            self.map[key] = labels

    def map_row(self, row: pd.Series) -> tuple[str, str, str]:
        subject = canonical_values(
            row, [self.subject_column], field="participant identity"
        )[0]
        session_values = canonical_values(
            row, self.session_columns, field="session identity"
        )
        image_values = canonical_values(row, self.image_columns, field="image identity")
        if self.uses_external_map:
            map_key = tuple([subject, *session_values, *image_values])
            if map_key not in self.map:
                raise MidsExportError(
                    "No external ID-map entry exists for this source volume."
                )
            participant, session, image_key = self.map[map_key]
        else:
            assert self.key is not None
            participant = _digest(self.key, "participant", [subject], self.length)
            session = _digest(
                self.key, "session", [subject, *session_values], self.length
            )
            image_key = _digest(
                self.key,
                "image",
                [subject, *session_values, *image_values],
                self.length,
            )
        return participant, session, image_key
