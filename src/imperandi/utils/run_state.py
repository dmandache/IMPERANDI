from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from imperandi.utils.logging import log_task_summary


def log_finished_resume_summary(
    logger: logging.Logger,
    task_name: str,
    state: Mapping[str, Any] | None,
    error_checkpoint_path: Path,
) -> None:
    """Report checkpoint skips without treating prior failures as successes."""
    resumed_ids = normalize_source_ids((state or {}).get("completed_indices", []))
    prior_failed_ids = set()
    if error_checkpoint_path.exists():
        prior_errors = pd.read_csv(error_checkpoint_path)
        error_key = "_source_idx" if "_source_idx" in prior_errors else "idx"
        if error_key in prior_errors:
            prior_failed_ids = normalize_source_ids(prior_errors[error_key])
    prior_failed_count = len(resumed_ids & prior_failed_ids)
    log_task_summary(
        logger,
        task_name,
        processed_rows=0,
        succeeded_rows=0,
        skipped_rows=len(resumed_ids),
        resumed_rows=len(resumed_ids) - prior_failed_count,
        extra_counts={"skipped by resume after prior failure": prior_failed_count},
    )


STATE_SCHEMA_VERSION = 3
DEFAULT_HASH_EXCLUDE_KEYS = frozenset(
    {
        "resume",
        "checkpoint_every_rows",
        "checkpoint_every_sec",
        "strict_resume",
    }
)


@dataclass(frozen=True)
class CheckpointPaths:
    state_path: Path
    main_checkpoint_path: Path
    error_checkpoint_path: Path


@dataclass(frozen=True)
class CheckpointConfig:
    command: str
    args_hash: str
    input_fingerprint: Any
    checkpoint_every_rows: int
    checkpoint_every_sec: int
    resume_enabled: bool
    row_fingerprints: dict[str, str] | None = None


@dataclass(frozen=True)
class SemanticFingerprint:
    """Task-specific identity for a mutable cohort table."""

    input_fingerprint: dict[str, Any]
    row_fingerprints: dict[str, str]


def build_checkpoint_paths(
    output_path: str | Path,
    error_path: str | Path,
    command: str,
) -> CheckpointPaths:
    out = Path(output_path)
    err = Path(error_path)
    return CheckpointPaths(
        state_path=out.parent / f".{out.stem}.{command}.state.json",
        main_checkpoint_path=out.parent / f".{out.stem}.{command}.checkpoint.csv",
        error_checkpoint_path=err.parent / f".{err.stem}.{command}.checkpoint.csv",
    )


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _normalize_for_json(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_json(v) for v in value]
    return value


def compute_args_hash(
    args: Any,
    *,
    exclude_keys: Iterable[str] = (),
) -> str:
    raw = vars(args) if hasattr(args, "__dict__") else dict(args)
    payload = {
        str(k): _normalize_for_json(v)
        for k, v in raw.items()
        if not str(k).startswith("_") and str(k) not in set(exclude_keys)
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: str | Path, *, strict: bool = False) -> dict[str, Any]:
    p = Path(path).expanduser()
    try:
        rp = p.resolve()
    except Exception:
        rp = p
    out: dict[str, Any] = {"path": str(rp), "exists": rp.exists()}
    if not rp.exists() or not rp.is_file():
        return out
    stat = rp.stat()
    out.update(
        {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    )
    if strict:
        out["sha256"] = _sha256_file(rp)
    return out


def fingerprint_inputs(
    inputs: str | Path | Sequence[str | Path] | None,
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    if inputs is None:
        return []
    if isinstance(inputs, (str, Path)):
        values: list[str | Path] = [inputs]
    else:
        values = list(inputs)
    fps = [fingerprint_file(p, strict=strict) for p in values]
    return sorted(fps, key=lambda x: str(x.get("path", "")))


def _normalize_semantic_value(value: Any) -> Any:
    """Normalize dataframe cells into deterministic JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(k): _normalize_semantic_value(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        values = sorted(value, key=str) if isinstance(value, set) else value
        return [_normalize_semantic_value(v) for v in values]

    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return _normalize_semantic_value(value.item())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _fingerprint_artifact_value(value: Any, *, strict: bool) -> Any:
    """Fingerprint file-valued cells without coupling to ordinary mtime changes.

    Lightweight mode records path identity and existence only. Strict mode adds
    content hashes, which intentionally costs more but detects in-place file
    replacement.
    """
    normalized = _normalize_semantic_value(value)
    if isinstance(normalized, list):
        return [
            _fingerprint_artifact_value(item, strict=strict) for item in normalized
        ]
    if normalized is None or not isinstance(normalized, str):
        return normalized

    raw = normalized.strip()
    if not raw:
        return raw
    if raw[:1] in {"[", "("}:
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [
                _fingerprint_artifact_value(item, strict=strict) for item in parsed
            ]
    p = Path(raw).expanduser()
    try:
        rp = p.resolve()
    except Exception:
        rp = p
    out: dict[str, Any] = {"path": str(rp), "exists": rp.exists()}
    if strict and rp.exists() and rp.is_file():
        out["sha256"] = _sha256_file(rp)
    return out


def fingerprint_dataframe_semantic(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] = (),
    dynamic_prefixes: Sequence[str] = (),
    artifact_columns: Sequence[str] = (),
    artifact_prefixes: Sequence[str] = (),
    strict: bool = False,
    preferred_column: str = "volume_id",
    source_column: str = "_source_idx",
) -> SemanticFingerprint:
    """Fingerprint only the dataframe inputs that are semantically relevant.

    This is intended for pipeline stages that progressively enrich one mutable
    cohort CSV. Columns written by unrelated downstream stages do not affect the
    fingerprint, while relevant row changes can be detected independently.
    """
    work = ensure_source_id_column(
        df.copy(),
        preferred_column=preferred_column,
        source_column=source_column,
    )

    requested = [str(column) for column in columns if str(column)]
    dynamic = sorted(
        column
        for column in work.columns
        if any(str(column).startswith(prefix) for prefix in dynamic_prefixes)
    )
    selected_columns = list(dict.fromkeys([*requested, *dynamic]))
    missing_columns = [column for column in requested if column not in work.columns]
    present_columns = [
        column
        for column in selected_columns
        if column in work.columns and column != source_column
    ]

    artifact_names = set(str(column) for column in artifact_columns)
    artifact_names.update(
        column
        for column in present_columns
        if any(column.startswith(prefix) for prefix in artifact_prefixes)
    )

    row_fingerprints: dict[str, str] = {}
    ordered_rows: list[tuple[str, str]] = []
    for _, row in work.iterrows():
        source_id = normalize_source_id(row.get(source_column))
        values: dict[str, Any] = {}
        for column in present_columns:
            value = row.get(column)
            normalized_value = (
                _fingerprint_artifact_value(value, strict=strict)
                if column in artifact_names
                else _normalize_semantic_value(value)
            )
            # A missing column and a present-but-null cell are semantically the
            # same for one row. Omitting nulls keeps row-level invalidation
            # selective when a new optional mask/metadata column is introduced.
            if normalized_value is not None:
                values[column] = normalized_value
        payload = {
            "source_id": source_id,
            "values": values,
        }
        blob = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        row_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        row_fingerprints[source_id] = row_hash
        ordered_rows.append((source_id, row_hash))

    dataset_blob = json.dumps(
        ordered_rows,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    dataset_hash = hashlib.sha256(dataset_blob.encode("utf-8")).hexdigest()
    return SemanticFingerprint(
        input_fingerprint={
            "kind": "semantic_dataframe",
            "schema": 1,
            "columns": present_columns,
            "missing_columns": missing_columns,
            "dynamic_prefixes": list(dynamic_prefixes),
            "artifact_columns": sorted(artifact_names),
            "strict_artifacts": bool(strict),
            "row_count": int(len(work)),
            "dataset_sha256": dataset_hash,
        },
        row_fingerprints=row_fingerprints,
    )


def fingerprint_csv_semantic(
    inputs: str | Path | Sequence[str | Path],
    *,
    columns: Sequence[str] = (),
    dynamic_prefixes: Sequence[str] = (),
    artifact_columns: Sequence[str] = (),
    artifact_prefixes: Sequence[str] = (),
    strict: bool = False,
    preferred_column: str = "volume_id",
) -> SemanticFingerprint:
    """Read one or more cohort CSVs and compute a task-specific fingerprint."""
    paths = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
    frames = [pd.read_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return fingerprint_dataframe_semantic(
        df,
        columns=columns,
        dynamic_prefixes=dynamic_prefixes,
        artifact_columns=artifact_columns,
        artifact_prefixes=artifact_prefixes,
        strict=strict,
        preferred_column=preferred_column,
    )


def matching_completed_indices(
    state: Mapping[str, Any] | None,
    row_fingerprints: Mapping[str, str],
) -> set[str]:
    """Return completed rows whose semantic inputs still match saved state."""
    if not state:
        return set()
    saved = state.get("completed_fingerprints")
    if not isinstance(saved, Mapping):
        return set()
    completed = normalize_source_ids(state.get("completed_indices", []))
    return {
        source_id
        for source_id in completed
        if source_id in row_fingerprints
        and str(saved.get(source_id, "")) == str(row_fingerprints[source_id])
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    p = Path(path)
    blob = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode(
        "utf-8"
    )
    _atomic_write_bytes(p, blob)


def atomic_write_csv(
    df: pd.DataFrame, path: str | Path, *, index: bool = False
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, index=index)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, p)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def load_state(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            return value
    except Exception:
        return None
    return None


def task_state_matches(
    state: Mapping[str, Any] | None,
    *,
    command: str,
    args_hash: str,
) -> bool:
    if not state:
        return False
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        return False
    return state.get("command") == command and state.get("args_hash") == args_hash


def state_matches(
    state: Mapping[str, Any] | None,
    *,
    command: str,
    args_hash: str,
    input_fingerprint: Any,
) -> bool:
    return task_state_matches(
        state,
        command=command,
        args_hash=args_hash,
    ) and state.get("input_fingerprint") == input_fingerprint


def now_epoch() -> float:
    return float(time.time())


def normalize_source_id(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def normalize_source_ids(values: Iterable[Any]) -> set[str]:
    return {
        source_id for source_id in (normalize_source_id(v) for v in values) if source_id
    }


def ensure_source_id_column(
    df: pd.DataFrame,
    *,
    preferred_column: str = "volume_id",
    source_column: str = "_source_idx",
) -> pd.DataFrame:
    fallback = pd.Series(df.index, index=df.index).map(normalize_source_id)

    if preferred_column in df.columns:
        preferred = df[preferred_column].map(normalize_source_id)
        df[source_column] = preferred.where(preferred != "", fallback)
        return df

    if source_column in df.columns:
        existing = df[source_column].map(normalize_source_id)
        df[source_column] = existing.where(existing != "", fallback)
        return df

    df[source_column] = fallback
    return df


def source_id_resume_signature(
    inputs: str | Path | Sequence[str | Path] | None,
    *,
    preferred_column: str = "volume_id",
) -> dict[str, Any] | None:
    if inputs is None:
        return None
    paths = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
    for path in paths:
        try:
            columns = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        if preferred_column in columns:
            return {
                "source_id_schema": 1,
                "preferred_source_id_column": preferred_column,
            }
    return None


def prepare_resume_context(
    *,
    args: Any,
    command: str,
    inputs: str | Path | Sequence[str | Path] | None,
    output_path: str | Path,
    error_path: str | Path,
    exclude_hash_args: Iterable[str] = (),
    input_fingerprint: Any | None = None,
    row_fingerprints: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    paths = build_checkpoint_paths(output_path, error_path, command)
    hash_exclude_keys = tuple(
        sorted(set(DEFAULT_HASH_EXCLUDE_KEYS).union(set(exclude_hash_args)))
    )
    args_hash = compute_args_hash(args, exclude_keys=hash_exclude_keys)
    input_fp = (
        fingerprint_inputs(inputs, strict=bool(getattr(args, "strict_resume", False)))
        if input_fingerprint is None
        else input_fingerprint
    )
    state = load_state(paths.state_path)
    resume_enabled = bool(getattr(args, "resume", False))
    task_is_compatible = resume_enabled and task_state_matches(
        state,
        command=command,
        args_hash=args_hash,
    )
    state_is_compatible = task_is_compatible and state_matches(
        state,
        command=command,
        args_hash=args_hash,
        input_fingerprint=input_fp,
    )
    finished_state = state_is_compatible and bool((state or {}).get("finished"))
    output_exists = Path(output_path).exists()
    checkpoint_exists = paths.main_checkpoint_path.exists()
    already_finished = finished_state and output_exists
    can_resume = state_is_compatible and checkpoint_exists
    can_partial_resume = (
        task_is_compatible
        and not state_is_compatible
        and bool((state or {}).get("finished"))
        and output_exists
        and row_fingerprints is not None
        and isinstance((state or {}).get("completed_fingerprints"), Mapping)
    )
    return {
        "paths": paths,
        "state": state,
        "can_resume": can_resume,
        "can_partial_resume": can_partial_resume,
        "already_finished": already_finished,
        "config": CheckpointConfig(
            command=command,
            args_hash=args_hash,
            input_fingerprint=input_fp,
            checkpoint_every_rows=max(
                1, int(getattr(args, "checkpoint_every_rows", 1))
            ),
            checkpoint_every_sec=max(1, int(getattr(args, "checkpoint_every_sec", 1))),
            resume_enabled=resume_enabled,
            row_fingerprints=(
                dict(row_fingerprints) if row_fingerprints is not None else None
            ),
        ),
    }


def _is_safe_unique_key(df: pd.DataFrame, key: str) -> bool:
    if key not in df.columns:
        return False
    series = df[key]
    if series.empty:
        return True
    non_null = series.dropna()
    try:
        return bool(non_null.is_unique)
    except TypeError:
        # Some columns (e.g. list-valued dicom_path) contain unhashable values.
        # Treat them as unsafe merge keys so callers can try the next key/fallback path.
        return False


def merge_with_existing_output(
    new_df: pd.DataFrame,
    output_path: str | Path,
    preferred_keys: Sequence[str],
    *,
    strict: bool = True,
) -> pd.DataFrame:
    p = Path(output_path)
    if not p.exists():
        return new_df

    existing_df = pd.read_csv(p)
    if existing_df.empty or new_df.empty:
        return new_df

    foreign_columns = [c for c in existing_df.columns if c not in new_df.columns]
    if not foreign_columns:
        return new_df

    merged_df = new_df.copy()
    for key in preferred_keys:
        if (
            key in merged_df.columns
            and key in existing_df.columns
            and _is_safe_unique_key(merged_df, key)
            and _is_safe_unique_key(existing_df, key)
        ):
            right = existing_df[[key, *foreign_columns]].copy()
            return merged_df.merge(right, on=key, how="left")

    if len(existing_df) == len(merged_df):
        for col in foreign_columns:
            merged_df[col] = existing_df[col].values
        return merged_df

    if strict:
        tried = ", ".join(preferred_keys) if preferred_keys else "(none)"
        raise ValueError(
            "Cannot safely preserve existing output columns while writing "
            f"{p}: no unique shared key matched among [{tried}] and row counts differ "
            f"(new={len(merged_df)}, existing={len(existing_df)})."
        )

    return merged_df


class CheckpointManager:
    def __init__(self, *, paths: CheckpointPaths, config: CheckpointConfig) -> None:
        self.paths = paths
        self.config = config
        self._processed_since_checkpoint = 0
        self._last_checkpoint_time = now_epoch()

    def mark_processed(self, amount: int = 1) -> None:
        self._processed_since_checkpoint += max(0, int(amount))

    def should_flush(self, *, force: bool = False) -> bool:
        if force:
            return True
        elapsed = now_epoch() - self._last_checkpoint_time
        return (
            self._processed_since_checkpoint >= self.config.checkpoint_every_rows
            or elapsed >= self.config.checkpoint_every_sec
        )

    def _build_state_payload(
        self,
        *,
        completed_indices: Iterable[Any],
        finished: bool,
        extra_state: Mapping[str, Any] | None,
        input_fingerprint: Any | None = None,
        row_fingerprints: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        completed = sorted(normalize_source_ids(completed_indices))
        payload: dict[str, Any] = {
            "schema_version": STATE_SCHEMA_VERSION,
            "command": self.config.command,
            "args_hash": self.config.args_hash,
            "input_fingerprint": (
                self.config.input_fingerprint
                if input_fingerprint is None
                else input_fingerprint
            ),
            "completed_indices": completed,
            "updated_at_epoch": now_epoch(),
        }
        fingerprints = (
            self.config.row_fingerprints
            if row_fingerprints is None
            else dict(row_fingerprints)
        )
        if fingerprints is not None:
            payload["completed_fingerprints"] = {
                source_id: str(fingerprints[source_id])
                for source_id in completed
                if source_id in fingerprints
            }
        if finished:
            payload["finished"] = True
        if extra_state:
            payload.update(dict(extra_state))
        return payload

    def flush(
        self,
        *,
        main_df: pd.DataFrame,
        error_df: pd.DataFrame | None,
        completed_indices: Iterable[Any],
        force: bool = False,
        extra_state: Mapping[str, Any] | None = None,
    ) -> bool:
        if not self.should_flush(force=force):
            return False

        atomic_write_csv(main_df, self.paths.main_checkpoint_path, index=False)
        if error_df is not None and not error_df.empty:
            atomic_write_csv(error_df, self.paths.error_checkpoint_path, index=False)
        elif self.paths.error_checkpoint_path.exists():
            self.paths.error_checkpoint_path.unlink()
        atomic_write_json(
            self.paths.state_path,
            self._build_state_payload(
                completed_indices=completed_indices,
                finished=False,
                extra_state=extra_state,
            ),
        )
        self._processed_since_checkpoint = 0
        self._last_checkpoint_time = now_epoch()
        return True

    def finalize_state(
        self,
        *,
        completed_indices: Iterable[Any],
        extra_state: Mapping[str, Any] | None = None,
        input_fingerprint: Any | None = None,
        row_fingerprints: Mapping[str, str] | None = None,
    ) -> None:
        """Mark a run complete, optionally recording a refreshed input snapshot.

        A refreshed fingerprint matters for commands that overwrite their input CSV:
        the file on disk after a successful run is then the correct resume baseline.
        """
        atomic_write_json(
            self.paths.state_path,
            self._build_state_payload(
                completed_indices=completed_indices,
                finished=True,
                extra_state=extra_state,
                input_fingerprint=input_fingerprint,
                row_fingerprints=row_fingerprints,
            ),
        )
