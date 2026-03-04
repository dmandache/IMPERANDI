from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

STATE_SCHEMA_VERSION = 2
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
    input_fingerprint: list[dict[str, Any]]
    checkpoint_every_rows: int
    checkpoint_every_sec: int
    resume_enabled: bool


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


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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
    blob = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")
    _atomic_write_bytes(p, blob)


def atomic_write_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> None:
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


def state_matches(
    state: Mapping[str, Any] | None,
    *,
    command: str,
    args_hash: str,
    input_fingerprint: list[dict[str, Any]],
) -> bool:
    if not state:
        return False
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        return False
    return (
        state.get("command") == command
        and state.get("args_hash") == args_hash
        and state.get("input_fingerprint") == input_fingerprint
    )


def now_epoch() -> float:
    return float(time.time())


def prepare_resume_context(
    *,
    args: Any,
    command: str,
    inputs: str | Path | Sequence[str | Path] | None,
    output_path: str | Path,
    error_path: str | Path,
    exclude_hash_args: Iterable[str] = (),
) -> dict[str, Any]:
    paths = build_checkpoint_paths(output_path, error_path, command)
    hash_exclude_keys = tuple(
        sorted(set(DEFAULT_HASH_EXCLUDE_KEYS).union(set(exclude_hash_args)))
    )
    args_hash = compute_args_hash(args, exclude_keys=hash_exclude_keys)
    input_fp = fingerprint_inputs(
        inputs, strict=bool(getattr(args, "strict_resume", False))
    )
    state = load_state(paths.state_path)
    resume_enabled = bool(getattr(args, "resume", False))
    state_is_compatible = resume_enabled and state_matches(
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
    return {
        "paths": paths,
        "state": state,
        "can_resume": can_resume,
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
        ),
    }


def _is_safe_unique_key(df: pd.DataFrame, key: str) -> bool:
    if key not in df.columns:
        return False
    series = df[key]
    if series.empty:
        return True
    non_null = series.dropna()
    return bool(non_null.is_unique)


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
        completed_indices: Iterable[int],
        finished: bool,
        extra_state: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": STATE_SCHEMA_VERSION,
            "command": self.config.command,
            "args_hash": self.config.args_hash,
            "input_fingerprint": self.config.input_fingerprint,
            "completed_indices": sorted(int(i) for i in completed_indices),
            "updated_at_epoch": now_epoch(),
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
        completed_indices: Iterable[int],
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
        completed_indices: Iterable[int],
        extra_state: Mapping[str, Any] | None = None,
    ) -> None:
        atomic_write_json(
            self.paths.state_path,
            self._build_state_payload(
                completed_indices=completed_indices,
                finished=True,
                extra_state=extra_state,
            ),
        )
