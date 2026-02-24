from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


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
    return (
        state.get("command") == command
        and state.get("args_hash") == args_hash
        and state.get("input_fingerprint") == input_fingerprint
    )


def now_epoch() -> float:
    return float(time.time())

