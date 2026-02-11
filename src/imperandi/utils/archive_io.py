"""Archive traversal and extraction helpers for zipped or tarred imaging data.

The definitions in this module are part of the Imperandi codebase and are
intended to be reused by higher-level workflows and CLI entry points.
"""

from __future__ import annotations

import hashlib
import io
import logging
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import quote, unquote
import tarfile
import zipfile

from pydicom import dcmread
from pydicom.errors import InvalidDicomError

logger = logging.getLogger(__name__)

ARCHIVE_URI_SCHEME = "archive://"
DEFAULT_ARCHIVE_MAX_DEPTH = 3
DEFAULT_ARCHIVE_MEMBER_LIMIT = 200000
ARCHIVE_EXTENSIONS = (".zip", ".tar", ".tar.gz", ".tgz")


def is_archive_filename(name: str) -> bool:
    """Return whether archive filename.

    Args:
        name (str): Input value for name.

    Returns:
        bool: True when the condition is satisfied; otherwise False.
    """
    lower = str(name).lower()
    return lower.endswith(ARCHIVE_EXTENSIONS)


def is_archive_uri(value: str) -> bool:
    """Return whether archive URI.

    Args:
        value (str): Input value to evaluate or transform.

    Returns:
        bool: True when the condition is satisfied; otherwise False.
    """
    return isinstance(value, str) and value.startswith(ARCHIVE_URI_SCHEME)


def _is_zip_name(name: str) -> bool:
    """Return whether zip name.

    Args:
        name (str): Input value for name.

    Returns:
        bool: True when the condition is satisfied; otherwise False.
    """
    return str(name).lower().endswith(".zip")


def _is_tar_name(name: str) -> bool:
    """Return whether tar name.

    Args:
        name (str): Input value for name.

    Returns:
        bool: True when the condition is satisfied; otherwise False.
    """
    lower = str(name).lower()
    return lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz")


def _normalize_member_name(name: str) -> str:
    """Normalize member name.

    Args:
        name (str): Input value for name.

    Returns:
        str: Normalized member name.
    """
    normalized = str(name).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def _is_safe_member_name(name: str) -> bool:
    """Return whether safe member name.

    Args:
        name (str): Input value for name.

    Returns:
        bool: True when the condition is satisfied; otherwise False.
    """
    normalized = str(name).replace("\\", "/")
    if not normalized:
        return False
    if normalized.startswith("/") or normalized.startswith("\\"):
        return False
    if len(normalized) >= 2 and normalized[1] == ":" and normalized[0].isalpha():
        return False
    parts = [p for p in PurePosixPath(normalized).parts if p not in ("", ".")]
    return all(part != ".." for part in parts)


def encode_archive_uri(outer_archive_path: str | Path, entry_chain: list[str]) -> str:
    """Encode archive URI.

    Args:
        outer_archive_path (str | Path): Filesystem path consumed by this operation.
        entry_chain (list[str]): Input value for entry chain.

    Returns:
        str: Encoded archive URI.

    Raises:
        ValueError: If provided inputs fail validation.
    """
    if not entry_chain:
        raise ValueError("entry_chain must contain at least one member name.")
    outer_abs = Path(outer_archive_path).resolve()
    encoded_parts = [quote(str(outer_abs), safe="")]
    encoded_parts.extend(
        quote(_normalize_member_name(name), safe="") for name in entry_chain
    )
    return ARCHIVE_URI_SCHEME + "!".join(encoded_parts)


def decode_archive_uri(uri: str) -> tuple[Path, list[str]]:
    """Decode archive URI.

    Args:
        uri (str): Input value for URI.

    Returns:
        tuple[Path, list[str]]: Resolved filesystem path.

    Raises:
        ValueError: If provided inputs fail validation.
    """
    if not is_archive_uri(uri):
        raise ValueError(f"Not an archive URI: {uri}")
    body = uri[len(ARCHIVE_URI_SCHEME) :]
    if not body:
        raise ValueError("Empty archive URI.")
    parts = body.split("!")
    if len(parts) < 2:
        raise ValueError("Archive URI must include outer path and at least one entry.")
    decoded = [unquote(p) for p in parts]
    outer = Path(decoded[0])
    chain = [_normalize_member_name(name) for name in decoded[1:]]
    return outer, chain


def _open_archive(source: Path | bytes, source_name: str):
    """Perform archive.

    Args:
        source (Path | bytes): Input value for source.
        source_name (str): Input value for source name.

    Returns:
        Any: Result of `_open_archive`.
    """
    if isinstance(source, Path):
        if _is_zip_name(source_name):
            return "zip", zipfile.ZipFile(source)
        if _is_tar_name(source_name):
            return "tar", tarfile.open(source, mode="r:*")
        try:
            return "zip", zipfile.ZipFile(source)
        except zipfile.BadZipFile:
            return "tar", tarfile.open(source, mode="r:*")

    bio = io.BytesIO(source)
    if _is_zip_name(source_name):
        return "zip", zipfile.ZipFile(bio)
    if _is_tar_name(source_name):
        return "tar", tarfile.open(fileobj=bio, mode="r:*")

    try:
        return "zip", zipfile.ZipFile(bio)
    except zipfile.BadZipFile:
        bio.seek(0)
        return "tar", tarfile.open(fileobj=bio, mode="r:*")


def _read_zip_member(zf: zipfile.ZipFile, target_name: str) -> bytes:
    """Read zip member.

    Args:
        zf (zipfile.ZipFile): Input value for zf.
        target_name (str): Input value for target name.

    Returns:
        bytes: Binary payload read from storage.

    Raises:
        KeyError: If required keys are missing from a mapping-like input.
    """
    target = _normalize_member_name(target_name)
    for info in zf.infolist():
        if info.is_dir():
            continue
        normalized = _normalize_member_name(info.filename)
        if normalized == target:
            return zf.read(info.filename)
    raise KeyError(f"Zip member not found: {target_name}")


def _read_tar_member(tf: tarfile.TarFile, target_name: str) -> bytes:
    """Read tar member.

    Args:
        tf (tarfile.TarFile): Input value for tf.
        target_name (str): Input value for target name.

    Returns:
        bytes: Binary payload read from storage.

    Raises:
        KeyError: If required keys are missing from a mapping-like input.
    """
    target = _normalize_member_name(target_name)
    for member in tf.getmembers():
        if not member.isfile():
            continue
        normalized = _normalize_member_name(member.name)
        if normalized != target:
            continue
        fp = tf.extractfile(member)
        if fp is None:
            raise KeyError(f"Tar member could not be extracted: {target_name}")
        return fp.read()
    raise KeyError(f"Tar member not found: {target_name}")


def _read_member_bytes_from_container(
    source: Path | bytes, source_name: str, member_name: str
) -> bytes:
    """Read member bytes from container.

    Args:
        source (Path | bytes): Archive source path or in-memory bytes payload.
        source_name (str): Display name for the current archive source.
        member_name (str): Archive member name to resolve.

    Returns:
        bytes: Binary payload read from storage.
    """
    kind, container = _open_archive(source, source_name)
    try:
        if kind == "zip":
            return _read_zip_member(container, member_name)
        return _read_tar_member(container, member_name)
    finally:
        container.close()


def _iter_container_members(
    source: Path | bytes,
    source_name: str,
    entry_chain: list[str],
    depth: int,
    max_depth: int,
    member_limit: int,
    counter: list[int],
):
    """Iterate over container members.

    Args:
        source (Path | bytes): Input value for source.
        source_name (str): Input value for source name.
        entry_chain (list[str]): Input value for entry chain.
        depth (int): Input value for depth.
        max_depth (int): Input value for max depth.
        member_limit (int): Input value for member limit.
        counter (list[int]): Input value for counter.

    Returns:
        Any: Iterator over container members.

    Raises:
        RuntimeError: If runtime prerequisites or optional dependencies are unavailable.
    """
    kind, container = _open_archive(source, source_name)
    try:
        if kind == "zip":
            members = [(info.filename, info.is_dir()) for info in container.infolist()]
        else:
            members = [
                (member.name, member.isdir()) for member in container.getmembers()
            ]

        for raw_name, is_dir in members:
            counter[0] += 1
            if counter[0] > member_limit:
                raise RuntimeError(
                    f"Archive member limit exceeded ({member_limit}) while reading {source_name}."
                )

            normalized = _normalize_member_name(raw_name)
            if not normalized or is_dir:
                continue

            if not _is_safe_member_name(normalized):
                logger.warning(
                    "[archive][discover] skipping unsafe member name: %s in %s",
                    raw_name,
                    source_name,
                )
                continue

            current_chain = entry_chain + [normalized]
            is_member_archive = is_archive_filename(normalized)
            yield {
                "entry_chain": current_chain,
                "depth": depth,
                "is_archive": is_member_archive,
            }

            if not is_member_archive:
                continue

            if depth >= max_depth:
                logger.warning(
                    "[archive][discover] max depth reached for nested archive member: %s",
                    normalized,
                )
                continue

            try:
                if kind == "zip":
                    nested_bytes = _read_zip_member(container, normalized)
                else:
                    nested_bytes = _read_tar_member(container, normalized)
            except Exception as exc:
                logger.warning(
                    "[archive][discover] unable to read nested archive member %s: %s",
                    normalized,
                    exc,
                )
                continue

            yield from _iter_container_members(
                source=nested_bytes,
                source_name=normalized,
                entry_chain=current_chain,
                depth=depth + 1,
                max_depth=max_depth,
                member_limit=member_limit,
                counter=counter,
            )
    finally:
        container.close()


def iter_archive_members(
    path_or_bytes: Path | bytes,
    depth: int = 0,
    max_depth: int = DEFAULT_ARCHIVE_MAX_DEPTH,
    member_limit: int = DEFAULT_ARCHIVE_MEMBER_LIMIT,
):
    """Iterate over archive members.

    Args:
        path_or_bytes (Path | bytes): Filesystem path consumed by this operation.
        depth (int): Input value for depth. Defaults to `0`.
        max_depth (int): Input value for max depth. Defaults to `DEFAULT_ARCHIVE_MAX_DEPTH`.
        member_limit (int): Input value for member limit. Defaults to `DEFAULT_ARCHIVE_MEMBER_LIMIT`.

    Returns:
        Any: Iterator over archive members.
    """
    if isinstance(path_or_bytes, Path):
        source_name = path_or_bytes.name
    else:
        source_name = "memory_archive"
    counter = [0]
    yield from _iter_container_members(
        source=path_or_bytes,
        source_name=source_name,
        entry_chain=[],
        depth=depth,
        max_depth=max_depth,
        member_limit=member_limit,
        counter=counter,
    )


def _record_sort_key(record: dict[str, Any]) -> tuple[str, str]:
    """Return record sort key.

    Args:
        record (dict[str, Any]): Record dictionary describing one discovered source item.

    Returns:
        tuple[str, str]: Tuple containing outputs from this step.
    """
    return str(record["source_uri_or_path"]), str(record.get("relative_path") or "")


def _record_is_dcm(record: dict[str, Any]) -> bool:
    """Return record is dcm.

    Args:
        record (dict[str, Any]): Record dictionary describing one discovered source item.

    Returns:
        bool: True when the condition is satisfied; otherwise False.
    """
    rel = str(record.get("relative_path", ""))
    src = str(record.get("source_uri_or_path", ""))
    return rel.lower().endswith(".dcm") or src.lower().endswith(".dcm")


def _validate_record_as_dicom(
    record: dict[str, Any], max_depth: int = DEFAULT_ARCHIVE_MAX_DEPTH
) -> bool:
    """Return validate record as DICOM.

    Args:
        record (dict[str, Any]): Record dictionary describing one discovered source item.
        max_depth (int): Maximum nested archive depth permitted during traversal. Defaults to `DEFAULT_ARCHIVE_MAX_DEPTH`.

    Returns:
        bool: True when the condition is satisfied; otherwise False.
    """
    source = record["source_uri_or_path"]
    try:
        if is_archive_uri(source):
            outer, entry_chain = decode_archive_uri(source)
            payload = read_archive_member_bytes(outer, entry_chain, max_depth=max_depth)
            dcmread(io.BytesIO(payload), stop_before_pixels=True, force=True)
            return True

        dcmread(Path(source), stop_before_pixels=True, force=True)
        return True
    except (InvalidDicomError, FileNotFoundError, PermissionError, OSError, ValueError):
        return False
    except Exception:
        return False


def discover_dicom_sources(
    root_entries: Iterable[Path],
    max_depth: int = DEFAULT_ARCHIVE_MAX_DEPTH,
    member_limit: int = DEFAULT_ARCHIVE_MEMBER_LIMIT,
) -> list[dict[str, Any]]:
    """Discover DICOM sources.

    Args:
        root_entries (Iterable[Path]): Input value for root entries.
        max_depth (int): Input value for max depth. Defaults to `DEFAULT_ARCHIVE_MAX_DEPTH`.
        member_limit (int): Input value for member limit. Defaults to `DEFAULT_ARCHIVE_MEMBER_LIMIT`.

    Returns:
        list[dict[str, Any]]: List of computed items.
    """
    dedup: dict[str, dict[str, Any]] = {}

    def _add_record(record: dict[str, Any]) -> None:
        """Add record.

        Args:
            record (dict[str, Any]): Record dictionary describing one discovered source item.
        """
        dedup.setdefault(str(record["source_uri_or_path"]), record)

    for root in sorted({Path(p) for p in root_entries}, key=lambda p: str(p)):
        if not root.exists():
            continue

        if root.is_dir():
            scan_root = str(root)
            for path in sorted(root.rglob("*"), key=lambda p: str(p)):
                if not path.is_file():
                    continue

                if is_archive_filename(path.name):
                    try:
                        members = iter_archive_members(
                            path,
                            depth=0,
                            max_depth=max_depth,
                            member_limit=member_limit,
                        )
                        for member in members:
                            if member["is_archive"]:
                                continue
                            member_chain = member["entry_chain"]
                            source_uri = encode_archive_uri(path, member_chain)
                            _add_record(
                                {
                                    "source_uri_or_path": source_uri,
                                    "scan_root": scan_root,
                                    "relative_path": member_chain[-1],
                                    "is_archive_member": True,
                                }
                            )
                    except Exception as exc:
                        logger.warning(
                            "[archive][discover] unable to inspect archive %s: %s",
                            path,
                            exc,
                        )
                    continue

                _add_record(
                    {
                        "source_uri_or_path": str(path),
                        "scan_root": scan_root,
                        "relative_path": str(path.relative_to(root)),
                        "is_archive_member": False,
                    }
                )
            continue

        if root.is_file() and is_archive_filename(root.name):
            scan_root = str(root.parent)
            try:
                members = iter_archive_members(
                    root,
                    depth=0,
                    max_depth=max_depth,
                    member_limit=member_limit,
                )
                for member in members:
                    if member["is_archive"]:
                        continue
                    member_chain = member["entry_chain"]
                    source_uri = encode_archive_uri(root, member_chain)
                    _add_record(
                        {
                            "source_uri_or_path": source_uri,
                            "scan_root": scan_root,
                            "relative_path": member_chain[-1],
                            "is_archive_member": True,
                        }
                    )
            except Exception as exc:
                logger.warning(
                    "[archive][discover] unable to inspect archive %s: %s",
                    root,
                    exc,
                )
            continue

        if root.is_file():
            _add_record(
                {
                    "source_uri_or_path": str(root),
                    "scan_root": str(root.parent),
                    "relative_path": root.name,
                    "is_archive_member": False,
                }
            )

    if not dedup:
        return []

    all_records = sorted(dedup.values(), key=_record_sort_key)
    dcm_records = [record for record in all_records if _record_is_dcm(record)]
    if dcm_records:
        return dcm_records

    logger.info(
        "[archive][discover] no *.dcm sources found; validating all candidates with dcmread."
    )
    validated = [
        record
        for record in all_records
        if _validate_record_as_dicom(record, max_depth=max_depth)
    ]
    return sorted(validated, key=_record_sort_key)


def read_archive_member_bytes(
    outer_archive_path: Path,
    entry_chain: list[str],
    max_depth: int = DEFAULT_ARCHIVE_MAX_DEPTH,
) -> bytes:
    """Read archive member bytes.

    Args:
        outer_archive_path (Path): Filesystem path consumed by this operation.
        entry_chain (list[str]): Nested archive/member traversal chain.
        max_depth (int): Maximum nested archive depth permitted during traversal. Defaults to `DEFAULT_ARCHIVE_MAX_DEPTH`.

    Returns:
        bytes: Binary payload read from storage.

    Raises:
        ValueError: If provided inputs fail validation.
        FileNotFoundError: If an expected input file cannot be found.
    """
    if not entry_chain:
        raise ValueError("entry_chain cannot be empty.")
    if not outer_archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {outer_archive_path}")
    if not is_archive_filename(outer_archive_path.name):
        raise ValueError(f"Not an archive file: {outer_archive_path}")

    archive_depth = 0
    current_source: Path | bytes = outer_archive_path
    current_name = outer_archive_path.name
    payload = b""

    for i, member_name in enumerate(entry_chain):
        normalized = _normalize_member_name(member_name)
        if not normalized or not _is_safe_member_name(normalized):
            raise ValueError(f"Unsafe archive member path: {member_name}")
        payload = _read_member_bytes_from_container(
            current_source, current_name, normalized
        )
        is_last = i == len(entry_chain) - 1
        if is_last:
            return payload
        if not is_archive_filename(normalized):
            raise ValueError(f"Intermediate member is not an archive: {normalized}")
        archive_depth += 1
        if archive_depth > max_depth:
            raise ValueError(
                f"Maximum nested archive depth exceeded ({max_depth}) while reading {normalized}"
            )
        current_source = payload
        current_name = normalized

    return payload


class ArchiveSession:
    """Class implementing archive session behavior.

    This class groups related state and behavior behind a single interface.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        keep_cache: bool = False,
        max_depth: int = DEFAULT_ARCHIVE_MAX_DEPTH,
    ):
        """Initialize `ArchiveSession` state.

        Args:
            cache_dir (str | Path | None): Directory path used for input or output data. Defaults to `None`.
            keep_cache (bool): Boolean flag controlling optional behavior. Defaults to `False`.
            max_depth (int): Maximum nested archive depth permitted during traversal. Defaults to `DEFAULT_ARCHIVE_MAX_DEPTH`.
        """
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path(tempfile.gettempdir()) / "imperandi_archive"
        )
        self.keep_cache = keep_cache
        self.max_depth = max_depth
        self.session_dir: Path | None = None
        self._materialized: dict[str, Path] = {}

    def __enter__(self):
        """Enter the context manager and return the active session object.

        Returns:
            Any: Context manager instance for chained use.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        session_id = hashlib.sha1(str(id(self)).encode("utf-8")).hexdigest()[:12]
        self.session_dir = self.cache_dir / f"session_{session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit the context manager and release allocated resources.

        Args:
            exc_type (Any): Input value for exc type.
            exc (Any): Input value for exc.
            tb (Any): Input value for tb.
        """
        if self.keep_cache:
            return
        if self.session_dir and self.session_dir.exists():
            shutil.rmtree(self.session_dir, ignore_errors=True)

    def materialize(self, uri_or_path: str | Path) -> Path:
        """Materialize the requested operation.

        Args:
            uri_or_path (str | Path): Filesystem path consumed by this operation.

        Returns:
            Path: Resolved filesystem path.

        Raises:
            RuntimeError: If runtime prerequisites or optional dependencies are unavailable.
        """
        if not is_archive_uri(str(uri_or_path)):
            return Path(uri_or_path)

        uri = str(uri_or_path)
        if uri in self._materialized:
            return self._materialized[uri]

        if self.session_dir is None:
            raise RuntimeError("ArchiveSession must be used as a context manager.")

        outer_path, entry_chain = decode_archive_uri(uri)
        payload = read_archive_member_bytes(
            outer_archive_path=outer_path,
            entry_chain=entry_chain,
            max_depth=self.max_depth,
        )

        leaf_name = Path(entry_chain[-1]).name or "member.dcm"
        key = hashlib.sha1(uri.encode("utf-8")).hexdigest()
        out_dir = self.session_dir / key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / leaf_name
        out_path.write_bytes(payload)

        self._materialized[uri] = out_path
        return out_path
