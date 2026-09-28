"""Generate deterministic external identifier maps for MIDS-style exports."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from .identity import validate_label
from .models import MidsExportError

IDENTITY_COLUMNS = ["patient_key", "study_id", "series_id", "volume_id"]
DEFAULT_SORT_COLUMNS = [
    "date",
    "StudyDate",
    "AcquisitionDate",
    "SeriesDate",
    "time",
    "StudyTime",
    "AcquisitionTime",
    "SeriesTime",
    "visit_order",
    "acquisition_order",
    "volume_ordinal_in_series",
    "SeriesNumber",
    "AcquisitionNumber",
]


def add_id_map_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--csv_path",
        "--csv-path",
        required=True,
        help="Curated IMPERANDI cohort CSV.",
    )
    parser.add_argument(
        "--output_path",
        "--output-path",
        default=None,
        help="External ID-map CSV (default: <input_stem>_mids_id_map.csv).",
    )
    parser.add_argument(
        "--patient-key-mode",
        "--patient_key_mode",
        choices=["map", "keep"],
        default="map",
        help=(
            "Map patient keys using the fixed-width numeric convention, or "
            "keep already-pseudonymized path-safe keys. Default: map."
        ),
    )
    parser.add_argument(
        "--sort-columns",
        "--sort_columns",
        nargs="+",
        default=None,
        help=(
            "Ordering columns in precedence order. Missing columns are ignored. "
            "Defaults to date/time, visit, acquisition, and volume-order fields."
        ),
    )
    parser.add_argument(
        "--minimum-digits",
        "--minimum_digits",
        type=int,
        default=4,
        help="Minimum mapped numeric label width (default: 4, producing 0001).",
    )
    parser.add_argument(
        "--overwrite",
        choices=["never", "replace"],
        default="never",
        help="Existing-map policy (default: never).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Report counts, widths, ordering fields, and output without writing.",
    )


def normalize_id_map_args(args: argparse.Namespace) -> argparse.Namespace:
    csv_path = Path(args.csv_path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Not a CSV file: {csv_path}")
    output = (
        Path(args.output_path).expanduser()
        if args.output_path
        else csv_path.with_name(f"{csv_path.stem}_mids_id_map.csv")
    )
    if output.suffix.lower() != ".csv":
        raise ValueError(f"ID-map output must be a CSV file: {output}")
    if output.resolve() == csv_path.resolve():
        raise MidsExportError("The ID map must not overwrite the source cohort CSV.")
    if args.minimum_digits < 1:
        raise MidsExportError("--minimum-digits must be at least 1.")
    args.csv_path = str(csv_path.resolve())
    args.output_path = str(output.resolve())
    args.sort_columns = list(args.sort_columns or DEFAULT_SORT_COLUMNS)
    return args


def _read_cohort(path: str) -> pd.DataFrame:
    available = list(pd.read_csv(path, nrows=0).columns)
    dtypes = {column: "string" for column in IDENTITY_COLUMNS if column in available}
    return pd.read_csv(path, dtype=dtypes, low_memory=False)


def _require_identities(frame: pd.DataFrame) -> None:
    missing_columns = [column for column in IDENTITY_COLUMNS if column not in frame]
    if missing_columns:
        raise MidsExportError(
            "ID-map generation requires column(s): " + ", ".join(missing_columns)
        )
    missing_values = {
        column: int(frame[column].fillna("").astype("string").str.strip().eq("").sum())
        for column in IDENTITY_COLUMNS
    }
    missing_values = {
        column: count for column, count in missing_values.items() if count
    }
    if missing_values:
        details = ", ".join(
            f"{column}={count}" for column, count in missing_values.items()
        )
        raise MidsExportError(f"Identity columns contain missing values: {details}")


def _sort_value(series: pd.Series, column: str) -> pd.Series:
    folded = column.casefold()
    if "date" in folded:
        parsed = pd.to_datetime(series, errors="coerce")
        return parsed.fillna(pd.Timestamp.max)
    if "time" in folded:
        cleaned = series.astype("string").str.replace(":", "", regex=False)
        numeric = pd.to_numeric(cleaned, errors="coerce").astype("Float64")
        return numeric.fillna(float("inf"))
    if any(token in folded for token in ("order", "number", "ordinal", "index")):
        # pandas may preserve a nullable Int64 dtype here. Convert before using
        # infinity as the missing-last sentinel because Int64 rejects floats.
        numeric = pd.to_numeric(series, errors="coerce").astype("Float64")
        return numeric.fillna(float("inf"))
    return series.astype("string").fillna("\uffff").str.strip().str.casefold()


def _ordered_rows(
    frame: pd.DataFrame, sort_columns: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    available = [column for column in sort_columns if column in frame]
    ordered = frame.copy()
    helper_columns: list[str] = []
    for index, column in enumerate(available):
        helper = f"__mids_sort_{index}"
        ordered[helper] = _sort_value(ordered[column], column)
        helper_columns.append(helper)
    # Raw identifiers are deterministic tie-breakers only; they never become
    # generated labels or exported dataset paths.
    ordered = ordered.sort_values(
        [*helper_columns, *IDENTITY_COLUMNS], kind="mergesort", na_position="last"
    ).reset_index(drop=True)
    return ordered, available


def _width(count: int, minimum_digits: int) -> int:
    return max(minimum_digits, len(str(max(count, 1))))


def _numbered_map(
    ordered: pd.DataFrame,
    keys: list[str],
    *,
    width: int,
) -> dict[tuple[str, ...], str]:
    entities = ordered.drop_duplicates(keys, keep="first")
    return {
        tuple(str(row[column]).strip() for column in keys): f"{number:0{width}d}"
        for number, (_, row) in enumerate(entities.iterrows(), 1)
    }


def generate_id_map(
    frame: pd.DataFrame,
    *,
    patient_key_mode: str = "map",
    sort_columns: list[str] | None = None,
    minimum_digits: int = 4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return an external ID map and a non-identifying generation report."""
    _require_identities(frame)
    if patient_key_mode not in {"map", "keep"}:
        raise MidsExportError("patient_key_mode must be 'map' or 'keep'.")
    if minimum_digits < 1:
        raise MidsExportError("minimum_digits must be at least 1.")
    ordered, used_sort_columns = _ordered_rows(
        frame, list(sort_columns or DEFAULT_SORT_COLUMNS)
    )
    hierarchy = {
        "patient": ["patient_key"],
        "study": ["patient_key", "study_id"],
        "series": ["patient_key", "study_id", "series_id"],
        "volume": IDENTITY_COLUMNS,
    }
    counts = {
        name: len(ordered.drop_duplicates(keys)) for name, keys in hierarchy.items()
    }
    widths = {name: _width(count, minimum_digits) for name, count in counts.items()}
    maps = {
        name: _numbered_map(ordered, keys, width=widths[name])
        for name, keys in hierarchy.items()
    }

    volumes = ordered.drop_duplicates(IDENTITY_COLUMNS, keep="first").copy()
    default_export_keys = ["patient_key", "study_id", "volume_id"]
    ambiguous = volumes.duplicated(default_export_keys, keep=False)
    if ambiguous.any():
        raise MidsExportError(
            "volume_id is not unique within patient_key/study_id. The generated "
            "map would be ambiguous for the default MIDS export identity schema."
        )
    rows: list[dict[str, str]] = []
    for _, row in volumes.iterrows():
        raw = {column: str(row[column]).strip() for column in IDENTITY_COLUMNS}
        patient_key = (raw["patient_key"],)
        study_key = (raw["patient_key"], raw["study_id"])
        series_key = (*study_key, raw["series_id"])
        volume_key = (*series_key, raw["volume_id"])
        if patient_key_mode == "keep":
            participant = validate_label(
                raw["patient_key"],
                field="patient_key in --patient-key-mode keep",
                lowercase=False,
            )
        else:
            participant = maps["patient"][patient_key]
        study = maps["study"][study_key]
        series = maps["series"][series_key]
        volume = maps["volume"][volume_key]
        rows.append(
            {
                **raw,
                "mapped_patient_key": participant,
                "mapped_study_id": study,
                "mapped_series_id": series,
                "mapped_volume_id": volume,
                # These aliases make the file directly consumable by
                # `imperandi export mids --id-map` with its default identity columns.
                "participant_label": participant,
                "session_label": study,
                "image_label": volume,
            }
        )
    result = pd.DataFrame(rows)
    result = result.sort_values(
        [
            "mapped_patient_key",
            "mapped_study_id",
            "mapped_series_id",
            "mapped_volume_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    report = {
        "status": "ready",
        "patient_key_mode": patient_key_mode,
        "counts": counts,
        "digits": widths,
        "sort_columns_used": used_sort_columns,
        "rows": len(result),
    }
    return result, report


def _write_map(frame: pd.DataFrame, output_path: Path, overwrite: str) -> None:
    if output_path.exists() and overwrite == "never":
        raise MidsExportError(
            f"ID-map output already exists: {output_path}. Use --overwrite replace."
        )
    if output_path.exists() and not output_path.is_file():
        raise MidsExportError(f"ID-map output is not a regular file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.imperandi-{uuid.uuid4().hex}.tmp"
    )
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def existing_id_map_status(frame: pd.DataFrame, output_path: Path) -> str:
    """Return missing, identical, conflicting, or invalid for an output path."""
    if not output_path.exists():
        return "missing"
    if not output_path.is_file():
        return "invalid"
    try:
        existing = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    except Exception:
        return "invalid"
    expected = frame.fillna("").astype(str)
    existing = existing.fillna("").astype(str)
    if list(existing.columns) != list(expected.columns):
        return "conflicting"
    return "identical" if existing.equals(expected) else "conflicting"


def write_or_reuse_id_map(
    frame: pd.DataFrame, output_path: Path, *, overwrite: str = "never"
) -> str:
    """Atomically create, reuse an identical map, or explicitly replace it."""
    status = existing_id_map_status(frame, output_path)
    if status == "identical":
        return "reused"
    if status == "invalid":
        raise MidsExportError(
            f"ID-map output is not a readable regular CSV: {output_path}"
        )
    if status == "conflicting" and overwrite == "never":
        raise MidsExportError(
            f"A differing ID map already exists: {output_path}. "
            "Use --id-map-overwrite replace to replace it."
        )
    _write_map(frame, output_path, "replace" if status == "conflicting" else "never")
    return "replaced" if status == "conflicting" else "created"


def run_id_map(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    frame = _read_cohort(args.csv_path)
    id_map, report = generate_id_map(
        frame,
        patient_key_mode=args.patient_key_mode,
        sort_columns=args.sort_columns,
        minimum_digits=args.minimum_digits,
    )
    report["output_path"] = args.output_path
    if args.dry_run:
        report["status"] = (
            "invalid"
            if Path(args.output_path).exists() and args.overwrite == "never"
            else "ready"
        )
        if report["status"] == "invalid":
            report["conflict"] = "output exists and overwrite policy is 'never'"
        print(json.dumps(report, indent=2, sort_keys=True))
        return (2 if report["status"] == "invalid" else 0), report
    _write_map(id_map, Path(args.output_path), args.overwrite)
    return 0, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a protected external ID map for MIDS-style export."
    )
    add_id_map_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = normalize_id_map_args(build_parser().parse_args(argv))
    exit_code, _ = run_id_map(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
