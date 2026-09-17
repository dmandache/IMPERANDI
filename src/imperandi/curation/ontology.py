"""External ontology tables and complete, normalized composite-key lookups."""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

KeyPart = str | Decimal
OntologyKey = tuple[KeyPart, ...]


def normalize_key_part(value: Any) -> KeyPart | None:
    """Normalize scalar keys; a varying grouped value cannot identify a volume."""
    if isinstance(value, (list, tuple, set)):
        parts = {normalize_key_part(item) for item in value}
        return parts.pop() if len(parts) == 1 else None
    if not pd.api.types.is_scalar(value) or pd.isna(value):
        return None
    text = str(value).strip().casefold()
    if not text:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    return number if number.is_finite() else None


def ontology_key(values: Iterable[Any]) -> OntologyKey | None:
    parts: list[KeyPart] = []
    for value in values:
        part = normalize_key_part(value)
        if part is None:
            return None
        parts.append(part)
    return tuple(parts)


@dataclass(frozen=True)
class OntologyIndex:
    columns: tuple[str, ...]
    phases: Mapping[OntologyKey, str]

    def validate_cohort(self, df: pd.DataFrame, *, name: str) -> None:
        missing = sorted(set(self.columns) - set(df.columns))
        if missing:
            raise ValueError(
                f"Ontology strategy {name!r} match_columns missing from cohort: {missing}."
            )

    def match(self, row: Mapping[str, Any]) -> str | None:
        key = ontology_key(row[column] for column in self.columns)
        return self.phases.get(key) if key is not None else None


def _read_table(path: Path, *, field: str) -> pd.DataFrame:
    if path.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError(f"{field} must use a .csv or .parquet extension.")
    if not path.is_file():
        raise ValueError(f"{field}: ontology file {path.name!r} not found or not a file.")
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        # Check every record: pandas can interpret extra fields as an index or
        # silently pad short rows. Neither is appropriate for an ontology.
        with path.open(encoding="utf-8-sig", newline="") as handle:
            records = csv.reader(handle, strict=True)
            header = next(records)
            if not header or len(set(header)) != len(header):
                raise ValueError("CSV requires unique column names")
            rows = []
            for record in records:
                if record and len(record) != len(header):
                    raise ValueError(f"CSV record {records.line_num} has the wrong width")
                if record:
                    rows.append(record)
        # Keep text and numeric precision intact; key normalization handles
        # numeric equivalence. Empty fields, rather than strings like 'NA', are null.
        return pd.DataFrame(rows, columns=header)
    except ImportError as exc:
        raise ValueError(
            f"{field}: Parquet requires an optional engine; install pyarrow "
            "with `pip install pyarrow` (or install fastparquet)."
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"{field}: cannot read ontology {path.name!r}; malformed or unreadable "
            f"{path.suffix.lower()} table ({type(exc).__name__})."
        ) from exc


def load_ontology(
    strategy: Mapping[str, Any],
    *,
    normalize_phase: Callable[[Any], str | None],
    progress_logger: logging.Logger,
) -> OntologyIndex:
    name = strategy["name"]
    field = f"phase_curation ontology {name!r}"
    path = Path(strategy["source"])
    if not path.is_absolute():
        raise ValueError(
            f"{field}.source must be resolved with resolve_phase_curation(manifest) "
            "before loading a relative resource."
        )
    table = _read_table(path, field=f"{field}.source")
    columns = tuple(strategy["match_columns"])
    value_column = strategy["value_column"]
    for option, required in (("match_columns", columns), ("value_column", [value_column])):
        missing = sorted(set(required) - set(table.columns))
        if missing:
            raise ValueError(f"{field}.{option} missing from ontology table: {missing}.")
    if not table.columns.is_unique:
        raise ValueError(f"{field}.source must contain unique column names.")

    mapping = strategy.get("mapping")
    phases: dict[OntologyKey, str] = {}
    for row_number, values in enumerate(
        table[list(columns) + [value_column]].itertuples(index=False, name=None), start=1
    ):
        raw = values[-1]
        phase = None
        if isinstance(raw, str):
            phase = (
                mapping.get(raw.strip().casefold())
                if mapping is not None
                else normalize_phase(raw)
            )
        if phase is None:
            raise ValueError(
                f"{field}.value_column {value_column!r}: invalid phase at data row "
                f"{row_number} after mapping or normalization."
            )
        key = ontology_key(values[:-1])
        if key is None:
            continue
        if key in phases and phases[key] != phase:
            raise ValueError(
                f"{field}.match_columns: duplicate ontology key with conflicting "
                f"phase values at data row {row_number}."
            )
        phases[key] = phase
    progress_logger.info(
        "Ontology strategy %s: source from %s; loaded %d row(s)",
        name,
        strategy.get("_source_origin", "manifest"),
        len(table),
    )
    return OntologyIndex(columns=columns, phases=phases)
