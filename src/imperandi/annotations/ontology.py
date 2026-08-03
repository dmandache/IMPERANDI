"""Deterministic composite-key ontology mapping."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from imperandi.config.models import CLINICAL_SLOTS, CONTRAST_PHASES, OntologyConfig
from imperandi.io.tables import read_table

VOCABULARIES = {
    "contrast_phase": CONTRAST_PHASES,
    "clinical_slot": CLINICAL_SLOTS,
}


def _normalized_text(value) -> str:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        values = value.tolist() if isinstance(value, np.ndarray) else list(value)
        if isinstance(value, set):
            values = sorted(values, key=str)
        normalized = [_normalized_text(item) for item in values]
        normalized = [item for item in normalized if item]
        return " | ".join(dict.fromkeys(normalized))
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value.strip().lower())


def _key_value(value, match: str):
    if isinstance(value, (list, tuple, set, np.ndarray)):
        values = value.tolist() if isinstance(value, np.ndarray) else list(value)
        if isinstance(value, set):
            values = sorted(values, key=str)
        normalized = [_key_value(item, match) for item in values]
        normalized = [item for item in normalized if item != ""]
        if len(normalized) == 1:
            return normalized[0]
        return tuple(normalized)
    if match == "exact":
        return "" if value is None or pd.isna(value) else str(value)
    if match == "normalized_exact":
        return _normalized_text(value)
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(numeric) else float(numeric)


def _normalized_keys(frame: pd.DataFrame, config: OntologyConfig) -> pd.DataFrame:
    missing = set(config.keys) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Ontology {config.id!r} requires missing key columns: {sorted(missing)}"
        )
    keys = pd.DataFrame(index=frame.index)
    for column, key_config in config.keys.items():
        match = key_config.match
        keys[f"_key_{column}"] = frame[column].apply(
            lambda value, match=match: _key_value(value, match)
        )
    return keys


def _validate_vocabulary(values: pd.Series, config: OntologyConfig) -> None:
    vocabulary_name = config.output.vocabulary
    if not vocabulary_name:
        return
    if vocabulary_name not in VOCABULARIES:
        raise ValueError(f"Unknown ontology vocabulary: {vocabulary_name}")
    values = set(values.dropna().astype(str))
    invalid = values - VOCABULARIES[vocabulary_name]
    if invalid:
        raise ValueError(
            f"Ontology {config.id!r} contains values outside {vocabulary_name}: "
            f"{sorted(invalid)}"
        )


def apply_ontology(df: pd.DataFrame, config: OntologyConfig) -> pd.DataFrame:
    ontology = read_table(config.source)
    value_column = config.output.value_column
    target_column = config.output.target_column
    if value_column not in ontology.columns:
        raise ValueError(
            f"Ontology {config.id!r} is missing value column {value_column!r}"
        )
    _validate_vocabulary(ontology[value_column], config)

    left_keys = _normalized_keys(df, config)
    right_keys = _normalized_keys(ontology, config)
    key_columns = list(left_keys.columns)
    lookup = right_keys.copy()
    lookup["_ontology_value"] = ontology[value_column].values
    lookup["_ontology_row"] = ontology.index.astype(str)

    conflicts = lookup.groupby(key_columns, dropna=False)["_ontology_value"].nunique()
    conflicting_keys = conflicts[conflicts > 1]
    if not conflicting_keys.empty and config.conflicts == "error":
        raise ValueError(f"Ontology {config.id!r} contains conflicting composite keys")
    if conflicting_keys.empty:
        lookup["_ontology_key_conflict"] = False
    else:
        conflict_lookup = conflicting_keys.rename("_count").reset_index()
        conflict_lookup["_ontology_key_conflict"] = True
        lookup = lookup.merge(
            conflict_lookup[[*key_columns, "_ontology_key_conflict"]],
            on=key_columns,
            how="left",
        )
        lookup["_ontology_key_conflict"] = lookup["_ontology_key_conflict"].fillna(
            False
        )
    lookup = lookup.drop_duplicates(key_columns, keep="first")

    left = left_keys.copy()
    left["_row_order"] = range(len(left))
    mapped = left.merge(lookup, on=key_columns, how="left", sort=False).sort_values(
        "_row_order"
    )
    mapped.index = df.index

    if config.unmatched == "error" and mapped["_ontology_value"].isna().any():
        count = int(mapped["_ontology_value"].isna().sum())
        raise ValueError(f"Ontology {config.id!r} left {count} row(s) unmatched")

    out = df.copy()
    existing = out.get(target_column, pd.Series(pd.NA, index=out.index))
    incoming = mapped["_ontology_value"]
    ontology_id_column = f"{target_column}_ontology_id"
    ontology_row_column = f"{target_column}_ontology_row"
    conflict_column = f"{target_column}_conflict"
    existing_ontology_id = out.get(
        ontology_id_column, pd.Series(pd.NA, index=out.index)
    )
    existing_ontology_row = out.get(
        ontology_row_column, pd.Series(pd.NA, index=out.index)
    )
    previous_conflict = out.get(
        conflict_column, pd.Series(False, index=out.index)
    ).fillna(False)
    disagreement = existing.notna() & incoming.notna() & existing.ne(incoming)
    if disagreement.any() and config.conflicts == "error":
        raise ValueError(
            f"Ontology {config.id!r} conflicts with existing {target_column!r} values"
        )
    if config.conflicts == "first":
        out[target_column] = existing.fillna(incoming)
        selected_incoming = existing.isna() & incoming.notna()
    else:
        out[target_column] = incoming.combine_first(existing)
        selected_incoming = incoming.notna()
    out[ontology_id_column] = existing_ontology_id
    out[ontology_row_column] = existing_ontology_row
    out.loc[selected_incoming, ontology_id_column] = config.id
    out.loc[selected_incoming, ontology_row_column] = mapped.loc[
        selected_incoming, "_ontology_row"
    ]
    out[conflict_column] = (
        previous_conflict
        | disagreement
        | mapped["_ontology_key_conflict"].fillna(False)
    )
    return out


def apply_ontologies(
    df: pd.DataFrame, ontologies: list[OntologyConfig]
) -> pd.DataFrame:
    out = df.copy()
    for ontology in ontologies:
        out = apply_ontology(out, ontology)
    return out
