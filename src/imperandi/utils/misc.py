import argparse
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def print_args(args: argparse.Namespace) -> None:
    items = vars(args)
    width = max(len(k) for k in items)

    for arg in sorted(items):
        value = items[arg]
        value_str = "None" if value is None else repr(value)
        print(f"{arg:<{width}} : {value_str}")


def report_volumes(df, step_name=None):
    unique_counts = df[["patient_key", "study_id", "series_id"]].nunique()
    if step_name:
        logger.info("After %s:", step_name)
    logger.info("\tUnique patients: %s", unique_counts["patient_key"])
    logger.info("\tUnique studies:  %s", unique_counts["study_id"])
    logger.info("\tUnique series:   %s", unique_counts["series_id"])

    if "volume_id" in df.columns:
        unique_volumes = df["volume_id"].nunique()
        logger.info("\tUnique volumes:  %s", unique_volumes)


def _display_value(value, max_length=40):
    """Format one report-table value without allowing unbounded log lines."""
    try:
        is_missing = pd.isna(value)
        if not hasattr(is_missing, "__iter__") and bool(is_missing):
            return "<NA>"
    except (TypeError, ValueError, AttributeError):
        pass

    text = str(value).replace("\n", " ")
    if len(text) > max_length:
        return f"{text[: max_length - 1]}…"
    return text


def _summarize_values(values, max_values=3):
    """Collapse repeated row-level values into a compact entity-level cell."""
    unique = list(dict.fromkeys(_display_value(value) for value in values))
    shown = unique[:max_values]
    text = ", ".join(shown)
    if len(unique) > max_values:
        text += f", … (+{len(unique) - max_values})"
    return text


def _report_table(rows, *, entity_name, total, max_rows):
    """Log a bounded table and make any truncation explicit."""
    shown = rows[:max_rows]
    if shown:
        table = pd.DataFrame(shown).to_string(index=False)
        suffix = ""
        if total > len(shown):
            suffix = f"\n… {total - len(shown)} more omitted"
        logger.warning(
            "⚠️  %s %s removed in this step (showing %s):\n%s%s",
            total,
            entity_name,
            len(shown),
            table,
            suffix,
        )
    else:
        logger.warning("⚠️  %s %s removed in this step", total, entity_name)


def report_change(df, previous_df, columns=None, max_rows=10, *, col=None):
    """Report patients and studies that disappeared between two dataframes.

    ``columns`` may be a single column name for backwards compatibility or an
    ordered collection of step-relevant columns. Row-level values are collapsed
    to unique, bounded summaries so the output remains useful on DICOM tables.
    """
    if col is not None:
        if columns is not None:
            raise ValueError(
                "Use either 'columns' or the legacy 'col' argument, not both."
            )
        columns = col
    if isinstance(columns, str):
        columns = [columns]
    columns = list(dict.fromkeys(columns or []))
    value_columns = [
        column
        for column in columns
        if column in previous_df.columns
        and column not in {"patient_key", "study_id", "date"}
    ]
    max_rows = max(0, int(max_rows))

    prev_patients = set(previous_df["patient_key"].unique())
    curr_patients = set(df["patient_key"].unique())
    missing_patients = sorted(prev_patients - curr_patients, key=str)
    if missing_patients:
        patient_rows = []
        for patient_key in missing_patients:
            affected = previous_df[previous_df["patient_key"].eq(patient_key)]
            row = {
                "patient_key": _display_value(patient_key),
                "studies": affected["study_id"].nunique(dropna=False),
            }
            for column in value_columns:
                row[column] = _summarize_values(affected[column])
            patient_rows.append(row)
        _report_table(
            patient_rows,
            entity_name="patients",
            total=len(missing_patients),
            max_rows=max_rows,
        )

    prev_studies = set(
        previous_df[["patient_key", "study_id"]].itertuples(index=False, name=None)
    )
    curr_studies = set(
        df[["patient_key", "study_id"]].itertuples(index=False, name=None)
    )
    missing_studies = sorted(
        prev_studies - curr_studies,
        key=lambda key: (str(key[0]), str(key[1])),
    )
    if missing_studies:
        study_rows = []
        for patient_key, study_id in missing_studies:
            affected = previous_df[
                previous_df["patient_key"].eq(patient_key)
                & previous_df["study_id"].eq(study_id)
            ]
            row = {
                "patient_key": _display_value(patient_key),
                "study_id": _display_value(study_id),
            }
            if "date" in affected.columns:
                dates = pd.to_datetime(affected["date"], errors="coerce").dt.strftime(
                    "%Y-%m-%d"
                )
                row["date"] = _summarize_values(dates)
            for column in value_columns:
                row[column] = _summarize_values(affected[column])
            study_rows.append(row)
        _report_table(
            study_rows,
            entity_name="studies",
            total=len(missing_studies),
            max_rows=max_rows,
        )
