"""Miscellaneous helper routines for reporting and diagnostics.

The definitions in this module are part of the Imperandi codebase and are
intended to be reused by higher-level workflows and CLI entry points.
"""

import argparse
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def print_args(args: argparse.Namespace) -> None:
    """Print arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments namespace.
    """
    items = vars(args)
    width = max(len(k) for k in items)

    for arg in sorted(items):
        value = items[arg]
        value_str = "None" if value is None else repr(value)
        print(f"{arg:<{width}} : {value_str}")


def report_volumes(df, step_name=None):
    """Perform volumes.

    Args:
        df (Any): Input pandas DataFrame to process.
        step_name (Optional[Any]): Input value for step name. Defaults to `None`.
    """
    unique_counts = df[["patient_key", "study_id", "series_id"]].nunique()
    if step_name:
        logger.info("After %s:", step_name)
    logger.info("Unique patients: %s", unique_counts["patient_key"])
    logger.info("Unique studies: %s", unique_counts["study_id"])
    logger.info("Unique series: %s", unique_counts["series_id"])

    if "volume_id" in df.columns:
        unique_volumes = df["volume_id"].nunique()
        logger.info("Unique volumes: %s", unique_volumes)


def report_change(df, previous_df, col=None):
    """Report change.

    Args:
        df: Input pandas DataFrame to process.
        previous_df: Input pandas DataFrame to process.
        col: Column name(s) used during tabular transformations.
    """
    prev_patients = set(previous_df["patient_key"].unique())
    curr_patients = set(df["patient_key"].unique())
    missing_patients = sorted(prev_patients - curr_patients)
    if missing_patients:
        logger.warning(
            "⚠️  %s patients removed in this step: %s",
            len(missing_patients),
            missing_patients,
        )
        if col is not None:
            logger.info("%s :", col)
            logger.info(
                "%s",
                previous_df[previous_df["patient_key"].isin(missing_patients)][
                    col
                ].value_counts(dropna=False),
            )

    prev_studies = set(previous_df["study_id"].unique())
    curr_studies = set(df["study_id"].unique())
    missing_studies = sorted(prev_studies - curr_studies)
    if missing_studies:
        missing_df = previous_df[previous_df["study_id"].isin(missing_studies)]
        if "date" in missing_df.columns:
            missing_df["date_str"] = pd.to_datetime(missing_df["date"]).dt.strftime(
                "%Y-%m-%d"
            )
            missing_info = missing_df[["patient_key", "date_str"]].drop_duplicates()
            missing_list = list(missing_info.itertuples(index=False, name=None))
            logger.warning(
                "⚠️  %s studies removed in this step: %s",
                len(missing_list),
                missing_list,
            )
        if col is not None:
            logger.info("%s :", col)
            logger.info("%s", missing_df[col].value_counts(dropna=False))
