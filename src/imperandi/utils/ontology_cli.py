"""The shared external-ontology option for phase-curation commands."""

import argparse
from pathlib import Path


def add_ontology_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ontology_path",
        type=str,
        default=None,
        help=(
            "CSV or Parquet ontology source overriding the manifest source only. "
            "Relative paths use the current working directory; the manifest must "
            "define ontology match_columns and value_column."
        ),
    )


def normalize_ontology_arg(args: argparse.Namespace) -> argparse.Namespace:
    source = getattr(args, "ontology_path", None)
    if source is not None:
        if not source.strip():
            raise ValueError("--ontology_path must be a non-empty path.")
        path = Path(source)
        args.ontology_path = str(path if path.is_absolute() else path.resolve())
    return args
