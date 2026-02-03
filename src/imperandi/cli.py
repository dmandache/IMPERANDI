"""Command-line interface for imperandi ingest workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pandarallel import pandarallel

from imperandi.ingest import clean as clean_module
from imperandi.ingest import parse as parse_module
from imperandi.utils.manifest import load_manifest

def _add_parse_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "parse",
        help="Parse DICOM headers into an index CSV.",
    )
    parse_module.add_parse_arguments(parser)
    parser.set_defaults(_handler=_handle_parse)


def _add_clean_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "clean",
        help="Clean DICOM metadata CSVs.",
    )
    clean_module.add_clean_arguments(parser)
    parser.set_defaults(_handler=_handle_clean)


def _add_ingest_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "ingest",
        help="Run parse then clean in a single step.",
    )
    parse_module.add_parse_arguments(parser, include_manifest=True)
    clean_module.add_clean_arguments(
        parser,
        include_manifest=False,
        include_csv_path=False,
        include_csv_path_out=False,
        include_dry_run=False,
    )
    parser.add_argument(
        "--csv_path_out",
        "--csv-path-out",
        dest="csv_path_out",
        type=str,
        default=None,
        help=(
            "Optional path to save the cleaned CSV file. "
            "Defaults to <output_dir>/dicom_index_clean.csv."
        ),
    )
    parser.set_defaults(_handler=_handle_ingest)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imperandi",
        description="IMPERANDI CLI for ingest parsing and cleaning.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_parse_subcommand(subparsers)
    _add_clean_subcommand(subparsers)
    _add_ingest_subcommand(subparsers)

    return parser


def _handle_parse(args: argparse.Namespace) -> int:
    if args.dry_run:
        print("Dry run: parse")
        print(f"  root_path={args.root_path}")
        print(f"  output_dir={args.output_dir}")
        print(f"  manifest={args.manifest}")
        print(f"  flatten_all_tags={args.flatten_all_tags}")
        print(f"  tags={args.tags}")
        print(f"  force_dicom_read={args.force_dicom_read}")
        print(f"  id_source={args.id_source}")
        print(f"  patient_key_from={args.patient_key_from}")
        print(f"  study_id_from={args.study_id_from}")
        print(f"  series_id_from={args.series_id_from}")
        print(f"  checkpoint_frequency={args.checkpoint_frequency}")
        print(f"  num_workers={args.num_workers}")
        print(f"  verbose={args.verbose}")
        return 0
    pandarallel.initialize(progress_bar=args.verbose, nb_workers=args.num_workers)
    parse_module.main(args)
    return 0


def _handle_clean(args: argparse.Namespace) -> int:
    if args.dry_run:
        print("Dry run: clean")
        print(f"  csv_path={args.csv_path}")
        print(f"  csv_path_out={args.csv_path_out}")
        print(f"  csv_dict_path={args.csv_dict_path}")
        print(f"  manifest={args.manifest}")
        print(f"  volume_min={args.volume_min}")
        print(f"  volume_max={args.volume_max}")
        return 0
    manifest = load_manifest(args.manifest, base_path=Path(__file__).resolve().parents[1])
    clean_module.clean_and_save_data(
        args.csv_path,
        args.csv_path_out,
        args.csv_dict_path,
        manifest,
        args.volume_min,
        args.volume_max,
    )
    return 0


def _handle_ingest(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    parsed_csv = output_dir / "dicom_index.csv"
    clean_out = (
        Path(args.csv_path_out)
        if args.csv_path_out
        else output_dir / "dicom_index_clean.csv"
    )
    if args.dry_run:
        print("Dry run: ingest (parse -> clean)")
        print(f"  parse.root_path={args.root_path}")
        print(f"  parse.output_dir={args.output_dir}")
        print(f"  parse.manifest={args.manifest}")
        print(f"  parse.flatten_all_tags={args.flatten_all_tags}")
        print(f"  parse.tags={args.tags}")
        print(f"  parse.force_dicom_read={args.force_dicom_read}")
        print(f"  parse.id_source={args.id_source}")
        print(f"  parse.patient_key_from={args.patient_key_from}")
        print(f"  parse.study_id_from={args.study_id_from}")
        print(f"  parse.series_id_from={args.series_id_from}")
        print(f"  parse.checkpoint_frequency={args.checkpoint_frequency}")
        print(f"  parse.num_workers={args.num_workers}")
        print(f"  parse.verbose={args.verbose}")
        print(f"  clean.csv_path=[{parsed_csv}]")
        print(f"  clean.csv_path_out={clean_out}")
        print(f"  clean.csv_dict_path={args.csv_dict_path}")
        print(f"  clean.manifest={args.manifest}")
        print(f"  clean.volume_min={args.volume_min}")
        print(f"  clean.volume_max={args.volume_max}")
        return 0
    pandarallel.initialize(progress_bar=args.verbose, nb_workers=args.num_workers)
    parse_module.main(args)
    manifest = load_manifest(args.manifest, base_path=Path(__file__).resolve().parents[1])

    clean_module.clean_and_save_data(
        [str(parsed_csv)],
        str(clean_out),
        args.csv_dict_path,
        manifest,
        args.volume_min,
        args.volume_max,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args._handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
