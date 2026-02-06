from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from pandarallel import pandarallel

from imperandi.ingest import clean as clean_module
from imperandi.ingest import parse as parse_module
from imperandi.process import convert as convert_module
from imperandi.utils.manifest import load_manifest
from imperandi.utils.misc import print_args


def _load_segment_module():
    try:
        from imperandi.process import segment as segment_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'segment' command requires optional dependencies. "
            "Install with: pip install -e .[segment]"
        ) from exc
    return segment_module


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


def _add_convert_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "convert",
        help="Convert DICOM series to NIfTI files.",
    )
    convert_module.add_convert_arguments(parser)
    parser.set_defaults(_handler=_handle_convert)


def _add_segment_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "segment",
        help="Segment NIfTI volumes with configurable tasks.",
    )
    try:
        segment_module = _load_segment_module()
    except RuntimeError as exc:
        parser.add_argument("segment_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
        parser.set_defaults(
            _handler=_handle_segment_unavailable,
            _segment_unavailable_msg=str(exc),
        )
        return

    segment_module.add_segment_arguments(parser)
    parser.set_defaults(_handler=_handle_segment)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imperandi",
        description="IMPERANDI CLI for ingest parsing and cleaning.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_parse_subcommand(subparsers)
    _add_clean_subcommand(subparsers)
    _add_ingest_subcommand(subparsers)
    _add_convert_subcommand(subparsers)
    _add_segment_subcommand(subparsers)

    return parser


def _handle_parse(args: argparse.Namespace) -> int:
    args = parse_module.normalize_parse_args(args)
    if args.dry_run:
        print("Dry run: parse")
        print_args(args)
        return 0
    pandarallel.initialize(progress_bar=args.verbose, nb_workers=args.num_workers)
    parse_module.main(args)
    return 0


def _handle_clean(args: argparse.Namespace) -> int:
    args = clean_module.normalize_clean_args(args)
    if args.dry_run:
        print("Dry run: clean")
        print_args(args)
        return 0
    manifest = load_manifest(
        args.manifest, base_path=Path(__file__).resolve().parents[0]
    )
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
    args = parse_module.normalize_parse_args(args)
    output_dir = Path(args.output_dir)
    parsed_csv = output_dir / "dicom_index.csv"
    clean_out = (
        Path(args.csv_path_out)
        if args.csv_path_out
        else output_dir / "dicom_index_clean.csv"
    )
    if args.dry_run:
        print("Dry run: ingest (parse -> clean)")
        print_args(args)
        return 0
    pandarallel.initialize(progress_bar=args.verbose, nb_workers=args.num_workers)
    parse_module.main(args)
    manifest = load_manifest(
        args.manifest, base_path=Path(__file__).resolve().parents[0]
    )

    clean_module.clean_and_save_data(
        [str(parsed_csv)],
        str(clean_out),
        args.csv_dict_path,
        manifest,
        args.volume_min,
        args.volume_max,
    )
    return 0


def _handle_convert(args: argparse.Namespace) -> int:
    args = convert_module.normalize_convert_args(args)
    if args.dry_run:
        print("Dry run: convert")
        print_args(args)
        return 0
    convert_module.main(args)
    return 0


def _handle_segment(args: argparse.Namespace) -> int:
    segment_module = _load_segment_module()
    args = segment_module.normalize_segment_args(args)
    if args.dry_run:
        print("Dry run: segment")
        print_args(args)
        return 0
    segment_module.main(args)
    return 0


def _handle_segment_unavailable(args: argparse.Namespace) -> int:
    print(getattr(args, "_segment_unavailable_msg", "Segment command unavailable."))
    return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args._handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
