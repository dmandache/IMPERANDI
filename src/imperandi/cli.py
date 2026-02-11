"""Command-line interface wiring for Imperandi ingestion, extraction, and processing tasks.

The definitions in this module are part of the Imperandi codebase and are
intended to be reused by higher-level workflows and CLI entry points.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

try:
    from pandarallel import pandarallel
except ModuleNotFoundError:
    pandarallel = None

from imperandi.ingest import clean as clean_module
from imperandi.ingest import parse as parse_module
from imperandi.process import convert as convert_module
from imperandi.utils.logging import setup_logging
from imperandi.utils.manifest import load_manifest
from imperandi.utils.misc import print_args

logger = logging.getLogger(__name__)


def _load_phase_module():
    """Lazily import and return the `phase` module.
    
    Returns:
        Any: Loaded object returned by this routine.
    """
    from imperandi.extract import phase as phase_module

    return phase_module


def _load_radiomics_module():
    """Lazily import and return the `radiomics` module.
    
    Returns:
        Any: Loaded object returned by this routine.
    """
    from imperandi.extract import radiomics as radiomics_module

    return radiomics_module


def _load_segment_module():
    """Lazily import and return the `segment` module.
    
    Returns:
        Any: Loaded object returned by this routine.
    
    Raises:
        RuntimeError: If runtime prerequisites or optional dependencies are unavailable.
    """
    try:
        from imperandi.process import segment as segment_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'segment' command requires optional dependencies. "
            "Install with: pip install -e .[segment]"
        ) from exc
    return segment_module


def _ensure_pandarallel() -> None:
    """Ensure runtime prerequisites are satisfied.
    
    Raises:
        RuntimeError: If runtime prerequisites or optional dependencies are unavailable.
    """
    if pandarallel is None:
        raise RuntimeError(
            "The 'parse' and 'ingest' commands require optional dependencies. "
            "Install with: pip install pandarallel"
        )


def _add_parse_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """Register the `parse` subcommand on the root parser.
    
    Args:
        subparsers (argparse._SubParsersAction): Subparser registry used to register command handlers.
    """
    parser = subparsers.add_parser(
        "parse",
        help="Parse DICOM headers into an index CSV.",
    )
    parse_module.add_parse_arguments(parser)
    parser.set_defaults(_handler=_handle_parse)


def _add_clean_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """Register the `clean` subcommand on the root parser.
    
    Args:
        subparsers (argparse._SubParsersAction): Subparser registry used to register command handlers.
    """
    parser = subparsers.add_parser(
        "clean",
        help="Clean DICOM metadata CSVs.",
    )
    clean_module.add_clean_arguments(parser)
    parser.set_defaults(_handler=_handle_clean)


def _add_ingest_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """Register the `ingest` subcommand on the root parser.
    
    Args:
        subparsers (argparse._SubParsersAction): Subparser registry used to register command handlers.
    """
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
    """Register the `convert` subcommand on the root parser.
    
    Args:
        subparsers (argparse._SubParsersAction): Subparser registry used to register command handlers.
    """
    parser = subparsers.add_parser(
        "convert",
        help="Convert DICOM series to NIfTI files.",
    )
    convert_module.add_convert_arguments(parser)
    parser.set_defaults(_handler=_handle_convert)


def _add_phase_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """Register the `phase` subcommand on the root parser.
    
    Args:
        subparsers (argparse._SubParsersAction): Subparser registry used to register command handlers.
    """
    parser = subparsers.add_parser(
        "phase",
        help="Extract contrast phase metadata from NIfTI volumes.",
    )
    phase_module = _load_phase_module()
    phase_module.add_phase_arguments(parser)
    parser.set_defaults(_handler=_handle_phase)


def _add_radiomics_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """Register the `radiomics` subcommand on the root parser.
    
    Args:
        subparsers (argparse._SubParsersAction): Subparser registry used to register command handlers.
    """
    parser = subparsers.add_parser(
        "radiomics",
        help="Extract radiomics features from NIfTI volumes and masks.",
    )
    radiomics_module = _load_radiomics_module()
    radiomics_module.add_radiomics_arguments(parser)
    parser.set_defaults(_handler=_handle_radiomics)


def _add_segment_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """Register the `segment` subcommand on the root parser.
    
    Args:
        subparsers (argparse._SubParsersAction): Subparser registry used to register command handlers.
    """
    parser = subparsers.add_parser(
        "segment",
        help="Segment NIfTI volumes with configurable tasks.",
    )
    try:
        segment_module = _load_segment_module()
    except RuntimeError as exc:
        parser.add_argument(
            "segment_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS
        )
        parser.set_defaults(
            _handler=_handle_segment_unavailable,
            _segment_unavailable_msg=str(exc),
        )
        return

    segment_module.add_segment_arguments(parser)
    parser.set_defaults(_handler=_handle_segment)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser instance.
    """
    parser = argparse.ArgumentParser(
        prog="imperandi",
        description="IMPERANDI CLI for ingest parsing and cleaning.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Set logging level (e.g. DEBUG, INFO, WARNING).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to a log file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Reduce log verbosity (sets WARNING level unless overridden).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_parse_subcommand(subparsers)
    _add_clean_subcommand(subparsers)
    _add_ingest_subcommand(subparsers)
    _add_convert_subcommand(subparsers)
    _add_phase_subcommand(subparsers)
    _add_radiomics_subcommand(subparsers)
    _add_segment_subcommand(subparsers)

    return parser


def _handle_parse(args: argparse.Namespace) -> int:
    """Handle execution of the `parse` CLI command.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments namespace.
    
    Returns:
        int: Process exit status code.
    """
    args = parse_module.normalize_parse_args(args)
    if args.dry_run:
        logger.info("Dry run: parse")
        print_args(args)
        return 0
    try:
        _ensure_pandarallel()
    except RuntimeError as exc:
        logger.error("%s", str(exc))
        return 2
    pandarallel.initialize(progress_bar=args.verbose, nb_workers=args.num_workers)
    parse_module.main(args)
    return 0


def _handle_clean(args: argparse.Namespace) -> int:
    """Handle execution of the `clean` CLI command.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments namespace.
    
    Returns:
        int: Process exit status code.
    """
    args = clean_module.normalize_clean_args(args)
    if args.dry_run:
        logger.info("Dry run: clean")
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
    """Handle execution of the `ingest` CLI command.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments namespace.
    
    Returns:
        int: Process exit status code.
    """
    args = parse_module.normalize_parse_args(args)
    output_dir = Path(args.output_dir)
    parsed_csv = output_dir / "dicom_index.csv"
    clean_out = (
        Path(args.csv_path_out)
        if args.csv_path_out
        else output_dir / "dicom_index_clean.csv"
    )
    if args.dry_run:
        logger.info("Dry run: ingest (parse -> clean)")
        print_args(args)
        return 0
    try:
        _ensure_pandarallel()
    except RuntimeError as exc:
        logger.error("%s", str(exc))
        return 2
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
    """Handle execution of the `convert` CLI command.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments namespace.
    
    Returns:
        int: Process exit status code.
    """
    args = convert_module.normalize_convert_args(args)
    if args.dry_run:
        logger.info("Dry run: convert")
        print_args(args)
        return 0
    convert_module.main(args)
    return 0


def _handle_phase(args: argparse.Namespace) -> int:
    """Handle execution of the `phase` CLI command.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments namespace.
    
    Returns:
        int: Process exit status code.
    """
    phase_module = _load_phase_module()
    args = phase_module.normalize_phase_args(args)
    if args.dry_run:
        logger.info("Dry run: phase")
        print_args(args)
        return 0
    try:
        phase_module.main(args)
    except RuntimeError as exc:
        message = str(exc)
        if "requires optional dependencies" in message:
            logger.error("%s", message)
            return 2
        raise
    return 0


def _handle_radiomics(args: argparse.Namespace) -> int:
    """Handle execution of the `radiomics` CLI command.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments namespace.
    
    Returns:
        int: Process exit status code.
    """
    radiomics_module = _load_radiomics_module()
    args = radiomics_module.normalize_radiomics_args(args)
    if args.dry_run:
        logger.info("Dry run: radiomics")
        print_args(args)
        return 0
    try:
        radiomics_module.main(args)
    except RuntimeError as exc:
        message = str(exc)
        if "requires optional dependencies" in message:
            logger.error("%s", message)
            return 2
        raise
    return 0


def _handle_segment(args: argparse.Namespace) -> int:
    """Handle execution of the `segment` CLI command.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments namespace.
    
    Returns:
        int: Process exit status code.
    """
    segment_module = _load_segment_module()
    args = segment_module.normalize_segment_args(args)
    if args.dry_run:
        logger.info("Dry run: segment")
        print_args(args)
        return 0
    segment_module.main(args)
    return 0


def _handle_segment_unavailable(args: argparse.Namespace) -> int:
    """Handle execution of the `unavailable` CLI command.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments namespace.
    
    Returns:
        int: Process exit status code.
    """
    logger.error(
        "%s", getattr(args, "_segment_unavailable_msg", "Segment command unavailable.")
    )
    return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the module entry point.
    
    Args:
        argv (Optional[Sequence[str]]): Optional argument vector used instead of `sys.argv`. Defaults to `None`.
    
    Returns:
        int: Process exit status code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(
        level=args.log_level,
        verbose=getattr(args, "verbose", False),
        quiet=args.quiet,
        log_file=args.log_file,
    )
    return args._handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
