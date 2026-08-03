"""User-facing CLI for project-based IMPERANDI pipelines."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import ValidationError

from imperandi.config import config_hash, load_config, resolved_config
from imperandi.pipeline.defaults import build_default_runner
from imperandi.utils.logging import setup_logging

logger = logging.getLogger(__name__)


STARTER_CONFIG = """version: 1

project:
  name: my-cohort
  profile: liver_ct_mri

input:
  sources:
    - /path/to/dicom

output:
  root: ./imperandi-results
  table_format: parquet
  publish_formats: [parquet, csv]

identity:
  source:
    patient_id_columns: [PatientID]
    namespace_columns: [site_id, IssuerOfPatientID]
    fallback:
      columns: []
      on_missing: error
  canonical:
    strategy: source

phase_prediction:
  enabled: false

conversion:
  enabled: false

segmentation:
  enabled: false

registration:
  enabled: false

radiomics:
  enabled: false

execution:
  workers: 4
  resume: true
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imperandi",
        description="Build traceable analysis-ready CT/MR imaging cohorts.",
    )
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--quiet", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a starter project YAML.")
    init_parser.add_argument("path", nargs="?", default="imperandi.yaml")
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(_handler=_handle_init)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate and resolve a project YAML."
    )
    validate_parser.add_argument("config")
    validate_parser.set_defaults(_handler=_handle_validate)

    plan_parser = subparsers.add_parser(
        "plan", help="Show the resolved stage plan without executing it."
    )
    plan_parser.add_argument("config")
    plan_parser.set_defaults(_handler=_handle_plan)

    run_parser = subparsers.add_parser("run", help="Execute a project pipeline.")
    run_parser.add_argument("config")
    run_parser.set_defaults(_handler=_handle_run)

    config_parser = subparsers.add_parser("config", help="Configuration utilities.")
    config_subparsers = config_parser.add_subparsers(
        dest="config_command", required=True
    )
    resolve_parser = config_subparsers.add_parser(
        "resolve", help="Print the fully resolved configuration."
    )
    resolve_parser.add_argument("config")
    resolve_parser.set_defaults(_handler=_handle_config_resolve)

    status_parser = subparsers.add_parser("status", help="Show stage states for a run.")
    status_parser.add_argument("run_dir")
    status_parser.set_defaults(_handler=_handle_status)
    return parser


def _handle_init(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser()
    if path.exists() and not args.force:
        raise FileExistsError(f"Configuration already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STARTER_CONFIG, encoding="utf-8")
    print(f"Created {path}")
    return 0


def _validated(path: str):
    return load_config(path)


def _handle_validate(args: argparse.Namespace) -> int:
    config = _validated(args.config)
    build_default_runner(config)
    print(f"Configuration is valid (sha256:{config_hash(config)})")
    return 0


def _handle_plan(args: argparse.Namespace) -> int:
    config = _validated(args.config)
    runner = build_default_runner(config)
    print(
        yaml.safe_dump(
            {"config_hash": config_hash(config), "stages": runner.plan()},
            sort_keys=False,
        )
    )
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    config = _validated(args.config)
    setup_logging(level=args.log_level or config.execution.log_level)
    runner = build_default_runner(config)
    results = runner.run()
    cohort = results["11_publish"].artifacts["cohort_index"]
    print(f"Pipeline completed: {cohort}")
    return 0


def _handle_config_resolve(args: argparse.Namespace) -> int:
    config = _validated(args.config)
    print(yaml.safe_dump(resolved_config(config), sort_keys=False))
    return 0


def _handle_status(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser()
    states = []
    for path in sorted(run_dir.glob("*/stage.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        states.append({"stage": path.parent.name, **state})
    if not states:
        raise FileNotFoundError(f"No stage state files found under {run_dir}")
    print(yaml.safe_dump(states, sort_keys=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        setup_logging(
            level=args.log_level,
            quiet=args.quiet,
            log_file=args.log_file,
        )
        return args._handler(args)
    except (OSError, TypeError, ValueError, RuntimeError, ValidationError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
