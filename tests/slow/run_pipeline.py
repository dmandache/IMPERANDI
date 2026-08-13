"""Run the complete IMPERANDI pipeline for a slow-test dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGES = ("ingest", "convert", "segment", "phase", "radiomics")


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_pipeline(
    dataset_dir: Path,
    *,
    input_dir: Path | None = None,
    work_dir: Path | None = None,
    manifest: str = "generic",
    stop_after: str = "radiomics",
) -> Path:
    """Run all requested stages and return the work directory."""
    dataset_dir = dataset_dir.resolve()
    input_dir = (input_dir or dataset_dir / "data" / "input").resolve()
    work_dir = (work_dir or dataset_dir / "data" / "work").resolve()

    if not input_dir.is_dir() or not any(
        path.is_file() for path in input_dir.rglob("*")
    ):
        raise FileNotFoundError(
            f"No dataset files found in {input_dir}. Run {dataset_dir / 'download.py'} first."
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    imperandi = [python, "-m", "imperandi"]

    commands = {
        "ingest": imperandi
        + [
            "ingest",
            str(input_dir),
            str(work_dir),
            "--manifest",
            manifest,
            "--snapshot_tags",
            "--num_workers",
            "1",
        ],
        "convert": imperandi
        + [
            "convert",
            str(work_dir / "dicom_index_clean.csv"),
            str(work_dir / "NIFTI"),
            "--csv_path_out",
            str(work_dir / "nifti_index.csv"),
            "--manifest",
            manifest,
            "--num_workers",
            "1",
        ],
        "segment": imperandi
        + [
            "segment",
            str(work_dir / "nifti_index.csv"),
            "--manifest",
            manifest,
            "--num_workers",
            "1",
        ],
        "phase": imperandi
        + [
            "phase",
            str(work_dir / "nifti_index.csv"),
            "--manifest",
            manifest,
        ],
        "radiomics": imperandi
        + [
            "radiomics",
            str(work_dir / "nifti_index.csv"),
            str(work_dir / "nifti_index_radiomics.csv"),
            "--manifest",
            manifest,
            "--skip_filter",
        ],
    }

    for stage in STAGES:
        _run(commands[stage])
        if stage == stop_after:
            break
    return work_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--manifest", default="generic")
    parser.add_argument("--stop-after", choices=STAGES, default=STAGES[-1])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_pipeline(
        args.dataset_dir,
        input_dir=args.input_dir,
        work_dir=args.work_dir,
        manifest=args.manifest,
        stop_after=args.stop_after,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
