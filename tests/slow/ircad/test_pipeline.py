"""Full-pipeline test for the small IRCAD cohort."""

from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

assert_full_pipeline_outputs = import_module(
    "tests.slow.assertions"
).assert_full_pipeline_outputs

DATASET_DIR = Path(__file__).resolve().parent
RUNNER = DATASET_DIR.parent / "run_pipeline.py"


@pytest.mark.slow
def test_full_ircad_pipeline(tmp_path, dataset_input):
    input_dir = dataset_input(DATASET_DIR, "IMPERANDI_IRCAD_INPUT")
    work_dir = tmp_path / "work"
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(DATASET_DIR),
            "--input-dir",
            str(input_dir),
            "--work-dir",
            str(work_dir),
        ],
        check=True,
    )
    assert_full_pipeline_outputs(work_dir)


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-m", "slow", "-s"]))
