"""Fixtures shared by the dataset-backed slow tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def dataset_input():
    """Resolve a dataset input folder, with an environment override."""

    def _resolve(dataset_dir: Path, env_var: str) -> Path:
        configured = os.getenv(env_var)
        input_dir = (
            Path(configured).expanduser().resolve()
            if configured
            else dataset_dir / "data" / "input"
        )
        if not input_dir.is_dir() or not any(
            path.is_file() for path in input_dir.rglob("*")
        ):
            pytest.skip(
                f"Slow-test input is missing at {input_dir}. "
                f"Run {dataset_dir / 'download.py'} or set {env_var}."
            )
        return input_dir

    return _resolve
