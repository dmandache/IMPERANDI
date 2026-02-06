from __future__ import annotations

import os
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_dataset_root(env_var: str, fallback_rel: str, label: str) -> Path:
    env_val = os.getenv(env_var)
    if env_val:
        root = Path(env_val).expanduser()
    else:
        root = _repo_root() / fallback_rel

    if not root.exists() or not root.is_dir():
        pytest.skip(f"{label} not found at {root}. Set {env_var} to enable.")

    try:
        next(root.iterdir())
    except StopIteration:
        pytest.skip(f"{label} directory is empty at {root}.")

    return root


@pytest.fixture(scope="session")
def ircad_dicom_root() -> Path:
    return _resolve_dataset_root(
        "IRCAD_ROOT", Path("tests") / "data" / "IRCAD_DICOM", "IRCAD DICOM dataset"
    )


@pytest.fixture(scope="session")
def ircad_nifti_root() -> Path:
    return _resolve_dataset_root(
        "IRCAD_NIFTI_ROOT", Path("tests") / "data" / "IRCAD_nifti", "IRCAD NIfTI dataset"
    )


@pytest.fixture
def ircad_reference_csv():
    def _get(filename: str) -> Path:
        path = _repo_root() / "tests" / "data" / filename
        if not path.exists():
            pytest.skip(f"Reference CSV missing: {path}")
        return path

    return _get
