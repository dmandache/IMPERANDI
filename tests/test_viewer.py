import sys
from pathlib import Path

import pytest
import ipywidgets as widgets
import nibabel as nib
import numpy as np

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imperandi.qc.viewer import CTScanViewer, load_nifti


def _build_viewer(patient_values, date_values, patient_value, date_value):
    viewer = CTScanViewer.__new__(CTScanViewer)
    viewer.patient_dropdown = widgets.Dropdown(
        options=[(value, value) for value in patient_values],
        value=patient_value,
    )
    viewer.date_dropdown = widgets.Dropdown(
        options=[(value, value) for value in date_values],
        value=date_value,
    )
    viewer.prev_patient_button = widgets.Button()
    viewer.next_patient_button = widgets.Button()
    viewer.prev_date_button = widgets.Button()
    viewer.next_date_button = widgets.Button()
    return viewer


@pytest.mark.parametrize(
    ("current_exam", "expected_prev_disabled", "expected_next_disabled"),
    [
        ("2024-01-01", True, False),
        ("2024-02-01", False, False),
        ("2024-03-01", False, True),
    ],
)
def test_exam_nav_buttons_disabled_at_edges(
    current_exam, expected_prev_disabled, expected_next_disabled
):
    viewer = _build_viewer(
        patient_values=["P1", "P2"],
        date_values=["2024-01-01", "2024-02-01", "2024-03-01"],
        patient_value="P1",
        date_value=current_exam,
    )

    viewer._update_jump_nav_buttons()

    assert viewer.prev_date_button.disabled is expected_prev_disabled
    assert viewer.next_date_button.disabled is expected_next_disabled
    assert viewer.prev_patient_button.disabled is False
    assert viewer.next_patient_button.disabled is False


def test_exam_nav_buttons_disabled_when_single_exam():
    viewer = _build_viewer(
        patient_values=["P1"],
        date_values=["2024-01-01"],
        patient_value="P1",
        date_value="2024-01-01",
    )

    viewer._update_jump_nav_buttons()

    assert viewer.prev_date_button.disabled is True
    assert viewer.next_date_button.disabled is True
    assert viewer.prev_patient_button.disabled is True
    assert viewer.next_patient_button.disabled is True


def test_load_nifti_resamples_to_default_one_mm_isotropic(tmp_path):
    path = tmp_path / "scan.nii.gz"
    data = np.arange(8, dtype=np.float32).reshape((2, 2, 2))
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    nib.save(nib.Nifti1Image(data, affine), path)

    loaded = load_nifti(path, orientation="RAS")

    assert loaded.shape == (4, 6, 8)


def test_load_nifti_uses_configurable_isotropic_resolution(tmp_path):
    path = tmp_path / "scan.nii.gz"
    data = np.arange(8, dtype=np.float32).reshape((2, 2, 2))
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    nib.save(nib.Nifti1Image(data, affine), path)

    loaded = load_nifti(path, orientation="RAS", isotropic_resolution_mm=2.0)

    assert loaded.shape == (2, 3, 4)
