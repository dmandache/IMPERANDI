import sys
from pathlib import Path

import pytest
import ipywidgets as widgets
import numpy as np
import pandas as pd

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imperandi.qc.viewer import CTScanViewer


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


def test_display_to_voxel_and_ct_coordinates_axial():
    viewer = CTScanViewer.__new__(CTScanViewer)
    viewer.view_plane = "axial"
    viewer.ct_scan_raw = np.zeros((8, 9, 10))
    viewer.slice_slider = widgets.IntSlider(value=4)
    viewer.ct_affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    voxel = viewer._display_to_voxel(6.0, 5.0)
    assert voxel.tolist() == [5.0, 6.0, 4.0]

    coord = viewer._voxel_to_ct_coordinate(voxel)
    assert coord.tolist() == [20.0, 38.0, 46.0]


def test_record_annotation_persists_in_dataframe():
    viewer = CTScanViewer.__new__(CTScanViewer)
    viewer.view_plane = "axial"
    viewer.ct_scan_raw = np.zeros((10, 10, 10))
    viewer.slice_slider = widgets.IntSlider(value=3)
    viewer.ct_affine = np.eye(4)
    viewer.current_index = 0
    viewer.df = pd.DataFrame([{"patient_key": "p1"}])
    viewer.annotations_current_scan = []
    viewer.annotation_summary = widgets.HTML(value="")
    viewer.update_display = lambda *_: None

    viewer._record_annotation("bounding_box", [(1.0, 2.0), (4.0, 5.0)])

    stored = viewer.df.at[0, "annotations"]
    assert isinstance(stored, list)
    assert len(stored) == 1
    ann = stored[0]
    assert ann["label"] == "tumor"
    assert ann["shape"] == "bounding_box"
    assert ann["slice_idx"] == 3
    assert ann["voxel_points"][0] == [2.0, 1.0, 3.0]
