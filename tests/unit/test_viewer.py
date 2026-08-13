import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import ipywidgets as widgets
import nibabel as nib
import numpy as np
import pandas as pd

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from imperandi.qc import viewer as viewer_module
from imperandi.qc.viewer import CTScanViewer, clip_hu_values, load_nifti


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


@pytest.fixture
def viewer(monkeypatch):
    """A fully initialized viewer with deterministic in-memory image data."""
    viewer_module.plt.switch_backend("Agg")
    scans = {
        "scan-1": np.arange(24, dtype=float).reshape(2, 3, 4),
        "scan-2": np.full((2, 3, 4), 20.0),
        "mask-1": np.array(
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            ]
        ),
        "mask-2": np.zeros((2, 3, 4)),
    }

    def fake_load(path, **_kwargs):
        return scans[path]

    monkeypatch.setattr(viewer_module, "load_nifti", fake_load)
    monkeypatch.setattr(viewer_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(viewer_module, "clear_output", lambda **_kwargs: None)
    monkeypatch.setattr(viewer_module, "display", lambda *_widgets: None)
    monkeypatch.setattr(CTScanViewer, "_try_enable_widget_backend", lambda _self: None)

    frame = pd.DataFrame(
        [
            {
                "patient_key": "P1",
                "date": "2024-01-01",
                "phase": "portal",
                "scan": "scan-1",
                "mask_liver": "mask-1",
                "SeriesDescription": "Baseline",
            },
            {
                "patient_key": "P2",
                "date": "2024-02-02",
                "phase": "arterial",
                "scan": "scan-2",
                "mask_liver": "mask-2",
            },
        ]
    )
    instance = CTScanViewer(frame, "scan", isotropic_resolution_mm=2.0)
    yield instance
    plt = viewer_module.plt
    plt.close(instance.fig)


def test_viewer_initialization_loads_data_and_builds_jump_controls(viewer):
    assert viewer.segmentation_cols == ["mask_liver"]
    assert viewer.seg_colormaps == {"mask_liver": "jet"}
    assert viewer.ct_scan_raw.shape == (2, 3, 4)
    assert viewer.segmentations["mask_liver"].shape == (2, 3, 4)
    assert viewer.patient_dropdown.options == (("P1", "P1"), ("P2", "P2"))
    assert viewer.date_dropdown.options == (("2024-01-01", "2024-01-01"),)
    assert viewer.phase_dropdown.options == (("portal", "portal"),)
    assert "Baseline" in viewer.info_display.value
    assert viewer.progress_bar.layout.visibility == "hidden"


def test_viewer_auto_detects_legacy_segmentations_and_ignores_empty_values(monkeypatch):
    monkeypatch.setattr(CTScanViewer, "init_widgets", lambda _self: None)
    monkeypatch.setattr(CTScanViewer, "load_data", lambda _self: None)
    frame = pd.DataFrame(
        {
            "scan": ["scan-1"],
            "liver_path": ["liver-mask"],
            "liver_tumor_path": [None],
            "totalseg_phase": ["venous"],
        }
    )

    instance = CTScanViewer(frame, "scan", exploration_mode="random")

    assert instance.phase_col == "totalseg_phase"
    assert instance.segmentation_cols == ["liver_path"]
    assert instance.explored_history == [0]
    assert instance.history_index == 0


def test_viewer_helpers_format_filter_and_select_values(viewer):
    assert clip_hu_values(np.array([-200, 0, 500]), -100, 400).tolist() == [
        -100,
        0,
        400,
    ]
    assert viewer._option_values([("P1", "id-1"), "P2"]) == ["id-1", "P2"]
    assert viewer._format_date(None) == "?"
    assert viewer._format_date(pd.NA) == "?"
    assert viewer._format_date(pd.Timestamp("2024-05-06")) == "2024-05-06"
    assert viewer._format_date("not-a-date") == "not-a-date"
    assert viewer._format_value(None) == ""
    assert viewer._format_value(pd.NA) == ""
    assert viewer._format_value(np.float64(1.2349)) == "1.235"
    assert viewer._format_value(pd.Timestamp("2024-05-06")) == "2024-05-06"
    assert viewer._is_empty_value([])
    assert viewer._is_empty_value({})
    assert not viewer._is_empty_value(["mask"])

    assert viewer._build_options_for_column(None, viewer._format_value) == [
        ("N/A", None)
    ]
    assert viewer._filter_frame_for_jump("P2", "2024-02-02").index.tolist() == [1]
    assert viewer._build_date_options("P2") == [("2024-02-02", "2024-02-02")]
    assert viewer._build_phase_options("P1", "2024-01-01") == [("portal", "portal")]


def test_dropdown_stepping_and_refresh_preserve_selected_row(viewer):
    viewer.patient_dropdown.value = "P2"
    viewer._step_dropdown(viewer.patient_dropdown, 1, wrap=True)
    assert viewer.patient_dropdown.value == "P1"
    viewer._step_dropdown(viewer.patient_dropdown, -1, wrap=True)
    assert viewer.patient_dropdown.value == "P2"

    viewer.date_dropdown.value = None
    viewer._step_dropdown(viewer.date_dropdown, -1, wrap=True)
    assert viewer.date_dropdown.value == "2024-02-02"
    viewer._refresh_jump_dropdowns(use_current_row=True)
    assert viewer.patient_dropdown.value == "P2"
    assert viewer.date_dropdown.value == "2024-02-02"
    assert viewer.phase_dropdown.value == "arterial"

    viewer._set_dropdown_options(viewer.phase_dropdown, [], disabled=False)
    assert viewer.phase_dropdown.value is None
    assert viewer.phase_dropdown.disabled is False
    viewer._set_dropdown_options(viewer.phase_dropdown, [("N/A", None)])
    assert viewer.phase_dropdown.disabled is True


def test_jump_handlers_load_the_row_matching_all_selected_filters(viewer, monkeypatch):
    calls = []
    monkeypatch.setattr(viewer, "load_data", lambda: calls.append(viewer.current_index))

    viewer.patient_dropdown.value = "P2"
    viewer.date_dropdown.value = "2024-02-02"
    viewer.phase_dropdown.value = "arterial"
    viewer._jump_to_selected_filters()

    assert viewer.current_index == 1
    assert calls == [1]
    viewer._jump_to_selected_filters()
    assert calls == [1]

    viewer._suspend_jump = True
    viewer.on_patient_change({})
    viewer.on_date_change({})
    viewer.on_phase_change({})
    assert calls == [1]


@pytest.mark.parametrize(
    ("plane", "expected_slice", "expected_center"),
    [
        ("axial", 2, 1),
        ("sagittal", 0, 1),
        ("coronal", 1, 1),
    ],
)
def test_slice_controls_and_lesion_centering_follow_plane(
    viewer, plane, expected_slice, expected_center
):
    seg = np.zeros((2, 3, 4))
    if plane == "axial":
        seg[:, :, expected_center] = 1
    elif plane == "sagittal":
        seg[expected_center, :, :] = 1
    else:
        seg[:, expected_center, :] = 1
    viewer.segmentations = {"mask_liver": seg}
    viewer.plane_selector.value = plane
    viewer.update_slice_slider()

    assert viewer.slice_slider.value == expected_center
    assert viewer._compute_center_slice(seg) == expected_center
    viewer.on_center_on_lesion(None)
    assert viewer.slice_slider.value == expected_center

    viewer.slice_slider.value = expected_slice
    viewer.on_prev_slice(None)
    assert viewer.slice_slider.value == max(0, expected_slice - 3)
    viewer.on_next_slice_manual(None)
    assert viewer.slice_slider.value <= viewer.slice_slider.max


def test_update_display_handles_all_planes_visibility_and_output_fallback(
    viewer, monkeypatch
):
    for plane in ("axial", "sagittal", "coronal"):
        viewer.plane_selector.value = plane
        viewer.update_slice_slider()
        viewer.update_display()
        assert viewer.ax.images

    viewer.seg_visibility["mask_liver"].value = False
    viewer.update_display()
    assert viewer.ax.get_legend() is None

    calls = []
    viewer._uses_output_fallback = True
    monkeypatch.setattr(viewer, "_render_output_figure", lambda: calls.append(True))
    viewer.update_display()
    assert calls == [True]


def test_load_data_skips_missing_masks_and_reports_failed_mask_load(
    viewer, monkeypatch, capsys
):
    viewer.df = pd.DataFrame(
        [
            {
                "patient_key": "P3",
                "date": "2024-03-03",
                "phase": "delayed",
                "scan": "scan",
                "mask_none": None,
                "mask_bad": "bad-mask",
            }
        ]
    )
    viewer.current_index = 0
    viewer.segmentation_cols = ["mask_none", "mask_bad"]
    viewer.seg_visibility = {}

    def fake_load(path, **_kwargs):
        if path == "bad-mask":
            raise OSError("not readable")
        return np.zeros((2, 3, 4))

    monkeypatch.setattr(viewer_module, "load_nifti", fake_load)
    viewer.load_data()

    assert viewer.segmentations == {}
    assert "failed to load segmentation mask_bad" in capsys.readouterr().out


def test_window_resolution_and_keyboard_handlers(viewer, monkeypatch):
    displayed = []
    loaded = []
    monkeypatch.setattr(viewer, "update_display", lambda: displayed.append(True))
    monkeypatch.setattr(viewer, "load_data", lambda: loaded.append(True))

    viewer.on_window_preset_change({"new": "Custom"})
    assert displayed == []
    viewer.on_window_preset_change({"new": "Liver"})
    assert (viewer.HU_min, viewer.HU_max) == (-15.0, 135.0)
    assert displayed == [True]
    viewer.on_resolution_change({"new": 1.5})
    assert viewer.isotropic_resolution_mm == 1.5
    assert loaded == [True]

    actions = []
    monkeypatch.setattr(viewer, "on_prev", lambda _button: actions.append("prev"))
    monkeypatch.setattr(viewer, "on_next", lambda _button: actions.append("next"))
    monkeypatch.setattr(
        viewer, "on_prev_slice", lambda _button: actions.append("prev-slice")
    )
    monkeypatch.setattr(
        viewer, "on_next_slice_manual", lambda _button: actions.append("next-slice")
    )
    for key in ("shift+left", "shift+down", "left", "down", None, "home"):
        viewer.on_key_press(SimpleNamespace(key=key))

    assert actions == ["prev", "next", "prev-slice", "next-slice"]


def test_scan_navigation_uses_ordered_and_random_history(monkeypatch):
    instance = CTScanViewer.__new__(CTScanViewer)
    instance.df = pd.DataFrame({"scan": ["a", "b", "c"]})
    instance.current_index = 2
    instance.exploration_mode = "ordered"
    calls = []
    instance.load_data = lambda: calls.append(instance.current_index)

    instance.on_next(None)
    instance.on_prev(None)
    assert calls == [0, 2]

    instance.exploration_mode = "random"
    instance.current_index = 0
    instance.explored_history = [0]
    instance.history_index = 0
    monkeypatch.setattr(np.random, "choice", lambda values: list(values)[0])
    instance.on_next(None)
    assert (
        instance.current_index,
        instance.explored_history,
        instance.history_index,
    ) == (
        1,
        [0, 1],
        1,
    )
    instance.on_prev(None)
    assert instance.current_index == 0
    instance.on_next(None)
    assert instance.current_index == 1


def test_random_previous_at_history_start_prints_message(capsys):
    instance = CTScanViewer.__new__(CTScanViewer)
    instance.exploration_mode = "random"
    instance.history_index = 0

    instance.on_prev(None)

    assert "Already at the first explored scan" in capsys.readouterr().out
