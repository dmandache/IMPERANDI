import builtins
import importlib.util
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
from imperandi.qc.viewer import CTScanViewer, ScanViewer, clip_hu_values, load_nifti
from imperandi.qc.viewer_windowing import normalize_modality, percentile_window


def test_viewer_can_be_loaded_directly_without_imperandi_import(monkeypatch):
    """The notebook viewer can be copied and loaded outside the package."""
    real_import = builtins.__import__

    def import_without_imperandi(name, *args, **kwargs):
        if name == "imperandi" or name.startswith("imperandi."):
            raise ModuleNotFoundError("No module named 'imperandi'", name="imperandi")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_imperandi)
    viewer_path = Path(viewer_module.__file__).resolve()
    spec = importlib.util.spec_from_file_location("standalone_viewer", viewer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.CTScanViewer is module.ScanViewer


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
    monkeypatch.setattr(viewer_module, "clear_output", lambda **_kwargs: None)
    monkeypatch.setattr(viewer_module, "display", lambda *_widgets: None)
    monkeypatch.setattr(CTScanViewer, "_try_enable_widget_backend", lambda _self: None)

    frame = pd.DataFrame(
        [
            {
                "patient_key": "P1",
                "date": "2024-01-01",
                "phase": "portal",
                "Modality": "CT",
                "scan": "scan-1",
                "mask_liver": "mask-1",
                "SeriesDescription": "Baseline",
            },
            {
                "patient_key": "P2",
                "date": "2024-02-02",
                "phase": "arterial",
                "Modality": "MR",
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
    assert isinstance(viewer, ScanViewer)
    assert viewer.segmentation_cols == ["mask_liver"]
    assert viewer.seg_colormaps == {"mask_liver": "jet"}
    assert viewer.ct_scan_raw.shape == (2, 3, 4)
    assert viewer.segmentations["mask_liver"].shape == (2, 3, 4)
    assert viewer.patient_dropdown.options == (("P1", "P1"), ("P2", "P2"))
    assert viewer.patient_dropdown.disabled is False
    assert viewer.date_dropdown.options == (("2024-01-01", "2024-01-01"),)
    assert viewer.date_dropdown.disabled is True
    assert viewer.phase_dropdown.options == (("portal", "portal"),)
    assert viewer.phase_dropdown.disabled is True
    assert viewer.modality_dropdown.options == (("CT", "CT"),)
    assert viewer.modality_dropdown.disabled is True
    assert viewer.window_preset.disabled is False
    assert viewer.percentile_min_input.disabled is True
    assert viewer.prev_button.layout.width == "50%"
    assert viewer.next_button.layout.width == "50%"
    rendering_inputs = [
        viewer.hu_min_input,
        viewer.hu_max_input,
        viewer.percentile_min_input,
        viewer.percentile_max_input,
    ]
    assert {widget.layout.width for widget in rendering_inputs} == {"50%"}
    assert {widget.style.description_width for widget in rendering_inputs} == {
        "105px"
    }
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


def test_viewer_stably_sorts_rows_for_navigation(monkeypatch):
    monkeypatch.setattr(CTScanViewer, "init_widgets", lambda _self: None)
    monkeypatch.setattr(CTScanViewer, "load_data", lambda _self: None)
    frame = pd.DataFrame(
        [
            {
                "patient_key": "P2",
                "study_id": "S1",
                "date": "2024-01-01",
                "Modality": "CT",
                "phase": "arterial",
                "scan": "p2",
            },
            {
                "patient_key": "P1",
                "study_id": "S2",
                "date": "2024-01-01",
                "Modality": "CT",
                "phase": "portal",
                "scan": "study-2",
            },
            {
                "patient_key": "P1",
                "study_id": "S1",
                "date": "2024-02-01",
                "Modality": "MR",
                "phase": "arterial",
                "scan": "mr",
            },
            {
                "patient_key": "P1",
                "study_id": "S1",
                "date": "2024-02-01",
                "Modality": "CT",
                "phase": "portal",
                "scan": "portal-first",
            },
            {
                "patient_key": "P1",
                "study_id": "S1",
                "date": "2024-02-01",
                "Modality": "CT",
                "phase": "arterial",
                "scan": "arterial",
            },
            {
                "patient_key": "P1",
                "study_id": "S1",
                "date": "2024-02-01",
                "Modality": "CT",
                "phase": "portal",
                "scan": "portal-second",
            },
        ],
        index=[9, 8, 7, 6, 5, 4],
    )

    instance = CTScanViewer(frame, "scan")

    assert instance.df["scan"].tolist() == [
        "study-2",
        "arterial",
        "portal-first",
        "portal-second",
        "mr",
        "p2",
    ]
    assert instance.df.index.tolist() == list(range(len(frame)))
    assert frame.index.tolist() == [9, 8, 7, 6, 5, 4]


def test_viewer_sorts_phases_in_clinical_order_then_preserves_rest(monkeypatch):
    monkeypatch.setattr(CTScanViewer, "init_widgets", lambda _self: None)
    monkeypatch.setattr(CTScanViewer, "load_data", lambda _self: None)
    phases = [
        "OTHER",
        "HEPATOBILIARY",
        "PORTAL_VENOUS",
        "NATIVE",
        "UNKNOWN",
        "DELAYED",
        "ARTERIAL",
    ]
    frame = pd.DataFrame(
        {
            "patient_key": ["P1"] * len(phases),
            "study_id": ["S1"] * len(phases),
            "Modality": ["MR"] * len(phases),
            "phase": phases,
            "scan": phases,
        }
    )

    instance = CTScanViewer(frame, "scan")

    assert instance.df["phase"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
        "HEPATOBILIARY",
        "OTHER",
        "UNKNOWN",
    ]


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


def test_load_data_disables_navigation_while_scan_is_loading(viewer, monkeypatch):
    observed_states = []
    viewer.segmentation_cols = []

    def fake_load(_path, **_kwargs):
        observed_states.append(
            (
                viewer._is_loading,
                viewer.prev_button.disabled,
                viewer.next_button.disabled,
                viewer.patient_dropdown.disabled,
                viewer.phase_dropdown.disabled,
            )
        )
        return np.zeros((2, 3, 4))

    monkeypatch.setattr(viewer_module, "load_nifti", fake_load)

    assert viewer.load_data() is True

    assert observed_states == [(True, True, True, True, True)]
    assert viewer._is_loading is False
    assert viewer.prev_button.disabled is False
    assert viewer.next_button.disabled is False


def test_scan_load_failure_clears_previous_image_and_reports_row(viewer, monkeypatch):
    viewer.df = pd.DataFrame(
        [
            {
                "patient_key": "P3",
                "study_id": "S3",
                "date": "2024-03-03",
                "Modality": "CT",
                "phase": "portal",
                "volume_id": "volume-bad",
                "scan": "missing-scan.nii.gz",
            }
        ]
    )
    viewer.current_index = 0
    viewer.segmentation_cols = []

    def fail_load(_path, **_kwargs):
        raise OSError("not readable")

    monkeypatch.setattr(viewer_module, "load_nifti", fail_load)

    assert viewer.load_data() is False

    assert viewer.current_index == 0
    assert viewer.ct_scan_raw is None
    assert viewer.segmentations == {}
    assert len(viewer.ax.images) == 0
    assert "Failed to load scan" in viewer.info_display.value
    assert "volume-bad" in viewer.info_display.value
    assert "missing-scan.nii.gz" in viewer.info_display.value
    assert "OSError: not readable" in viewer.info_display.value
    assert "Failed to load scan" in viewer.ax.texts[0].get_text()
    assert viewer.progress_bar.bar_style == "danger"
    assert viewer.progress_bar.description == "Load failed"
    assert viewer.prev_button.disabled is False
    assert viewer.next_button.disabled is False


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


def test_mri_uses_default_percentile_window_and_disables_hu_controls(viewer):
    viewer.current_index = 1
    viewer.ct_scan_raw = np.arange(100, dtype=float).reshape(2, 5, 10)
    viewer._update_window_controls()

    assert viewer._current_modality() == "MR"
    assert viewer._display_window() == pytest.approx((0.99, 98.01))
    assert viewer.window_preset.disabled is True
    assert viewer.hu_min_input.disabled is True
    assert viewer.hu_max_input.disabled is True
    assert viewer.percentile_min_input.disabled is False
    assert viewer.percentile_max_input.disabled is False


def test_mri_slices_share_volume_wide_rendering_bounds(viewer):
    viewer.current_index = 1
    viewer.ct_scan_raw = np.arange(100, dtype=float).reshape(2, 5, 10)
    viewer.plane_selector.value = "axial"
    expected_window = percentile_window(viewer.ct_scan_raw, 1, 99)

    observed_windows = []
    for slice_index in (0, 9):
        viewer.slice_slider.value = slice_index
        viewer.update_display()
        observed_windows.append(viewer.ax.images[0].get_clim())

    assert observed_windows == [expected_window, expected_window]


def test_invalid_percentile_edit_reverts_to_rendered_values(viewer):
    viewer.current_index = 1
    viewer._update_window_controls()

    viewer.percentile_min_input.value = 100

    assert viewer.percentile_min_input.value == 1.0
    assert viewer.percentile_max_input.value == 99.0
    assert (viewer.percentile_min, viewer.percentile_max) == (1.0, 99.0)


def test_modality_dropdown_is_enabled_for_mixed_modality_exam(viewer):
    mixed_row = viewer.df.iloc[0].copy()
    mixed_row["Modality"] = "MRI"
    mixed_row["phase"] = "delayed"
    viewer.df = pd.concat([viewer.df, mixed_row.to_frame().T], ignore_index=True)
    viewer.current_index = 0

    viewer._refresh_jump_dropdowns(use_current_row=True)

    assert viewer.modality_dropdown.options == (("CT", "CT"), ("MR", "MR"))
    assert viewer.modality_dropdown.disabled is False
    viewer.modality_dropdown.value = "MR"
    assert viewer._build_phase_options("P1", "2024-01-01") == [
        ("delayed", "delayed")
    ]


def test_single_option_dropdowns_enable_when_alternatives_become_available(viewer):
    second_phase = viewer.df.iloc[0].copy()
    second_phase["phase"] = "delayed"
    second_date = viewer.df.iloc[0].copy()
    second_date["date"] = "2024-01-02"
    viewer.df = pd.concat(
        [viewer.df, second_phase.to_frame().T, second_date.to_frame().T],
        ignore_index=True,
    )
    viewer.current_index = 0

    viewer._refresh_jump_dropdowns(use_current_row=True)

    assert viewer.patient_dropdown.disabled is False
    assert viewer.date_dropdown.disabled is False
    assert viewer.phase_dropdown.disabled is False


def test_modality_and_percentile_helpers_handle_mri_and_nonfinite_voxels():
    assert normalize_modality("mri") == "MR"
    assert normalize_modality(" ct ") == "CT"
    assert normalize_modality(["MRI"]) == "MR"
    values = np.array([0.0, 10.0, np.nan, np.inf])
    assert percentile_window(values, 0, 100) == (0.0, 10.0)
    with pytest.raises(ValueError, match="lower < upper"):
        percentile_window(values, 99, 1)


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


def test_duplicate_phase_scan_navigation_visits_every_row_forward_and_backward(
    viewer, monkeypatch
):
    loaded = []
    viewer.segmentation_cols = []
    frame = pd.DataFrame(
        [
            {
                "patient_key": "P1",
                "date": "2024-01-01",
                "Modality": "CT",
                "phase": "portal",
                "scan": "portal-first",
            },
            {
                "patient_key": "P1",
                "date": "2024-01-01",
                "Modality": "CT",
                "phase": "arterial",
                "scan": "arterial",
            },
            {
                "patient_key": "P1",
                "date": "2024-01-01",
                "Modality": "CT",
                "phase": "portal",
                "scan": "portal-second",
            },
            {
                "patient_key": "P1",
                "date": "2024-01-01",
                "Modality": "CT",
                "phase": "delayed",
                "scan": "delayed",
            },
        ]
    )
    viewer.df = viewer._sort_dataframe_for_navigation(frame)

    def fake_load(path, **_kwargs):
        loaded.append(path)
        return np.zeros((2, 3, 4))

    monkeypatch.setattr(viewer_module, "load_nifti", fake_load)
    assert viewer.df["scan"].tolist() == [
        "arterial",
        "portal-first",
        "portal-second",
        "delayed",
    ]

    viewer.current_index = 0
    viewer.load_data()
    loaded.clear()
    viewer.on_next(None)
    assert (viewer.current_index, loaded[-1]) == (1, "portal-first")
    viewer.on_next(None)
    assert (viewer.current_index, loaded[-1]) == (2, "portal-second")

    viewer.current_index = 3
    viewer.load_data()
    loaded.clear()
    viewer.on_prev(None)
    assert (viewer.current_index, loaded[-1]) == (2, "portal-second")
    viewer.on_prev(None)
    assert (viewer.current_index, loaded[-1]) == (1, "portal-first")


def test_random_previous_at_history_start_prints_message(capsys):
    instance = CTScanViewer.__new__(CTScanViewer)
    instance.exploration_mode = "random"
    instance.history_index = 0

    instance.on_prev(None)

    assert "Already at the first explored scan" in capsys.readouterr().out
