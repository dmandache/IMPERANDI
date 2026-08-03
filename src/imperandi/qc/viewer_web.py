import argparse
import warnings
from contextlib import suppress
from html import escape

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import panel as pn
import param

try:
    from imperandi.qc.viewer_resample import (
        DEFAULT_ISOTROPIC_RESOLUTION_MM,
        load_nifti_isotropic,
        validate_isotropic_resolution,
    )
    from imperandi.qc.viewer_web_data import (
        FILTER_ALL_COLUMNS,
        filter_dataframe,
        get_image_path_columns,
        guess_ct_scan_col,
        guess_phase_col,
        guess_segmentation_cols,
        load_dataframe,
        validate_image_path_column,
    )
except ModuleNotFoundError:
    from viewer_resample import (
        DEFAULT_ISOTROPIC_RESOLUTION_MM,
        load_nifti_isotropic,
        validate_isotropic_resolution,
    )
    from viewer_web_data import (
        FILTER_ALL_COLUMNS,
        filter_dataframe,
        get_image_path_columns,
        guess_ct_scan_col,
        guess_phase_col,
        guess_segmentation_cols,
        load_dataframe,
        validate_image_path_column,
    )

warnings.filterwarnings("ignore")

pn.extension(sizing_mode="stretch_width")


DICOM_TAGS_TO_DISPLAY = [
    "patient_id",
    "date",
    "visit_order",
    "phase",
    "totalseg_phase",
    "SeriesDescription",
    "ImageType",
    "PixelSpacing",
    "SpacingBetweenSlices",
    "SliceThickness",
    "liver_noise",
    "liver_volume",
    "liver_median_hu",
    "vessels_median_hu",
    "tumor_median_hu",
    "num_tumors",
    "tumor_volume",
]

WINDOW_PRESETS = {
    "Custom": None,
    "Soft Tissue": (40, 400),
    "Liver": (60, 150),
    "Lung": (-600, 1500),
    "Bone": (300, 1500),
}

COLORMAPS = ["jet", "autumn", "summer", "winter", "viridis"]


def load_nifti(
    file_path,
    orientation="LAS",
    isotropic_resolution_mm=DEFAULT_ISOTROPIC_RESOLUTION_MM,
    order=1,
):
    return load_nifti_isotropic(
        file_path,
        orientation=orientation,
        resolution_mm=isotropic_resolution_mm,
        order=order,
    )


def clip_hu_values(ct_scan, min_hu, max_hu):
    return np.clip(ct_scan, min_hu, max_hu)


def is_empty_value(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip() == ""


def format_value(value):
    if is_empty_value(value):
        return "?"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def format_date(value):
    if is_empty_value(value):
        return "?"
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return str(value)
    return dt.strftime("%Y-%m-%d")


class CTScanPanelViewer(param.Parameterized):
    """
    Minimal Panel version of the current ipywidgets viewer.

    Keeps the app state in param.Parameters and lets Panel generate/react
    to widgets.
    """

    current_index = param.Integer(default=0)

    patient = param.Selector(default=None, objects=[])
    date = param.Selector(default=None, objects=[])
    phase = param.Selector(default=None, objects=[])

    view_plane = param.Selector(
        default="axial", objects=["axial", "sagittal", "coronal"]
    )
    slice_idx = param.Integer(default=0, bounds=(0, 1))

    window_preset = param.Selector(
        default="Custom", objects=list(WINDOW_PRESETS.keys())
    )
    HU_min = param.Integer(default=-100, step=10, bounds=(-1000, 1000))
    HU_max = param.Integer(default=400, step=10, bounds=(-1000, 1000))

    alpha = param.Number(default=0.10, bounds=(0, 1))
    isotropic_resolution_mm = param.Number(
        default=DEFAULT_ISOTROPIC_RESOLUTION_MM,
        bounds=(0.01, 100.0),
    )

    center_segmentation = param.Selector(default=None, objects=[])

    def __init__(
        self,
        df,
        ct_scan_col,
        segmentation_cols=None,
        phase_col=None,
        orientation="LAS",
        isotropic_resolution_mm=DEFAULT_ISOTROPIC_RESOLUTION_MM,
        **params,
    ):
        params.setdefault(
            "isotropic_resolution_mm",
            validate_isotropic_resolution(isotropic_resolution_mm),
        )
        super().__init__(**params)

        self.df = df.reset_index(drop=True)
        self.ct_scan_col = ct_scan_col
        self.orientation = orientation

        self.patient_col = "patient_id" if "patient_id" in self.df.columns else None
        self.date_col = "date" if "date" in self.df.columns else None

        if phase_col is not None and phase_col in self.df.columns:
            self.phase_col = phase_col
        elif "phase" in self.df.columns:
            self.phase_col = "phase"
        elif "totalseg_phase" in self.df.columns:
            self.phase_col = "totalseg_phase"
        else:
            self.phase_col = None

        if segmentation_cols is None:
            auto_cols = [c for c in self.df.columns if str(c).startswith("mask_")]
            if not auto_cols:
                auto_cols = [
                    c
                    for c in ["liver_path", "liver_tumor_path"]
                    if c in self.df.columns
                ]
            self.segmentation_cols = auto_cols
        elif isinstance(segmentation_cols, str):
            self.segmentation_cols = [segmentation_cols]
        else:
            self.segmentation_cols = list(segmentation_cols)

        self.segmentation_cols = [
            c
            for c in self.segmentation_cols
            if c in self.df.columns
            and self.df[c].apply(lambda x: not is_empty_value(x)).any()
        ]

        self.seg_visible = {c: True for c in self.segmentation_cols}
        self.segmentations = {}
        self.ct_scan_raw = np.zeros((2, 2, 2))

        self.figure_pane = pn.pane.Matplotlib(height=700, sizing_mode="stretch_both")
        self.info_pane = pn.pane.HTML(height=320, sizing_mode="stretch_width")

        self.progress = pn.indicators.Progress(
            name="Loading",
            value=0,
            max=100,
            active=False,
            sizing_mode="stretch_width",
        )

        self._build_widgets()
        self._refresh_jump_options()
        self.load_data()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_widgets(self):
        self.patient_widget = pn.widgets.Select.from_param(
            self.param.patient,
            name="Patient",
            sizing_mode="stretch_width",
        )
        self.date_widget = pn.widgets.Select.from_param(
            self.param.date,
            name="Date",
            sizing_mode="stretch_width",
        )
        self.phase_widget = pn.widgets.Select.from_param(
            self.param.phase,
            name=self.phase_col or "Phase",
            sizing_mode="stretch_width",
        )

        self.plane_widget = pn.widgets.RadioButtonGroup.from_param(
            self.param.view_plane,
            name="Plane",
            button_type="primary",
        )

        self.slice_widget = pn.widgets.IntSlider.from_param(
            self.param.slice_idx,
            name="Slice",
            sizing_mode="stretch_width",
        )

        self.window_widget = pn.widgets.Select.from_param(
            self.param.window_preset,
            name="HU Window",
            sizing_mode="stretch_width",
        )

        self.hu_min_widget = pn.widgets.IntInput.from_param(
            self.param.HU_min,
            name="HU min",
        )
        self.hu_max_widget = pn.widgets.IntInput.from_param(
            self.param.HU_max,
            name="HU max",
        )

        self.alpha_widget = pn.widgets.FloatSlider.from_param(
            self.param.alpha,
            name="Mask alpha",
            start=0,
            end=1,
            step=0.05,
            sizing_mode="stretch_width",
        )

        self.resolution_widget = pn.widgets.FloatInput.from_param(
            self.param.isotropic_resolution_mm,
            name="Voxel mm",
            step=0.25,
        )

        self.center_widget = pn.widgets.Select.from_param(
            self.param.center_segmentation,
            name="Center mask",
            sizing_mode="stretch_width",
        )

        self.prev_scan_button = pn.widgets.Button(
            name="< Prev Scan", button_type="default"
        )
        self.next_scan_button = pn.widgets.Button(
            name="Next Scan >", button_type="default"
        )
        self.prev_patient_button = pn.widgets.Button(name="< Prev Patient")
        self.next_patient_button = pn.widgets.Button(name="Next Patient >")
        self.prev_date_button = pn.widgets.Button(name="< Prev Exam")
        self.next_date_button = pn.widgets.Button(name="Next Exam >")
        self.prev_slice_button = pn.widgets.Button(name="< Prev Slice")
        self.next_slice_button = pn.widgets.Button(name="Next Slice >")
        self.center_button = pn.widgets.Button(
            name="Center on largest mask slice", button_type="primary"
        )

        self.prev_scan_button.on_click(lambda _: self.step_scan(-1))
        self.next_scan_button.on_click(lambda _: self.step_scan(1))
        self.prev_patient_button.on_click(
            lambda _: self.step_selector("patient", -1, wrap=True)
        )
        self.next_patient_button.on_click(
            lambda _: self.step_selector("patient", 1, wrap=True)
        )
        self.prev_date_button.on_click(
            lambda _: self.step_selector("date", -1, wrap=False)
        )
        self.next_date_button.on_click(
            lambda _: self.step_selector("date", 1, wrap=False)
        )
        self.prev_slice_button.on_click(lambda _: self.step_slice(-3))
        self.next_slice_button.on_click(lambda _: self.step_slice(3))
        self.center_button.on_click(lambda _: self.center_on_mask())

        self.seg_checkboxes = {}
        for col in self.segmentation_cols:
            cb = pn.widgets.Checkbox(name=col, value=True)
            cb.param.watch(self._on_seg_visibility_change, "value")
            self.seg_checkboxes[col] = cb

        self.param.watch(self._on_jump_change, ["patient", "date", "phase"])
        self.param.watch(self._on_plane_change, "view_plane")
        self.param.watch(self._on_window_change, "window_preset")
        self.param.watch(
            self._update_display_event, ["slice_idx", "HU_min", "HU_max", "alpha"]
        )
        self.param.watch(self._on_resolution_change, "isotropic_resolution_mm")

    # ------------------------------------------------------------------
    # Navigation / dropdown logic
    # ------------------------------------------------------------------

    def _unique_options(self, frame, column, formatter=format_value):
        if column is None:
            return []
        values = []
        seen = set()
        for value in frame[column].tolist():
            v = formatter(value)
            if v not in seen:
                values.append(v)
                seen.add(v)
        return values

    def _filtered_frame(self, patient=None, date=None):
        frame = self.df

        if self.patient_col and patient not in [None, "?"]:
            frame = frame[frame[self.patient_col].apply(format_value) == patient]

        if self.date_col and date not in [None, "?"]:
            frame = frame[frame[self.date_col].apply(format_date) == date]

        return frame

    def _refresh_jump_options(self, use_current_row=True):
        row = self.df.iloc[self.current_index]

        patient_options = self._unique_options(self.df, self.patient_col, format_value)
        self.param.patient.objects = patient_options or [None]

        if use_current_row and self.patient_col:
            self.patient = format_value(row[self.patient_col])
        elif self.patient not in self.param.patient.objects:
            self.patient = self.param.patient.objects[0]

        date_frame = self._filtered_frame(patient=self.patient)
        date_options = self._unique_options(date_frame, self.date_col, format_date)
        self.param.date.objects = date_options or [None]

        if use_current_row and self.date_col:
            self.date = format_date(row[self.date_col])
        elif self.date not in self.param.date.objects:
            self.date = self.param.date.objects[0]

        phase_frame = self._filtered_frame(patient=self.patient, date=self.date)
        phase_options = self._unique_options(phase_frame, self.phase_col, format_value)
        self.param.phase.objects = phase_options or [None]

        if use_current_row and self.phase_col:
            self.phase = format_value(row[self.phase_col])
        elif self.phase not in self.param.phase.objects:
            self.phase = self.param.phase.objects[0]

        self.param.center_segmentation.objects = self.segmentation_cols or [None]
        if (
            self.segmentation_cols
            and self.center_segmentation not in self.segmentation_cols
        ):
            self.center_segmentation = self.segmentation_cols[0]

    def _on_jump_change(self, event):
        matches = self.df.copy()

        if self.patient_col and self.patient not in [None, "?"]:
            matches = matches[
                matches[self.patient_col].apply(format_value) == self.patient
            ]

        if self.date_col and self.date not in [None, "?"]:
            matches = matches[matches[self.date_col].apply(format_date) == self.date]

        if self.phase_col and self.phase not in [None, "?"]:
            matches = matches[matches[self.phase_col].apply(format_value) == self.phase]

        if len(matches) == 0:
            return

        new_index = int(matches.index[0])
        if new_index != self.current_index:
            self.current_index = new_index
            self.load_data(refresh_jump=False)

    def step_selector(self, name, step, wrap=False):
        objects = list(getattr(self.param, name).objects)
        current = getattr(self, name)

        if not objects or current not in objects:
            return

        idx = objects.index(current)
        new_idx = idx + step

        if wrap:
            new_idx %= len(objects)
        else:
            new_idx = max(0, min(len(objects) - 1, new_idx))

        setattr(self, name, objects[new_idx])

    def step_scan(self, step):
        self.current_index = (self.current_index + step) % len(self.df)
        self.load_data(refresh_jump=True)

    def step_slice(self, step):
        lo, hi = self.param.slice_idx.bounds
        self.slice_idx = int(max(lo, min(hi, self.slice_idx + step)))

    # ------------------------------------------------------------------
    # Loading and rendering
    # ------------------------------------------------------------------

    def load_data(self, refresh_jump=True):
        self.progress.active = True
        self.progress.value = 10

        row = self.df.iloc[self.current_index]

        ct_path = row[self.ct_scan_col]
        self.ct_scan_raw = load_nifti(
            ct_path,
            orientation=self.orientation,
            isotropic_resolution_mm=self.isotropic_resolution_mm,
            order=1,
        )

        self.progress.value = 50

        self.segmentations = {}
        for col in self.segmentation_cols:
            path = row.get(col)
            if not is_empty_value(path):
                try:
                    self.segmentations[col] = load_nifti(
                        path,
                        orientation=self.orientation,
                        isotropic_resolution_mm=self.isotropic_resolution_mm,
                        order=0,
                    )
                except Exception as e:
                    print(f"Could not load segmentation {col}: {e}")

        if refresh_jump:
            self._refresh_jump_options(use_current_row=True)

        self._update_slice_bounds()
        self.slice_idx = self.param.slice_idx.bounds[1] // 2

        self.progress.value = 90
        self.update_display()

        self.progress.value = 100
        self.progress.active = False

    def _on_plane_change(self, event):
        self._update_slice_bounds()
        self.slice_idx = self.param.slice_idx.bounds[1] // 2
        self.update_display()

    def _on_resolution_change(self, event):
        self.isotropic_resolution_mm = validate_isotropic_resolution(event.new)
        self.load_data(refresh_jump=False)

    def _update_slice_bounds(self):
        shape = self.ct_scan_raw.shape

        if self.view_plane == "axial":
            max_slice = shape[2] - 1
        elif self.view_plane == "sagittal":
            max_slice = shape[0] - 1
        else:
            max_slice = shape[1] - 1

        self.param.slice_idx.bounds = (0, int(max_slice))

    def _on_window_change(self, event):
        preset = WINDOW_PRESETS.get(self.window_preset)
        if preset is not None:
            center, width = preset
            self.HU_min = int(center - width / 2)
            self.HU_max = int(center + width / 2)
        self.update_display()

    def _on_seg_visibility_change(self, event):
        for col, cb in self.seg_checkboxes.items():
            self.seg_visible[col] = cb.value
        self.update_display()

    def _update_display_event(self, event):
        self.update_display()

    def get_slice(self, volume):
        if self.view_plane == "axial":
            return volume[:, :, self.slice_idx]
        if self.view_plane == "sagittal":
            return volume[self.slice_idx, :, :]
        if self.view_plane == "coronal":
            return volume[:, self.slice_idx, :]
        raise ValueError(self.view_plane)

    def center_on_mask(self):
        seg = self.segmentations.get(self.center_segmentation)
        if seg is None:
            return

        if self.view_plane == "axial":
            sums = np.sum(seg > 0, axis=(0, 1))
        elif self.view_plane == "sagittal":
            sums = np.sum(seg > 0, axis=(1, 2))
        else:
            sums = np.sum(seg > 0, axis=(0, 2))

        if sums.size:
            self.slice_idx = int(np.argmax(sums))

    def update_display(self):
        ct = clip_hu_values(self.ct_scan_raw, self.HU_min, self.HU_max)
        img = self.get_slice(ct)

        fig, ax = plt.subplots(figsize=(7, 7), dpi=100)

        ax.imshow(np.rot90(img), cmap="gray", vmin=self.HU_min, vmax=self.HU_max)

        for i, col in enumerate(self.segmentation_cols):
            if not self.seg_visible.get(col, True):
                continue

            seg = self.segmentations.get(col)
            if seg is None:
                continue

            seg_slice = self.get_slice(seg)
            seg_slice = np.rot90(seg_slice)

            masked = np.ma.masked_where(seg_slice <= 0, seg_slice)
            cmap = COLORMAPS[i % len(COLORMAPS)]
            ax.imshow(masked, cmap=cmap, alpha=self.alpha)

            with suppress(Exception):
                ax.contour(seg_slice > 0, levels=[0.5], linewidths=0.8)

        ax.set_title(
            f"Scan {self.current_index + 1}/{len(self.df)} | "
            f"{self.view_plane} | slice {self.slice_idx}"
        )
        ax.axis("off")

        self.figure_pane.object = fig
        plt.close(fig)

        self.info_pane.object = self._build_info_html()

    def _build_info_html(self):
        row = self.df.iloc[self.current_index]

        rows = []
        for tag in DICOM_TAGS_TO_DISPLAY:
            if tag in row.index:
                rows.append(
                    f"<tr><td><b>{tag}</b></td><td>{format_value(row[tag])}</td></tr>"
                )

        path = row.get(self.ct_scan_col, "")
        rows.append(f"<tr><td><b>{self.ct_scan_col}</b></td><td>{path}</td></tr>")

        return f"""
        <div style="max-height:320px; overflow-y:auto; font-size:13px;">
        <table>
        {''.join(rows)}
        </table>
        </div>
        """

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def panel(self):
        view_controls = pn.Card(
            self.plane_widget,
            pn.Row(self.prev_slice_button, self.next_slice_button),
            self.slice_widget,
            title="View",
            collapsed=False,
        )

        explore_controls = pn.Card(
            pn.Row(self.prev_patient_button, self.next_patient_button),
            self.patient_widget,
            pn.Row(self.prev_date_button, self.next_date_button),
            self.date_widget,
            pn.Row(self.prev_scan_button, self.next_scan_button),
            self.phase_widget,
            title="Explore",
            collapsed=False,
        )

        overlay_controls = pn.Card(
            self.alpha_widget,
            *self.seg_checkboxes.values(),
            title="Mask Overlay",
            collapsed=False,
        )

        rendering_controls = pn.Card(
            self.window_widget,
            pn.Row(self.hu_min_widget, self.hu_max_widget),
            self.resolution_widget,
            title="Rendering",
            collapsed=False,
        )

        center_controls = pn.Card(
            self.center_widget,
            self.center_button,
            title="Largest Surface Slice",
            collapsed=False,
        )

        sidebar = pn.Column(
            self.progress,
            view_controls,
            explore_controls,
            center_controls,
            overlay_controls,
            rendering_controls,
            self.info_pane,
            width=430,
            sizing_mode="fixed",
        )

        return pn.Row(
            self.figure_pane,
            sidebar,
            sizing_mode="stretch_both",
        )


class CTScanPanelApp:
    def __init__(
        self,
        initial_csv=None,
        ct_scan_col="nifti_path",
        segmentation_cols=None,
        phase_col=None,
        orientation="LAS",
        isotropic_resolution_mm=DEFAULT_ISOTROPIC_RESOLUTION_MM,
    ):
        self.orientation = orientation
        self.isotropic_resolution_mm = validate_isotropic_resolution(
            isotropic_resolution_mm
        )
        self.preferred_ct_col = ct_scan_col
        self.preferred_phase_col = phase_col
        self.preferred_seg_cols = segmentation_cols

        self.source_df = pd.DataFrame()
        self.filtered_df = pd.DataFrame()
        self.viewer = None

        self.status_pane = pn.pane.HTML(
            "<div style='font-size:13px; color:#555;'>Load a dataframe to begin.</div>",
            sizing_mode="stretch_width",
        )
        self.count_pane = pn.pane.HTML("", sizing_mode="stretch_width")
        self.preview_pane = pn.pane.DataFrame(
            pd.DataFrame(),
            height=220,
            sizing_mode="stretch_width",
        )
        self.viewer_container = pn.Column(
            pn.pane.HTML(
                "<div style='padding:24px; border:1px dashed #ccc; border-radius:8px;'>"
                "Load a dataframe, choose the NIfTI column, then apply filters "
                "if needed."
                "</div>",
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_both",
        )

        self._build_widgets()

        if initial_csv:
            self.df_path_input.value = str(initial_csv)
            self._load_dataframe_from_path()

    def _build_widgets(self):
        self.df_path_input = pn.widgets.TextInput(
            name="Dataframe path",
            placeholder="/path/to/index.csv",
            sizing_mode="stretch_width",
        )
        self.load_path_button = pn.widgets.Button(
            name="Load Path",
            button_type="primary",
        )
        self.load_path_button.on_click(self._load_dataframe_from_path)

        self.df_upload = pn.widgets.FileInput(
            name="Upload dataframe",
            accept=".csv,.tsv,.tab,.txt,.json,.pkl,.pickle,.parquet",
            sizing_mode="stretch_width",
        )
        self.load_upload_button = pn.widgets.Button(
            name="Load Upload",
            button_type="default",
        )
        self.load_upload_button.on_click(self._load_uploaded_dataframe)

        self.ct_col_widget = pn.widgets.Select(
            name="CT column",
            options=[],
            sizing_mode="stretch_width",
        )
        self.phase_col_widget = pn.widgets.Select(
            name="Phase column",
            options=[("None", None)],
            value=None,
            sizing_mode="stretch_width",
        )
        self.seg_cols_widget = pn.widgets.MultiSelect(
            name="Segmentation columns",
            options=[],
            value=[],
            size=6,
            sizing_mode="stretch_width",
        )
        self.resolution_widget = pn.widgets.FloatInput(
            name="Voxel mm",
            value=self.isotropic_resolution_mm,
            step=0.25,
            start=0.01,
            end=100.0,
        )
        self.refresh_viewer_button = pn.widgets.Button(
            name="Refresh Viewer",
            button_type="primary",
        )
        self.refresh_viewer_button.on_click(self._refresh_viewer)

        self.filter_column_widget = pn.widgets.Select(
            name="Filter column",
            options=[("All columns", FILTER_ALL_COLUMNS)],
            value=FILTER_ALL_COLUMNS,
            sizing_mode="stretch_width",
        )
        self.filter_mode_widget = pn.widgets.Select(
            name="Match mode",
            options=["contains", "exact", "regex"],
            value="contains",
            sizing_mode="stretch_width",
        )
        self.filter_value_widget = pn.widgets.TextInput(
            name="Filter text",
            placeholder="portal / P001 / liver",
            sizing_mode="stretch_width",
        )
        self.filter_query_widget = pn.widgets.TextInput(
            name="Pandas query",
            placeholder="phase == 'portal' and visit_order >= 2",
            sizing_mode="stretch_width",
        )
        self.case_sensitive_widget = pn.widgets.Checkbox(
            name="Case sensitive",
            value=False,
        )
        self.apply_filter_button = pn.widgets.Button(
            name="Apply Filter",
            button_type="primary",
        )
        self.clear_filter_button = pn.widgets.Button(
            name="Clear Filter",
            button_type="default",
        )
        self.apply_filter_button.on_click(self._apply_filter)
        self.clear_filter_button.on_click(self._clear_filter)

    def _set_status(self, message, kind="info"):
        colors = {
            "info": "#555",
            "success": "#1f6f43",
            "error": "#a12622",
        }
        color = colors.get(kind, colors["info"])
        self.status_pane.object = (
            f"<div style='font-size:13px; color:{color};'>{escape(str(message))}</div>"
        )

    def _set_viewer_message(self, message):
        self.viewer_container.objects = [
            pn.pane.HTML(
                (
                    "<div style='padding:24px; border:1px dashed #ccc; "
                    "border-radius:8px;'>"
                    f"{escape(str(message))}"
                    "</div>"
                ),
                sizing_mode="stretch_width",
            )
        ]

    def _update_preview(self):
        if self.filtered_df.empty:
            self.preview_pane.object = pd.DataFrame()
        else:
            self.preview_pane.object = self.filtered_df.head(25)

    def _update_counts(self):
        total = len(self.source_df)
        active = len(self.filtered_df)
        self.count_pane.object = (
            "<div style='font-size:13px; color:#555;'>"
            f"Active rows: {active} / {total}</div>"
        )

    def _configure_dataframe_widgets(self):
        columns = list(self.source_df.columns)
        path_columns = get_image_path_columns(self.source_df, allow_empty=True)

        ct_default = guess_ct_scan_col(path_columns, self.preferred_ct_col)
        phase_default = guess_phase_col(columns, self.preferred_phase_col)
        seg_defaults = guess_segmentation_cols(self.source_df, self.preferred_seg_cols)

        self.ct_col_widget.options = path_columns
        self.ct_col_widget.value = ct_default if ct_default in path_columns else None

        self.phase_col_widget.options = [("None", None)] + [(c, c) for c in columns]
        self.phase_col_widget.value = (
            phase_default if phase_default in columns else None
        )

        self.seg_cols_widget.options = path_columns
        self.seg_cols_widget.value = [c for c in seg_defaults if c in path_columns]

        self.filter_column_widget.options = [("All columns", FILTER_ALL_COLUMNS)] + [
            (c, c) for c in columns
        ]
        if self.filter_column_widget.value not in [
            value for _, value in self.filter_column_widget.options
        ]:
            self.filter_column_widget.value = FILTER_ALL_COLUMNS

        if not path_columns:
            self._set_status(
                "No image path columns were detected. CT and segmentation "
                "selections require file path columns.",
                kind="error",
            )

    def _load_dataframe_from_path(self, event=None):
        csv_path = self.df_path_input.value.strip()
        if not csv_path:
            self._set_status("Enter a dataframe path first.", kind="error")
            return

        try:
            df = load_dataframe(csv_path)
        except Exception as exc:
            self._set_status(f"Could not load dataframe: {exc}", kind="error")
            return

        self._on_dataframe_loaded(df, source_label=csv_path)

    def _load_uploaded_dataframe(self, event=None):
        if not self.df_upload.value:
            self._set_status("Choose a dataframe file to upload first.", kind="error")
            return

        file_name = getattr(self.df_upload, "filename", None) or "uploaded.csv"

        try:
            df = load_dataframe(self.df_upload.value, source_name=file_name)
        except Exception as exc:
            self._set_status(f"Could not load uploaded dataframe: {exc}", kind="error")
            return

        self._on_dataframe_loaded(df, source_label=file_name)

    def _on_dataframe_loaded(self, df, source_label):
        self.source_df = df.reset_index(drop=True)
        self._configure_dataframe_widgets()
        self._apply_filter(update_status=False)
        self._set_status(
            f"Loaded {len(self.source_df)} rows from {source_label}.",
            kind="success",
        )

    def _get_selected_segmentation_cols(self):
        return [
            column
            for column in list(self.seg_cols_widget.value or [])
            if column in self.filtered_df.columns
        ]

    def _apply_filter(self, event=None, update_status=True):
        if self.source_df.empty:
            self.filtered_df = pd.DataFrame()
            self._update_preview()
            self._update_counts()
            self._set_viewer_message("Load a dataframe to begin.")
            return

        try:
            self.filtered_df = filter_dataframe(
                self.source_df,
                text=self.filter_value_widget.value,
                column=self.filter_column_widget.value,
                mode=self.filter_mode_widget.value,
                query=self.filter_query_widget.value,
                case_sensitive=self.case_sensitive_widget.value,
            )
        except Exception as exc:
            if update_status:
                self._set_status(f"Filter error: {exc}", kind="error")
            return

        self._update_preview()
        self._update_counts()

        if self.filtered_df.empty:
            self._set_viewer_message("No rows match the current filters.")
            if update_status:
                self._set_status("Filters applied, but no rows matched.", kind="info")
            return

        self._refresh_viewer(update_status=update_status)

    def _clear_filter(self, event=None):
        self.filter_column_widget.value = FILTER_ALL_COLUMNS
        self.filter_mode_widget.value = "contains"
        self.filter_value_widget.value = ""
        self.filter_query_widget.value = ""
        self.case_sensitive_widget.value = False
        self._apply_filter()

    def _refresh_viewer(self, event=None, update_status=True):
        if self.filtered_df.empty:
            self._set_viewer_message("No rows are available to display.")
            return

        ct_scan_col = self.ct_col_widget.value
        if ct_scan_col is None:
            self._set_viewer_message(
                "Select the CT scan path column to open the viewer."
            )
            if update_status:
                self._set_status(
                    "Choose the CT scan column before opening the viewer.", kind="error"
                )
            return

        try:
            validate_image_path_column(
                self.filtered_df,
                ct_scan_col,
                allow_empty=False,
                label=f"CT column '{ct_scan_col}'",
            )
        except Exception as exc:
            self._set_viewer_message(str(exc))
            if update_status:
                self._set_status(str(exc), kind="error")
            return

        phase_col = self.phase_col_widget.value
        if phase_col not in self.filtered_df.columns:
            phase_col = None

        segmentation_cols = self._get_selected_segmentation_cols()
        try:
            for column in segmentation_cols:
                validate_image_path_column(
                    self.filtered_df,
                    column,
                    allow_empty=True,
                    label=f"Segmentation column '{column}'",
                )
        except Exception as exc:
            self._set_viewer_message(str(exc))
            if update_status:
                self._set_status(str(exc), kind="error")
            return

        try:
            self.viewer = CTScanPanelViewer(
                df=self.filtered_df,
                ct_scan_col=ct_scan_col,
                segmentation_cols=segmentation_cols or None,
                phase_col=phase_col,
                orientation=self.orientation,
                isotropic_resolution_mm=self.resolution_widget.value,
            )
        except Exception as exc:
            self._set_viewer_message(f"Viewer error: {exc}")
            if update_status:
                self._set_status(f"Could not open viewer: {exc}", kind="error")
            return

        self.preferred_ct_col = ct_scan_col
        self.preferred_phase_col = phase_col
        self.preferred_seg_cols = segmentation_cols
        self.viewer_container.objects = [self.viewer.panel()]

        if update_status:
            self._set_status(
                f"Showing {len(self.filtered_df)} filtered rows.",
                kind="success",
            )

    def panel(self):
        dataframe_controls = pn.Card(
            pn.Row(self.df_path_input, self.load_path_button),
            pn.Row(self.df_upload, self.load_upload_button),
            self.status_pane,
            title="Dataframe",
            collapsed=False,
        )

        schema_controls = pn.Card(
            self.ct_col_widget,
            self.phase_col_widget,
            self.seg_cols_widget,
            self.resolution_widget,
            self.refresh_viewer_button,
            title="Columns",
            collapsed=False,
        )

        filter_controls = pn.Card(
            pn.Row(self.filter_column_widget, self.filter_mode_widget),
            self.filter_value_widget,
            self.filter_query_widget,
            self.case_sensitive_widget,
            pn.Row(self.apply_filter_button, self.clear_filter_button),
            self.count_pane,
            title="Filter",
            collapsed=False,
        )

        preview_controls = pn.Card(
            self.preview_pane,
            title="Preview",
            collapsed=False,
        )

        return pn.Column(
            pn.Row(
                dataframe_controls,
                schema_controls,
                filter_controls,
                sizing_mode="stretch_width",
            ),
            preview_controls,
            self.viewer_container,
            sizing_mode="stretch_both",
        )


# ----------------------------------------------------------------------
# Example entry point
#
# Run the viewer with :
# panel serve src/imperandi/qc/viewer_web.py --args \
#   --csv /mnt/Data/Code/IMPERANDI/tests/data/nifti_index.csv
#
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=None)
parser.add_argument("--ct-col", default="nifti_path")
parser.add_argument("--phase-col", default=None)
parser.add_argument("--seg-cols", nargs="*", default=None)
parser.add_argument(
    "--isotropic-resolution-mm", type=float, default=DEFAULT_ISOTROPIC_RESOLUTION_MM
)
args = parser.parse_args()

app = CTScanPanelApp(
    initial_csv=args.csv,
    ct_scan_col=args.ct_col,
    segmentation_cols=args.seg_cols,
    phase_col=args.phase_col,
    isotropic_resolution_mm=args.isotropic_resolution_mm,
)

app.panel().servable(title="IMPERANDI QC Viewer")
