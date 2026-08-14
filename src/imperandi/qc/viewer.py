# %matplotlib widget

import sys
import warnings
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ipywidgets as widgets
from IPython.display import clear_output, display

try:
    from imperandi.qc.viewer_resample import (
        DEFAULT_ISOTROPIC_RESOLUTION_MM,
        load_nifti_isotropic,
        validate_isotropic_resolution,
    )
    from imperandi.qc.viewer_windowing import normalize_modality, percentile_window
except ModuleNotFoundError as exc:
    # Allow this file to be loaded directly (for example with
    # importlib.util.spec_from_file_location) without installing IMPERANDI.
    if exc.name != "imperandi":
        raise
    viewer_dir = str(Path(__file__).resolve().parent)
    if viewer_dir not in sys.path:
        sys.path.insert(0, viewer_dir)
    from viewer_resample import (  # noqa: E402
        DEFAULT_ISOTROPIC_RESOLUTION_MM,
        load_nifti_isotropic,
        validate_isotropic_resolution,
    )
    from viewer_windowing import normalize_modality, percentile_window  # noqa: E402

warnings.filterwarnings("ignore")  # Ignore warnings

# List of DICOM tags to display
DICOM_TAGS_TO_DISPLAY = [
    "patient_key",
    "date",
    "Modality",
    "visit_order",
    "phase",
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

SLICE_NAV_STEP = 3

WINDOW_PRESETS = {
    "Soft Tissue": (40, 400),
    "Liver": (60, 150),
    "Lung": (-600, 1500),
    "Bone": (300, 1500),
}

COLORMAPS = ["jet", "autumn", "summer", "winter", "viridis"]
CONTOUR_COLORS = ["blue", "red", "green", "cyan", "magenta"]
DISPLAY_CANVAS_PX = 700
FIGURE_DPI = 100
JUMP_DROPDOWN_WIDTH = "250px"
JUMP_NAV_BUTTON_WIDTH = "120px"

PHASE_SORT_RANK = {
    "NATIVE": 0,
    "ARTERIAL": 1,
    "PORTAL": 2,
    "PORTAL_VENOUS": 2,
    "DELAYED": 3,
    "HEPATOBILIARY": 4,
}


def load_nifti(
    file_path,
    orientation="LAS",
    isotropic_resolution_mm=DEFAULT_ISOTROPIC_RESOLUTION_MM,
    order=1,
):
    """Load a NIfTI file, orient it, and resample to isotropic display spacing."""
    return load_nifti_isotropic(
        file_path,
        orientation=orientation,
        resolution_mm=isotropic_resolution_mm,
        order=order,
    )


def clip_hu_values(ct_scan, min_hu, max_hu):
    """Clip CT values to the selected Hounsfield Unit (HU) window."""
    return np.clip(ct_scan, min_hu, max_hu)


class ScanViewer:
    """Interactive Jupyter viewer for CT/MRI volumes and segmentation overlays.

    The viewer navigates cohort rows, patients, exams, phases, and anatomical
    planes with modality-aware intensity windowing and isotropic resampling.
    """

    def __init__(
        self,
        df,
        ct_scan_col,
        segmentation_cols=None,
        phase_col=None,
        HU_min=-100,
        HU_max=400,
        exploration_mode="ordered",
        isotropic_resolution_mm=DEFAULT_ISOTROPIC_RESOLUTION_MM,
    ):
        self.ct_scan_col = ct_scan_col
        self.patient_col = "patient_key" if "patient_key" in df.columns else None
        self.study_col = "study_id" if "study_id" in df.columns else None
        self.date_col = "date" if "date" in df.columns else None
        self.modality_col = next(
            (column for column in ("Modality", "modality") if column in df.columns),
            None,
        )
        if phase_col is not None and phase_col in df.columns:
            self.phase_col = phase_col
        elif "phase" in df.columns:
            self.phase_col = "phase"
        elif "totalseg_phase" in df.columns:
            self.phase_col = "totalseg_phase"
        else:
            self.phase_col = None

        self.df = self._sort_dataframe_for_navigation(df)
        df = self.df

        # Handle segmentation_cols gracefully
        if segmentation_cols is None:
            auto_cols = [col for col in df.columns if str(col).startswith("mask_")]
            if not auto_cols:
                auto_cols = [
                    col for col in ["liver_path", "liver_tumor_path"] if col in df
                ]
            self.segmentation_cols = [
                col
                for col in auto_cols
                if df[col].apply(lambda v: not self._is_empty_value(v)).any()
            ]
        elif isinstance(segmentation_cols, str):
            self.segmentation_cols = [segmentation_cols]
        else:
            self.segmentation_cols = segmentation_cols

        # Filter out missing segmentation columns
        self.segmentation_cols = [
            col for col in self.segmentation_cols if col in df.columns
        ]
        if segmentation_cols and not self.segmentation_cols:
            print("Warning: No valid segmentation columns found in DataFrame.")

        self.seg_colormaps = {}
        self.seg_contour_colors = {}
        for i, seg_name in enumerate(self.segmentation_cols):
            self.seg_colormaps[seg_name] = COLORMAPS[i % len(COLORMAPS)]
            self.seg_contour_colors[seg_name] = CONTOUR_COLORS[i % len(CONTOUR_COLORS)]

        self.HU_min = HU_min
        self.HU_max = HU_max
        self.percentile_min = 1.0
        self.percentile_max = 99.0
        self.current_index = 0
        self.view_plane = "axial"
        self.slice_idx = 0
        self.ct_scan_raw = np.zeros([2, 2, 2])
        self.segmentations = {}
        self.seg_visibility = {}
        self.fig = None
        self.ax = None
        self.display_widget = None
        self._uses_output_fallback = False
        self._suspend_jump = False
        self._suspend_window = False
        self._is_loading = False
        self._last_load_error = None
        self.exploration_mode = exploration_mode
        self.canvas_size_px = DISPLAY_CANVAS_PX
        self.figure_dpi = FIGURE_DPI
        self.image_aspect = "auto"
        self.isotropic_resolution_mm = validate_isotropic_resolution(
            isotropic_resolution_mm
        )

        if self.exploration_mode == "random":
            self.explored_history = [self.current_index]
            self.history_index = 0

        self.init_widgets()
        self.load_data()

    def _sort_dataframe_for_navigation(self, df):
        """Group rows and apply clinical phase order with stable ties."""
        sort_columns = [
            column
            for column in (
                self.patient_col,
                self.study_col,
                self.modality_col,
                self.phase_col,
            )
            if column is not None
        ]
        ordered = df.copy().reset_index(drop=True)
        if len(ordered) <= 1 or not sort_columns:
            return ordered

        sort_keys = pd.DataFrame(index=ordered.index)
        key_columns = []
        for position, column in enumerate(sort_columns):
            if column == self.phase_col:
                normalized_phase = (
                    ordered[column]
                    .apply(self._format_value)
                    .str.strip()
                    .str.upper()
                    .str.replace(r"[^A-Z0-9]+", "_", regex=True)
                    .str.strip("_")
                )
                rank_column = f"phase_rank_{position}"
                sort_keys[rank_column] = normalized_phase.map(PHASE_SORT_RANK).fillna(
                    5
                )
                key_columns.append(rank_column)
                continue
            if column == self.modality_col:
                values = ordered[column].apply(normalize_modality)
            else:
                values = ordered[column].apply(self._format_value)
            values = values.fillna("").astype(str).str.casefold()
            missing_column = f"missing_{position}"
            value_column = f"value_{position}"
            sort_keys[missing_column] = values.eq("")
            sort_keys[value_column] = values
            key_columns.extend([missing_column, value_column])

        sort_keys["input_position"] = np.arange(len(ordered))
        sorted_positions = sort_keys.sort_values(
            key_columns + ["input_position"],
            kind="mergesort",
        ).index
        return ordered.iloc[sorted_positions].reset_index(drop=True)

    def init_widgets(self):
        self.slice_slider = widgets.IntSlider(
            min=0,
            max=100,
            step=1,
            value=0,
            description="Slice",
            layout=widgets.Layout(width="100%", min_width="0px"),
        )
        self.slice_slider.observe(self.on_slice_change, names="value")

        nav_button_layout = widgets.Layout(
            width="auto", min_width=JUMP_NAV_BUTTON_WIDTH
        )
        scan_button_layout = widgets.Layout(width="50%", min_width="180px")
        self.prev_slice_button = widgets.Button(
            description="< Prev Slice", layout=nav_button_layout
        )
        self.next_slice_button = widgets.Button(
            description="Next Slice>", layout=nav_button_layout
        )
        self.prev_slice_button.on_click(self.on_prev_slice)
        self.next_slice_button.on_click(self.on_next_slice_manual)

        self.alpha_slider = widgets.FloatSlider(
            value=0.1,
            min=0,
            max=1,
            step=0.1,
            description="alpha",
            orientation="horizontal",
            layout=widgets.Layout(width="100%", min_width="0px"),
        )
        self.alpha_slider.observe(self.update_display, names="value")

        self.plane_selector = widgets.ToggleButtons(
            options=["axial", "sagittal", "coronal"],
            # layout=widgets.Layout(width="100%", min_width="0px"),
        )
        self.plane_selector.observe(self.on_plane_change, names="value")

        self.window_preset = widgets.Dropdown(
            options=["Custom"] + list(WINDOW_PRESETS.keys()),
            value="Custom",
            description="HU Window",
            layout=widgets.Layout(width="100%", min_width="0px"),
        )
        self.window_preset.observe(self.on_window_preset_change, names="value")

        self.hu_min_input = widgets.BoundedFloatText(
            value=self.HU_min,
            min=-10000,
            max=10000,
            step=10,
            description="HU min",
            layout=widgets.Layout(width="50%", min_width="0px"),
            style={"description_width": "105px"},
        )
        self.hu_max_input = widgets.BoundedFloatText(
            value=self.HU_max,
            min=-10000,
            max=10000,
            step=10,
            description="HU max",
            layout=widgets.Layout(width="50%", min_width="0px"),
            style={"description_width": "105px"},
        )
        self.percentile_min_input = widgets.BoundedFloatText(
            value=self.percentile_min,
            min=0,
            max=100,
            step=0.5,
            description="Percentile min",
            layout=widgets.Layout(width="50%", min_width="0px"),
            style={"description_width": "105px"},
        )
        self.percentile_max_input = widgets.BoundedFloatText(
            value=self.percentile_max,
            min=0,
            max=100,
            step=0.5,
            description="Percentile max",
            layout=widgets.Layout(width="50%", min_width="0px"),
            style={"description_width": "105px"},
        )
        self.hu_min_input.observe(self.on_hu_change, names="value")
        self.hu_max_input.observe(self.on_hu_change, names="value")
        self.percentile_min_input.observe(self.on_percentile_change, names="value")
        self.percentile_max_input.observe(self.on_percentile_change, names="value")

        self.resolution_input = widgets.BoundedFloatText(
            value=self.isotropic_resolution_mm,
            min=0.01,
            max=100.0,
            step=0.25,
            description="Voxel mm",
            layout=widgets.Layout(width="100%", min_width="0px"),
        )
        self.resolution_input.observe(self.on_resolution_change, names="value")

        phase_desc = self.phase_col if self.phase_col else "phase"
        self.patient_dropdown = widgets.Dropdown(
            description="Patient", layout=widgets.Layout(width="100%", min_width="0px")
        )
        self.date_dropdown = widgets.Dropdown(
            description="Date", layout=widgets.Layout(width="100%", min_width="0px")
        )
        self.modality_dropdown = widgets.Dropdown(
            description="Modality",
            layout=widgets.Layout(width="50%", min_width="0px"),
        )
        self.phase_dropdown = widgets.Dropdown(
            description=phase_desc, layout=widgets.Layout(width="50%", min_width="0px")
        )
        self.prev_patient_button = widgets.Button(
            description="< Prev Patient",
            layout=nav_button_layout,
        )
        self.next_patient_button = widgets.Button(
            description="Next Patient >",
            layout=nav_button_layout,
        )
        self.prev_date_button = widgets.Button(
            description="< Prev Exam",
            layout=nav_button_layout,
        )
        self.next_date_button = widgets.Button(
            description="Next Exam >",
            layout=nav_button_layout,
        )
        self._refresh_jump_dropdowns()
        self.patient_dropdown.observe(self.on_patient_change, names="value")
        self.date_dropdown.observe(self.on_date_change, names="value")
        self.modality_dropdown.observe(self.on_modality_change, names="value")
        self.phase_dropdown.observe(self.on_phase_change, names="value")
        self.prev_patient_button.on_click(self.on_prev_patient)
        self.next_patient_button.on_click(self.on_next_patient)
        self.prev_date_button.on_click(self.on_prev_date)
        self.next_date_button.on_click(self.on_next_date)

        self.next_button = widgets.Button(
            description="Next Scan >", layout=scan_button_layout
        )
        self.next_button.on_click(self.on_next)
        self.prev_button = widgets.Button(
            description="< Prev Scan", layout=scan_button_layout
        )
        self.prev_button.on_click(self.on_prev)

        self.progress_bar = widgets.FloatProgress(
            value=0,
            min=0,
            max=1,
            description="Loading:",
            bar_style="info",
            layout=widgets.Layout(width="100%", min_width="0px"),
        )

        self.info_display = widgets.HTML(value="")

        if self.segmentation_cols:
            for seg_name in self.segmentation_cols:
                cb = widgets.Checkbox(value=True, description=seg_name, indent=True)
                cb.observe(self.on_seg_visibility_change, names="value")
                self.seg_visibility[seg_name] = cb
            self.seg_visibility_box = widgets.VBox(
                list(self.seg_visibility.values()),
                layout=widgets.Layout(width="100%", min_width="100px"),
            )
        else:
            self.seg_visibility_box = widgets.HTML(
                "<i>No segmentations</i>",
                layout=widgets.Layout(width="100%", min_width="100px"),
            )

        if self.segmentation_cols:
            self.center_seg_dropdown = widgets.Dropdown(
                options=self.segmentation_cols, description=""
            )
            self.center_button = widgets.Button(description="Center on")
            self.center_button.on_click(self.on_center_on_lesion)
        else:
            self.center_seg_dropdown = widgets.Dropdown(
                options=[], description="", disabled=True
            )
            self.center_button = widgets.Button(description="Center on", disabled=True)

        self._try_enable_widget_backend()
        was_interactive = plt.isinteractive()
        try:
            # Avoid backend auto-publishing a second figure output in notebooks.
            plt.ioff()
            figure_size_in = self.canvas_size_px / float(self.figure_dpi)
            self.fig, self.ax = plt.subplots(
                figsize=(figure_size_in, figure_size_in),
                dpi=self.figure_dpi,
            )
        finally:
            if was_interactive:
                plt.ion()
        self._pin_axes_to_canvas()
        if hasattr(self.fig.canvas, "header_visible"):
            self.fig.canvas.header_visible = False

        canvas = self.fig.canvas
        if isinstance(canvas, widgets.Widget):
            canvas.layout = widgets.Layout(
                width="100%",
                height="100%",
            )
            self.display_widget = canvas
            self._uses_output_fallback = False
            # Keep widget canvas alive; closing it clears the widget comm.
        else:
            self._uses_output_fallback = True
            output = widgets.Output(
                layout=widgets.Layout(
                    width="100%",
                    height="100%",
                )
            )
            self.display_widget = output
            self._render_output_figure()
            # Prevent duplicate auto-render below widgets for inline backends.
            plt.close(self.fig)

        if self.fig is not None:
            self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        group_layout = widgets.Layout(
            border="1px solid #d9d9d9",
            padding="6px",
            margin="0 0 6px 0",
            align_items="center",
            justify_content="center",
            overflow="hidden",
        )
        grid_three = widgets.Layout(
            grid_template_columns="auto 1fr auto",
            width="100%",
            align_items="center",
            grid_gap="6px",
        )
        # grid_two = widgets.Layout(
        #     grid_template_columns="auto 1fr",
        #     width="100%",
        #     align_items="center",
        #     grid_gap="6px",
        # )
        view_grid_three = widgets.Layout(
            grid_template_columns="auto 1fr auto",
            width="100%",
            align_items="center",
            justify_content="center",
            grid_gap="6px",
        )
        view_grid_two = widgets.Layout(
            grid_template_columns="auto 1fr",
            width="100%",
            align_items="center",
            justify_content="center",
            justify_items="center",
            grid_gap="6px",
        )

        top_left_controls = widgets.VBox(
            [
                widgets.HTML("<b>View</b>"),
                widgets.GridBox(
                    [widgets.HTML("Plane"), self.plane_selector],
                    layout=view_grid_two,
                ),
                widgets.GridBox(
                    [self.prev_slice_button, self.slice_slider, self.next_slice_button],
                    layout=view_grid_three,
                ),
            ],
            layout=widgets.Layout(
                border=group_layout.border,
                padding=group_layout.padding,
                margin=group_layout.margin,
                align_items=group_layout.align_items,
                justify_content="center",
                overflow=group_layout.overflow,
                height="100%",
            ),
        )
        top_right_jump = widgets.VBox(
            [
                widgets.HTML("<b>Explore</b>"),
                widgets.GridBox(
                    [
                        self.prev_patient_button,
                        self.patient_dropdown,
                        self.next_patient_button,
                    ],
                    layout=grid_three,
                ),
                widgets.GridBox(
                    [self.prev_date_button, self.date_dropdown, self.next_date_button],
                    layout=grid_three,
                ),
                widgets.HBox(
                    [self.modality_dropdown, self.phase_dropdown],
                    layout=widgets.Layout(width="100%"),
                ),
                widgets.HBox(
                    [self.prev_button, self.next_button],
                    layout=widgets.Layout(width="100%"),
                ),
            ],
            layout=widgets.Layout(
                border=group_layout.border,
                padding=group_layout.padding,
                margin=group_layout.margin,
                align_items=group_layout.align_items,
                justify_content=group_layout.justify_content,
                overflow=group_layout.overflow,
                height="100%",
            ),
        )
        ui_top = widgets.VBox(
            [
                widgets.HTML("<h3 style='margin:0 0 8px 0'>3D Scan Viewer</h3>"),
                widgets.GridBox(
                    [top_left_controls, top_right_jump],
                    layout=widgets.Layout(
                        grid_template_columns="1fr 1fr",
                        width="100%",
                        align_items="stretch",
                        grid_gap="6px",
                    ),
                ),
            ]
        )
        overlay_group = widgets.VBox(
            [
                widgets.HTML("<b>Mask Overlay</b>"),
                self.alpha_slider,
                self.seg_visibility_box,
            ],
            layout=group_layout,
        )
        window_group = widgets.VBox(
            [
                widgets.HTML("<b>Rendering</b>"),
                self.window_preset,
                widgets.HBox(
                    [self.hu_min_input, self.hu_max_input],
                    layout=widgets.Layout(width="100%"),
                ),
                widgets.HBox(
                    [self.percentile_min_input, self.percentile_max_input],
                    layout=widgets.Layout(width="100%"),
                ),
                self.resolution_input,
            ],
            layout=group_layout,
        )
        center_button_and_dropdown = widgets.HBox(
            [self.center_button, self.center_seg_dropdown],
            layout=widgets.Layout(width="100%", align_items="stretch"),
        )
        center_group = widgets.VBox(
            [widgets.HTML("<b>Largest Surface Slice</b>"), center_button_and_dropdown],
            layout=group_layout,
        )
        progress_group = widgets.VBox(
            [self.progress_bar],
            layout=widgets.Layout(
                width="100%", align_items="stretch", overflow="hidden"
            ),
        )

        self.info_container = widgets.Box(
            [self.info_display],
            layout=widgets.Layout(
                max_height="350px",
                overflow="auto",
                overflow_x="hidden",
            ),
        )

        right_items = [
            progress_group,
            center_group,
            overlay_group,
            window_group,
            self.info_container,
        ]
        right_panel = widgets.VBox(
            right_items,
            layout=widgets.Layout(width="500px"),
        )

        self.display_widget.layout = widgets.Layout(
            width="100%",
            flex="1 1 auto",
            min_width="500px",
        )
        ui_bot = widgets.HBox(
            [self.display_widget, right_panel],
            layout=widgets.Layout(width="100%", align_items="flex-start"),
        )
        display(ui_top, ui_bot)

    def _option_values(self, options):
        values = []
        for option in options:
            if isinstance(option, tuple) and len(option) == 2:
                values.append(option[1])
            else:
                values.append(option)
        return values

    def _step_dropdown(self, dropdown, direction, wrap=False):
        options = self._option_values(dropdown.options)
        values = [opt for opt in options if opt is not None]
        if not values:
            return
        current = dropdown.value
        if current not in values:
            if wrap and direction < 0:
                dropdown.value = values[-1]
            else:
                dropdown.value = values[0]
            return
        idx = values.index(current)
        if wrap:
            next_idx = (idx + direction) % len(values)
        else:
            next_idx = max(0, min(len(values) - 1, idx + direction))
        if next_idx != idx:
            dropdown.value = values[next_idx]

    def _update_jump_nav_buttons(self):
        patient_values = [
            opt
            for opt in self._option_values(self.patient_dropdown.options)
            if opt is not None
        ]
        date_values = [
            opt
            for opt in self._option_values(self.date_dropdown.options)
            if opt is not None
        ]
        self.prev_patient_button.disabled = len(patient_values) <= 1
        self.next_patient_button.disabled = len(patient_values) <= 1
        if len(date_values) <= 1:
            self.prev_date_button.disabled = True
            self.next_date_button.disabled = True
            return

        current_date = self.date_dropdown.value
        if current_date not in date_values:
            self.prev_date_button.disabled = True
            self.next_date_button.disabled = True
            return

        current_idx = date_values.index(current_date)
        self.prev_date_button.disabled = current_idx == 0
        self.next_date_button.disabled = current_idx == (len(date_values) - 1)

    def _build_options_for_column(self, column, formatter, frame=None):
        if column is None:
            return [("N/A", None)]
        source = self.df if frame is None else frame
        seen = set()
        options = []
        for _, row in source.iterrows():
            value = formatter(row.get(column))
            if value == "":
                continue
            if value in seen:
                continue
            seen.add(value)
            options.append((value, value))
        if not options:
            return [("N/A", None)]
        return options

    def _filter_frame_for_jump(
        self, patient_value=None, date_value=None, modality_value=None
    ):
        frame = self.df
        if self.patient_col and patient_value is not None:
            frame = frame[
                frame[self.patient_col].apply(
                    lambda v: self._format_value(v) == patient_value
                )
            ]
        if self.date_col and date_value is not None:
            frame = frame[
                frame[self.date_col].apply(lambda v: self._format_date(v) == date_value)
            ]
        if self.modality_col and modality_value is not None:
            frame = frame[
                frame[self.modality_col].apply(normalize_modality) == modality_value
            ]
        return frame

    def _build_patient_options(self):
        return self._build_options_for_column(self.patient_col, self._format_value)

    def _build_date_options(self, patient_value):
        if self.date_col is None:
            return [("N/A", None)]
        frame = self._filter_frame_for_jump(patient_value=patient_value)
        return self._build_options_for_column(
            self.date_col, self._format_date, frame=frame
        )

    def _build_phase_options(self, patient_value, date_value):
        if self.phase_col is None:
            return [("N/A", None)]
        frame = self._filter_frame_for_jump(
            patient_value=patient_value,
            date_value=date_value,
            modality_value=getattr(self.modality_dropdown, "value", None),
        )
        return self._build_options_for_column(
            self.phase_col, self._format_value, frame=frame
        )

    def _set_dropdown_options(self, dropdown, options, preferred=None, disabled=False):
        values = self._option_values(options)
        dropdown.options = options
        if preferred in values:
            dropdown.value = preferred
        elif values:
            dropdown.value = values[0]
        else:
            dropdown.value = None
        dropdown.disabled = disabled or (len(values) == 1 and values[0] is None)

    def _refresh_jump_dropdowns(self, *, use_current_row=False):
        if not hasattr(self, "patient_dropdown"):
            return

        row = self.df.iloc[self.current_index]
        patient_pref = (
            self._format_value(row.get(self.patient_col))
            if use_current_row and self.patient_col is not None
            else getattr(self.patient_dropdown, "value", None)
        )
        date_pref = (
            self._format_date(row.get(self.date_col))
            if use_current_row and self.date_col is not None
            else getattr(self.date_dropdown, "value", None)
        )
        phase_pref = (
            self._format_value(row.get(self.phase_col))
            if use_current_row and self.phase_col is not None
            else getattr(self.phase_dropdown, "value", None)
        )
        modality_pref = (
            normalize_modality(row.get(self.modality_col))
            if use_current_row and self.modality_col is not None
            else getattr(self.modality_dropdown, "value", None)
        )

        self._suspend_jump = True
        try:
            patient_options = self._build_patient_options()
            self._set_dropdown_options(
                self.patient_dropdown,
                patient_options,
                preferred=patient_pref,
                disabled=self.patient_col is None or len(patient_options) <= 1,
            )

            date_options = self._build_date_options(self.patient_dropdown.value)
            self._set_dropdown_options(
                self.date_dropdown,
                date_options,
                preferred=date_pref,
                disabled=self.date_col is None or len(date_options) <= 1,
            )

            modality_frame = self._filter_frame_for_jump(
                patient_value=self.patient_dropdown.value,
                date_value=self.date_dropdown.value,
            )
            modality_options = self._build_options_for_column(
                self.modality_col,
                normalize_modality,
                frame=modality_frame,
            )
            self._set_dropdown_options(
                self.modality_dropdown,
                modality_options,
                preferred=modality_pref,
                disabled=self.modality_col is None or len(modality_options) <= 1,
            )

            phase_options = self._build_phase_options(
                self.patient_dropdown.value,
                self.date_dropdown.value,
            )
            self._set_dropdown_options(
                self.phase_dropdown,
                phase_options,
                preferred=phase_pref,
                disabled=self.phase_col is None or len(phase_options) <= 1,
            )
        finally:
            self._suspend_jump = False
        self._update_jump_nav_buttons()

    def _jump_to_selected_filters(self):
        if len(self.df) == 0:
            return
        patient_value = self.patient_dropdown.value if self.patient_col else None
        date_value = self.date_dropdown.value if self.date_col else None
        phase_value = self.phase_dropdown.value if self.phase_col else None
        modality_value = (
            self.modality_dropdown.value if self.modality_col else None
        )

        for pos in range(len(self.df)):
            row = self.df.iloc[pos]
            if self.patient_col and patient_value is not None:
                if self._format_value(row.get(self.patient_col)) != patient_value:
                    continue
            if self.date_col and date_value is not None:
                if self._format_date(row.get(self.date_col)) != date_value:
                    continue
            if self.modality_col and modality_value is not None:
                if normalize_modality(row.get(self.modality_col)) != modality_value:
                    continue
            if self.phase_col and phase_value is not None:
                if self._format_value(row.get(self.phase_col)) != phase_value:
                    continue
            if int(pos) == int(self.current_index):
                return
            self.current_index = int(pos)
            self.load_data()
            return

    def _format_date(self, value):
        if value is None:
            return "?"
        try:
            if pd.isna(value):
                return "?"
        except Exception:
            pass
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        try:
            dt = pd.to_datetime(value, errors="coerce")
            if pd.isna(dt):
                return str(value)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return str(value)

    def _format_value(self, value):
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    def _is_empty_value(self, value):
        if value is None:
            return True
        try:
            if pd.isna(value):
                return True
        except Exception:
            pass
        if isinstance(value, (list, tuple, dict)) and len(value) == 0:
            return True
        return False

    def _get_selected_segmentation(self):
        if not self.segmentation_cols:
            return None
        seg_name = self.center_seg_dropdown.value
        if seg_name in self.segmentations:
            return self.segmentations[seg_name]
        return None

    def _compute_center_slice(self, seg):
        if seg is None:
            return None
        if self.view_plane == "axial":
            sums = np.sum(seg, axis=(0, 1))
        elif self.view_plane == "sagittal":
            sums = np.sum(seg, axis=(1, 2))
        else:
            sums = np.sum(seg, axis=(0, 2))
        if sums.size == 0:
            return None
        return int(np.argmax(sums))

    def _set_jump_value(self):
        self._refresh_jump_dropdowns(use_current_row=True)

    def _try_enable_widget_backend(self):
        """Best-effort switch to ipympl when available."""
        backend = (plt.get_backend() or "").lower()
        if "ipympl" in backend or "nbagg" in backend:
            return
        try:
            import ipympl  # noqa: F401

            plt.switch_backend("module://ipympl.backend_nbagg")
        except Exception:
            # Keep current backend; output fallback will still be interactive.
            return

    def _render_output_figure(self):
        if not self._uses_output_fallback:
            return
        if not isinstance(self.display_widget, widgets.Output):
            return
        with self.display_widget:
            clear_output(wait=True)
            display(self.fig)

    def _pin_axes_to_canvas(self):
        if self.fig is None or self.ax is None:
            return
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self.ax.set_position([0.0, 0.0, 1.0, 1.0])
        self.ax.set_aspect(self.image_aspect, adjustable="box")
        self.ax.margins(0)

    def on_window_preset_change(self, change):
        if self._current_modality() != "CT":
            return
        preset = change["new"]
        if preset == "Custom":
            return
        wl, ww = WINDOW_PRESETS[preset]
        window_min = wl - ww / 2.0
        window_max = wl + ww / 2.0
        self._suspend_window = True
        try:
            self.hu_max_input.value = window_max
            self.hu_min_input.value = window_min
        finally:
            self._suspend_window = False
        self.HU_min = window_min
        self.HU_max = window_max
        self.update_display()

    def _current_modality(self):
        if self.modality_col is None or self.df.empty:
            return "CT"
        return normalize_modality(
            self.df.iloc[self.current_index].get(self.modality_col)
        )

    def _update_window_controls(self):
        is_ct = self._current_modality() == "CT"
        self.window_preset.disabled = not is_ct
        self.hu_min_input.disabled = not is_ct
        self.hu_max_input.disabled = not is_ct
        self.percentile_min_input.disabled = is_ct
        self.percentile_max_input.disabled = is_ct

    def _display_window(self):
        if self._current_modality() == "MR":
            return percentile_window(
                self.ct_scan_raw,
                self.percentile_min,
                self.percentile_max,
            )
        return float(self.HU_min), float(self.HU_max)

    def on_hu_change(self, change):
        if self._suspend_window or self._current_modality() != "CT":
            return
        lower = float(self.hu_min_input.value)
        upper = float(self.hu_max_input.value)
        if lower >= upper:
            self._restore_window_inputs()
            return
        self.HU_min = lower
        self.HU_max = upper
        self.update_display()

    def on_percentile_change(self, change):
        if self._suspend_window or self._current_modality() != "MR":
            return
        lower = float(self.percentile_min_input.value)
        upper = float(self.percentile_max_input.value)
        if lower >= upper:
            self._restore_window_inputs()
            return
        self.percentile_min = lower
        self.percentile_max = upper
        self.update_display()

    def _restore_window_inputs(self):
        """Keep controls synchronized with the last valid rendered window."""
        self._suspend_window = True
        try:
            self.hu_min_input.value = self.HU_min
            self.hu_max_input.value = self.HU_max
            self.percentile_min_input.value = self.percentile_min
            self.percentile_max_input.value = self.percentile_max
        finally:
            self._suspend_window = False

    def on_resolution_change(self, change):
        self.isotropic_resolution_mm = validate_isotropic_resolution(change["new"])
        self.load_data()

    def on_patient_change(self, change):
        if self._suspend_jump:
            return
        self._refresh_jump_dropdowns(use_current_row=False)
        self._jump_to_selected_filters()

    def on_date_change(self, change):
        if self._suspend_jump:
            return
        self._refresh_jump_dropdowns(use_current_row=False)
        self._jump_to_selected_filters()

    def on_modality_change(self, change):
        if self._suspend_jump:
            return
        self._refresh_jump_dropdowns(use_current_row=False)
        self._jump_to_selected_filters()

    def on_phase_change(self, change):
        if self._suspend_jump:
            return
        self._jump_to_selected_filters()

    def on_prev_patient(self, button):
        self._step_dropdown(self.patient_dropdown, -1, wrap=True)

    def on_next_patient(self, button):
        self._step_dropdown(self.patient_dropdown, 1, wrap=True)

    def on_prev_date(self, button):
        self._step_dropdown(self.date_dropdown, -1)

    def on_next_date(self, button):
        self._step_dropdown(self.date_dropdown, 1)

    def on_seg_visibility_change(self, change):
        self.update_display()

    def on_center_on_lesion(self, button):
        seg = self._get_selected_segmentation()
        if seg is None:
            return
        center_idx = self._compute_center_slice(seg)
        if center_idx is not None:
            self.slice_slider.value = int(center_idx)

    def on_key_press(self, event):
        if getattr(self, "_is_loading", False):
            return
        key = (event.key or "").lower()
        if "shift+" in key:
            if "left" in key or "up" in key:
                self.on_prev(None)
            elif "right" in key or "down" in key:
                self.on_next(None)
            return

        if key in {"left", "up"}:
            self.on_prev_slice(None)
        elif key in {"right", "down"}:
            self.on_next_slice_manual(None)

    def _set_navigation_loading(self, loading):
        if not loading:
            for name in ("prev_button", "next_button"):
                control = getattr(self, name, None)
                if control is not None:
                    control.disabled = False
            return

        control_names = (
            "patient_dropdown",
            "date_dropdown",
            "modality_dropdown",
            "phase_dropdown",
            "prev_patient_button",
            "next_patient_button",
            "prev_date_button",
            "next_date_button",
            "prev_button",
            "next_button",
        )
        for name in control_names:
            control = getattr(self, name, None)
            if control is not None:
                control.disabled = True

    def _show_scan_load_error(self, exc):
        self.ct_scan_raw = None
        self.segmentations = {}
        self._last_load_error = exc
        self.update_info_display(load_error=exc)

        if self.ax is None:
            return
        self.ax.clear()
        self.ax.text(
            0.5,
            0.5,
            "Failed to load scan\n" f"{type(exc).__name__}: {exc}",
            ha="center",
            va="center",
            wrap=True,
            transform=self.ax.transAxes,
        )
        self.ax.axis("off")
        if self._uses_output_fallback:
            self._render_output_figure()
        else:
            self.fig.canvas.draw_idle()

    def load_data(self):
        if self._is_loading:
            return False

        self._is_loading = True
        self._set_navigation_loading(True)
        self.progress_bar.layout.visibility = "visible"
        self.progress_bar.value = 0
        self.progress_bar.bar_style = "info"
        self.progress_bar.description = "Loading..."

        try:
            row = self.df.iloc[self.current_index]
            self.progress_bar.value = 0.1
            try:
                scan = load_nifti(
                    row[self.ct_scan_col],
                    isotropic_resolution_mm=self.isotropic_resolution_mm,
                    order=1,
                )
            except Exception as exc:
                self.progress_bar.bar_style = "danger"
                self.progress_bar.description = "Load failed"
                self._show_scan_load_error(exc)
                return False

            self.ct_scan_raw = scan
            self._last_load_error = None
            self._update_window_controls()

            self.segmentations = {}
            if self.segmentation_cols:
                for seg_col in self.segmentation_cols:
                    seg_path = row.get(seg_col, None)
                    if seg_path is None:
                        continue
                    if isinstance(seg_path, float) and np.isnan(seg_path):
                        continue
                    try:
                        self.segmentations[seg_col] = load_nifti(
                            seg_path,
                            isotropic_resolution_mm=self.isotropic_resolution_mm,
                            order=0,
                        )
                    except Exception as exc:
                        print(
                            f"Warning: failed to load segmentation {seg_col} "
                            f"for index {self.current_index}: {exc}"
                        )

            self.progress_bar.value = 0.6
            self.update_info_display()
            self.update_slice_slider()
            self.progress_bar.value = 1
            self.progress_bar.bar_style = "success"
            self.progress_bar.description = "Loaded"
            self.progress_bar.layout.visibility = "hidden"
            return True
        finally:
            self._is_loading = False
            try:
                self._set_jump_value()
            finally:
                self._set_navigation_loading(False)

    def update_slice_slider(self):
        if self.ct_scan_raw is None:
            return
        self.view_plane = self.plane_selector.value
        if self.view_plane == "axial":
            self.num_slices = self.ct_scan_raw.shape[2]
        elif self.view_plane == "sagittal":
            self.num_slices = self.ct_scan_raw.shape[0]
        else:
            self.num_slices = self.ct_scan_raw.shape[1]

        center_idx = None
        seg = self._get_selected_segmentation()
        if seg is not None:
            center_idx = self._compute_center_slice(seg)
        if center_idx is None:
            center_idx = self.num_slices // 2

        self.slice_idx = int(center_idx)

        self.slice_slider.unobserve(self.on_slice_change, names="value")
        self.slice_slider.max = max(0, self.num_slices - 1)
        self.slice_slider.value = min(self.slice_idx, self.slice_slider.max)
        self.slice_slider.observe(self.on_slice_change, names="value")
        self.update_display()

    def update_display(self, *_):
        if self.ct_scan_raw is None or self.ax is None:
            return

        self.view_plane = self.plane_selector.value
        slice_idx = int(self.slice_slider.value)
        alpha = self.alpha_slider.value

        if self.view_plane == "axial":
            ct_slice = self.ct_scan_raw[:, :, slice_idx]
            seg_slices = {
                name: seg[:, :, slice_idx] for name, seg in self.segmentations.items()
            }
        elif self.view_plane == "sagittal":
            ct_slice = self.ct_scan_raw[slice_idx, :, :]
            seg_slices = {
                name: seg[slice_idx, :, :] for name, seg in self.segmentations.items()
            }
        else:
            ct_slice = self.ct_scan_raw[:, slice_idx, :]
            seg_slices = {
                name: seg[:, slice_idx, :] for name, seg in self.segmentations.items()
            }

        window_min, window_max = self._display_window()
        ct_slice = np.clip(ct_slice, window_min, window_max)

        self.ax.clear()
        self._pin_axes_to_canvas()
        self.ax.imshow(
            ct_slice.T,
            cmap="gray",
            origin="lower",
            aspect=self.image_aspect,
            interpolation="nearest",
            vmin=window_min,
            vmax=window_max,
        )

        visible_names = []
        for seg_name in self.segmentation_cols:
            if seg_name not in seg_slices:
                continue
            cb = self.seg_visibility.get(seg_name)
            if cb is not None and not cb.value:
                continue
            visible_names.append(seg_name)
            seg_slice = seg_slices[seg_name]
            cmap = self.seg_colormaps.get(seg_name, "jet")
            contour_color = self.seg_contour_colors.get(seg_name, "red")
            self.ax.imshow(
                np.ma.masked_where(seg_slice == 0, seg_slice).T,
                cmap=cmap,
                alpha=alpha,
                origin="lower",
                aspect=self.image_aspect,
                interpolation="nearest",
            )
            self.ax.contour(
                seg_slice.T,
                colors=contour_color,
                linewidths=0.8,
                alpha=min(1.0, alpha + 0.1),
                origin="lower",
            )

        if visible_names:
            handles = [
                mpatches.Patch(
                    color=self.seg_contour_colors.get(name, "red"), label=name
                )
                for name in visible_names
            ]
            self.ax.legend(
                handles=handles,
                loc="upper right",
                fontsize="small",
                framealpha=0.6,
            )

        self.ax.axis("off")
        if self._uses_output_fallback:
            self._render_output_figure()
        else:
            self.fig.canvas.draw_idle()

    def update_info_display(self, load_error=None):
        row = self.df.iloc[self.current_index]
        rows = []
        for column in DICOM_TAGS_TO_DISPLAY:
            if column not in row.index:
                continue
            value = row[column]
            if self._is_empty_value(value):
                continue
            formatted = self._format_value(value)
            if formatted == "":
                continue
            rows.append(
                "<tr>"
                f"<td style='word-wrap: break-word; overflow-wrap: anywhere;'><b>{column}</b></td>"
                f"<td style='word-wrap: break-word; overflow-wrap: anywhere;'>{formatted}</td>"
                "</tr>"
            )
        if rows:
            metadata_html = (
                "<table style='width: 100%; table-layout: fixed; border-collapse: collapse;'>"
                + "".join(rows)
                + "</table>"
            )
        else:
            metadata_html = "<i>No metadata</i>"

        error_html = ""
        if load_error is not None:
            identity = [f"Scan {self.current_index + 1}/{len(self.df)}"]
            for column in ("volume_id", "study_id", self.ct_scan_col):
                if column in row.index and not self._is_empty_value(row[column]):
                    identity.append(f"{column}={self._format_value(row[column])}")
            error_text = f"{type(load_error).__name__}: {load_error}"
            error_html = (
                "<div style='color: #b00020; margin-bottom: 8px;'>"
                "<b>Failed to load scan</b><br>"
                f"{escape(' | '.join(identity))}<br>"
                f"<code>{escape(error_text)}</code>"
                "</div>"
            )
        html = error_html + metadata_html
        self.info_display.value = html

    def on_slice_change(self, change):
        self.slice_idx = self.slice_slider.value
        self.update_display()

    def on_plane_change(self, change):
        self.view_plane = self.plane_selector.value
        self.update_slice_slider()

    def on_prev_slice(self, button):
        new_val = max(0, self.slice_slider.value - SLICE_NAV_STEP)
        self.slice_slider.value = new_val

    def on_next_slice_manual(self, button):
        new_val = min(self.slice_slider.max, self.slice_slider.value + SLICE_NAV_STEP)
        self.slice_slider.value = new_val

    def on_next(self, button):
        if getattr(self, "_is_loading", False):
            return
        if self.exploration_mode == "ordered":
            self.current_index = (self.current_index + 1) % len(self.df)
        else:
            if self.history_index == len(self.explored_history) - 1:
                unexplored = set(range(len(self.df))) - set(self.explored_history)
                if unexplored:
                    new_index = np.random.choice(list(unexplored))
                else:
                    new_index = np.random.choice(range(len(self.df)))
                self.explored_history.append(new_index)
                self.history_index += 1
                self.current_index = new_index
            else:
                self.history_index += 1
                self.current_index = self.explored_history[self.history_index]
        self.load_data()

    def on_prev(self, button):
        if getattr(self, "_is_loading", False):
            return
        if self.exploration_mode == "ordered":
            self.current_index = (self.current_index - 1) % len(self.df)
            self.load_data()
        else:
            if self.history_index > 0:
                self.history_index -= 1
                self.current_index = self.explored_history[self.history_index]
                self.load_data()
            else:
                print("Already at the first explored scan.")


# Backwards-compatible import for notebooks using the former CT-specific name.
CTScanViewer = ScanViewer
