# %matplotlib widget

from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ipywidgets as widgets
from IPython.display import clear_output, display

# List of DICOM tags to display
DICOM_TAGS_TO_DISPLAY = [
    "patient_key",
    "date",
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


def load_nifti(file_path, orientation="LAS"):
    """Load a NIfTI file and return the image data oriented in RAS+."""
    img = nib.load(Path(file_path).resolve())
    data = img.get_fdata()
    affine = img.affine
    current_ornt = nib.orientations.io_orientation(affine)
    if orientation == "RAS":
        new_ornt = np.array([[0, 1], [1, 1], [2, 1]])
    elif orientation == "LAS":
        new_ornt = np.array([[0, -1], [1, 1], [2, 1]])
    transform = nib.orientations.ornt_transform(current_ornt, new_ornt)
    return nib.orientations.apply_orientation(data, transform)


def clip_hu_values(ct_scan, min_hu, max_hu):
    """Clip the Hounsfield Unit (HU) values of the CT scan."""
    return np.clip(ct_scan, min_hu, max_hu)


class CTScanViewer:
    def __init__(
        self,
        df,
        ct_scan_col,
        segmentation_cols=None,
        HU_min=-100,
        HU_max=400,
        exploration_mode="ordered",
    ):
        self.df = df
        self.ct_scan_col = ct_scan_col

        # Handle segmentation_cols gracefully
        if segmentation_cols is None:
            self.segmentation_cols = []
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
        self.exploration_mode = exploration_mode

        if self.exploration_mode == "random":
            self.explored_history = [self.current_index]
            self.history_index = 0

        self.init_widgets()
        self.load_data()

    def init_widgets(self):
        self.slice_slider = widgets.IntSlider(
            min=0,
            max=100,
            step=1,
            value=0,
            description="Slice",
            layout=widgets.Layout(width="400px"),
        )
        self.slice_slider.observe(self.on_slice_change, names="value")

        self.prev_slice_button = widgets.Button(
            description="Prev", layout=widgets.Layout(width="60px")
        )
        self.next_slice_button = widgets.Button(
            description="Next", layout=widgets.Layout(width="60px")
        )
        self.prev_slice_button.on_click(self.on_prev_slice)
        self.next_slice_button.on_click(self.on_next_slice_manual)

        self.alpha_slider = widgets.FloatSlider(
            value=0.1,
            min=0,
            max=1,
            step=0.1,
            description="alpha",
            orientation="vertical",
            layout=widgets.Layout(height="200px"),
        )
        self.alpha_slider.observe(self.update_display, names="value")

        self.plane_selector = widgets.ToggleButtons(
            options=["axial", "sagittal", "coronal"], description="Plane"
        )
        self.plane_selector.observe(self.on_plane_change, names="value")

        self.window_preset = widgets.Dropdown(
            options=["Custom"] + list(WINDOW_PRESETS.keys()),
            value="Custom",
            description="Window",
        )
        self.window_preset.observe(self.on_window_preset_change, names="value")

        self.jump_dropdown = widgets.Dropdown(
            options=self._build_jump_options(),
            description="Jump",
        )
        self.jump_dropdown.observe(self.on_jump_change, names="value")

        self.next_button = widgets.Button(description="Next Scan")
        self.next_button.on_click(self.on_next)
        self.prev_button = widgets.Button(description="Prev Scan")
        self.prev_button.on_click(self.on_prev)

        self.progress_bar = widgets.FloatProgress(
            value=0, min=0, max=1, description="Loading:", bar_style="info"
        )

        self.info_display = widgets.HTML(value="")

        if self.segmentation_cols:
            for seg_name in self.segmentation_cols:
                cb = widgets.Checkbox(
                    value=True, description=seg_name, indent=False
                )
                cb.observe(self.on_seg_visibility_change, names="value")
                self.seg_visibility[seg_name] = cb
            self.seg_visibility_box = widgets.VBox(
                list(self.seg_visibility.values())
            )
        else:
            self.seg_visibility_box = widgets.HTML("<i>No segmentations</i>")

        if self.segmentation_cols:
            self.center_seg_dropdown = widgets.Dropdown(
                options=self.segmentation_cols, description="Center"
            )
            self.center_button = widgets.Button(description="Center on lesion")
            self.center_button.on_click(self.on_center_on_lesion)
        else:
            self.center_seg_dropdown = widgets.Dropdown(
                options=[], description="Center", disabled=True
            )
            self.center_button = widgets.Button(
                description="Center on lesion", disabled=True
            )

        self._try_enable_widget_backend()
        was_interactive = plt.isinteractive()
        try:
            # Avoid backend auto-publishing a second figure output in notebooks.
            plt.ioff()
            self.fig, self.ax = plt.subplots(figsize=(9, 9))
        finally:
            if was_interactive:
                plt.ion()
        if hasattr(self.fig.canvas, "header_visible"):
            self.fig.canvas.header_visible = False

        canvas = self.fig.canvas
        if isinstance(canvas, widgets.Widget):
            canvas.layout = widgets.Layout(width="650px", height="650px")
            self.display_widget = canvas
            self._uses_output_fallback = False
            # Keep widget canvas alive; closing it clears the widget comm.
        else:
            self._uses_output_fallback = True
            output = widgets.Output(
                layout=widgets.Layout(width="650px", height="650px")
            )
            self.display_widget = output
            self._render_output_figure()
            # Prevent duplicate auto-render below widgets for inline backends.
            plt.close(self.fig)

        if self.fig is not None:
            self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        ui_top = widgets.VBox(
            [
                widgets.HBox(
                    [self.plane_selector, self.window_preset, self.jump_dropdown]
                ),
                widgets.HBox(
                    [self.prev_slice_button, self.slice_slider, self.next_slice_button]
                ),
            ]
        )

        right_items = [
            self.alpha_slider,
            self.seg_visibility_box,
            self.center_seg_dropdown,
            self.center_button,
            self.info_display,
            widgets.HBox([self.prev_button, self.next_button]),
            self.progress_bar,
        ]
        right_panel = widgets.VBox(
            right_items, layout=widgets.Layout(width="320px")
        )

        ui_bot = widgets.HBox([self.display_widget, right_panel])
        display(ui_top, ui_bot)

    def _build_jump_options(self):
        options = []
        for pos in range(len(self.df)):
            row = self.df.iloc[pos]
            patient = row.get("patient_key", f"row {pos}")
            if pd.isna(patient):
                patient = f"row {pos}"
            date_str = self._format_date(row.get("date", None))
            label = f"{patient} | {date_str}"
            options.append((label, int(pos)))
        return options

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
        if not hasattr(self, "jump_dropdown"):
            return
        self._suspend_jump = True
        try:
            self.jump_dropdown.value = int(self.current_index)
        finally:
            self._suspend_jump = False

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

    def on_window_preset_change(self, change):
        preset = change["new"]
        if preset == "Custom":
            return
        wl, ww = WINDOW_PRESETS[preset]
        self.HU_min = wl - ww / 2.0
        self.HU_max = wl + ww / 2.0
        self.update_display()

    def on_jump_change(self, change):
        if self._suspend_jump:
            return
        new_index = change["new"]
        if new_index is None:
            return
        if new_index == self.current_index:
            return
        self.current_index = int(new_index)
        self.load_data()

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

    def load_data(self):
        self.progress_bar.layout.visibility = "visible"
        self.progress_bar.value = 0
        self.progress_bar.bar_style = "info"
        self.progress_bar.description = "Loading..."

        row = self.df.iloc[self.current_index]
        self.progress_bar.value = 0.1
        self.ct_scan_raw = load_nifti(row[self.ct_scan_col])

        self.segmentations = {}
        if self.segmentation_cols:
            for seg_col in self.segmentation_cols:
                seg_path = row.get(seg_col, None)
                if seg_path is None:
                    continue
                if isinstance(seg_path, float) and np.isnan(seg_path):
                    continue
                try:
                    self.segmentations[seg_col] = load_nifti(seg_path)
                except Exception as exc:
                    print(
                        f"Warning: failed to load segmentation {seg_col} "
                        f"for index {self.current_index}: {exc}"
                    )

        self.progress_bar.value = 0.6
        self.update_info_display()
        self.update_slice_slider()
        self._set_jump_value()
        self.progress_bar.value = 1
        self.progress_bar.bar_style = "success"
        self.progress_bar.description = "Loaded"
        time.sleep(0.5)
        self.progress_bar.layout.visibility = "hidden"

    def update_slice_slider(self):
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
                name: seg[:, :, slice_idx]
                for name, seg in self.segmentations.items()
            }
        elif self.view_plane == "sagittal":
            ct_slice = self.ct_scan_raw[slice_idx, :, :]
            seg_slices = {
                name: seg[slice_idx, :, :]
                for name, seg in self.segmentations.items()
            }
        else:
            ct_slice = self.ct_scan_raw[:, slice_idx, :]
            seg_slices = {
                name: seg[:, slice_idx, :]
                for name, seg in self.segmentations.items()
            }

        ct_slice = clip_hu_values(ct_slice, self.HU_min, self.HU_max)

        self.ax.clear()
        self.ax.imshow(ct_slice.T, cmap="gray", origin="lower")

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
                mpatches.Patch(color=self.seg_contour_colors.get(name, "red"), label=name)
                for name in visible_names
            ]
            self.ax.legend(
                handles=handles,
                loc="upper right",
                fontsize="small",
                framealpha=0.6,
            )

        self.ax.axis("off")
        self.fig.tight_layout()
        if self._uses_output_fallback:
            self._render_output_figure()
        else:
            self.fig.canvas.draw_idle()

    def update_info_display(self):
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
                f"<tr><td><b>{column}</b></td><td>{formatted}</td></tr>"
            )
        if rows:
            html = (
                "<table style='width: 100%; border-collapse: collapse;'>"
                + "".join(rows)
                + "</table>"
            )
        else:
            html = "<i>No metadata</i>"
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
