# %matplotlib widget

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive_output
from IPython.display import display, clear_output
import time
import pandas as pd

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

        self.HU_min = HU_min
        self.HU_max = HU_max
        self.current_index = 0
        self.view_plane = "axial"
        self.slice_idx = 0
        self.ct_scan = np.zeros([2, 2, 2])
        self.segmentations = {}
        self.fig_event = None
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
            description="Slice ",
            layout=widgets.Layout(width="400px"),
        )
        self.slice_slider.observe(self.on_slice_change, names="value")

        self.prev_slice_button = widgets.Button(
            description="←", layout=widgets.Layout(width="40px")
        )
        self.next_slice_button = widgets.Button(
            description="→", layout=widgets.Layout(width="40px")
        )
        self.prev_slice_button.on_click(self.on_prev_slice)
        self.next_slice_button.on_click(self.on_next_slice_manual)

        self.alpha_slider = widgets.FloatSlider(
            value=0.1, min=0, max=1, step=0.1, description="α", orientation="vertical"
        )

        self.plane_selector = widgets.ToggleButtons(
            options=["axial", "sagittal", "coronal"], description="Plane "
        )
        self.plane_selector.observe(self.on_plane_change, names="value")

        self.next_button = widgets.Button(description="Next Scan")
        self.next_button.on_click(self.on_next)
        self.prev_button = widgets.Button(
            description="↵", layout=widgets.Layout(width="40px")
        )
        self.prev_button.on_click(self.on_prev)

        self.progress_bar = widgets.FloatProgress(
            value=0, min=0, max=1, description="Loading:", bar_style="info"
        )

        self.info_display = widgets.HTML(value="")

        ui_top = widgets.VBox(
            [
                self.plane_selector,
                widgets.HBox(
                    [self.prev_slice_button, self.slice_slider, self.next_slice_button]
                ),
            ]
        )

        out = widgets.interactive_output(
            self.update_display,
            {
                "slice_idx": self.slice_slider,
                "view_plane": self.plane_selector,
                "alpha": self.alpha_slider,
            },
        )

        ui_bot = widgets.HBox(
            [
                out,
                self.alpha_slider,
                self.info_display,
                self.prev_button,
                self.next_button,
                self.progress_bar,
            ]
        )
        display(ui_top, ui_bot)

    def load_data(self):
        self.progress_bar.layout.visibility = "visible"
        self.progress_bar.value = 0
        self.progress_bar.bar_style = "info"
        self.progress_bar.description = "Loading..."

        row = self.df.iloc[self.current_index]
        self.progress_bar.value = 0.1
        self.ct_scan = load_nifti(row[self.ct_scan_col])
        self.progress_bar.value = 0.4
        self.ct_scan = clip_hu_values(self.ct_scan, self.HU_min, self.HU_max)
        self.progress_bar.value = 0.6

        self.segmentations = {}
        if self.segmentation_cols:
            for seg_col in self.segmentation_cols:
                self.segmentations[seg_col] = load_nifti(row[seg_col])

        self.progress_bar.value = 0.8
        self.update_info_display()
        self.update_slice_slider()
        self.progress_bar.value = 1
        self.progress_bar.bar_style = "success"
        self.progress_bar.description = "Loaded"
        time.sleep(0.5)
        self.progress_bar.layout.visibility = "hidden"

    def update_slice_slider(self):
        self.slice_slider.value = 0

        if self.segmentations:
            first_segmentation = next(iter(self.segmentations.values()))
            if self.view_plane == "axial":
                self.num_slices = self.ct_scan.shape[2]
                self.slice_idx = np.argmax(np.sum(first_segmentation, axis=(0, 1)))
            elif self.view_plane == "sagittal":
                self.num_slices = self.ct_scan.shape[0]
                self.slice_idx = np.argmax(np.sum(first_segmentation, axis=(1, 2)))
            elif self.view_plane == "coronal":
                self.num_slices = self.ct_scan.shape[1]
                self.slice_idx = np.argmax(np.sum(first_segmentation, axis=(0, 2)))
        else:
            if self.view_plane == "axial":
                self.num_slices = self.ct_scan.shape[2]
            elif self.view_plane == "sagittal":
                self.num_slices = self.ct_scan.shape[0]
            elif self.view_plane == "coronal":
                self.num_slices = self.ct_scan.shape[1]
            self.slice_idx = self.num_slices // 2

        self.slice_slider.unobserve(self.on_slice_change, names="value")
        self.slice_slider.max = self.num_slices - 1
        self.slice_slider.value = self.slice_idx
        self.slice_slider.observe(self.on_slice_change, names="value")

    def update_display(self, slice_idx, view_plane, alpha=0.5):
        self.view_plane = view_plane

        if view_plane == "axial":
            ct_slice = self.ct_scan[:, :, slice_idx]
            seg_slices = {
                name: seg[:, :, slice_idx] for name, seg in self.segmentations.items()
            }
        elif view_plane == "sagittal":
            ct_slice = self.ct_scan[slice_idx, :, :]
            seg_slices = {
                name: seg[slice_idx, :, :] for name, seg in self.segmentations.items()
            }
        elif view_plane == "coronal":
            ct_slice = self.ct_scan[:, slice_idx, :]
            seg_slices = {
                name: seg[:, slice_idx, :] for name, seg in self.segmentations.items()
            }

        fig, ax = plt.subplots(figsize=(9, 9))
        fig.canvas.header_visible = False
        plt.imshow(ct_slice.T, cmap="gray", origin="lower")

        colormaps = ["jet", "autumn", "summer", "winter", "viridis"]
        contour_colors = ["blue", "red", "green", "cyan", "magenta"]

        for i, (seg_name, seg_slice) in enumerate(seg_slices.items()):
            cmap = colormaps[i % len(colormaps)]
            contour_color = contour_colors[i % len(contour_colors)]
            plt.imshow(
                np.ma.masked_where(seg_slice == 0, seg_slice).T,
                cmap=cmap,
                alpha=alpha,
                origin="lower",
            )
            plt.contour(
                seg_slice.T,
                colors=contour_color,
                linewidths=0.8,
                alpha=alpha + 0.1,
                origin="lower",
            )

        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def update_info_display(self):
        row = self.df.iloc[self.current_index]
        info = ""
        for column in row.index:
            if column in DICOM_TAGS_TO_DISPLAY:
                info += f"<b>{column}:</b> {row[column]}<br>"
        self.info_display.value = info

    def on_slice_change(self, change):
        self.slice_idx = self.slice_slider.value

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