"""
Liver registration toolkit (arteriel -> portal) with plotting, geometry helpers,
and a batch routine that updates paths in-place (phase == 'arteriel').

USAGE EXAMPLE (single run):
---------------------------
out = register_arterial_to_portal(
    portal_img="/path/portal.nii.gz",
    arterial_img="/path/arteriel.nii.gz",
    portal_liver_mask="/path/portal_liver.nii.gz",
    arterial_liver_mask="/path/arteriel_liver.nii.gz",
    portal_tumor_mask="/path/portal_tumor.nii.gz",
    arterial_tumor_mask="/path/arteriel_tumor.nii.gz",
    verbose=False,
    compute_dice=True,
)

USAGE EXAMPLE (batch + update df in place):
-------------------------------------------
updated_df, log_df = register_update_paths_inplace(
    df=df,  # long/tidy with columns: patient_key, phase, nifti_path, liver_path, liver_tumor_path
    out_root="~/workspace/nifti_registered",
    df_save_path="~/workspace/nifti_registered/df_updated.csv",
    log_save_path="~/workspace/nifti_registered/registration_log.csv",
    pad_mm=25.0,
    bspline_ctrl_spacing_mm=90.0,
    band_mm=15.0,
    compute_dice=True,
    verbose=False,
    preview=False,  # set True to show small previews (guarded)
)
"""

import os, time, traceback
from typing import Tuple, Dict, Optional, Iterable, Literal
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import math
import SimpleITK as sitk

import matplotlib

print(f"plt backend : {matplotlib.rcParams['backend']}")
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# Global helpers (comparison, resampling, geometry)
# =============================================================================


def _same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    """Check if two SimpleITK images share the exact same grid (size, spacing, origin, direction)."""
    return (
        a.GetSize() == b.GetSize()
        and a.GetSpacing() == b.GetSpacing()
        and a.GetOrigin() == b.GetOrigin()
        and a.GetDirection() == b.GetDirection()
    )


def resample_like(
    ref: sitk.Image,
    img: sitk.Image,
    tx: Optional[sitk.Transform] = None,
    interp: int = sitk.sitkNearestNeighbor,
    default: float = 0.0,
    pixel_id: Optional[int] = None,
) -> sitk.Image:
    """
    Resample 'img' onto 'ref' grid with optional transform 'tx'.
    - interp: use sitk.sitkLinear for intensities, sitk.sitkNearestNeighbor for labels.
    - default: background fill value.
    - pixel_id: target pixel ID; defaults to 'img' pixel ID if None.

    Example:
        out = resample_like(portal_img, arteriel_img, tx=transform, interp=sitk.sitkLinear)
    """
    if pixel_id is None:
        pixel_id = img.GetPixelID()
    if tx is None:
        tx = sitk.Transform(3, sitk.sitkIdentity)
    return sitk.Resample(img, ref, tx, interp, default, pixel_id)


def _dice_coeff(A: sitk.Image, B: sitk.Image) -> float:
    """Compute Dice on binary masks; resamples B onto A if geometry differs."""
    A8 = sitk.Cast(A > 0, sitk.sitkUInt8)
    if (A8.GetSize(), A8.GetSpacing(), A8.GetOrigin(), A8.GetDirection()) != (
        B.GetSize(),
        B.GetSpacing(),
        B.GetOrigin(),
        B.GetDirection(),
    ):
        B = resample_like(A8, B, None, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    B8 = sitk.Cast(B > 0, sitk.sitkUInt8)
    # fast Dice with LabelOverlapMeasuresImageFilter
    f = sitk.LabelOverlapMeasuresImageFilter()
    f.Execute(A8, B8)
    return f.GetDiceCoefficient()


def _calc_vol_ml(mask_img, ref_img):
    if mask_img is None:
        return None
    # ensure binary
    mask_img = sitk.Cast(mask_img > 0, sitk.sitkUInt8)
    stats = sitk.StatisticsImageFilter()
    stats.Execute(mask_img)
    voxel_count = stats.GetSum()  # since values are 0/1
    sx, sy, sz = ref_img.GetSpacing()
    vol_mm3 = voxel_count * (sx * sy * sz)
    return vol_mm3 / 1000.0  # mL


def _bbox_index(mask: sitk.Image) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (start[3], size[3]) of label-1 bbox, or None if empty."""
    mask8 = sitk.Cast(mask > 0, sitk.sitkUInt8)
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(mask8)
    if not ls.HasLabel(1):
        return None
    ix, iy, iz, nx, ny, nz = ls.GetBoundingBox(1)
    return np.array([ix, iy, iz], int), np.array([nx, ny, nz], int)


def _bbox_world(mask: sitk.Image):
    """Physical-space AABB (min, max) for label-1 using continuous indices."""
    out = _bbox_index(mask)
    if out is None:
        return None
    start, size = out
    end = start + size  # exclusive
    corners = np.array(
        [
            [start[0], start[1], start[2]],
            [end[0], start[1], start[2]],
            [start[0], end[1], start[2]],
            [start[0], start[1], end[2]],
            [end[0], end[1], start[2]],
            [end[0], start[1], end[2]],
            [start[0], end[1], end[2]],
            [end[0], end[1], end[2]],
        ],
        dtype=float,
    )
    phys = np.array(
        [mask.TransformContinuousIndexToPhysicalPoint(tuple(c)) for c in corners]
    )
    return phys.min(axis=0), phys.max(axis=0)


def _expand_world_bbox(
    min_w: np.ndarray, max_w: np.ndarray, pad_mm: float
) -> Tuple[np.ndarray, np.ndarray]:
    pad = float(pad_mm)
    return min_w - pad, max_w + pad


def _world_bbox_union(mask_a: sitk.Image, mask_b: sitk.Image, pad_mm: float = 0.0):
    """Union of two mask AABBs (world space) with optional per-side padding."""
    a = _bbox_world(mask_a)
    b = _bbox_world(mask_b)
    if a is None and b is None:
        return None
    if a is None:
        min_w, max_w = b
    elif b is None:
        min_w, max_w = a
    else:
        min_w = np.minimum(a[0], b[0])
        max_w = np.maximum(a[1], b[1])
    return _expand_world_bbox(min_w, max_w, pad_mm)


def _compute_crop_spec_from_world_bbox(
    img: sitk.Image,
    world_min: np.ndarray,
    world_max: np.ndarray,
    ensure_odd_vox: bool = True,
) -> Dict[str, Tuple[int, int, int]]:
    """
    Build an ROI (index/size) covering [world_min, world_max] on 'img' grid.
    Expands to integer indices; can force odd sizes to center transforms nicely.
    """
    lo_f = np.array(img.TransformPhysicalPointToContinuousIndex(tuple(world_min)))
    hi_f = np.array(img.TransformPhysicalPointToContinuousIndex(tuple(world_max)))
    lo, hi = np.minimum(lo_f, hi_f), np.maximum(lo_f, hi_f)

    start = np.floor(lo).astype(int)
    stop = np.ceil(hi).astype(int)
    size = np.maximum(1, stop - start).astype(int)

    if ensure_odd_vox:
        size += size % 2 == 0  # make odd

    img_size = np.array(img.GetSize(), int)
    pad_lower = np.maximum(0, -start)
    pad_upper = np.maximum(0, (start + size) - img_size)

    adj_start = start + pad_lower
    padded_size = img_size + pad_lower + pad_upper
    adj_start = np.maximum(0, np.minimum(adj_start, padded_size - size))

    return {
        "roi_index": (int(adj_start[0]), int(adj_start[1]), int(adj_start[2])),
        "roi_size": (int(size[0]), int(size[1]), int(size[2])),
        "pad_lower": (int(pad_lower[0]), int(pad_lower[1]), int(pad_lower[2])),
        "pad_upper": (int(pad_upper[0]), int(pad_upper[1]), int(pad_upper[2])),
    }


def _apply_crop_spec(
    img: sitk.Image, spec: Dict[str, Tuple[int, int, int]], pad_value=0
) -> sitk.Image:
    """Pad+crop image according to 'spec' returned by _compute_crop_spec_from_world_bbox."""
    pl, pu = list(spec["pad_lower"]), list(spec["pad_upper"])
    if any(v > 0 for v in (pl + pu)):
        padf = sitk.ConstantPadImageFilter()
        padf.SetPadLowerBound(pl)
        padf.SetPadUpperBound(pu)
        padf.SetConstant(pad_value)
        img = padf.Execute(img)
    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex(list(spec["roi_index"]))
    roi.SetSize(list(spec["roi_size"]))
    return roi.Execute(img)


def _iter_subtransforms(tx: sitk.Transform):
    if tx.GetName() == "CompositeTransform":
        for i in range(tx.GetNumberOfTransforms()):
            yield tx.GetNthTransform(i)
    else:
        yield tx


def _find_affine_in(tx: sitk.Transform):
    for t in _iter_subtransforms(tx):
        if t.GetName() == "AffineTransform":
            return sitk.AffineTransform(t)
    return None


def _affine_summary(tx: sitk.Transform) -> Dict[str, Optional[tuple]]:
    a = _find_affine_in(tx)
    if a is None:
        return {
            "affine_name": tx.GetName(),
            "affine_matrix": None,
            "affine_translation": None,
            "affine_center": None,
        }
    return {
        "affine_name": "AffineTransform",
        "affine_matrix": tuple(a.GetMatrix()),
        "affine_translation": tuple(a.GetTranslation()),
        "affine_center": tuple(a.GetCenter()),
    }


def bspline_transform_summary(tx: sitk.Transform) -> Dict[str, Optional[tuple]]:
    """Summarize the last BSplineTransform inside 'tx' if present."""
    b = None
    idx = None
    if tx.GetName() == "CompositeTransform":
        for i in range(tx.GetNumberOfTransforms()):
            t = tx.GetNthTransform(i)
            if t.GetName() == "BSplineTransform":
                b, idx = t, i
    elif tx.GetName() == "BSplineTransform":
        b, idx = tx, 0

    if b is None:
        return {
            "composite_name": tx.GetName(),
            "composite_num_subtransforms": (
                tx.GetNumberOfTransforms()
                if tx.GetName() == "CompositeTransform"
                else 0
            ),
            "bspline_found": False,
            "bspline_index": None,
            "bspline_order": None,
            "bspline_mesh_size": None,
            "bspline_origin_mm": None,
            "bspline_phys_dims_mm": None,
            "bspline_ctrl_spacing_mm": None,
            "bspline_direction_rowmajor": None,
            "bspline_num_params": None,
            "bspline_num_ctrl_points_per_axis": None,
            "bspline_total_ctrl_points": None,
        }

    mesh = tuple(b.GetTransformDomainMeshSize())
    origin = tuple(b.GetTransformDomainOrigin())
    phys_dims = tuple(b.GetTransformDomainPhysicalDimensions())
    direction = tuple(b.GetTransformDomainDirection())
    order = int(b.GetOrder())
    n_params = int(b.GetNumberOfParameters())

    ctrl_spacing = tuple((p / m) if m > 0 else 0.0 for p, m in zip(phys_dims, mesh))
    n_ctrl_axis = tuple(m + order for m in mesh)
    total_ctrl = int(math.prod(n_ctrl_axis))

    return {
        "composite_name": tx.GetName(),
        "composite_num_subtransforms": (
            tx.GetNumberOfTransforms() if tx.GetName() == "CompositeTransform" else 0
        ),
        "bspline_found": True,
        "bspline_index": idx,
        "bspline_order": order,
        "bspline_mesh_size": mesh,
        "bspline_origin_mm": origin,
        "bspline_phys_dims_mm": phys_dims,
        "bspline_ctrl_spacing_mm": ctrl_spacing,
        "bspline_direction_rowmajor": direction,
        "bspline_num_params": n_params,
        "bspline_num_ctrl_points_per_axis": n_ctrl_axis,
        "bspline_total_ctrl_points": total_ctrl,
    }


# =============================================================================
# Visualization helpers
# =============================================================================


def myshow(
    img: sitk.Image,
    mask: Optional[Iterable[sitk.Image]] = None,
    title: str = "",
    wl: float = 50,
    ww: float = 300,
    k: int = 4,
    pmin: float = 0.4,
    pmax: float = 0.8,
):
    """
    Show up to k axial slices between percentages pmin..pmax.
    Any provided mask(s) are resampled to 'img' grid and contoured.

    Notes:
    - Handles 3D inputs only.
    - Uses nearest-neighbor for masks and linear for base image.

    Example:
        myshow(portal_img, [portal_liver_mask, portal_tumor_mask], title="Portal")
    """
    if img.GetDimension() != 3:
        raise ValueError("myshow expects a 3D volume.")

    nda = sitk.GetArrayViewFromImage(img)  # (z, y, x)
    depth = nda.shape[0]

    # slice indices
    pmin, pmax = float(np.clip(pmin, 0, 1)), float(np.clip(pmax, 0, 1))
    if pmax < pmin:
        pmin, pmax = pmax, pmin
    percs = (
        [0.5]
        if (k <= 1 or depth <= 1 or abs(pmax - pmin) < 1e-9)
        else np.linspace(pmin, pmax, int(k))
    )
    indices = np.unique(
        np.clip(np.round(np.asarray(percs) * (depth - 1)).astype(int), 0, depth - 1)
    )

    # overlays
    masks_nda = []
    if mask is not None:
        masks = mask if isinstance(mask, (list, tuple)) else [mask]
        for m in masks:
            if m is None:
                continue
            mb = sitk.Cast(m > 0, sitk.sitkUInt8)
            if not _same_grid(img, mb):
                mb = resample_like(
                    img,
                    mb,
                    tx=None,
                    interp=sitk.sitkNearestNeighbor,
                    default=0,
                    pixel_id=sitk.sitkUInt8,
                )
            masks_nda.append(sitk.GetArrayViewFromImage(mb))

    # WL/WW
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    if ww <= 0:
        lo, hi = np.percentile(nda, [5, 95])

    # layout
    n = len(indices)
    ncols = min(max(n, 1), 6)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=60)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    colors = ["b", "r", "g", "y", "c", "m", "w"]

    for pi, (ax, z) in enumerate(zip(axes, indices)):
        sl = nda[z].astype(np.float32)
        sl = np.clip((sl - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        ax.imshow(sl, cmap="gray", interpolation="nearest")
        for i, m in enumerate(masks_nda):
            mz = m[z, :, :]
            if np.any(mz):
                ax.contour(
                    mz > 0, levels=[0.5], colors=colors[i % len(colors)], linewidths=0.9
                )
        ax.set_title(f"{title} - z={z}" if (pi == 0 and title) else f"z={z}")
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    # plt.tight_layout()
    try:
        with plt.ioff():
            fig.savefig(f"{title}.png", dpi=60, bbox_inches="tight")
    finally:
        plt.close(fig)
    return


def bspline_to_displacement_field(
    bspline_tx: sitk.Transform, reference: sitk.Image
) -> sitk.Image:
    """Convert a (nonlinear) transform into a displacement field defined on 'reference'."""
    f = sitk.TransformToDisplacementFieldFilter()
    f.SetReferenceImage(reference)
    f.SetOutputPixelType(sitk.sitkVectorFloat64)
    return f.Execute(bspline_tx)


def plot_displacement_quiver(
    df_img: sitk.Image,
    ref_img: Optional[sitk.Image] = None,
    slice_index: Optional[int] = None,
    step: int = 8,
    scale: float = 1.0,
    show_background: bool = True,
    wl: float = 50,
    ww: float = 300,
):
    """
    Quiver visualization of a displacement field on one axial slice.
    df_img must be a vector image (dx, dy, dz) in mm.

    Example:
        disp = bspline_to_displacement_field(bspline_tx, crop_ref)
        plot_displacement_quiver(disp, ref_img=crop_ref, step=8, scale=1.5)
    """
    dfa = sitk.GetArrayFromImage(df_img)  # (Z, Y, X, 3)
    spacing = np.array(df_img.GetSpacing())  # (sx, sy, sz) mm

    Z, Y, X, _ = dfa.shape
    z = Z // 2 if slice_index is None else int(slice_index)

    # convert mm -> pixels for display
    sx = spacing[0] if spacing[0] else 1.0
    sy = spacing[1] if spacing[1] else 1.0
    dx_pix = dfa[z, :, :, 0] / sx
    dy_pix = dfa[z, :, :, 1] / sy
    yy, xx = np.mgrid[0:Y, 0:X]

    xx_s, yy_s = xx[::step, ::step], yy[::step, ::step]
    u, v = (dx_pix * scale)[::step, ::step], (dy_pix * scale)[::step, ::step]

    plt.figure()
    if show_background and ref_img is not None:
        img = sitk.GetArrayFromImage(ref_img).astype(np.float32)
        sl = img[z, :, :]
        lo, hi = wl - ww / 2.0, wl + ww / 2.0
        sl = np.clip((sl - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        plt.imshow(sl, cmap="gray")

    plt.quiver(
        xx_s,
        yy_s,
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="cyan",
        width=0.002,
    )
    plt.title(f"Displacement field (axial z={z})")
    plt.axis("off")
    plt.show()


# =============================================================================
# Registration
# =============================================================================


def register_arterial_to_portal(
    portal_img,
    arterial_img,
    portal_liver_mask,
    arterial_liver_mask,
    portal_tumor_mask,
    arterial_tumor_mask,
    pad_mm: float = 50.0,
    bspline_ctrl_spacing_mm: float = 35.0,
    band_mm: float = 15.0,  # narrow-band around boundary for DM metric
    verbose: bool = False,  # when True: plot stages (resamples as needed)
    compute_dice: bool = True,
):
    """
    Deformably register arterial -> portal using:
      1) affine init (MI, masked to liver)
      2) B-spline on distance maps (liver-only), narrow-band for speed

    Returns dict with:
      - transform_parameter_object (Selected best: Identity | Affine | Affine ∘ BSpline)
      - log (dice + transform summaries + selected_stage)
    """

    compute_dice = True  # force computing dice, finally used in logic

    # -------------------- small local utils --------------------
    def as_image(x):
        if isinstance(x, sitk.Image):
            return x
        if isinstance(x, str):
            return sitk.ReadImage(x)
        raise ValueError("Inputs must be SimpleITK Images or readable file paths.")

    def bin_mask(i):
        i = sitk.Cast(i, sitk.sitkFloat32)
        return sitk.Cast(sitk.BinaryThreshold(i, 0.5, 1e9, 1, 0), sitk.sitkUInt8)

    def match_geom(mask, ref):
        mask = bin_mask(mask)
        if (
            list(mask.GetSize()) == list(ref.GetSize())
            and mask.GetSpacing() == ref.GetSpacing()
            and mask.GetOrigin() == ref.GetOrigin()
            and mask.GetDirection() == ref.GetDirection()
        ):
            return mask
        return sitk.Resample(
            mask,
            ref,
            sitk.Transform(3, sitk.sitkIdentity),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )

    # -------------------- load / harmonize geometry --------------------
    portal_img = as_image(portal_img)
    arterial_img = as_image(arterial_img)
    portal_liver_mask = match_geom(as_image(portal_liver_mask), portal_img)
    arterial_liver_mask = match_geom(as_image(arterial_liver_mask), arterial_img)
    portal_tumor_mask = match_geom(as_image(portal_tumor_mask), portal_img)
    arterial_tumor_mask = match_geom(as_image(arterial_tumor_mask), arterial_img)

    # 1) Get union bbox in world space across both liver masks (add per-side padding)
    union_bbox = _world_bbox_union(
        portal_liver_mask, arterial_liver_mask, pad_mm=pad_mm
    )
    if verbose:
        print(f"union_bbox : {union_bbox}")

    if union_bbox is None:
        # nothing to crop; proceed with originals
        portal_crop = sitk.Cast(portal_img, sitk.sitkFloat32)
        arterial_crop = sitk.Cast(arterial_img, sitk.sitkFloat32)
        portal_liver_crop = sitk.Cast(portal_liver_mask > 0, sitk.sitkUInt8)
        arterial_liver_crop = sitk.Cast(arterial_liver_mask > 0, sitk.sitkUInt8)
        portal_tumor_crop = sitk.Cast(portal_tumor_mask > 0, sitk.sitkUInt8)
        arterial_tumor_crop = sitk.Cast(arterial_tumor_mask > 0, sitk.sitkUInt8)
    else:
        world_min, world_max = union_bbox

        # 2) Build per-image crop specs from the SAME world bbox
        spec_portal = _compute_crop_spec_from_world_bbox(
            portal_img, world_min, world_max, ensure_odd_vox=True
        )
        spec_arterial = _compute_crop_spec_from_world_bbox(
            arterial_img, world_min, world_max, ensure_odd_vox=True
        )

        if verbose:
            print(f"spec_portal : {spec_portal}")
            print(f"spec_arterial : {spec_arterial}")

        # 3) Apply spec to images and masks (pad_value=0 for images, 0 for masks)
        portal_crop = _apply_crop_spec(
            sitk.Cast(portal_img, sitk.sitkFloat32), spec_portal, pad_value=0
        )
        arterial_crop = _apply_crop_spec(
            sitk.Cast(arterial_img, sitk.sitkFloat32), spec_arterial, pad_value=0
        )

        portal_liver_crop = _apply_crop_spec(
            sitk.Cast(portal_liver_mask > 0, sitk.sitkUInt8), spec_portal, pad_value=0
        )
        arterial_liver_crop = _apply_crop_spec(
            sitk.Cast(arterial_liver_mask > 0, sitk.sitkUInt8),
            spec_arterial,
            pad_value=0,
        )

        portal_tumor_crop = _apply_crop_spec(
            sitk.Cast(portal_tumor_mask > 0, sitk.sitkUInt8), spec_portal, pad_value=0
        )
        arterial_tumor_crop = _apply_crop_spec(
            sitk.Cast(arterial_tumor_mask > 0, sitk.sitkUInt8),
            spec_arterial,
            pad_value=0,
        )

    crop_ref = portal_crop  # output/reference grid

    # -------------------- baseline Dice --------------------
    if compute_dice:
        dice_init_liver = _dice_coeff(portal_liver_crop, arterial_liver_crop)
        dice_init_tumor = _dice_coeff(portal_tumor_crop, arterial_tumor_crop)
        print(
            f"[INIT]    Dice  liver: {dice_init_liver:.4f} | tumor: {dice_init_tumor:.4f}"
        )
    else:
        dice_init_liver = dice_init_tumor = None

    if verbose:
        myshow(portal_crop, [portal_liver_crop, portal_tumor_crop], "Portal crop")
        myshow(
            arterial_crop,
            [arterial_liver_crop, arterial_tumor_crop, portal_tumor_crop],
            "Arterial crop",
        )

    # -------------------- affine (masked to liver) --------------------
    init_tx = sitk.CenteredTransformInitializer(
        portal_liver_crop,
        arterial_liver_crop,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    reg1 = sitk.ImageRegistrationMethod()
    reg1.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    if compute_dice and dice_init_liver is not None and dice_init_liver > 0.9:
        # constrain metric to liver boundaries if segmentation overlaps well
        reg1.SetMetricFixedMask(portal_liver_crop)
        reg1.SetMetricMovingMask(arterial_liver_crop)
    reg1.SetMetricSamplingStrategy(reg1.REGULAR)
    reg1.SetMetricSamplingPercentage(0.4, True)
    reg1.SetInterpolator(sitk.sitkLinear)
    reg1.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=300, relaxationFactor=0.5
    )
    reg1.SetOptimizerScalesFromPhysicalShift()
    reg1.SetShrinkFactorsPerLevel([8, 4, 2])
    reg1.SetSmoothingSigmasPerLevel([4, 2, 1])
    reg1.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg1.SetInitialTransform(init_tx, inPlace=False)
    affine_tx = reg1.Execute(crop_ref, arterial_crop)

    # Dice after affine (on reference grid)
    if compute_dice:
        aff_arterial_liver_in_portal = resample_like(
            crop_ref,
            arterial_liver_crop,
            affine_tx,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        aff_arterial_tumor_in_portal = resample_like(
            crop_ref,
            arterial_tumor_crop,
            affine_tx,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        dice_aff_liver = _dice_coeff(portal_liver_crop, aff_arterial_liver_in_portal)
        dice_aff_tumor = _dice_coeff(portal_tumor_crop, aff_arterial_tumor_in_portal)
        print(
            f"[AFFINE]  Dice  liver: {dice_aff_liver:.4f} | tumor: {dice_aff_tumor:.4f}"
        )
    else:
        aff_arterial_liver_in_portal = aff_arterial_tumor_in_portal = None
        dice_aff_liver = dice_aff_tumor = None

    if verbose:
        aff_arterial_in_portal = resample_like(
            crop_ref,
            arterial_crop,
            affine_tx,
            sitk.sitkLinear,
            0.0,
            arterial_img.GetPixelID(),
        )
        myshow(
            aff_arterial_in_portal,
            [
                aff_arterial_liver_in_portal,
                aff_arterial_tumor_in_portal,
                portal_tumor_crop,
            ],
            "Arterial → Portal (affine)",
        )

    # -------------------- B-spline on distance maps (narrow-band) --------------------
    # Domain for BSpline on portal crop
    phys = [(sz - 1) * sp for sz, sp in zip(crop_ref.GetSize(), crop_ref.GetSpacing())]
    mesh_size = [max(1, int(round(L / float(bspline_ctrl_spacing_mm)))) for L in phys]
    if verbose:
        print(f"B-spline mesh size : {mesh_size} \t phys : {phys}")

    bspline_tx = sitk.BSplineTransformInitializer(crop_ref, mesh_size, order=3)

    if compute_dice and dice_aff_liver is not None and dice_aff_liver > 0.9:
        # REGISTER MASKS
        fixed_dm = sitk.SignedMaurerDistanceMap(portal_liver_crop, True, False, True)
        moving_dm = sitk.SignedMaurerDistanceMap(arterial_liver_crop, True, False, True)
        # limit to ±band_mm to focus on boundary
        fixed_dm = sitk.Clamp(fixed_dm, lowerBound=-band_mm, upperBound=band_mm)
        moving_dm = sitk.Clamp(moving_dm, lowerBound=-band_mm, upperBound=band_mm)

        reg2 = sitk.ImageRegistrationMethod()
        reg2.SetMetricAsMeanSquares()
        reg2.SetInterpolator(sitk.sitkLinear)
        reg2.SetMetricSamplingStrategy(reg2.REGULAR)
        reg2.SetMetricSamplingPercentage(0.4, True)
        reg2.SetOptimizerAsGradientDescentLineSearch(
            learningRate=1.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-3,
            convergenceWindowSize=5,
        )
        reg2.SetOptimizerScalesFromPhysicalShift()
        reg2.SetShrinkFactorsPerLevel([4, 2])
        reg2.SetSmoothingSigmasPerLevel([2, 1])
        reg2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg2.SetMovingInitialTransform(affine_tx)  # affine = fixed pre-warp
        reg2.SetInitialTransform(bspline_tx, inPlace=True)  # optimize this object
        _ = reg2.Execute(fixed_dm, moving_dm)
        final_bspline = bspline_tx  # updated in-place
    else:
        # REGISTER INTENSITIES
        fixed_image = portal_crop
        moving_image = arterial_crop

        reg2 = sitk.ImageRegistrationMethod()
        reg2.SetMetricAsCorrelation()
        reg2.SetInterpolator(sitk.sitkLinear)
        reg2.SetMetricSamplingStrategy(reg2.REGULAR)
        reg2.SetMetricSamplingPercentage(0.4, True)
        reg2.SetOptimizerAsGradientDescentLineSearch(
            learningRate=1.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-3,
            convergenceWindowSize=5,
        )
        reg2.SetOptimizerScalesFromPhysicalShift()
        reg2.SetShrinkFactorsPerLevel([4, 2])
        reg2.SetSmoothingSigmasPerLevel([2, 1])
        reg2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg2.SetMovingInitialTransform(affine_tx)
        reg2.SetInitialTransform(bspline_tx, inPlace=True)
        _ = reg2.Execute(fixed_image, moving_image)
        final_bspline = bspline_tx

    # Compose (affine ∘ bspline)
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(affine_tx)
    composite.AddTransform(final_bspline)

    # -------------------- resample outputs on the portal crop grid --------------------
    if compute_dice:
        # For BSpline
        arterial_in_portal_bs = resample_like(
            crop_ref,
            arterial_crop,
            composite,
            sitk.sitkLinear,
            0.0,
            arterial_img.GetPixelID(),
        )
        arterial_liver_in_portal_bs = resample_like(
            crop_ref,
            arterial_liver_crop,
            composite,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        arterial_tumor_in_portal_bs = resample_like(
            crop_ref,
            arterial_tumor_crop,
            composite,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        dice_bs_liver = _dice_coeff(portal_liver_crop, arterial_liver_in_portal_bs)
        dice_bs_tumor = _dice_coeff(portal_tumor_crop, arterial_tumor_in_portal_bs)
        print(
            f"[BSPLINE] Dice  liver: {dice_bs_liver:.4f} | tumor: {dice_bs_tumor:.4f}"
        )
    else:
        dice_bs_liver = dice_bs_tumor = None

    if verbose and compute_dice:
        myshow(
            arterial_in_portal_bs,
            [
                arterial_liver_in_portal_bs,
                arterial_tumor_in_portal_bs,
                portal_tumor_crop,
            ],
            "Arterial → Portal (affine + B-spline)",
        )

    # -------------------- Select best-performing stage --------------------
    # If Dice isn't computed, keep composite by default.
    selected_stage = "bspline"
    final_transform = composite

    if compute_dice and all(
        v is not None for v in [dice_init_liver, dice_aff_liver, dice_bs_liver]
    ):
        # Prepare candidates with priority for simpler transforms on ties
        eps = 1e-6
        candidates = [
            (
                "init",
                sitk.Transform(3, sitk.sitkIdentity),
                dice_init_liver,
                dice_init_tumor,
                0,
            ),
            ("affine", affine_tx, dice_aff_liver, dice_aff_tumor, 1),
            ("bspline", composite, dice_bs_liver, dice_bs_tumor, 2),
        ]

        # Best by liver Dice, tie-break by tumor Dice, then by simplicity rank (lower is simpler)
        def _score_key(item):
            name, tx, d_liv, d_tum, simplicity_rank = item
            return (float(d_liv), float(d_tum), -simplicity_rank)

        # Manually pick best honoring epsilon and simplicity preference
        best = candidates[0]
        for cand in candidates[1:]:
            dL_best, dT_best = best[2], best[3]
            dL_c, dT_c = cand[2], cand[3]

            if (
                (dL_c > dL_best + eps)
                or (abs(dL_c - dL_best) <= eps and dT_c > dT_best + eps)
                or (
                    abs(dL_c - dL_best) <= eps
                    and abs(dT_c - dT_best) <= eps
                    and cand[4] < best[4]
                )
            ):
                best = cand

        selected_stage, final_transform = best[0], best[1]
        if verbose:
            print(
                f"[SELECT] stage={selected_stage} | Dice(liver) init/aff/bs = "
                f"{dice_init_liver:.4f}/{dice_aff_liver:.4f}/{dice_bs_liver:.4f}"
            )

    # -------------------- optional vector field viz for the selected transform --------------------
    if verbose:
        df_filter = sitk.TransformToDisplacementFieldFilter()
        df_filter.SetReferenceImage(crop_ref)
        df_filter.SetOutputPixelType(sitk.sitkVectorFloat64)
        fwd_df_portal_grid = df_filter.Execute(final_transform)
        plot_displacement_quiver(
            fwd_df_portal_grid, ref_img=crop_ref, step=8, scale=1.5
        )

    # -------------------- logs --------------------
    aff_sum = _affine_summary(affine_tx)
    bs_sum = bspline_transform_summary(composite)
    log = {
        "dice_init_liver": dice_init_liver,
        "dice_init_tumor": dice_init_tumor,
        "dice_aff_liver": dice_aff_liver,
        "dice_aff_tumor": dice_aff_tumor,
        "dice_bs_liver": dice_bs_liver,
        "dice_bs_tumor": dice_bs_tumor,
        **aff_sum,
        **bs_sum,
        "bspline_band_mm": float(band_mm),
        "bspline_ctrl_spacing_mm": float(bspline_ctrl_spacing_mm),
        "selected_stage": selected_stage,
    }

    return {
        "transform_parameter_object": final_transform,  # Identity | Affine | Affine ∘ BSpline
        "log": log,
    }


# =============================================================================
# Main batch: update paths in place (phase == 'arteriel')
# =============================================================================


def register_update_paths_inplace(
    df: Optional[pd.DataFrame] = None,
    df_path: Optional[str] = None,
    out_root: str = "",
    df_save_path: str = "",
    log_save_path: str = "",
    pad_mm: float = 25.0,
    bspline_ctrl_spacing_mm: float = 90.0,
    band_mm: float = 15.0,
    compute_dice: bool = True,
    verbose: bool = False,
    preview: bool = False,
    tumor_mask_strategy: Literal[
        "transfer_portal", "arterial_only", "union", "intersection"
    ] = "transfer_portal",
):
    """
    Pipeline (no new rows):
      - Accepts either a Pandas DataFrame (`df`) or a CSV path (`df_path`)
      - Pivot df to find portal/arteriel paths
      - For each patient: run register_arterial_to_portal(...)
      - Save registered outputs to <out_root>/<patient_key>/      [skipped if preview=True]
      - Update df rows (phase == 'arteriel') with new registered paths
      - Save updated df and a separate log CSV                     [skipped if preview=True]
    """
    # --- Input handling ---
    if df is None and df_path is None:
        raise ValueError("Provide either `df` or `df_path`.")
    if df is None:
        df_path = os.path.expanduser(df_path)
        df = pd.read_csv(df_path)

    # --- Paths + optional dirs ---
    if not preview:
        out_root = os.path.expanduser(out_root)
        df_save_path = os.path.expanduser(df_save_path)
        log_save_path = os.path.expanduser(log_save_path)

        os.makedirs(out_root, exist_ok=True)
        os.makedirs(os.path.dirname(df_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(log_save_path), exist_ok=True)

    # --- Proceed with registration ---
    pv = df.pivot_table(
        index="patient_key",
        columns="phase",
        values=["nifti_path", "liver_path", "liver_tumor_path"],
        aggfunc="first",
    )
    pv = pv.dropna(subset=[("nifti_path", "portal"), ("nifti_path", "arteriel")])

    logs = []

    for pid in tqdm(pv.index, desc="Registering", leave=False):
        print(f"\n >>> Running on Patient {pid} <<< \n")
        try:
            portal_img_p = pv.loc[pid, ("nifti_path", "portal")]
            arteriel_img_p = pv.loc[pid, ("nifti_path", "arteriel")]
            portal_liver_p = (
                pv.loc[pid, ("liver_path", "portal")]
                if ("liver_path", "portal") in pv.columns
                else None
            )
            arteriel_liver_p = (
                pv.loc[pid, ("liver_path", "arteriel")]
                if ("liver_path", "arteriel") in pv.columns
                else None
            )
            portal_tumor_p = (
                pv.loc[pid, ("liver_tumor_path", "portal")]
                if ("liver_tumor_path", "portal") in pv.columns
                else None
            )
            arteriel_tumor_p = (
                pv.loc[pid, ("liver_tumor_path", "arteriel")]
                if ("liver_tumor_path", "arteriel") in pv.columns
                else None
            )

            t0 = time.perf_counter()
            tx = None
            try:
                out = register_arterial_to_portal(
                    portal_img=portal_img_p,
                    arterial_img=arteriel_img_p,
                    portal_liver_mask=portal_liver_p,
                    arterial_liver_mask=arteriel_liver_p,
                    portal_tumor_mask=portal_tumor_p,
                    arterial_tumor_mask=arteriel_tumor_p,
                    pad_mm=pad_mm,
                    bspline_ctrl_spacing_mm=bspline_ctrl_spacing_mm,
                    band_mm=band_mm,
                    verbose=verbose,
                    compute_dice=compute_dice,
                )

                tx = out.get("transform_parameter_object", None)
                L = out.get("log", {})

            except Exception:
                # No frills: if registration raises, there is no output to read from.
                # Proceed with "init" transform (identity) and keep going.
                tx = None
                exec_time_s = 0
                L = {"selected_stage": "init"}

            if tx is None or not isinstance(tx, sitk.Transform):
                tx = sitk.Transform(3, sitk.sitkIdentity)

            exec_time_s = time.perf_counter() - t0

            fixed_portal = sitk.ReadImage(portal_img_p)
            moving_art = sitk.ReadImage(arteriel_img_p)

            # Arterial image registered onto portal grid
            reg_art = sitk.Resample(
                moving_art,
                fixed_portal,
                tx,
                sitk.sitkLinear,
                0.0,
                moving_art.GetPixelID(),
            )

            pat_dir = os.path.join(out_root, str(pid))
            reg_art_p = os.path.join(pat_dir, f"{pid}_arteriel_reg.nii.gz")
            if not preview:
                os.makedirs(pat_dir, exist_ok=True)
                sitk.WriteImage(reg_art, reg_art_p)

            # -------------------------------
            # LIVER masks on portal geometry
            # -------------------------------
            portal_liver_img = None
            if portal_liver_p and os.path.exists(portal_liver_p):
                portal_liver_img_native = sitk.ReadImage(portal_liver_p)
                portal_liver_img_native = sitk.Cast(
                    portal_liver_img_native > 0, sitk.sitkUInt8
                )
                # Resample to fixed_portal with identity so geometry matches exactly
                portal_liver_img = sitk.Resample(
                    portal_liver_img_native,
                    fixed_portal,
                    sitk.Transform(3, sitk.sitkIdentity),
                    sitk.sitkNearestNeighbor,
                    0,
                    sitk.sitkUInt8,
                )

            reg_liver_img, reg_liver_p = None, None
            if arteriel_liver_p and os.path.exists(arteriel_liver_p):
                reg_liver_native = sitk.ReadImage(arteriel_liver_p)
                reg_liver_native = sitk.Cast(reg_liver_native > 0, sitk.sitkUInt8)
                reg_liver_img = sitk.Resample(
                    reg_liver_native,
                    fixed_portal,
                    tx,
                    sitk.sitkNearestNeighbor,
                    0,
                    sitk.sitkUInt8,
                )
                reg_liver_p = os.path.join(pat_dir, f"{pid}_arteriel_reg_liver.nii.gz")
                if not preview:
                    sitk.WriteImage(reg_liver_img, reg_liver_p)

            # --------------------------------
            # TUMOR masks on portal geometry
            # --------------------------------
            tumor_portal_on_reg_img = None
            if portal_tumor_p and os.path.exists(portal_tumor_p):
                tumor_portal_native = sitk.ReadImage(portal_tumor_p)
                tumor_portal_native = sitk.Cast(tumor_portal_native > 0, sitk.sitkUInt8)
                tumor_portal_on_reg_img = sitk.Resample(
                    tumor_portal_native,
                    fixed_portal,
                    sitk.Transform(3, sitk.sitkIdentity),
                    sitk.sitkNearestNeighbor,
                    0,
                    sitk.sitkUInt8,
                )

            tumor_arterial_reg_img = None
            if arteriel_tumor_p and os.path.exists(arteriel_tumor_p):
                tumor_arterial_native = sitk.ReadImage(arteriel_tumor_p)
                tumor_arterial_native = sitk.Cast(
                    tumor_arterial_native > 0, sitk.sitkUInt8
                )
                tumor_arterial_reg_img = sitk.Resample(
                    tumor_arterial_native,
                    fixed_portal,
                    tx,
                    sitk.sitkNearestNeighbor,
                    0,
                    sitk.sitkUInt8,
                )

            # STRATEGY (all images now share fixed_portal geometry)
            final_tumor_img = None
            if tumor_mask_strategy == "transfer_portal":
                final_tumor_img = tumor_portal_on_reg_img or tumor_arterial_reg_img
            elif tumor_mask_strategy == "arterial_only":
                final_tumor_img = tumor_arterial_reg_img or tumor_portal_on_reg_img
            elif tumor_mask_strategy == "union":
                if (
                    tumor_portal_on_reg_img is not None
                    and tumor_arterial_reg_img is not None
                ):
                    final_tumor_img = sitk.Cast(
                        sitk.Or(
                            tumor_portal_on_reg_img > 0, tumor_arterial_reg_img > 0
                        ),
                        sitk.sitkUInt8,
                    )
                else:
                    final_tumor_img = tumor_portal_on_reg_img or tumor_arterial_reg_img
            elif tumor_mask_strategy == "intersection":
                if (
                    tumor_portal_on_reg_img is not None
                    and tumor_arterial_reg_img is not None
                ):
                    final_tumor_img = sitk.Cast(
                        sitk.And(
                            tumor_portal_on_reg_img > 0, tumor_arterial_reg_img > 0
                        ),
                        sitk.sitkUInt8,
                    )
                else:
                    final_tumor_img = tumor_portal_on_reg_img or tumor_arterial_reg_img
            else:
                raise ValueError(f"Unknown tumor_mask_strategy: {tumor_mask_strategy}")

            # TRIM TO LIVER (prefer registered arterial liver; fallback to portal liver)
            if final_tumor_img is not None and reg_liver_img is not None:
                final_tumor_img = sitk.Cast(
                    sitk.And(final_tumor_img > 0, reg_liver_img > 0), sitk.sitkUInt8
                )

            # SAVE final tumor (skip if preview)
            final_tumor_p = None
            if final_tumor_img is not None:
                final_tumor_p = os.path.join(
                    pat_dir, f"{pid}_arteriel_reg_tumor.nii.gz"
                )
                if not preview:
                    sitk.WriteImage(final_tumor_img, final_tumor_p)

            # PLOT if preview
            if preview:
                myshow(
                    fixed_portal,
                    [portal_liver_img, tumor_portal_on_reg_img],
                    crop_first_mask=True,
                    title="Original Portal",
                )
                myshow(
                    reg_art,
                    [reg_liver_img, final_tumor_img],
                    crop_first_mask=True,
                    title="Final Arterial",
                )

            # Update in-memory DataFrame paths (note: in preview mode, these files won't exist)
            if not preview:
                mask_df = (df["patient_key"] == pid) & (df["phase"] == "arteriel")
                df.loc[mask_df, "nifti_path"] = reg_art_p
                if reg_liver_img is not None:
                    df.loc[mask_df, "liver_path"] = reg_liver_p
                if final_tumor_p is not None:
                    df.loc[mask_df, "liver_tumor_path"] = final_tumor_p

            logs.append(
                {
                    "patient_key": pid,
                    **L,
                    "exec_time_s": exec_time_s,
                    "tumor_mask_strategy": tumor_mask_strategy,
                    "status": "ok",
                    "error": None,
                }
            )

        except Exception as e:
            print(traceback.format_exc())
            logs.append({"patient_key": pid, "status": "error", "error": str(e)})

    log_df = pd.DataFrame(logs)

    if not preview:
        df.to_csv(df_save_path, index=False)
        log_df.to_csv(log_save_path, index=False)

    return df, log_df


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Portal-to-arterial tumor registration."
    )
    parser.add_argument("--df_path", required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--out_root",
        required=True,
        help="Output root directory for registered NIfTI files.",
    )
    parser.add_argument(
        "--df_save_path", required=True, help="Path to save the updated DataFrame CSV."
    )
    parser.add_argument(
        "--log_save_path", required=True, help="Path to save the log CSV."
    )
    parser.add_argument(
        "--pad_mm", type=float, default=25.0, help="Padding margin in mm."
    )
    parser.add_argument(
        "--bspline_ctrl_spacing_mm",
        type=float,
        default=90.0,
        help="B-spline control point spacing in mm.",
    )
    parser.add_argument(
        "--band_mm", type=float, default=15.0, help="Distance map band thickness in mm."
    )
    parser.add_argument(
        "--compute_dice", type=lambda x: x.lower() in ["true", "1", "yes"], default=True
    )
    parser.add_argument(
        "--verbose", type=lambda x: x.lower() in ["true", "1", "yes"], default=False
    )

    args = parser.parse_args()
    print(args)

    updated_df, log_df = register_update_paths_inplace(
        df_path=args.df_path,
        out_root=args.out_root,
        df_save_path=args.df_save_path,
        log_save_path=args.log_save_path,
        pad_mm=args.pad_mm,
        bspline_ctrl_spacing_mm=args.bspline_ctrl_spacing_mm,
        band_mm=args.band_mm,
        compute_dice=args.compute_dice,
        verbose=args.verbose,
    )
