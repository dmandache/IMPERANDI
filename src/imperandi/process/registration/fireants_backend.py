"""Optional FireANTs GPU backend for linear signed-distance registration.

FireANTs and SimpleITK both expose the transform used for resampling: a pull
mapping from a point on the fixed/output grid to physical space in the moving
image.  Keeping that convention explicit here prevents image direction and
anisotropic spacing from leaking into the transform conversion.
"""

from dataclasses import dataclass, field
import time

import numpy as np


class FireANTsUnavailable(RuntimeError):
    """FireANTs was requested but its optional CUDA runtime is unavailable."""


def _imports():
    try:
        import torch
        from fireants.io.image import BatchedImages, Image
        from fireants.registration.affine import AffineRegistration
        from fireants.registration.rigid import RigidRegistration
    except (ImportError, OSError) as exc:
        raise FireANTsUnavailable(
            "FireANTs is not installed; install 'imperandi[fireants]'"
        ) from exc
    try:
        available = bool(torch.cuda.is_available())
    except (RuntimeError, OSError) as exc:
        raise FireANTsUnavailable(f"CUDA initialization failed: {exc}") from exc
    if not available:
        raise FireANTsUnavailable("FireANTs requires a CUDA-capable PyTorch runtime")
    return torch, Image, BatchedImages, RigidRegistration, AffineRegistration


def sitk_transform_to_physical_matrix(transform) -> np.ndarray:
    """Convert a centered SimpleITK linear transform to ``y = A x + b``."""
    matrix = np.asarray(transform.GetMatrix(), dtype=float).reshape(3, 3)
    translation = np.asarray(transform.GetTranslation(), dtype=float)
    center = (
        np.asarray(transform.GetCenter(), dtype=float)
        if hasattr(transform, "GetCenter")
        else np.zeros(3, dtype=float)
    )
    result = np.eye(4, dtype=float)
    result[:3, :3] = matrix
    result[:3, 3] = translation + center - matrix @ center
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite initial physical transform")
    return result


def physical_matrix_to_sitk(matrix, *, center, rigid):
    """Convert ``y = A x + b`` to an equivalent centered SimpleITK transform."""
    import SimpleITK as sitk

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("FireANTs returned an invalid physical transform matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5, rtol=0):
        raise ValueError("FireANTs returned a non-affine homogeneous transform")
    linear = matrix[:3, :3]
    center = np.asarray(center, dtype=float)
    translation = matrix[:3, 3] - center + linear @ center
    transform = sitk.Euler3DTransform() if rigid else sitk.AffineTransform(3)
    transform.SetCenter(center.tolist())
    if rigid:
        transform.SetMatrix(linear.ravel().tolist(), 1e-4)
    else:
        transform.SetMatrix(linear.ravel().tolist())
    transform.SetTranslation(translation.tolist())
    return transform


def coarse_image(image, spacing_mm, outside_value):
    """Resample on a coarser lattice while preserving physical end points."""
    import SimpleITK as sitk

    old_size = np.asarray(image.GetSize(), dtype=int)
    old_spacing = np.asarray(image.GetSpacing(), dtype=float)
    requested = np.maximum(old_spacing, float(spacing_mm))
    extent = np.maximum(old_size - 1, 0) * old_spacing
    new_size = np.maximum(2, np.floor(extent / requested).astype(int) + 1)
    new_spacing = np.divide(
        extent,
        new_size - 1,
        out=requested.copy(),
        where=(new_size - 1) > 0,
    )
    reference = sitk.Image([int(value) for value in new_size], sitk.sitkFloat32)
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    reference.SetSpacing([float(value) for value in new_spacing])
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        float(outside_value),
        sitk.sitkFloat32,
    )


@dataclass
class FireANTsContext:
    torch: object
    fixed: object
    moving: object
    fixed_batch: object
    moving_batch: object
    fixed_size: tuple[int, ...]
    moving_size: tuple[int, ...]
    fixed_spacing: tuple[float, ...]
    moving_spacing: tuple[float, ...]
    preparation_elapsed_seconds: float
    reference_gpu_data_reused: bool
    uses: int = 0


@dataclass
class OptimizationReport:
    stop: str
    iteration: int
    diagnostics: dict = field(default_factory=dict)

    def GetOptimizerStopConditionDescription(self):
        return self.stop

    def GetOptimizerIteration(self):
        return self.iteration


def prepare_context(
    fixed_dm, moving_dm, config, reference_cache=None
) -> FireANTsContext:
    """Upload one coarse distance-field pair for reuse by rigid and affine."""
    started = time.perf_counter()
    torch, Image, BatchedImages, _, _ = _imports()
    outside = float(config.organ_boundary_band_half_width_mm)
    reference_cache = {} if reference_cache is None else reference_cache
    cached = reference_cache.get("fireants_fixed")
    reference_reused = cached is not None
    if cached is None:
        fixed_coarse = coarse_image(
            fixed_dm, config.fireants_coarse_spacing_mm, outside
        )
        try:
            fixed = Image(fixed_coarse, device="cuda")
            fixed_batch = BatchedImages([fixed])
        except (RuntimeError, OSError) as exc:
            raise FireANTsUnavailable(f"CUDA image preparation failed: {exc}") from exc
        cached = {
            "image": fixed,
            "batch": fixed_batch,
            "size": tuple(fixed_coarse.GetSize()),
            "spacing": tuple(fixed_coarse.GetSpacing()),
        }
        reference_cache["fireants_fixed"] = cached
    moving_coarse = coarse_image(moving_dm, config.fireants_coarse_spacing_mm, outside)
    try:
        moving = Image(moving_coarse, device="cuda")
        moving_batch = BatchedImages([moving])
        torch.cuda.synchronize()
    except (RuntimeError, OSError) as exc:
        raise FireANTsUnavailable(f"CUDA image preparation failed: {exc}") from exc
    return FireANTsContext(
        torch=torch,
        fixed=cached["image"],
        moving=moving,
        fixed_batch=cached["batch"],
        moving_batch=moving_batch,
        fixed_size=cached["size"],
        moving_size=tuple(moving_coarse.GetSize()),
        fixed_spacing=cached["spacing"],
        moving_spacing=tuple(moving_coarse.GetSpacing()),
        preparation_elapsed_seconds=time.perf_counter() - started,
        reference_gpu_data_reused=reference_reused,
    )


def refine(context, initial, config, *, affine):
    """Optimize one physical pull transform and return a SimpleITK transform."""
    torch, _, _, RigidRegistration, AffineRegistration = _imports()
    initial_matrix = sitk_transform_to_physical_matrix(initial)
    init = torch.as_tensor(initial_matrix[None], dtype=torch.float32, device="cuda")
    kwargs = dict(
        scales=[1],
        iterations=[config.maximum_optimizer_iterations],
        fixed_images=context.fixed_batch,
        moving_images=context.moving_batch,
        loss_type="mse",
        optimizer="Adam",
        optimizer_lr=(
            config.fireants_affine_learning_rate
            if affine
            else config.fireants_rigid_learning_rate
        ),
        progress_bar=False,
    )
    if affine:
        registration = AffineRegistration(init_rigid=init, **kwargs)
        matrix_getter = registration.get_affine_matrix
    else:
        registration = RigidRegistration(
            init_translation=init[:, :3, 3],
            init_moment=init[:, :3, :3],
            **kwargs,
        )
        matrix_getter = registration.get_rigid_matrix

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    registration.optimize()
    torch.cuda.synchronize()
    optimization_elapsed = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    result_matrix = matrix_getter().detach().cpu().numpy()[0]
    context.uses += 1
    center = initial.GetCenter() if hasattr(initial, "GetCenter") else (0.0, 0.0, 0.0)
    transform = physical_matrix_to_sitk(result_matrix, center=center, rigid=not affine)
    diagnostics = {
        "backend": "fireants",
        "algorithm": "AffineRegistration" if affine else "RigidRegistration",
        "optimization_elapsed_seconds": optimization_elapsed,
        "gpu_peak_memory_allocated_bytes": peak_allocated,
        "gpu_peak_memory_reserved_bytes": peak_reserved,
        "coarse_fixed_size": list(context.fixed_size),
        "coarse_moving_size": list(context.moving_size),
        "coarse_fixed_spacing_mm": list(context.fixed_spacing),
        "coarse_moving_spacing_mm": list(context.moving_spacing),
        "gpu_context_reused": context.uses > 1,
        "reference_gpu_data_reused": context.reference_gpu_data_reused,
        "gpu_preparation_elapsed_seconds": (
            context.preparation_elapsed_seconds if context.uses == 1 else 0.0
        ),
        "optimizer_iteration_limit": config.maximum_optimizer_iterations,
    }
    return transform, OptimizationReport(
        "FireANTs optimization completed; actual iteration count unavailable",
        -1,
        diagnostics,
    )
