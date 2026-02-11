"""segment.py
=================
Batch‑process a list of 3‑D volumes to obtain masks with a configurable
segmentation backend (default: TotalSegmentator v2).

The module supports both CLI and library usage:

1. reads a CSV containing a ``nifti_path`` column,
2. spawns a multiprocessing pool (``spawn`` context – required for
   PyTorch + CUDA),
3. runs config‑driven segmentation tasks per volume,
4. optionally merges / cleans masks, and
5. writes updated CSVs with output paths and a separate error CSV.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.measure import label, regionprops
from skimage.morphology import ball
from tqdm import tqdm

from imperandi.utils.logging import setup_logging
from imperandi.utils.manifest import DEFAULT_MANIFEST_NAME, load_manifest
from imperandi.utils.misc import report_volumes  # type: ignore

# -----------------------------------------------------------------------------
# Configuration & logging
# -----------------------------------------------------------------------------
# Path where TotalSegmentator models are cached (edit as needed)
# os.environ.setdefault("TOTALSEG_HOME_DIR", str(Path.home() / ".totalsegmentator_v2"))

DEFAULT_TIMEOUT = 15 * 60  # seconds – hard wall per study inside the pool

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------


def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    """Return image data, affine matrix and voxel sizes (zoom)."""
    img = nib.load(str(path))
    return img.get_fdata(), img.affine, img.header.get_zooms()


def save_nifti(
    data: np.ndarray, affine: np.ndarray, out_path: Path, *, dtype=np.uint8
) -> None:
    """Write *data* to *out_path* as a NIfTI‑1 file with *dtype*."""
    img = nib.Nifti1Image(data.astype(dtype, copy=False), affine)
    nib.save(img, str(out_path))


def compute_struct_elem(zooms: Tuple[float, ...], radius_mm: float = 5.0) -> np.ndarray:
    """Create a spherical structuring element with *radius_mm* in real units."""
    radii_vox = [max(1, int(round(radius_mm / z))) for z in zooms]
    return ball(max(radii_vox))


# -----------------------------------------------------------------------------
# Mask post‑processing
# -----------------------------------------------------------------------------


def clean_and_merge_masks(
    dir_path: Path,
    mask_files: List[str],
    *,
    output_name: str,
    radius_mm: float = 5.0,
    verbose: bool = False,
    close: bool = True,
    fill_holes: bool = True,
    largest_cc: bool = True,
) -> bool:
    """Merge masks and optionally apply morphological cleanup."""

    masks: Dict[str, np.ndarray] = {}
    ref_affine: np.ndarray | None = None
    voxel_zooms: Tuple[float, ...] | None = None

    for fname in mask_files:
        src = dir_path / fname
        if not src.exists():
            logger.warning("Mask missing – skipping merge: %s", src)
            continue

        data, affine, zooms = load_nifti(src)
        if ref_affine is None:
            ref_affine, voxel_zooms = affine, zooms
        elif not np.allclose(ref_affine, affine):
            logger.error("Affine mismatch for %s – aborting merge.", src.name)
            return False
        masks[fname] = data > 0

    if not masks:
        logger.error("No valid masks found to merge in %s", dir_path)
        return False

    if len({m.shape for m in masks.values()}) > 1:
        logger.error("Mask shape mismatch in %s – aborting merge.", dir_path)
        return False

    merged = np.logical_or.reduce(list(masks.values()))
    if close:
        merged = binary_closing(
            merged, structure=compute_struct_elem(voxel_zooms, radius_mm)
        )
    if fill_holes:
        merged = binary_fill_holes(merged)

    # Keep only the largest connected component (CC)
    if largest_cc:
        labeled, n_cc = label(merged, return_num=True)
        if n_cc > 1:
            largest = max(regionprops(labeled), key=lambda r: r.area)
            merged = labeled == largest.label
            if verbose:
                logger.info(
                    f"{dir_path} : kept largest CC ({largest.area} voxels) out of {n_cc}"
                )
        elif verbose:
            logger.info(f"{dir_path} : single connected component")

    save_nifti(merged, ref_affine, dir_path / output_name)

    # Optional: overwrite the originals with their cleaned‑up version
    for fname, mask in masks.items():
        save_nifti(mask & merged, ref_affine, dir_path / fname)

    return True


# -----------------------------------------------------------------------------
# Segmentation of one 3‑D volume
# -----------------------------------------------------------------------------


class TotalSegmentatorBackend:
    """Thin wrapper for TotalSegmentator to keep dependency optional."""

    def __init__(self) -> None:
        self._ts = None

    def _ensure_imported(self) -> None:
        if self._ts is None:
            from totalsegmentator.python_api import totalsegmentator

            self._ts = totalsegmentator

    def run(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        task: str,
        fast: bool,
        **kwargs: Any,
    ) -> None:
        self._ensure_imported()
        self._ts(
            input=input_path,
            output=output_dir,
            task=task,
            fast=fast,
            quiet=True,
            verbose=False,
            **kwargs,
        )


def _default_tasks_config() -> Dict[str, Any]:
    return {
        "backend": "totalsegmentator",
        "tasks": [
            {
                "key": "liver",
                "task": "total",
                "output": "liver.nii.gz",
                "extra": {"roi_subset_robust": ["liver"]},
            },
            {
                "key": "liver_tumor",
                "task": "liver_vessels",
                "output": "liver_tumor.nii.gz",
                "extra": {},
            },
        ],
        "postprocess": {
            "merge_keys": ["liver", "liver_tumor"],
            "output": "liver_all.nii.gz",
            "radius_mm": 5.0,
            "largest_cc": True,
            "fill_holes": True,
            "close": True,
        },
    }


def load_tasks_config(
    path: Path | None,
    *,
    manifest: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Load segmentation tasks from JSON, then manifest, then builtin defaults."""
    if path is None:
        manifest_segmentation = (manifest or {}).get("segmentation")
        if manifest_segmentation is not None:
            if not isinstance(manifest_segmentation, dict):
                raise ValueError("Manifest key 'segmentation' must be a JSON object.")
            return copy.deepcopy(manifest_segmentation)
        return _default_tasks_config()

    if not path.exists():
        raise FileNotFoundError(f"Tasks config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prefetch_totalsegmentator_models(
    tasks_config: Dict[str, Any], *, fast: bool
) -> None:
    """Download required TotalSegmentator weights before multiprocessing."""
    if tasks_config.get("backend", "totalsegmentator") != "totalsegmentator":
        return

    tasks = tasks_config.get("tasks", [])
    if not tasks:
        return

    task_to_id = {
        "total": [291, 292, 293, 294, 295],
        "total_fast": [297, 298],
        "total_mr": [850, 851],
        "total_fast_mr": [852, 853],
        "lung_vessels": [258],
        "cerebral_bleed": [150],
        "hip_implant": [260],
        "pleural_pericard_effusion": [315],
        "body": [299],
        "body_fast": [300],
        "body_mr": [597],
        "body_mr_fast": [598],
        "vertebrae_mr": [756],
        "head_glands_cavities": [775],
        "headneck_bones_vessels": [776],
        "head_muscles": [777],
        "headneck_muscles": [778, 779],
        "liver_vessels": [8],
        "lung_nodules": [913],
        "kidney_cysts": [789],
        "oculomotor_muscles": [351],
        "breasts": [527],
        "ventricle_parts": [552],
        "liver_segments": [570],
        "liver_segments_mr": [576],
        "craniofacial_structures": [115],
        "abdominal_muscles": [952],
        "teeth": [113],
        "trunk_cavities": [343],
        "brain_aneurysm": [615],
        "heartchambers_highres": [301],
        "appendicular_bones": [304],
        "appendicular_bones_mr": [855],
        "tissue_types": [481],
        "tissue_types_mr": [925],
        "tissue_4_types": [485],
        "vertebrae_body": [305],
        "face": [303],
        "face_mr": [856],
        "brain_structures": [409],
        "thigh_shoulder_muscles": [857],
        "thigh_shoulder_muscles_mr": [857],
        "coronary_arteries": [507],
    }

    task_names = {task["task"] for task in tasks if "task" in task}
    if not task_names:
        return

    resolved_tasks: List[str] = []
    for name in sorted(task_names):
        if name == "total" and fast:
            resolved_tasks.append("total_fast")
        elif name == "total_mr" and fast:
            resolved_tasks.append("total_fast_mr")
        elif name == "body" and fast:
            resolved_tasks.append("body_fast")
        elif name == "body_mr" and fast:
            resolved_tasks.append("body_mr_fast")
        else:
            resolved_tasks.append(name)

    missing = [name for name in resolved_tasks if name not in task_to_id]
    if missing:
        logger.warning(
            "Skipping model prefetch for unknown tasks: %s", ", ".join(missing)
        )

    task_ids: List[int] = []
    for name in resolved_tasks:
        ids = task_to_id.get(name)
        if not ids:
            continue
        task_ids.extend(ids)

    if not task_ids:
        return

    from totalsegmentator.python_api import download_pretrained_weights

    logger.info(
        "Prefetching TotalSegmentator models for tasks: %s",
        ", ".join(resolved_tasks),
    )
    for task_id in sorted(set(task_ids)):
        download_pretrained_weights(task_id)


def segment_volume(
    nifti_path: Path,
    output_dir: Path,
    tasks_config: Dict[str, Any],
    *,
    fast: bool,
    verbose: bool = False,
    force: bool = False,
    backend: TotalSegmentatorBackend | None = None,
) -> List[str]:
    """Run segmentation tasks and optional post‐processing."""
    warnings: List[str] = []
    tasks = tasks_config.get("tasks", [])
    if not tasks:
        raise ValueError("No tasks provided in config")

    backend_name = tasks_config.get("backend", "totalsegmentator")
    if backend_name != "totalsegmentator":
        raise ValueError(f"Unsupported backend: {backend_name}")

    backend = backend or TotalSegmentatorBackend()

    for task in tasks:
        task_name = task["task"]
        task_output = task["output"]
        extra = task.get("extra", {})

        dst = output_dir / task_output
        if dst.exists() and not force:
            if verbose:
                logger.info("Skip %s – file exists", dst)
            continue

        try:
            backend.run(
                input_path=nifti_path,
                output_dir=output_dir,
                task=task_name,
                fast=fast,
                **extra,
            )
        except Exception as exc:
            logger.error(
                "Segmentation failed on %s (%s): %s", nifti_path, task_name, exc
            )
            raise

        if not dst.exists():
            raise RuntimeError(f"Expected mask not produced: {dst}")
        else:
            logger.info("Mask saved at %s", dst)

    postprocess = tasks_config.get("postprocess")
    if not postprocess:
        return warnings

    merge_keys = postprocess.get("merge_keys", [])
    if not merge_keys:
        return warnings

    key_to_output = {task["key"]: task["output"] for task in tasks}
    merge_files = [key_to_output[k] for k in merge_keys if k in key_to_output]
    if not merge_files:
        return warnings

    merged_name = postprocess.get("output", "mask_merged.nii.gz")
    dst = output_dir / merged_name
    if dst.exists() and not force:
        if verbose:
            logger.info("Skip %s – file exists", dst)
        return warnings

    on_failure = str(postprocess.get("on_failure", "warn_only")).strip().lower()
    if on_failure not in {"warn_only", "fail"}:
        raise ValueError(
            f"Invalid postprocess.on_failure='{on_failure}'. Use 'warn_only' or 'fail'."
        )

    merged_ok = clean_and_merge_masks(
        output_dir,
        merge_files,
        output_name=merged_name,
        radius_mm=float(postprocess.get("radius_mm", 5.0)),
        verbose=verbose,
        close=bool(postprocess.get("close", True)),
        fill_holes=bool(postprocess.get("fill_holes", True)),
        largest_cc=bool(postprocess.get("largest_cc", True)),
    )
    if not merged_ok or not dst.exists():
        message = f"Postprocess merge did not produce expected output: {dst}"
        if on_failure == "fail":
            raise RuntimeError(message)
        logger.warning(message)
        warnings.append(message)

    return warnings


# -----------------------------------------------------------------------------
# Worker wrapper (called in pool)
# -----------------------------------------------------------------------------


def process_single_volume(
    idx: int,
    row: Dict[str, Any],  # must be JSON‑serialisable
    tasks_config: Dict[str, Any],
    *,
    fast: bool,
    verbose: bool,
    force: bool,
    backend: TotalSegmentatorBackend | None = None,
) -> Tuple[int, str | None, str | None, str | None]:
    """Return ``(idx, output_dir|None, error_msg|None, warning_msg|None)``."""

    setup_logging(verbose=verbose)

    try:
        nifti_path = Path(row["nifti_path"])
    except KeyError:
        return idx, None, "column 'nifti_path' missing", None

    if not nifti_path.exists():
        return idx, None, "file not found", None

    try:
        warnings = segment_volume(
            nifti_path,
            nifti_path.parent,
            tasks_config,
            fast=fast,
            verbose=verbose,
            force=force,
            backend=backend,
        )
        warning_msg = " | ".join(warnings) if warnings else None
        return idx, str(nifti_path.parent), None, warning_msg
    except Exception as exc:
        # Capture full traceback for later debugging
        logger.debug("Traceback for %s:\n%s", nifti_path.name, traceback.format_exc())
        return idx, None, str(exc), None


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def add_segment_arguments(
    parser: argparse.ArgumentParser,
    include_manifest: bool = True,
    include_dry_run: bool = True,
) -> None:
    parser.add_argument(
        "csv_path_pos",
        nargs="?",
        type=str,
        default=None,
        help="Path to the input CSV file. Defaults to ./nifti_index.csv.",
    )
    parser.add_argument(
        "--csv_path",
        dest="csv_path_opt",
        type=str,
    )
    parser.add_argument(
        "--csv_path_out",
        type=str,
        required=False,
        default=None,
        help="Output CSV (default: overwrite input).",
    )
    parser.add_argument(
        "--error_csv_path",
        type=str,
        default=None,
        help="CSV for failures only (default: alongside input CSV).",
    )
    parser.add_argument(
        "--tasks_config",
        type=str,
        default=None,
        help=(
            "JSON config for segmentation tasks. "
            "If omitted, use the manifest's segmentation section."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Pool size")
    parser.add_argument(
        "--fast", action="store_true", help="Use TotalSegmentator fast mode"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re‑run even if output masks already exist",
    )
    parser.add_argument(
        "--start_method",
        choices=["spawn", "fork", "forkserver"],
        default="spawn",
        help="multiprocessing start method: spawn=robust, fork=faster (Linux)",
    )
    parser.add_argument(
        "--timeout_sec",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-volume timeout in seconds",
    )
    if include_manifest:
        parser.add_argument(
            "--manifest",
            type=str,
            default=DEFAULT_MANIFEST_NAME,
            help=(
                "Dataset manifest name or path to manifest JSON "
                f"(default: {DEFAULT_MANIFEST_NAME})."
            ),
        )
    if include_dry_run:
        parser.add_argument(
            "--dry-run",
            dest="dry_run",
            action="store_true",
            default=False,
            help="Print planned actions without running.",
        )


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch segmentation with TotalSegmentator v2",
        add_help=add_help,
    )
    add_segment_arguments(parser)
    return parser


def normalize_segment_args(args: argparse.Namespace) -> argparse.Namespace:
    csv_in = args.csv_path_opt if args.csv_path_opt is not None else args.csv_path_pos

    if csv_in is None:
        csv_path = Path.cwd() / "nifti_index.csv"
    else:
        csv_path = Path(csv_in)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    args.csv_path = str(csv_path.resolve())

    if not args.csv_path_out:
        args.csv_path_out = args.csv_path
    else:
        args.csv_path_out = str(Path(args.csv_path_out))

    if args.error_csv_path:
        args.error_csv_path = str(Path(args.error_csv_path))
    else:
        args.error_csv_path = str(Path(args.csv_path).parent / "seg_errors.csv")

    if args.tasks_config:
        args.tasks_config = str(Path(args.tasks_config))

    del args.csv_path_pos
    del args.csv_path_opt

    return args


def main(args: argparse.Namespace) -> None:
    setup_logging(verbose=getattr(args, "verbose", False))
    manifest = load_manifest(
        getattr(args, "manifest", None), base_path=Path(__file__).resolve().parents[1]
    )
    tasks_config = load_tasks_config(
        Path(args.tasks_config) if args.tasks_config else None,
        manifest=manifest,
    )
    prefetch_totalsegmentator_models(tasks_config, fast=args.fast)

    # --- read and pre‑clean CSV ------------------------------------------------
    df = pd.read_csv(args.csv_path).copy()
    if "nifti_path" not in df.columns:
        unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
        if unnamed:
            df = df.drop(columns=unnamed)
    if "nifti_path" not in df.columns:
        raise KeyError("column 'nifti_path' missing")
    df = df.drop_duplicates("nifti_path").copy()
    for task in tasks_config.get("tasks", []):
        df[f"mask_{task['key']}"] = None
    if tasks_config.get("postprocess"):
        df["mask_merged"] = None
    df["warning_message"] = None

    # --- spawn multiprocessing pool -------------------------------------------
    ctx = mp.get_context(
        args.start_method
    )  # 'spawn' required for torch / CUDA stability

    results: List[Tuple[int, str | None, str | None, str | None]] = []

    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(
                process_single_volume,
                idx,
                row.to_dict(),
                tasks_config,
                fast=args.fast,
                verbose=args.verbose,
                force=args.force,
            ): idx
            for idx, row in df.iterrows()
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Segment"):
            i = futures[fut]
            try:
                res = fut.result(timeout=args.timeout_sec)
            except TimeoutError:
                logger.warning("Row %d exceeded %ds – killed", i, args.timeout_sec)
                fut.cancel()
                res = (i, None, "timeout", None)
            except Exception as exc:
                res = (i, None, f"worker crash: {exc}", None)
            results.append(res)

        # Force‑kill any orphaned processes (≤ Python 3.10)
        pool.shutdown(wait=False, cancel_futures=True)
        processes = getattr(pool, "_processes", None)
        if processes:
            for p in processes.values():
                if p.is_alive():
                    p.kill()

    # --- consolidate results ---------------------------------------------------
    errors: List[Dict[str, Any]] = []
    for idx, out_dir, err_msg, warning_msg in results:
        if out_dir:
            base = Path(out_dir)
            row_warnings: List[str] = []
            for task in tasks_config.get("tasks", []):
                mask_path = base / task["output"]
                if mask_path.exists():
                    df.at[idx, f"mask_{task['key']}"] = str(mask_path)
                else:
                    row_warnings.append(f"missing mask: {mask_path}")
            if tasks_config.get("postprocess"):
                merged_name = tasks_config["postprocess"].get(
                    "output", "mask_merged.nii.gz"
                )
                merged_path = base / merged_name
                if merged_path.exists():
                    df.at[idx, "mask_merged"] = str(merged_path)
                else:
                    row_warnings.append(f"missing merged mask: {merged_path}")
            if warning_msg:
                row_warnings.append(warning_msg)
            if row_warnings:
                df.at[idx, "warning_message"] = " | ".join(row_warnings)
        else:
            errors.append({"idx": idx, "error_message": err_msg or "unknown"})

    # --- write output tables ---------------------------------------------------
    df.to_csv(args.csv_path_out, index=False)
    logger.info("Wrote main table → %s", args.csv_path_out)

    if errors:
        err_idx = [r["idx"] for r in errors]
        err_df = df.loc[err_idx].copy()
        err_df["error_message"] = [r["error_message"] for r in errors]
        err_df.to_csv(args.error_csv_path, index=False)
        logger.warning("%d rows failed – see %s", len(err_df), args.error_csv_path)

        # Optional project‑specific volume report
        try:
            report_volumes(err_df)
        except Exception:
            logger.debug("report_volumes() failed – continuing")

    logger.info("All done ✔")


if __name__ == "__main__":
    setup_logging()
    args = build_parser().parse_args()
    args = normalize_segment_args(args)
    if getattr(args, "dry_run", False):
        logger.info("Dry run: segment")
        logger.info("%s", args)
        raise SystemExit(0)
    main(args)
