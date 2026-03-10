from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from imperandi.process import _registration_common as reg_common
from imperandi.utils.checkpoint_cli import add_checkpoint_arguments
from imperandi.utils.misc import print_args
from imperandi.utils.run_state import (
    CheckpointManager,
    atomic_write_csv,
    merge_with_existing_output,
    prepare_resume_context,
)

logger = logging.getLogger(__name__)
DEFAULT_CHECKPOINT_EVERY_ROWS = 50
DEFAULT_CHECKPOINT_EVERY_SEC = 5 * 60


def add_register_population_arguments(
    parser: argparse.ArgumentParser,
    include_dry_run: bool = True,
) -> None:
    parser.add_argument(
        "csv_path_pos",
        nargs="?",
        type=str,
        default=None,
        help="Path to input CSV with `nifti_path` and organ mask columns. Defaults to ./nifti_index.csv.",
    )
    parser.add_argument(
        "csv_path_out_pos",
        nargs="?",
        type=str,
        default=None,
        help="Optional output CSV path (positional alternative to --csv_path_out).",
    )
    parser.add_argument("--csv_path", dest="csv_path_opt", type=str)
    parser.add_argument(
        "--csv_path_out",
        type=str,
        default=None,
        help=(
            "Path to save the population-registration CSV. "
            "Defaults to <csv_dir>/<csv_stem>_registered_population.csv."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory used for template artifacts and optional registered outputs.",
    )
    parser.add_argument(
        "--error_csv_path",
        type=str,
        default=None,
        help="Path to save failed rows (default: <csv_dir>/register_population_errors.csv).",
    )
    parser.add_argument(
        "--log_csv_path",
        type=str,
        default=None,
        help="Path to save row-level registration logs (default: <csv_dir>/register_population_log.csv).",
    )
    parser.add_argument(
        "--organ",
        type=str,
        default="liver",
        help="Organ name used to resolve the default mask column (default: liver).",
    )
    parser.add_argument(
        "--mask_column",
        type=str,
        default=None,
        help="Explicit organ mask column to use instead of the default mask_<organ>.",
    )
    parser.add_argument(
        "--template_sample_size",
        type=int,
        default=reg_common.DEFAULT_TEMPLATE_SAMPLE_SIZE,
        help="Maximum number of valid rows to use when building the median-shape template.",
    )
    parser.add_argument(
        "--template_seed",
        type=int,
        default=reg_common.DEFAULT_TEMPLATE_SEED,
        help="Random seed used when sampling template candidates.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=reg_common.DEFAULT_NUM_WORKERS,
        help="Number of concurrent row workers used for registration.",
    )
    parser.add_argument(
        "--pad_mm",
        type=float,
        default=reg_common.DEFAULT_PAD_MM,
        help="Reserved registration padding in mm for future template refinements.",
    )
    parser.add_argument(
        "--save_registered_outputs",
        action="store_true",
        default=False,
        help="Also resample and save registered images and masks and rewrite their paths in the output CSV.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    add_checkpoint_arguments(
        parser,
        default_rows=DEFAULT_CHECKPOINT_EVERY_ROWS,
        default_sec=DEFAULT_CHECKPOINT_EVERY_SEC,
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
        description="Build a population liver template and rigidly align cohort rows to it.",
        add_help=add_help,
    )
    add_register_population_arguments(parser)
    return parser


def normalize_register_population_args(args: argparse.Namespace) -> argparse.Namespace:
    csv_in = args.csv_path_opt if args.csv_path_opt is not None else args.csv_path_pos
    csv_path = Path(csv_in) if csv_in else (Path.cwd() / "nifti_index.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Not a CSV file: {csv_path}")

    csv_path = csv_path.resolve()
    args.csv_path = str(csv_path)
    args.output_dir = str(Path(args.output_dir))
    args.organ = str(args.organ).strip().lower()
    args.mask_column = reg_common.resolve_mask_column(
        organ=args.organ,
        mask_column=args.mask_column,
    )
    args.template_sample_size = max(1, int(args.template_sample_size))
    args.template_seed = int(args.template_seed)
    args.num_workers = max(1, int(args.num_workers))
    args.pad_mm = float(args.pad_mm)

    csv_path_out_pos = getattr(args, "csv_path_out_pos", None)
    csv_out = args.csv_path_out if args.csv_path_out else csv_path_out_pos
    if csv_out:
        args.csv_path_out = str(Path(csv_out))
    else:
        args.csv_path_out = str(
            csv_path.parent / f"{csv_path.stem}_registered_population.csv"
        )

    if args.error_csv_path:
        args.error_csv_path = str(Path(args.error_csv_path))
    else:
        args.error_csv_path = str(csv_path.parent / "register_population_errors.csv")

    if args.log_csv_path:
        args.log_csv_path = str(Path(args.log_csv_path))
    else:
        args.log_csv_path = str(csv_path.parent / "register_population_log.csv")

    del args.csv_path_pos
    del args.csv_path_opt
    if hasattr(args, "csv_path_out_pos"):
        del args.csv_path_out_pos
    return args


def parse_arguments() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    args = normalize_register_population_args(args)
    logger.info("Running %s with args: %s", Path(__file__).name, args)
    return args


def _template_paths(args: argparse.Namespace) -> dict[str, Path]:
    template_dir = Path(args.output_dir) / "template"
    return {
        "template_dir": template_dir,
        "reference_image_path": template_dir / "template_reference.nii.gz",
        "mask_path": template_dir / f"{args.mask_column}.nii.gz",
    }


def _build_population_template(
    df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    sitk_module,
) -> dict[str, Any]:
    sampled_rows = reg_common.sample_valid_rows_for_template(
        df,
        mask_column=args.mask_column,
        sample_size=args.template_sample_size,
        seed=args.template_seed,
        sitk_module=sitk_module,
    )
    if not sampled_rows:
        raise RuntimeError(
            f"No valid rows found for template creation using column '{args.mask_column}'."
        )

    exemplar = reg_common.choose_median_exemplar(sampled_rows)
    exemplar_image = reg_common.read_image(exemplar["nifti_path"], sitk_module)
    exemplar_mask = reg_common.read_binary_mask(
        exemplar["mask_path"],
        reference_image=exemplar_image,
        sitk_module=sitk_module,
    )

    aligned_arrays = [
        sitk_module.GetArrayFromImage(
            sitk_module.Cast(exemplar_mask > 0, sitk_module.sitkFloat32)
        )
    ]
    for sampled in sampled_rows:
        if int(sampled["source_idx"]) == int(exemplar["source_idx"]):
            continue
        try:
            moving_mask = reg_common.read_binary_mask(
                sampled["mask_path"],
                sitk_module=sitk_module,
            )
            rigid_tx = reg_common.rigid_register_mask_pair(
                fixed_mask=exemplar_mask,
                moving_mask=moving_mask,
                sitk_module=sitk_module,
            )
            aligned_mask = reg_common.resample_like(
                exemplar_image,
                moving_mask,
                tx=rigid_tx,
                interp=sitk_module.sitkNearestNeighbor,
                default=0,
                pixel_id=sitk_module.sitkUInt8,
                sitk_module=sitk_module,
            )
            aligned_arrays.append(
                sitk_module.GetArrayFromImage(
                    sitk_module.Cast(aligned_mask > 0, sitk_module.sitkFloat32)
                )
            )
        except Exception as exc:
            logger.warning(
                "Skipping sampled template row %s after registration error: %s",
                sampled["source_idx"],
                exc,
            )

    mean_array = np.mean(np.stack(aligned_arrays, axis=0), axis=0).astype(np.float32)
    mean_image = reg_common.image_from_array_like(
        mean_array,
        reference_image=exemplar_image,
        sitk_module=sitk_module,
        cast_to=sitk_module.sitkFloat32,
    )
    template_mask = reg_common.threshold_to_largest_component(
        mean_image,
        reference_image=exemplar_image,
        threshold=0.5,
        sitk_module=sitk_module,
    )

    paths = _template_paths(args)
    reg_common.write_image(
        exemplar_image,
        paths["reference_image_path"],
        sitk_module=sitk_module,
    )
    reg_common.write_image(
        template_mask,
        paths["mask_path"],
        sitk_module=sitk_module,
    )
    return {
        "template_source_idx": int(exemplar["source_idx"]),
        "reference_image_path": str(paths["reference_image_path"]),
        "mask_path": str(paths["mask_path"]),
        "sample_count": len(aligned_arrays),
    }


def _row_log_columns() -> list[str]:
    return [
        "_source_idx",
        "patient_key",
        "population_register_template_source_idx",
        "population_register_mask_column",
        "population_register_stage",
        "population_register_dice_before",
        "population_register_dice_after",
        "population_register_status",
        "population_register_error_message",
        *reg_common.POPULATION_MATRIX_COLUMNS,
    ]


def _register_population_row(
    row: dict[str, Any],
    *,
    template_info: dict[str, Any],
    args: argparse.Namespace,
    path_columns: list[str],
) -> dict[str, Any]:
    sitk_module = reg_common._load_register_dependencies()
    source_idx = int(row.get("_source_idx", -1))
    base_updates: dict[str, Any] = {
        "population_register_template_source_idx": int(
            template_info["template_source_idx"]
        ),
        "population_register_mask_column": args.mask_column,
        "population_register_stage": "rigid",
        "population_register_status": "error",
        "population_register_error_message": None,
    }

    nifti_path = row.get("nifti_path")
    mask_path = row.get(args.mask_column)
    if not reg_common._is_existing_path(nifti_path):
        return {
            "source_idx": source_idx,
            "updates": {
                **base_updates,
                "population_register_error_message": f"invalid nifti_path: {nifti_path}",
            },
            "error_message": f"invalid nifti_path: {nifti_path}",
        }
    if not reg_common._is_existing_path(mask_path):
        return {
            "source_idx": source_idx,
            "updates": {
                **base_updates,
                "population_register_error_message": (
                    f"invalid {args.mask_column}: {mask_path}"
                ),
            },
            "error_message": f"invalid {args.mask_column}: {mask_path}",
        }

    try:
        template_image = reg_common.read_image(
            template_info["reference_image_path"],
            sitk_module,
        )
        template_mask = reg_common.read_binary_mask(
            template_info["mask_path"],
            reference_image=template_image,
            sitk_module=sitk_module,
        )
        moving_image = reg_common.read_image(str(nifti_path), sitk_module)
        moving_mask = reg_common.read_binary_mask(
            str(mask_path),
            reference_image=moving_image,
            sitk_module=sitk_module,
        )
        dice_before = reg_common.dice_coeff(
            template_mask,
            moving_mask,
            sitk_module=sitk_module,
        )
        rigid_tx = reg_common.rigid_register_mask_pair(
            fixed_mask=template_mask,
            moving_mask=moving_mask,
            sitk_module=sitk_module,
        )
        dice_after = reg_common.dice_coeff(
            template_mask,
            moving_mask,
            sitk_module=sitk_module,
            tx=rigid_tx,
        )
        updates = {
            **base_updates,
            **reg_common.transform_to_flat_3x4(rigid_tx),
            "population_register_dice_before": dice_before,
            "population_register_dice_after": dice_after,
            "population_register_status": "ok",
            "population_register_error_message": None,
        }
        if args.save_registered_outputs:
            row_dir = reg_common.build_row_output_dir(args.output_dir, source_idx)
            rewritten_paths = reg_common.warp_row_files(
                row,
                row_dir=row_dir,
                path_columns=path_columns,
                reference_image=template_image,
                transform=rigid_tx,
                sitk_module=sitk_module,
            )
            updates.update(rewritten_paths)
        return {"source_idx": source_idx, "updates": updates, "error_message": None}
    except Exception as exc:
        return {
            "source_idx": source_idx,
            "updates": {
                **base_updates,
                "population_register_error_message": str(exc),
            },
            "error_message": str(exc),
        }


def _apply_row_result(
    df: pd.DataFrame,
    idx: int,
    *,
    result: dict[str, Any],
    errors_by_idx: dict[int, dict[str, Any]],
) -> None:
    source_idx = int(df.at[idx, "_source_idx"])
    for key, value in result["updates"].items():
        df.at[idx, key] = value
    if result["error_message"]:
        error_row = df.loc[idx].to_dict()
        error_row["error_message"] = result["error_message"]
        errors_by_idx[source_idx] = error_row
    elif source_idx in errors_by_idx:
        del errors_by_idx[source_idx]


def _build_log_df(df: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in _row_log_columns() if col in df.columns]
    return df[columns].copy()


def main(args: argparse.Namespace) -> None:
    output_path = Path(args.csv_path_out)
    error_path = Path(args.error_csv_path)
    exclude_hash_args = {
        "csv_path_out",
        "error_csv_path",
        "log_csv_path",
        "dry_run",
        "verbose",
        "resume",
        "checkpoint_every_rows",
        "checkpoint_every_sec",
        "strict_resume",
    }
    resume_ctx = prepare_resume_context(
        args=args,
        command="register_population",
        inputs=args.csv_path,
        output_path=output_path,
        error_path=error_path,
        exclude_hash_args=exclude_hash_args,
    )
    paths = resume_ctx["paths"]
    state = resume_ctx["state"]
    can_resume = resume_ctx["can_resume"]
    already_finished = resume_ctx["already_finished"]
    ckpt = CheckpointManager(paths=paths, config=resume_ctx["config"])

    if already_finished:
        logger.info(
            "Resume enabled and matching register-population run already finished; skipping execution."
        )
        return

    sitk_module = reg_common._load_register_dependencies()
    if can_resume and paths.main_checkpoint_path.exists():
        logger.info(
            "Resuming register-population from checkpoint: %s",
            paths.main_checkpoint_path,
        )
        df = pd.read_csv(paths.main_checkpoint_path).copy()
    else:
        df = pd.read_csv(args.csv_path).copy()
        df["_source_idx"] = df.index.astype(int)
    if "_source_idx" not in df.columns:
        df["_source_idx"] = df.index.astype(int)
    if "nifti_path" not in df.columns:
        raise KeyError("column 'nifti_path' missing")
    if args.mask_column not in df.columns:
        raise KeyError(f"column '{args.mask_column}' missing")

    source_df = pd.read_csv(args.csv_path).copy()
    source_df["_source_idx"] = source_df.index.astype(int)
    template_info = _build_population_template(
        source_df,
        args=args,
        sitk_module=sitk_module,
    )
    path_columns = ["nifti_path", *reg_common.get_mask_columns(df)]

    completed_indices: set[int] = set()
    if can_resume:
        completed_indices = {
            int(i)
            for i in (state or {}).get("completed_indices", [])
            if isinstance(i, int)
        }
    errors_by_idx: dict[int, dict[str, Any]] = {}
    if can_resume and paths.error_checkpoint_path.exists():
        err_ckpt = pd.read_csv(paths.error_checkpoint_path)
        for _, row in err_ckpt.iterrows():
            if "_source_idx" in row:
                try:
                    errors_by_idx[int(row["_source_idx"])] = row.to_dict()
                except Exception:
                    pass

    def _checkpoint_write(*, force: bool = False) -> None:
        err_df = (
            pd.DataFrame(list(errors_by_idx.values()))
            if errors_by_idx
            else pd.DataFrame()
        )
        ckpt.flush(
            main_df=df,
            error_df=err_df,
            completed_indices=completed_indices,
            force=force,
            extra_state={
                "template_source_idx": int(template_info["template_source_idx"]),
                "template_mask_path": template_info["mask_path"],
                "template_reference_image_path": template_info["reference_image_path"],
            },
        )

    pending_indices = [
        idx
        for idx in df.index.tolist()
        if int(df.at[idx, "_source_idx"]) not in completed_indices
    ]
    row_payloads = [
        (idx, df.loc[idx].to_dict())
        for idx in pending_indices
    ]

    if args.num_workers > 1 and len(row_payloads) > 1:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_map = {
                executor.submit(
                    _register_population_row,
                    row_dict,
                    template_info=template_info,
                    args=args,
                    path_columns=path_columns,
                ): idx
                for idx, row_dict in row_payloads
            }
            for future in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc="RegisterPopulation",
                unit="row",
            ):
                idx = future_map[future]
                result = future.result()
                _apply_row_result(df, idx, result=result, errors_by_idx=errors_by_idx)
                completed_indices.add(int(df.at[idx, "_source_idx"]))
                ckpt.mark_processed()
                _checkpoint_write(force=False)
    else:
        for idx, row_dict in tqdm(
            row_payloads,
            total=len(row_payloads),
            desc="RegisterPopulation",
            unit="row",
        ):
            result = _register_population_row(
                row_dict,
                template_info=template_info,
                args=args,
                path_columns=path_columns,
            )
            _apply_row_result(df, idx, result=result, errors_by_idx=errors_by_idx)
            completed_indices.add(int(df.at[idx, "_source_idx"]))
            ckpt.mark_processed()
            _checkpoint_write(force=False)

    _checkpoint_write(force=True)
    df_out = df.drop(columns=["_source_idx"], errors="ignore")
    df_out = merge_with_existing_output(
        df_out,
        args.csv_path_out,
        preferred_keys=["nifti_path", "patient_key"],
        strict=True,
    )
    atomic_write_csv(df_out, args.csv_path_out, index=False)
    logger.info("Wrote main table -> %s", args.csv_path_out)

    log_df = _build_log_df(df)
    atomic_write_csv(log_df, args.log_csv_path, index=False)
    logger.info("Wrote log table -> %s", args.log_csv_path)

    if errors_by_idx:
        df_err = pd.DataFrame(list(errors_by_idx.values())).drop(
            columns=["_source_idx"], errors="ignore"
        )
        atomic_write_csv(df_err, args.error_csv_path, index=False)
        logger.warning("%d rows failed -> %s", len(df_err), args.error_csv_path)

    ckpt.finalize_state(
        completed_indices=completed_indices,
        extra_state={
            "template_source_idx": int(template_info["template_source_idx"]),
            "template_mask_path": template_info["mask_path"],
            "template_reference_image_path": template_info["reference_image_path"],
        },
    )
    logger.info("Population registration done")


if __name__ == "__main__":
    args = parse_arguments()
    if getattr(args, "dry_run", False):
        logger.info("Dry run: register-population")
        print_args(args)
        raise SystemExit(0)
    main(args)
