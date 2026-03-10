from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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


def add_register_intra_patient_arguments(
    parser: argparse.ArgumentParser,
    include_dry_run: bool = True,
) -> None:
    parser.add_argument(
        "csv_path_pos",
        nargs="?",
        type=str,
        default=None,
        help="Path to input CSV with `patient_key`, `nifti_path`, and mask columns. Defaults to ./nifti_index.csv.",
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
            "Path to save the intra-patient registration CSV. "
            "Defaults to <csv_dir>/<csv_stem>_registered_intra_patient.csv."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory used to store copied anchors and registered outputs.",
    )
    parser.add_argument(
        "--error_csv_path",
        type=str,
        default=None,
        help="Path to save failed rows (default: <csv_dir>/register_intra_patient_errors.csv).",
    )
    parser.add_argument(
        "--log_csv_path",
        type=str,
        default=None,
        help="Path to save row-level registration logs (default: <csv_dir>/register_intra_patient_log.csv).",
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
        "--num_workers",
        type=int,
        default=reg_common.DEFAULT_NUM_WORKERS,
        help="Number of concurrent patient workers used for registration.",
    )
    parser.add_argument(
        "--pad_mm",
        type=float,
        default=reg_common.DEFAULT_PAD_MM,
        help="Reserved registration padding in mm for future cropping refinements.",
    )
    parser.add_argument(
        "--band_mm",
        type=float,
        default=reg_common.DEFAULT_BAND_MM,
        help="Signed-distance-map clamp range for B-spline registration.",
    )
    parser.add_argument(
        "--bspline_ctrl_spacing_mm",
        type=float,
        default=reg_common.DEFAULT_BSPLINE_CTRL_SPACING_MM,
        help="Control point spacing for B-spline registration.",
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
        description="Perform rigid + elastic registration within each patient.",
        add_help=add_help,
    )
    add_register_intra_patient_arguments(parser)
    return parser


def normalize_register_intra_patient_args(args: argparse.Namespace) -> argparse.Namespace:
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
    args.num_workers = max(1, int(args.num_workers))
    args.pad_mm = float(args.pad_mm)
    args.band_mm = float(args.band_mm)
    args.bspline_ctrl_spacing_mm = float(args.bspline_ctrl_spacing_mm)

    csv_path_out_pos = getattr(args, "csv_path_out_pos", None)
    csv_out = args.csv_path_out if args.csv_path_out else csv_path_out_pos
    if csv_out:
        args.csv_path_out = str(Path(csv_out))
    else:
        args.csv_path_out = str(
            csv_path.parent / f"{csv_path.stem}_registered_intra_patient.csv"
        )

    if args.error_csv_path:
        args.error_csv_path = str(Path(args.error_csv_path))
    else:
        args.error_csv_path = str(csv_path.parent / "register_intra_patient_errors.csv")

    if args.log_csv_path:
        args.log_csv_path = str(Path(args.log_csv_path))
    else:
        args.log_csv_path = str(csv_path.parent / "register_intra_patient_log.csv")

    del args.csv_path_pos
    del args.csv_path_opt
    if hasattr(args, "csv_path_out_pos"):
        del args.csv_path_out_pos
    return args


def parse_arguments() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    args = normalize_register_intra_patient_args(args)
    logger.info("Running %s with args: %s", Path(__file__).name, args)
    return args


def _build_intra_log_columns() -> list[str]:
    return [
        "_source_idx",
        "patient_key",
        "intra_register_anchor_source_idx",
        "intra_register_anchor_phase",
        "intra_register_stage",
        "intra_register_dice_before",
        "intra_register_dice_after_rigid",
        "intra_register_dice_after_elastic",
        "intra_register_status",
        "intra_register_error_message",
    ]


def _select_anchor_row(patient_df: pd.DataFrame, *, mask_column: str) -> dict[str, Any] | None:
    if patient_df.empty:
        return None

    working = patient_df.copy()
    working["_phase_norm"] = working.apply(
        lambda row: reg_common.infer_phase_from_row(row.to_dict()),
        axis=1,
    )
    working["_has_required_inputs"] = working.apply(
        lambda row: reg_common._is_existing_path(row.get("nifti_path"))
        and reg_common._is_existing_path(row.get(mask_column)),
        axis=1,
    )
    candidates = working[working["_has_required_inputs"]].copy()
    if candidates.empty:
        return None

    candidates["_portal_rank"] = candidates["_phase_norm"].apply(
        lambda value: 0 if value == "portal" else 1
    )
    if "visit_order" in candidates.columns:
        candidates["_sort_visit_order"] = reg_common.to_numeric_sort_series(
            candidates["visit_order"]
        )
    else:
        candidates["_sort_visit_order"] = float("inf")

    if "date" in candidates.columns:
        dates = pd.to_datetime(candidates["date"], errors="coerce")
        candidates["_sort_date_missing"] = dates.isna().astype(int)
        candidates["_sort_date"] = dates
    else:
        candidates["_sort_date_missing"] = 1
        candidates["_sort_date"] = pd.NaT

    if "followup_months" in candidates.columns:
        candidates["_sort_followup_months"] = reg_common.to_numeric_sort_series(
            candidates["followup_months"]
        )
    else:
        candidates["_sort_followup_months"] = float("inf")

    candidates["_sort_source_idx"] = reg_common.to_numeric_sort_series(
        candidates["_source_idx"],
        missing=float("inf"),
    )
    ordered = candidates.sort_values(
        by=[
            "_portal_rank",
            "_sort_visit_order",
            "_sort_date_missing",
            "_sort_date",
            "_sort_followup_months",
            "_sort_source_idx",
        ],
        kind="stable",
    )
    return ordered.iloc[0].to_dict()


def _build_anchor_success_updates(
    *,
    anchor_source_idx: int,
    anchor_phase: str | None,
    copied_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        **copied_paths,
        "intra_register_anchor_source_idx": anchor_source_idx,
        "intra_register_anchor_phase": anchor_phase,
        "intra_register_stage": "anchor",
        "intra_register_dice_before": 1.0,
        "intra_register_dice_after_rigid": 1.0,
        "intra_register_dice_after_elastic": 1.0,
        "intra_register_status": "ok",
        "intra_register_error_message": None,
    }


def _build_error_result(
    *,
    source_idx: int,
    anchor_source_idx: int | None,
    anchor_phase: str | None,
    stage: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "source_idx": int(source_idx),
        "updates": {
            "intra_register_anchor_source_idx": anchor_source_idx,
            "intra_register_anchor_phase": anchor_phase,
            "intra_register_stage": stage,
            "intra_register_status": "error",
            "intra_register_error_message": error_message,
        },
        "error_message": error_message,
    }


def _process_patient_group(
    patient_df: pd.DataFrame,
    *,
    pending_source_indices: set[int],
    args: argparse.Namespace,
    path_columns: list[str],
) -> list[dict[str, Any]]:
    sitk_module = reg_common._load_register_dependencies()
    anchor = _select_anchor_row(patient_df, mask_column=args.mask_column)
    if anchor is None:
        return [
            _build_error_result(
                source_idx=int(row["_source_idx"]),
                anchor_source_idx=None,
                anchor_phase=None,
                stage="anchor",
                error_message=(
                    f"no valid anchor row found for patient using {args.mask_column}"
                ),
            )
            for _, row in patient_df.iterrows()
            if int(row["_source_idx"]) in pending_source_indices
        ]

    anchor_source_idx = int(anchor["_source_idx"])
    anchor_phase = reg_common.infer_phase_from_row(anchor)
    anchor_working = dict(anchor)
    results: list[dict[str, Any]] = []

    if anchor_source_idx in pending_source_indices:
        try:
            copied_paths = reg_common.copy_row_files(
                anchor_working,
                row_dir=reg_common.build_row_output_dir(args.output_dir, anchor_source_idx),
                path_columns=path_columns,
            )
            if not copied_paths.get("nifti_path") or not copied_paths.get(args.mask_column):
                raise RuntimeError("failed to copy anchor nifti or organ mask")
            anchor_working.update(copied_paths)
            results.append(
                {
                    "source_idx": anchor_source_idx,
                    "updates": _build_anchor_success_updates(
                        anchor_source_idx=anchor_source_idx,
                        anchor_phase=anchor_phase,
                        copied_paths=copied_paths,
                    ),
                    "error_message": None,
                }
            )
        except Exception as exc:
            error_message = f"anchor copy failed: {exc}"
            return [
                _build_error_result(
                    source_idx=int(row["_source_idx"]),
                    anchor_source_idx=anchor_source_idx,
                    anchor_phase=anchor_phase,
                    stage="anchor",
                    error_message=error_message,
                )
                for _, row in patient_df.iterrows()
                if int(row["_source_idx"]) in pending_source_indices
            ]

    try:
        fixed_image = reg_common.read_image(str(anchor_working["nifti_path"]), sitk_module)
        fixed_mask = reg_common.read_binary_mask(
            str(anchor_working[args.mask_column]),
            reference_image=fixed_image,
            sitk_module=sitk_module,
        )
    except Exception as exc:
        error_message = f"anchor inputs are not readable: {exc}"
        return [
            _build_error_result(
                source_idx=int(row["_source_idx"]),
                anchor_source_idx=anchor_source_idx,
                anchor_phase=anchor_phase,
                stage="anchor",
                error_message=error_message,
            )
            for _, row in patient_df.iterrows()
            if int(row["_source_idx"]) in pending_source_indices
            and int(row["_source_idx"]) != anchor_source_idx
        ]

    for _, row_series in patient_df.iterrows():
        row = row_series.to_dict()
        source_idx = int(row["_source_idx"])
        if source_idx not in pending_source_indices or source_idx == anchor_source_idx:
            continue
        nifti_path = row.get("nifti_path")
        mask_path = row.get(args.mask_column)
        if not reg_common._is_existing_path(nifti_path):
            results.append(
                _build_error_result(
                    source_idx=source_idx,
                    anchor_source_idx=anchor_source_idx,
                    anchor_phase=anchor_phase,
                    stage="rigid",
                    error_message=f"invalid nifti_path: {nifti_path}",
                )
            )
            continue
        if not reg_common._is_existing_path(mask_path):
            results.append(
                _build_error_result(
                    source_idx=source_idx,
                    anchor_source_idx=anchor_source_idx,
                    anchor_phase=anchor_phase,
                    stage="rigid",
                    error_message=f"invalid {args.mask_column}: {mask_path}",
                )
            )
            continue

        try:
            moving_mask = reg_common.read_binary_mask(
                str(mask_path),
                sitk_module=sitk_module,
            )
            rigid_tx = reg_common.rigid_register_mask_pair(
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
                sitk_module=sitk_module,
            )
            dice_before = reg_common.dice_coeff(
                fixed_mask,
                moving_mask,
                sitk_module=sitk_module,
            )
            dice_after_rigid = reg_common.dice_coeff(
                fixed_mask,
                moving_mask,
                sitk_module=sitk_module,
                tx=rigid_tx,
            )

            final_tx = rigid_tx
            final_stage = "rigid"
            dice_after_elastic: float | None = None
            try:
                elastic_tx = reg_common.bspline_register_mask_pair(
                    fixed_mask=fixed_mask,
                    moving_mask=moving_mask,
                    initial_transform=rigid_tx,
                    sitk_module=sitk_module,
                    band_mm=args.band_mm,
                    ctrl_spacing_mm=args.bspline_ctrl_spacing_mm,
                )
                dice_after_elastic = reg_common.dice_coeff(
                    fixed_mask,
                    moving_mask,
                    sitk_module=sitk_module,
                    tx=elastic_tx,
                )
                if dice_after_elastic >= dice_after_rigid:
                    final_tx = elastic_tx
                    final_stage = "bspline"
            except Exception as exc:
                logger.warning(
                    "Elastic registration fallback to rigid for row %s: %s",
                    source_idx,
                    exc,
                )

            rewritten_paths = reg_common.warp_row_files(
                row,
                row_dir=reg_common.build_row_output_dir(args.output_dir, source_idx),
                path_columns=path_columns,
                reference_image=fixed_image,
                transform=final_tx,
                sitk_module=sitk_module,
            )
            results.append(
                {
                    "source_idx": source_idx,
                    "updates": {
                        **rewritten_paths,
                        "intra_register_anchor_source_idx": anchor_source_idx,
                        "intra_register_anchor_phase": anchor_phase,
                        "intra_register_stage": final_stage,
                        "intra_register_dice_before": dice_before,
                        "intra_register_dice_after_rigid": dice_after_rigid,
                        "intra_register_dice_after_elastic": dice_after_elastic,
                        "intra_register_status": "ok",
                        "intra_register_error_message": None,
                    },
                    "error_message": None,
                }
            )
        except Exception as exc:
            results.append(
                _build_error_result(
                    source_idx=source_idx,
                    anchor_source_idx=anchor_source_idx,
                    anchor_phase=anchor_phase,
                    stage="rigid",
                    error_message=str(exc),
                )
            )

    return results


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
    columns = [col for col in _build_intra_log_columns() if col in df.columns]
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
        command="register_intra_patient",
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
            "Resume enabled and matching register-intra-patient run already finished; skipping execution."
        )
        return

    reg_common._load_register_dependencies()
    if can_resume and paths.main_checkpoint_path.exists():
        logger.info(
            "Resuming register-intra-patient from checkpoint: %s",
            paths.main_checkpoint_path,
        )
        df = pd.read_csv(paths.main_checkpoint_path).copy()
    else:
        df = pd.read_csv(args.csv_path).copy()
        df["_source_idx"] = df.index.astype(int)
    if "_source_idx" not in df.columns:
        df["_source_idx"] = df.index.astype(int)
    if "patient_key" not in df.columns:
        raise KeyError("column 'patient_key' missing")
    if "nifti_path" not in df.columns:
        raise KeyError("column 'nifti_path' missing")
    if args.mask_column not in df.columns:
        raise KeyError(f"column '{args.mask_column}' missing")

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
        )

    pending_indices = [
        idx
        for idx in df.index.tolist()
        if int(df.at[idx, "_source_idx"]) not in completed_indices
    ]
    missing_patient_indices = [
        idx
        for idx in pending_indices
        if pd.isna(df.at[idx, "patient_key"])
        or not str(df.at[idx, "patient_key"]).strip()
    ]
    for idx in missing_patient_indices:
        result = _build_error_result(
            source_idx=int(df.at[idx, "_source_idx"]),
            anchor_source_idx=None,
            anchor_phase=None,
            stage="anchor",
            error_message="missing patient_key value",
        )
        _apply_row_result(df, idx, result=result, errors_by_idx=errors_by_idx)
        completed_indices.add(int(df.at[idx, "_source_idx"]))
        ckpt.mark_processed()
        _checkpoint_write(force=False)

    pending_source_indices = {
        int(df.at[idx, "_source_idx"])
        for idx in pending_indices
        if idx not in missing_patient_indices
    }
    group_payloads: list[tuple[pd.DataFrame, set[int]]] = []
    if pending_source_indices:
        for _, patient_df in df.groupby("patient_key", sort=False):
            group_pending = {
                int(source_idx)
                for source_idx in patient_df["_source_idx"].tolist()
                if int(source_idx) in pending_source_indices
            }
            if group_pending:
                group_payloads.append((patient_df.copy(), group_pending))
    source_idx_to_idx = {
        int(df.at[idx, "_source_idx"]): idx
        for idx in df.index.tolist()
    }

    if args.num_workers > 1 and len(group_payloads) > 1:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_map = {
                executor.submit(
                    _process_patient_group,
                    patient_df,
                    pending_source_indices=group_pending,
                    args=args,
                    path_columns=path_columns,
                ): group_pending
                for patient_df, group_pending in group_payloads
            }
            for future in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc="RegisterIntraPatient",
                unit="patient",
            ):
                results = future.result()
                for result in results:
                    src_idx = int(result["source_idx"])
                    idx = source_idx_to_idx[src_idx]
                    _apply_row_result(df, idx, result=result, errors_by_idx=errors_by_idx)
                    completed_indices.add(src_idx)
                    ckpt.mark_processed()
                _checkpoint_write(force=False)
    else:
        for patient_df, group_pending in tqdm(
            group_payloads,
            total=len(group_payloads),
            desc="RegisterIntraPatient",
            unit="patient",
        ):
            results = _process_patient_group(
                patient_df,
                pending_source_indices=group_pending,
                args=args,
                path_columns=path_columns,
            )
            for result in results:
                src_idx = int(result["source_idx"])
                idx = source_idx_to_idx[src_idx]
                _apply_row_result(df, idx, result=result, errors_by_idx=errors_by_idx)
                completed_indices.add(src_idx)
                ckpt.mark_processed()
            _checkpoint_write(force=False)

    _checkpoint_write(force=True)
    df_out = df.drop(columns=["_source_idx"], errors="ignore")
    df_out = merge_with_existing_output(
        df_out,
        args.csv_path_out,
        preferred_keys=["patient_key", "nifti_path"],
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

    ckpt.finalize_state(completed_indices=completed_indices)
    logger.info("Intra-patient registration done")


if __name__ == "__main__":
    args = parse_arguments()
    if getattr(args, "dry_run", False):
        logger.info("Dry run: register-intra-patient")
        print_args(args)
        raise SystemExit(0)
    main(args)
