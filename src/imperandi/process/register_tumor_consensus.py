from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from imperandi.process import _registration_common as reg_common
from imperandi.process.registration import (
    ConsensusConfig,
    LongitudinalAuditConfig,
    build_longitudinal_audit,
    build_visit_consensus,
)
from imperandi.utils.misc import print_args
from imperandi.utils.run_state import atomic_write_csv, merge_with_existing_output

logger = logging.getLogger(__name__)


def add_register_tumor_consensus_arguments(
    parser: argparse.ArgumentParser,
    include_dry_run: bool = True,
) -> None:
    parser.add_argument(
        "csv_path_pos",
        nargs="?",
        type=str,
        default=None,
        help="Path to input CSV with patient/visit/phase image and tumor masks.",
    )
    parser.add_argument(
        "csv_path_out_pos",
        nargs="?",
        type=str,
        default=None,
        help="Optional output CSV path for per-visit consensus rows.",
    )
    parser.add_argument("--csv_path", dest="csv_path_opt", type=str)
    parser.add_argument(
        "--csv_path_out",
        type=str,
        default=None,
        help=(
            "Per-visit consensus output CSV. "
            "Defaults to <csv_dir>/<csv_stem>_tumor_consensus.csv."
        ),
    )
    parser.add_argument(
        "--components_csv_path",
        type=str,
        default=None,
        help=(
            "Tumor-component summary CSV path. "
            "Defaults to <csv_dir>/tumor_consensus_components.csv."
        ),
    )
    parser.add_argument(
        "--audit_csv_path",
        type=str,
        default=None,
        help=(
            "Longitudinal consistency audit CSV path. "
            "Defaults to <csv_dir>/tumor_consistency_audit.csv."
        ),
    )
    parser.add_argument(
        "--error_csv_path",
        type=str,
        default=None,
        help=(
            "Rows/groups that failed consensus generation. "
            "Defaults to <csv_dir>/register_tumor_consensus_errors.csv."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where consensus masks and metadata are saved.",
    )
    parser.add_argument(
        "--organ",
        type=str,
        default="liver",
        help="Organ name used to resolve default mask columns.",
    )
    parser.add_argument(
        "--patient_column",
        type=str,
        default="patient_key",
        help="Patient grouping column.",
    )
    parser.add_argument(
        "--visit_column",
        type=str,
        default="visit_order",
        help="Visit/timepoint grouping column.",
    )
    parser.add_argument(
        "--phase_column",
        type=str,
        default="phase",
        help="Phase column used to select a reference phase.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="nifti_path",
        help="Image path column.",
    )
    parser.add_argument(
        "--organ_mask_column",
        type=str,
        default=None,
        help="Optional organ mask column (default: mask_<organ>).",
    )
    parser.add_argument(
        "--tumor_mask_column",
        type=str,
        default=None,
        help="Tumor mask column (default: mask_<organ>_tumor).",
    )
    parser.add_argument(
        "--consensus_rule",
        type=str,
        default="majority",
        choices=["union", "intersection", "majority"],
        help="Consensus rule across aligned phases.",
    )
    parser.add_argument(
        "--majority_threshold",
        type=float,
        default=0.5,
        help="Threshold used by the majority rule.",
    )
    parser.add_argument(
        "--min_component_voxels",
        type=int,
        default=10,
        help="Drop consensus connected components smaller than this size.",
    )
    parser.add_argument(
        "--disable_elastic",
        action="store_true",
        default=False,
        help="Use rigid-only alignment when building visit consensus.",
    )
    parser.add_argument(
        "--band_mm",
        type=float,
        default=reg_common.DEFAULT_BAND_MM,
        help="Elastic distance-map clamp range in mm.",
    )
    parser.add_argument(
        "--bspline_ctrl_spacing_mm",
        type=float,
        default=reg_common.DEFAULT_BSPLINE_CTRL_SPACING_MM,
        help="Elastic B-spline control spacing in mm.",
    )
    parser.add_argument(
        "--max_centroid_shift_mm",
        type=float,
        default=25.0,
        help="Longitudinal audit threshold for suspicious centroid shifts.",
    )
    parser.add_argument(
        "--max_total_volume_change_ratio",
        type=float,
        default=0.6,
        help="Longitudinal audit threshold for unstable total volume change.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
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
        description=(
            "Build per-visit tumor consensus across phases and run longitudinal "
            "tumor consistency audit."
        ),
        add_help=add_help,
    )
    add_register_tumor_consensus_arguments(parser)
    return parser


def normalize_register_tumor_consensus_args(args: argparse.Namespace) -> argparse.Namespace:
    csv_in = args.csv_path_opt if args.csv_path_opt is not None else args.csv_path_pos
    csv_path = Path(csv_in) if csv_in else (Path.cwd() / "nifti_index.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Not a CSV file: {csv_path}")
    csv_path = csv_path.resolve()
    args.csv_path = str(csv_path)

    csv_path_out_pos = getattr(args, "csv_path_out_pos", None)
    csv_out = args.csv_path_out if args.csv_path_out else csv_path_out_pos
    if csv_out:
        args.csv_path_out = str(Path(csv_out))
    else:
        args.csv_path_out = str(csv_path.parent / f"{csv_path.stem}_tumor_consensus.csv")

    if args.components_csv_path:
        args.components_csv_path = str(Path(args.components_csv_path))
    else:
        args.components_csv_path = str(csv_path.parent / "tumor_consensus_components.csv")

    if args.audit_csv_path:
        args.audit_csv_path = str(Path(args.audit_csv_path))
    else:
        args.audit_csv_path = str(csv_path.parent / "tumor_consistency_audit.csv")

    if args.error_csv_path:
        args.error_csv_path = str(Path(args.error_csv_path))
    else:
        args.error_csv_path = str(csv_path.parent / "register_tumor_consensus_errors.csv")

    args.output_dir = str(Path(args.output_dir))
    args.organ = str(args.organ).strip().lower()
    args.patient_column = str(args.patient_column).strip()
    args.visit_column = str(args.visit_column).strip()
    args.phase_column = str(args.phase_column).strip()
    args.image_column = str(args.image_column).strip()
    args.organ_mask_column = reg_common.resolve_mask_column(
        organ=args.organ,
        mask_column=args.organ_mask_column,
    )
    if args.tumor_mask_column is not None and str(args.tumor_mask_column).strip():
        args.tumor_mask_column = str(args.tumor_mask_column).strip()
    else:
        args.tumor_mask_column = f"{args.organ_mask_column}_tumor"
    args.majority_threshold = float(args.majority_threshold)
    if args.majority_threshold <= 0.0 or args.majority_threshold > 1.0:
        raise ValueError("--majority_threshold must be in (0, 1]")
    args.min_component_voxels = max(1, int(args.min_component_voxels))
    args.band_mm = float(args.band_mm)
    args.bspline_ctrl_spacing_mm = float(args.bspline_ctrl_spacing_mm)
    args.max_centroid_shift_mm = float(args.max_centroid_shift_mm)
    args.max_total_volume_change_ratio = float(args.max_total_volume_change_ratio)

    del args.csv_path_pos
    del args.csv_path_opt
    if hasattr(args, "csv_path_out_pos"):
        del args.csv_path_out_pos
    return args


def parse_arguments() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    args = normalize_register_tumor_consensus_args(args)
    logger.info("Running %s with args: %s", Path(__file__).name, args)
    return args


def _safe_label(value: Any, *, fallback: str) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        text = fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or fallback


def _visit_sort_value(value: Any) -> tuple[int, Any]:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return (0, float(numeric))
    date_value = pd.to_datetime(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(date_value):
        return (1, date_value.to_pydatetime())
    return (2, str(value))


def _build_consensus_config(args: argparse.Namespace) -> ConsensusConfig:
    return ConsensusConfig(
        rule=str(args.consensus_rule),
        majority_threshold=float(args.majority_threshold),
        min_component_voxels=int(args.min_component_voxels),
        use_elastic_registration=not bool(args.disable_elastic),
        band_mm=float(args.band_mm),
        bspline_ctrl_spacing_mm=float(args.bspline_ctrl_spacing_mm),
    )


def _build_audit_config(args: argparse.Namespace) -> LongitudinalAuditConfig:
    return LongitudinalAuditConfig(
        max_centroid_shift_mm=float(args.max_centroid_shift_mm),
        max_total_volume_change_ratio=float(args.max_total_volume_change_ratio),
    )


def main(args: argparse.Namespace) -> None:
    sitk_module = reg_common._load_register_dependencies()
    df = pd.read_csv(args.csv_path).copy()
    df["_source_idx"] = df.index.astype(int)

    required_columns = [
        args.patient_column,
        args.image_column,
        args.tumor_mask_column,
    ]
    for column in required_columns:
        if column not in df.columns:
            raise KeyError(f"column '{column}' missing")
    if args.visit_column not in df.columns:
        logger.warning(
            "visit column '%s' missing; all rows per patient will be grouped as a single visit.",
            args.visit_column,
        )
        df[args.visit_column] = "visit_0"
    if args.phase_column not in df.columns:
        logger.warning(
            "phase column '%s' missing; reference phase fallback will use source order.",
            args.phase_column,
        )
        df[args.phase_column] = None
    if args.organ_mask_column not in df.columns:
        logger.warning(
            "organ mask column '%s' missing; consensus registration will fallback to identity.",
            args.organ_mask_column,
        )
        args.organ_mask_column = None

    consensus_config = _build_consensus_config(args)
    audit_config = _build_audit_config(args)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    consensus_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    components_by_patient_visit: dict[tuple[str, str], list[Any]] = {}

    grouped = list(
        df.groupby([args.patient_column, args.visit_column], sort=False, dropna=False)
    )
    for (patient_value, visit_value), group_df in tqdm(
        grouped,
        total=len(grouped),
        desc="TumorConsensus",
        unit="visit",
    ):
        patient_key = str(patient_value)
        visit_key = str(visit_value)
        rows = [row for _, row in group_df.iterrows()]
        rows_dict = [row.to_dict() for row in rows]
        try:
            result = build_visit_consensus(
                rows_dict,
                patient_key=patient_key,
                visit_key=visit_key,
                tumor_mask_column=args.tumor_mask_column,
                organ_mask_column=args.organ_mask_column,
                image_column=args.image_column,
                phase_column=args.phase_column,
                config=consensus_config,
                sitk_module=sitk_module,
            )

            patient_label = _safe_label(patient_key, fallback="patient")
            visit_label = _safe_label(visit_key, fallback="visit")
            visit_dir = output_root / "consensus" / patient_label / visit_label
            visit_dir.mkdir(parents=True, exist_ok=True)
            consensus_mask_path = visit_dir / f"{args.tumor_mask_column}_consensus.nii.gz"
            reg_common.write_image(
                result.consensus_mask,
                consensus_mask_path,
                sitk_module=sitk_module,
            )
            metadata_path = visit_dir / "consensus_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "patient_key": patient_key,
                        "visit_key": visit_key,
                        "reference_source_idx": int(result.reference_source_idx),
                        "aligned_mask_count": int(result.aligned_mask_count),
                        "config": {
                            "rule": consensus_config.rule,
                            "majority_threshold": consensus_config.majority_threshold,
                            "min_component_voxels": consensus_config.min_component_voxels,
                            "use_elastic_registration": consensus_config.use_elastic_registration,
                            "band_mm": consensus_config.band_mm,
                            "bspline_ctrl_spacing_mm": consensus_config.bspline_ctrl_spacing_mm,
                        },
                        "component_count": len(result.components),
                        "component_rows": [
                            {
                                "component_id": int(c.component_id),
                                "volume_vox": int(c.volume_vox),
                                "volume_ml": float(c.volume_ml),
                                "centroid_x_mm": float(c.centroid_x_mm),
                                "centroid_y_mm": float(c.centroid_y_mm),
                                "centroid_z_mm": float(c.centroid_z_mm),
                                "bbox_x_min": int(c.bbox_x_min),
                                "bbox_y_min": int(c.bbox_y_min),
                                "bbox_z_min": int(c.bbox_z_min),
                                "bbox_x_size": int(c.bbox_x_size),
                                "bbox_y_size": int(c.bbox_y_size),
                                "bbox_z_size": int(c.bbox_z_size),
                            }
                            for c in result.components
                        ],
                        "transform_metadata_by_source_idx": result.transform_metadata_by_source_idx,
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
            consensus_rows.append(
                {
                    args.patient_column: patient_key,
                    args.visit_column: visit_key,
                    "consensus_reference_source_idx": int(result.reference_source_idx),
                    "consensus_aligned_mask_count": int(result.aligned_mask_count),
                    "consensus_component_count": int(len(result.components)),
                    "consensus_mask_path": str(consensus_mask_path),
                    "consensus_metadata_path": str(metadata_path),
                    "consensus_rule": consensus_config.rule,
                    "consensus_status": "ok",
                    "consensus_error_message": None,
                }
            )
            components_by_patient_visit[(patient_key, visit_key)] = list(result.components)
            for component in result.components:
                component_rows.append(
                    {
                        args.patient_column: patient_key,
                        args.visit_column: visit_key,
                        "component_id": int(component.component_id),
                        "volume_vox": int(component.volume_vox),
                        "volume_ml": float(component.volume_ml),
                        "centroid_x_mm": float(component.centroid_x_mm),
                        "centroid_y_mm": float(component.centroid_y_mm),
                        "centroid_z_mm": float(component.centroid_z_mm),
                        "bbox_x_min": int(component.bbox_x_min),
                        "bbox_y_min": int(component.bbox_y_min),
                        "bbox_z_min": int(component.bbox_z_min),
                        "bbox_x_size": int(component.bbox_x_size),
                        "bbox_y_size": int(component.bbox_y_size),
                        "bbox_z_size": int(component.bbox_z_size),
                    }
                )
        except Exception as exc:
            error_message = str(exc)
            consensus_rows.append(
                {
                    args.patient_column: patient_key,
                    args.visit_column: visit_key,
                    "consensus_reference_source_idx": None,
                    "consensus_aligned_mask_count": 0,
                    "consensus_component_count": 0,
                    "consensus_mask_path": None,
                    "consensus_metadata_path": None,
                    "consensus_rule": consensus_config.rule,
                    "consensus_status": "error",
                    "consensus_error_message": error_message,
                }
            )
            error_rows.append(
                {
                    args.patient_column: patient_key,
                    args.visit_column: visit_key,
                    "error_message": error_message,
                }
            )

    audit_rows: list[dict[str, Any]] = []
    if consensus_rows:
        grouped_visits: dict[str, list[str]] = {}
        for row in consensus_rows:
            if row.get("consensus_status") != "ok":
                continue
            patient_key = str(row[args.patient_column])
            grouped_visits.setdefault(patient_key, []).append(str(row[args.visit_column]))
        for patient_key, visits in grouped_visits.items():
            unique_visits = sorted(set(visits), key=_visit_sort_value)
            components_map = {
                visit: components_by_patient_visit.get((patient_key, visit), [])
                for visit in unique_visits
            }
            findings = build_longitudinal_audit(
                patient_key=patient_key,
                sorted_visits=unique_visits,
                components_by_visit=components_map,
                config=audit_config,
            )
            for finding in findings:
                audit_rows.append(
                    {
                        args.patient_column: finding.patient_key,
                        "visit_prev": finding.visit_prev,
                        "visit_curr": finding.visit_curr,
                        "audit_flag": finding.flag,
                        "audit_severity": finding.severity,
                        "audit_value": finding.value,
                        "audit_detail": finding.detail,
                    }
                )

    consensus_df = pd.DataFrame(consensus_rows)
    if not consensus_df.empty:
        consensus_df = merge_with_existing_output(
            consensus_df,
            args.csv_path_out,
            preferred_keys=[args.patient_column, args.visit_column],
            strict=True,
        )
    atomic_write_csv(consensus_df, args.csv_path_out, index=False)
    logger.info("Wrote per-visit consensus table -> %s", args.csv_path_out)

    components_df = pd.DataFrame(component_rows)
    atomic_write_csv(components_df, args.components_csv_path, index=False)
    logger.info("Wrote tumor component table -> %s", args.components_csv_path)

    audit_df = pd.DataFrame(audit_rows)
    atomic_write_csv(audit_df, args.audit_csv_path, index=False)
    logger.info("Wrote longitudinal audit table -> %s", args.audit_csv_path)

    if error_rows:
        errors_df = pd.DataFrame(error_rows)
        atomic_write_csv(errors_df, args.error_csv_path, index=False)
        logger.warning("%d groups failed -> %s", len(errors_df), args.error_csv_path)
    logger.info("Tumor consensus + longitudinal audit done")


if __name__ == "__main__":
    args = parse_arguments()
    if getattr(args, "dry_run", False):
        logger.info("Dry run: register-tumor-consensus")
        print_args(args)
        raise SystemExit(0)
    main(args)
