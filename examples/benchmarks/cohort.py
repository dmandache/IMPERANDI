"""Build paper-inspired cohorts independently, then compare public references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PHASES = ("NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED")
EXAM = ["patient_key", "study_id", "date"]
MRI_EXAM = ["patient_key", "date"]


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def require(df, columns, *, allow_empty=()):
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    allow_empty = set(allow_empty)
    for col in columns:
        if col not in allow_empty and df[col].str.strip().eq("").any():
            raise ValueError(f"Empty {col}")


def normalize_dates(values):
    # DICOM YYYYMMDD and exported ISO dates are both accepted, never inferred.
    return values.map(
        lambda value: pd.to_datetime(
            value, format="%Y%m%d" if len(value) == 8 else "%Y-%m-%d"
        ).strftime("%Y-%m-%d")
    )


def mri_eligibility(data):
    """Use resolved canonical T1 MRI phases for both comparison methods."""
    require(data, ["Modality", "phase"], allow_empty={"phase"})
    if "mri_sequence" not in data:
        raise ValueError("LiverHccSeg needs mri_sequence")
    eligible = (
        data["Modality"].isin(["MR", "MRI"])
        & data["mri_sequence"].eq("T1")
        & data["phase"].isin(PHASES)
    )
    if "phase_status" in data:
        eligible &= data["phase_status"].eq("RESOLVED")
    return eligible


def build_cohort(df, target):
    """Use IMPERANDI's best-per-exam export, without reference membership."""
    # phase is intentionally empty for phase_status=NOT_APPLICABLE.
    # Keep the column mandatory, but do not require every row to carry a phase.
    require(
        df,
        [*EXAM, "volume_id", "series_id", "Modality", "phase"],
        allow_empty={"phase"},
    )
    if df["volume_id"].duplicated().any():
        raise ValueError("Expected one row per volume_id")
    data = df.copy()
    data["date"] = normalize_dates(data["date"])
    if target == "mri_multiphase":
        # A canonical phase counts as solved; reject inconsistent status
        # annotations when the optional phase_status column is present.
        eligible = mri_eligibility(data)
    else:
        eligible = data["Modality"].eq("CT") & data["phase"].eq("PORTAL_VENOUS")
    candidates = data.loc[eligible].copy()
    if candidates.duplicated([*EXAM, "phase"]).any():
        raise ValueError("Use dicom_index_curated.csv: duplicate exam/phase candidates")
    candidates["_score"] = pd.to_numeric(
        candidates.get("selection_score", pd.Series(0, index=candidates.index)),
        errors="coerce",
    ).fillna(0)
    if target == "mri_multiphase":
        # Rank dates by distinct canonical phase coverage across same-day
        # study UIDs. Repeated reconstructions must not inflate coverage.
        dates = candidates.groupby(MRI_EXAM, as_index=False).agg(
            n_resolved_phases=("phase", "nunique")
        )
        dates = dates.sort_values(
            ["patient_key", "n_resolved_phases", "date"],
            ascending=[True, False, True],
        ).drop_duplicates("patient_key")
        # Choose one best candidate per phase on the selected date. Retain
        # source study UIDs, and never combine phases across different dates.
        selected = candidates.sort_values(
            [*MRI_EXAM, "phase", "_score", "study_id", "volume_id"],
            ascending=[True, True, True, False, True, True],
        ).drop_duplicates([*MRI_EXAM, "phase"])
        selected = selected.merge(
            dates,
            on=MRI_EXAM,
            validate="many_to_one",
        )
        selected["complete_multiphase"] = selected["n_resolved_phases"].eq(len(PHASES))
    else:
        selected = candidates.sort_values(
            ["patient_key", "date", "_score", "study_id", "volume_id"],
            ascending=[True, True, False, True, True],
        ).drop_duplicates("patient_key")
    selected = selected.drop(columns="_score").sort_values([*EXAM, "phase"])
    audit = data.copy()
    audit["selected"] = audit["volume_id"].isin(selected["volume_id"]).astype(int)
    audit["selection_reason"] = "outside_required_modality_sequence_or_phase"
    audit.loc[eligible, "selection_reason"] = (
        "fewer_resolved_phases_later_date_or_lower_ranked_candidate"
        if target == "mri_multiphase"
        else "later_date_or_lower_ranked_scan"
    )
    audit.loc[audit["selected"].eq(1), "selection_reason"] = "selected"
    return selected, audit


def load_reference(path, target):
    ref = read_csv(path)
    if "subject" in ref:
        parsed = ref["subject"].str.extract(
            r"^lihc__(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})__(\d{4}-\d{2}-\d{2})__VEN\.nii\.gz$"
        )
        if parsed.isna().any().any():
            raise ValueError("Unexpected WSP reference filename")
        ref["patient_key"], ref["date"] = parsed[0], parsed[1]
        ref["phase"] = "PORTAL_VENOUS"
    require(ref, ["patient_key", "date"])
    ref["date"] = normalize_dates(ref["date"])
    if "phase" not in ref and target == "mri_multiphase":
        ref = ref.merge(pd.DataFrame({"phase": PHASES}), how="cross")
    require(ref, ["phase"])
    if ref.duplicated(["patient_key", "date", "phase"]).any():
        raise ValueError("Duplicate patient/date/phase reference rows")
    return ref


def mri_phases_by_exam(data):
    """Count distinct canonical phases across study UIDs on the same date."""
    require(data, [*MRI_EXAM, "phase"])
    data = data.loc[data["phase"].isin(PHASES)].copy()
    data["date"] = normalize_dates(data["date"])
    return data.groupby(MRI_EXAM)["phase"].agg(set).to_dict()


def compare_mri_phases(selected, reference):
    """Report phase recovery for every reference patient/date, including misses."""
    predicted = mri_phases_by_exam(selected)
    truth = mri_phases_by_exam(reference)
    rows = []
    for (patient, date), expected in sorted(truth.items()):
        phases = predicted.get((patient, date), set())
        matched = phases & expected
        rows.append(
            {
                "patient_key": patient,
                "date": date,
                "n_selected_phases": len(phases),
                "n_reference_phases": len(expected),
                "n_matched_phases": len(matched),
                "matched_phases": "|".join(p for p in PHASES if p in matched),
                "missing_phases": "|".join(p for p in PHASES if p in expected - phases),
                "full_phase_match": len(matched) == len(PHASES),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            *MRI_EXAM,
            "n_selected_phases",
            "n_reference_phases",
            "n_matched_phases",
            "matched_phases",
            "missing_phases",
            "full_phase_match",
        ],
    )


def compare_reference_mri_phases(inventory, reference):
    """Check required phases on published dates before cohort date selection."""
    candidates = inventory.loc[mri_eligibility(inventory)]
    return compare_mri_phases(candidates, reference).rename(
        columns={"n_selected_phases": "n_detected_phases"}
    )


def compare_cohorts(selected, reference, *, target=None):
    require(selected, ["patient_key", "date", "phase"])
    selected = selected.copy()
    selected["date"] = normalize_dates(selected["date"])
    levels = {
        "patient": ["patient_key"],
        "patient_date": ["patient_key", "date"],
        "patient_date_phase": ["patient_key", "date", "phase"],
    }
    # Published dates are study proxies, not proof of identical DICOM series.
    if "series_id" in reference and reference["series_id"].str.strip().ne("").all():
        require(selected, ["series_id"])
        levels["series"] = ["patient_key", "date", "phase", "series_id"]
    if target == "mri_multiphase":
        levels["patient_date_full_phase"] = MRI_EXAM
    metrics, mismatches = [], []
    for level, columns in levels.items():
        if level == "patient_date_full_phase":
            pred = {
                exam
                for exam, phases in mri_phases_by_exam(selected).items()
                if len(phases) == len(PHASES)
            }
            truth = {
                exam
                for exam, phases in mri_phases_by_exam(reference).items()
                if len(phases) == len(PHASES)
            }
        else:
            pred = set(selected[columns].itertuples(index=False, name=None))
            truth = set(reference[columns].itertuples(index=False, name=None))
        tp, fp, fn = len(pred & truth), len(pred - truth), len(truth - pred)
        metrics.append(
            {
                "level": level,
                "predicted": len(pred),
                "reference": len(truth),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / len(pred) if pred else None,
                "recall": tp / len(truth) if truth else None,
                "f1": 2 * tp / (len(pred) + len(truth)) if pred or truth else None,
                "jaccard": tp / len(pred | truth) if pred or truth else None,
            }
        )
        for status, items in (
            ("false_positive", pred - truth),
            ("false_negative", truth - pred),
        ):
            for item in sorted(items):
                mismatches.append(
                    {"level": level, "status": status, **dict(zip(columns, item))}
                )
    details = pd.DataFrame(mismatches).reindex(
        columns=["level", "status", "patient_key", "date", "phase", "series_id"]
    )
    return pd.DataFrame(metrics), details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("target", choices=["mri_multiphase", "ct_portal_venous"])
    build.add_argument("inventory", type=Path)
    build.add_argument("output_dir", type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("target", choices=["mri_multiphase", "ct_portal_venous"])
    compare.add_argument("selected", type=Path)
    compare.add_argument("reference", type=Path)
    compare.add_argument("output_dir", type=Path)
    compare.add_argument(
        "--inventory",
        type=Path,
        help="All-date MRI candidate inventory for alternate reference-date phase detection.",
    )
    args = parser.parse_args()
    if (
        args.command == "compare"
        and args.inventory is not None
        and args.target != "mri_multiphase"
    ):
        parser.error("--inventory is supported only for mri_multiphase comparison")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "build":
        selected, audit = build_cohort(read_csv(args.inventory), args.target)
        selected.to_csv(args.output_dir / "selected_cohort.csv", index=False)
        audit.to_csv(args.output_dir / "selection_audit.csv", index=False)
        print(
            f"Selected {len(selected)} volumes / {selected['patient_key'].nunique()} patients"
        )
    else:
        selected = read_csv(args.selected)
        selected["date"] = normalize_dates(selected["date"])
        ref = load_reference(args.reference, args.target)
        metrics, mismatches = compare_cohorts(selected, ref, target=args.target)
        metrics.to_csv(args.output_dir / "cohort_metrics.csv", index=False)
        mismatches.to_csv(args.output_dir / "cohort_mismatches.csv", index=False)
        summary = {
            "target": args.target,
            "series_comparison_available": "series" in set(metrics["level"]),
        }
        if args.target == "mri_multiphase":
            # Patient coverage is intentionally permissive; phase-level recall
            # still penalizes all missing acquisitions in partial studies.
            phase_counts = selected.groupby(MRI_EXAM)["phase"].nunique()
            summary["selected_complete_mri_exams"] = int(
                phase_counts.eq(len(PHASES)).sum()
            )
            summary["selected_partial_mri_exams"] = int(
                phase_counts.lt(len(PHASES)).sum()
            )
            summary["selected_mri_phase_count_distribution"] = {
                str(count): int(n)
                for count, n in phase_counts.value_counts().sort_index().items()
            }
            phase_matches = compare_mri_phases(selected, ref)
            phase_matches.to_csv(args.output_dir / "mri_phase_matches.csv", index=False)
            summary["reference_mri_exams"] = len(phase_matches)
            summary["matched_mri_phases"] = int(phase_matches["n_matched_phases"].sum())
            summary["matched_complete_mri_exams"] = int(
                phase_matches["full_phase_match"].sum()
            )
            summary["matched_partial_mri_exams"] = int(
                (
                    phase_matches["n_matched_phases"].gt(0)
                    & ~phase_matches["full_phase_match"].astype(bool)
                ).sum()
            )
            summary["matched_mri_phase_count_distribution"] = {
                str(count): int(phase_matches["n_matched_phases"].eq(count).sum())
                for count in range(len(PHASES) + 1)
            }
            if args.inventory is not None:
                reference_phase_matches = compare_reference_mri_phases(
                    read_csv(args.inventory), ref
                )
                reference_phase_matches.to_csv(
                    args.output_dir / "mri_reference_phase_matches.csv", index=False
                )
                summary["mri_reference_phase_detection"] = {
                    "inventory": str(args.inventory),
                    "reference_exams": len(reference_phase_matches),
                    "required_phases": int(
                        reference_phase_matches["n_reference_phases"].sum()
                    ),
                    "detected_required_phases": int(
                        reference_phase_matches["n_matched_phases"].sum()
                    ),
                    "full_phase_match_exams": int(
                        reference_phase_matches["full_phase_match"].sum()
                    ),
                    "partial_phase_match_exams": int(
                        (
                            reference_phase_matches["n_matched_phases"].gt(0)
                            & ~reference_phase_matches["full_phase_match"].astype(bool)
                        ).sum()
                    ),
                    "matched_phase_count_distribution": {
                        str(count): int(
                            reference_phase_matches["n_matched_phases"].eq(count).sum()
                        )
                        for count in range(len(PHASES) + 1)
                    },
                }
        if "include_tumor" in ref:
            tumor = set(ref.loc[ref["include_tumor"].eq("1"), "patient_key"])
            summary["reference_tumor_patients"] = len(tumor)
            summary["reference_tumor_patients_recovered"] = len(
                tumor & set(selected["patient_key"])
            )
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(metrics.to_string(index=False))
        if args.target == "mri_multiphase":
            print("\nMRI matched phases per reference exam (same patient/date):")
            print(
                phase_matches[
                    [*MRI_EXAM, "n_matched_phases", "full_phase_match"]
                ].to_string(index=False)
            )
            print(
                f"Full four-phase match: {summary['matched_complete_mri_exams']} / "
                f"{summary['reference_mri_exams']} reference exams"
            )
            if args.inventory is not None:
                print("\nAlternate MRI phase detection on reference patient/dates:")
                print(
                    reference_phase_matches[
                        [*MRI_EXAM, "n_matched_phases", "full_phase_match"]
                    ].to_string(index=False)
                )
                detection = summary["mri_reference_phase_detection"]
                print(
                    f"Detected required phases: {detection['detected_required_phases']} / "
                    f"{detection['required_phases']}; full four-phase match: "
                    f"{detection['full_phase_match_exams']} / "
                    f"{detection['reference_exams']} reference exams"
                )


if __name__ == "__main__":
    main()
