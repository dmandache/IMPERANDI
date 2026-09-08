"""Cohort orchestration; backend and consensus functions are replaceable interfaces."""

from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import uuid
import numpy as np
import pandas as pd

from imperandi.utils.run_state import atomic_write_json
from .config import RegistrationConfig
from .consensus import fuse_tumors
from .organ import backend, read_image, read_mask, register_pair, resample

logger = logging.getLogger(__name__)


def _key(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()[:20]


def _label(value):
    return "" if pd.isna(value) else str(value).strip().upper()


def _rank(row, priorities):
    rank = next(
        (
            i
            for i, selector in enumerate(priorities)
            if all(_label(row.get(k)) == _label(v) for k, v in selector.items())
        ),
        len(priorities),
    )
    return rank, str(row["registration_scan_id"])


def _write(image, path, *, transform=False):
    """Publish a complete artifact with a sibling temporary file."""
    sitk = backend()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{uuid.uuid4().hex}.{path.name}")
    try:
        (sitk.WriteTransform if transform else sitk.WriteImage)(image, str(temporary))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path.resolve())


def prepare_cohort(table, config):
    """Validate identities and initialize derived columns without loading images."""
    df = table.copy().reset_index(drop=True)
    required = [
        "patient_key",
        config.visit_column,
        "Modality",
        "nifti_path",
        config.organ_column,
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing registration columns: {sorted(missing)}")
    for col in ["patient_key", config.visit_column, "Modality", "nifti_path"]:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing registration identity: {col}")
    modalities = df.Modality.map(_label).replace({"MRI": "MR"})
    if not modalities.isin(["CT", "MR"]).all():
        raise ValueError("Registration supports CT and MR only")
    # Paths distinguish converted volumes even when series IDs are repeated.
    identities = [
        [
            str(row.patient_key),
            str(row[config.visit_column]),
            modalities[i],
            str(Path(row.nifti_path).expanduser().resolve()),
        ]
        for i, row in df.iterrows()
    ]
    ids = [_key(identity) for identity in identities]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scan identity in registration input")
    df["registration_scan_id"] = ids
    columns = [
        "registration_report_path",
        "registration_reference_id",
        "registration_status",
        "consensus_status",
        "reg_reference_to_scan_path",
        "reg_scan_to_reference_path",
        "reg_tumor_common_path",
        "reg_tumor_native_path",
        "reg_tumor_coverage_common_path",
        "reg_tumor_coverage_native_path",
        "reg_tumor_probability_common_path",
        "reg_tumor_probability_native_path",
        "reg_nifti_path",
        "reg_organ_path",
    ]
    for col in columns:
        df[col] = pd.Series([None] * len(df), dtype=object)
    df["registration_group_id"] = [
        _key((str(row.patient_key), str(row[config.visit_column]), modalities[i]))
        for i, row in df.iterrows()
    ]
    return df


def register_cohort(
    table,
    output_dir,
    config=None,
    *,
    pair_registration=register_pair,
    fusion=fuse_tumors,
):
    """Return (cohort, errors), grouping strictly by patient, visit and modality.

    This image-processing API performs fresh work. The CLI runner adds group
    scheduling, checkpointing, and resume around this function.
    pair_registration and fusion may be injected by alternative implementations.
    """
    config = config or RegistrationConfig()
    sitk = backend()
    df = prepare_cohort(table, config)
    ids = df.registration_scan_id.tolist()
    modalities = df.Modality.map(_label).replace({"MRI": "MR"})
    root = Path(output_dir).expanduser().resolve() / uuid.uuid4().hex
    root.mkdir(parents=True)
    errors = []

    def error(index, stage, exc):
        errors.append(
            {"registration_scan_id": ids[index], "stage": stage, "error": str(exc)}
        )

    groups = df.groupby(
        [df.patient_key, df[config.visit_column], modalities], sort=True
    )
    for group_key, group in groups:
        group_id = _key(tuple(str(value) for value in group_key))
        logger.info("Registering group %s (%d scans)", group_id, len(group))
        directory = root / group_id
        df.loc[group.index, "registration_group_id"] = group_id
        priorities = config.reference_priority.get(group_key[2], [])
        order = sorted(group.index, key=lambda i: _rank(df.loc[i], priorities))
        loaded = {}
        for i in order:
            try:
                image = read_image(df.at[i, "nifti_path"])
                organ = read_mask(df.at[i, config.organ_column], image)
                if np.count_nonzero(sitk.GetArrayViewFromImage(organ)) < 4:
                    raise ValueError("Empty or insufficient organ mask")
                loaded[i] = image, organ
            except (RuntimeError, ValueError, TypeError) as exc:
                df.at[i, "registration_status"] = "failed"
                error(i, "input", exc)
        if not loaded:
            df.loc[group.index, "consensus_status"] = "no_reference"
            continue
        ref = next(iter(loaded))
        reference, reference_organ = loaded[ref]
        df.loc[group.index, "registration_reference_id"] = ids[ref]
        transforms = {}
        pair_reports = []
        for i, (image, organ) in loaded.items():
            try:
                if i == ref:
                    tx = sitk.Euler3DTransform()
                    report = {
                        "stage": "reference",
                        "dice_before": 1.0,
                        "dice_after": 1.0,
                        "warnings": [],
                    }
                else:
                    result = pair_registration(reference_organ, organ, config)
                    tx = result.reference_to_scan
                    report = {
                        k: v
                        for k, v in vars(result).items()
                        if k != "reference_to_scan"
                    }
                inverse = tx.GetInverse()
                pair_dir = directory / ids[i]
                df.at[i, "reg_reference_to_scan_path"] = _write(
                    tx, pair_dir / "reference_to_scan.tfm", transform=True
                )
                df.at[i, "reg_scan_to_reference_path"] = _write(
                    inverse, pair_dir / "scan_to_reference.tfm", transform=True
                )
                df.at[i, "reg_nifti_path"] = _write(
                    resample(image, reference, tx, label=False),
                    pair_dir / "image_registered.nii.gz",
                )
                df.at[i, "reg_organ_path"] = _write(
                    resample(organ, reference, tx), pair_dir / "organ_registered.nii.gz"
                )
                transforms[i] = tx
                df.at[i, "registration_status"] = "reference" if i == ref else "ok"
                pair_reports.append({"scan_id": ids[i], **report})
            except (RuntimeError, ValueError) as exc:
                df.at[i, "registration_status"] = "failed"
                error(i, "organ", exc)
        if ref not in transforms:
            df.loc[group.index, "consensus_status"] = "no_reference"
            continue
        masks, coverages, contributors, native_masks = [], [], [], {}
        # Reference first, then availability in configured priority order.
        for i in [ref] + [i for i in order if i != ref]:
            if i not in transforms:
                continue
            path = df.at[i, config.tumor_column] if config.tumor_column in df else None
            if pd.isna(path) or not str(path).strip():
                continue
            try:
                tumor = read_mask(path, loaded[i][0])
                coverage = sitk.Image(tumor.GetSize(), sitk.sitkUInt8) + 1
                coverage.CopyInformation(tumor)
                masks.append(resample(tumor, reference, transforms[i]))
                coverages.append(resample(coverage, reference, transforms[i]))
                contributors.append(i)
                native_masks[i] = tumor
                if config.method == "anchor":
                    break
            except (RuntimeError, ValueError) as exc:
                error(i, "tumor_input", exc)
        df.loc[group.index, "consensus_status"] = "registration_failed"
        df.loc[list(transforms), "consensus_status"] = "no_tumor_input"
        if masks:
            try:
                fused = fusion(
                    masks, coverages, method=config.method, threshold=config.threshold
                )
                common = _write(fused.mask, directory / "tumor_common.nii.gz")
                support = _write(fused.coverage, directory / "coverage_common.nii.gz")
                probability = (
                    _write(fused.probability, directory / "probability_common.nii.gz")
                    if fused.probability is not None
                    else None
                )
                for i, tx in transforms.items():
                    try:
                        inverse = tx.GetInverse()
                        image = loaded[i][0]
                        pair_dir = directory / ids[i]
                        if config.method == "anchor":
                            composed = sitk.CompositeTransform(3)
                            composed.AddTransform(transforms[contributors[0]])
                            composed.AddTransform(inverse)
                            native = resample(
                                native_masks[contributors[0]], image, composed
                            )
                        else:
                            native = resample(fused.mask, image, inverse)
                        native_coverage = resample(fused.coverage, image, inverse)
                        native = sitk.And(native, native_coverage)
                        df.at[i, "reg_tumor_native_path"] = _write(
                            native, pair_dir / "tumor_native.nii.gz"
                        )
                        df.at[i, "reg_tumor_coverage_native_path"] = _write(
                            native_coverage,
                            pair_dir / "coverage_native.nii.gz",
                        )
                        if probability:
                            df.at[
                                i,
                                "reg_tumor_probability_native_path",
                            ] = _write(
                                resample(
                                    fused.probability, image, inverse, label=False
                                ),
                                pair_dir / "probability_native.nii.gz",
                            )
                        df.at[i, "reg_tumor_common_path"] = common
                        df.at[i, "reg_tumor_coverage_common_path"] = support
                        df.at[i, "reg_tumor_probability_common_path"] = probability
                        df.at[i, "consensus_status"] = (
                            "single_contributor" if len(masks) == 1 else "ok"
                        )
                    except (RuntimeError, ValueError) as exc:
                        df.at[i, "consensus_status"] = "failed"
                        error(i, "transfer", exc)
            except (RuntimeError, ValueError) as exc:
                df.loc[group.index, "consensus_status"] = "failed"
                error(ref, "consensus", exc)
        report_path = directory / "group.json"
        atomic_write_json(
            report_path,
            {
                "group": [str(value) for value in group_key],
                "reference": ids[ref],
                "config": asdict(config),
                "contributors": [ids[i] for i in contributors],
                "pairs": pair_reports,
                "simpleitk_version": sitk.Version_VersionString(),
                "errors": [
                    e
                    for e in errors
                    if e["registration_scan_id"] in set(group.registration_scan_id)
                ],
            },
        )
        df.loc[group.index, "registration_report_path"] = str(report_path.resolve())
    return df, pd.DataFrame(errors, columns=["registration_scan_id", "stage", "error"])
