"""Core metadata stages: index, identity, volume assembly, curation, selection."""

from __future__ import annotations

import glob
import hashlib
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pydicom.datadict import tag_for_keyword

from imperandi.annotations import apply_ontologies, apply_rule_packs, resolve_annotation
from imperandi.config.models import CLINICAL_SLOTS, CONTRAST_PHASES
from imperandi.curation import curate_by_modality
from imperandi.identity import resolve_patient_identities
from imperandi.ingest import clean as clean_module
from imperandi.ingest import parse as parse_module
from imperandi.io.tables import read_table, table_schema_path, warn_if_csv_is_large
from imperandi.utils.datetime import to_dates, to_times
from imperandi.utils.geometry import standardize_iop

from ..base import PipelineStage, RunContext, StageResult

logger = logging.getLogger(__name__)


def _read_required(context: RunContext, artifact: str) -> pd.DataFrame:
    try:
        path = context.artifacts[artifact]
    except KeyError as exc:
        raise ValueError(f"Required artifact {artifact!r} is unavailable") from exc
    return read_table(path)


def _concat_existing(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    return (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )


class IndexStage(PipelineStage):
    name = "01_index"
    produces = frozenset({"instances_raw"})

    def resume_token(self, config) -> str:
        """Fingerprint source inventory paths, sizes, and modification times."""
        digest = hashlib.sha256()
        digest.update(str(super().resume_token(config)).encode())

        def add_path(path: Path) -> None:
            try:
                resolved = path.expanduser().resolve()
                stat = resolved.stat()
                digest.update(
                    f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
                )
                if resolved.is_dir():
                    for child in sorted(
                        (item for item in resolved.rglob("*") if item.is_file()),
                        key=str,
                    ):
                        child_stat = child.stat()
                        digest.update(
                            (
                                f"{child.relative_to(resolved)}|{child_stat.st_size}|"
                                f"{child_stat.st_mtime_ns}\n"
                            ).encode()
                        )
            except OSError as exc:
                digest.update(f"unreadable:{path}:{exc.__class__.__name__}\n".encode())

        for source in config.input.sources:
            digest.update(f"source:{source}\n".encode())
            matches = sorted(glob.glob(source, recursive=True))
            if matches:
                for match in matches:
                    add_path(Path(match))
            else:
                add_path(Path(source))
        return digest.hexdigest()

    def _parse_source(
        self, context: RunContext, source: str, ordinal: int
    ) -> tuple[Path, Path]:
        source_path = Path(source)
        source_dir = context.stage_dir(self.name) / f"source_{ordinal:03d}"
        source_dir.mkdir(parents=True, exist_ok=True)

        if (
            source_path.suffix.lower() in {".csv", ".parquet"}
            and not source_path.exists()
        ):
            raise FileNotFoundError(f"Pre-indexed input table does not exist: {source}")
        if source_path.exists() and source_path.suffix.lower() in {".csv", ".parquet"}:
            destination = source_dir / f"dicom_index{source_path.suffix.lower()}"
            shutil.copy2(source_path, destination)
            source_schema = table_schema_path(source_path)
            if source_schema.exists():
                shutil.copy2(source_schema, table_schema_path(destination))
            return destination, source_dir / "dicom_index_errors.csv"

        identity_columns = [
            *context.config.identity.source.patient_id_columns,
            *context.config.identity.source.namespace_columns,
            *context.config.identity.source.fallback.columns,
        ]
        identity_tags = [
            column
            for column in dict.fromkeys(identity_columns)
            if tag_for_keyword(column)
        ]
        patient_tag = next(
            (
                column
                for column in context.config.identity.source.patient_id_columns
                if tag_for_keyword(column)
            ),
            "PatientID",
        )
        argv = [
            source,
            str(source_dir),
            "--patient_key_from",
            patient_tag,
            "--tags",
            ",".join(identity_tags),
            "--num_workers",
            str(context.config.execution.workers),
            "--archive_max_depth",
            str(context.config.input.archive_depth),
            "--checkpoint_every_rows",
            str(context.config.execution.checkpoint_every_rows),
            "--checkpoint_every_sec",
            str(context.config.execution.checkpoint_every_seconds),
        ]
        args = parse_module.build_parser(include_manifest=False).parse_args(argv)
        # The v2 project contract does not expose legacy dataset manifests, but
        # the parser backend still records this field in its run state.
        args.manifest = None
        args = parse_module.normalize_parse_args(args)
        parse_module.main(args)
        return source_dir / "dicom_index.csv", source_dir / "dicom_index_errors.csv"

    def run(self, context: RunContext) -> StageResult:
        tables: list[Path] = []
        errors: list[Path] = []
        for ordinal, source in enumerate(context.config.input.sources, start=1):
            table_path, error_path = self._parse_source(context, source, ordinal)
            tables.append(table_path)
            errors.append(error_path)

        frames = [read_table(path) for path in tables]
        instances = pd.concat(frames, ignore_index=True, sort=False)
        warn_if_csv_is_large(
            context.config.output.table_format,
            len(instances),
            log=logger,
        )
        output = context.write_table(self.name, "instances_raw", instances)
        error_df = _concat_existing(errors)
        artifacts = {"instances_raw": output}
        if not error_df.empty:
            artifacts["index_errors"] = context.write_table(
                self.name, "index_errors", error_df
            )
        return StageResult(
            artifacts=artifacts,
            errors=error_df,
            metrics={"instances": len(instances), "errors": len(error_df)},
        )


class IdentityStage(PipelineStage):
    name = "02_identity"
    requires = frozenset({"instances_raw"})
    produces = frozenset({"instances", "identity_map"})

    def run(self, context: RunContext) -> StageResult:
        raw = _read_required(context, "instances_raw")
        result = resolve_patient_identities(raw, context.config.identity)
        resolved = result.cohort[result.cohort["patient_id"].notna()].copy()
        unresolved = result.cohort[result.cohort["patient_id"].isna()].copy()
        instances_path = context.write_table(self.name, "instances", resolved)
        identity_path = context.write_table(self.name, "identity_map", result.sensitive)
        artifacts = {"instances": instances_path, "identity_map": identity_path}
        if not unresolved.empty:
            artifacts["instances_unresolved_identity"] = context.write_table(
                self.name, "instances_unresolved_identity", unresolved
            )
        if not result.qc_flags.empty:
            artifacts["identity_qc"] = context.write_table(
                self.name, "identity_qc", result.qc_flags
            )
        return StageResult(
            artifacts=artifacts,
            qc_flags=result.qc_flags,
            metrics={
                "rows": len(resolved),
                "unresolved_rows": len(unresolved),
                "patients": int(resolved["patient_id"].nunique(dropna=True)),
            },
        )


def _derive_pixel_spacing(df: pd.DataFrame) -> pd.DataFrame:
    if "PixelSpacing" not in df.columns:
        return df
    out = df.copy()

    def first(value):
        if isinstance(value, (list, tuple, np.ndarray)) and len(value):
            return value[0]
        if isinstance(value, str):
            try:
                parsed = clean_module.literal_eval(value)
                return (
                    parsed[0]
                    if isinstance(parsed, (list, tuple)) and parsed
                    else np.nan
                )
            except (ValueError, SyntaxError):
                return np.nan
        return np.nan

    out["PixelSpacingXY"] = pd.to_numeric(
        out["PixelSpacing"].apply(first), errors="coerce"
    )
    return out


class AssembleStage(PipelineStage):
    name = "03_assemble"
    requires = frozenset({"instances"})
    produces = frozenset({"volumes"})

    def run(self, context: RunContext) -> StageResult:
        df = _read_required(context, "instances")
        if df.empty:
            output = context.write_table(self.name, "volumes", df)
            return StageResult(artifacts={"volumes": output}, metrics={"volumes": 0})

        # Legacy geometry functions still use patient_key internally. It is a
        # transient alias; patient_id remains the public canonical identifier.
        df = df.copy()
        df["patient_key"] = df["patient_id"]
        df = to_dates(df)
        df = to_times(df)
        df = clean_module.add_date(df)
        df = clean_module.add_time(df)
        df = _derive_pixel_spacing(df)
        if "ImageOrientationPatient" in df.columns:
            df["ImageOrientationPatient"] = df["ImageOrientationPatient"].apply(
                standardize_iop
            )

        df = clean_module.generate_volume_id(df)
        df = clean_module.correct_volume_ids(df)
        volumes = clean_module.group_volumes(df)
        if "dicom_path" in volumes.columns:
            volumes = clean_module.calculate_volume_length(volumes)
        volumes = clean_module.compute_visit_order(volumes)
        volumes = clean_module.compute_acquisition_order(volumes)
        volumes = volumes.drop(columns=["patient_key"], errors="ignore")
        output = context.write_table(self.name, "volumes", volumes)
        return StageResult(
            artifacts={"volumes": output},
            metrics={
                "instances": len(df),
                "volumes": len(volumes),
                "patients": int(volumes["patient_id"].nunique(dropna=True)),
            },
        )


def _copy_builtin_annotations(curated: pd.DataFrame) -> pd.DataFrame:
    out = curated.copy()
    if "eligible" not in out.columns:
        out["eligible"] = True

    def fill_missing(target: str, mask: pd.Series, values) -> None:
        if target not in out.columns:
            out[target] = pd.NA
        apply_mask = mask & out[target].isna()
        out.loc[apply_mask, target] = (
            values.loc[apply_mask] if isinstance(values, pd.Series) else values
        )

    modality = (
        out.get(
            "curation_modality",
            out.get("Modality", pd.Series("", index=out.index)),
        )
        .astype("string")
        .str.upper()
    )
    is_ct = modality.eq("CT").fillna(False)
    is_mr = modality.isin(["MR", "MRI"]).fillna(False)

    if "ct_phase" in out.columns:
        valid = (
            is_ct & out["ct_phase"].notna() & out["ct_phase"].ne("OTHER").fillna(False)
        )
        fill_missing("phase_rules_explicit", valid, out["ct_phase"])
    if "mri_perfusion_label" in out.columns:
        valid = (
            is_mr
            & out["mri_perfusion_label"].notna()
            & out["mri_perfusion_label"].ne("OTHER").fillna(False)
        )
        source = out.get(
            "mri_perfusion_source", pd.Series("none", index=out.index)
        ).astype("string")
        explicit_source = source.eq("explicit_text").fillna(False)
        explicit = valid & explicit_source
        inferred = valid & ~explicit_source
        fill_missing("phase_rules_explicit", explicit, out["mri_perfusion_label"])
        fill_missing("phase_rules_inferred", inferred, out["mri_perfusion_label"])

    if "selection_slot" in out.columns:
        ct_slot = is_ct & out["selection_slot"].ne("CT_OTHER").fillna(False)
        fill_missing("slot_rules_explicit", ct_slot, out["selection_slot"])

        mr_slot = is_mr & out["selection_slot"].isin(["T2", "DWI"]).fillna(False)
        mr_slot_values = "MR_" + out["selection_slot"].astype(str)
        fill_missing("slot_rules_explicit", mr_slot, mr_slot_values)

        mr_t1 = is_mr & out["selection_slot"].astype("string").str.startswith(
            "T1_"
        ).fillna(False)
        mr_valid = mr_t1 & out["selection_slot"].ne("T1_OTHER").fillna(False)
        mr_source = out.get(
            "mri_perfusion_source", pd.Series("none", index=out.index)
        ).astype("string")
        mr_explicit_source = mr_source.eq("explicit_text").fillna(False)
        explicit = mr_valid & mr_explicit_source
        inferred = mr_valid & ~mr_explicit_source
        mr_t1_values = "MR_" + out["selection_slot"].astype(str)
        fill_missing("slot_rules_explicit", explicit, mr_t1_values)
        fill_missing("slot_rules_inferred", inferred, mr_t1_values)

    ct_localizer = out.get("ct_is_localizer", pd.Series(False, index=out.index))
    ct_derived = out.get("ct_is_derived_low_value", pd.Series(False, index=out.index))
    ct_rejected = is_ct & (
        ct_localizer.fillna(False).astype(bool) | ct_derived.fillna(False).astype(bool)
    )
    ct_newly_rejected = ct_rejected & out["eligible"].fillna(True).astype(bool)
    out.loc[ct_rejected, "eligible"] = False
    out.loc[ct_newly_rejected, "exclusion_reason"] = "ct_localizer_or_derived"

    if "mri_sequence" in out.columns:
        mr_rejected = is_mr & out["mri_sequence"].isin(["LOCALIZER", "KEY_IMAGES"])
        mr_newly_rejected = mr_rejected & out["eligible"].fillna(True).astype(bool)
        out.loc[mr_rejected, "eligible"] = False
        out.loc[mr_newly_rejected, "exclusion_reason"] = "mri_localizer_or_key_image"
    return out


class AnnotateStage(PipelineStage):
    name = "04_annotate"
    requires = frozenset({"volumes"})
    produces = frozenset({"volumes_annotated", "volumes_shortlist"})

    def run(self, context: RunContext) -> StageResult:
        volumes = _read_required(context, "volumes")
        annotated = apply_ontologies(volumes, context.config.annotations.ontologies)
        annotated = apply_rule_packs(annotated, context.config.annotations.rule_packs)
        results = curate_by_modality(
            annotated,
            patient_col="patient_id",
            study_col="study_id",
            date_col="date",
            contextual_strategies=context.config.annotations.contextual_strategies,
            curators=[
                reference
                for reference in context.config.annotations.rule_packs
                if reference.startswith("builtin:")
            ],
        )
        supported = results["curated_all"]
        other = results["other"].copy()
        if not other.empty:
            other["curation_modality"] = "OTHER"
            previously_eligible = other.get(
                "eligible", pd.Series(True, index=other.index)
            ).fillna(True)
            missing_reason = other.get(
                "exclusion_reason", pd.Series(pd.NA, index=other.index)
            ).isna()
            other["eligible"] = False
            other.loc[previously_eligible | missing_reason, "exclusion_reason"] = (
                "unsupported_modality"
            )
        annotated = pd.concat([supported, other], ignore_index=True, sort=False)
        annotated = _copy_builtin_annotations(annotated)
        annotated["eligible"] = annotated["eligible"].fillna(True).astype(bool)
        shortlist = annotated[annotated["eligible"]].copy()
        rejected = annotated[~annotated["eligible"]].copy()
        conflict_columns = [
            column for column in annotated.columns if column.endswith("_conflict")
        ]
        qc_mask = annotated.get(
            "qc_rule", pd.Series(pd.NA, index=annotated.index)
        ).notna()
        for column in conflict_columns:
            qc_mask |= annotated[column].fillna(False).astype(bool)
        annotation_qc = annotated[qc_mask].copy()

        artifacts = {
            "volumes_annotated": context.write_table(
                self.name, "volumes_annotated", annotated
            ),
            "volumes_shortlist": context.write_table(
                self.name, "volumes_shortlist", shortlist
            ),
        }
        if not rejected.empty:
            artifacts["volumes_rejected"] = context.write_table(
                self.name, "volumes_rejected", rejected
            )
        if not annotation_qc.empty:
            artifacts["annotation_qc"] = context.write_table(
                self.name, "annotation_qc", annotation_qc
            )
        return StageResult(
            artifacts=artifacts,
            qc_flags=annotation_qc,
            metrics={
                "annotated": len(annotated),
                "shortlisted": len(shortlist),
                "rejected": len(rejected),
                "qc_rows": len(annotation_qc),
            },
        )


def _phase_to_slot(row: pd.Series) -> str | None:
    phase_source = row.get("phase_source")
    if pd.isna(phase_source) or phase_source != "phase_image":
        return None
    phase = row.get("phase_resolved")
    if pd.isna(phase) or str(phase).upper() in {"", "OTHER", "UNKNOWN"}:
        return None
    modality = str(row.get("curation_modality", row.get("Modality", ""))).upper()
    if modality == "CT":
        return f"CT_{str(phase).upper()}"
    if modality in {"MR", "MRI"} and str(row.get("mri_sequence", "")).upper() == "T1":
        return f"MR_T1_{str(phase).upper()}"
    return None


def _validate_controlled_values(
    values: pd.Series, allowed: frozenset[str], label: str
) -> None:
    present = values.dropna().astype(str)
    invalid = set(present) - allowed
    if invalid:
        raise ValueError(f"Invalid {label} values: {sorted(invalid)}")


class ResolveSelectStage(PipelineStage):
    name = "07_resolve_select"
    requires = frozenset({"volumes_predicted"})
    produces = frozenset({"volumes_resolved", "selected_volumes"})

    def run(self, context: RunContext) -> StageResult:
        predicted = _read_required(context, "volumes_predicted")
        resolution = context.config.phase_prediction.resolution
        resolved = resolve_annotation(
            predicted,
            candidates=resolution.precedence,
            target="phase_resolved",
            source_target="phase_source",
            conflict_target="phase_conflict",
            disagreement=resolution.disagreement,
        )
        _validate_controlled_values(
            resolved["phase_resolved"], CONTRAST_PHASES, "contrast phase"
        )
        resolved["slot_image"] = resolved.apply(_phase_to_slot, axis=1)
        selection_config = context.config.selection
        resolved = resolve_annotation(
            resolved,
            candidates=selection_config.precedence,
            target="clinical_slot",
            source_target="clinical_slot_source",
            conflict_target="clinical_slot_conflict",
            disagreement=selection_config.disagreement,
        )
        _validate_controlled_values(
            resolved["clinical_slot"], CLINICAL_SLOTS, "clinical slot"
        )
        route = (
            resolved.get(
                "curation_modality",
                resolved.get("Modality", pd.Series("", index=resolved.index)),
            )
            .astype("string")
            .str.upper()
            .replace({"MRI": "MR"})
        )
        slot_route = resolved["clinical_slot"].astype("string").str.split("_").str[0]
        route_mismatch = (
            resolved["clinical_slot"].notna()
            & route.isin(["CT", "MR"])
            & slot_route.ne(route)
        )
        if route_mismatch.any():
            mismatches = sorted(
                {
                    f"{route_value}->{slot_value}"
                    for route_value, slot_value in zip(
                        route[route_mismatch],
                        resolved.loc[route_mismatch, "clinical_slot"],
                        strict=True,
                    )
                }
            )
            raise ValueError(
                "Clinical slot modality does not match routed modality: "
                f"{mismatches}"
            )

        evidence_order = {
            source: len(selection_config.precedence) - index
            for index, source in enumerate(selection_config.precedence)
        }
        candidates = resolved[
            resolved["eligible"].fillna(False) & resolved["clinical_slot"].notna()
        ].copy()
        candidates["_evidence_rank"] = (
            candidates["clinical_slot_source"].map(evidence_order).fillna(0)
        )
        quality_source = candidates.get(
            "selection_score", pd.Series(0, index=candidates.index)
        )
        candidates["_quality_rank"] = pd.to_numeric(
            quality_source, errors="coerce"
        ).fillna(0)
        candidates["_stable_volume"] = candidates.get(
            "volume_id", pd.Series(candidates.index.astype(str), index=candidates.index)
        ).astype(str)
        exam_columns = [
            column
            for column in ["patient_id", "study_id", "date"]
            if column in candidates.columns
        ]
        candidates = candidates.sort_values(
            [
                *exam_columns,
                "clinical_slot",
                "_evidence_rank",
                "_quality_rank",
                "_stable_volume",
            ],
            ascending=[True] * (len(exam_columns) + 1) + [False, False, True],
            kind="mergesort",
            na_position="last",
        )
        selected = (
            candidates.groupby(
                [*exam_columns, "clinical_slot"], as_index=False, dropna=False
            )
            .head(1)
            .drop(columns=["_evidence_rank", "_quality_rank", "_stable_volume"])
            .reset_index(drop=True)
        )

        qc_rows = []
        required = context.config.selection.required_slots
        if required and not resolved.empty:
            exams = resolved[exam_columns].drop_duplicates()
            for _, exam in exams.iterrows():
                mask = pd.Series(True, index=selected.index)
                for column in exam_columns:
                    mask &= (
                        selected[column]
                        .astype("string")
                        .fillna("")
                        .eq(str(exam[column]) if pd.notna(exam[column]) else "")
                    )
                modality_values = resolved.copy()
                for column in exam_columns:
                    modality_values = modality_values[
                        modality_values[column]
                        .astype("string")
                        .fillna("")
                        .eq(str(exam[column]) if pd.notna(exam[column]) else "")
                    ]
                modalities = {
                    str(value).upper()
                    for value in modality_values.get(
                        "curation_modality", pd.Series(dtype=str)
                    )
                    if pd.notna(value)
                }
                present = set(selected.loc[mask, "clinical_slot"].dropna())
                for modality in modalities:
                    modality_key = "MR" if modality == "MRI" else modality
                    for slot in required.get(modality_key, []):
                        if slot not in present:
                            qc_rows.append(
                                {
                                    **exam.to_dict(),
                                    "qc_code": "MISSING_CLINICAL_SLOT",
                                    "clinical_slot": slot,
                                    "severity": "warning",
                                }
                            )
        qc = pd.DataFrame(qc_rows)
        artifacts = {
            "volumes_resolved": context.write_table(
                self.name, "volumes_resolved", resolved
            ),
            "selected_volumes": context.write_table(
                self.name, "selected_volumes", selected
            ),
        }
        if not qc.empty:
            artifacts["selection_qc"] = context.write_table(
                self.name, "selection_qc", qc
            )
        return StageResult(
            artifacts=artifacts,
            qc_flags=qc,
            metrics={"resolved": len(resolved), "selected": len(selected)},
        )
