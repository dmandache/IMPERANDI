"""Adapters for image conversion, phase prediction, segmentation and radiomics."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

from imperandi.io.tables import read_table
from imperandi.process.registration import register_pairs

from ..base import PipelineStage, RunContext, StageResult
from .core import _read_required

logger = logging.getLogger(__name__)


def _bridge_csv(stage_dir: Path, name: str, frame: pd.DataFrame) -> Path:
    path = stage_dir / f"_{name}.csv"
    frame.to_csv(path, index=False)
    return path


def _load_error(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


class ConvertStage(PipelineStage):
    name = "05_convert"
    requires = frozenset({"volumes_shortlist"})
    produces = frozenset({"volumes_converted"})

    def mode(self, config) -> str:
        return "active" if config.conversion.enabled else "pass_through"

    def run(self, context: RunContext) -> StageResult:
        volumes = _read_required(context, "volumes_shortlist")
        if not context.config.conversion.enabled or volumes.empty:
            output = context.write_table(self.name, "volumes_converted", volumes)
            return StageResult(
                artifacts={"volumes_converted": output},
                metrics={"converted": 0, "skipped": len(volumes)},
            )

        from imperandi.process import convert as convert_module

        stage_dir = context.stage_dir(self.name)
        if "patient_id" not in volumes.columns:
            raise ValueError("Conversion input requires canonical patient_id")
        # The conversion backend still uses patient_key to construct its private
        # image directory layout. Keep that alias inside the bridge only.
        backend_input = volumes.copy()
        backend_input["patient_key"] = backend_input["patient_id"]
        input_csv = _bridge_csv(stage_dir, "volumes_shortlist", backend_input)
        output_csv = stage_dir / "_volumes_converted.csv"
        error_csv = stage_dir / "_convert_errors.csv"
        image_dir = stage_dir / "images"
        workers = context.config.conversion.workers or context.config.execution.workers
        argv = [
            str(input_csv),
            str(image_dir),
            "--csv_path_out",
            str(output_csv),
            "--error_csv_path",
            str(error_csv),
            "--num_workers",
            str(workers),
            "--archive_max_depth",
            str(context.config.input.archive_depth),
            "--checkpoint_every_rows",
            str(context.config.execution.checkpoint_every_rows),
            "--checkpoint_every_sec",
            str(context.config.execution.checkpoint_every_seconds),
        ]
        args = convert_module.build_parser().parse_args(argv)
        args = convert_module.normalize_convert_args(args)
        convert_module.main(args)
        converted = pd.read_csv(output_csv).drop(
            columns=["patient_key"], errors="ignore"
        )
        errors = _load_error(error_csv).drop(columns=["patient_key"], errors="ignore")
        output = context.write_table(self.name, "volumes_converted", converted)
        artifacts = {"volumes_converted": output}
        if not errors.empty:
            artifacts["convert_errors"] = context.write_table(
                self.name, "convert_errors", errors
            )
        return StageResult(
            artifacts=artifacts,
            errors=errors,
            metrics={"converted": len(converted), "errors": len(errors)},
        )


PHASE_LABELS = {
    "native": "NATIVE",
    "non_contrast": "NATIVE",
    "non-contrast": "NATIVE",
    "arterial": "ARTERIAL",
    "arterial_early": "ARTERIAL",
    "arterial_late": "ARTERIAL",
    "portal": "PORTAL_VENOUS",
    "portal_venous": "PORTAL_VENOUS",
    "venous": "PORTAL_VENOUS",
    "delayed": "DELAYED",
}


def _canonical_image_phase(value):
    if pd.isna(value):
        return pd.NA
    key = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return PHASE_LABELS.get(key, pd.NA)


class PredictPhaseStage(PipelineStage):
    name = "06_predict_phase"
    requires = frozenset({"volumes_converted"})
    produces = frozenset({"volumes_predicted"})

    def mode(self, config) -> str:
        return "active" if config.phase_prediction.enabled else "pass_through"

    @staticmethod
    def _prediction_mask(
        df: pd.DataFrame, scope: str, modalities: list[str]
    ) -> pd.Series:
        modality_source = df.get(
            "curation_modality",
            df.get("Modality", pd.Series("", index=df.index)),
        )
        modality = modality_source.astype("string").str.upper().replace({"MRI": "MR"})
        eligible_modality = modality.isin(modalities)
        explicit = pd.Series(False, index=df.index)
        for column in ["phase_ontology", "phase_rules_explicit"]:
            if column in df.columns:
                explicit |= df[column].notna() & df[column].astype(
                    "string"
                ).str.upper().ne("OTHER")
        if scope == "all_eligible":
            return eligible_modality
        if scope == "selected_and_unresolved":
            selected = df.get("slot_ontology", pd.Series(pd.NA, index=df.index)).notna()
            selected |= df.get(
                "slot_rules_explicit", pd.Series(pd.NA, index=df.index)
            ).notna()
            return eligible_modality & (selected | ~explicit)
        return eligible_modality & ~explicit

    def run(self, context: RunContext) -> StageResult:
        converted = _read_required(context, "volumes_converted")
        config = context.config.phase_prediction
        if not config.enabled or converted.empty:
            output = context.write_table(self.name, "volumes_predicted", converted)
            return StageResult(
                artifacts={"volumes_predicted": output},
                metrics={"predicted": 0},
            )

        from imperandi.extract import phase as phase_module

        mask = self._prediction_mask(converted, config.scope, config.modalities)
        subset = converted[mask].copy()
        if subset.empty:
            output = context.write_table(self.name, "volumes_predicted", converted)
            return StageResult(
                artifacts={"volumes_predicted": output}, metrics={"predicted": 0}
            )

        stage_dir = context.stage_dir(self.name)
        input_csv = _bridge_csv(stage_dir, "phase_candidates", subset)
        output_csv = stage_dir / "_phase_predictions.csv"
        error_csv = stage_dir / "_phase_errors.csv"
        argv = [
            str(input_csv),
            str(output_csv),
            "--error_csv_path",
            str(error_csv),
            "--checkpoint_every_rows",
            str(context.config.execution.checkpoint_every_rows),
            "--checkpoint_every_sec",
            str(context.config.execution.checkpoint_every_seconds),
        ]
        args = phase_module.build_parser().parse_args(argv)
        args = phase_module.normalize_phase_args(args)
        phase_module.main(args)
        predictions = pd.read_csv(output_csv)
        if "volume_id" in predictions.columns:
            merge_key = "volume_id"
        elif "nifti_path" in predictions.columns:
            merge_key = "nifti_path"
        else:
            raise ValueError("Phase prediction output requires volume_id or nifti_path")
        if merge_key not in converted.columns:
            raise ValueError(
                f"Phase prediction output cannot be joined: missing {merge_key!r}"
            )
        predictions = predictions.drop_duplicates(merge_key, keep="last")
        prediction_columns = [
            column for column in predictions.columns if column.startswith("totalseg_")
        ]
        predicted = converted.merge(
            predictions[[merge_key, *prediction_columns]],
            on=merge_key,
            how="left",
            suffixes=("", "_new"),
        )
        for column in prediction_columns:
            new_column = f"{column}_new"
            if new_column in predicted.columns:
                predicted[column] = predicted[new_column].combine_first(
                    predicted[column]
                )
                predicted = predicted.drop(columns=[new_column])
        predicted["phase_image"] = predicted.get(
            "totalseg_phase", pd.Series(pd.NA, index=predicted.index)
        ).apply(_canonical_image_phase)
        confidence = pd.to_numeric(
            predicted.get(
                "totalseg_probability",
                predicted.get(
                    "totalseg_confidence", pd.Series(pd.NA, index=predicted.index)
                ),
            ),
            errors="coerce",
        )
        predicted["phase_image_confidence"] = confidence
        predicted.loc[
            confidence.notna() & confidence.lt(config.minimum_confidence), "phase_image"
        ] = pd.NA
        errors = _load_error(error_csv)
        output = context.write_table(self.name, "volumes_predicted", predicted)
        artifacts = {"volumes_predicted": output}
        if not errors.empty:
            artifacts["phase_prediction_errors"] = context.write_table(
                self.name, "phase_prediction_errors", errors
            )
        return StageResult(
            artifacts=artifacts,
            errors=errors,
            metrics={"predicted": len(predictions), "errors": len(errors)},
        )


def _segmentation_backend_config(context: RunContext, modality: str) -> dict:
    tasks = []
    for task in context.config.segmentation.tasks:
        if modality != task.modality:
            continue
        item = {
            "task": task.task,
            "output": task.output,
            "extra": task.parameters,
        }
        if task.fetch_output:
            item["fetch_output"] = task.fetch_output
        tasks.append(item)
    result: dict = {"backend": "totalsegmentator", "tasks": tasks}
    postprocess = context.config.segmentation.postprocess.get(modality)
    if postprocess is None:
        postprocess = context.config.segmentation.postprocess.get("default")
    if postprocess:
        result["postprocess"] = postprocess
    return result


class SegmentStage(PipelineStage):
    name = "08_segment"
    requires = frozenset({"selected_volumes"})
    produces = frozenset({"volumes_segmented"})

    def mode(self, config) -> str:
        return "active" if config.segmentation.enabled else "pass_through"

    def run(self, context: RunContext) -> StageResult:
        selected = _read_required(context, "selected_volumes")
        if not context.config.segmentation.enabled or selected.empty:
            output = context.write_table(self.name, "volumes_segmented", selected)
            return StageResult(
                artifacts={"volumes_segmented": output}, metrics={"segmented": 0}
            )

        from imperandi.process import segment as segment_module

        stage_dir = context.stage_dir(self.name)
        outputs = []
        error_frames = []
        modality_series = (
            selected.get(
                "curation_modality",
                selected.get("Modality", pd.Series("", index=selected.index)),
            )
            .astype("string")
            .str.upper()
            .replace({"MRI": "MR"})
        )
        for modality in sorted(set(modality_series.dropna())):
            subset = selected[modality_series.eq(modality)].copy()
            backend_config = _segmentation_backend_config(context, modality)
            if not backend_config["tasks"]:
                outputs.append(subset)
                continue
            prefix = modality.lower()
            input_csv = _bridge_csv(stage_dir, f"{prefix}_selected", subset)
            output_csv = stage_dir / f"_{prefix}_segmented.csv"
            error_csv = stage_dir / f"_{prefix}_segment_errors.csv"
            backend_config_path = stage_dir / f"_{prefix}_segment_backend.json"
            backend_config_path.write_text(
                json.dumps({"segmentation": backend_config}, indent=2),
                encoding="utf-8",
            )
            argv = [
                str(input_csv),
                str(output_csv),
                "--error_csv_path",
                str(error_csv),
                "--manifest",
                str(backend_config_path),
                "--num_workers",
                str(context.config.execution.workers),
                "--checkpoint_every_rows",
                str(context.config.execution.checkpoint_every_rows),
                "--checkpoint_every_sec",
                str(context.config.execution.checkpoint_every_seconds),
            ]
            args = segment_module.build_parser().parse_args(argv)
            args = segment_module.normalize_segment_args(args)
            segment_module.main(args)
            outputs.append(pd.read_csv(output_csv))
            errors = _load_error(error_csv)
            if not errors.empty:
                error_frames.append(errors)
        segmented = (
            pd.concat(outputs, ignore_index=True, sort=False) if outputs else selected
        )
        errors = (
            pd.concat(error_frames, ignore_index=True, sort=False)
            if error_frames
            else pd.DataFrame()
        )
        output = context.write_table(self.name, "volumes_segmented", segmented)
        artifacts = {"volumes_segmented": output}
        if not errors.empty:
            artifacts["segment_errors"] = context.write_table(
                self.name, "segment_errors", errors
            )
        return StageResult(
            artifacts=artifacts,
            errors=errors,
            metrics={"segmented": len(segmented), "errors": len(errors)},
        )


class RegistrationStage(PipelineStage):
    name = "09_register"
    requires = frozenset({"volumes_segmented"})
    produces = frozenset({"volumes_registered"})

    def mode(self, config) -> str:
        return "active" if config.registration.enabled else "pass_through"

    def run(self, context: RunContext) -> StageResult:
        volumes = _read_required(context, "volumes_segmented")
        config = context.config.registration
        if not config.enabled:
            output = context.write_table(self.name, "volumes_registered", volumes)
            return StageResult(
                artifacts={"volumes_registered": output}, metrics={"registered": 0}
            )
        if config.pairs is None:
            raise ValueError(
                "registration.pairs is required when registration is enabled"
            )
        pairs = read_table(config.pairs)
        registrations, errors = register_pairs(
            pairs,
            volumes,
            output_dir=context.stage_dir(self.name) / "images",
            transform=config.transform,
        )
        if registrations.empty:
            enriched = volumes.copy()
        else:

            def collapse(values: pd.Series):
                present = list(dict.fromkeys(values.dropna().tolist()))
                if not present:
                    return pd.NA
                return present[0] if len(present) == 1 else present

            registration_summary = registrations.groupby(
                "moving_volume_id", as_index=False, dropna=False
            ).agg(collapse)
            enriched = volumes.merge(
                registration_summary,
                left_on="volume_id",
                right_on="moving_volume_id",
                how="left",
            )
        output = context.write_table(self.name, "volumes_registered", enriched)
        artifacts = {"volumes_registered": output}
        if not registrations.empty:
            artifacts["registration_pairs"] = context.write_table(
                self.name, "registration_pairs", registrations
            )
        if not errors.empty:
            artifacts["registration_errors"] = context.write_table(
                self.name, "registration_errors", errors
            )
        return StageResult(
            artifacts=artifacts,
            errors=errors,
            metrics={"registered": len(registrations), "errors": len(errors)},
        )


class RadiomicsStage(PipelineStage):
    name = "10_radiomics"
    requires = frozenset({"volumes_registered"})
    produces = frozenset({"radiomics_table"})

    def mode(self, config) -> str:
        return "active" if config.radiomics.enabled else "pass_through"

    def run(self, context: RunContext) -> StageResult:
        volumes = _read_required(context, "volumes_registered")
        config = context.config.radiomics
        if not config.enabled or volumes.empty:
            output = context.write_table(self.name, "radiomics_table", volumes)
            return StageResult(
                artifacts={"radiomics_table": output}, metrics={"radiomics_rows": 0}
            )

        from imperandi.extract import radiomics as radiomics_module

        subset = volumes
        if config.slots:
            subset = subset[subset["clinical_slot"].isin(config.slots)].copy()
        if config.masks:
            keep_masks = {f"mask_{name}" for name in config.masks}
            missing_masks = keep_masks - set(subset.columns)
            if missing_masks:
                raise ValueError(
                    f"Radiomics requested missing mask columns: {sorted(missing_masks)}"
                )
            drop_masks = [
                column
                for column in subset.columns
                if column.startswith("mask_") and column not in keep_masks
            ]
            subset = subset.drop(columns=drop_masks)
        if subset.empty:
            output = context.write_table(self.name, "radiomics_table", volumes)
            return StageResult(
                artifacts={"radiomics_table": output},
                metrics={"radiomics_rows": 0, "eligible_rows": 0},
            )
        stage_dir = context.stage_dir(self.name)
        input_csv = _bridge_csv(stage_dir, "radiomics_input", subset)
        output_csv = stage_dir / "_radiomics.csv"
        error_csv = stage_dir / "_radiomics_errors.csv"
        argv = [
            str(input_csv),
            str(output_csv),
            "--error_csv_path",
            str(error_csv),
            "--skip_filter",
            "--checkpoint_every_rows",
            str(context.config.execution.checkpoint_every_rows),
            "--checkpoint_every_sec",
            str(context.config.execution.checkpoint_every_seconds),
        ]
        if config.settings:
            argv.extend(["--pyradiomics_settings", str(config.settings)])
        args = radiomics_module.build_parser().parse_args(argv)
        args = radiomics_module.normalize_radiomics_args(args)
        radiomics_module.main(args)
        features = pd.read_csv(output_csv)
        errors = _load_error(error_csv)
        join_key = next(
            (
                column
                for column in ["volume_id", "nifti_path"]
                if column in volumes.columns and column in features.columns
            ),
            None,
        )
        if join_key is None:
            if len(subset) != len(volumes):
                raise ValueError(
                    "Filtered radiomics output requires volume_id or nifti_path "
                    "to rejoin the full cohort"
                )
            enriched = features
        else:
            feature_columns = [
                column for column in features.columns if column not in volumes.columns
            ]
            additions = features[[join_key, *feature_columns]].drop_duplicates(
                join_key, keep="last"
            )
            enriched = volumes.merge(additions, on=join_key, how="left")
        output = context.write_table(self.name, "radiomics_table", enriched)
        artifacts = {"radiomics_table": output}
        if not errors.empty:
            artifacts["radiomics_errors"] = context.write_table(
                self.name, "radiomics_errors", errors
            )
        return StageResult(
            artifacts=artifacts,
            errors=errors,
            metrics={"radiomics_rows": len(features), "errors": len(errors)},
        )


class PublishStage(PipelineStage):
    name = "11_publish"
    requires = frozenset({"radiomics_table"})
    produces = frozenset({"cohort_index"})

    def run(self, context: RunContext) -> StageResult:
        cohort = _read_required(context, "radiomics_table")
        stage_dir = context.stage_dir(self.name)
        artifacts = {}
        for table_format in context.config.output.publish_formats:
            path = stage_dir / f"cohort_index.{table_format.value}"
            from imperandi.io.tables import write_table

            write_table(cohort, path)
            artifacts[f"cohort_index_{table_format.value}"] = path
        canonical = artifacts.get(
            f"cohort_index_{context.config.output.table_format.value}"
        ) or next(iter(artifacts.values()))
        artifacts["cohort_index"] = canonical
        return StageResult(
            artifacts=artifacts,
            metrics={
                "rows": len(cohort),
                "patients": int(
                    cohort.get("patient_id", pd.Series(dtype=str)).nunique()
                ),
            },
        )
