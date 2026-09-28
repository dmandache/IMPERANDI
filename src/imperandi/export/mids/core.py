"""Planning and orchestration for the strict CT/MR volumetric MIDS profile."""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from imperandi import __version__
from imperandi.utils.manifest import load_manifest

from .identity import IdentityMapper
from .metadata import project, validate_allowlist
from .models import ExportFile, ExportPlan, MidsExportError
from .naming import (
    destination_for_image,
    image_basename,
    mask_label,
    nifti_extension,
    normalized_modality,
)
from .validation import validate_plan
from .writer import write_plan

logger = logging.getLogger(__name__)

PROFILE_NAME = "IMPERANDI strict CT/MR volumetric MIDS-style profile"
MIDS_PROPOSAL = "https://arxiv.org/abs/2010.00434"
MIDS_PUBLICATION = "https://pubmed.ncbi.nlm.nih.gov/35612110/"


def add_mids_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--csv_path", "--csv-path", required=True, help="Curated IMPERANDI cohort CSV."
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        required=True,
        help="New MIDS-style dataset directory.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="YAML with a mids_export block (default: bundled strict CT/MR profile).",
    )
    parser.add_argument(
        "--id-map",
        default=None,
        help=(
            "External CSV mapping each participant/examination to safe labels, "
            "or 'auto' to generate it. Never copied into the dataset."
        ),
    )
    parser.add_argument(
        "--id-map-output",
        "--id_map_output",
        default=None,
        help=(
            "Protected output for --id-map auto. Defaults to "
            "<cohort_stem>_mids_id_map.csv beside the cohort."
        ),
    )
    parser.add_argument(
        "--id-map-overwrite",
        "--id_map_overwrite",
        choices=["never", "replace"],
        default="never",
        help="Policy for a differing existing automatically generated ID map.",
    )
    parser.add_argument(
        "--patient-key-mode",
        "--patient_key_mode",
        choices=["map", "keep"],
        default="map",
        help=(
            "Map patients using fixed-width numeric labels, or keep path-safe "
            "patient keys (default: map)."
        ),
    )
    parser.add_argument(
        "--sort-columns",
        "--sort_columns",
        nargs="+",
        default=None,
        help="Ordering columns passed to the automatic ID-map generator.",
    )
    parser.add_argument(
        "--minimum-digits",
        "--minimum_digits",
        type=int,
        default=4,
        help=(
            "Minimum mapped patient/study/series/volume numeric width used by "
            "--id-map auto (default: 4)."
        ),
    )
    parser.add_argument(
        "--key-file",
        default=None,
        help="File containing a secret key for deterministic HMAC identifiers; never copied into the dataset.",
    )
    parser.add_argument(
        "--overwrite",
        choices=["never", "replace"],
        default="never",
        help="Existing-output policy. 'replace' stages and validates a complete replacement before swapping it in.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the complete plan, exclusions, missing fields, and collisions without writing.",
    )


def normalize_mids_args(args: argparse.Namespace) -> argparse.Namespace:
    csv_path = Path(args.csv_path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Not a CSV file: {csv_path}")
    output = Path(args.output_dir).expanduser()
    if output.is_symlink():
        raise MidsExportError("output_dir must not be a symbolic link.")
    if (
        output.resolve() == csv_path.resolve()
        or output.resolve() in csv_path.resolve().parents
    ):
        raise MidsExportError(
            "output_dir must be a separate dataset directory, not the source CSV or its parent."
        )
    args.csv_path = str(csv_path.resolve())
    args.output_dir = str(output.resolve())
    if args.minimum_digits < 1:
        raise MidsExportError("--minimum-digits must be at least 1.")
    if args.id_map and str(args.id_map).casefold() == "auto":
        args.id_map = "auto"
        auto_output = (
            Path(args.id_map_output).expanduser()
            if args.id_map_output
            else csv_path.with_name(f"{csv_path.stem}_mids_id_map.csv")
        ).resolve()
        if auto_output.suffix.lower() != ".csv":
            raise MidsExportError("--id-map-output must be a CSV file.")
        if auto_output == csv_path.resolve():
            raise MidsExportError(
                "The automatic ID map must not overwrite the cohort CSV."
            )
        if auto_output == output.resolve() or output.resolve() in auto_output.parents:
            raise MidsExportError(
                "The automatic ID map must remain outside output_dir."
            )
        args.id_map_output = str(auto_output)
    elif args.id_map:
        id_map = Path(args.id_map).expanduser().resolve()
        if id_map == output.resolve() or output.resolve() in id_map.parents:
            raise MidsExportError("The external ID map must remain outside output_dir.")
        args.id_map = str(id_map)
        if args.id_map_output:
            raise MidsExportError("--id-map-output is only valid with --id-map auto.")
    elif args.id_map_output:
        raise MidsExportError("--id-map-output requires --id-map auto.")
    if args.key_file:
        key_file = Path(args.key_file).expanduser().resolve()
        if key_file == output.resolve() or output.resolve() in key_file.parents:
            raise MidsExportError(
                "The identity key file must remain outside output_dir."
            )
        args.key_file = str(key_file)
    if args.id_map and args.key_file:
        raise MidsExportError("Use either --id-map or --key-file, not both.")
    return args


def _export_config(manifest_arg: str | None) -> dict[str, Any]:
    manifest = load_manifest(
        manifest_arg or "mids",
        base_path=Path(__file__).resolve().parents[2],
    )
    config = manifest.get("mids_export", manifest if manifest_arg else {})
    if not isinstance(config, dict) or not config:
        raise MidsExportError(
            "The YAML manifest must contain a non-empty mids_export mapping."
        )
    if config.get("profile") != "strict-ct-mr-volumetric-v1":
        raise MidsExportError(
            "Only mids_export.profile=strict-ct-mr-volumetric-v1 is supported."
        )
    for key in ("identity", "modality", "filename", "metadata"):
        if not isinstance(config.get(key), dict):
            raise MidsExportError(f"mids_export.{key} must be a mapping.")
    validate_allowlist(config["metadata"])
    mapping = config["modality"].get("mapping", {})
    if not isinstance(mapping, dict) or not {"CT", "MR"}.issubset(mapping):
        raise MidsExportError(
            "mids_export.modality.mapping must explicitly define CT and MR."
        )
    normalized_modality("CT", config["modality"])
    normalized_modality("MR", config["modality"])
    derivatives = config.get("derivatives", {})
    if not isinstance(derivatives, dict):
        raise MidsExportError("mids_export.derivatives must be a mapping.")
    masks = derivatives.get("masks", {})
    if not isinstance(masks, dict):
        raise MidsExportError("mids_export.derivatives.masks must be a mapping.")
    unsupported_mask_kinds = set(masks) - {"source", "consensus"}
    if unsupported_mask_kinds:
        raise MidsExportError(
            "Strict profile mask kinds are source and consensus; unsupported: "
            + ", ".join(sorted(unsupported_mask_kinds))
        )
    pipelines = [
        value.get("pipeline", f"imperandi-{kind}")
        for kind, value in masks.items()
        if isinstance(value, dict) and value.get("enabled", True)
    ]
    radiomics = derivatives.get("radiomics", {})
    if not isinstance(radiomics, dict):
        raise MidsExportError("mids_export.derivatives.radiomics must be a mapping.")
    if radiomics.get("enabled", True):
        pipelines.append(radiomics.get("pipeline", "imperandi-radiomics"))
    pipelines = [str(value) for value in pipelines if value]
    if any(not re.fullmatch(r"[a-z0-9][a-z0-9-]*", value) for value in pipelines):
        raise MidsExportError(
            "Derivative pipeline names must contain lowercase letters, digits, or hyphens."
        )
    if len(pipelines) != len(set(pipelines)):
        raise MidsExportError(
            "Every derivative kind must use a separate pipeline directory."
        )
    return config


def _row_ref(index: int) -> str:
    return f"row-{index + 1:06d}"


def _source_extension(path: Path) -> str:
    return ".nii.gz" if path.name.lower().endswith(".nii.gz") else ".nii"


def _relative_scan_path(image_path: Path) -> str:
    # Scan tables live directly in the subject/session directory.
    return Path(*image_path.parts[2:]).as_posix()


def _matches(column: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(column, pattern) for pattern in patterns)


def _derivative_status(row: pd.Series, config: dict[str, Any]) -> tuple[bool, str]:
    status = "not_provided"
    for column in config.get("qc_status_columns", []):
        if column in row and not pd.isna(row[column]) and str(row[column]).strip():
            status = str(row[column]).strip()
            break
    folded = status.casefold()
    rejected = {str(value).casefold() for value in config.get("rejected_statuses", [])}
    accepted = {str(value).casefold() for value in config.get("accepted_statuses", [])}
    if folded in rejected:
        return False, f"rejected QC status: {status}"
    if status != "not_provided" and accepted and folded not in accepted:
        return False, f"QC status is not accepted: {status}"
    threshold = config.get("minimum_confidence")
    if threshold is not None:
        for column in config.get("confidence_columns", []):
            if column in row and not pd.isna(row[column]):
                try:
                    confidence = float(row[column])
                except (TypeError, ValueError):
                    return False, f"invalid confidence in {column}"
                if confidence < float(threshold):
                    return False, f"confidence {confidence:g} is below {threshold}"
                status = f"{status}; confidence={confidence:g}"
                break
    return True, status


def _dataset_description(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset", {})
    description: dict[str, Any] = {
        "Name": str(dataset.get("name", "IMPERANDI MIDS-style export")),
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "IMPERANDI", "Version": __version__}],
        "MIDSCompatibility": {
            "Profile": "strict CT/MR volumetric",
            "Claim": "MIDS-style organization; full conformance is not claimed because no validation target is defined.",
            "Proposal": MIDS_PROPOSAL,
            "Publication": MIDS_PUBLICATION,
        },
    }
    for source, target in (
        ("license", "License"),
        ("authors", "Authors"),
        ("acknowledgements", "Acknowledgements"),
    ):
        if source in dataset:
            description[target] = dataset[source]
    return description


def _derivative_description(
    pipeline: str, config: dict[str, Any], kind: str
) -> dict[str, Any]:
    generated = config.get("generated_by", {})
    return {
        "Name": str(config.get("name", pipeline)),
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": str(generated.get("name", "IMPERANDI")),
                "Version": str(generated.get("version", __version__)),
                "Parameters": _sanitize_settings(generated.get("settings", {})),
            }
        ],
        "SourceDatasets": [{"URL": "../../", "Description": "Images in this export"}],
        "DerivativeKind": kind,
    }


def _sanitize_settings(value: Any, *, key: str = "") -> Any:
    """Remove secrets and local paths before recording effective settings."""
    folded = key.casefold().replace("-", "_")
    if folded in {"external_map", "id_map", "key_env", "key_file"} or (
        folded.endswith("_path")
        or "secret" in folded
        or "password" in folded
        or "token" in folded
    ):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            item_key: _sanitize_settings(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_settings(item, key=key) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_settings(item, key=key) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return "<redacted>"
    return value


def _safe_manifest(config: dict[str, Any]) -> str:
    safe = _sanitize_settings(copy.deepcopy(config))
    return yaml.safe_dump({"mids_export": safe}, sort_keys=False)


def _read_cohort(path: str, config: dict[str, Any]) -> pd.DataFrame:
    """Read identity columns as strings so leading zeros remain stable."""
    identity = config["identity"]
    identity_columns = {
        str(identity.get("subject_column", "patient_key")),
        *identity.get("session", {}).get("source_columns", ["study_id"]),
        *identity.get("image_source_columns", ["volume_id"]),
        "series_id",
    }
    available = set(pd.read_csv(path, nrows=0).columns)
    dtypes = {column: "string" for column in identity_columns & available}
    return pd.read_csv(path, dtype=dtypes, low_memory=False)


def build_plan(
    frame: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any],
    *,
    id_map_path: str | None = None,
    id_map_frame: pd.DataFrame | None = None,
    key_file: str | None = None,
) -> ExportPlan:
    """Build a deterministic plan without creating or modifying any files."""
    plan = ExportPlan(output_dir=output_dir)
    mapper = IdentityMapper(
        config["identity"],
        id_map_path=id_map_path,
        id_map_frame=id_map_frame,
        key_file=key_file,
    )
    modality_config = config["modality"]
    filename_config = config["filename"]
    metadata_config = config["metadata"]
    records: list[dict[str, Any]] = []
    participant_sources: dict[str, str] = {}
    session_sources: dict[tuple[str, str], tuple[str, ...]] = {}
    image_sources: dict[str, tuple[str, str, str]] = {}

    for position, (_, row) in enumerate(frame.iterrows()):
        ref = _row_ref(position)
        reasons: list[str] = []
        source_value = row.get("nifti_path")
        source = (
            Path(str(source_value)).expanduser()
            if not pd.isna(source_value) and str(source_value).strip()
            else None
        )
        if source is None:
            reasons.append("missing nifti_path")
        elif not nifti_extension(source):
            reasons.append("source is not a .nii or .nii.gz volume")
        elif not source.is_file():
            reasons.append("nifti_path does not exist")
        try:
            directory, suffix = normalized_modality(
                row.get("Modality"), modality_config
            )
        except MidsExportError as exc:
            reasons.append(str(exc))
            directory, suffix = "", ""
        try:
            participant, session, image_id = mapper.map_row(row)
        except MidsExportError as exc:
            reasons.append(str(exc))
            participant = session = image_id = ""
        if reasons:
            plan.excluded_rows.append({"row": ref, "reasons": reasons})
            for reason in reasons:
                if reason.startswith("Missing ") or reason.startswith("missing "):
                    field = reason.split(":", 1)[0]
                    plan.missing_fields.setdefault(field, []).append(ref)
            continue

        raw_subject = str(row[mapper.subject_column]).strip()
        if (
            participant in participant_sources
            and participant_sources[participant] != raw_subject
        ):
            plan.collisions.setdefault(f"sub-{participant}", []).append(ref)
        participant_sources[participant] = raw_subject
        raw_session = tuple(
            str(row[column]).strip() for column in mapper.session_columns
        )
        session_key = (participant, session)
        if (
            session_key in session_sources
            and session_sources[session_key] != raw_session
        ):
            plan.collisions.setdefault(f"sub-{participant}/ses-{session}", []).append(
                ref
            )
        session_sources[session_key] = raw_session
        if image_id in image_sources:
            _, _, first_ref = image_sources[image_id]
            plan.collisions.setdefault(
                f"sub-{participant}/ses-{session}/image-{image_id}",
                [first_ref],
            ).append(ref)
        else:
            image_sources[image_id] = (participant, session, ref)
        basename = image_basename(
            row,
            participant=participant,
            session=session,
            suffix=suffix,
            filename_config=filename_config,
        )
        assert source is not None
        preliminary = destination_for_image(participant, session, directory, basename)
        preliminary = preliminary.with_name(basename + _source_extension(source))
        records.append(
            {
                "row": row,
                "row_ref": ref,
                "participant": participant,
                "session": session,
                "image_id": image_id,
                "modality": suffix.upper(),
                "directory": directory,
                "suffix": suffix,
                "basename": basename,
                "source": source,
                "destination": preliminary,
            }
        )

    # MIDS run-<index> disambiguates volumes whose curated descriptions coincide.
    grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["destination"]].append(record)
    for destination, group in grouped.items():
        if len(group) == 1:
            continue
        ids = [record["image_id"] for record in group]
        if len(ids) != len(set(ids)) or "_run-" in group[0]["basename"]:
            plan.collisions[destination.as_posix()] = [
                record["row_ref"] for record in group
            ]
        if "_run-" in group[0]["basename"]:
            continue
        for run, record in enumerate(
            sorted(group, key=lambda item: (item["image_id"], item["row_ref"])), 1
        ):
            stem = record["basename"]
            prefix, suffix = stem.rsplit("_", 1)
            basename = f"{prefix}_run-{run:02d}_{suffix}"
            record["basename"] = basename
            record["destination"] = destination.with_name(
                basename + _source_extension(record["source"])
            )
        plan.resolved_collisions[destination.as_posix()] = [
            record["destination"].as_posix()
            for record in sorted(group, key=lambda item: item["destination"].as_posix())
        ]

    participant_rows: dict[str, dict[str, Any]] = {}
    session_rows: dict[tuple[str, str], dict[str, Any]] = {}
    scan_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    derivative_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    derivative_configs = config.get("derivatives", {}).get("masks", {})
    radiomics_config = config.get("derivatives", {}).get("radiomics", {})

    for record in sorted(records, key=lambda item: item["destination"].as_posix()):
        row = record["row"]
        destination: Path = record["destination"]
        participant = record["participant"]
        session = record["session"]
        image_id = record["image_id"]
        plan.files.append(ExportFile(record["source"], destination, "image", image_id))
        participant_rows.setdefault(
            participant,
            {
                "participant_id": f"sub-{participant}",
                **project(row, metadata_config.get("participants", {})),
            },
        )
        session_rows.setdefault(
            (participant, session),
            {
                "session_id": f"ses-{session}",
                **project(row, metadata_config.get("sessions", {})),
            },
        )
        scan_rows[(participant, session)].append(
            {
                "filename": _relative_scan_path(destination),
                "source_image_id": image_id,
                "modality": record["modality"],
                **project(row, metadata_config.get("scans", {})),
            }
        )

        for kind, derivative_config in derivative_configs.items():
            if not isinstance(derivative_config, dict) or not derivative_config.get(
                "enabled", True
            ):
                continue
            patterns = list(derivative_config.get("column_patterns", []))
            excludes = list(derivative_config.get("exclude_patterns", []))
            pipeline = str(derivative_config.get("pipeline", f"imperandi-{kind}"))
            allowed, qc_status = _derivative_status(row, derivative_config)
            for column in sorted(row.index):
                if not _matches(str(column), patterns) or _matches(
                    str(column), excludes
                ):
                    continue
                value = row[column]
                if pd.isna(value) or not str(value).strip():
                    continue
                if not allowed:
                    plan.excluded_rows.append(
                        {
                            "row": record["row_ref"],
                            "kind": kind,
                            "field": str(column),
                            "reasons": [qc_status],
                        }
                    )
                    continue
                mask_source = Path(str(value)).expanduser()
                if not mask_source.is_file() or not nifti_extension(mask_source):
                    plan.excluded_rows.append(
                        {
                            "row": record["row_ref"],
                            "kind": kind,
                            "field": str(column),
                            "reasons": ["mask path is missing or unsupported"],
                        }
                    )
                    continue
                label = mask_label(str(column), patterns)
                base_without_suffix = record["basename"].rsplit("_", 1)[0]
                descriptor = "consensus" if kind == "consensus" else "source"
                mask_name = f"{base_without_suffix}_desc-{descriptor}_label-{label}_seg{_source_extension(mask_source)}"
                mask_destination = (
                    Path("derivatives") / pipeline / destination.parent / mask_name
                )
                plan.files.append(
                    ExportFile(
                        mask_source,
                        mask_destination,
                        "consensus_mask" if kind == "consensus" else "source_mask",
                        image_id,
                        source_image=destination.as_posix(),
                    )
                )
                derivative_rows[pipeline].append(
                    {
                        "mask_path": mask_destination.as_posix(),
                        "source_image": destination.as_posix(),
                        "source_image_id": image_id,
                        "mask_kind": descriptor,
                        "label": label,
                        "qc_status": qc_status,
                    }
                )

        if radiomics_config.get("enabled", True):
            feature_patterns = list(radiomics_config.get("column_patterns", []))
            features: dict[str, Any] = {}
            for column in sorted(row.index):
                if _matches(str(column), feature_patterns) and not pd.isna(row[column]):
                    try:
                        features[str(column)] = float(row[column])
                    except (TypeError, ValueError):
                        continue
            if features:
                allowed, qc_status = _derivative_status(row, radiomics_config)
                if not allowed:
                    plan.excluded_rows.append(
                        {
                            "row": record["row_ref"],
                            "kind": "radiomics",
                            "reasons": [qc_status],
                        }
                    )
                    continue
                pipeline = str(radiomics_config.get("pipeline", "imperandi-radiomics"))
                derivative_rows[pipeline].append(
                    {
                        "source_image": destination.as_posix(),
                        "source_image_id": image_id,
                        "qc_status": qc_status,
                        **features,
                    }
                )

    plan.tables[Path("participants.tsv")] = [
        participant_rows[key] for key in sorted(participant_rows)
    ]
    for participant in sorted(participant_rows):
        sessions = [
            session_rows[key] for key in sorted(session_rows) if key[0] == participant
        ]
        plan.tables[Path(f"sub-{participant}") / f"sub-{participant}_sessions.tsv"] = (
            sessions
        )
    for (participant, session), rows in sorted(scan_rows.items()):
        path = (
            Path(f"sub-{participant}")
            / f"ses-{session}"
            / f"sub-{participant}_ses-{session}_scans.tsv"
        )
        plan.tables[path] = rows

    for pipeline, rows in sorted(derivative_rows.items()):
        is_radiomics = pipeline == str(
            radiomics_config.get("pipeline", "imperandi-radiomics")
        )
        filename = "radiomics.tsv" if is_radiomics else "masks.tsv"
        plan.tables[Path("derivatives") / pipeline / filename] = rows
        source_config = (
            radiomics_config
            if is_radiomics
            else next(
                (
                    value
                    for value in derivative_configs.values()
                    if value.get("pipeline") == pipeline
                ),
                {},
            )
        )
        plan.json_files[Path("derivatives") / pipeline / "dataset_description.json"] = (
            _derivative_description(
                pipeline, source_config, "radiomics" if is_radiomics else "segmentation"
            )
        )

    plan.json_files[Path("dataset_description.json")] = _dataset_description(config)
    plan.text_files[Path("code") / "imperandi_mids_export.yaml"] = _safe_manifest(
        config
    )
    plan.text_files[Path("README")] = (
        f"{PROFILE_NAME}\n\n"
        "This dataset uses the hierarchy and naming concepts proposed by MIDS for "
        "curated CT/MR NIfTI volumes. It is not a claim of full MIDS conformance: "
        "the cited proposal does not define a validation target used by this export.\n\n"
        "Participant/session/image identifiers are safe export identifiers. Reverse "
        "identity maps and secret keys are intentionally excluded. Metadata is an "
        "explicit allowlisted projection; the source IMPERANDI CSV is not included.\n"
    )
    if not records:
        plan.missing_fields.setdefault("exportable_images", []).append("dataset")
    destinations: dict[str, list[str]] = defaultdict(list)
    for item in plan.files:
        destinations[item.relative_path.as_posix()].append(item.image_id)
    for destination, image_ids in destinations.items():
        if len(image_ids) > 1:
            plan.collisions.setdefault(destination, image_ids)
    return plan


def run_export(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    config = _export_config(args.manifest)
    frame = _read_cohort(args.csv_path, config)
    automatic_map = None
    automatic_map_report = None
    id_map_path = args.id_map
    if args.id_map == "auto":
        from .id_map import generate_id_map

        automatic_map, automatic_map_report = generate_id_map(
            frame,
            patient_key_mode=args.patient_key_mode,
            sort_columns=args.sort_columns,
            minimum_digits=args.minimum_digits,
        )
        id_map_path = None
    plan = build_plan(
        frame,
        Path(args.output_dir),
        config,
        id_map_path=id_map_path,
        id_map_frame=automatic_map,
        key_file=args.key_file,
    )
    if args.dry_run:
        try:
            validate_plan(plan)
        except MidsExportError as exc:
            plan.validation_errors.append(str(exc))
        if Path(args.output_dir).exists() and args.overwrite == "never":
            plan.validation_errors.append(
                "Output already exists and the overwrite policy is 'never'."
            )
        automatic_map_state = None
        if automatic_map is not None:
            from .id_map import existing_id_map_status

            automatic_map_state = existing_id_map_status(
                automatic_map, Path(args.id_map_output)
            )
            if (
                automatic_map_state in {"conflicting", "invalid"}
                and args.id_map_overwrite == "never"
            ):
                plan.validation_errors.append(
                    "The automatic ID-map output exists and is not identical."
                )
        report = plan.report()
        if automatic_map_report is not None:
            report["automatic_id_map"] = {
                **automatic_map_report,
                "output_path": args.id_map_output,
                "existing_status": automatic_map_state,
                "written": False,
            }
        print(json.dumps(report, indent=2, sort_keys=True))
        return (2 if report["status"] == "invalid" else 0), report
    validate_plan(plan)
    if Path(args.output_dir).exists() and args.overwrite == "never":
        raise MidsExportError(
            f"Output already exists: {args.output_dir}. "
            "Use --overwrite replace to rebuild it safely."
        )
    if automatic_map is not None:
        from .id_map import write_or_reuse_id_map

        map_action = write_or_reuse_id_map(
            automatic_map,
            Path(args.id_map_output),
            overwrite=args.id_map_overwrite,
        )
        logger.info("Automatic external ID map %s: %s", map_action, args.id_map_output)
    report = plan.report()
    write_plan(plan, overwrite=args.overwrite)
    logger.info(
        "MIDS-style export complete: %d images, %d masks, %d excluded rows -> %s",
        report["counts"]["images"],
        report["counts"]["masks"],
        report["counts"]["excluded_rows"],
        args.output_dir,
    )
    return 0, report
