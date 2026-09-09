"""Shared-framework checkpoints around complete registration groups."""

import argparse
from dataclasses import asdict
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from imperandi.utils.logging import log_task_summary
from imperandi.utils.run_state import (
    CheckpointManager,
    atomic_write_csv,
    ensure_source_id_column,
    fingerprint_inputs,
    prepare_resume_context,
)
from .execution import iter_group_results
from .grouping import prepare_cohort
from .labels import group_label
from .organ import backend
from .qc import ERROR_COLUMNS, build_error_record, build_qc

logger = logging.getLogger(__name__)


def _artifact_paths(rows):
    columns = [
        c
        for c in rows
        if c == "registration_log_path"
        or (c.startswith("reg_") and c.endswith("_path"))
    ]
    return sorted(
        {str(v) for col in columns for v in rows[col].dropna() if str(v).strip()}
    )


def _read_checkpoint(path):
    try:
        return pd.read_csv(path, dtype=str)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def run_registration(args, table, config, manifest):
    """Resume only committed groups with intact artifacts and unchanged inputs."""
    df = ensure_source_id_column(
        prepare_cohort(table, config), preferred_column="registration_scan_id"
    )
    sitk = backend()
    signature = argparse.Namespace(
        settings=asdict(config),
        manifest_config=manifest,
        registration_schema=8,
        backend_version=sitk.Version_VersionString(),
        output_dir=args.output_dir,
        threads_per_worker=args.threads_per_worker,
        timeout_sec=args.timeout_sec,
        resume=args.resume and not args.force,
        strict_resume=args.strict_resume,
        checkpoint_every_rows=args.checkpoint_every_rows,
        checkpoint_every_sec=args.checkpoint_every_sec,
    )
    inputs = {args.csv_path}
    for col in [
        "nifti_path",
        f"source_{config.organ_column}",
        f"source_{config.tumor_column}",
    ]:
        if col in df:
            inputs.update(str(p) for p in df[col].dropna() if str(p).strip())
    context = prepare_resume_context(
        args=signature,
        command="register",
        inputs=sorted(inputs),
        output_path=args.csv_path_out,
        error_path=args.error_csv_path,
    )
    manager = CheckpointManager(paths=context["paths"], config=context["config"])
    state = context["state"] or {}
    completed, artifacts, errors = set(), {}, pd.DataFrame(columns=ERROR_COLUMNS)
    df["registration_qc_path"] = args.qc_csv_path
    groups = dict(tuple(df.groupby("registration_group_id", sort=True)))
    derived = [
        c
        for c in df
        if c != "registration_qc_path"
        and (
            c.startswith("reg_")
            or c.startswith("registration_")
            or c in {"consensus_status", config.organ_column, config.tumor_column}
        )
    ]
    if context["can_resume"]:
        saved = _read_checkpoint(context["paths"].main_checkpoint_path)
        saved_errors = _read_checkpoint(context["paths"].error_checkpoint_path)
        # Missing error data is unsafe when the state says errors were committed.
        errors_valid = not state.get("error_count", 0) or (
            saved_errors is not None
            and len(saved_errors) == state["error_count"]
            and set(ERROR_COLUMNS).issubset(saved_errors.columns)
        )
        if saved is not None and set(derived).issubset(saved) and errors_valid:
            saved = saved.set_index("registration_scan_id", drop=False)
            if saved.index.is_unique:
                for group_id in state.get("completed_indices", []):
                    if group_id not in groups:
                        continue
                    scan_ids = groups[group_id].registration_scan_id.tolist()
                    if not set(scan_ids).issubset(saved.index):
                        continue
                    previous = saved.loc[scan_ids]
                    expected = state.get("artifacts", {}).get(group_id)
                    paths = _artifact_paths(previous)
                    if expected is None or any(not Path(p).is_file() for p in paths):
                        continue
                    if fingerprint_inputs(paths, strict=args.strict_resume) != expected:
                        continue
                    if args.retry_failed and (
                        previous.registration_status.eq("failed").any()
                        or previous.consensus_status.eq("failed").any()
                        or (
                            saved_errors is not None
                            and saved_errors.registration_scan_id.isin(scan_ids).any()
                        )
                    ):
                        continue
                    # Merge only derived fields; native input types and values remain authoritative.
                    df.loc[groups[group_id].index, derived] = previous[
                        derived
                    ].to_numpy()
                    completed.add(group_id)
                    artifacts[group_id] = expected
                    logger.info(
                        "Registration group reused: %s, series=%d",
                        group_label(groups[group_id].iloc[0], config.visit_column),
                        len(groups[group_id]),
                    )
        if saved_errors is not None and set(ERROR_COLUMNS).issubset(saved_errors):
            reused_ids = set(
                df.loc[df.registration_group_id.isin(completed), "registration_scan_id"]
            )
            errors = saved_errors.loc[
                saved_errors.registration_scan_id.isin(reused_ids), ERROR_COLUMNS
            ].copy()

    skipped_rows = int(df.registration_group_id.isin(completed).sum())
    pending = [(key, group) for key, group in groups.items() if key not in completed]
    reused_groups = len(completed)
    logger.info(
        "Registration plan: series=%d, groups=%d, groups_pending=%d, groups_reused=%d",
        len(df),
        len(groups),
        len(pending),
        len(completed),
    )

    def checkpoint(force=False):
        if not manager.should_flush(force=force):
            return
        atomic_write_csv(
            build_qc(df.loc[df.registration_group_id.isin(completed)], errors, config),
            args.qc_csv_path,
            index=False,
        )
        manager.flush(
            main_df=df,
            error_df=errors,
            completed_indices=completed,
            force=force,
            extra_state={"artifacts": artifacts, "error_count": len(errors)},
        )

    final_paths = [args.csv_path_out, args.error_csv_path, args.qc_csv_path]
    if (
        not pending
        and context["already_finished"]
        and all(Path(p).is_file() for p in final_paths)
        and (
            fingerprint_inputs(final_paths, strict=args.strict_resume)
            == state.get("final_outputs")
        )
    ):
        logger.info(
            "Resume enabled and matching registration run already finished; "
            "skipping execution."
        )
        log_task_summary(
            logger,
            "Registration",
            total_rows=len(df),
            processed_rows=0,
            succeeded_rows=0,
            skipped_rows=len(df),
            success_label="registered",
            skipped_label="reused",
            extra_counts={"groups reused": len(completed)},
        )
        logger.info("Registration done ✔")
        return df.drop(columns="_source_idx"), errors

    # Replace incompatible state before any new work. Also covers empty cohorts.
    checkpoint(force=True)
    results = iter_group_results(
        pending,
        args.output_dir,
        config,
        num_workers=args.num_workers,
        timeout_sec=args.timeout_sec,
        start_method=args.start_method,
        threads=args.threads_per_worker,
        heartbeat=checkpoint,
    )
    try:
        with tqdm(
            total=len(df),
            initial=skipped_rows,
            desc="Registration",
            unit="series",
            disable=getattr(args, "quiet", False),
        ) as progress:
            for group_id, result, worker_error in results:
                group = groups[group_id]
                label = group_label(group.iloc[0], config.visit_column)
                if worker_error is not None:
                    logger.error(
                        "Registration group failed: %s, series=%d; "
                        "see the error table for details",
                        label,
                        len(group),
                    )
                    rows = group.copy()
                    rows["registration_status"] = "failed"
                    rows["consensus_status"] = "registration_failed"
                    group_errors = pd.DataFrame(
                        [
                            build_error_record(
                                row,
                                config,
                                stage="worker",
                                error=worker_error,
                            )
                            for _, row in group.iterrows()
                        ],
                        columns=ERROR_COLUMNS,
                    )
                else:
                    rows, group_errors = result
                rows = rows.set_index("registration_scan_id", drop=False).loc[
                    group.registration_scan_id
                ]
                df.loc[group.index, derived] = rows[derived].to_numpy()
                if not group_errors.empty:
                    errors = pd.concat(
                        [errors, group_errors[ERROR_COLUMNS]], ignore_index=True
                    )
                paths = _artifact_paths(rows)
                if any(not Path(p).is_file() for p in paths):
                    raise RuntimeError(
                        "Registration worker returned missing output artifacts"
                    )
                artifacts[group_id] = fingerprint_inputs(
                    paths, strict=args.strict_resume
                )
                completed.add(group_id)
                manager.mark_processed(len(group))
                checkpoint()
                progress.update(len(group))
    finally:
        # Generator.close terminates any outstanding subprocesses on interruption.
        results.close()
        checkpoint(force=True)

    atomic_write_csv(df.drop(columns="_source_idx"), args.csv_path_out, index=False)
    atomic_write_csv(errors, args.error_csv_path, index=False)
    manager.finalize_state(
        completed_indices=completed,
        extra_state={
            "artifacts": artifacts,
            "error_count": len(errors),
            "final_outputs": fingerprint_inputs(final_paths, strict=args.strict_resume),
        },
    )
    failed_ids = set(errors.registration_scan_id)
    pending_ids = {key for key, _ in pending}
    processed_ids = set(
        df.loc[df.registration_group_id.isin(pending_ids), "registration_scan_id"]
    )
    log_task_summary(
        logger,
        "Registration",
        total_rows=len(df),
        processed_rows=len(df) - skipped_rows,
        skipped_rows=skipped_rows,
        failed_rows=len(failed_ids & processed_ids),
        success_label="registered",
        skipped_label="reused",
        extra_counts={
            "groups processed": len(pending),
            "groups reused": reused_groups,
        },
    )
    logger.info("Registration done ✔")
    return df.drop(columns="_source_idx"), errors
