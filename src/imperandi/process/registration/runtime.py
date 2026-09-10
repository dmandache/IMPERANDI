"""Shared-framework checkpoints around complete registration groups."""

import argparse
from dataclasses import asdict
import logging
import multiprocessing as mp
from multiprocessing.connection import wait
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from imperandi.utils.logging import log_task_summary, setup_logging
from imperandi.utils.run_state import (
    CheckpointManager,
    atomic_write_csv,
    ensure_source_id_column,
    fingerprint_inputs,
    prepare_resume_context,
)
from .cohort import prepare_cohort, register_cohort
from .alignment import backend
from .reporting import ERROR_COLUMNS, build_error_record, build_qc, group_label

logger = logging.getLogger(__name__)

# Increment when registration behavior changes so old artifacts are not reused.
REGISTRATION_SCHEMA = 11


def _run_group(table, output_dir, config):
    """Apply the same group failure policy in-process and in worker processes."""
    try:
        return register_cohort(table, output_dir, config), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _process_group(connection, table, output_dir, config, threads, log_level):
    setup_logging(level=log_level)
    try:
        backend().ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
        connection.send(_run_group(table, output_dir, config))
    except Exception as exc:
        connection.send((None, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _stop(process):
    if process.is_alive():
        process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join()
    process.close()


def iter_group_results(
    groups,
    output_dir,
    config,
    *,
    num_workers=1,
    timeout_sec=900,
    start_method="spawn",
    threads=1,
    heartbeat=lambda: None,
):
    """Yield (group_id, (cohort, errors) or None, worker_error or None).

    Each subprocess owns a single group. Killing a timed-out worker cannot kill
    unrelated work or publish a completed checkpoint for its partial artifacts.
    One worker with timeout zero runs in-process for library/debugging use.
    """
    if num_workers == 1 and timeout_sec == 0:
        sitk = backend()
        previous = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
        try:
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
            for group_id, table in groups:
                heartbeat()
                result, error = _run_group(table, output_dir, config)
                yield group_id, result, error
        finally:
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(previous)
        return

    context = mp.get_context(start_method)
    pending = iter(groups)
    active = {}
    exhausted = False
    try:
        while active or not exhausted:
            while not exhausted and len(active) < num_workers:
                try:
                    group_id, table = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(
                    target=_process_group,
                    args=(
                        sender,
                        table,
                        str(output_dir),
                        config,
                        threads,
                        logging.getLogger().getEffectiveLevel(),
                    ),
                )
                try:
                    process.start()
                except BaseException:
                    receiver.close()
                    sender.close()
                    process.close()
                    raise
                sender.close()
                active[receiver] = (group_id, process, time.monotonic())
            if not active:
                break
            ready = wait(list(active), timeout=0.1)
            for receiver in list(active):
                group_id, process, started = active[receiver]
                result, error = None, None
                timed_out = (
                    timeout_sec > 0 and time.monotonic() - started >= timeout_sec
                )
                if receiver in ready:
                    try:
                        result, error = receiver.recv()
                    except (EOFError, OSError):
                        error = f"Worker exited without a result (exit code {process.exitcode})"
                elif timed_out:
                    error = f"Group timeout after {timeout_sec}s"
                elif not process.is_alive():
                    error = (
                        f"Worker exited without a result (exit code {process.exitcode})"
                    )
                else:
                    continue
                del active[receiver]
                receiver.close()
                _stop(process)
                yield group_id, result, error
            heartbeat()
    finally:
        for receiver, (_, process, _) in active.items():
            receiver.close()
            _stop(process)


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
        registration_schema=REGISTRATION_SCHEMA,
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
            or c
            in {
                "consensus_status",
                "tumor_consensus_input_status",
                config.organ_column,
                config.tumor_column,
            }
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
                        previous.registration_status.isin(
                            ["failed", "low_confidence", "invalid_organ_mask"]
                        ).any()
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
    failed_ids = set(errors.registration_scan_id) | set(
        df.loc[
            df.registration_status.isin(
                ["failed", "low_confidence", "invalid_organ_mask"]
            ),
            "registration_scan_id",
        ]
    )
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
