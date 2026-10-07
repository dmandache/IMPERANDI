"""Bounded group workers with hard timeouts and deterministic cleanup."""

import logging
import multiprocessing as mp
from multiprocessing.connection import wait
import time

from imperandi.utils.logging import setup_logging
from .organ import backend
from .pipeline import register_cohort


def _process_group(connection, table, output_dir, config, threads, log_level):
    setup_logging(level=log_level)
    try:
        backend().ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
        connection.send((register_cohort(table, output_dir, config), None))
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
                yield group_id, register_cohort(table, output_dir, config), None
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
