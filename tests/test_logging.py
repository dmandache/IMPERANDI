import importlib.util
import logging
from pathlib import Path

_LOGGING_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "imperandi" / "utils" / "logging.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "imperandi_utils_logging", _LOGGING_MODULE_PATH
)
assert _SPEC is not None
assert _SPEC.loader is not None
_LOGGING_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LOGGING_MODULE)
log_task_summary = _LOGGING_MODULE.log_task_summary


def test_log_task_summary_reports_shared_row_counts(caplog):
    logger = logging.getLogger("imperandi.tests.summary")

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_task_summary(
            logger,
            "Phase extraction",
            total_rows=5,
            processed_rows=3,
            succeeded_rows=2,
            skipped_rows=2,
            failed_rows=1,
            success_label="phase extracted",
            extra_counts={
                "skipped by resume": 1,
                "skipped with existing phase": 1,
                "skipped by filters": 0,
            },
        )

    assert (
        "Phase extraction summary: 5 total row(s), 3 processed, "
        "2 phase extracted, 2 skipped, 1 failed, 1 skipped by resume, "
        "1 skipped with existing phase"
    ) in caplog.text
    assert "skipped by filters" not in caplog.text


def test_log_task_summary_defaults_successes_to_processed_minus_failed(caplog):
    logger = logging.getLogger("imperandi.tests.summary.default")

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_task_summary(
            logger,
            "Conversion",
            processed_rows=4,
            skipped_rows=1,
            failed_rows=1,
            success_label="converted",
            skipped_label="skipped already valid",
        )

    assert (
        "Conversion summary: 4 processed, 3 converted, "
        "1 skipped already valid, 1 failed"
    ) in caplog.text
