import importlib.util
import io
import logging
import sys
from pathlib import Path

import pytest

_LOGGING_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "imperandi" / "utils" / "logging.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "imperandi_utils_logging", _LOGGING_MODULE_PATH
)
assert _SPEC is not None
assert _SPEC.loader is not None
_LOGGING_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LOGGING_MODULE)
log_task_summary = _LOGGING_MODULE.log_task_summary


@pytest.mark.parametrize(
    "encoding, expected_console",
    [
        (
            "cp1252",
            "Warning \\u26a0\\ufe0f for Jos\u00e9; Cleaning done \\u2714",
        ),
        (
            "ascii",
            "Warning \\u26a0\\ufe0f for Jos\\xe9; Cleaning done \\u2714",
        ),
        ("utf-8", "Warning \u26a0\ufe0f for Jos\u00e9; Cleaning done \u2714"),
    ],
)
@pytest.mark.parametrize("use_environment", [False, True])
def test_setup_logging_handles_unicode_in_console_and_file(
    monkeypatch, tmp_path, capsys, encoding, expected_console, use_environment
):
    root = logging.RootLogger(logging.WARNING)
    output = io.BytesIO()
    console = io.TextIOWrapper(output, encoding=encoding, errors="strict")
    log_path = tmp_path / "pipeline.log"
    try:
        with monkeypatch.context() as patch:
            patch.setattr(logging, "root", root)
            patch.setattr(sys, "stdout", console)
            if use_environment:
                patch.setenv("IMPERANDI_LOG_FILE", str(log_path))
            else:
                patch.delenv("IMPERANDI_LOG_FILE", raising=False)
            _LOGGING_MODULE.setup_logging(
                level="INFO",
                log_file=None if use_environment else str(log_path),
                fmt="%(message)s",
            )
            root.warning(
                "Warning %s for %s; Cleaning done %s",
                "\u26a0\ufe0f",
                "Jos\u00e9",
                "\u2714",
            )

        assert "--- Logging error ---" not in capsys.readouterr().err
        assert output.getvalue().decode(encoding).strip() == expected_console
        assert log_path.read_text(encoding="utf-8").strip() == (
            "Warning \u26a0\ufe0f for Jos\u00e9; Cleaning done \u2714"
        )
        assert console.encoding == encoding
        assert console.errors == "strict"
    finally:
        for handler in root.handlers:
            handler.close()
        console.close()


def test_setup_logging_escapes_unicode_in_tracebacks(monkeypatch, capsys):
    root = logging.RootLogger(logging.WARNING)
    output = io.BytesIO()
    console = io.TextIOWrapper(output, encoding="cp1252", errors="strict")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(logging, "root", root)
            patch.setattr(sys, "stdout", console)
            patch.delenv("IMPERANDI_LOG_FILE", raising=False)
            _LOGGING_MODULE.setup_logging(level="INFO", fmt="%(message)s")
            try:
                raise ValueError("Invalid phase \u2714")
            except ValueError:
                root.exception("Phase extraction failed")

        assert "--- Logging error ---" not in capsys.readouterr().err
        text = output.getvalue().decode("cp1252")
        assert "Phase extraction failed" in text
        assert "Traceback (most recent call last):" in text
        assert "ValueError: Invalid phase \\u2714" in text
    finally:
        for handler in root.handlers:
            handler.close()
        console.close()


def test_setup_logging_supports_streams_without_encoding(monkeypatch, capsys):
    root = logging.RootLogger(logging.WARNING)
    console = io.StringIO()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(logging, "root", root)
            patch.setattr(sys, "stdout", console)
            patch.delenv("IMPERANDI_LOG_FILE", raising=False)
            _LOGGING_MODULE.setup_logging(level="INFO", fmt="%(message)s")
            root.info("Cleaning done \u2714")

        assert "--- Logging error ---" not in capsys.readouterr().err
        assert console.getvalue().strip() == "Cleaning done \u2714"
    finally:
        for handler in root.handlers:
            handler.close()
        console.close()


def test_log_task_summary_reports_shared_row_counts(caplog):
    logger = logging.getLogger("imperandi.tests.summary")

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        log_task_summary(
            logger,
            "Phase extraction",
            total_rows=5,
            processed_rows=3,
            succeeded_rows=2,
            skipped_rows=2,
            resumed_rows=1,
            failed_rows=1,
            success_label="phase extracted",
            extra_counts={
                "skipped with existing phase": 1,
                "skipped by filters": 0,
            },
        )

    assert (
        "Phase extraction summary: 5 total, "
        "2 phase extracted, 1 resumed, 1 skipped, 1 failed"
    ) in caplog.text
    assert "Phase extraction details: 3 processed, 1 skipped with existing phase" in caplog.text
    assert "skipped by filters" not in caplog.text


def test_log_task_summary_reports_zero_resumed_rows(caplog):
    logger = logging.getLogger("imperandi.tests.summary.zero_resume")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_task_summary(logger, "Conversion", processed_rows=2, resumed_rows=0)

    assert "0 resumed" in caplog.text


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
        "Conversion summary: 3 converted, "
        "1 skipped already valid, 1 failed"
    ) in caplog.text
