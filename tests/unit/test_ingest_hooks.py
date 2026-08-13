import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dataset_configs.hooks import operandi
from imperandi.ingest.hooks import (
    apply_derived_columns,
    apply_id_standardization,
    apply_patient_key_standardization,
    clean_hook,
    get_clean_hook_outputs,
)


def test_clean_hook_declares_outputs_without_wrapping_function():
    def original(value):
        return value

    decorated = clean_hook(outputs=["center", "source"])(original)

    assert decorated is original
    assert get_clean_hook_outputs(decorated) == ["center", "source"]
    assert get_clean_hook_outputs(lambda: None) == []


def test_apply_patient_key_standardization_requires_patient_key_column():
    df = pd.DataFrame({"study_id": ["S1"]})

    assert apply_patient_key_standardization(df, str.upper) is df


def test_apply_patient_key_standardization_preserves_raw_values_without_hook():
    df = pd.DataFrame(
        {
            "patient_key": ["P1"],
            "_patient_key_raw": ["original"],
        }
    )

    result = apply_patient_key_standardization(df, None)

    assert result is not df
    assert result.to_dict("records") == [
        {"patient_key": "P1", "_patient_key_raw": "original"}
    ]


def test_apply_patient_key_standardization_marks_and_logs_real_failures(caplog):
    df = pd.DataFrame({"patient_key": ["ok", "bad", "", None]})

    def standardize(value):
        if value == "bad":
            return None
        if isinstance(value, str):
            return value.upper()
        return value

    logger = logging.getLogger("imperandi.tests.hooks")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        result = apply_patient_key_standardization(
            df,
            standardize,
            logger=logger,
            log_prefix="custom_hook",
        )

    assert result.loc[0, "patient_key"] == "OK"
    assert pd.isna(result.loc[1, "patient_key"])
    assert result["patient_key_std_failed"].tolist() == [False, True, False, False]
    assert result["_patient_key_raw"].tolist()[:3] == ["ok", "bad", ""]
    assert "[custom_hook] failed on unique raw keys=1" in caplog.text


def test_apply_id_standardization_resolves_manifest_hook():
    seen = []

    def resolver(config):
        seen.append(config)
        return lambda value: value.strip().upper()

    result = apply_id_standardization(
        pd.DataFrame({"patient_key": [" p1 "]}),
        {"id_standardization": {"function": "example:normalize"}},
        hook_resolver=resolver,
    )

    assert seen == [{"function": "example:normalize"}]
    assert result.loc[0, "patient_key"] == "P1"


def test_apply_id_standardization_passes_empty_config_to_resolver():
    seen = []

    result = apply_id_standardization(
        pd.DataFrame({"patient_key": ["P1"]}),
        {},
        hook_resolver=lambda config: seen.append(config),
    )

    assert seen == [{}]
    assert result.loc[0, "_patient_key_raw"] == "P1"


def test_apply_derived_columns_supports_missing_only_and_overwrite():
    df = pd.DataFrame(
        {
            "patient_key": ["P1", "P2"],
            "center": ["curated", "curated"],
        }
    )
    hooks = {
        "derive_missing": lambda value: pd.Series(
            {"center": f"derived-{value}", "source": f"source-{value}"}
        ),
        "derive_overwrite": lambda value: pd.Series({"center": f"overwritten-{value}"}),
    }
    manifest = {
        "derived_columns": [
            {
                "from_column": "patient_key",
                "function": "derive_missing",
            },
            {
                "from_column": "patient_key",
                "function": "derive_overwrite",
                "join_mode": "overwrite",
            },
        ]
    }

    result = apply_derived_columns(
        df,
        manifest,
        hook_resolver=lambda config: hooks[config["function"]],
    )

    assert result["center"].tolist() == ["overwritten-P1", "overwritten-P2"]
    assert result["source"].tolist() == ["source-P1", "source-P2"]


def test_apply_derived_columns_skips_unusable_definitions():
    df = pd.DataFrame({"patient_key": ["P1"]})
    manifest = {
        "derived_columns": [
            {},
            {"from_column": "missing", "function": "unused"},
            {"from_column": "patient_key", "function": "unresolved"},
        ]
    }

    result = apply_derived_columns(df, manifest, hook_resolver=lambda _config: None)

    assert result is df


def test_apply_derived_columns_handles_empty_input_and_manifest():
    df = pd.DataFrame({"patient_key": pd.Series(dtype=str)})
    manifest = {
        "derived_columns": [
            {"from_column": "patient_key", "function": "example:derive"}
        ]
    }

    empty_result = apply_derived_columns(
        df,
        manifest,
        hook_resolver=lambda _config: lambda _value: pd.Series({"center": "BJN"}),
    )

    assert empty_result.empty
    assert apply_derived_columns(df, {}) is df


def test_operandi_patient_key_validation_and_standardization():
    assert operandi.check_operandi_patient_key("001_01-02-0007-02")
    assert operandi.standardize_operandi_patient_key("001_01-02-0007-02") == "1-2-7-2"


def test_operandi_patient_key_validation_rejects_unknown_codes():
    for patient_key in ["99-2-7-2", "1-99-7-2", "1-2-7-99"]:
        with pytest.raises(AssertionError):
            operandi.check_operandi_patient_key(patient_key)


def test_operandi_patient_key_falls_back_to_legacy_format():
    assert operandi.transform_operandi_patient_key("legacy patient 4 99") == "4-2-99-2"
    assert (
        operandi.standardize_operandi_patient_key("legacy patient 4 99") == "4-2-99-2"
    )
    assert operandi.transform_operandi_patient_key("invalid") is None
    assert operandi.standardize_operandi_patient_key("invalid") is None


def test_operandi_derived_columns_use_standardized_key():
    result = operandi.extract_from_patient_key("001_01-02-0007-02")

    assert result.to_dict() == {
        "center": "BJN",
        "source": "CIRSE",
        "tumor_type": "CHC",
    }
    assert get_clean_hook_outputs(operandi.standardize_operandi_patient_key) == [
        "patient_key"
    ]
    assert get_clean_hook_outputs(operandi.extract_from_patient_key) == [
        "center",
        "source",
        "tumor_type",
    ]
