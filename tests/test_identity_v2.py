import pandas as pd
import pytest

from imperandi.config.models import IdentityConfig
from imperandi.identity import resolve_patient_identities


def test_source_identity_is_namespaced_and_raw_id_is_separated():
    data = pd.DataFrame(
        {"PatientID": [" 001 "], "site_id": ["HEGP"], "SeriesDescription": ["x"]}
    )
    config = IdentityConfig.model_validate(
        {
            "source": {
                "patient_id_columns": ["PatientID"],
                "namespace_columns": ["site_id"],
                "fallback": {"on_missing": "error"},
            },
            "canonical": {"strategy": "source"},
        }
    )
    result = resolve_patient_identities(data, config)

    assert result.cohort.loc[0, "patient_id"] == "site_id=HEGP|001"
    assert "PatientID" not in result.cohort
    assert result.sensitive.loc[0, "dicom_patient_id"] == "001"


def test_crosswalk_then_hmac_uses_crosswalk_first(tmp_path, monkeypatch):
    crosswalk = tmp_path / "identities.csv"
    pd.DataFrame(
        {
            "site_id": ["A"],
            "dicom_patient_id": ["ONE"],
            "patient_id": ["P-CROSSWALK"],
        }
    ).to_csv(crosswalk, index=False)
    monkeypatch.setenv("TEST_ID_SECRET", "not-committed")
    data = pd.DataFrame({"PatientID": ["ONE", "TWO"], "site_id": ["A", "A"]})
    config = IdentityConfig.model_validate(
        {
            "source": {
                "patient_id_columns": ["PatientID"],
                "namespace_columns": ["site_id"],
                "fallback": {"on_missing": "error"},
            },
            "canonical": {
                "strategy": "crosswalk_then_hmac",
                "crosswalk": crosswalk,
                "crosswalk_keys": ["site_id", "dicom_patient_id"],
                "hmac": {
                    "secret_env": "TEST_ID_SECRET",
                    "namespace": "test-v1",
                    "prefix": "P",
                    "length": 16,
                },
            },
        }
    )
    result = resolve_patient_identities(data, config)

    assert result.cohort.loc[0, "patient_id"] == "P-CROSSWALK"
    assert result.cohort.loc[0, "patient_id_method"] == "crosswalk"
    assert result.cohort.loc[1, "patient_id"].startswith("P")
    assert result.cohort.loc[1, "patient_id_method"] == "hmac"


def test_never_policy_does_not_persist_raw_identity_mapping():
    data = pd.DataFrame({"PatientID": ["SECRET-1"]})
    config = IdentityConfig.model_validate(
        {
            "source": {
                "patient_id_columns": ["PatientID"],
                "namespace_columns": [],
                "fallback": {"on_missing": "error"},
            },
            "canonical": {"strategy": "source"},
            "sensitive_fields": {"persist_raw_identifiers": "never"},
        }
    )

    result = resolve_patient_identities(data, config)

    assert "PatientID" not in result.cohort
    assert list(result.sensitive.columns) == ["patient_id"]
    assert "dicom_patient_id" not in result.sensitive


def test_keep_policy_marks_missing_identity_for_safe_exclusion():
    data = pd.DataFrame(
        {
            "PatientID": [None],
            "site_id": ["SITE-A"],
            "SeriesDescription": ["unknown"],
        }
    )
    config = IdentityConfig.model_validate(
        {
            "source": {
                "patient_id_columns": ["PatientID"],
                "namespace_columns": ["site_id"],
                "fallback": {"on_missing": "keep"},
            },
            "canonical": {"strategy": "source"},
        }
    )

    result = resolve_patient_identities(data, config)

    assert pd.isna(result.cohort.loc[0, "patient_id"])
    assert result.qc_flags.loc[0, "qc_code"] == "IDENTITY_MISSING"


def test_preindexed_patient_id_can_be_used_as_source_without_being_dropped():
    data = pd.DataFrame({"patient_id": ["P-001"]})
    config = IdentityConfig.model_validate(
        {
            "source": {
                "patient_id_columns": ["patient_id"],
                "namespace_columns": [],
                "fallback": {"on_missing": "error"},
            },
            "canonical": {"strategy": "source"},
        }
    )

    result = resolve_patient_identities(data, config)

    assert result.cohort.loc[0, "patient_id"] == "P-001"


def test_legacy_parser_identity_alias_does_not_bypass_sensitive_policy():
    data = pd.DataFrame({"PatientID": ["SECRET-1"], "patient_key": ["SECRET-1"]})
    config = IdentityConfig.model_validate(
        {
            "source": {
                "patient_id_columns": ["PatientID"],
                "namespace_columns": [],
                "fallback": {"on_missing": "error"},
            },
            "canonical": {"strategy": "source"},
        }
    )

    result = resolve_patient_identities(data, config)

    assert "PatientID" not in result.cohort
    assert "patient_key" not in result.cohort
    assert result.sensitive.loc[0, "dicom_patient_id"] == "SECRET-1"


def test_cohort_policy_uses_explicit_raw_fields_not_legacy_patient_key():
    data = pd.DataFrame({"PatientID": ["SECRET-1"], "patient_key": ["legacy-secret"]})
    config = IdentityConfig.model_validate(
        {
            "source": {
                "patient_id_columns": ["PatientID"],
                "namespace_columns": [],
            },
            "sensitive_fields": {"persist_raw_identifiers": "cohort"},
        }
    )

    result = resolve_patient_identities(data, config)

    assert "patient_key" not in result.cohort
    assert result.cohort.loc[0, "dicom_patient_id"] == "SECRET-1"
    assert result.cohort.loc[0, "source_patient_key"] == "SECRET-1"


def test_empty_identity_table_preserves_a_stable_output_schema():
    data = pd.DataFrame(columns=["PatientID", "SeriesDescription"])
    config = IdentityConfig.model_validate(
        {
            "source": {
                "patient_id_columns": ["PatientID"],
                "namespace_columns": [],
                "fallback": {"on_missing": "error"},
            }
        }
    )

    result = resolve_patient_identities(data, config)

    assert result.cohort.empty
    assert "patient_id" in result.cohort
    assert list(result.sensitive.columns) == [
        "patient_id",
        "dicom_patient_id",
        "source_patient_key",
        "patient_id_source_column",
    ]
    assert result.qc_flags.empty


def test_multiple_source_ids_can_be_flagged_without_failing(tmp_path):
    crosswalk = tmp_path / "identities.csv"
    pd.DataFrame(
        {
            "dicom_patient_id": ["ONE", "TWO"],
            "patient_id": ["P-SAME", "P-SAME"],
        }
    ).to_csv(crosswalk, index=False)
    config = IdentityConfig.model_validate(
        {
            "source": {
                "patient_id_columns": ["PatientID"],
                "namespace_columns": [],
                "fallback": {"on_missing": "error"},
            },
            "canonical": {
                "strategy": "crosswalk",
                "crosswalk": crosswalk,
                "crosswalk_keys": ["dicom_patient_id"],
            },
            "validation": {"multiple_source_ids_per_patient": "flag"},
        }
    )

    result = resolve_patient_identities(
        pd.DataFrame({"PatientID": ["ONE", "TWO"]}), config
    )

    assert result.qc_flags.loc[0, "qc_code"] == "IDENTITY_MULTIPLE_SOURCE_IDS"
    assert result.qc_flags.loc[0, "severity"] == "warning"


def test_crosswalk_canonical_ids_are_trimmed_strings(tmp_path):
    crosswalk = tmp_path / "identities.csv"
    pd.DataFrame({"dicom_patient_id": ["ONE"], "patient_id": [" 101 "]}).to_csv(
        crosswalk, index=False
    )
    config = IdentityConfig.model_validate(
        {
            "source": {
                "patient_id_columns": ["PatientID"],
                "namespace_columns": [],
            },
            "canonical": {
                "strategy": "crosswalk",
                "crosswalk": crosswalk,
                "crosswalk_keys": ["dicom_patient_id"],
            },
        }
    )

    result = resolve_patient_identities(pd.DataFrame({"PatientID": ["ONE"]}), config)

    assert result.cohort.loc[0, "patient_id"] == "101"
    assert str(result.cohort["patient_id"].dtype) == "string"


def test_crosswalk_rejects_empty_key_values(tmp_path):
    crosswalk = tmp_path / "identities.csv"
    pd.DataFrame({"dicom_patient_id": [None], "patient_id": ["P-001"]}).to_csv(
        crosswalk, index=False
    )
    config = IdentityConfig.model_validate(
        {
            "source": {
                "patient_id_columns": ["PatientID"],
                "namespace_columns": [],
            },
            "canonical": {
                "strategy": "crosswalk",
                "crosswalk": crosswalk,
                "crosswalk_keys": ["dicom_patient_id"],
            },
        }
    )

    with pytest.raises(ValueError, match="empty key values"):
        resolve_patient_identities(
            pd.DataFrame({"PatientID": ["ONE"]}),
            config,
        )
