"""Manifest overrides for metadata text fields and their precedence."""

import copy
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pandas as pd
import pytest

from imperandi.curation import curate_by_modality
from imperandi.curation.ct import curate as ct
from imperandi.curation.mri import curate as mr
from imperandi.curation.phase import validate_phase_curation


def dataset(modality):
    return pd.DataFrame(
        [
            {
                "patient_key": "p1",
                "study_id": "s1",
                "series_id": "series1",
                "volume_id": "volume1",
                "date": "2020-01-01",
                "Modality": modality,
                "SeriesDescription": "AX T1 arterial",
                "ProtocolName": "AX T1 portal venous",
                "CustomText": "AX T1 native",
                "ImageType": "ORIGINAL PRIMARY",
                "Rows": 512,
                "Columns": 512,
                "n_files": 120,
                "SliceThickness": 2,
            }
        ]
    )


def config(columns):
    return {"strategies": [{"type": "rules"}], "text_columns": columns}


@pytest.mark.parametrize(
    "modality,curate", [("CT", ct.curate_ct), ("MR", mr.curate_mri)]
)
def test_override_replaces_defaults_and_respects_column_order(modality, curate):
    df = dataset(modality)
    original = df.copy(deep=True)
    options = config({modality: ["MissingTag", "ProtocolName", "SeriesDescription"]})
    original_options = copy.deepcopy(options)

    out = curate(df, phase_curation=options)
    assert out["curated"].iloc[0]["phase"] == "PORTAL_VENOUS"
    assert len(out["selected_long"]) == 1
    assert "portal venous" in out["selected_long"].iloc[0]["selected_candidate"].lower()
    assert (
        curate(df, phase_curation=config({modality: ["CustomText"]}))["curated"].iloc[
            0
        ]["phase"]
        == "NATIVE"
    )
    assert curate(df)["curated"].iloc[0]["phase"] == "ARTERIAL"
    pd.testing.assert_frame_equal(df, original)
    assert options == original_options


def test_router_uses_independent_modality_overrides_and_mri_alias():
    df = pd.concat([dataset("CT"), dataset("MR")], ignore_index=True)
    out = curate_by_modality(
        df, phase_curation=config({"CT": ["CustomText"], "MRI": ["ProtocolName"]})
    )
    assert out["ct"]["curated"].iloc[0]["phase"] == "NATIVE"
    assert out["mri"]["curated"].iloc[0]["phase"] == "PORTAL_VENOUS"


def test_omitted_modality_keeps_defaults():
    out = mr.curate_mri(dataset("MR"), phase_curation=config({"CT": ["ProtocolName"]}))
    assert out["curated"].iloc[0]["phase"] == "ARTERIAL"


def test_mr_custom_fields_classify_sequence_and_phase():
    df = dataset("MR")
    df["SeriesDescription"] = "T2 localizer"
    out = mr.annotate_mri(df, phase_curation=config({"MR": ["CustomText"]}))
    assert out.iloc[0]["mri_sequence"] == "T1"
    assert out.iloc[0]["phase"] == "NATIVE"


def test_ct_override_controls_bolus_monitoring():
    df = dataset("CT")
    df["SeriesDescription"] = "bolus tracking"
    out = ct.curate_ct(df, phase_curation=config({"CT": ["CustomText"]}))
    assert out["curated"].iloc[0]["phase"] == "NATIVE"
    df["CustomText"] = "bolus tracking"
    out = ct.curate_ct(df, phase_curation=config({"CT": ["CustomText"]}))
    assert out["curated"].iloc[0]["phase_status"] == "NOT_APPLICABLE"


def test_text_column_scope_is_reset_after_failure(monkeypatch):
    def fail(_):
        raise RuntimeError("curation failed")

    with monkeypatch.context() as patch:
        patch.setattr(ct, "detect_ct_phase", fail)
        with pytest.raises(RuntimeError, match="curation failed"):
            ct.curate_ct(dataset("CT"), phase_curation=config({"CT": ["CustomText"]}))
    assert ct.curate_ct(dataset("CT"))["curated"].iloc[0]["phase"] == "ARTERIAL"


def test_concurrent_curation_calls_keep_their_own_columns(monkeypatch):
    barrier = Barrier(2)
    detect_phase = ct.detect_ct_phase

    def synchronized_detect_phase(row):
        barrier.wait(timeout=10)
        return detect_phase(row)

    monkeypatch.setattr(ct, "detect_ct_phase", synchronized_detect_phase)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                ct.curate_ct,
                dataset("CT"),
                phase_curation=config({"CT": [column]}),
            )
            for column in ["CustomText", "ProtocolName"]
        ]
        phases = [future.result()["curated"].iloc[0]["phase"] for future in futures]
    assert phases == ["NATIVE", "PORTAL_VENOUS"]


@pytest.mark.parametrize(
    "columns",
    [
        None,
        [],
        {"US": ["SeriesDescription"]},
        {"CT": []},
        {"CT": "SeriesDescription"},
        {"CT": [""]},
        {"CT": [1]},
        {"CT": ["ProtocolName", " ProtocolName "]},
        {"MR": ["ProtocolName"], "MRI": ["SeriesDescription"]},
    ],
)
def test_invalid_text_columns_are_rejected(columns):
    with pytest.raises(ValueError, match="text_columns"):
        validate_phase_curation(config(columns))
