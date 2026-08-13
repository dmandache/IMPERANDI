"""Tests for phase-curation rules."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from imperandi.curation.ct.curate import curate_ct
from imperandi.curation.phase import (
    apply_phase_curation,
    phase_needs_strategy,
    validate_phase_curation,
)

PHASE_CONFIG = {
    "strategies": [
        {
            "type": "ontology",
            "name": "site_ontology",
            "columns": ["site_phase"],
            "mapping": {"portal-site": "PORTAL_VENOUS"},
        },
        {"type": "rules", "name": "metadata_rules"},
        {
            "type": "totalsegmentator",
            "name": "totalseg_prediction",
            "column": "totalseg_phase",
            "mapping": {"native": "NATIVE", "portal": "PORTAL_VENOUS"},
        },
    ],
    "fallback": "OTHER",
}


def test_phase_curation_uses_ordered_fallbacks_and_provenance():
    df = pd.DataFrame(
        [
            {
                "site_phase": "portal-site",
                "rule_phase": "ARTERIAL",
                "totalseg_phase": "native",
            },
            {"site_phase": "unmapped", "rule_phase": "ARTERIAL"},
            {"rule_phase": "OTHER", "totalseg_phase": "portal"},
            {"rule_phase": "UNKNOWN"},
        ]
    )

    out = apply_phase_curation(df, PHASE_CONFIG)

    assert out["phase"].tolist() == [
        "PORTAL_VENOUS",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "OTHER",
    ]
    assert out["phase_source"].tolist() == [
        "site_ontology",
        "metadata_rules",
        "totalseg_prediction",
        "fallback",
    ]


def test_phase_curation_logs_resolution_counts_for_each_strategy_and_fallback():
    df = pd.DataFrame(
        [
            {"site_phase": "portal-site", "rule_phase": "ARTERIAL"},
            {"site_phase": "unmapped", "rule_phase": "ARTERIAL"},
            {"rule_phase": "OTHER", "totalseg_phase": "portal"},
            {"rule_phase": "UNKNOWN"},
        ]
    )
    progress_logger = MagicMock()

    apply_phase_curation(df, PHASE_CONFIG, progress_logger=progress_logger)

    assert [call.args[5:] for call in progress_logger.info.call_args_list[:3]] == [
        (1, 3),
        (1, 2),
        (1, 1),
    ]
    assert progress_logger.info.call_args_list[3].args[1:] == ("OTHER", 1, 0)


def test_phase_curation_order_can_prefer_totalsegmentator():
    config = {
        "strategies": [
            {"type": "totalsegmentator"},
            {"type": "rules"},
        ],
        "fallback": None,
    }
    df = pd.DataFrame([{"rule_phase": "ARTERIAL", "totalseg_phase": "portal_venous"}])

    out = apply_phase_curation(df, config)

    assert out.loc[0, "phase"] == "PORTAL_VENOUS"
    assert out.loc[0, "phase_source"] == "totalsegmentator"


def test_totalsegmentator_defaults_to_ct_and_mri_falls_back_to_rules():
    config = {
        "strategies": [
            {"type": "totalsegmentator"},
            {"type": "rules"},
        ]
    }
    df = pd.DataFrame(
        [
            {
                "Modality": "MR",
                "rule_phase": "ARTERIAL",
                "totalseg_phase": "portal_venous",
            },
            {
                "Modality": "CT",
                "rule_phase": "ARTERIAL",
                "totalseg_phase": "portal_venous",
            },
        ]
    )

    needs_prediction = phase_needs_strategy(df, config, "totalsegmentator")
    out = apply_phase_curation(df, config)

    assert needs_prediction.tolist() == [False, True]
    assert out["phase"].tolist() == ["ARTERIAL", "PORTAL_VENOUS"]
    assert out["phase_source"].tolist() == ["rules", "totalsegmentator"]


def test_existing_phase_can_be_an_explicit_ontology_source():
    config = {
        "strategies": [
            {
                "type": "ontology",
                "columns": ["phase"],
                "mapping": {"portal": "PORTAL_VENOUS"},
            }
        ]
    }

    out = apply_phase_curation(pd.DataFrame([{"phase": "portal"}]), config)

    assert out.loc[0, "phase"] == "PORTAL_VENOUS"
    assert out.loc[0, "phase_source"] == "ontology"


def test_phase_curation_preserves_duplicate_indices():
    df = pd.DataFrame(
        [{"rule_phase": "ARTERIAL"}, {"rule_phase": "OTHER"}],
        index=[7, 7],
    )

    out = apply_phase_curation(df, {"strategies": [{"type": "rules"}]})

    assert out.index.tolist() == [7, 7]
    assert out["phase"].tolist() == ["ARTERIAL", "OTHER"]


def test_phase_needs_totalsegmentator_only_after_prior_strategies_fail():
    df = pd.DataFrame(
        [
            {"site_phase": "portal-site", "rule_phase": "OTHER"},
            {"site_phase": "unmapped", "rule_phase": "ARTERIAL"},
            {"site_phase": "unmapped", "rule_phase": "OTHER"},
        ]
    )

    needs_prediction = phase_needs_strategy(df, PHASE_CONFIG, "totalsegmentator")

    assert needs_prediction.tolist() == [False, False, True]


def test_ct_selection_uses_resolved_ontology_phase():
    df = pd.DataFrame(
        [
            {
                "patient_key": "p1",
                "study_id": "s1",
                "series_id": "series",
                "volume_id": "volume",
                "date": "2020-01-01",
                "Modality": "CT",
                "SeriesDescription": "unclassified acquisition",
                "site_phase": "portal-site",
                "ImageType": "ORIGINAL PRIMARY AXIAL",
                "Rows": 512,
                "Columns": 512,
                "SliceThickness": 2.0,
                "n_files": 120,
            }
        ]
    )

    results = curate_ct(df, phase_curation=PHASE_CONFIG)

    assert results["curated"].loc[0, "rule_phase"] == "OTHER"
    assert results["curated"].loc[0, "phase"] == "PORTAL_VENOUS"
    assert results["selected_long"].loc[0, "selection_slot"] == "CT_PORTAL_VENOUS"


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"strategies": [{"type": "ontology", "columns": ["phase"]}]},
        {"strategies": [{"type": "unsupported"}]},
        {"strategies": [{"type": "rules"}, {"type": "rules"}]},
        {"strategies": [{"type": "totalsegmentator", "modalities": []}]},
    ],
)
def test_phase_curation_validation_rejects_invalid_configs(config):
    with pytest.raises(ValueError):
        validate_phase_curation(config)
