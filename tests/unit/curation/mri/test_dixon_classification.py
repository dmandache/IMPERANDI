"""Tests for Dixon classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imperandi.curation.mri import curate as mc
from imperandi.ingest.clean import compute_acquisition_order


@pytest.mark.parametrize(
    "image_type,expected",
    [
        (["DERIVED", "PRIMARY", "W", "W", "DERIVED"], "WATER"),
        (["DERIVED", "PRIMARY", "IP", "IP", "DERIVED"], "IN_PHASE"),
        (["DERIVED", "PRIMARY", "OP", "OP", "DERIVED"], "OPPOSED_PHASE"),
        (["DERIVED", "PRIMARY", "F", "F", "DERIVED"], "FAT"),
        ("['DERIVED', 'PRIMARY', 'W', 'W']", "WATER"),
        (r"DERIVED\PRIMARY\IP\IP", "IN_PHASE"),
        ("DERIVED, PRIMARY, OP, OP", "OPPOSED_PHASE"),
        ("DERIVED; PRIMARY; OUTOFPhase", "OPPOSED_PHASE"),
        ("DERIVED PRIMARY F F", "FAT"),
        (["DERIVED", "PRIMARY", "FF"], "FAT_FRACTION"),
        (["DERIVED", "PRIMARY", "R2S"], "R2STAR"),
        (("DERIVED", "PRIMARY", "F"), "FAT"),
        (np.array(["DERIVED", "PRIMARY", "W"]), "WATER"),
        (None, None),
    ],
)
def test_detect_dixon_component_from_structured_image_type(image_type, expected):
    assert mc.detect_dixon_component_from_image_type(image_type) == expected


def _classify(series_description: str, image_type: object = None) -> pd.Series:
    result = mc.add_basic_feature_columns(
        pd.DataFrame(
            [
                {
                    "SeriesDescription": series_description,
                    "ImageType": image_type,
                }
            ]
        )
    )
    return result.iloc[0]


@pytest.mark.parametrize(
    "series_description,image_type,expected,source",
    [
        ("mDIXON acquisition", ["DERIVED", "PRIMARY", "W"], "WATER", "image_type"),
        (
            "mDIXON water fat reconstruction",
            ["DERIVED", "PRIMARY", "IP"],
            "IN_PHASE",
            "image_type",
        ),
        (
            "mDIXON PDFF map",
            ["DERIVED", "PRIMARY", "F"],
            "FAT_FRACTION",
            "explicit_text",
        ),
        ("mDIXON R2* map", ["DERIVED", "PRIMARY", "W"], "R2STAR", "explicit_text"),
        (
            "mDIXON acquisition",
            ["DERIVED", "PRIMARY", "W", "F"],
            "DIXON_UNKNOWN",
            "image_type",
        ),
        ("DW acquisition", ["DERIVED", "PRIMARY", "DW"], "NOT_DIXON", "none"),
        ("FIESTA acquisition", ["ORIGINAL", "PRIMARY"], "NOT_DIXON", "none"),
        ("generic mDIXON acquisition", None, "DIXON_UNKNOWN", "dixon_context"),
        ("T1 spin echo acquisition", None, "NOT_DIXON", "none"),
    ],
)
def test_dixon_classification_precedence(
    series_description: str,
    image_type,
    expected: str,
    source: str,
):
    classified = _classify(series_description, image_type)

    assert classified["dixon_component"] == expected
    assert classified["dixon_component_source"] == source
    assert classified["dixon_component_reason"]


@pytest.mark.parametrize(
    "series_description,expected",
    [
        ("T1 VIBE DIXON SANS IV CAIPI_W", "WATER"),
        ("T1 VIBE DIXON ART-PORT CAIPI_W", "WATER"),
        ("T1 weighted IDEAL pre gad_W", "WATER"),
        ("mDIXON-All_BH", "DIXON_ALL"),
        ("mDIXON-Quant_BH", "FAT_FRACTION"),
        ("4D mDIXON-W", "WATER"),
    ],
)
def test_dixon_free_text_compatibility(series_description: str, expected: str):
    assert _classify(series_description)["dixon_component"] == expected


def test_dixon_components_are_independent_from_acquisition_order():
    rows = []
    for component, image_type, offset in [
        ("IP", ["DERIVED", "PRIMARY", "IP", "IP", "DERIVED"], 0),
        ("W", ["DERIVED", "PRIMARY", "W", "W", "DERIVED"], 400),
    ]:
        for phase in range(4):
            first_instance = offset + phase + 1
            rows.append(
                {
                    "patient_key": "p1",
                    "study_id": "s1",
                    "volume_id": f"{component}_{phase + 1}",
                    "SeriesDescription": "4D mDIXON dynamic",
                    "ImageType": image_type,
                    "InstanceNumber": list(
                        range(first_instance, first_instance + 16, 4)
                    ),
                }
            )

    ordered = compute_acquisition_order(pd.DataFrame(rows))
    classified = mc.add_basic_feature_columns(ordered)
    ip_rows = classified[classified["volume_id"].str.startswith("IP_")]
    water_rows = classified[classified["volume_id"].str.startswith("W_")]

    assert set(ip_rows["dixon_component"]) == {"IN_PHASE"}
    assert set(water_rows["dixon_component"]) == {"WATER"}
    assert ip_rows["acquisition_order"].tolist() == [0, 1, 2, 3]
    assert water_rows["acquisition_order"].tolist() == [4, 5, 6, 7]
    assert "volume_index_in_series" not in classified.columns
    assert "volume_order_in_series" not in classified.columns

    phase_input = mc.add_mri_sequence_columns(classified)
    phased = mc.add_mri_perfusion_columns(
        phase_input,
        exam_group_cols=["patient_key", "study_id"],
    )
    ip_phases = phased[phased["volume_id"].str.startswith("IP_")]
    water_phases = phased[phased["volume_id"].str.startswith("W_")]
    expected_phases = ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]

    assert ip_phases["mri_perfusion_label"].tolist() == expected_phases
    assert water_phases["mri_perfusion_label"].tolist() == expected_phases
    assert set(phased["mri_perfusion_source"]) == {"acquisition_order_dixon_component"}
