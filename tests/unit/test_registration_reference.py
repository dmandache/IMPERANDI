"""Ordered reference criteria for categorical preferences and numeric extrema."""

import numpy as np
import pandas as pd
import pytest

from imperandi.process.registration.config import RegistrationConfig
from imperandi.process.registration.cohort import prepare_cohort, reference_rank


def ordered_ids(rows, priorities):
    return [
        row["registration_scan_id"]
        for row in sorted(rows, key=lambda row: reference_rank(row, priorities))
    ]


def test_reference_criteria_apply_in_order_and_normalize_categories():
    priorities = RegistrationConfig().reference_priority["MR"]
    rows = [
        dict(registration_scan_id=str(i), **row)
        for i, row in enumerate(
            [
                dict(mri_sequence="T2", phase="PORTAL_VENOUS", PixelSpacingXY=0.1),
                dict(mri_sequence="T1", phase="ARTERIAL", PixelSpacingXY=0.1),
                dict(mri_sequence="T1", phase="PORTAL_VENOUS", PixelSpacingXY=1),
                dict(
                    mri_sequence="T1",
                    phase="PORTAL_VENOUS",
                    PixelSpacingXY=0.8,
                    SliceThickness=5,
                ),
                dict(
                    mri_sequence="T1",
                    phase="PORTAL_VENOUS",
                    PixelSpacingXY=0.8,
                    SliceThickness=2,
                    registration_organ_volume_mm3=1000,
                ),
                dict(
                    mri_sequence=" t1 ",
                    phase=" portal_venous ",
                    PixelSpacingXY="0.8",
                    SliceThickness="2",
                    registration_organ_volume_mm3=2000,
                ),
                dict(mri_sequence="OTHER", phase="PORTAL_VENOUS", PixelSpacingXY=0.01),
            ]
        )
    ]
    expected = ["5", "4", "3", "2", "1", "0", "6"]
    assert ordered_ids(rows, priorities) == expected
    assert ordered_ids(rows[::-1], priorities) == expected


@pytest.mark.parametrize(
    "direction,expected", [("min", ["two", "ten"]), ("max", ["ten", "two"])]
)
@pytest.mark.parametrize(
    "missing",
    [
        None,
        pd.NA,
        np.nan,
        np.inf,
        -np.inf,
        "",
        "bad",
        "nan",
        "inf",
        True,
        np.bool_(False),
    ],
)
def test_numeric_strings_sort_numerically_and_unusable_values_rank_last(
    direction, expected, missing
):
    rows = [
        dict(registration_scan_id="bad", value=missing),
        dict(registration_scan_id="ten", value="10"),
        dict(registration_scan_id="two", value="2"),
    ]
    assert ordered_ids(rows, [{"value": direction}]) == [*expected, "bad"]


def test_unlisted_and_missing_categories_allow_later_criteria_to_break_ties():
    rows = [
        dict(registration_scan_id="missing", SliceThickness=1),
        dict(registration_scan_id="unknown", mri_sequence="OTHER", SliceThickness=2),
        dict(registration_scan_id="preferred", mri_sequence="T1"),
    ]
    assert ordered_ids(rows, [{"mri_sequence": ["T1"]}, {"SliceThickness": "min"}]) == [
        "preferred",
        "missing",
        "unknown",
    ]


def test_volume_criterion_position_controls_its_influence():
    rows = [
        dict(
            registration_scan_id="small",
            mri_sequence="T1",
            registration_organ_volume_mm3=700,
        ),
        dict(
            registration_scan_id="large",
            mri_sequence="T2",
            registration_organ_volume_mm3=1000,
        ),
    ]
    priorities = [
        {"mri_sequence": ["T1", "T2"]},
        {"registration_organ_volume_mm3": "max"},
    ]
    assert ordered_ids(rows, priorities) == ["small", "large"]
    assert ordered_ids(rows, priorities[::-1]) == ["large", "small"]


@pytest.mark.parametrize(
    "priorities", [[], [{"missing_column": "min"}], [{"value": "max"}]]
)
def test_complete_ties_use_stable_scan_ids(priorities):
    rows = [dict(registration_scan_id=name, value=2) for name in ["c", "a", "b"]]
    assert ordered_ids(rows, priorities) == ["a", "b", "c"]
    assert ordered_ids(rows[::-1], priorities) == ["a", "b", "c"]


def test_planning_discards_stale_organ_volume_without_loading_images():
    config = RegistrationConfig(
        reference_priority={
            "MR": [
                {"registration_organ_volume_mm3": "max"},
                {"mri_sequence": ["T1", "T2"]},
            ]
        }
    )
    rows = [
        dict(
            patient_key="p1",
            study_id="v1",
            Modality="MRI",
            mri_sequence=sequence,
            nifti_path=f"/unavailable/{sequence}.nii.gz",
            mask_liver="/unavailable/organ.nii.gz",
            registration_organ_volume_mm3=volume,
        )
        for sequence, volume in [("T1", 1), ("T2", 1000)]
    ]
    planned = prepare_cohort(pd.DataFrame(rows), config)
    assert planned.registration_series_number.tolist() == [1, 2]
    assert planned.registration_organ_volume_mm3.isna().all()
