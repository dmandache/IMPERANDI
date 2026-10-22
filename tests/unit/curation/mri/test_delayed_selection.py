"""Delayed selection prefers 5-10 minutes without changing phase classification."""

import pandas as pd
import pytest

from imperandi.curation.mri import mri_curation as mc


def candidates(descriptions):
    return pd.DataFrame(
        [
            {
                "patient_key": "p1",
                "study_id": "exam1",
                "date": "2020-01-01",
                "Modality": "MR",
                "series_id": f"s{i}",
                "volume_id": f"v{i}",
                "SeriesDescription": description,
                "ImageType": "ORIGINAL PRIMARY",
                "n_rows_in_volume": 100,
                "PixelSpacing": "1\\1",
                "SliceThickness": 3,
            }
            for i, description in enumerate(descriptions)
        ]
    )


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (179, "DELAYED"),
        (180, "DELAYED"),
        (300, "DELAYED"),
        (600, "DELAYED"),
        (900, "DELAYED"),
        (901, "DELAYED"),
        (1200, "HEPATOBILIARY"),
        (7200, "HEPATOBILIARY"),
    ],
)
def test_selection_preference_preserves_phase_classification(seconds, expected):
    result = mc.annotate_mri(candidates([f"AX T1 VIBE tardif {seconds}s"]))
    assert result.phase.tolist() == [expected]


@pytest.mark.parametrize("preferred", ["5min", "7min30sec", "10min"])
def test_preferred_interval_beats_higher_image_score(preferred):
    rows = candidates(["AX T1 VIBE delayed 3min", f"COR T1 VIBE delayed {preferred}"])
    result = mc.curate_mri(rows)
    scores = result["curated"].set_index("volume_id").selection_score
    assert scores["v0"] > scores["v1"]
    assert result["selected_long"].volume_id.tolist() == ["v1"]
    assert (
        "preferred delayed interval=5-10min"
        in result["selected_wide"].iloc[0].T1_DELAYED
    )


def test_elapsed_timing_prefers_interval_when_description_has_no_duration():
    rows = mc.annotate_mri(candidates(["AX T1 VIBE delayed"] * 2))
    rows["time_delta"] = [180, 420]
    rows = mc.add_scores(rows)
    selected, _ = mc.select_best_candidates(rows)
    assert selected.volume_id.tolist() == ["v1"]


@pytest.mark.parametrize(
    "descriptions",
    [
        ["AX T1 VIBE delayed 3min", "COR T1 VIBE delayed 15min"],
        ["AX T1 VIBE delayed", "COR T1 VIBE delayed"],
        ["AX T1 VIBE delayed 5min", "COR T1 VIBE delayed 10min"],
    ],
)
def test_normal_quality_ranking_within_same_interval_tier(descriptions):
    result = mc.curate_mri(candidates(descriptions))
    assert result["selected_long"].volume_id.tolist() == ["v0"]


def test_timing_does_not_change_other_phase_ranking():
    rows = candidates(["AX T1 VIBE portal", "COR T1 VIBE portal"])
    rows["time_delta"] = [90, 420]
    result = mc.curate_mri(rows)
    assert result["selected_long"].volume_id.tolist() == ["v0"]


def test_preference_uses_text_timing_before_elapsed_time():
    rows = mc.annotate_mri(
        candidates(["AX T1 VIBE delayed 3min", "COR T1 VIBE delayed 7min"])
    )
    rows["time_delta"] = [420, 180]
    selected, _ = mc.select_best_candidates(mc.add_scores(rows))
    assert selected.volume_id.tolist() == ["v1"]


def test_conflicting_text_durations_do_not_receive_preference():
    row = pd.Series(
        {
            "phase": "DELAYED",
            "SeriesDescription": "T1 VIBE delayed 5min 15min",
            "time_delta": 420,
        }
    )
    assert not mc._delayed_in_preferred_window(row)
