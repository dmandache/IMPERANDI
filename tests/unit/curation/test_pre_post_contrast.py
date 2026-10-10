"""Mixed contrast context is resolved by acquisition order, not keyword priority."""

import pandas as pd
import pytest

from imperandi.curation import rules
from imperandi.curation.ct.curate import curate_ct
from imperandi.curation.mri.curate import curate_mri

CONFIG = {
    "strategies": [{"type": "rules"}],
    "text_columns": {
        "CT": ["SeriesDescription", "ImageComments", "StudyDescription"],
        "MR": ["SeriesDescription", "ImageComments", "StudyDescription"],
    },
}


def _row(modality, volume, time, description=None, **extra):
    result = {
        "patient_key": "P1",
        "study_id": "S1",
        "date": "2020-01-01",
        "series_id": volume,
        "volume_id": volume,
        "Modality": modality,
        "time": time,
        "SeriesDescription": description
        or ("T1 VIBE" if modality == "MR" else "Abdomen"),
        "StudyDescription": "with and without contrast",
        "ImageType": "ORIGINAL PRIMARY AXIAL",
        "Rows": 512,
        "Columns": 512,
        "n_files": 100,
        "SliceThickness": 3.0,
        "PixelSpacing": "1\\1",
    }
    result.update(extra)
    return result


def _curate(modality, rows):
    function = curate_mri if modality == "MR" else curate_ct
    return function(pd.DataFrame(rows), phase_curation=CONFIG)["curated"].set_index(
        "volume_id"
    )


@pytest.mark.parametrize(
    "text",
    [
        "pre-post",
        "precontrast and postcontrast",
        "with and without contrast",
        "w/wo contrast",
        "w+wo",
        "wo&w",
        "w&w/o",
        "nonenhanced & enhanced",
        "sans et avec injection",
    ],
)
def test_mixed_context_is_not_an_explicit_native_label(text):
    assert rules.has_pre_post_contrast_text(text)
    assert rules.match_phase(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "w/wo",
        "w+wo",
        "wo&w",
        "w&w/o",
        "wo/w",
        "wo+w",
        "W / WO",
        "W, W/O Contrast",
        "with/without",
        "without+with",
        "with and without",
        "without and with",
        "pre/post",
        "pre+post",
        "post/pre",
        "pre and post",
    ],
)
def test_explicit_mixed_patterns_are_independent_of_native_and_post_rules(text):
    assert rules.has_pre_post_contrast_text(text, native=r"(?!)", post=r"(?!)")
    assert rules.match_phase(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "w/o",
        "wo",
        "w",
        "with",
        "without",
        "pre contrast",
        "post contrast",
        "w/woody",
        "shadow/wo",
        "wo/water",
        "T1 DIXON pre_W",
        "T1 DIXON SS IV_W",
        "T1 DIXON wo_W",
        "T1 DIXON wo-W",
    ],
)
def test_explicit_mixed_patterns_require_two_complete_contrast_conditions(text):
    assert not rules.has_pre_post_contrast_text(text)


@pytest.mark.parametrize("modality", ["CT", "MR"])
@pytest.mark.parametrize("mixed_text", ["w/wo", "w+wo", "wo&w", "wo/w", "with/without"])
def test_explicit_mixed_shorthand_infers_only_first_acquisition_as_native(
    modality, mixed_text
):
    out = _curate(
        modality,
        [
            _row(modality, "second", "120100", StudyDescription=mixed_text),
            _row(modality, "first", "120000", StudyDescription=mixed_text),
        ],
    )

    assert out.loc["first", "phase"] == "NATIVE"
    assert out.loc["first", "phase_confidence"] == "inferred"
    assert out.loc["second", "phase"] == "OTHER"
    if modality == "MR":
        assert out.loc["first", "dixon_component"] == "NOT_DIXON"
        assert out.loc["first", "mri_perfusion_source"] == "group_pre_post_order"


@pytest.mark.parametrize(
    "text",
    [
        "pre contrast",
        "pre injection",
        "without contrast",
        "non enhanced",
        "unenhanced",
        "sans iv",
        "w/o contrast",
        "precontrast",
    ],
)
def test_native_phrase_does_not_supply_its_own_postcontrast_evidence(text):
    assert not rules.has_pre_post_contrast_text(text)
    assert rules.match_phase(text)[0] == "NATIVE"


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_only_first_acquisition_is_native_with_explicit_phases_preserved(modality):
    rows = [
        _row(modality, "late", "120200", "T1 portal" if modality == "MR" else "portal"),
        _row(modality, "middle", "120100"),
        _row(modality, "first", "120000"),
    ]
    out = _curate(modality, rows)
    assert out.loc["first", "phase"] == "NATIVE"
    assert out.loc["first", "phase_confidence"] == "inferred"
    assert "StudyDescription=" in out.loc["first", "phase_reason"]
    assert out.loc["middle", "phase"] == "OTHER"
    assert out.loc["late", "phase"] == "PORTAL_VENOUS"
    if modality == "MR":
        assert out.loc["first", "mri_perfusion_source"] == "group_pre_post_order"


@pytest.mark.parametrize("modality", ["CT", "MR"])
@pytest.mark.parametrize(
    "phase_text", ["arterial", "portal", "delayed", "post contrast"]
)
def test_explicit_first_acquisition_blocks_fallback_without_shifting_to_second(
    modality, phase_text
):
    prefix = "T1 " if modality == "MR" else ""
    out = _curate(
        modality,
        [
            _row(modality, "first", "120000", prefix + phase_text),
            _row(modality, "second", "120100"),
        ],
    )
    assert not out.phase.eq("NATIVE").any()


@pytest.mark.parametrize("modality", ["CT", "MR"])
@pytest.mark.parametrize(
    "mixed_text,expected",
    [
        ("pre post portal", "PORTAL_VENOUS"),
        ("w/wo portal", "PORTAL_VENOUS"),
        ("w+wo arterial", "ARTERIAL"),
        ("wo&w delayed", "DELAYED"),
    ],
)
def test_named_phase_in_same_mixed_text_wins(modality, mixed_text, expected):
    prefix = "T1 " if modality == "MR" else ""
    out = _curate(
        modality,
        [
            _row(modality, "first", "120000", prefix + mixed_text),
            _row(modality, "second", "120100"),
        ],
    )
    assert out.loc["first", "phase"] == expected
    assert not out.phase.eq("NATIVE").any()


def test_no_mixed_context_synthesized_across_fields_or_list_entries():
    out = _curate(
        "MR",
        [
            _row("MR", "first", "120000", "T1 pre", StudyDescription="post contrast"),
            _row("MR", "second", "120100", "T1 pre", StudyDescription="post contrast"),
        ],
    )
    assert set(out.mri_perfusion_source) == {"explicit_text"}
    out = _curate(
        "MR",
        [
            _row(
                "MR",
                "first",
                "120000",
                StudyDescription=["pre contrast", "post contrast"],
            ),
            _row(
                "MR",
                "second",
                "120100",
                StudyDescription=["pre contrast", "post contrast"],
            ),
        ],
    )
    assert not out.mri_perfusion_source.eq("group_pre_post_order").any()


def test_native_only_and_post_only_examinations_are_unchanged():
    for context, expected in [
        ("without contrast", "NATIVE"),
        ("with contrast", "OTHER"),
    ]:
        out = _curate(
            "MR",
            [
                _row("MR", "first", "120000", StudyDescription=context),
                _row("MR", "second", "120100", StudyDescription=context),
            ],
        )
        assert set(out.phase) == {expected}


@pytest.mark.parametrize(
    "description",
    [
        "T1 VIBE DIXON SANS IV CAIPI_W",
        "AX T1 DIXON SS IV_W",
        "T1 IDEAL pre gad_W",
        "T1 pre gad",
        "T1 sans gad",
        "T1 DIXON wo_W",
        "T1 DIXON wo-W",
    ],
)
def test_mri_native_agent_and_water_suffix_are_not_mixed_context(description):
    out = _curate(
        "MR",
        [
            _row("MR", "first", "120000", description, StudyDescription=""),
        ],
    )
    assert out.loc["first", "phase"] == "NATIVE"
    assert out.loc["first", "mri_perfusion_source"] == "explicit_text"


def test_study_and_dixon_component_boundaries_are_preserved():
    rows = []
    for study in ["S1", "S2"]:
        for component in ["water", "fat"]:
            for rank in [1, 2]:
                rows.append(
                    _row(
                        "MR",
                        f"{study}-{component}-{rank}",
                        f"120{rank}00",
                        f"T1 Dixon {component}",
                        study_id=study,
                    )
                )
    out = _curate("MR", rows)
    assert set(out.index[out.phase.eq("NATIVE")]) == {
        "S1-water-1",
        "S1-fat-1",
        "S2-water-1",
        "S2-fat-1",
    }


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_time_wins_over_conflicting_rank_and_tied_reconstructions_share_phase(modality):
    out = _curate(
        modality,
        [
            _row(modality, "first-thin", "120000", acquisition_order=3),
            _row(modality, "first-thick", "120000", acquisition_order=4),
            _row(modality, "second", "120100", acquisition_order=0),
        ],
    )
    assert set(out.index[out.phase.eq("NATIVE")]) == {"first-thin", "first-thick"}


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_missing_time_uses_acquisition_rank_and_missing_chronology_stays_unresolved(
    modality,
):
    out = _curate(
        modality,
        [
            _row(modality, "second", None, acquisition_order=1),
            _row(modality, "first", None, acquisition_order=0),
        ],
    )
    assert out.loc["first", "phase"] == "NATIVE"
    assert out.loc["second", "phase"] == "OTHER"
    out = _curate(
        modality,
        [
            _row(modality, "second", None),
            _row(modality, "first", None),
        ],
    )
    assert set(out.phase) == {"OTHER"}


def test_multivolume_mixed_series_uses_volume_order_and_preserves_existing_dynamic_rules():
    rows = [
        _row(
            "MR",
            f"V{rank}",
            None,
            "T1 pre-post",
            series_id="DYNAMIC",
            volume_order_in_series=rank,
            n_volumes_in_series=2,
        )
        for rank in [2, 1]
    ]
    out = _curate("MR", rows)
    assert out.loc["V1", "phase"] == "NATIVE"
    assert out.loc["V2", "phase"] == "OTHER"
    rows = [
        _row(
            "MR",
            f"V{rank}",
            f"120{rank}00",
            "T1 dynamic pre-post",
            series_id="DYNAMIC",
            volume_order_in_series=rank,
            n_volumes_in_series=4,
            acquisition_order=rank - 1,
        )
        for rank in [1, 2, 3, 4]
    ]
    rows[-1]["time"] = "120500"
    out = _curate("MR", rows)
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]


def test_localizer_does_not_become_native_or_displace_first_t1():
    out = _curate(
        "MR",
        [
            _row("MR", "scout", "115900", "localizer"),
            _row("MR", "first", "120000"),
            _row("MR", "second", "120100"),
        ],
    )
    assert out.loc["scout", "phase_status"] == "NOT_APPLICABLE"
    assert out.loc["first", "phase"] == "NATIVE"


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_first_explicit_acquisition_participates_even_without_mixed_text(modality):
    prefix = "T1 " if modality == "MR" else ""
    out = _curate(
        modality,
        [
            _row(modality, "first", "120000", prefix + "portal", StudyDescription=""),
            _row(modality, "second", "120100"),
            _row(modality, "third", "120200"),
        ],
    )
    assert out.loc["first", "phase"] == "PORTAL_VENOUS"
    assert not out.phase.eq("NATIVE").any()


def test_dixon_water_suffix_does_not_block_mixed_group_native_fallback():
    out = _curate(
        "MR",
        [
            _row("MR", "first", "120000", "T1 DIXON_W"),
            _row("MR", "second", "120100", "T1 DIXON_W"),
        ],
    )
    assert out.loc["first", "phase"] == "NATIVE"
    assert out.loc["second", "phase"] == "OTHER"


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_explicit_phase_on_first_reconstruction_blocks_native_for_tied_reconstruction(
    modality,
):
    prefix = "T1 " if modality == "MR" else ""
    out = _curate(
        modality,
        [
            _row(modality, "first-explicit", "120000", prefix + "portal"),
            _row(modality, "first-unlabelled", "120000"),
            _row(modality, "second", "120100"),
        ],
    )
    assert not out.phase.eq("NATIVE").any()


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_derived_product_does_not_displace_first_diagnostic_acquisition(modality):
    prefix = "T1 " if modality == "MR" else ""
    out = _curate(
        modality,
        [
            _row(modality, "mpr", "115900", prefix + "MPR"),
            _row(modality, "first", "120000"),
            _row(modality, "second", "120100"),
        ],
    )
    assert out.loc["mpr", "phase"] == "OTHER"
    assert out.loc["first", "phase"] == "NATIVE"
