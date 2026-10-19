"""Generic dynamic chronology must retain explicit phase anchors."""

import pandas as pd
import pytest

from imperandi.curation.mri.mri_curation import curate_mri


def volume(name, description, time, **kwargs):
    return {
        "patient_key": "P1",
        "study_id": "S1",
        "date": "2020-01-01",
        "series_id": "DYNAMIC",
        "volume_id": name,
        "Modality": "MR",
        "SeriesDescription": "AX T1 LAVA " + description,
        "time": time,
        "time_source": "AcquisitionTime",
        "ImageType": "ORIGINAL PRIMARY",
        "Rows": 512,
        "Columns": 512,
        "n_files": 100,
        "SliceThickness": 3,
        **kwargs,
    }


def curate(rows, **config):
    return curate_mri(
        pd.DataFrame(rows),
        phase_curation={
            "strategies": [{"type": "rules"}],
            "text_columns": {
                "MR": ["SeriesDescription", "ProtocolName", "StudyDescription"]
            },
            **config,
        },
    )["curated"].set_index("volume_id")


def anchored_rows(description="dynamic", same_series=False):
    # After the initial 20-second assumption, later clocks exercise order fallback.
    return [
        volume("pre", "pre", "115959", series_id="DYNAMIC" if same_series else "PRE"),
        volume("d1", description, "120000"),
        volume("d2", description, "120016"),
        volume("d3", description, "120111"),
    ]


@pytest.mark.parametrize("same_series", [False, True])
@pytest.mark.parametrize("description", ["dynamic", "dynamic post"])
def test_explicit_native_anchors_same_or_separate_series(same_series, description):
    out = curate(anchored_rows(description, same_series))
    assert out.loc[["pre", "d1", "d2", "d3"], "phase"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert out.loc["pre", "mri_perfusion_source"] == "explicit_text"
    assert (
        out.loc[["d2", "d3"], "mri_perfusion_source"]
        .eq("dynamic_explicit_anchor")
        .all()
    )
    assert "explicit anchors=pre" in out.loc["d1", "mri_perfusion_reason"]
    assert out.loc["d1", "mri_perfusion_source"] == "time_delta"


def test_native_anchor_counts_toward_minimum_acquisition_count():
    out = curate(anchored_rows()[:3])
    assert out.loc[["pre", "d1", "d2"], "phase"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
    ]
    out = curate(anchored_rows()[:2])
    assert out.loc["d1", "phase"] == "ARTERIAL"


def test_named_portal_stays_in_chronology_instead_of_restarting_generic_ranks():
    rows = anchored_rows()
    rows[2]["SeriesDescription"] = "AX T1 LAVA portal"
    out = curate(rows)
    assert out.loc[["pre", "d1", "d2", "d3"], "phase"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert out.loc["d2", "mri_perfusion_source"] == "explicit_text"


@pytest.mark.parametrize(
    "next_phase,expected",
    [
        (None, ["ARTERIAL", "PORTAL_VENOUS", "DELAYED"]),
        ("arterial", ["OTHER", "OTHER", "OTHER"]),
        ("portal", ["ARTERIAL", "ARTERIAL", "ARTERIAL"]),
        ("delayed", ["ARTERIAL", "PORTAL_VENOUS", "PORTAL_VENOUS"]),
    ],
)
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("clocks", [False, True])
def test_order_inference_reserves_succeeding_explicit_phase_only_when_available(
    next_phase, expected, native, clocks
):
    rows = anchored_rows("dynamic post")
    if not native:
        rows = rows[1:]
    if next_phase:
        rows.append(volume("next", next_phase, "120112", series_id="NEXT"))
    for rank, row in enumerate(rows):
        row["acquisition_order"] = rank
        if not clocks:
            row["time"] = None

    out = curate(rows[::-1])

    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == expected
    if next_phase:
        assert out.loc["next", "mri_perfusion_source"] == "explicit_text"
        assert (
            not out.loc[["d1", "d2", "d3"], "phase"].eq(out.loc["next", "phase"]).any()
        )
    expected_source = (
        "dynamic_explicit_anchor"
        if native or next_phase
        else "acquisition_order_dixon_component"
    )
    resolved = out.loc[["d1", "d2", "d3"]].loc[lambda frame: frame.phase.ne("OTHER")]
    assert resolved.mri_perfusion_source.isin(
        [expected_source, "time_delta"] if clocks else [expected_source]
    ).all()
    if "OTHER" in expected:
        assert (
            out.loc[["d1", "d2", "d3"], "mri_perfusion_reason"]
            .str.contains("no compatible postcontrast phase remains")
            .all()
        )


def test_generic_dynamic_starts_arterial_before_succeeding_portal():
    rows = anchored_rows()[1:]
    rows.append(volume("next", "portal", "120112", series_id="NEXT"))

    out = curate(rows)

    assert out.loc[["d1", "d2", "d3", "next"], "phase"].tolist() == [
        "ARTERIAL",
        "ARTERIAL",
        "ARTERIAL",
        "PORTAL_VENOUS",
    ]


def test_repeated_explicit_phase_does_not_advance_unresolved_interval():
    rows = [
        volume("pre", "pre", "115959"),
        volume("art", "arterial", "120000"),
        volume("between", "dynamic post", "120016"),
        volume("art_again", "arterial", "120111"),
        volume("after", "dynamic post", "120112"),
        volume("portal", "portal", "120113"),
    ]

    out = curate(rows)

    assert out.loc[["between", "after"], "phase"].tolist() == [
        "OTHER",
        "ARTERIAL",
    ]
    assert out.loc[["art", "art_again"], "phase"].eq("ARTERIAL").all()
    assert out.loc["portal", "phase"] == "PORTAL_VENOUS"


def test_succeeding_anchor_in_different_image_type_does_not_bound_phases():
    rows = anchored_rows()
    rows.append(
        volume(
            "next", "portal", "120112", series_id="NEXT", ImageType="DERIVED PRIMARY"
        )
    )

    out = curate(rows)

    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]


def test_explicit_phase_at_same_acquisition_position_can_still_be_shared():
    rows = anchored_rows()
    rows.append(volume("portal", "portal", "120016", series_id="PORTAL"))

    out = curate(rows)

    assert out.loc[["d2", "portal"], "phase"].eq("PORTAL_VENOUS").all()
    assert out.loc["d1", "phase"] == "ARTERIAL"
    assert out.loc["d3", "phase"] == "DELAYED"


def test_succeeding_explicit_phase_bounds_matching_time_delta():
    rows = [
        volume("pre", "pre", "120000"),
        volume("d1", "dynamic post", "120025"),
        volume("d2", "dynamic post", "120115"),
        volume("portal", "portal", "120130"),
    ]

    out = curate(rows)

    assert out.loc["d1", "mri_perfusion_source"] == "time_delta"
    assert out.loc["d2", "phase"] == "ARTERIAL"
    assert out.loc["d2", "mri_perfusion_source"] == "dynamic_explicit_anchor"
    assert (
        "reserved succeeding explicit PORTAL_VENOUS phase"
        in out.loc["d2", "mri_perfusion_reason"]
    )
    assert out.loc["portal", "mri_perfusion_source"] == "explicit_text"


def test_explicit_arterial_can_anchor_postcontrast_sequence_without_native():
    rows = anchored_rows()[1:]
    rows[0]["SeriesDescription"] = "AX T1 LAVA arterial"
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert out.loc["d1", "mri_perfusion_source"] == "explicit_text"
    assert not out.phase.eq("NATIVE").any()
    assert "phase evidence=order only" in out.loc["d2", "mri_perfusion_reason"]
    assert "arterial start inferred" not in out.loc["d2", "mri_perfusion_reason"]


def test_conflicting_explicit_anchor_clock_does_not_block_supported_volumes():
    rows = anchored_rows()
    rows[2]["SeriesDescription"] = "AX T1 LAVA delayed"
    out = curate(rows)
    assert out.loc["pre", "phase"] == "NATIVE"
    assert out.loc["d2", "phase"] == "DELAYED"
    assert out.loc[["d1", "d3"], "phase"].tolist() == ["ARTERIAL", "DELAYED"]
    assert out.loc["d2", "mri_perfusion_source"] == "explicit_text"


def test_repeated_explicit_native_acquisitions_do_not_consume_postcontrast_ranks():
    rows = [volume("earlier_pre", "pre", "115800", series_id="PRE0"), *anchored_rows()]
    out = curate(rows)
    assert out.loc[["earlier_pre", "pre"], "phase"].eq("NATIVE").all()
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]


@pytest.mark.parametrize(
    "description,expected",
    [
        ("dynamic", ["ARTERIAL", "PORTAL_VENOUS", "DELAYED"]),
        ("dynamic post", ["ARTERIAL", "PORTAL_VENOUS", "DELAYED"]),
    ],
)
def test_full_and_post_only_dynamic_templates_without_native_anchor(
    description, expected
):
    rows = anchored_rows(description)[1:]
    rows[2]["time"] = "120200"
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == expected


@pytest.mark.parametrize(
    "anchor_change",
    [
        {"study_id": "OTHER"},
        {"date": "2020-01-02"},
        {"patient_key": "OTHER"},
        {"SeriesDescription": "AX T1 VIBE pre"},
        {"acquisition_plane": "CORONAL"},
        {"ImageType": "ORIGINAL PRIMARY F"},
        {"ImageType": "DERIVED PRIMARY"},
    ],
)
def test_unrelated_native_cannot_anchor_post_only_group(anchor_change):
    rows = anchored_rows("dynamic post")
    for row in rows:
        row["acquisition_plane"] = "AXIAL"
    rows[0].update(anchor_change)
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert (
        not out.loc[["d1", "d2", "d3"], "mri_perfusion_reason"]
        .str.contains("explicit anchors=pre")
        .any()
    )


def test_unknown_native_component_can_anchor_only_its_own_dixon_series():
    rows = anchored_rows("dynamic post", same_series=True)
    rows[0]["SeriesDescription"] = "AX T1 mDIXON pre"
    for row in rows[1:]:
        row["SeriesDescription"] = "AX T1 mDIXON water dynamic post"
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    rows[0]["series_id"] = "PRE"
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert (
        not out.loc[["d1", "d2", "d3"], "mri_perfusion_reason"]
        .str.contains("explicit anchors=pre")
        .any()
    )


def test_acquisition_times_take_priority_over_conflicting_ranks_and_input_order():
    rows = anchored_rows("dynamic post")
    for row, rank in zip(rows, [9, 7, 5, 3]):
        row["acquisition_order"] = rank
    out = curate(rows[::-1])
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert "ordered by time" in out.loc["d1", "mri_perfusion_reason"]


def test_raw_acquisition_time_cannot_override_canonical_shared_time():
    rows = anchored_rows("dynamic post")
    for row in rows:
        row["AcquisitionTime"] = row["time"]
        row["time"] = "120000"
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].eq("ARTERIAL").all()


@pytest.mark.parametrize(
    "index,bad_time,expected,reliable",
    [
        (0, None, ["ARTERIAL", "PORTAL_VENOUS", "DELAYED"], [False, False, False]),
        (2, None, ["ARTERIAL", "PORTAL_VENOUS", "DELAYED"], [False, False, False]),
        (1, "246060", ["ARTERIAL", "PORTAL_VENOUS", "DELAYED"], [False, False, False]),
    ],
)
def test_partial_or_invalid_times_keep_individual_evidence(
    index, bad_time, expected, reliable
):
    rows = anchored_rows("dynamic post")
    for rank, row in enumerate(rows):
        row["acquisition_order"] = rank
    rows[index]["time"] = bad_time
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == expected
    if bad_time is not None:
        assert "ordered by acquisition_order" in out.loc["d1", "mri_perfusion_reason"]


def test_completely_missing_times_allow_complete_acquisition_rank():
    rows = anchored_rows("dynamic post")
    for rank, row in enumerate(rows):
        row["time"] = None
        row["acquisition_order"] = rank
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert "ordered by acquisition_order" in out.loc["d1", "mri_perfusion_reason"]


def test_series_local_acquisition_numbers_are_not_compared_across_series():
    rows = anchored_rows("dynamic post")
    for rank, row in enumerate(rows):
        row["time"] = None
        row["AcquisitionNumber"] = rank
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].eq("OTHER").all()
    for position, row in enumerate(rows, start=1):
        row["series_id"] = "DYNAMIC"
        row["volume_order_in_series"] = position
        row["n_volumes_in_series"] = len(rows)
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]


def test_second_complete_clock_can_fill_a_partial_clock():
    rows = anchored_rows("dynamic post")
    for row in rows:
        row["AcquisitionTime"] = row["time"]
    rows[2]["AcquisitionTime"] = None
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert "ordered by time" in out.loc["d1", "mri_perfusion_reason"]


def test_native_does_not_shift_estimated_post_injection_windows():
    rows = anchored_rows("dynamic post")
    for row, clock in zip(rows, ["115000", "120000", "120100", "120500"]):
        row["time"] = clock
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "time_delta"].tolist() == [20, 80, 320]
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert out.loc[["d1", "d2", "d3"], "mri_perfusion_source"].eq("time_delta").all()


@pytest.mark.parametrize(
    "clocks,expected",
    [
        (["115900", "120000", "120001", "120002"], ["ARTERIAL"] * 3),
        (
            ["115959", "120000", "120100", "120500"],
            ["ARTERIAL", "PORTAL_VENOUS", "DELAYED"],
        ),
        (
            ["115900", "120000", "120010", "120500"],
            ["ARTERIAL", "ARTERIAL", "DELAYED"],
        ),
        (
            ["115900", "120000", "120200", "120500"],
            ["ARTERIAL", "PORTAL_VENOUS", "DELAYED"],
        ),
        (
            ["115900", "120000", "120100", "120200"],
            ["ARTERIAL", "PORTAL_VENOUS", "DELAYED"],
        ),
        (
            ["115900", "120000", "120100", "121501"],
            ["ARTERIAL", "PORTAL_VENOUS", "DELAYED"],
        ),
    ],
    ids=[
        "rapid-repeated-frames",
        "native-after-possible-injection",
        "portal-too-early",
        "portal-too-late",
        "delayed-too-early",
        "delayed-too-late",
    ],
)
def test_elapsed_windows_take_precedence_before_order_fallback(clocks, expected):
    rows = anchored_rows("dynamic post")
    for row, clock in zip(rows, clocks):
        row["time"] = clock
    out = curate(rows)
    assert out.loc["pre", "phase"] == "NATIVE"
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == expected


@pytest.mark.parametrize(
    "clocks",
    [
        ["120000", "120035", "120100", "120300"],
        ["120000", "120020", "120130", "121500"],
    ],
)
def test_shared_timing_windows_include_their_boundaries(clocks):
    rows = anchored_rows("dynamic post")
    for row, clock in zip(rows, clocks):
        row["time"] = clock
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]


def test_incompatible_clock_preserves_named_postcontrast_anchor():
    rows = anchored_rows("dynamic post")
    rows[1]["SeriesDescription"] = "AX T1 LAVA arterial"
    rows[2]["time"] = "120016"
    out = curate(rows)
    assert out.loc[["pre", "d1"], "phase"].tolist() == ["NATIVE", "ARTERIAL"]
    assert out.loc[["d2", "d3"], "phase"].tolist() == ["PORTAL_VENOUS", "DELAYED"]
    assert out.loc["d1", "mri_perfusion_source"] == "explicit_text"


def test_no_chronology_does_not_infer_from_lexical_volume_ids():
    rows = anchored_rows()
    for row in rows:
        row["time"] = None
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].eq("OTHER").all()


def test_acquisitions_before_or_tied_with_native_are_not_shifted_to_postcontrast():
    rows = anchored_rows("dynamic post")
    rows[1]["time"] = "115959"
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "OTHER",
        "ARTERIAL",
        "PORTAL_VENOUS",
    ]
    rows[0]["time"] = "120600"
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].eq("OTHER").all()


def test_tied_reconstructions_share_phase_without_consuming_another_rank():
    rows = anchored_rows("dynamic post")
    rows.append(volume("d1_recon", "dynamic post", "120000", series_id="RECON"))
    out = curate(rows)
    assert out.loc[["d1", "d1_recon"], "phase"].eq("ARTERIAL").all()
    assert out.loc[["d2", "d3"], "phase"].tolist() == ["PORTAL_VENOUS", "DELAYED"]


@pytest.mark.parametrize(
    "clock_changes,expected",
    [
        (
            ["110000", "120000", "120100", "120200"],
            ["ARTERIAL", "PORTAL_VENOUS", "DELAYED"],
        ),
        (
            ["115959", "120000", "120010", "121701"],
            ["ARTERIAL", "ARTERIAL", "PORTAL_VENOUS"],
        ),
    ],
)
def test_long_clock_gaps_do_not_block_ordered_dynamic_acquisitions(
    clock_changes, expected
):
    rows = anchored_rows()
    for row, clock in zip(rows, clock_changes):
        row["time"] = clock
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == expected
    assert "phase evidence=order only" in out.loc["d3", "mri_perfusion_reason"]


def test_derived_native_and_derived_dynamics_do_not_anchor_or_shift_ranks():
    rows = anchored_rows("dynamic post")
    rows[0]["SeriesDescription"] += " SUB"
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert (
        not out.loc[["d1", "d2", "d3"], "mri_perfusion_reason"]
        .str.contains("explicit anchors=pre")
        .any()
    )
    rows = anchored_rows()
    rows.append(volume("derived", "dynamic SUB", "115930"))
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]


def test_mixed_pre_post_dynamic_uses_order_when_timing_is_uninformative():
    rows = anchored_rows("dynamic pre/post")[1:]
    rows[2]["time"] = "120220"
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert "phase evidence=order only" in out.loc["d3", "mri_perfusion_reason"]


@pytest.mark.parametrize(
    "identifier",
    [
        "acquisition_order",
        "volume_order_in_series",
    ],
)
def test_same_series_canonical_ranks_preserve_volumes_with_one_clock(identifier):
    rows = anchored_rows("dynamic C+")
    for position, row in enumerate(rows[1:], start=1):
        row.update(
            time="134717",
            AcquisitionTime="134717",
            volume_split_method="repeated_slice_stack",
            n_volumes_in_series=3,
            **{identifier: position},
        )
    rows[0]["time"] = "134617"
    rows.append(volume("late", "delayed", "135444", series_id="LATE"))
    out = curate(rows[::-1])
    assert out.loc[["pre", "d1", "d2", "d3", "late"], "phase"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "ARTERIAL",
        "ARTERIAL",
        "DELAYED",
    ]
    assert (
        out.loc[["d1", "d2", "d3"], "mri_dynamic_order_source"]
        .str.contains(identifier)
        .all()
    )
    assert (
        out.loc[["d1", "d2", "d3"], "mri_perfusion_reason"]
        .str.contains("time_delta=20s")
        .all()
    )
    assert out.loc["late", "mri_perfusion_source"] == "explicit_text"


def test_full_dynamic_resolves_supported_frames_around_incompatible_transition():
    rows = [
        volume(
            f"v{i}",
            "dynamic",
            clock,
            TemporalPositionIdentifier=i,
            AcquisitionTime=clock,
        )
        for i, clock in enumerate(["104139", "104206", "104253", "104345"], start=1)
    ]
    rows.append(
        volume("late", "delayed", "105303", series_id="LATE", AcquisitionTime="105303")
    )
    out = curate(rows[::-1])
    assert out.loc[["v1", "v2", "v3", "v4", "late"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "PORTAL_VENOUS",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert "phase evidence=order only" in out.loc["v4", "mri_perfusion_reason"]
    assert out.loc["late", "mri_perfusion_source"] == "explicit_text"


@pytest.mark.parametrize(
    "source", [None, "SeriesTime", "ContentTime", "InstanceCreationTime", "StudyTime"]
)
def test_canonical_time_supplies_elapsed_windows_regardless_of_source(source):
    rows = anchored_rows("dynamic post")
    for row, clock in zip(rows, ["115900", "120000", "120001", "120002"]):
        row["time"] = clock
        row["time_source"] = source
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].eq("ARTERIAL").all()
    assert out.loc[["d1", "d2", "d3"], "mri_perfusion_source"].eq("time_delta").all()


@pytest.mark.parametrize("marker", ["C+", "+C"])
def test_postcontrast_marker_cannot_supply_an_inferred_native(marker):
    rows = [
        volume(
            f"v{i}",
            f"dynamic {marker}",
            "120000",
            volume_order_in_series=i,
            n_volumes_in_series=3,
        )
        for i in range(1, 4)
    ]
    out = curate(rows)
    assert out.phase.eq("ARTERIAL").all()
    rows.append(volume("portal", "portal", "120300", series_id="PORTAL"))
    out = curate(rows)
    assert not out.phase.eq("NATIVE").any()
    assert out.loc[["v1", "v2", "v3"], "phase"].tolist() == [
        "ARTERIAL",
        "ARTERIAL",
        "ARTERIAL",
    ]


def test_temporal_position_values_are_not_compared_between_series():
    rows = anchored_rows("dynamic post")
    for position, row in enumerate(rows):
        row.update(
            time=None,
            TemporalPositionIdentifier=position,
            series_id=f"SERIES{position}",
        )
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].eq("OTHER").all()


def test_simultaneous_series_temporal_positions_are_not_aligned():
    rows = [volume("pre", "pre", "115959", series_id="PRE")]
    for sid in ["A", "B"]:
        for position in range(1, 4):
            rows.append(
                volume(
                    f"{sid}{position}",
                    "dynamic post",
                    "120000",
                    series_id=sid,
                    volume_order_in_series=position,
                    n_volumes_in_series=3,
                )
            )
    out = curate(rows)
    assert (
        out.loc[[f"{sid}{i}" for sid in ["A", "B"] for i in range(1, 4)], "phase"]
        .eq("ARTERIAL")
        .all()
    )
    assert out.loc["A1", "mri_dynamic_order_source"].endswith(
        "ambiguous volume order across series"
    )


def test_no_temporal_identity_does_not_split_one_clock_using_lexical_volume_ids():
    rows = anchored_rows("dynamic post")
    for row in rows[1:]:
        row["time"] = "120000"
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].eq("ARTERIAL").all()


def test_shared_native_clock_can_be_ordered_with_supplied_series_volume_ranks():
    rows = anchored_rows("dynamic post", same_series=True)
    for position, row in enumerate(rows, start=1):
        row.update(
            time="120000",
            volume_order_in_series=position,
            n_volumes_in_series=len(rows),
        )
    out = curate(rows[::-1])
    assert out.loc[["pre", "d1", "d2", "d3"], "phase"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "ARTERIAL",
        "ARTERIAL",
    ]


def test_unusable_early_generic_volume_does_not_block_explicitly_calibrated_later_volumes():
    rows = anchored_rows("dynamic post")
    rows[1]["time"] = "115959"
    rows[2]["SeriesDescription"] = "AX T1 LAVA portal"
    out = curate(rows)
    assert out.loc[["pre", "d1", "d2", "d3"], "phase"].tolist() == [
        "NATIVE",
        "OTHER",
        "PORTAL_VENOUS",
        "DELAYED",
    ]


def test_succeeding_delayed_anchor_reserves_its_phase_even_after_a_long_gap():
    rows = anchored_rows("dynamic post")
    rows.append(volume("far_late", "delayed", "123000", series_id="FAR"))
    out = curate(rows)
    assert out.loc[["pre", "d1", "d2", "d3", "far_late"], "phase"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "PORTAL_VENOUS",
        "DELAYED",
    ]


def test_canonical_time_is_used_regardless_of_original_clock_source():
    rows = anchored_rows("dynamic post", same_series=True)
    for position, (row, clock) in enumerate(
        zip(rows, ["120500", "120200", "120100", "120000"]), start=1
    ):
        row.update(
            time=clock,
            time_source="InstanceCreationTime",
            volume_order_in_series=position,
            n_volumes_in_series=len(rows),
        )
    out = curate(rows[::-1])
    assert out.loc["pre", "phase"] == "NATIVE"
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "OTHER",
        "OTHER",
        "OTHER",
    ]
    assert out.loc["d2", "mri_perfusion_source"] == "generic_dynamic_context_pending"
    assert out.loc[["d1", "d2", "d3"], "mri_dynamic_order_source"].eq("time").all()
