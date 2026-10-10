"""Generic dynamic chronology must retain explicit phase anchors."""

import pandas as pd
import pytest

from imperandi.curation.mri.curate import curate_mri


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
    return [
        volume("pre", "pre", "115900", series_id="DYNAMIC" if same_series else "PRE"),
        volume("d1", description, "120000"),
        volume("d2", description, "120100"),
        volume("d3", description, "120500"),
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
        out.loc[["d1", "d2", "d3"], "mri_perfusion_source"]
        .eq("dynamic_explicit_anchor")
        .all()
    )
    assert "explicit anchors=pre" in out.loc["d1", "mri_perfusion_reason"]
    assert (
        "arterial start inferred from acquisition order"
        in out.loc["d1", "mri_perfusion_reason"]
    )


def test_native_anchor_counts_toward_minimum_acquisition_count():
    out = curate(anchored_rows()[:3])
    assert out.loc[["pre", "d1", "d2"], "phase"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
    ]
    out = curate(anchored_rows()[:2])
    assert out.loc["d1", "phase"] == "OTHER"


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
    assert (
        "explicit postcontrast acquisition clocks"
        in out.loc["d2", "mri_perfusion_reason"]
    )
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
        ("dynamic", ["NATIVE", "ARTERIAL", "PORTAL_VENOUS"]),
        ("dynamic post", ["OTHER", "OTHER", "OTHER"]),
    ],
)
def test_missing_native_keeps_full_dynamic_fallback_but_not_post_only(
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
    ],
)
def test_unrelated_native_cannot_anchor_post_only_group(anchor_change):
    rows = anchored_rows("dynamic post")
    for row in rows:
        row["acquisition_plane"] = "AXIAL"
    rows[0].update(anchor_change)
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].eq("OTHER").all()


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
    assert out.loc[["d1", "d2", "d3"], "phase"].eq("OTHER").all()


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


def test_acquisition_time_tag_takes_priority_over_coalesced_series_time():
    rows = anchored_rows("dynamic post")
    for row in rows:
        row["AcquisitionTime"] = row["time"]
        row["time"] = "120000"
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert "ordered by AcquisitionTime" in out.loc["d1", "mri_perfusion_reason"]


@pytest.mark.parametrize(
    "index,bad_time,expected,reliable",
    [
        (0, None, ["ARTERIAL", "PORTAL_VENOUS", "DELAYED"], [True, True, True]),
        (2, None, ["ARTERIAL", "PORTAL_VENOUS", "DELAYED"], [True, False, True]),
        (1, "246060", ["OTHER", "PORTAL_VENOUS", "DELAYED"], [False, False, False]),
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
    assert (
        out.loc[["d1", "d2", "d3"], "mri_dynamic_timing_reliable"].tolist() == reliable
    )
    if bad_time is not None:
        assert "invalid acquisition clock" in out.loc["d1", "mri_perfusion_reason"]


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
    for row in rows:
        row["series_id"] = "DYNAMIC"
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


def test_native_anchor_does_not_supply_injection_delay_windows():
    rows = anchored_rows("dynamic post")
    for row, clock in zip(rows, ["115000", "120000", "120100", "120500"]):
        row["time"] = clock
    out = curate(rows)
    # Native acquisition started ten minutes before the postcontrast series;
    # these offsets are chronology evidence, not ten-minute injection delays.
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]


@pytest.mark.parametrize(
    "clocks,expected",
    [
        (["115900", "120000", "120001", "120002"], ["ARTERIAL"] * 3),
        (["115959", "120000", "120100", "120500"], ["OTHER"] * 3),
        (["115900", "120000", "120010", "120500"], ["ARTERIAL", "ARTERIAL", "DELAYED"]),
        (["115900", "120000", "120200", "120500"], ["ARTERIAL", "OTHER", "DELAYED"]),
        (
            ["115900", "120000", "120100", "120200"],
            ["ARTERIAL", "PORTAL_VENOUS", "OTHER"],
        ),
        (
            ["115900", "120000", "120100", "121501"],
            ["ARTERIAL", "PORTAL_VENOUS", "OTHER"],
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
def test_timing_windows_resolve_each_acquisition_including_repeated_phases(
    clocks, expected
):
    rows = anchored_rows("dynamic post")
    for row, clock in zip(rows, clocks):
        row["time"] = clock
    out = curate(rows)
    assert out.loc["pre", "phase"] == "NATIVE"
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == expected
    for name, phase in zip(["d1", "d2", "d3"], expected):
        if phase == "OTHER":
            assert (
                out.loc[name, "mri_perfusion_source"]
                == "generic_dynamic_context_blocked"
            )
            assert "timing windows" in out.loc[name, "mri_perfusion_reason"]


@pytest.mark.parametrize(
    "clocks",
    [
        ["115959", "120035", "120100", "120300"],
        ["115959", "120020", "120130", "121500"],
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
    rows[2]["time"] = "120001"
    out = curate(rows)
    assert out.loc[["pre", "d1"], "phase"].tolist() == ["NATIVE", "ARTERIAL"]
    assert out.loc[["d2", "d3"], "phase"].tolist() == ["ARTERIAL", "DELAYED"]
    assert out.loc["d1", "mri_perfusion_source"] == "explicit_text"


def test_no_chronology_does_not_infer_from_lexical_volume_ids():
    rows = anchored_rows()
    for row in rows:
        row["time"] = None
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].eq("OTHER").all()


def test_acquisitions_before_or_tied_with_native_are_not_shifted_to_postcontrast():
    rows = anchored_rows("dynamic post")
    rows[1]["time"] = "115900"
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].eq("OTHER").all()
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
        (["110000", "120000", "120100", "120200"], ["OTHER"] * 3),
        (
            ["115900", "120000", "120100", "121701"],
            ["ARTERIAL", "PORTAL_VENOUS", "OTHER"],
        ),
    ],
)
def test_disconnected_acquisition_does_not_invalidate_supported_segment(
    clock_changes, expected
):
    rows = anchored_rows()
    for row, clock in zip(rows, clock_changes):
        row["time"] = clock
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == expected
    assert out.loc["d3", "mri_perfusion_source"] == "generic_dynamic_context_blocked"


def test_derived_native_and_derived_dynamics_do_not_anchor_or_shift_ranks():
    rows = anchored_rows("dynamic post")
    rows[0]["SeriesDescription"] += " SUB"
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].eq("OTHER").all()
    rows = anchored_rows()
    rows.append(volume("derived", "dynamic SUB", "115930"))
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]


def test_blocked_generic_group_cannot_be_resolved_by_later_native_fallback():
    rows = anchored_rows("dynamic pre/post")[1:]
    rows[2]["time"] = "120220"
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "OTHER",
    ]
    assert out.loc["d3", "mri_perfusion_source"] == "generic_dynamic_context_blocked"


@pytest.mark.parametrize(
    "identifier",
    [
        "TemporalPositionIdentifier",
        "TemporalPositionIndex",
        "AcquisitionNumber",
        "volume_order_in_series",
    ],
)
def test_same_series_temporal_identifiers_preserve_volumes_with_one_clock(identifier):
    rows = anchored_rows("dynamic C+")
    for position, row in enumerate(rows[1:], start=1):
        row.update(
            time="134717",
            AcquisitionTime="134717",
            volume_split_method="repeated_slice_stack",
            **{identifier: position},
        )
    rows[0]["time"] = "134148"
    rows.append(volume("late", "delayed", "135444", series_id="LATE"))
    out = curate(rows[::-1])
    assert out.loc[["pre", "d1", "d2", "d3", "late"], "phase"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
        "DELAYED",
    ]
    assert (
        out.loc[["d1", "d2", "d3"], "mri_dynamic_order_source"]
        .str.contains(identifier)
        .all()
    )
    assert not out.loc[["d1", "d2", "d3"], "mri_dynamic_timing_reliable"].any()
    assert out.loc[["d1", "d2", "d3"], "mri_dynamic_timing_source"].eq("").all()
    assert (
        out.loc[["d1", "d2", "d3"], "mri_perfusion_reason"]
        .str.contains("phase evidence=order only")
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
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "OTHER",
        "DELAYED",
    ]
    assert out.loc["v4", "mri_perfusion_source"] == "generic_dynamic_context_blocked"
    assert out.loc[["v2", "v3", "v4"], "mri_dynamic_timing_reliable"].all()
    assert (
        out.loc[["v2", "v3", "v4"], "mri_dynamic_timing_source"]
        .eq("AcquisitionTime")
        .all()
    )
    assert out.loc["late", "mri_perfusion_source"] == "explicit_text"


def test_reliable_timing_allows_multiple_acquisitions_of_each_postcontrast_phase():
    rows = [volume("pre", "pre", "115900", series_id="PRE")]
    clocks = ["120000", "120004", "120100", "120110", "120400", "120430"]
    for position, clock in enumerate(clocks, start=1):
        rows.append(
            volume(
                f"d{position}",
                "dynamic post",
                clock,
                AcquisitionTime=clock,
                TemporalPositionIdentifier=position,
            )
        )
    out = curate(rows[::-1])
    assert out.loc[[f"d{i}" for i in range(1, 7)], "phase"].tolist() == [
        "ARTERIAL",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "PORTAL_VENOUS",
        "DELAYED",
        "DELAYED",
    ]
    assert out.loc[[f"d{i}" for i in range(1, 7)], "mri_dynamic_timing_reliable"].all()


@pytest.mark.parametrize(
    "source", [None, "SeriesTime", "ContentTime", "InstanceCreationTime", "StudyTime"]
)
def test_non_acquisition_clocks_supply_order_without_phase_timing(source):
    rows = anchored_rows("dynamic post")
    for row, clock in zip(rows, ["115900", "120000", "120001", "120002"]):
        row["time"] = clock
        row["time_source"] = source
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert not out.loc[["d1", "d2", "d3"], "mri_dynamic_timing_reliable"].any()
    assert out.loc[["d1", "d2", "d3"], "mri_dynamic_timing_source"].eq("").all()


@pytest.mark.parametrize("marker", ["C+", "+C"])
def test_postcontrast_marker_cannot_supply_an_inferred_native(marker):
    rows = [
        volume(f"v{i}", f"dynamic {marker}", "120000", TemporalPositionIdentifier=i)
        for i in range(1, 4)
    ]
    out = curate(rows)
    assert out.phase.eq("OTHER").all()
    rows.append(volume("portal", "portal", "120300", series_id="PORTAL"))
    out = curate(rows)
    assert not out.phase.eq("NATIVE").any()
    assert out.loc[["v1", "v2", "v3"], "phase"].tolist() == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "PORTAL_VENOUS",
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
    rows = [volume("pre", "pre", "115900", series_id="PRE")]
    for sid in ["A", "B"]:
        for position in range(1, 4):
            rows.append(
                volume(
                    f"{sid}{position}",
                    "dynamic post",
                    "120000",
                    series_id=sid,
                    TemporalPositionIdentifier=position,
                )
            )
    out = curate(rows)
    assert (
        out.loc[[f"{sid}{i}" for sid in ["A", "B"] for i in range(1, 4)], "phase"]
        .eq("OTHER")
        .all()
    )
    assert out.loc["A1", "mri_dynamic_order_source"].endswith(
        "ambiguous temporal order across series"
    )


def test_no_temporal_identity_does_not_split_one_clock_using_lexical_volume_ids():
    rows = anchored_rows("dynamic post")
    for row in rows[1:]:
        row["time"] = "120000"
    out = curate(rows)
    assert out.loc[["d1", "d2", "d3"], "phase"].eq("OTHER").all()


def test_shared_native_clock_can_be_ordered_only_with_same_series_temporal_positions():
    rows = anchored_rows("dynamic post", same_series=True)
    for position, row in enumerate(rows, start=1):
        row.update(
            time="120000", AcquisitionTime="120000", TemporalPositionIdentifier=position
        )
    out = curate(rows[::-1])
    assert out.loc[["pre", "d1", "d2", "d3"], "phase"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert not out.loc[["d1", "d2", "d3"], "mri_dynamic_timing_reliable"].any()


def test_unusable_early_generic_volume_does_not_block_explicitly_calibrated_later_volumes():
    rows = anchored_rows("dynamic post")
    rows[1]["time"] = "115900"
    rows[2]["SeriesDescription"] = "AX T1 LAVA portal"
    out = curate(rows)
    assert out.loc[["pre", "d1", "d2", "d3"], "phase"].tolist() == [
        "NATIVE",
        "OTHER",
        "PORTAL_VENOUS",
        "DELAYED",
    ]


def test_far_delayed_anchor_does_not_block_a_complete_prefix_of_supported_acquisitions():
    rows = anchored_rows("dynamic post")
    rows.append(volume("far_late", "delayed", "123000", series_id="FAR"))
    out = curate(rows)
    assert out.loc[["pre", "d1", "d2", "d3", "far_late"], "phase"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
        "DELAYED",
    ]


def test_creation_clocks_cannot_override_same_series_temporal_identifiers():
    rows = anchored_rows("dynamic post", same_series=True)
    for position, (row, clock) in enumerate(
        zip(rows, ["120500", "120200", "120100", "120000"]), start=1
    ):
        row.update(
            time=clock,
            time_source="InstanceCreationTime",
            TemporalPositionIdentifier=position,
        )
    out = curate(rows[::-1])
    assert out.loc[["pre", "d1", "d2", "d3"], "phase"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert (
        out.loc[["d1", "d2", "d3"], "mri_dynamic_order_source"]
        .eq("TemporalPositionIdentifier within series")
        .all()
    )
    assert not out.loc[["d1", "d2", "d3"], "mri_dynamic_timing_reliable"].any()
