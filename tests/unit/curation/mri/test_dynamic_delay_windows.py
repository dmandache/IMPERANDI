"""Generic dynamic phases prefer matching elapsed times, then acquisition order."""

from datetime import datetime, timedelta

import pandas as pd
import pytest
import yaml

from imperandi import cli
from imperandi.curation.mri import mri_curation as mc


def dynamic_rows(deltas, description="AX T1 LAVA dynamic"):
    start = datetime(2020, 1, 1, 12)
    return pd.DataFrame(
        {
            "patient_key": "p1",
            "study_id": "exam1",
            "series_id": "dynamic",
            "volume_id": [f"v{i}" for i in range(len(deltas))],
            "Modality": "MR",
            "SeriesDescription": description,
            "ImageType": "ORIGINAL PRIMARY",
            "time": [(start + timedelta(seconds=s)).time().isoformat() for s in deltas],
            "acquisition_order": range(len(deltas)),
        }
    )


def test_matching_deltas_allow_repeated_phases_before_order_fallback():
    rows = dynamic_rows([0, 5, 10, 25, 220])
    out = mc.annotate_mri(rows.sample(frac=1, random_state=7)).set_index("volume_id")
    out = out.loc[rows.volume_id]

    assert out.phase.tolist() == [
        "ARTERIAL",
        "ARTERIAL",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert (
        out.loc[["v0", "v1", "v2", "v4"], "mri_perfusion_source"].eq("time_delta").all()
    )
    assert "time_delta=25s" in out.loc["v1", "mri_perfusion_reason"]
    assert "phase evidence=order only" in out.loc["v3", "mri_perfusion_reason"]


def test_time_delta_below_succeeding_explicit_phase_retains_priority():
    rows = dynamic_rows([0, 5, 10, 25, 220])
    rows.loc[4, "SeriesDescription"] = "AX T1 LAVA delayed"

    out = mc.annotate_mri(rows)

    assert out.phase.tolist() == [
        "ARTERIAL",
        "ARTERIAL",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert out.loc[[0, 1, 2], "mri_perfusion_source"].eq("time_delta").all()
    assert out.loc[4, "mri_perfusion_source"] == "explicit_text"


def test_upper_anchor_applies_to_time_delta_before_minimum_order_count():
    rows = dynamic_rows([0, 90])
    rows.loc[1, "SeriesDescription"] = "AX T1 LAVA portal"
    annotated = mc.annotate_mri(rows, resolve_phase=False)
    annotated.loc[0, "time_delta"] = 75
    annotated.loc[0, "mri_perfusion_source"] = mc.GENERIC_DYNAMIC_CONTEXT_PENDING
    annotated.loc[0, "mri_perfusion_label"] = "OTHER"

    assignments = mc.infer_generic_dynamic_phases_from_exam_context(annotated)

    assert assignments[0][1] == "OTHER"
    assert "reserved succeeding explicit PORTAL_VENOUS phase" in assignments[0][2]


@pytest.mark.parametrize(
    "seconds,phase",
    [
        (20, "ARTERIAL"),
        (35, "ARTERIAL"),
        (60, "PORTAL_VENOUS"),
        (90, "PORTAL_VENOUS"),
        (180, "DELAYED"),
        (900, "DELAYED"),
        (1200, "HEPATOBILIARY"),
        (7200, "HEPATOBILIARY"),
        (10799, "HEPATOBILIARY"),
    ],
)
def test_default_window_boundaries_resolve_without_three_ordered_acquisitions(
    seconds, phase
):
    out = mc.annotate_mri(dynamic_rows([0, seconds - 20]))

    assert out.loc[0, "phase"] == "ARTERIAL"
    assert out.loc[1, "phase"] == phase
    assert out.loc[1, "mri_perfusion_source"] == "time_delta"
    assert out.loc[1, "mri_perfusion_confidence"] == "inferred"


@pytest.mark.parametrize(
    "delta",
    [
        None,
        "invalid",
        float("nan"),
        float("inf"),
        -1,
        0,
        19.99,
        35.01,
        59.99,
        90.01,
        179.99,
        900.01,
        1200.01,
        7199.99,
        10799.01,
    ],
)
def test_invalid_or_out_of_window_deltas_use_order_fallback(delta):
    rows = mc.annotate_mri(dynamic_rows([0, 1, 2, 3]), resolve_phase=False)
    rows["mri_perfusion_source"] = mc.GENERIC_DYNAMIC_CONTEXT_PENDING
    rows["time_delta"] = delta

    assignments = mc.infer_generic_dynamic_phases_from_exam_context(rows)

    assert [label for _, label, *_ in assignments] == [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
        "DELAYED",
    ]
    assert all(source != "time_delta" for _, _, _, source, _ in assignments)


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize("missing", [None, "invalid"])
def test_unresolved_current_or_reference_time_uses_order_fallback(position, missing):
    rows = dynamic_rows([0, 16, 71, 100])
    rows.loc[position, "time"] = missing

    out = mc.annotate_mri(rows)

    assert out.phase.tolist() == ["ARTERIAL", "PORTAL_VENOUS", "DELAYED", "DELAYED"]
    assert pd.isna(out.loc[position, "time_delta"])


@pytest.mark.parametrize("upper_anchor", [False, True])
def test_matching_deltas_with_ambiguous_local_order_respect_upper_anchor(upper_anchor):
    native = dynamic_rows([0], "AX T1 LAVA pre")
    series = []
    for sid in ["A", "B"]:
        rows = dynamic_rows([25, 25])
        rows["series_id"] = sid
        rows["volume_id"] = sid + rows.volume_id
        rows["volume_order_in_series"] = [1, 2]
        rows["n_volumes_in_series"] = 2
        series.append(rows)
    if upper_anchor:
        following = dynamic_rows([30], "AX T1 LAVA arterial")
        following["series_id"] = "following"
        following["volume_id"] = "following"
        series.append(following)
    rows = pd.concat([native, *series], ignore_index=True).drop(
        columns="acquisition_order"
    )

    out = mc.annotate_mri(rows)

    assert out.loc[1:4, "time_delta"].eq(20).all()
    assert out.loc[1:4, "phase"].eq("OTHER" if upper_anchor else "ARTERIAL").all()
    assert (
        out.loc[1:4, "mri_perfusion_source"]
        .eq(mc.GENERIC_DYNAMIC_CONTEXT_PENDING if upper_anchor else "time_delta")
        .all()
    )
    assert out.loc[1:4, "mri_dynamic_order_source"].str.contains("ambiguous").all()


@pytest.mark.parametrize(
    "description,expected",
    [
        ("AX T1 LAVA dynamic arterial", ["ARTERIAL"] * 4),
        ("AX T1 LAVA Multiart", ["ARTERIAL"] * 4),
        ("AX T1 LAVA Mask+Multiart", ["NATIVE", "ARTERIAL", "ARTERIAL", "ARTERIAL"]),
        ("AX T1 LAVA ART-PORT", ["ARTERIAL", "OTHER", "OTHER", "PORTAL_VENOUS"]),
        ("AX T1 LAVA ART/PORT/LATE", ["ARTERIAL", "OTHER", "OTHER", "DELAYED"]),
    ],
)
def test_explicit_labels_and_special_profiles_keep_precedence(description, expected):
    out = mc.annotate_mri(dynamic_rows([0, 60, 180, 240], description))

    assert out.phase.tolist() == expected
    assert not out.mri_perfusion_source.eq("time_delta").any()


@pytest.mark.parametrize("command", ["clean", "phase"])
def test_cli_uses_time_windows_and_order_fallback(tmp_path, command):
    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    dynamic_rows([0, 5, 10, 25]).to_csv(source, index=False)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "phase_curation": {"strategies": [{"type": "rules"}]},
                "cleaning": {
                    "steps": [{"type": "modality_curation"}, {"type": "phase_curation"}]
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        cli.main([command, str(source), str(output), "--manifest", str(manifest)]) == 0
    )

    out = pd.read_csv(output)
    assert out.phase.tolist() == ["ARTERIAL", "ARTERIAL", "ARTERIAL", "PORTAL_VENOUS"]
    assert out.loc[[0, 1, 2], "mri_perfusion_source"].eq("time_delta").all()
