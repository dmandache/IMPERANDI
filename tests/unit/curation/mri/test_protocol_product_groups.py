"""Protocol context is isolated by reconstruction product flags."""

import pandas as pd
import pytest

from imperandi.curation.mri import mri_curation as mc


def row(image_type="ORIGINAL PRIMARY", **extra):
    return {
        "patient_key": "p1",
        "study_id": "exam1",
        "date": "2020-01-01",
        "Modality": "MR",
        "SeriesDescription": "AX T1 LAVA dynamic",
        "ImageType": image_type,
        **extra,
    }


def test_product_flags_split_protocol_groups_without_using_dixon_component():
    rows = mc.add_basic_feature_columns(
        pd.DataFrame(
            [
                row("ORIGINAL PRIMARY WATER"),
                row("ORIGINAL PRIMARY IN_PHASE"),
                row("DERIVED SECONDARY MPR WATER"),
                row("DERIVED PRIMARY SUB WATER"),
                row("DERIVED SECONDARY MPR SUB WATER"),
            ]
        )
    )
    keys = rows.apply(mc._generic_dynamic_key, axis=1)
    assert keys[0] == keys[1]
    assert len(set(keys)) == 4
    for i in range(1, len(rows)):
        assert mc._matches_dynamic_group(rows.iloc[i], keys[0]) == (i == 1)


@pytest.mark.parametrize("flag", ["is_mip_mpr", "is_subtraction"])
def test_missing_product_flags_default_to_false(flag):
    base = pd.Series(row())
    missing = base.copy()
    missing[flag] = pd.NA
    assert mc._generic_dynamic_key(base) == mc._generic_dynamic_key(missing)


def test_mpr_and_source_dynamic_runs_progress_independently():
    rows = pd.DataFrame(
        [
            row(
                "DERIVED SECONDARY MPR" if mpr else "ORIGINAL PRIMARY",
                volume_id=f"{'mpr' if mpr else 'source'}{phase}",
                series_id=f"s{phase}_{mpr}",
                acquisition_order=phase * 2 + int(mpr),
                n_volumes_in_series=1,
                volume_order_in_series=1,
            )
            for phase in range(4)
            for mpr in [False, True]
        ]
    )
    out = mc.annotate_mri(rows).set_index("volume_id")
    for product in ["source", "mpr"]:
        run = out.loc[[f"{product}{phase}" for phase in range(4)]]
        assert run.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]
        assert run.mri_perfusion_reason.str.contains(
            f"is_mip_mpr={product == 'mpr'}"
        ).all()


def test_mpr_phase_does_not_supply_native_context_to_source_protocol():
    rows = pd.DataFrame(
        [
            row(SeriesDescription="AX T1 LAVA", volume_id="source", series_id="s0"),
            row(
                "DERIVED SECONDARY MPR",
                SeriesDescription="AX T1 LAVA portal",
                volume_id="mpr",
                series_id="s1",
            ),
        ]
    )
    out = mc.annotate_mri(rows).set_index("volume_id")
    assert out.loc["source", "phase"] == "NATIVE"
    assert out.loc["source", "mri_perfusion_source"] == "native_no_perfusion_evidence"
    assert out.loc["mpr", "phase"] == "PORTAL_VENOUS"
