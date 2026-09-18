"""Best-series export and manifest-defined exam identification."""

import pandas as pd
import pytest
import yaml

from imperandi import cli
from imperandi.curation import curate_by_modality
from imperandi.curation.export import (
    save_selected_candidates,
    selected_output_path,
    selected_qc_path,
)
from imperandi.curation.phase import validate_phase_curation
from imperandi.extract import phase


def cohort(modality="CT"):
    return pd.DataFrame(
        [
            {
                "patient_key": "p1",
                "study_id": "same-study",
                "visit": visit,
                "series_id": f"s{i}",
                "volume_id": f"v{i}",
                "Modality": modality,
                "SeriesDescription": "AX T1 portal venous",
                "ImageType": "ORIGINAL PRIMARY AXIAL",
                "Rows": 512,
                "Columns": 512,
                "SliceThickness": thickness,
                "n_files": 120,
                "nifti_path": f"volume{i}.nii.gz",
                "_patient_key_raw": "original-id",
            }
            for i, (visit, thickness) in enumerate([(1, 2), (1, 9), (2, 2)])
        ]
    )


def config():
    return {
        "strategies": [{"type": "rules"}],
        "exam_group_columns": ["patient_key", "visit"],
    }


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_exam_columns_control_selection_and_null_exams_are_retained(modality):
    df = cohort(modality)
    df.loc[df["visit"].eq(2), "visit"] = None
    results = curate_by_modality(df, phase_curation=config())
    selected = results["selected_long_all"]
    assert set(selected["volume_id"]) == {"v0", "v2"}
    assert selected["visit"].isna().sum() == 1


@pytest.mark.parametrize("columns", [[], "visit", [""], [1], ["visit", "visit"]])
def test_invalid_exam_columns_are_rejected(columns):
    with pytest.raises(ValueError, match="exam_group_columns"):
        validate_phase_curation({**config(), "exam_group_columns": columns})


def test_missing_explicit_exam_column_is_rejected():
    with pytest.raises(ValueError, match="missing columns.*visit"):
        curate_by_modality(cohort().drop(columns="visit"), phase_curation=config())


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_export_uses_final_phase_preserves_paths_and_excludes_helpers(
    tmp_path, modality
):
    df = cohort(modality)
    df["phase"] = "ARTERIAL"
    df["phase_source"] = "model"
    df["phase_confidence"] = 0.9
    df["phase_reason"] = "prediction"
    path = tmp_path / "nested" / "selected.csv"
    assert save_selected_candidates(df, path, config()) == 2
    selected = pd.read_csv(path)
    assert set(selected["volume_id"]) == {"v0", "v2"}
    assert selected["phase"].tolist() == ["ARTERIAL", "ARTERIAL"]
    assert set(selected["phase_source"]) == {"model"}
    assert set(selected["selection_slot"]) == {
        "CT_ARTERIAL" if modality == "CT" else "T1_ARTERIAL"
    }
    assert set(selected["nifti_path"]) == {"volume0.nii.gz", "volume2.nii.gz"}
    assert "_patient_key_raw" in selected
    assert "_row_order" not in selected
    assert len(df) == 3
    qc = pd.read_csv(selected_qc_path(path))
    slot = "CT_ARTERIAL" if modality == "CT" else "T1_ARTERIAL"
    assert qc["visit"].tolist() == [1, 2]
    assert qc[slot].tolist() == selected["selected_candidate"].tolist()
    assert qc["curation_modality"].tolist() == [modality, modality]
    assert "volume_id" not in qc
    if modality == "MR":
        assert pd.notna(qc.loc[0, "T1_ARTERIAL_other_candidates"])
        assert pd.isna(qc.loc[1, "T1_ARTERIAL_other_candidates"])


@pytest.mark.parametrize("modality", ["CT", "MR", "PT"])
def test_export_with_no_candidates_has_readable_headers(tmp_path, modality):
    df = cohort(modality)
    df["SeriesDescription"] = "localizer scout"
    df["phase"] = "OTHER"
    path = tmp_path / "selected.csv"
    assert save_selected_candidates(df, path, config()) == 0
    assert pd.read_csv(path).empty
    qc = pd.read_csv(selected_qc_path(path))
    assert qc.empty
    assert {"patient_key", "visit", "curation_modality"}.issubset(qc.columns)


def test_mixed_modality_export_retains_both_wide_tables(tmp_path):
    df = pd.concat([cohort("CT"), cohort("MR")], ignore_index=True)
    df["phase"] = "PORTAL_VENOUS"
    path = tmp_path / "selected.csv"
    assert save_selected_candidates(df, path, config()) == 4
    qc = pd.read_csv(selected_qc_path(path))
    assert len(qc) == 4
    assert {"CT_PORTAL_VENOUS", "T1_PORTAL_VENOUS"}.issubset(qc.columns)
    assert qc.loc[qc["curation_modality"].eq("CT"), "CT_PORTAL_VENOUS"].notna().all()
    assert qc.loc[qc["curation_modality"].eq("MR"), "T1_PORTAL_VENOUS"].notna().all()


def test_export_path_cannot_overwrite_input_or_main_output(tmp_path):
    main = tmp_path / "output.csv"
    source = tmp_path / "input.csv"
    for path in (main, source):
        with pytest.raises(ValueError, match="must differ"):
            selected_output_path(path, main, protected_paths=[source])
    with pytest.raises(ValueError, match="end in .csv"):
        selected_output_path(tmp_path / "selected.txt", main)
    path = tmp_path / "selected.csv"
    with pytest.raises(ValueError, match="QC path must differ"):
        selected_output_path(path, selected_qc_path(path))
    with pytest.raises(ValueError, match="QC path must differ"):
        selected_output_path(path, main, protected_paths=[selected_qc_path(path)])


def test_default_export_paths_follow_input_even_with_different_main_output(tmp_path):
    source = tmp_path / "cohort.v2.csv"
    main = tmp_path / "results" / "phase.csv"
    selected = selected_output_path(None, main, input_path=source)
    assert selected == tmp_path / "cohort.v2_selected.csv"
    assert selected_qc_path(selected) == tmp_path / "cohort.v2_selected_qc.csv"
    custom = tmp_path / "custom.csv"
    assert selected_output_path(custom, main, input_path=source) == custom
    with pytest.raises(ValueError, match="must differ"):
        selected_output_path(None, selected, input_path=source)


def test_clean_cli_exports_selected_and_loads_custom_exam_columns(tmp_path):
    source, main, selected = [
        tmp_path / name for name in ("in.csv", "out.csv", "in_selected.csv")
    ]
    cohort().to_csv(source, index=False)
    manifest = tmp_path / "site.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "phase_curation": config(),
                "cleaning": {"steps": [{"type": "modality_curation"}]},
            }
        )
    )
    assert (
        cli.main(
            [
                "clean",
                str(source),
                str(main),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    assert len(pd.read_csv(main)) == 3
    assert pd.read_csv(selected)["visit"].tolist() == [1, 2]
    qc = pd.read_csv(selected_qc_path(selected))
    assert qc["visit"].tolist() == [1, 2]
    assert "CT_PORTAL_VENOUS" in qc


def test_missing_exam_identifiers_produce_empty_default_exports(tmp_path, caplog):
    source = tmp_path / "volumes.csv"
    pd.DataFrame([{"nifti_path": "volume.nii.gz", "rule_phase": "ARTERIAL"}]).to_csv(
        source, index=False
    )
    assert cli.main(["phase", str(source)]) == 0
    assert pd.read_csv(tmp_path / "volumes_selected.csv").empty
    assert pd.read_csv(tmp_path / "volumes_selected_qc.csv").empty
    assert "No exam grouping columns" in caplog.text


def test_phase_cli_can_export_on_finished_resume_without_prediction(
    tmp_path, monkeypatch
):
    source, main, selected = [
        tmp_path / name for name in ("in.csv", "out.csv", "selected.csv")
    ]
    cohort().to_csv(source, index=False)
    manifest = tmp_path / "site.yaml"
    phase_config = {**config(), "strategies": [{"type": "totalsegmentator"}]}
    manifest.write_text(yaml.safe_dump({"phase_curation": phase_config}))
    calls = []

    def predict(idx, row, **kwargs):
        calls.append(idx)
        return idx, {"totalseg_phase": "arterial"}, None

    monkeypatch.setattr(phase, "_load_phase_extractor", lambda: object())
    monkeypatch.setattr(phase, "process_single_volume", predict)
    args = ["phase", str(source), str(main), "--manifest", str(manifest)]
    assert cli.main(args) == 0
    assert calls == [0, 1, 2]
    assert not selected.exists()
    assert not selected_qc_path(selected).exists()
    default_selected = tmp_path / "in_selected.csv"
    assert set(pd.read_csv(default_selected)["volume_id"]) == {"v0", "v2"}
    assert pd.read_csv(selected_qc_path(default_selected))["CT_ARTERIAL"].notna().all()
    assert cli.main([*args, "--selected_csv_path", str(selected)]) == 0
    assert calls == [0, 1, 2]
    assert len(pd.read_csv(main)) == 3
    out = pd.read_csv(selected)
    assert set(out["volume_id"]) == {"v0", "v2"}
    assert set(out["phase"]) == {"ARTERIAL"}
    qc_path = selected_qc_path(selected)
    assert pd.read_csv(qc_path)["CT_ARTERIAL"].notna().all()
    # Replacing a stale export must not retain rows absent from the selection.
    pd.DataFrame([{"volume_id": "stale"}]).to_csv(selected, index=False)
    pd.DataFrame([{"stale": True}]).to_csv(qc_path, index=False)
    assert cli.main([*args, "--selected_csv_path", str(selected)]) == 0
    assert set(pd.read_csv(selected)["volume_id"]) == {"v0", "v2"}
    qc = pd.read_csv(qc_path)
    assert "stale" not in qc
    assert qc["visit"].tolist() == [1, 2]
