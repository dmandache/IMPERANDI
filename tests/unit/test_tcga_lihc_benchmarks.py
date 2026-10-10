"""Small public-cohort fixtures; no image downloads or model weights."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pandas as pd
import pytest

from imperandi.ingest.clean import validate_cleaning_manifest
from imperandi.utils.manifest import load_manifest

BENCHMARK = Path(__file__).resolve().parents[2] / "examples/benchmarks"


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, BENCHMARK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cohort = load_example("cohort")
download = load_example("download")


def run_benchmark_runner(tmp_path, target, *, totalseg=None):
    """Record runner commands without loading DICOMs or running a model."""
    log = tmp_path / "calls.jsonl"
    interpreter = tmp_path / "record_python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['BENCHMARK_CALL_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    interpreter.chmod(0o755)
    env = {**os.environ, "PYTHON": str(interpreter), "BENCHMARK_CALL_LOG": str(log)}
    env.pop("TOTALSEG_PHASE", None)
    if totalseg is not None:
        env["TOTALSEG_PHASE"] = totalseg
    work = tmp_path / "work"
    result = subprocess.run(
        [
            "bash",
            str(BENCHMARK / "run.sh"),
            target,
            str(tmp_path / "dicoms"),
            str(work),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    calls = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    return result, calls, work


@pytest.mark.parametrize("totalseg", [None, "0", "1"])
def test_ct_runner_prediction_controls_conversion_and_cohort_input(tmp_path, totalseg):
    result, calls, work = run_benchmark_runner(
        tmp_path, "ct_portal_venous", totalseg=totalseg
    )
    assert result.returncode == 0, result.stderr
    commands = [
        call[4] if call[:2] == ["-m", "imperandi"] else call[1] for call in calls
    ]
    assert commands == (
        ["ingest", "convert", "phase", "build", "compare"]
        if totalseg == "1"
        else ["ingest", "build", "compare"]
    )
    inventory = (
        "nifti_index_phased_curated.csv"
        if totalseg == "1"
        else "dicom_index_curated.csv"
    )
    assert calls[-2][3] == str(work / inventory)
    assert "--inventory" not in calls[-1]
    if totalseg == "1":
        assert calls[1][5] == str(work / "dicom_index_clean.csv")
        assert calls[2][5] == str(work / "nifti_index.csv")


def test_mri_runner_stays_metadata_only_without_conversion(tmp_path):
    result, calls, work = run_benchmark_runner(tmp_path, "mri_multiphase")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 3
    assert calls[1][3] == str(work / "dicom_index_curated.csv")
    assert calls[-1][-2:] == ["--inventory", str(work / "dicom_index_curated.csv")]


@pytest.mark.parametrize("totalseg", ["0", "1", "", "invalid"])
def test_mri_runner_explains_ct_only_backend_and_continues_metadata_only(
    tmp_path, totalseg
):
    result, calls, work = run_benchmark_runner(
        tmp_path, "mri_multiphase", totalseg=totalseg
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr.strip() == (
        "The backend used for phase prediction from image contrast does not "
        "currently support MRI, only CT. Running with `TOTALSEG_PHASE=0` "
        "(metadata only)."
    )
    commands = [
        call[4] if call[:2] == ["-m", "imperandi"] else call[1] for call in calls
    ]
    assert commands == ["ingest", "build", "compare"]
    assert calls[1][3] == str(work / "dicom_index_curated.csv")
    assert calls[-1][-2:] == ["--inventory", str(work / "dicom_index_curated.csv")]


@pytest.mark.parametrize("totalseg", ["", "2", "true"])
def test_ct_runner_rejects_invalid_prediction_values_before_processing(
    tmp_path, totalseg
):
    result, calls, work = run_benchmark_runner(
        tmp_path, "ct_portal_venous", totalseg=totalseg
    )
    assert result.returncode == 2
    assert "TOTALSEG_PHASE must be 0 or 1" in result.stderr
    assert calls == []
    assert not work.exists()


@pytest.mark.parametrize("patients", [[], ["p1"], ["p1", "p2"]])
def test_download_selects_ct_mr_patients_without_paper_cohort_filtering(
    tmp_path, monkeypatch, patients
):
    rows = []
    for patient, modality, collection in [
        ("p1", "CT", "tcga_lihc"),
        ("p2", "MR", "tcga_lihc"),
        ("p3", "PT", "tcga_lihc"),
        ("p4", "SM", "tcga_lihc"),
        ("p1", "SEG", "tcga_lihc"),
        ("p5", "CT", "other_collection"),
    ]:
        rows.append(
            {
                "PatientID": patient,
                "Modality": modality,
                "collection_id": collection,
                "StudyInstanceUID": patient,
                "SeriesInstanceUID": f"{patient}-{modality}",
                "series_size_MB": 1.0,
            }
        )
    calls = []
    client = types.SimpleNamespace(
        index=pd.DataFrame(rows),
        download_from_selection=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setitem(
        sys.modules,
        "idc_index",
        types.SimpleNamespace(IDCClient=lambda: client),
    )
    patient_args = [arg for patient in patients for arg in ("--patient", patient)]
    monkeypatch.setattr("sys.argv", ["download.py", str(tmp_path), *patient_args])
    download.main()
    expected = patients or ["p1", "p2"]
    assert calls == [
        {
            "downloadDir": str(tmp_path),
            "seriesInstanceUID": [
                f"{p}-{'CT' if p == 'p1' else 'MR'}" for p in expected
            ],
            "dirTemplate": "%PatientID/%StudyInstanceUID/%Modality/%SeriesInstanceUID",
        }
    ]
    assert [path.name for path in tmp_path.iterdir()] == ["idc_selection.csv"]
    selection = pd.read_csv(tmp_path / "idc_selection.csv")
    assert set(selection.PatientID) == set(expected)
    assert set(selection.Modality).issubset({"CT", "MR"})


def volume(patient, study, phase, date="2000-01-01", score="10", modality="MR"):
    return {
        "patient_key": patient,
        "study_id": study,
        "series_id": f"{study}-{phase}",
        "volume_id": f"{patient}-{study}-{phase}",
        "date": date,
        "Modality": modality,
        "mri_sequence": "T1",
        "phase": phase,
        "selection_score": score,
    }


def test_mri_multiphase_selects_earliest_date_with_most_phases_then_best_candidates():
    rows = [
        volume("p1", "old-low", phase, date="2000-01-01", score="10")
        for phase in cohort.PHASES
    ]
    rows += [
        volume("p1", "old-high", phase, date="2000-01-01", score="15")
        for phase in cohort.PHASES
    ]
    rows += [
        volume("p1", "new-high", phase, date="2001-01-01", score="100")
        for phase in cohort.PHASES
    ]
    rows += [volume("p2", "incomplete", phase) for phase in cohort.PHASES[:3]]
    rows += [volume("p2", "different-study", "DELAYED")]
    selected, audit = cohort.build_cohort(pd.DataFrame(rows), "mri_multiphase")
    assert set(selected["study_id"]) == {"old-high", "incomplete", "different-study"}
    assert selected.loc[selected["patient_key"].eq("p1"), "date"].unique().tolist() == [
        "2000-01-01"
    ]
    assert set(selected.loc[selected["patient_key"].eq("p1"), "phase"]) == set(
        cohort.PHASES
    )
    assert set(selected.loc[selected["patient_key"].eq("p2"), "phase"]) == set(
        cohort.PHASES
    )
    assert selected.groupby("patient_key")["n_resolved_phases"].first().to_dict() == {
        "p1": 4,
        "p2": 4,
    }
    assert selected.groupby("patient_key")["complete_multiphase"].first().to_dict() == {
        "p1": True,
        "p2": True,
    }
    assert audit["selected"].sum() == 8
    assert audit.loc[audit["study_id"].eq("different-study"), "selected"].all()


def test_mri_partial_studies_prefer_coverage_then_earliest_date():
    rows = [
        volume("p1", "old-one", "NATIVE", date="2000-01-01"),
        volume("p1", "new-two", "ARTERIAL", date="2001-01-01"),
        volume("p1", "new-two", "PORTAL_VENOUS", date="2001-01-01"),
        volume("p2", "one", "DELAYED"),
        {**volume("p3", "unresolved", "OTHER"), "phase_status": "UNRESOLVED"},
    ]
    for row in rows:
        row.setdefault("phase_status", "RESOLVED")
    selected, audit = cohort.build_cohort(pd.DataFrame(rows), "mri_multiphase")
    assert set(selected["study_id"]) == {"new-two", "one"}
    assert selected.groupby("patient_key")["phase"].nunique().to_dict() == {
        "p1": 2,
        "p2": 1,
    }
    assert not audit.loc[audit["patient_key"].eq("p3"), "selected"].any()


def test_mri_counts_distinct_phases_per_date_across_study_uids():
    rows = [volume("p1", f"old-native-{i}", "NATIVE") for i in range(5)]
    rows += [volume("p1", "old-portal", "PORTAL_VENOUS")]
    rows += [
        volume("p1", "new-pre", "NATIVE", date="2001-01-01"),
        volume("p1", "new-post", "ARTERIAL", date="2001-01-01"),
        volume("p1", "new-post", "PORTAL_VENOUS", date="2001-01-01"),
    ]
    selected, audit = cohort.build_cohort(pd.DataFrame(rows), "mri_multiphase")

    assert selected["date"].unique().tolist() == ["2001-01-01"]
    assert len(selected) == 3
    assert selected["n_resolved_phases"].eq(3).all()
    assert not selected["complete_multiphase"].any()
    assert audit.loc[audit["date"].eq("2000-01-01"), "selected"].eq(0).all()


def test_mri_date_ties_choose_earliest_and_do_not_combine_different_dates():
    rows = [
        volume("p1", "old", "NATIVE", score="10"),
        volume("p1", "new", "ARTERIAL", date="2001-01-01", score="100"),
        volume("p2", "native", "NATIVE"),
        volume("p2", "arterial", "ARTERIAL", date="2001-01-01"),
    ]
    selected, _ = cohort.build_cohort(pd.DataFrame(rows), "mri_multiphase")

    assert selected["date"].eq("2000-01-01").all()
    assert selected["phase"].eq("NATIVE").all()
    assert selected["n_resolved_phases"].eq(1).all()


def test_mri_same_date_phase_duplicates_choose_best_candidate_deterministically():
    rows = [
        volume("p1", "low", "NATIVE", score="1"),
        volume("p1", "z-high", "NATIVE", score="20"),
        volume("p1", "a-high", "NATIVE", score="20"),
        volume("p1", "post", "ARTERIAL", score="10"),
    ]
    original = pd.DataFrame(rows)
    selected, audit = cohort.build_cohort(original, "mri_multiphase")
    shuffled, _ = cohort.build_cohort(
        original.sample(frac=1, random_state=7), "mri_multiphase"
    )

    assert set(selected["study_id"]) == {"a-high", "post"}
    assert selected["n_resolved_phases"].eq(2).all()
    assert audit["selected"].sum() == 2
    assert set(selected["volume_id"]) == set(shuffled["volume_id"])


def test_mri_summary_counts_dates_with_multiple_study_uids_as_one_exam(
    tmp_path, monkeypatch
):
    rows = [
        volume("p1", "pre", "NATIVE"),
        *[volume("p1", "post", phase) for phase in cohort.PHASES[1:]],
    ]
    selected, _ = cohort.build_cohort(pd.DataFrame(rows), "mri_multiphase")
    selected_path = tmp_path / "selected.csv"
    reference_path = tmp_path / "reference.csv"
    selected.to_csv(selected_path, index=False)
    pd.DataFrame([{"patient_key": "p1", "date": "2000-01-01"}]).to_csv(
        reference_path, index=False
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "cohort.py",
            "compare",
            "mri_multiphase",
            str(selected_path),
            str(reference_path),
            str(tmp_path),
        ],
    )
    cohort.main()
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["selected_complete_mri_exams"] == 1
    assert summary["selected_partial_mri_exams"] == 0
    assert summary["selected_mri_phase_count_distribution"] == {"4": 1}
    assert summary["matched_complete_mri_exams"] == 1
    assert summary["matched_mri_phases"] == 4


@pytest.fixture
def mri_comparison():
    reference = pd.DataFrame(
        [
            {"patient_key": patient, "date": "2000-01-01", "phase": phase}
            for patient in [
                "full",
                "three",
                "two",
                "one",
                "missing",
                "wrong-date",
                "split-date",
            ]
            for phase in cohort.PHASES
        ]
    )
    rows = [volume("full", f"study-{phase}", phase) for phase in cohort.PHASES]
    # Alternate date spelling and repeated phases must not inflate coverage.
    rows[0]["date"] = "20000101"
    rows.append(volume("full", "duplicate-native", "NATIVE"))
    for patient, count in [("three", 3), ("two", 2), ("one", 1)]:
        rows.extend(volume(patient, patient, phase) for phase in cohort.PHASES[:count])
    rows.extend(
        volume("wrong-date", "wrong", phase, date="2001-01-01")
        for phase in cohort.PHASES
    )
    rows.extend(volume("split-date", "old", phase) for phase in cohort.PHASES[:2])
    rows.extend(
        volume("split-date", "new", phase, date="2001-01-01")
        for phase in cohort.PHASES[2:]
    )
    rows.extend(volume("extra", "extra", phase) for phase in cohort.PHASES)
    return pd.DataFrame(rows), reference


def test_mri_phase_matches_count_distinct_phases_on_each_reference_date(mri_comparison):
    selected, reference = mri_comparison
    matches = cohort.compare_mri_phases(selected, reference).set_index("patient_key")

    assert matches["n_matched_phases"].to_dict() == {
        "full": 4,
        "three": 3,
        "two": 2,
        "one": 1,
        "missing": 0,
        "wrong-date": 0,
        "split-date": 2,
    }
    assert matches.index[matches["full_phase_match"]].tolist() == ["full"]
    assert matches.loc["full", "n_selected_phases"] == 4
    assert matches.loc["full", "matched_phases"] == "|".join(cohort.PHASES)
    assert matches.loc["full", "missing_phases"] == ""
    assert matches.loc["two", "missing_phases"] == "PORTAL_VENOUS|DELAYED"
    assert matches.loc["wrong-date", "missing_phases"] == "|".join(cohort.PHASES)


def test_mri_full_phase_metrics_require_complete_matching_dates(mri_comparison):
    selected, reference = mri_comparison
    metrics, mismatches = cohort.compare_cohorts(
        selected, reference, target="mri_multiphase"
    )
    full = metrics.set_index("level").loc["patient_date_full_phase"]

    assert full[["predicted", "reference", "tp", "fp", "fn"]].tolist() == [
        3,
        7,
        1,
        2,
        6,
    ]
    assert full["precision"] == pytest.approx(1 / 3)
    assert full["recall"] == pytest.approx(1 / 7)
    assert set(
        mismatches.loc[
            mismatches["level"].eq("patient_date_full_phase")
            & mismatches["status"].eq("false_positive"),
            "patient_key",
        ]
    ) == {"wrong-date", "extra"}


def test_mri_full_phase_match_requires_all_four_canonical_phases():
    selected = pd.DataFrame([volume("p1", "partial", p) for p in cohort.PHASES[:2]])
    reference = selected[[*cohort.MRI_EXAM, "phase"]].copy()
    matches = cohort.compare_mri_phases(selected, reference)

    assert matches["n_reference_phases"].tolist() == [2]
    assert matches["n_matched_phases"].tolist() == [2]
    assert matches["full_phase_match"].tolist() == [False]
    metrics, _ = cohort.compare_cohorts(selected, reference, target="mri_multiphase")
    full = metrics.set_index("level").loc["patient_date_full_phase"]
    assert full["predicted"] == full["reference"] == 0
    assert pd.isna(full["recall"])


@pytest.mark.parametrize("empty", [None, "selected", "reference"])
def test_mri_comparison_writes_phase_recovery_and_highlights_full_matches(
    tmp_path, monkeypatch, capsys, mri_comparison, empty
):
    selected, reference = mri_comparison
    if empty == "selected":
        selected = selected.iloc[:0]
    elif empty == "reference":
        reference = reference.iloc[:0]
    selected_path = tmp_path / "selected.csv"
    reference_path = tmp_path / "reference.csv"
    selected.to_csv(selected_path, index=False)
    reference.to_csv(reference_path, index=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "cohort.py",
            "compare",
            "mri_multiphase",
            str(selected_path),
            str(reference_path),
            str(tmp_path),
        ],
    )
    cohort.main()

    summary = json.loads((tmp_path / "summary.json").read_text())
    matches = pd.read_csv(tmp_path / "mri_phase_matches.csv")
    metrics = pd.read_csv(tmp_path / "cohort_metrics.csv").set_index("level")
    if empty == "reference":
        assert matches.empty
        assert summary["reference_mri_exams"] == 0
        assert summary["matched_mri_phase_count_distribution"] == {
            str(count): 0 for count in range(5)
        }
    else:
        assert len(matches) == summary["reference_mri_exams"] == 7
        assert summary["matched_mri_phase_count_distribution"] == (
            {"0": 7, "1": 0, "2": 0, "3": 0, "4": 0}
            if empty == "selected"
            else {"0": 2, "1": 1, "2": 2, "3": 1, "4": 1}
        )
    full_count = 0 if empty else 1
    assert summary["matched_complete_mri_exams"] == full_count
    assert metrics.loc["patient_date_full_phase", "tp"] == full_count
    assert summary["matched_mri_phases"] == (0 if empty else 12)
    assert summary["matched_partial_mri_exams"] == (0 if empty else 4)
    assert summary["selected_complete_mri_exams"] == (0 if empty == "selected" else 3)
    output = capsys.readouterr().out
    assert "full_phase_match" in output
    assert (
        f"Full four-phase match: {full_count} / "
        f"{summary['reference_mri_exams']} reference exams"
    ) in output


def test_ct_comparison_keeps_existing_metric_levels(mri_comparison):
    selected, reference = mri_comparison
    metrics, _ = cohort.compare_cohorts(selected, reference, target="ct_portal_venous")
    assert metrics["level"].tolist() == [
        "patient",
        "patient_date",
        "patient_date_phase",
    ]


@pytest.fixture
def mri_reference_inventory():
    rows = [volume("partial", "old", p) for p in cohort.PHASES[:3]]
    rows.extend(volume("partial", "new", p, date="2001-01-01") for p in cohort.PHASES)
    rows.extend(
        volume("full", f"reference-{p}", p, date="2001-01-01") for p in cohort.PHASES
    )
    rows.append(volume("full", "duplicate-native", "NATIVE", date="20010101"))
    rows.extend(volume("full", "earlier", p) for p in cohort.PHASES)
    rows.extend(
        volume("wrong-date", "new", p, date="2001-01-01") for p in cohort.PHASES
    )
    rows.extend(volume("split", "old", p) for p in cohort.PHASES[:2])
    rows.extend(volume("split", "new", p, date="2001-01-01") for p in cohort.PHASES[2:])
    rows.extend(
        {**volume("missing", "t2", p), "mri_sequence": "T2"} for p in cohort.PHASES
    )
    rows.extend(
        [
            {**volume("partial", "t2", "DELAYED"), "mri_sequence": "T2"},
            volume("partial", "ct", "DELAYED", modality="CT"),
            {
                **volume("partial", "unresolved", "DELAYED"),
                "phase_status": "UNRESOLVED",
            },
            {**volume("missing", "scout", ""), "phase_status": "NOT_APPLICABLE"},
        ]
    )
    for row in rows:
        row.setdefault("phase_status", "RESOLVED")
    reference = pd.DataFrame(
        [
            {"patient_key": patient, "date": date, "phase": phase}
            for patient, date in [
                ("partial", "2000-01-01"),
                ("full", "2001-01-01"),
                ("wrong-date", "2000-01-01"),
                ("missing", "2000-01-01"),
                ("split", "2000-01-01"),
            ]
            for phase in cohort.PHASES
        ]
    )
    return pd.DataFrame(rows), reference


def test_alternate_mri_detection_checks_reference_dates_before_cohort_selection(
    mri_reference_inventory,
):
    inventory, reference = mri_reference_inventory
    selected, _ = cohort.build_cohort(inventory, "mri_multiphase")
    selected_matches = cohort.compare_mri_phases(selected, reference)
    detected = cohort.compare_reference_mri_phases(inventory, reference).set_index(
        "patient_key"
    )

    assert selected_matches["full_phase_match"].sum() == 0
    assert detected["n_matched_phases"].to_dict() == {
        "partial": 3,
        "full": 4,
        "wrong-date": 0,
        "missing": 0,
        "split": 2,
    }
    assert detected.loc["full", "n_detected_phases"] == 4
    assert detected.index[detected["full_phase_match"]].tolist() == ["full"]
    assert detected.loc["partial", "missing_phases"] == "DELAYED"
    assert "n_selected_phases" not in detected.columns


def test_alternate_mri_detection_accepts_inventory_without_phase_status(
    mri_reference_inventory,
):
    inventory, reference = mri_reference_inventory
    detected = cohort.compare_reference_mri_phases(
        inventory.drop(columns="phase_status"), reference
    ).set_index("patient_key")
    # With no status annotation, the canonical T1 delayed phase is eligible.
    assert detected.loc["partial", "n_matched_phases"] == 4
    assert detected.loc["missing", "n_matched_phases"] == 0


@pytest.mark.parametrize("empty", [None, "selected", "inventory", "reference"])
def test_alternate_mri_comparison_writes_inventory_recovery(
    tmp_path, monkeypatch, capsys, mri_reference_inventory, empty
):
    inventory, reference = mri_reference_inventory
    selected, _ = cohort.build_cohort(inventory, "mri_multiphase")
    if empty == "selected":
        selected = selected.iloc[:0]
    elif empty == "inventory":
        inventory = inventory.iloc[:0]
    elif empty == "reference":
        reference = reference.iloc[:0]
    selected_path, reference_path, inventory_path = (
        tmp_path / "selected.csv",
        tmp_path / "reference.csv",
        tmp_path / "inventory.csv",
    )
    selected.to_csv(selected_path, index=False)
    reference.to_csv(reference_path, index=False)
    inventory.to_csv(inventory_path, index=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "cohort.py",
            "compare",
            "mri_multiphase",
            str(selected_path),
            str(reference_path),
            str(tmp_path),
            "--inventory",
            str(inventory_path),
        ],
    )
    cohort.main()

    summary = json.loads((tmp_path / "summary.json").read_text())
    detection = summary["mri_reference_phase_detection"]
    matches = pd.read_csv(tmp_path / "mri_reference_phase_matches.csv")
    assert (
        len(matches)
        == detection["reference_exams"]
        == (0 if empty == "reference" else 5)
    )
    assert detection["inventory"] == str(inventory_path)
    assert detection["required_phases"] == (0 if empty == "reference" else 20)
    assert detection["detected_required_phases"] == (
        0 if empty in ("inventory", "reference") else 9
    )
    assert detection["full_phase_match_exams"] == (
        0 if empty in ("inventory", "reference") else 1
    )
    assert detection["partial_phase_match_exams"] == (
        0 if empty in ("inventory", "reference") else 2
    )
    assert detection["matched_phase_count_distribution"] == (
        {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}
        if empty == "reference"
        else (
            {"0": 5, "1": 0, "2": 0, "3": 0, "4": 0}
            if empty == "inventory"
            else {"0": 2, "1": 0, "2": 1, "3": 1, "4": 1}
        )
    )
    assert summary["matched_complete_mri_exams"] == 0
    assert summary["matched_mri_phases"] == (
        0 if empty in ("selected", "reference") else 2
    )
    assert (
        "Alternate MRI phase detection on reference patient/dates:"
        in capsys.readouterr().out
    )


def test_inventory_comparison_option_is_mri_only(tmp_path, monkeypatch, capsys):
    output = tmp_path / "results"
    monkeypatch.setattr(
        "sys.argv",
        [
            "cohort.py",
            "compare",
            "ct_portal_venous",
            "selected.csv",
            "reference.csv",
            str(output),
            "--inventory",
            "inventory.csv",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        cohort.main()
    assert exc.value.code == 2
    assert "--inventory is supported only for mri_multiphase" in capsys.readouterr().err
    assert not output.exists()


def test_build_cohort_allows_not_applicable_rows_with_empty_phase():
    rows = [
        volume("p1", "portal", "PORTAL_VENOUS", modality="CT"),
        {
            **volume("p1", "scout", "", modality="CT"),
            "phase_status": "NOT_APPLICABLE",
            "SeriesDescription": "SCOUT",
        },
    ]
    rows[0]["phase_status"] = "RESOLVED"

    selected, audit = cohort.build_cohort(pd.DataFrame(rows), "ct_portal_venous")

    assert selected["volume_id"].tolist() == ["p1-portal-PORTAL_VENOUS"]
    assert not audit.loc[audit["study_id"].eq("scout"), "selected"].any()


def test_ct_portal_venous_earliest_date_precedes_quality_and_ties_use_score():
    rows = [
        volume(
            "p1", "old-low", "PORTAL_VENOUS", date="20000101", score="10", modality="CT"
        ),
        volume(
            "p1",
            "old-high",
            "PORTAL_VENOUS",
            date="2000-01-01",
            score="20",
            modality="CT",
        ),
        volume(
            "p1", "new", "PORTAL_VENOUS", date="2001-01-01", score="100", modality="CT"
        ),
        volume("p2", "native", "NATIVE", modality="CT"),
    ]
    selected, _ = cohort.build_cohort(pd.DataFrame(rows), "ct_portal_venous")
    assert selected["study_id"].tolist() == ["old-high"]
    assert selected["date"].tolist() == ["2000-01-01"]


def test_duplicate_exam_phase_is_rejected():
    row = volume("p1", "s1", "NATIVE")
    other = {**row, "volume_id": "other-volume"}
    with pytest.raises(ValueError, match="duplicate exam/phase"):
        cohort.build_cohort(pd.DataFrame([row, other]), "mri_multiphase")


def test_metrics_distinguish_patient_date_phase_and_series():
    pred = pd.DataFrame(
        [
            volume("p1", "s1", "ARTERIAL"),
            volume("p2", "s2", "PORTAL_VENOUS"),
            volume("extra", "s3", "ARTERIAL"),
        ]
    )
    ref = pred.iloc[:2].copy()
    ref.loc[1, "date"] = "2001-01-01"
    ref.loc[0, "series_id"] = "different-uid"
    scores, mismatches = cohort.compare_cohorts(pred, ref)
    scores = scores.set_index("level")
    assert scores.loc["patient", ["tp", "fp", "fn"]].tolist() == [2, 1, 0]
    assert scores.loc["patient_date", ["tp", "fp", "fn"]].tolist() == [1, 2, 1]
    assert scores.loc["series", "tp"] == 0
    assert {"false_positive", "false_negative"} == set(mismatches["status"])


def test_empty_predictions_report_zero_recovery():
    ref = pd.DataFrame([volume("p1", "s1", "ARTERIAL")])
    scores, missing = cohort.compare_cohorts(ref.iloc[:0], ref)
    assert scores["recall"].eq(0).all()
    assert scores["precision"].isna().all()
    assert missing["status"].eq("false_negative").all()


@pytest.mark.parametrize(
    "target,patients,phases", [("mri_multiphase", 17, 68), ("ct_portal_venous", 49, 49)]
)
def test_manifests_and_published_reference_counts(target, patients, phases):
    validate_cleaning_manifest(
        load_manifest(BENCHMARK / target / "manifest.yaml", base_path=BENCHMARK)
    )
    ref = cohort.load_reference(BENCHMARK / target / "reference_cohort.csv", target)
    assert ref["patient_key"].nunique() == patients
    assert len(ref) == phases
    if target == "mri_multiphase":
        assert ref.loc[ref["include_tumor"].eq("1"), "patient_key"].nunique() == 14
    else:
        assert ref["label"].value_counts().to_dict() == {"0": 34, "1": 15}
