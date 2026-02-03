import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from imperandi.ingest.parse import choose_ids, apply_id_standardization


def test_choose_ids_path_only(tmp_path):
    root = tmp_path
    p1 = root / "patient1" / "studyA" / "series1" / "img1.dcm"
    p2 = root / "patient2" / "img2.dcm"
    df = pd.DataFrame({"dicom_path": [str(p1), str(p2)]})

    out = choose_ids(
        df.copy(),
        root,
        id_source="path",
        patient_tag="PatientName",
        study_tag="StudyInstanceUID",
        series_tag="SeriesInstanceUID",
    )

    assert "patient_key" in out.columns
    assert out.loc[0, "patient_key"] == "patient1"
    assert out.loc[0, "study_id"] == "studyA"
    assert out.loc[0, "series_id"] == "series1"
    assert out.loc[1, "patient_key"] == "patient2"
    assert pd.isna(out.loc[1, "study_id"])
    assert pd.isna(out.loc[1, "series_id"])
    assert out.loc[0, "dicom_filename"] == "img1.dcm"
    assert out.loc[1, "dicom_filename"] == "img2.dcm"
    # intermediate columns removed
    assert "patient_key_path" not in out.columns
    assert "study_path" not in out.columns
    assert "series_path" not in out.columns


def test_choose_ids_tags_only_and_trimming(tmp_path):
    root = tmp_path
    p1 = root / "p" / "s" / "sr" / "a.dcm"
    p2 = root / "p2" / "b.dcm"
    df = pd.DataFrame(
        {
            "dicom_path": [str(p1), str(p2)],
            "PatientName": [" Alice ", None],
            "StudyInstanceUID": ["S1", ""],
            "SeriesInstanceUID": [None, "  "],
        }
    )

    out = choose_ids(
        df.copy(),
        root,
        id_source="tags",
        patient_tag="PatientName",
        study_tag="StudyInstanceUID",
        series_tag="SeriesInstanceUID",
    )

    assert out.loc[0, "patient_key"] == "Alice"
    assert pd.isna(out.loc[1, "patient_key"])
    # study/series missing or blank -> "0"
    assert out.loc[0, "study_id"] == "S1"
    assert out.loc[1, "study_id"] == "0"
    assert out.loc[0, "series_id"] == "0"
    assert out.loc[1, "series_id"] == "0"


def test_choose_ids_auto_fallback_to_path(tmp_path):
    root = tmp_path
    p1 = root / "patientA" / "studyX" / "seriesX" / "img1.dcm"
    p2 = root / "patientB" / "studyY" / "img2.dcm"
    df = pd.DataFrame(
        {
            "dicom_path": [str(p1), str(p2)],
            "PatientName": [None, " Bob "],
            "StudyInstanceUID": [None, " STUDY2 "],
            "SeriesInstanceUID": [None, None],
        }
    )

    out = choose_ids(
        df.copy(),
        root,
        id_source="auto",
        patient_tag="PatientName",
        study_tag="StudyInstanceUID",
        series_tag="SeriesInstanceUID",
    )

    assert out.loc[0, "patient_key"] == "patientA"
    assert out.loc[1, "patient_key"] == "Bob"
    assert out.loc[0, "study_id"] == "0"
    assert out.loc[1, "study_id"] == "STUDY2"
    assert out.loc[0, "series_id"] == "0"
    assert out.loc[1, "series_id"] == "0"


def test_apply_id_standardization_monkeypatched_hook(monkeypatch):
    # prepare df
    df = pd.DataFrame({"patient_key": [" Alice ", "X", None]})

    # hook: strip & upper except for 'X' -> simulate failing standardization
    def hook(v):
        if v is None:
            return None
        v = str(v).strip()
        if v == "X":
            return None
        return v.upper()

    # monkeypatch resolve_hook used inside apply_id_standardization
    import imperandi.ingest.parse as parse_module

    monkeypatch.setattr(parse_module, "resolve_hook", lambda _cfg: hook)

    out = apply_id_standardization(df.copy(), manifest={"id_standardization": {"dummy": True}})

    # raw preserved
    assert "patient_key_raw" in out.columns
    assert out.loc[0, "patient_key"] == "ALICE"
    # failing raw ('X') should mark std_failed True
    assert out.loc[1, "patient_key_raw"] == "X"
    assert out.loc[1, "patient_key_std_failed"] is True
    # None remains None and not marked as failed
    assert pd.isna(out.loc[2, "patient_key"])
    assert "patient_key_std_failed" in out.columns