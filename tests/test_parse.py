import sys
import io
import json
import tarfile
import zipfile
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from imperandi.ingest import parse
from imperandi.ingest import hook_manifests


def _make_archive_dataset(root: Path) -> Path:
    inner_tar = root / "inner.tar.gz"
    with tarfile.open(inner_tar, "w:gz") as tf:
        payload = b"DICM_FAKE"
        info = tarfile.TarInfo(name="patientA/study1/series1/img1.dcm")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    outer_zip = root / "outer.zip"
    with zipfile.ZipFile(outer_zip, "w") as zf:
        zf.write(inner_tar, arcname="nested/inner.tar.gz")
    return outer_zip


def test_normalize_parse_args_prefers_flags(tmp_path):
    root_pos = tmp_path / "root_pos"
    out_pos = tmp_path / "out_pos"
    root_opt = tmp_path / "root_opt"
    out_opt = tmp_path / "out_opt"

    args = parse.normalize_parse_args(
        parse.argparse.Namespace(
            root_path_pos=str(root_pos),
            output_dir_pos=str(out_pos),
            root_path_opt=str(root_opt),
            output_dir_opt=str(out_opt),
        )
    )

    assert args.root_path == str(root_opt)
    assert args.output_dir == str(out_opt)
    assert not hasattr(args, "root_path_pos")
    assert not hasattr(args, "root_path_opt")
    assert not hasattr(args, "output_dir_pos")
    assert not hasattr(args, "output_dir_opt")


def test_normalize_parse_args_defaults_to_cwd():
    args = parse.normalize_parse_args(
        parse.argparse.Namespace(
            root_path_pos=None,
            output_dir_pos=None,
            root_path_opt=None,
            output_dir_opt=None,
        )
    )

    expected_root = Path.cwd()
    assert args.root_path == str(expected_root)
    assert args.output_dir == str(expected_root.parent)
    assert args.archive_detect_sample_size == 128


def test_normalize_parse_args_glob_defaults_output_to_matched_parent(tmp_path):
    (tmp_path / "site_a").mkdir()
    (tmp_path / "site_b").mkdir()
    root_pattern = str(tmp_path / "site_*")

    args = parse.normalize_parse_args(
        parse.argparse.Namespace(
            root_path_pos=None,
            output_dir_pos=None,
            root_path_opt=root_pattern,
            output_dir_opt=None,
        )
    )

    assert args.root_path == root_pattern
    assert args.output_dir == str(tmp_path)


def test_choose_ids_path_only(tmp_path):
    root = tmp_path
    p1 = root / "patient1" / "studyA" / "series1" / "img1.dcm"
    p2 = root / "patient2" / "img2.dcm"
    df = pd.DataFrame({"dicom_path": [str(p1), str(p2)]})

    out = parse.choose_ids(
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
    # assert pd.isna(out.loc[1, "study_id"])
    # assert pd.isna(out.loc[1, "series_id"])
    assert out.loc[1, "study_id"] == "0"
    assert out.loc[1, "series_id"] == "0"
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

    out = parse.choose_ids(
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
            "PatientName": [pd.NA, "Bob"],
            "StudyInstanceUID": [None, "STUDY2"],
            "SeriesInstanceUID": [None, None],
        }
    )

    out = parse.choose_ids(
        df.copy(),
        root,
        id_source="auto",
        patient_tag="PatientName",
        study_tag="StudyInstanceUID",
        series_tag="SeriesInstanceUID",
    )

    assert out.loc[0, "patient_key"] == "patientA"
    assert out.loc[1, "patient_key"] == "Bob"
    assert out.loc[0, "study_id"] == "studyX"
    assert out.loc[1, "study_id"] == "STUDY2"
    assert out.loc[0, "series_id"] == "seriesX"
    assert out.loc[1, "series_id"] == "0"


def test_choose_ids_path_uses_scan_root_col_for_glob_inputs(tmp_path):
    root_a = tmp_path / "site_a"
    root_b = tmp_path / "site_b"
    p1 = root_a / "patientA" / "study1" / "series1" / "img1.dcm"
    p2 = root_b / "patientB" / "study2" / "series2" / "img2.dcm"

    df = pd.DataFrame(
        {
            "dicom_path": [str(p1), str(p2)],
            "_scan_root": [str(root_a), str(root_b)],
        }
    )

    out = parse.choose_ids(
        df.copy(),
        tmp_path / "site_*",
        id_source="path",
        patient_tag="PatientName",
        study_tag="StudyInstanceUID",
        series_tag="SeriesInstanceUID",
    )

    assert out.loc[0, "patient_key"] == "patientA"
    assert out.loc[1, "patient_key"] == "patientB"
    assert out.loc[0, "study_id"] == "study1"
    assert out.loc[1, "study_id"] == "study2"
    assert out.loc[0, "series_id"] == "series1"
    assert out.loc[1, "series_id"] == "series2"
    assert "_scan_root" not in out.columns


def test_get_dicom_paths_supports_glob_root(tmp_path):
    root_a = tmp_path / "site_a" / "patient1"
    root_b = tmp_path / "site_b" / "patient2"
    root_a.mkdir(parents=True)
    root_b.mkdir(parents=True)

    p1 = root_a / "a.dcm"
    p2 = root_b / "b.dcm"
    p1.write_text("")
    p2.write_text("")

    paths = parse.get_dicom_paths(str(tmp_path / "site_*"))
    assert set(paths) == {p1, p2}


def test_get_dicom_path_entries_supports_archives(tmp_path):
    archive = _make_archive_dataset(tmp_path)
    entries = parse.get_dicom_path_entries(str(archive), archive_max_depth=3)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["is_archive_member"]
    assert entry["source_uri_or_path"].startswith("archive://")
    assert entry["relative_path"] == "patientA/study1/series1/img1.dcm"


def test_choose_ids_uses_relative_path_for_archive_sources(tmp_path):
    archive = _make_archive_dataset(tmp_path)
    entry = parse.get_dicom_path_entries(str(archive), archive_max_depth=3)[0]
    df = pd.DataFrame(
        {
            "dicom_path": [entry["source_uri_or_path"]],
            "_scan_root": [entry["scan_root"]],
            "_relative_path": [entry["relative_path"]],
        }
    )
    out = parse.choose_ids(
        df.copy(),
        tmp_path,
        id_source="path",
        patient_tag="PatientName",
        study_tag="StudyInstanceUID",
        series_tag="SeriesInstanceUID",
    )

    assert out.loc[0, "patient_key"] == "patientA"
    assert out.loc[0, "study_id"] == "study1"
    assert out.loc[0, "series_id"] == "series1"


def test_get_dicom_path_entries_skips_corrupt_archive(tmp_path):
    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_bytes(b"not-a-valid-archive")

    entries = parse.get_dicom_path_entries(str(bad_zip), archive_max_depth=3)
    assert entries == []


def test_detect_archive_mode_by_subsample(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for i in range(5):
        (root / f"{i:03d}.dcm").write_text("")
    assert not parse.detect_archive_mode_by_subsample([root], sample_size=4)

    (root / "010.zip").write_text("not-a-zip")
    assert parse.detect_archive_mode_by_subsample([root], sample_size=20)


def test_apply_id_standardization_monkeypatched_hook(monkeypatch):
    # prepare df
    df = pd.DataFrame({"patient_key": [" Alice ", "X", None, pd.NA]})

    # hook: strip & upper except for 'X' -> simulate failing standardization
    def hook(v):
        if v is None or pd.isna(v):
            return None
        v = str(v).strip()
        if v == "X":
            return None
        return v.upper()

    monkeypatch.setattr(hook_manifests, "resolve_hook", lambda _cfg: hook)

    out = hook_manifests.apply_id_standardization(
        df.copy(),
        manifest={"id_standardization": {"dummy": True}},
        hook_resolver=hook_manifests.resolve_hook,
        logger=parse.logger,
    )

    print(out)
    # raw preserved
    assert "_patient_key_raw" in out.columns
    assert out.loc[0, "patient_key"] == "ALICE"
    # failing raw ('X') should mark std_failed True
    assert out.loc[1, "_patient_key_raw"] == "X"
    assert out.loc[1, "patient_key_std_failed"]
    # None remains None and not marked as failed
    assert pd.isna(out.loc[2, "patient_key"])
    assert "patient_key_std_failed" in out.columns


def test_adds_new_columns(monkeypatch):
    df = pd.DataFrame({"value": [1, 2]})
    monkeypatch.setattr(
        hook_manifests,
        "resolve_hook",
        lambda d: (lambda x: {"double": x * 2, "is_odd": x % 2 == 1}),
    )
    manifest = {"derived_columns": [{"from_column": "value"}]}
    out = hook_manifests.apply_derived_columns(
        df.copy(), manifest, hook_resolver=hook_manifests.resolve_hook
    )
    assert "double" in out.columns and "is_odd" in out.columns
    assert out["double"].tolist() == [2, 4]
    assert out["is_odd"].tolist() == [True, False]


def test_missing_only_does_not_overwrite(monkeypatch):
    df = pd.DataFrame({"value": [1, 2], "double": [99, 99]})
    monkeypatch.setattr(
        hook_manifests, "resolve_hook", lambda d: (lambda x: {"double": x * 2})
    )
    manifest = {"derived_columns": [{"from_column": "value"}]}
    out = hook_manifests.apply_derived_columns(
        df.copy(), manifest, hook_resolver=hook_manifests.resolve_hook
    )
    # default join_mode is 'missing_only' -> existing 'double' stays unchanged
    assert out["double"].tolist() == [99, 99]


def test_overwrite_replaces_columns(monkeypatch):
    df = pd.DataFrame({"value": [1, 2], "double": [99, 99]})
    monkeypatch.setattr(
        hook_manifests, "resolve_hook", lambda d: (lambda x: {"double": x * 2})
    )
    manifest = {"derived_columns": [{"from_column": "value", "join_mode": "overwrite"}]}
    out = hook_manifests.apply_derived_columns(
        df.copy(), manifest, hook_resolver=hook_manifests.resolve_hook
    )
    assert out["double"].tolist() == [2, 4]


def test_noop_when_no_derived_columns():
    df = pd.DataFrame({"a": [1]})
    out = hook_manifests.apply_derived_columns(df.copy(), {})
    assert out.equals(df)


def test_build_effective_tags_includes_defaults_user_and_id_tags():
    tags = parse.build_effective_tags(
        default_tags=["PatientName", "StudyInstanceUID"],
        user_tags=["Modality", "PatientName"],
        patient_tag="PatientID",
        study_tag="StudyInstanceUID",
        series_tag="SeriesInstanceUID",
    )
    assert tags == [
        "PatientName",
        "StudyInstanceUID",
        "Modality",
        "PatientID",
        "SeriesInstanceUID",
    ]


def test_read_dicom_header_selected_returns_only_requested(monkeypatch):
    class FakeDataset:
        def __init__(self):
            self._values = {
                "PatientName": "Alice",
                "Modality": "CT",
                "UnusedTag": "x",
            }

        def get(self, key):
            return self._values.get(key)

    monkeypatch.setattr(parse, "dcmread", lambda *args, **kwargs: FakeDataset())
    out = parse.read_dicom_header_selected(
        "fake.dcm",
        tags=["PatientName", "Modality"],
        force=False,
    )
    assert set(out.index.tolist()) == {"PatientName", "Modality"}
    assert out["PatientName"] == "Alice"
    assert out["Modality"] == "CT"


def test_build_global_readers_auto_switches(monkeypatch):
    calls = []

    def fake_standard(source, *, tags, force):
        calls.append(("standard", source))
        return pd.Series({"PatientName": "A"})

    def fake_archive(source, *, tags, force, archive_max_depth):
        calls.append(("archive", source))
        return pd.Series({"PatientName": "B"})

    monkeypatch.setattr(parse, "read_dicom_header_selected_standard", fake_standard)
    monkeypatch.setattr(parse, "read_dicom_header_selected_archive_aware", fake_archive)
    monkeypatch.setattr(
        parse,
        "read_dicom_header_standard",
        lambda source, *, force: pd.Series({"x": 1}),
    )
    monkeypatch.setattr(
        parse,
        "read_dicom_header_archive_aware",
        lambda source, *, force, archive_max_depth: pd.Series({"x": 2}),
    )
    monkeypatch.setattr(
        parse, "is_archive_uri", lambda v: str(v).startswith("archive://")
    )

    selected, _full, state = parse.build_global_readers(
        initial_archive_mode=False,
        tags=["PatientName"],
        force=False,
        archive_max_depth=3,
    )

    selected("a.dcm")
    selected("archive://x!y")
    selected("b.dcm")

    assert calls[0][0] == "standard"
    assert calls[1][0] == "archive"
    assert calls[2][0] == "archive"
    assert state["archive_mode"]
    assert state["auto_switched"]


def test_process_with_checkpoint_keeps_csv_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(pd.Series, "parallel_apply", pd.Series.apply, raising=False)
    df_paths = pd.DataFrame(
        {"dicom_path": ["a.dcm", "b.dcm"], "_read_path": ["a.dcm", "b.dcm"]}
    )

    out = parse.process_with_checkpoint(
        df_paths=df_paths,
        read_func=lambda _: pd.Series({"PatientName": "x"}),
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
        output_dir=tmp_path,
        final_name="dicom_paths_with_tags.csv",
        read_path_col="_read_path",
    )

    assert (tmp_path / "dicom_paths_with_tags.csv").exists()
    assert (tmp_path / ".dicom_paths_with_tags.parse.state.json").exists()
    assert (tmp_path / ".dicom_paths_with_tags.parse.checkpoint.csv").exists()
    assert list(tmp_path.glob("dicom_paths_with_tags_*.csv")) == []
    assert len(out) == 2


def test_process_with_checkpoint_preserves_all_empty_columns_per_chunk(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pd.Series, "parallel_apply", pd.Series.apply, raising=False)
    df_paths = pd.DataFrame({"dicom_path": ["a.dcm", "b.dcm"]})

    out = parse.process_with_checkpoint(
        df_paths=df_paths,
        read_func=lambda _: pd.Series(
            {
                "AcquisitionTime": "120000",
                "InstanceCreationTime": None,
            }
        ),
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
        output_dir=tmp_path,
        final_name="dicom_paths_with_tags.csv",
    )

    ckpt = pd.read_csv(tmp_path / ".dicom_paths_with_tags.parse.checkpoint.csv")

    assert "InstanceCreationTime" in ckpt.columns
    assert "InstanceCreationTime" in out.columns
    assert out["InstanceCreationTime"].isna().all()


def test_process_with_checkpoint_reports_file_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(pd.Series, "parallel_apply", pd.Series.apply, raising=False)
    recorded = {"kwargs": None, "updates": []}

    class DummyProgressBar:
        def __init__(self, **kwargs):
            recorded["kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, amount):
            recorded["updates"].append(amount)

    monkeypatch.setattr(parse, "tqdm", lambda **kwargs: DummyProgressBar(**kwargs))
    df_paths = pd.DataFrame({"dicom_path": [f"f{i}.dcm" for i in range(5)]})

    parse.process_with_checkpoint(
        df_paths=df_paths,
        read_func=lambda _: pd.Series({"PatientName": "x"}),
        checkpoint_every_rows=2,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
        output_dir=tmp_path,
        final_name="dicom_paths_with_tags.csv",
    )

    assert recorded["kwargs"] == {"total": 5, "desc": "Parse files", "unit": "file"}
    assert recorded["updates"] == [2, 2, 1]
    assert sum(recorded["updates"]) == 5


def test_process_with_checkpoint_counts_resumed_rows_in_progress(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pd.Series, "parallel_apply", pd.Series.apply, raising=False)
    recorded = {"updates": []}
    calls = []

    class DummyProgressBar:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, amount):
            recorded["updates"].append(amount)

    monkeypatch.setattr(parse, "tqdm", lambda **kwargs: DummyProgressBar(**kwargs))
    df_paths = pd.DataFrame({"dicom_path": ["a.dcm", "b.dcm", "c.dcm"]})

    parse.process_with_checkpoint(
        df_paths=df_paths,
        read_func=lambda _: pd.Series({"PatientName": "x"}),
        checkpoint_every_rows=2,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
        output_dir=tmp_path,
        final_name="dicom_paths_with_tags.csv",
    )

    out = parse.process_with_checkpoint(
        df_paths=df_paths,
        read_func=lambda path: calls.append(path) or pd.Series({"PatientName": "x"}),
        checkpoint_every_rows=2,
        checkpoint_every_sec=3600,
        resume=True,
        strict_resume=False,
        output_dir=tmp_path,
        final_name="dicom_paths_with_tags.csv",
    )

    assert calls == []
    assert recorded["updates"] == [3]
    assert sum(recorded["updates"]) == 3
    assert len(out) == 3


def test_process_with_checkpoint_rejects_non_positive_frequency(tmp_path):
    df_paths = pd.DataFrame({"dicom_path": ["a.dcm"]})

    def read_func(_):
        return pd.Series({"PatientName": "x"})

    with pytest.raises(ValueError, match="positive integer"):
        parse.process_with_checkpoint(
            df_paths=df_paths,
            read_func=read_func,
            checkpoint_every_rows=0,
            checkpoint_every_sec=10,
            resume=False,
            strict_resume=False,
            output_dir=tmp_path,
            final_name="dicom_paths_with_tags.csv",
        )

    with pytest.raises(ValueError, match="positive integer"):
        parse.process_with_checkpoint(
            df_paths=df_paths,
            read_func=read_func,
            checkpoint_every_rows=2,
            checkpoint_every_sec=-1,
            resume=False,
            strict_resume=False,
            output_dir=tmp_path,
            final_name="dicom_paths_with_tags.csv",
        )


def test_process_with_checkpoint_without_checkpoints_writes_only_final(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pd.Series, "parallel_apply", pd.Series.apply, raising=False)
    df_paths = pd.DataFrame({"dicom_path": ["a.dcm", "b.dcm"]})

    out = parse.process_with_checkpoint(
        df_paths=df_paths,
        read_func=lambda _: pd.Series({"PatientName": "x"}),
        checkpoint_every_rows=10,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
        output_dir=tmp_path,
        final_name="dicom_paths_with_tags.csv",
    )

    assert (tmp_path / "dicom_paths_with_tags.csv").exists()
    assert list(tmp_path.glob("dicom_paths_with_tags_*.csv")) == []
    assert len(out) == 2
    assert "PatientName" in out.columns


def test_write_dicom_tags_snapshot_is_deterministic(tmp_path):
    df = pd.DataFrame(
        {
            "dicom_path": [f"p{i}.dcm" for i in range(10)],
            "_scan_root": ["root"] * 10,
            "_relative_path": [f"rel/{i}.dcm" for i in range(10)],
            "_read_path": [f"read/{i}.dcm" for i in range(10)],
        }
    )

    def fake_full_reader(fp):
        return pd.Series({"Source": fp})

    out1 = tmp_path / "snap1.ndjson"
    out2 = tmp_path / "snap2.ndjson"
    n1 = parse.write_dicom_tags_snapshot(
        df=df,
        output_path=out1,
        sample_size=5,
        seed=42,
        read_full_func=fake_full_reader,
    )
    n2 = parse.write_dicom_tags_snapshot(
        df=df,
        output_path=out2,
        sample_size=5,
        seed=42,
        read_full_func=fake_full_reader,
    )

    lines1 = out1.read_text(encoding="utf-8").strip().splitlines()
    lines2 = out2.read_text(encoding="utf-8").strip().splitlines()
    assert n1 == 5
    assert n2 == 5
    assert lines1 == lines2

    record = json.loads(lines1[0])
    assert set(record.keys()) == {
        "dicom_path",
        "_scan_root",
        "_relative_path",
        "snapshot_seed",
        "snapshot_index",
        "tags",
    }


def test_write_dicom_tags_snapshot_samples_unique_series_when_available(tmp_path):
    df = pd.DataFrame(
        {
            "dicom_path": [f"p{i}.dcm" for i in range(8)],
            "_scan_root": ["root"] * 8,
            "_relative_path": [f"rel/{i}.dcm" for i in range(8)],
            "_read_path": [f"read/{i}.dcm" for i in range(8)],
            "SeriesInstanceUID": ["S1", "S1", "S2", "S2", "S3", "S3", "S4", "S4"],
        }
    )

    out = tmp_path / "snap.ndjson"
    n = parse.write_dicom_tags_snapshot(
        df=df,
        output_path=out,
        sample_size=10,
        seed=42,
        series_col="SeriesInstanceUID",
        read_full_func=lambda fp: pd.Series({"Source": fp}),
    )

    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x]
    assert n == 4
    assert len(lines) == 4
    assert len({item["tags"]["Source"] for item in lines}) == 4


def test_write_dicom_tags_snapshot_normalizes_missing_empty_strings(tmp_path):
    df = pd.DataFrame(
        {
            "dicom_path": ["p0.dcm"],
            "_scan_root": ["root"],
            "_relative_path": ["rel/0.dcm"],
            "_read_path": ["read/0.dcm"],
        }
    )

    out = tmp_path / "snap_empty.ndjson"
    n = parse.write_dicom_tags_snapshot(
        df=df,
        output_path=out,
        sample_size=1,
        seed=42,
        read_full_func=lambda _: pd.Series(
            {
                "EmptyTag": "",
                "WhitespaceTag": "   ",
                "Nested": ["", "ok"],
                "PresentTag": "value",
            }
        ),
    )

    assert n == 1
    line = out.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    tags = record["tags"]
    assert tags["EmptyTag"] is None
    assert tags["WhitespaceTag"] is None
    assert tags["Nested"] == [None, "ok"]
    assert tags["PresentTag"] == "value"
