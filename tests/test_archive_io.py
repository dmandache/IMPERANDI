import io
import sys
import tarfile
import zipfile
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from imperandi.utils import archive_io


def _make_nested_zip_tar_with_dicom(root: Path) -> tuple[Path, str]:
    inner_tar = root / "inner.tar.gz"
    with tarfile.open(inner_tar, "w:gz") as tf:
        payload = b"DICM_FAKE"
        info = tarfile.TarInfo(name="patientA/study1/series1/img1.dcm")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    outer_zip = root / "outer.zip"
    with zipfile.ZipFile(outer_zip, "w") as zf:
        zf.write(inner_tar, arcname="nested/inner.tar.gz")

    return outer_zip, "nested/inner.tar.gz"


def test_encode_decode_archive_uri_roundtrip(tmp_path):
    outer = tmp_path / "dataset.zip"
    chain = ["a/b/inner.tar.gz", "patient/study/series/img.dcm"]
    uri = archive_io.encode_archive_uri(outer, chain)
    assert uri.startswith("archive://")

    decoded_outer, decoded_chain = archive_io.decode_archive_uri(uri)
    assert decoded_outer == outer.resolve()
    assert decoded_chain == chain


def test_iter_archive_members_nested_zip_tar(tmp_path):
    outer_zip, _ = _make_nested_zip_tar_with_dicom(tmp_path)
    members = list(archive_io.iter_archive_members(outer_zip, max_depth=3))
    leaf_members = [m for m in members if not m["is_archive"]]
    assert any(
        m["entry_chain"][-1] == "patientA/study1/series1/img1.dcm" for m in leaf_members
    )


def test_discover_dicom_sources_returns_archive_uri(tmp_path):
    outer_zip, _ = _make_nested_zip_tar_with_dicom(tmp_path)
    records = archive_io.discover_dicom_sources([outer_zip], max_depth=3)
    assert len(records) == 1
    rec = records[0]
    assert rec["is_archive_member"]
    assert rec["source_uri_or_path"].startswith("archive://")
    assert rec["relative_path"] == "patientA/study1/series1/img1.dcm"


def test_iter_archive_members_respects_depth_limit(tmp_path):
    outer_zip, _ = _make_nested_zip_tar_with_dicom(tmp_path)
    members = list(archive_io.iter_archive_members(outer_zip, max_depth=0))
    leaf_members = [m for m in members if not m["is_archive"]]
    assert not leaf_members


def test_materialize_rejects_zip_slip_member(tmp_path):
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.dcm", b"bad")

    uri = archive_io.encode_archive_uri(bad_zip, ["../escape.dcm"])
    with archive_io.ArchiveSession(cache_dir=tmp_path / ".cache") as session:
        with pytest.raises(ValueError):
            session.materialize(uri)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("dataset.ZIP", True),
        ("dataset.tar", True),
        ("dataset.tar.gz", True),
        ("dataset.tgz", True),
        ("dataset.gz", False),
        ("dataset", False),
    ],
)
def test_archive_filename_detection(name, expected):
    assert archive_io.is_archive_filename(name) is expected


def test_archive_uri_validation_errors(tmp_path):
    with pytest.raises(ValueError, match="at least one member"):
        archive_io.encode_archive_uri(tmp_path / "data.zip", [])
    with pytest.raises(ValueError, match="Not an archive URI"):
        archive_io.decode_archive_uri("data.zip")
    with pytest.raises(ValueError, match="Empty archive URI"):
        archive_io.decode_archive_uri("archive://")
    with pytest.raises(ValueError, match="outer path and at least one entry"):
        archive_io.decode_archive_uri("archive://outer-only")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("", False),
        ("/absolute/file.dcm", False),
        ("\\absolute\\file.dcm", False),
        ("C:/absolute/file.dcm", False),
        ("patient/../file.dcm", False),
        ("./patient/study/file.dcm", True),
    ],
)
def test_archive_member_path_safety(name, expected):
    assert archive_io._is_safe_member_name(name) is expected


def test_open_archive_detects_zip_and_tar_without_extensions(tmp_path):
    zip_path = tmp_path / "zip_payload.bin"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("image.dcm", b"zip payload")

    kind, container = archive_io._open_archive(zip_path, zip_path.name)
    try:
        assert kind == "zip"
        assert container.read("image.dcm") == b"zip payload"
    finally:
        container.close()

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tf:
        payload = b"tar payload"
        info = tarfile.TarInfo("image.dcm")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    kind, container = archive_io._open_archive(tar_buffer.getvalue(), "memory_archive")
    try:
        assert kind == "tar"
        assert container.getmember("image.dcm").size == len(b"tar payload")
    finally:
        container.close()


def test_open_archive_uses_named_byte_container_types():
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("image.dcm", b"zip")

    kind, container = archive_io._open_archive(zip_buffer.getvalue(), "data.zip")
    try:
        assert kind == "zip"
    finally:
        container.close()

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tf:
        info = tarfile.TarInfo("image.dcm")
        info.size = 0
        tf.addfile(info, io.BytesIO())

    kind, container = archive_io._open_archive(tar_buffer.getvalue(), "data.tar")
    try:
        assert kind == "tar"
    finally:
        container.close()


def test_archive_member_readers_report_missing_members(tmp_path):
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("folder/", b"")
        zf.writestr("folder/image.dcm", b"zip")
    with zipfile.ZipFile(zip_path) as zf:
        with pytest.raises(KeyError, match="Zip member not found"):
            archive_io._read_zip_member(zf, "missing.dcm")

    tar_path = tmp_path / "data.tar"
    with tarfile.open(tar_path, "w") as tf:
        directory = tarfile.TarInfo("folder")
        directory.type = tarfile.DIRTYPE
        tf.addfile(directory)
        payload = b"tar"
        info = tarfile.TarInfo("folder/image.dcm")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    with tarfile.open(tar_path) as tf:
        assert archive_io._read_tar_member(tf, "folder/image.dcm") == b"tar"
        with pytest.raises(KeyError, match="Tar member not found"):
            archive_io._read_tar_member(tf, "missing.dcm")


def test_iter_archive_members_accepts_bytes_and_enforces_member_limit(tmp_path):
    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("a.dcm", b"a")
        zf.writestr("b.dcm", b"b")

    members = list(archive_io.iter_archive_members(archive_path.read_bytes()))
    assert [member["entry_chain"] for member in members] == [["a.dcm"], ["b.dcm"]]

    with pytest.raises(RuntimeError, match="member limit exceeded"):
        list(archive_io.iter_archive_members(archive_path, member_limit=1))


def test_iter_archive_members_skips_unsafe_and_directory_entries(tmp_path, caplog):
    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("folder/", b"")
        zf.writestr("../escape.dcm", b"bad")
        zf.writestr("folder/image.dcm", b"good")

    with caplog.at_level("WARNING", logger=archive_io.logger.name):
        members = list(archive_io.iter_archive_members(archive_path))

    assert [member["entry_chain"] for member in members] == [["folder/image.dcm"]]
    assert "skipping unsafe member name" in caplog.text


def test_discover_dicom_sources_scans_directories_and_plain_files(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    plain_dicom = root / "plain.dcm"
    plain_dicom.write_bytes(b"plain")
    archive_path = root / "packed.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("patient/archive.dcm", b"packed")

    records = archive_io.discover_dicom_sources([root, plain_dicom])

    assert len(records) == 2
    assert {record["relative_path"] for record in records} == {
        "plain.dcm",
        "patient/archive.dcm",
    }
    assert {record["is_archive_member"] for record in records} == {False, True}


def test_discover_dicom_sources_handles_missing_and_corrupt_archives(tmp_path, caplog):
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a zip")

    with caplog.at_level("WARNING", logger=archive_io.logger.name):
        records = archive_io.discover_dicom_sources([tmp_path / "missing", corrupt])

    assert records == []
    assert "unable to inspect archive" in caplog.text


def test_discover_dicom_sources_validates_extensionless_candidates(
    tmp_path, monkeypatch
):
    keep = tmp_path / "keep"
    reject = tmp_path / "reject"
    keep.write_bytes(b"dicom")
    reject.write_bytes(b"text")
    seen = []

    def validate(record, max_depth):
        seen.append((Path(record["source_uri_or_path"]).name, max_depth))
        return Path(record["source_uri_or_path"]).name == "keep"

    monkeypatch.setattr(archive_io, "_validate_record_as_dicom", validate)

    records = archive_io.discover_dicom_sources([reject, keep], max_depth=7)

    assert [Path(record["source_uri_or_path"]).name for record in records] == ["keep"]
    assert seen == [("keep", 7), ("reject", 7)]


def test_read_archive_member_bytes_validates_inputs(tmp_path):
    missing = tmp_path / "missing.zip"
    plain = tmp_path / "plain.bin"
    plain.write_bytes(b"plain")

    with pytest.raises(ValueError, match="cannot be empty"):
        archive_io.read_archive_member_bytes(missing, [])
    with pytest.raises(FileNotFoundError, match="Archive not found"):
        archive_io.read_archive_member_bytes(missing, ["image.dcm"])
    with pytest.raises(ValueError, match="Not an archive file"):
        archive_io.read_archive_member_bytes(plain, ["image.dcm"])


def test_read_archive_member_bytes_nested_success_and_failures(tmp_path):
    outer_zip, nested_name = _make_nested_zip_tar_with_dicom(tmp_path)
    leaf = "patientA/study1/series1/img1.dcm"

    assert (
        archive_io.read_archive_member_bytes(
            outer_zip, [nested_name, leaf], max_depth=3
        )
        == b"DICM_FAKE"
    )
    with pytest.raises(ValueError, match="Maximum nested archive depth"):
        archive_io.read_archive_member_bytes(
            outer_zip, [nested_name, leaf], max_depth=0
        )
    with pytest.raises(KeyError, match="Zip member not found"):
        archive_io.read_archive_member_bytes(outer_zip, ["missing.dcm"])

    simple_zip = tmp_path / "simple.zip"
    with zipfile.ZipFile(simple_zip, "w") as zf:
        zf.writestr("plain.bin", b"plain")
    with pytest.raises(ValueError, match="Intermediate member is not an archive"):
        archive_io.read_archive_member_bytes(simple_zip, ["plain.bin", "image.dcm"])


def test_archive_session_requires_context_and_caches_materialized_member(tmp_path):
    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("patient/image.dcm", b"payload")
    uri = archive_io.encode_archive_uri(archive_path, ["patient/image.dcm"])

    session = archive_io.ArchiveSession(cache_dir=tmp_path / "cache")
    assert session.materialize(tmp_path / "plain.dcm") == tmp_path / "plain.dcm"
    with pytest.raises(RuntimeError, match="context manager"):
        session.materialize(uri)

    with session:
        materialized = session.materialize(uri)
        assert materialized.read_bytes() == b"payload"
        assert session.materialize(uri) == materialized
        session_dir = session.session_dir

    assert session_dir is not None
    assert not session_dir.exists()


def test_archive_session_can_keep_materialized_cache(tmp_path):
    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("image.dcm", b"payload")
    uri = archive_io.encode_archive_uri(archive_path, ["image.dcm"])

    with archive_io.ArchiveSession(
        cache_dir=tmp_path / "cache", keep_cache=True
    ) as session:
        materialized = session.materialize(uri)

    assert materialized.exists()
