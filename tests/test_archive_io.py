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
