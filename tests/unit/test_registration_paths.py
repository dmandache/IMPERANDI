"""Registration group folders retain identities and contain path-like values."""

from pathlib import Path

import pytest

from imperandi.process.registration.cohort import group_output_path, reset_group_output


@pytest.mark.parametrize(
    "value, expected",
    [
        ("patient001", "patient001"),
        ("001", "001"),
        ("patient/001", "patient%2F001"),
        ("patient\\001", "patient%5C001"),
        ("patient%2F001", "patient%252F001"),
        (".", "%2E"),
        ("..", "%2E%2E"),
        ("../patient001", "..%2Fpatient001"),
        ("/patient001", "%2Fpatient001"),
        ("C:\\patient001", "C%3A%5Cpatient001"),
    ],
)
def test_group_values_stay_in_distinct_directory_components(value, expected):
    path = group_output_path(
        {"patient_key": value, "exam_stage": "BASELINE"},
        ["patient_key", "exam_stage"],
    )

    assert path == Path(expected) / "BASELINE"
    assert not path.is_absolute()
    assert len(path.parts) == 2


def test_modality_folder_matches_normalized_group_identity():
    path = group_output_path(
        {"patient_key": " 001 ", "Modality": " mri "},
        ["patient_key", "Modality"],
    )

    assert path == Path("001") / "MR"


def test_reset_removes_nested_outputs_and_keeps_other_exams(tmp_path, caplog):
    directory = tmp_path / "patient001" / "BASELINE"
    stale = directory / "old-run" / "removed-scan" / "old.nii.gz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old output")
    sibling = directory.with_name("FOLLOWUP") / "keep.nii.gz"
    sibling.parent.mkdir()
    sibling.write_bytes(b"other exam")

    reset_group_output(
        directory, tmp_path, [], "patient_key=patient001, exam_stage=BASELINE"
    )

    assert directory.is_dir()
    assert not list(directory.iterdir())
    assert sibling.read_bytes() == b"other exam"
    assert "Overwriting registration outputs" in caplog.text


def test_reset_preserves_source_inputs_inside_output_group(tmp_path):
    directory = tmp_path / "patient001" / "BASELINE"
    directory.mkdir(parents=True)
    source = directory / "source.nii.gz"
    source.write_bytes(b"native input")

    with pytest.raises(ValueError, match="source inputs"):
        reset_group_output(directory, tmp_path, [str(source)], "baseline")

    assert source.read_bytes() == b"native input"


@pytest.mark.parametrize("through_parent", [False, True])
@pytest.mark.parametrize("within_root", [False, True])
def test_reset_refuses_symlink_output_folders(tmp_path, through_parent, within_root):
    root = tmp_path / "outputs"
    root.mkdir()
    target = (root if within_root else tmp_path) / "elsewhere"
    target.mkdir()
    protected = target / "keep.nii.gz"
    protected.write_bytes(b"keep")
    link = root / "patient001"
    link.symlink_to(target, target_is_directory=True)
    directory = link / "BASELINE" if through_parent else link

    with pytest.raises(ValueError, match="output root|symlink"):
        reset_group_output(directory, root, [], "baseline")

    assert protected.read_bytes() == b"keep"


def test_reset_refuses_the_output_root(tmp_path):
    protected = tmp_path / "keep.nii.gz"
    protected.write_bytes(b"keep")

    with pytest.raises(ValueError, match="output root"):
        reset_group_output(tmp_path, tmp_path, [], "baseline")

    assert protected.read_bytes() == b"keep"
