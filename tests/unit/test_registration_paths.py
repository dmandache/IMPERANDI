"""Registration group folders retain identities and contain path-like values."""

from pathlib import Path

import pytest

from imperandi.process.registration.cohort import group_output_path


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
