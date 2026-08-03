from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imperandi.process import registration


def test_registration_pair_table_accepts_external_fixed_template(tmp_path, monkeypatch):
    calls = []

    def fake_register_pair(**kwargs):
        calls.append(kwargs)
        return registration.RegistrationOutput(
            pair_id=kwargs["pair_id"],
            transform_path=Path(tmp_path / "pair.tfm"),
            registered_image_path=Path(tmp_path / "pair.nii.gz"),
            metric_value=0.5,
        )

    monkeypatch.setattr(registration, "register_pair", fake_register_pair)
    volumes = pd.DataFrame({"volume_id": ["moving"], "nifti_path": ["moving.nii.gz"]})
    pairs = pd.DataFrame(
        {
            "pair_id": ["template-1"],
            "fixed_nifti_path": ["template.nii.gz"],
            "moving_volume_id": ["moving"],
        }
    )

    outputs, errors = registration.register_pairs(
        pairs, volumes, output_dir=tmp_path, transform="rigid"
    )

    assert errors.empty
    assert calls[0]["fixed_path"] == "template.nii.gz"
    assert calls[0]["moving_path"] == "moving.nii.gz"
    assert outputs.loc[0, "moving_volume_id"] == "moving"


def test_registration_pair_failures_are_isolated_with_stable_empty_schema(
    tmp_path, monkeypatch
):
    def fail(**kwargs):
        raise RuntimeError(f"cannot register {kwargs['pair_id']}")

    monkeypatch.setattr(registration, "register_pair", fail)
    volumes = pd.DataFrame(
        {
            "volume_id": ["fixed", "moving"],
            "nifti_path": ["fixed.nii.gz", "moving.nii.gz"],
        }
    )
    pairs = pd.DataFrame({"fixed_volume_id": ["fixed"], "moving_volume_id": ["moving"]})

    outputs, errors = registration.register_pairs(
        pairs, volumes, output_dir=tmp_path, transform="rigid_affine"
    )

    assert outputs.empty
    assert "moving_volume_id" in outputs.columns
    assert len(errors) == 1
    assert "cannot register" in errors.loc[0, "error_message"]


def test_rigid_affine_registration_writes_transform_and_image(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    values = np.random.default_rng(42).normal(size=(32, 32, 32)).astype("float32")
    fixed = sitk.GetImageFromArray(values)
    moving = sitk.GetImageFromArray(values.copy())
    fixed_path = tmp_path / "fixed.nii.gz"
    moving_path = tmp_path / "moving.nii.gz"
    sitk.WriteImage(fixed, str(fixed_path))
    sitk.WriteImage(moving, str(moving_path))

    result = registration.register_pair(
        pair_id="pair",
        fixed_path=fixed_path,
        moving_path=moving_path,
        output_dir=tmp_path,
        transform="rigid_affine",
    )

    assert result.transform_path.exists()
    assert result.registered_image_path.exists()


def test_pair_id_is_sanitized_before_it_becomes_a_filename():
    stem = registration._safe_pair_stem("../../outside")

    assert "/" not in stem
    assert ".." not in stem
