import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import logging

import numpy as np
import pandas as pd
import pytest

from imperandi.process import _registration_common as reg_common
from imperandi.process import register_intra_patient as intra_module
from imperandi.process import register_population as population_module
from imperandi.process.registration import (
    load_rigid_transform_from_metadata,
    save_transform_artifacts,
)


def test_normalize_register_population_args_defaults(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path,mask_liver\n")

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=None,
        csv_path_out=None,
        output_dir=str(tmp_path / "registered"),
        error_csv_path=None,
        log_csv_path=None,
        organ="liver",
        mask_column=None,
        template_sample_size=128,
        template_mode="mean_shape",
        template_source_idx=None,
        principal_vectors=None,
        template_seed=0,
        num_workers=2,
        pad_mm=25.0,
        save_registered_outputs=False,
        verbose=False,
        dry_run=False,
    )

    out = population_module.normalize_register_population_args(args)

    assert out.csv_path == str(csv_path.resolve())
    assert out.csv_path_out == str(
        csv_path.parent / "nifti_index_registered_population.csv"
    )
    assert out.error_csv_path == str(
        csv_path.parent / "register_population_errors.csv"
    )
    assert out.log_csv_path == str(csv_path.parent / "register_population_log.csv")
    assert out.mask_column == "mask_liver"
    assert out.template_mode == "mean_shape"
    assert out.template_source_idx is None
    assert out.principal_vectors is None


def test_normalize_register_population_args_parses_principal_vectors(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path,mask_liver\n")

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=None,
        csv_path_out=None,
        output_dir=str(tmp_path / "registered"),
        error_csv_path=None,
        log_csv_path=None,
        organ="liver",
        mask_column=None,
        template_sample_size=8,
        template_mode="principal_vectors",
        template_source_idx=None,
        principal_vectors="1,0,0,0,1,0,0,0,1",
        template_seed=0,
        num_workers=2,
        pad_mm=25.0,
        save_registered_outputs=False,
        verbose=False,
        dry_run=False,
    )

    out = population_module.normalize_register_population_args(args)

    assert out.template_mode == "principal_vectors"
    assert out.principal_vectors == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_normalize_register_population_args_enables_save_when_normalize(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path,mask_liver\n")

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=None,
        csv_path_out=None,
        output_dir=str(tmp_path / "registered"),
        error_csv_path=None,
        log_csv_path=None,
        organ="liver",
        mask_column=None,
        template_sample_size=8,
        template_mode="mean_shape",
        template_source_idx=None,
        principal_vectors=None,
        template_seed=0,
        num_workers=2,
        pad_mm=25.0,
        save_registered_outputs=False,
        normalize_registered_outputs=True,
        normalize_crop_mode="margin",
        normalize_margin_mm=10.0,
        normalize_without_background=False,
        normalize_spacing="1.5,1.5,1.5",
        normalize_orientation="LPS",
        disable_normalize_center_organ=False,
        verbose=False,
        dry_run=False,
    )

    out = population_module.normalize_register_population_args(args)

    assert out.normalize_registered_outputs is True
    assert out.save_registered_outputs is True
    assert out.normalize_spacing == (1.5, 1.5, 1.5)


def test_normalize_register_population_args_rejects_incompatible_template_source_idx(
    tmp_path,
):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path,mask_liver\n")

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=None,
        csv_path_out=None,
        output_dir=str(tmp_path / "registered"),
        error_csv_path=None,
        log_csv_path=None,
        organ="liver",
        mask_column=None,
        template_sample_size=8,
        template_mode="mean_shape",
        template_source_idx=0,
        principal_vectors=None,
        template_seed=0,
        num_workers=2,
        pad_mm=25.0,
        save_registered_outputs=False,
        verbose=False,
        dry_run=False,
    )

    with pytest.raises(ValueError) as exc_info:
        population_module.normalize_register_population_args(args)
    assert "--template_source_idx is only supported" in str(exc_info.value)


def test_default_principal_vectors_for_liver():
    out = population_module._default_principal_vectors_for_organ("liver")
    assert out is not None
    expected = population_module._canonicalize_eigen_axes(
        np.asarray(
            population_module.DEFAULT_ORGAN_PRINCIPAL_VECTORS["liver"],
            dtype=float,
        )
    )
    np.testing.assert_allclose(out, expected)


def test_principal_frame_transform_aligns_permuted_axes():
    class FakeTx:
        def __init__(self):
            self.matrix = None
            self.translation = None

        def SetMatrix(self, matrix):
            self.matrix = matrix

        def SetTranslation(self, translation):
            self.translation = translation

    class FakeSitk:
        def AffineTransform(self, dim):
            assert dim == 3
            return FakeTx()

    tx = population_module._principal_frame_transform(
        moving_frame={
            "centroid_mm": [0.0, 0.0, 0.0],
            "axes": [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
        },
        reference_frame={
            "centroid_mm": [0.0, 0.0, 0.0],
            "axes": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
        sitk_module=FakeSitk(),
    )

    np.testing.assert_allclose(np.asarray(tx.matrix, dtype=float).reshape(3, 3), np.eye(3))
    np.testing.assert_allclose(np.asarray(tx.translation, dtype=float), np.zeros(3))


def test_rigid_pose_quaternion_roundtrip():
    class FakeTransform:
        def __init__(self, matrix, translation):
            self.matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
            self.translation = np.asarray(translation, dtype=float).reshape(3)

        def TransformPoint(self, point):
            return tuple(self.matrix @ np.asarray(point, dtype=float) + self.translation)

    class FakeAffine:
        def __init__(self):
            self.matrix = None
            self.translation = None

        def SetMatrix(self, matrix):
            self.matrix = list(matrix)

        def SetTranslation(self, translation):
            self.translation = list(translation)

    class FakeSitk:
        def AffineTransform(self, dim):
            assert dim == 3
            return FakeAffine()

    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    translation = np.asarray([12.5, -3.0, 4.25], dtype=float)
    flat_pose = reg_common.transform_to_flat_quaternion_translation(
        FakeTransform(rotation, translation)
    )

    assert flat_pose["population_tx_qx"] == pytest.approx(0.0)
    assert flat_pose["population_tx_qy"] == pytest.approx(0.0)
    assert flat_pose["population_tx_qz"] == pytest.approx(np.sqrt(0.5))
    assert flat_pose["population_tx_qw"] == pytest.approx(np.sqrt(0.5))
    assert flat_pose["population_tx_t0"] == pytest.approx(12.5)
    assert flat_pose["population_tx_t1"] == pytest.approx(-3.0)
    assert flat_pose["population_tx_t2"] == pytest.approx(4.25)

    loaded = reg_common.rigid_transform_from_population_pose(
        flat_pose,
        sitk_module=FakeSitk(),
    )

    np.testing.assert_allclose(
        np.asarray(loaded.matrix, dtype=float).reshape(3, 3),
        rotation,
    )
    np.testing.assert_allclose(np.asarray(loaded.translation, dtype=float), translation)


def test_save_transform_artifacts_persists_rigid_pose_as_quaternion(tmp_path):
    class FakeTransform:
        def __init__(self, matrix, translation):
            self.matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
            self.translation = np.asarray(translation, dtype=float).reshape(3)

        def TransformPoint(self, point):
            return tuple(self.matrix @ np.asarray(point, dtype=float) + self.translation)

    class FakeAffine:
        def __init__(self):
            self.matrix = None
            self.translation = None

        def SetMatrix(self, matrix):
            self.matrix = list(matrix)

        def SetTranslation(self, translation):
            self.translation = list(translation)

    class FakeSitk:
        @staticmethod
        def WriteTransform(_transform, path):
            Path(path).write_text("tfm", encoding="utf-8")

        def AffineTransform(self, dim):
            assert dim == 3
            return FakeAffine()

    rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    translation = np.asarray([1.0, 2.0, 3.0], dtype=float)
    artifact = save_transform_artifacts(
        row_dir=tmp_path,
        transform=FakeTransform(rotation, translation),
        sitk_module=FakeSitk(),
        prefix="rigid_pose",
        metadata={"stage": "rigid"},
    )

    payload = json.loads(Path(artifact.metadata_path).read_text(encoding="utf-8"))
    assert "rigid_pose" in payload
    assert "matrix_3x4" not in payload
    assert payload["rigid_pose"]["translation_xyz"] == [1.0, 2.0, 3.0]

    loaded = load_rigid_transform_from_metadata(
        artifact.metadata_path,
        sitk_module=FakeSitk(),
    )

    np.testing.assert_allclose(
        np.asarray(loaded.matrix, dtype=float).reshape(3, 3),
        rotation,
    )
    np.testing.assert_allclose(np.asarray(loaded.translation, dtype=float), translation)


def test_migrate_legacy_population_pose_columns_converts_rotation_matrix():
    class FakeAffine:
        def __init__(self):
            self.matrix = None
            self.translation = None

        def SetMatrix(self, matrix):
            self.matrix = list(matrix)

        def SetTranslation(self, translation):
            self.translation = list(translation)

        def TransformPoint(self, point):
            matrix = np.asarray(self.matrix, dtype=float).reshape(3, 3)
            translation = np.asarray(self.translation, dtype=float)
            return tuple(matrix @ np.asarray(point, dtype=float) + translation)

    class FakeSitk:
        def AffineTransform(self, dim):
            assert dim == 3
            return FakeAffine()

    df = pd.DataFrame(
        [
            {
                "nifti_path": "scan.nii.gz",
                "mask_liver": "mask.nii.gz",
                "population_tx_r00": 0.0,
                "population_tx_r01": -1.0,
                "population_tx_r02": 0.0,
                "population_tx_r10": 1.0,
                "population_tx_r11": 0.0,
                "population_tx_r12": 0.0,
                "population_tx_r20": 0.0,
                "population_tx_r21": 0.0,
                "population_tx_r22": 1.0,
                "population_tx_t0": 7.0,
                "population_tx_t1": 8.0,
                "population_tx_t2": 9.0,
            }
        ]
    )

    out = population_module._migrate_legacy_population_pose_columns(
        df,
        sitk_module=FakeSitk(),
    )

    assert "population_tx_r00" not in out.columns
    assert out.loc[0, "population_tx_qx"] == pytest.approx(0.0)
    assert out.loc[0, "population_tx_qy"] == pytest.approx(0.0)
    assert out.loc[0, "population_tx_qz"] == pytest.approx(np.sqrt(0.5))
    assert out.loc[0, "population_tx_qw"] == pytest.approx(np.sqrt(0.5))
    assert out.loc[0, "population_tx_t0"] == pytest.approx(7.0)
    assert out.loc[0, "population_tx_t1"] == pytest.approx(8.0)
    assert out.loc[0, "population_tx_t2"] == pytest.approx(9.0)


def test_project_to_rotation_preserves_existing_rotation():
    rotation = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    out = population_module._project_to_rotation(rotation)
    np.testing.assert_allclose(out, rotation)


def test_match_eigenvector_basis_keeps_positive_axis_alignment():
    reference = np.asarray(
        [
            [0.80193, 0.592863, 0.07363],
            [0.342035, -0.556673, 0.757052],
            [0.489816, -0.581918, -0.649193],
        ],
        dtype=float,
    )
    moving = np.asarray(
        [
            [-0.585992, 0.807823, 0.063527],
            [0.558565, 0.345895, 0.753898],
            [0.587042, 0.477262, -0.653913],
        ],
        dtype=float,
    )
    matched = population_module._match_eigenvector_basis(reference, moving)
    diagonal = np.diag(reference.T @ matched)
    assert np.all(diagonal > 0.0)


def test_build_principal_vectors_template_uses_default_liver_vectors(
    tmp_path, monkeypatch
):
    sample_rows = [
        {
            "source_idx": 0,
            "nifti_path": str(tmp_path / "scan.nii.gz"),
            "mask_path": str(tmp_path / "mask.nii.gz"),
        }
    ]
    exemplar = sample_rows[0]

    monkeypatch.setattr(
        population_module,
        "_sampled_rows_or_error",
        lambda *args, **kwargs: sample_rows,
    )
    monkeypatch.setattr(population_module.reg_common, "choose_median_exemplar", lambda rows: exemplar)
    monkeypatch.setattr(population_module.reg_common, "read_image", lambda *args, **kwargs: "img")
    monkeypatch.setattr(
        population_module.reg_common,
        "read_binary_mask",
        lambda *args, **kwargs: "mask",
    )
    monkeypatch.setattr(population_module.reg_common, "write_image", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        population_module,
        "_mask_principal_frame",
        lambda *args, **kwargs: {
            "centroid_mm": np.asarray([0.0, 0.0, 0.0]),
            "axes": np.eye(3),
        },
    )
    monkeypatch.setattr(
        population_module,
        "_compute_average_shape_principal_frame",
        lambda *args, **kwargs: (
            {"centroid_mm": np.asarray([1.0, 2.0, 3.0]), "axes": np.eye(3)},
            1,
        ),
    )

    args = argparse.Namespace(
        output_dir=str(tmp_path / "registered"),
        mask_column="mask_liver",
        principal_vectors=None,
        organ="liver",
        template_seed=0,
        template_sample_size=1,
    )
    out = population_module._build_principal_vectors_template(
        pd.DataFrame([{"_source_idx": 0}]),
        args=args,
        sitk_module=object(),
    )

    expected_axes = population_module._canonicalize_eigen_axes(
        np.asarray(
            population_module.DEFAULT_ORGAN_PRINCIPAL_VECTORS["liver"],
            dtype=float,
        )
    )
    np.testing.assert_allclose(
        np.asarray(out["principal_reference"]["axes"], dtype=float),
        expected_axes,
    )
    payload = json.loads(Path(out["principal_vectors_path"]).read_text(encoding="utf-8"))
    assert payload["source"] == "default_liver"


def test_normalize_register_intra_patient_args_defaults(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("patient_key,nifti_path,mask_liver\n")

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=None,
        csv_path_out=None,
        output_dir=str(tmp_path / "registered"),
        error_csv_path=None,
        log_csv_path=None,
        organ="liver",
        mask_column=None,
        num_workers=2,
        pad_mm=25.0,
        band_mm=15.0,
        bspline_ctrl_spacing_mm=90.0,
        verbose=False,
        dry_run=False,
    )

    out = intra_module.normalize_register_intra_patient_args(args)

    assert out.csv_path == str(csv_path.resolve())
    assert out.csv_path_out == str(
        csv_path.parent / "nifti_index_registered_intra_patient.csv"
    )
    assert out.error_csv_path == str(
        csv_path.parent / "register_intra_patient_errors.csv"
    )
    assert out.log_csv_path == str(
        csv_path.parent / "register_intra_patient_log.csv"
    )
    assert out.mask_column == "mask_liver"


def test_sample_valid_rows_for_template_is_deterministic(tmp_path, monkeypatch):
    rows = []
    metrics_by_path = {}
    for idx in range(4):
        nifti = tmp_path / f"case_{idx}.nii.gz"
        mask = tmp_path / f"case_{idx}_mask.nii.gz"
        nifti.write_text("nifti")
        mask.write_text("mask")
        rows.append({"_source_idx": idx, "nifti_path": str(nifti), "mask_liver": str(mask)})
        metrics_by_path[str(mask)] = {
            "volume_ml": 100.0 + idx,
            "bbox_x_mm": 10.0 + idx,
            "bbox_y_mm": 20.0 + idx,
            "bbox_z_mm": 30.0 + idx,
        }

    monkeypatch.setattr(
        reg_common,
        "mask_metrics",
        lambda path, *, sitk_module: dict(metrics_by_path[path]),
    )

    df = pd.DataFrame(rows)
    sampled_a = reg_common.sample_valid_rows_for_template(
        df,
        mask_column="mask_liver",
        sample_size=3,
        seed=7,
        sitk_module=object(),
    )
    sampled_b = reg_common.sample_valid_rows_for_template(
        df,
        mask_column="mask_liver",
        sample_size=3,
        seed=7,
        sitk_module=object(),
    )

    assert [row["source_idx"] for row in sampled_a] == [
        row["source_idx"] for row in sampled_b
    ]


def test_compute_mean_metrics():
    metrics = [
        {
            "volume_ml": 100.0,
            "bbox_x_mm": 10.0,
            "bbox_y_mm": 20.0,
            "bbox_z_mm": 30.0,
        },
        {
            "volume_ml": 300.0,
            "bbox_x_mm": 30.0,
            "bbox_y_mm": 40.0,
            "bbox_z_mm": 50.0,
        },
    ]

    out = reg_common.compute_mean_metrics(metrics)

    assert out == {
        "volume_ml": 200.0,
        "bbox_x_mm": 20.0,
        "bbox_y_mm": 30.0,
        "bbox_z_mm": 40.0,
    }


def test_build_output_path_uses_scan_and_prefixless_mask_names(tmp_path):
    row_dir = tmp_path / "rows" / "0"

    nifti_out = reg_common.build_output_path(
        row_dir,
        column_name="nifti_path",
        source_path="scan_source.nii.gz",
    )
    liver_out = reg_common.build_output_path(
        row_dir,
        column_name="mask_liver",
        source_path="mask_source.nii.gz",
    )
    tumor_out = reg_common.build_output_path(
        row_dir,
        column_name="mask_liver_tumor",
        source_path="mask_source.nii.gz",
    )
    passthrough_out = reg_common.build_output_path(
        row_dir,
        column_name="nifti_preview",
        source_path="preview.nii.gz",
    )

    assert nifti_out == row_dir / "scan.nii.gz"
    assert liver_out == row_dir / "liver.nii.gz"
    assert tumor_out == row_dir / "liver_tumor.nii.gz"
    assert passthrough_out == row_dir / "nifti_preview.nii.gz"


def test_select_anchor_row_prefers_portal_phase(tmp_path):
    nifti_a = tmp_path / "a.nii.gz"
    nifti_b = tmp_path / "b.nii.gz"
    mask_a = tmp_path / "a_mask.nii.gz"
    mask_b = tmp_path / "b_mask.nii.gz"
    for path in (nifti_a, nifti_b, mask_a, mask_b):
        path.write_text("x")

    df = pd.DataFrame(
        [
            {
                "_source_idx": 0,
                "patient_key": "p1",
                "phase": "arteriel",
                "visit_order": 0,
                "nifti_path": str(nifti_a),
                "mask_liver": str(mask_a),
            },
            {
                "_source_idx": 1,
                "patient_key": "p1",
                "phase": "portal",
                "visit_order": 1,
                "nifti_path": str(nifti_b),
                "mask_liver": str(mask_b),
            },
        ]
    )

    selected = intra_module._select_anchor_row(df, mask_column="mask_liver")

    assert selected["_source_idx"] == 1


def test_select_anchor_row_falls_back_to_totalseg_phase(tmp_path):
    nifti_a = tmp_path / "a.nii.gz"
    nifti_b = tmp_path / "b.nii.gz"
    mask_a = tmp_path / "a_mask.nii.gz"
    mask_b = tmp_path / "b_mask.nii.gz"
    for path in (nifti_a, nifti_b, mask_a, mask_b):
        path.write_text("x")

    df = pd.DataFrame(
        [
            {
                "_source_idx": 0,
                "patient_key": "p1",
                "phase": None,
                "totalseg_phase": "arterial_late",
                "visit_order": 0,
                "nifti_path": str(nifti_a),
                "mask_liver": str(mask_a),
            },
            {
                "_source_idx": 1,
                "patient_key": "p1",
                "phase": None,
                "totalseg_phase": "portal_venous",
                "visit_order": 1,
                "nifti_path": str(nifti_b),
                "mask_liver": str(mask_b),
            },
        ]
    )

    selected = intra_module._select_anchor_row(df, mask_column="mask_liver")

    assert selected["_source_idx"] == 1


def test_process_patient_group_returns_anchor_error_when_anchor_inputs_unreadable(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "a.nii.gz"
    mask = tmp_path / "a_mask.nii.gz"
    for path in (nifti, mask):
        path.write_text("x")

    df = pd.DataFrame(
        [
            {
                "_source_idx": 0,
                "patient_key": "p1",
                "phase": "portal",
                "nifti_path": str(nifti),
                "mask_liver": str(mask),
            }
        ]
    )

    monkeypatch.setattr(intra_module.reg_common, "_load_register_dependencies", lambda: object())
    monkeypatch.setattr(
        intra_module.reg_common,
        "copy_row_files",
        lambda *args, **kwargs: {"nifti_path": "copied_scan.nii.gz", "mask_liver": "copied_mask.nii.gz"},
    )
    monkeypatch.setattr(
        intra_module.reg_common,
        "read_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad anchor image")),
    )

    args = argparse.Namespace(
        output_dir=str(tmp_path / "registered"),
        mask_column="mask_liver",
        band_mm=15.0,
        bspline_ctrl_spacing_mm=90.0,
    )
    out = intra_module._process_patient_group(
        df,
        pending_source_indices={0},
        args=args,
        path_columns=["nifti_path", "mask_liver"],
    )

    assert len(out) == 1
    assert out[0]["source_idx"] == 0
    assert out[0]["updates"]["intra_register_status"] == "error"
    assert out[0]["updates"]["intra_register_stage"] == "anchor"
    assert "anchor inputs are not readable" in out[0]["error_message"]


def test_process_patient_group_debug_logging_keeps_only_milestones(
    tmp_path, monkeypatch, caplog
):
    nifti_a = tmp_path / "a.nii.gz"
    mask_a = tmp_path / "a_mask.nii.gz"
    nifti_b = tmp_path / "b.nii.gz"
    mask_b = tmp_path / "b_mask.nii.gz"
    for path in (nifti_a, mask_a, nifti_b, mask_b):
        path.write_text("x")

    df = pd.DataFrame(
        [
            {
                "_source_idx": 0,
                "patient_key": "p1",
                "phase": "portal",
                "visit_order": 0,
                "nifti_path": str(nifti_a),
                "mask_liver": str(mask_a),
            },
            {
                "_source_idx": 1,
                "patient_key": "p1",
                "phase": "arteriel",
                "visit_order": 1,
                "nifti_path": str(nifti_b),
                "mask_liver": str(mask_b),
            },
        ]
    )

    monkeypatch.setattr(intra_module.reg_common, "_load_register_dependencies", lambda: object())
    monkeypatch.setattr(intra_module.reg_common, "identity_transform", lambda *_args, **_kwargs: "identity")
    monkeypatch.setattr(
        intra_module.reg_common,
        "copy_row_files",
        lambda row, *, row_dir, path_columns: {
            column: str(
                reg_common.build_output_path(
                    row_dir,
                    column_name=column,
                    source_path=row.get(column),
                )
            )
            for column in path_columns
            if row.get(column)
        },
    )
    monkeypatch.setattr(intra_module.reg_common, "read_image", lambda *args, **kwargs: "image")
    monkeypatch.setattr(intra_module.reg_common, "read_binary_mask", lambda *args, **kwargs: "mask")
    monkeypatch.setattr(intra_module.reg_common, "rigid_register_mask_pair", lambda *args, **kwargs: "rigid")
    monkeypatch.setattr(
        intra_module.reg_common,
        "bspline_register_mask_pair",
        lambda *args, **kwargs: "elastic",
    )

    def fake_dice(*args, **kwargs):
        tx = kwargs.get("tx")
        if tx == "rigid":
            return 0.8
        if tx == "elastic":
            return 0.9
        return 0.5

    monkeypatch.setattr(intra_module.reg_common, "dice_coeff", fake_dice)
    monkeypatch.setattr(
        intra_module.reg_common,
        "warp_row_files",
        lambda row, *, row_dir, path_columns, **kwargs: {
            column: str(
                reg_common.build_output_path(
                    row_dir,
                    column_name=column,
                    source_path=row.get(column),
                )
            )
            for column in path_columns
            if row.get(column)
        },
    )
    monkeypatch.setattr(
        intra_module,
        "save_transform_artifacts",
        lambda **kwargs: argparse.Namespace(
            transform_path="transform.tfm",
            metadata_path="transform.json",
        ),
    )

    args = argparse.Namespace(
        output_dir=str(tmp_path / "registered"),
        mask_column="mask_liver",
        band_mm=15.0,
        bspline_ctrl_spacing_mm=90.0,
        intra_mode="longitudinal",
    )

    with caplog.at_level(logging.DEBUG):
        out = intra_module._process_patient_group(
            df,
            pending_source_indices={0, 1},
            args=args,
            path_columns=["nifti_path", "mask_liver"],
        )

    debug_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG]

    assert len(out) == 2
    assert sum("Planned intra-patient work" in msg for msg in debug_messages) == 1
    assert not any("Starting intra-patient task processing" in msg for msg in debug_messages)
    assert not any("Intra task builder resolved" in msg for msg in debug_messages)
    assert not any("Running intra task" in msg for msg in debug_messages)
    assert not any("Intra elastic stage" in msg for msg in debug_messages)
    assert not any("cached_paths" in msg for msg in debug_messages)
    anchor_message = next(
        msg for msg in debug_messages if "Prepared intra anchor" in msg
    )
    assert "row_dir=" in anchor_message
    assert any("Completed intra task" in msg for msg in debug_messages)


def test_register_intra_patient_main_keeps_state_resumable_when_rows_unprocessed(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "a.nii.gz"
    mask = tmp_path / "a_mask.nii.gz"
    for path in (nifti, mask):
        path.write_text("x")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [{"patient_key": "p1", "phase": "portal", "nifti_path": str(nifti), "mask_liver": str(mask)}]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(intra_module.reg_common, "_load_register_dependencies", lambda: object())
    calls = {"groups": 0}

    def fake_process_group(patient_df, *, pending_source_indices, args, path_columns):
        calls["groups"] += 1
        return []

    monkeypatch.setattr(intra_module, "_process_patient_group", fake_process_group)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        output_dir=str(tmp_path / "registered"),
        error_csv_path=str(tmp_path / "errors.csv"),
        log_csv_path=str(tmp_path / "log.csv"),
        organ="liver",
        mask_column="mask_liver",
        num_workers=1,
        pad_mm=25.0,
        band_mm=15.0,
        bspline_ctrl_spacing_mm=90.0,
        verbose=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )

    intra_module.main(args)
    assert calls["groups"] == 1

    state_path = Path(args.csv_path_out).parent / ".out.register_intra_patient.state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert not bool(state.get("finished"))
    assert state.get("completed_indices") == []

    args.resume = True
    intra_module.main(args)
    assert calls["groups"] == 2


def test_register_population_main_keeps_paths_unchanged_without_save(
    tmp_path, monkeypatch
):
    nifti_a = tmp_path / "a.nii.gz"
    mask_a = tmp_path / "a_mask.nii.gz"
    nifti_b = tmp_path / "b.nii.gz"
    mask_b = tmp_path / "b_mask.nii.gz"
    tumor_a = tmp_path / "a_tumor.nii.gz"
    for path in (nifti_a, mask_a, nifti_b, mask_b, tumor_a):
        path.write_text("x")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [
            {
                "patient_key": "p1",
                "nifti_path": str(nifti_a),
                "mask_liver": str(mask_a),
                "mask_liver_tumor": str(tumor_a),
            },
            {
                "patient_key": "p2",
                "nifti_path": str(nifti_b),
                "mask_liver": str(mask_b),
                "mask_liver_tumor": "",
            },
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(population_module.reg_common, "_load_register_dependencies", lambda: object())

    def fake_build_template(df, *, args, sitk_module):
        template_dir = Path(args.output_dir) / "template"
        template_dir.mkdir(parents=True, exist_ok=True)
        ref_path = template_dir / "template_reference.nii.gz"
        mask_path = template_dir / f"{args.mask_column}.nii.gz"
        ref_path.write_text("ref")
        mask_path.write_text("mask")
        return {
            "template_source_idx": 0,
            "reference_image_path": str(ref_path),
            "mask_path": str(mask_path),
        }

    def fake_register_row(row, *, template_info, args, path_columns):
        source_idx = int(row["_source_idx"])
        base = {
            "population_register_template_source_idx": 0,
            "population_register_mask_column": args.mask_column,
            "population_register_stage": "rigid",
        }
        if source_idx == 1:
            return {
                "source_idx": source_idx,
                "updates": {
                    **base,
                    "population_register_status": "error",
                    "population_register_error_message": "bad mask",
                },
                "error_message": "bad mask",
            }
        return {
            "source_idx": source_idx,
            "updates": {
                **base,
                "population_register_status": "ok",
                "population_register_error_message": None,
                "population_register_dice_before": 0.5,
                "population_register_dice_after": 0.9,
                **{
                    column: float(index)
                    for index, column in enumerate(reg_common.POPULATION_TRANSFORM_COLUMNS)
                },
            },
            "error_message": None,
        }

    monkeypatch.setattr(population_module, "_build_population_template", fake_build_template)
    monkeypatch.setattr(population_module, "_register_population_row", fake_register_row)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        output_dir=str(tmp_path / "registered"),
        error_csv_path=str(tmp_path / "errors.csv"),
        log_csv_path=str(tmp_path / "log.csv"),
        organ="liver",
        mask_column="mask_liver",
        template_sample_size=8,
        template_seed=0,
        num_workers=1,
        pad_mm=25.0,
        save_registered_outputs=False,
        verbose=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )

    population_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert out_df.loc[0, "nifti_path"] == str(nifti_a)
    assert out_df.loc[0, "mask_liver"] == str(mask_a)
    assert out_df.loc[0, "population_tx_qx"] == 0.0
    assert out_df.loc[0, "population_register_status"] == "ok"
    assert out_df.loc[1, "nifti_path"] == str(nifti_b)
    assert out_df.loc[1, "population_register_status"] == "error"

    err_df = pd.read_csv(args.error_csv_path)
    assert len(err_df) == 1
    assert err_df.loc[0, "error_message"] == "bad mask"

    log_df = pd.read_csv(args.log_csv_path)
    assert sorted(log_df["population_register_status"].tolist()) == ["error", "ok"]
    assert Path(args.output_dir, "template", "template_reference.nii.gz").exists()


def test_register_population_main_rewrites_paths_when_save_enabled(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "a.nii.gz"
    mask = tmp_path / "a_mask.nii.gz"
    tumor = tmp_path / "a_tumor.nii.gz"
    for path in (nifti, mask, tumor):
        path.write_text("x")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [
            {
                "patient_key": "p1",
                "nifti_path": str(nifti),
                "mask_liver": str(mask),
                "mask_liver_tumor": str(tumor),
            }
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(population_module.reg_common, "_load_register_dependencies", lambda: object())

    def fake_build_template(df, *, args, sitk_module):
        template_dir = Path(args.output_dir) / "template"
        template_dir.mkdir(parents=True, exist_ok=True)
        ref_path = template_dir / "template_reference.nii.gz"
        mask_path = template_dir / f"{args.mask_column}.nii.gz"
        ref_path.write_text("ref")
        mask_path.write_text("mask")
        return {
            "template_source_idx": 0,
            "reference_image_path": str(ref_path),
            "mask_path": str(mask_path),
        }

    def fake_register_row(row, *, template_info, args, path_columns):
        row_dir = Path(args.output_dir) / "rows" / str(int(row["_source_idx"]))
        row_dir.mkdir(parents=True, exist_ok=True)
        rewritten = {}
        for column in ("nifti_path", "mask_liver", "mask_liver_tumor"):
            out_path = reg_common.build_output_path(
                row_dir,
                column_name=column,
                source_path=str(row[column]),
            )
            out_path.write_text(column)
            rewritten[column] = str(out_path)
        return {
            "source_idx": int(row["_source_idx"]),
            "updates": {
                **rewritten,
                "population_register_template_source_idx": 0,
                "population_register_mask_column": args.mask_column,
                "population_register_stage": "rigid",
                "population_register_status": "ok",
                "population_register_error_message": None,
                "population_register_dice_before": 0.5,
                "population_register_dice_after": 0.9,
                **{column: 1.0 for column in reg_common.POPULATION_TRANSFORM_COLUMNS},
            },
            "error_message": None,
        }

    monkeypatch.setattr(population_module, "_build_population_template", fake_build_template)
    monkeypatch.setattr(population_module, "_register_population_row", fake_register_row)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        output_dir=str(tmp_path / "registered"),
        error_csv_path=str(tmp_path / "errors.csv"),
        log_csv_path=str(tmp_path / "log.csv"),
        organ="liver",
        mask_column="mask_liver",
        template_sample_size=8,
        template_seed=0,
        num_workers=1,
        pad_mm=25.0,
        save_registered_outputs=True,
        verbose=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )

    population_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert Path(out_df.loc[0, "nifti_path"]).name == "scan.nii.gz"
    assert Path(out_df.loc[0, "mask_liver"]).name == "liver.nii.gz"
    assert Path(out_df.loc[0, "mask_liver_tumor"]).name == "liver_tumor.nii.gz"
    assert Path(out_df.loc[0, "nifti_path"]).exists()


def test_register_population_main_resume_skips_completed_rows(tmp_path, monkeypatch):
    nifti = tmp_path / "a.nii.gz"
    mask = tmp_path / "a_mask.nii.gz"
    for path in (nifti, mask):
        path.write_text("x")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "mask_liver": str(mask)}]).to_csv(
        csv_path, index=False
    )

    monkeypatch.setattr(population_module.reg_common, "_load_register_dependencies", lambda: object())

    counts = {"template": 0, "rows": 0}

    def fake_build_template(df, *, args, sitk_module):
        counts["template"] += 1
        template_dir = Path(args.output_dir) / "template"
        template_dir.mkdir(parents=True, exist_ok=True)
        ref_path = template_dir / "template_reference.nii.gz"
        mask_path = template_dir / f"{args.mask_column}.nii.gz"
        ref_path.write_text("ref")
        mask_path.write_text("mask")
        return {
            "template_source_idx": 0,
            "reference_image_path": str(ref_path),
            "mask_path": str(mask_path),
        }

    def fake_register_row(row, *, template_info, args, path_columns):
        counts["rows"] += 1
        return {
            "source_idx": int(row["_source_idx"]),
            "updates": {
                "population_register_template_source_idx": 0,
                "population_register_mask_column": args.mask_column,
                "population_register_stage": "rigid",
                "population_register_status": "ok",
                "population_register_error_message": None,
                "population_register_dice_before": 0.5,
                "population_register_dice_after": 0.9,
                **{column: 1.0 for column in reg_common.POPULATION_TRANSFORM_COLUMNS},
            },
            "error_message": None,
        }

    monkeypatch.setattr(population_module, "_build_population_template", fake_build_template)
    monkeypatch.setattr(population_module, "_register_population_row", fake_register_row)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        output_dir=str(tmp_path / "registered"),
        error_csv_path=str(tmp_path / "errors.csv"),
        log_csv_path=str(tmp_path / "log.csv"),
        organ="liver",
        mask_column="mask_liver",
        template_sample_size=8,
        template_seed=0,
        num_workers=1,
        pad_mm=25.0,
        save_registered_outputs=False,
        verbose=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )

    population_module.main(args)
    assert counts["template"] == 1
    assert counts["rows"] == 1

    args.resume = True
    population_module.main(args)
    assert counts["template"] == 1
    assert counts["rows"] == 1


def test_register_population_row_principal_vectors_mode(tmp_path, monkeypatch):
    nifti = tmp_path / "a.nii.gz"
    mask = tmp_path / "a_mask.nii.gz"
    nifti.write_text("x")
    mask.write_text("x")

    class FakeSitk:
        sitkNearestNeighbor = 0
        sitkUInt8 = 1
        sitkFloat32 = 2

    monkeypatch.setattr(population_module.reg_common, "_load_register_dependencies", lambda: FakeSitk)
    monkeypatch.setattr(population_module.reg_common, "read_image", lambda *args, **kwargs: "img")
    monkeypatch.setattr(population_module.reg_common, "read_binary_mask", lambda *args, **kwargs: "mask")

    calls = {"dice": 0, "rigid": 0}

    def fake_dice(*args, **kwargs):
        calls["dice"] += 1
        return 0.5 if calls["dice"] == 1 else 0.9

    def fake_rigid(*args, **kwargs):
        calls["rigid"] += 1
        raise AssertionError("rigid registration must not be called in principal_vectors mode")

    monkeypatch.setattr(population_module.reg_common, "dice_coeff", fake_dice)
    monkeypatch.setattr(population_module.reg_common, "rigid_register_mask_pair", fake_rigid)
    monkeypatch.setattr(
        population_module,
        "_mask_principal_frame",
        lambda *args, **kwargs: {
            "centroid_mm": [0.0, 0.0, 0.0],
            "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
    )
    monkeypatch.setattr(population_module, "_principal_frame_transform", lambda *args, **kwargs: "tx")
    monkeypatch.setattr(
        population_module.reg_common,
        "transform_to_flat_quaternion_translation",
        lambda tx: {column: 0.0 for column in reg_common.POPULATION_TRANSFORM_COLUMNS},
    )

    args = argparse.Namespace(
        mask_column="mask_liver",
        save_registered_outputs=False,
        output_dir=str(tmp_path / "registered"),
        template_mode="principal_vectors",
    )
    template_info = {
        "template_source_idx": 0,
        "reference_image_path": str(nifti),
        "mask_path": str(mask),
        "principal_reference": {
            "centroid_mm": [0.0, 0.0, 0.0],
            "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
    }
    row = {"_source_idx": 0, "nifti_path": str(nifti), "mask_liver": str(mask)}

    result = population_module._register_population_row(
        row,
        template_info=template_info,
        args=args,
        path_columns=["nifti_path", "mask_liver"],
    )

    assert result["error_message"] is None
    assert result["updates"]["population_register_stage"] == "principal_vectors"
    assert result["updates"]["population_register_template_mode"] == "principal_vectors"
    assert calls["rigid"] == 0


def test_build_single_sample_template_debug_logging_omits_redundant_details(
    tmp_path, monkeypatch, caplog
):
    nifti = tmp_path / "a.nii.gz"
    mask = tmp_path / "a_mask.nii.gz"
    for path in (nifti, mask):
        path.write_text("x")

    df = pd.DataFrame([{"_source_idx": 0, "nifti_path": str(nifti), "mask_liver": str(mask)}])

    monkeypatch.setattr(
        population_module.reg_common,
        "sample_valid_rows_for_template",
        lambda *args, **kwargs: [
            {
                "source_idx": 0,
                "nifti_path": str(nifti),
                "mask_path": str(mask),
            }
        ],
    )
    monkeypatch.setattr(
        population_module.reg_common,
        "choose_median_exemplar",
        lambda sampled_rows: sampled_rows[0],
    )
    monkeypatch.setattr(population_module.reg_common, "read_image", lambda *args, **kwargs: "image")
    monkeypatch.setattr(population_module.reg_common, "read_binary_mask", lambda *args, **kwargs: "mask")
    monkeypatch.setattr(
        population_module.reg_common,
        "write_image",
        lambda image, path, **kwargs: Path(path).write_text("x"),
    )

    args = argparse.Namespace(
        output_dir=str(tmp_path / "registered"),
        mask_column="mask_liver",
        template_sample_size=4,
        template_seed=0,
        template_mode="single_sample",
        template_source_idx=None,
    )

    with caplog.at_level(logging.DEBUG):
        population_module._build_single_sample_template(
            df,
            args=args,
            sitk_module=object(),
        )

    debug_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG]

    assert not any("Template sampled source indices" in msg for msg in debug_messages)
    assert not any("Single-sample template chose median exemplar" in msg for msg in debug_messages)
    assert not any("Single-sample template using explicit source_idx" in msg for msg in debug_messages)


def test_register_population_row_debug_logging_keeps_success_and_normalization(
    tmp_path, monkeypatch, caplog
):
    nifti = tmp_path / "a.nii.gz"
    mask = tmp_path / "a_mask.nii.gz"
    tumor = tmp_path / "a_tumor.nii.gz"
    template_ref = tmp_path / "template_ref.nii.gz"
    template_mask = tmp_path / "template_mask.nii.gz"
    for path in (nifti, mask, tumor, template_ref, template_mask):
        path.write_text("x")

    class FakeImage:
        def GetSpacing(self):
            return (1.0, 1.0, 1.0)

    class FakeSitk:
        sitkNearestNeighbor = 0
        sitkUInt8 = 1
        sitkFloat32 = 2

    monkeypatch.setattr(population_module.reg_common, "_load_register_dependencies", lambda: FakeSitk)
    monkeypatch.setattr(population_module.reg_common, "read_image", lambda *args, **kwargs: FakeImage())
    monkeypatch.setattr(population_module.reg_common, "read_binary_mask", lambda *args, **kwargs: "mask")
    monkeypatch.setattr(population_module.reg_common, "rigid_register_mask_pair", lambda *args, **kwargs: "rigid")
    monkeypatch.setattr(
        population_module.reg_common,
        "dice_coeff",
        lambda *args, **kwargs: 0.5 if kwargs.get("tx") is None else 0.9,
    )
    monkeypatch.setattr(
        population_module.reg_common,
        "transform_to_flat_quaternion_translation",
        lambda tx: {column: 0.0 for column in reg_common.POPULATION_TRANSFORM_COLUMNS},
    )

    def fake_warp_row_files(row, *, row_dir, path_columns, **kwargs):
        out = {}
        for column in path_columns:
            value = row.get(column)
            if not value:
                continue
            out_path = reg_common.build_output_path(
                row_dir,
                column_name=column,
                source_path=value,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(column)
            out[column] = str(out_path)
        return out

    monkeypatch.setattr(population_module.reg_common, "warp_row_files", fake_warp_row_files)
    monkeypatch.setattr(
        population_module,
        "save_transform_artifacts",
        lambda **kwargs: argparse.Namespace(
            transform_path="population.tfm",
            metadata_path="population.json",
        ),
    )
    monkeypatch.setattr(
        population_module,
        "normalize_image_and_masks",
        lambda **kwargs: (
            FakeImage(),
            "organ_mask",
            {"mask_liver_tumor": "tumor_mask"},
            {"crop_mode": "margin"},
        ),
    )
    monkeypatch.setattr(
        population_module.reg_common,
        "write_image",
        lambda image, path, **kwargs: (
            Path(path).parent.mkdir(parents=True, exist_ok=True),
            Path(path).write_text("x"),
        ),
    )

    args = argparse.Namespace(
        mask_column="mask_liver",
        save_registered_outputs=True,
        normalize_registered_outputs=True,
        output_dir=str(tmp_path / "registered"),
        template_mode="mean_shape",
        normalize_crop_mode="margin",
        normalize_margin_mm=10.0,
        normalize_without_background=False,
        normalize_spacing=None,
        normalize_orientation="LPS",
        disable_normalize_center_organ=False,
    )
    template_info = {
        "template_source_idx": 0,
        "reference_image_path": str(template_ref),
        "mask_path": str(template_mask),
    }
    row = {
        "_source_idx": 0,
        "nifti_path": str(nifti),
        "mask_liver": str(mask),
        "mask_liver_tumor": str(tumor),
    }

    with caplog.at_level(logging.DEBUG):
        result = population_module._register_population_row(
            row,
            template_info=template_info,
            args=args,
            path_columns=["nifti_path", "mask_liver", "mask_liver_tumor"],
        )

    debug_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG]

    assert result["error_message"] is None
    assert not any("Starting population registration" in msg for msg in debug_messages)
    assert not any("wrote 3 warped paths" in msg for msg in debug_messages)
    assert any("registered successfully" in msg for msg in debug_messages)
    assert any("wrote normalized outputs" in msg for msg in debug_messages)


def test_register_population_main_debug_logging_omits_plumbing(
    tmp_path, monkeypatch, caplog
):
    nifti = tmp_path / "a.nii.gz"
    mask = tmp_path / "a_mask.nii.gz"
    for path in (nifti, mask):
        path.write_text("x")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "mask_liver": str(mask)}]).to_csv(
        csv_path, index=False
    )

    monkeypatch.setattr(population_module.reg_common, "_load_register_dependencies", lambda: object())
    monkeypatch.setattr(
        population_module,
        "_build_population_template",
        lambda df, *, args, sitk_module: {
            "template_source_idx": 0,
            "template_mode": "mean_shape",
            "reference_image_path": str(nifti),
            "mask_path": str(mask),
            "sample_count": 1,
        },
    )
    monkeypatch.setattr(
        population_module,
        "_register_population_row",
        lambda row, *, template_info, args, path_columns: {
            "source_idx": int(row["_source_idx"]),
            "updates": {
                "population_register_template_source_idx": 0,
                "population_register_mask_column": args.mask_column,
                "population_register_stage": "rigid",
                "population_register_status": "ok",
                "population_register_error_message": None,
                "population_register_dice_before": 0.5,
                "population_register_dice_after": 0.9,
                **{
                    column: 1.0 for column in reg_common.POPULATION_TRANSFORM_COLUMNS
                },
            },
            "error_message": None,
        },
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        output_dir=str(tmp_path / "registered"),
        error_csv_path=str(tmp_path / "errors.csv"),
        log_csv_path=str(tmp_path / "log.csv"),
        organ="liver",
        mask_column="mask_liver",
        template_sample_size=8,
        template_seed=0,
        num_workers=1,
        pad_mm=25.0,
        save_registered_outputs=False,
        normalize_registered_outputs=False,
        verbose=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )

    with caplog.at_level(logging.DEBUG):
        population_module.main(args)

    debug_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG]

    assert not any("Population path columns selected" in msg for msg in debug_messages)
    assert not any("population row payloads for execution" in msg for msg in debug_messages)


def test_register_intra_patient_main_rewrites_paths(tmp_path, monkeypatch):
    nifti_a = tmp_path / "a.nii.gz"
    mask_a = tmp_path / "a_mask.nii.gz"
    nifti_b = tmp_path / "b.nii.gz"
    mask_b = tmp_path / "b_mask.nii.gz"
    tumor_b = tmp_path / "b_tumor.nii.gz"
    for path in (nifti_a, mask_a, nifti_b, mask_b, tumor_b):
        path.write_text("x")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [
            {
                "patient_key": "p1",
                "phase": "portal",
                "nifti_path": str(nifti_a),
                "mask_liver": str(mask_a),
                "mask_liver_tumor": "",
            },
            {
                "patient_key": "p1",
                "phase": "arteriel",
                "nifti_path": str(nifti_b),
                "mask_liver": str(mask_b),
                "mask_liver_tumor": str(tumor_b),
            },
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(intra_module.reg_common, "_load_register_dependencies", lambda: object())

    def fake_process_group(patient_df, *, pending_source_indices, args, path_columns):
        results = []
        for _, row in patient_df.iterrows():
            source_idx = int(row["_source_idx"])
            if source_idx not in pending_source_indices:
                continue
            row_dir = Path(args.output_dir) / "rows" / str(source_idx)
            row_dir.mkdir(parents=True, exist_ok=True)
            rewritten = {}
            for column in ("nifti_path", "mask_liver"):
                out_path = reg_common.build_output_path(
                    row_dir,
                    column_name=column,
                    source_path=str(row[column]),
                )
                out_path.write_text(column)
                rewritten[column] = str(out_path)
            if source_idx == 1:
                out_path = reg_common.build_output_path(
                    row_dir,
                    column_name="mask_liver_tumor",
                    source_path=str(row["mask_liver_tumor"]),
                )
                out_path.write_text("tumor")
                rewritten["mask_liver_tumor"] = str(out_path)
            results.append(
                {
                    "source_idx": source_idx,
                    "updates": {
                        **rewritten,
                        "intra_register_anchor_source_idx": 0,
                        "intra_register_anchor_phase": "portal",
                        "intra_register_stage": "anchor" if source_idx == 0 else "bspline",
                        "intra_register_dice_before": 1.0 if source_idx == 0 else 0.5,
                        "intra_register_dice_after_rigid": 1.0 if source_idx == 0 else 0.8,
                        "intra_register_dice_after_elastic": 1.0 if source_idx == 0 else 0.9,
                        "intra_register_status": "ok",
                        "intra_register_error_message": None,
                    },
                    "error_message": None,
                }
            )
        return results

    monkeypatch.setattr(intra_module, "_process_patient_group", fake_process_group)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        output_dir=str(tmp_path / "registered"),
        error_csv_path=str(tmp_path / "errors.csv"),
        log_csv_path=str(tmp_path / "log.csv"),
        organ="liver",
        mask_column="mask_liver",
        num_workers=1,
        pad_mm=25.0,
        band_mm=15.0,
        bspline_ctrl_spacing_mm=90.0,
        verbose=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )

    intra_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert Path(out_df.loc[0, "nifti_path"]).name == "scan.nii.gz"
    assert Path(out_df.loc[1, "mask_liver_tumor"]).name == "liver_tumor.nii.gz"
    assert out_df.loc[1, "intra_register_stage"] == "bspline"
    assert Path(out_df.loc[0, "nifti_path"]).exists()

    log_df = pd.read_csv(args.log_csv_path)
    assert sorted(log_df["intra_register_stage"].tolist()) == ["anchor", "bspline"]


def test_register_intra_patient_main_handles_missing_patient_key(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "a.nii.gz"
    mask = tmp_path / "a_mask.nii.gz"
    for path in (nifti, mask):
        path.write_text("x")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [{"patient_key": "", "nifti_path": str(nifti), "mask_liver": str(mask)}]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(intra_module.reg_common, "_load_register_dependencies", lambda: object())
    calls = {"groups": 0}

    def fake_process_group(patient_df, *, pending_source_indices, args, path_columns):
        calls["groups"] += 1
        return []

    monkeypatch.setattr(intra_module, "_process_patient_group", fake_process_group)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        output_dir=str(tmp_path / "registered"),
        error_csv_path=str(tmp_path / "errors.csv"),
        log_csv_path=str(tmp_path / "log.csv"),
        organ="liver",
        mask_column="mask_liver",
        num_workers=1,
        pad_mm=25.0,
        band_mm=15.0,
        bspline_ctrl_spacing_mm=90.0,
        verbose=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )

    intra_module.main(args)

    assert calls["groups"] == 0
    out_df = pd.read_csv(args.csv_path_out)
    assert out_df.loc[0, "intra_register_status"] == "error"

    err_df = pd.read_csv(args.error_csv_path)
    assert err_df.loc[0, "error_message"] == "missing patient_key value"


def test_register_intra_patient_main_resume_skips_completed_rows(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "a.nii.gz"
    mask = tmp_path / "a_mask.nii.gz"
    for path in (nifti, mask):
        path.write_text("x")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [{"patient_key": "p1", "phase": "portal", "nifti_path": str(nifti), "mask_liver": str(mask)}]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(intra_module.reg_common, "_load_register_dependencies", lambda: object())
    calls = {"groups": 0}

    def fake_process_group(patient_df, *, pending_source_indices, args, path_columns):
        calls["groups"] += 1
        source_idx = int(patient_df.iloc[0]["_source_idx"])
        row_dir = Path(args.output_dir) / "rows" / str(source_idx)
        row_dir.mkdir(parents=True, exist_ok=True)
        out_path = reg_common.build_output_path(
            row_dir,
            column_name="nifti_path",
            source_path=str(patient_df.iloc[0]["nifti_path"]),
        )
        mask_path = reg_common.build_output_path(
            row_dir,
            column_name="mask_liver",
            source_path=str(patient_df.iloc[0]["mask_liver"]),
        )
        out_path.write_text("nifti")
        mask_path.write_text("mask")
        return [
            {
                "source_idx": source_idx,
                "updates": {
                    "nifti_path": str(out_path),
                    "mask_liver": str(mask_path),
                    "intra_register_anchor_source_idx": source_idx,
                    "intra_register_anchor_phase": "portal",
                    "intra_register_stage": "anchor",
                    "intra_register_dice_before": 1.0,
                    "intra_register_dice_after_rigid": 1.0,
                    "intra_register_dice_after_elastic": 1.0,
                    "intra_register_status": "ok",
                    "intra_register_error_message": None,
                },
                "error_message": None,
            }
        ]

    monkeypatch.setattr(intra_module, "_process_patient_group", fake_process_group)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        output_dir=str(tmp_path / "registered"),
        error_csv_path=str(tmp_path / "errors.csv"),
        log_csv_path=str(tmp_path / "log.csv"),
        organ="liver",
        mask_column="mask_liver",
        num_workers=1,
        pad_mm=25.0,
        band_mm=15.0,
        bspline_ctrl_spacing_mm=90.0,
        verbose=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )

    intra_module.main(args)
    assert calls["groups"] == 1

    args.resume = True
    intra_module.main(args)
    assert calls["groups"] == 1
