import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imperandi.process.registration import (
    ConsensusConfig,
    GroupingKeys,
    LongitudinalAuditConfig,
    TumorComponent,
    build_intra_patient_tasks,
    build_visit_consensus,
    build_longitudinal_audit,
    parse_spacing_csv_value,
)
import imperandi.process.registration.consensus as consensus_impl
from imperandi.process import register_tumor_consensus as consensus_module


def _write_case_files(tmp_path: Path, count: int) -> list[dict[str, str]]:
    rows = []
    for idx in range(count):
        image = tmp_path / f"scan_{idx}.nii.gz"
        mask = tmp_path / f"mask_{idx}.nii.gz"
        image.write_text("scan")
        mask.write_text("mask")
        rows.append({"nifti_path": str(image), "mask_liver": str(mask)})
    return rows


def test_build_intra_patient_tasks_multiphasic(tmp_path):
    paths = _write_case_files(tmp_path, 4)
    df = pd.DataFrame(
        [
            {
                "_source_idx": 0,
                "patient_key": "p1",
                "visit_order": 0,
                "phase": "portal",
                **paths[0],
            },
            {
                "_source_idx": 1,
                "patient_key": "p1",
                "visit_order": 0,
                "phase": "arteriel",
                **paths[1],
            },
            {
                "_source_idx": 2,
                "patient_key": "p1",
                "visit_order": 1,
                "phase": "portal",
                **paths[2],
            },
            {
                "_source_idx": 3,
                "patient_key": "p1",
                "visit_order": 1,
                "phase": "arteriel",
                **paths[3],
            },
        ]
    )

    tasks, anchors, mode = build_intra_patient_tasks(
        df,
        keys=GroupingKeys(patient="patient_key", visit="visit_order", phase="phase"),
        pending_source_indices={0, 1, 2, 3},
        mask_column="mask_liver",
        mode="multiphasic",
    )

    assert mode == "multiphasic"
    assert anchors == {0, 2}
    assert {task.task_kind for task in tasks} == {"multiphasic"}
    assert {(task.reference_source_idx, task.moving_source_idx) for task in tasks} == {
        (0, 1),
        (2, 3),
    }


def test_build_intra_patient_tasks_longitudinal(tmp_path):
    paths = _write_case_files(tmp_path, 3)
    df = pd.DataFrame(
        [
            {
                "_source_idx": 0,
                "patient_key": "p1",
                "visit_order": 0,
                "phase": "portal",
                **paths[0],
            },
            {
                "_source_idx": 1,
                "patient_key": "p1",
                "visit_order": 1,
                "phase": "arteriel",
                **paths[1],
            },
            {
                "_source_idx": 2,
                "patient_key": "p1",
                "visit_order": 2,
                "phase": "portal",
                **paths[2],
            },
        ]
    )

    tasks, anchors, mode = build_intra_patient_tasks(
        df,
        keys=GroupingKeys(patient="patient_key", visit="visit_order", phase="phase"),
        pending_source_indices={0, 1, 2},
        mask_column="mask_liver",
        mode="longitudinal",
    )

    assert mode == "longitudinal"
    assert anchors == {0}
    assert {task.task_kind for task in tasks} == {"longitudinal"}
    assert {(task.reference_source_idx, task.moving_source_idx) for task in tasks} == {
        (0, 1),
        (0, 2),
    }


def test_build_longitudinal_audit_flags():
    components = {
        "0": [
            TumorComponent(1, 100, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 3, 3, 3),
            TumorComponent(2, 80, 0.8, 20.0, 0.0, 0.0, 4, 0, 0, 3, 3, 3),
        ],
        "1": [
            TumorComponent(1, 20, 0.2, 120.0, 0.0, 0.0, 10, 0, 0, 2, 2, 2),
        ],
    }
    findings = build_longitudinal_audit(
        patient_key="p1",
        sorted_visits=["0", "1"],
        components_by_visit=components,
        config=LongitudinalAuditConfig(
            max_centroid_shift_mm=25.0,
            max_total_volume_change_ratio=0.2,
        ),
    )
    flags = {f.flag for f in findings}
    assert "tumor_count_mismatch" in flags
    assert "suspicious_position_shift" in flags
    assert "unstable_segmentation_pattern" in flags


def test_parse_spacing_csv_value():
    assert parse_spacing_csv_value("1.0,2.5,3") == (1.0, 2.5, 3.0)
    assert parse_spacing_csv_value(None) is None


def test_normalize_register_tumor_consensus_args_defaults(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("patient_key,visit_order,phase,nifti_path,mask_liver_tumor\n")

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=None,
        csv_path_out=None,
        components_csv_path=None,
        audit_csv_path=None,
        error_csv_path=None,
        output_dir=str(tmp_path / "out"),
        organ="liver",
        patient_column="patient_key",
        visit_column="visit_order",
        phase_column="phase",
        image_column="nifti_path",
        organ_mask_column=None,
        tumor_mask_column=None,
        consensus_rule="majority",
        majority_threshold=0.5,
        min_component_voxels=10,
        disable_elastic=False,
        band_mm=15.0,
        bspline_ctrl_spacing_mm=90.0,
        max_centroid_shift_mm=25.0,
        max_total_volume_change_ratio=0.6,
        verbose=False,
        dry_run=False,
    )

    out = consensus_module.normalize_register_tumor_consensus_args(args)

    assert out.csv_path == str(csv_path.resolve())
    assert out.csv_path_out == str(csv_path.parent / "nifti_index_tumor_consensus.csv")
    assert out.components_csv_path == str(
        csv_path.parent / "tumor_consensus_components.csv"
    )
    assert out.audit_csv_path == str(csv_path.parent / "tumor_consistency_audit.csv")
    assert out.tumor_mask_column == "mask_liver_tumor"
    assert out.organ_mask_column == "mask_liver"


def test_build_visit_consensus_debug_logging_keeps_single_visit_start_and_fallbacks(
    tmp_path, monkeypatch, caplog
):
    ref_image = tmp_path / "ref.nii.gz"
    ref_tumor = tmp_path / "ref_tumor.nii.gz"
    ref_organ = tmp_path / "ref_organ.nii.gz"
    moving_image = tmp_path / "moving.nii.gz"
    moving_tumor = tmp_path / "moving_tumor.nii.gz"
    moving_organ = tmp_path / "moving_organ.nii.gz"
    skipped_image = tmp_path / "skipped.nii.gz"
    for path in (
        ref_image,
        ref_tumor,
        ref_organ,
        moving_image,
        moving_tumor,
        moving_organ,
        skipped_image,
    ):
        path.write_text("x")

    class FakeImage:
        def GetSpacing(self):
            return (1.0, 1.0, 1.0)

        def GetSize(self):
            return (1, 1, 1)

        def TransformContinuousIndexToPhysicalPoint(self, point):
            return tuple(point)

    class FakeSitk:
        sitkNearestNeighbor = 0
        sitkUInt8 = 1

        @staticmethod
        def GetArrayFromImage(_image):
            return np.array([[[1]]], dtype=np.uint8)

    monkeypatch.setattr(consensus_impl.reg_common, "read_image", lambda *args, **kwargs: FakeImage())
    monkeypatch.setattr(consensus_impl.reg_common, "read_binary_mask", lambda *args, **kwargs: "mask")
    monkeypatch.setattr(consensus_impl.reg_common, "resample_like", lambda *args, **kwargs: "aligned")
    monkeypatch.setattr(consensus_impl.reg_common, "rigid_register_mask_pair", lambda *args, **kwargs: "rigid")
    monkeypatch.setattr(
        consensus_impl.reg_common,
        "bspline_register_mask_pair",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("elastic failed")),
    )
    monkeypatch.setattr(
        consensus_impl.reg_common,
        "dice_coeff",
        lambda *args, **kwargs: 0.5 if kwargs.get("tx") is None else 0.8,
    )
    monkeypatch.setattr(
        consensus_impl.reg_common,
        "image_from_array_like",
        lambda *args, **kwargs: "consensus_mask",
    )

    rows = [
        {
            "_source_idx": 0,
            "patient_key": "p1",
            "visit_order": 0,
            "phase": "portal",
            "nifti_path": str(ref_image),
            "mask_liver_tumor": str(ref_tumor),
            "mask_liver": str(ref_organ),
        },
        {
            "_source_idx": 1,
            "patient_key": "p1",
            "visit_order": 0,
            "phase": "arteriel",
            "nifti_path": str(moving_image),
            "mask_liver_tumor": str(moving_tumor),
            "mask_liver": str(moving_organ),
        },
        {
            "_source_idx": 2,
            "patient_key": "p1",
            "visit_order": 0,
            "phase": "arteriel",
            "nifti_path": str(skipped_image),
            "mask_liver_tumor": "",
            "mask_liver": "",
        },
    ]

    with caplog.at_level(logging.DEBUG):
        result = build_visit_consensus(
            rows,
            patient_key="p1",
            visit_key="0",
            tumor_mask_column="mask_liver_tumor",
            organ_mask_column="mask_liver",
            config=ConsensusConfig(),
            sitk_module=FakeSitk,
        )

    debug_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG]

    assert result.reference_source_idx == 0
    assert sum("Building visit consensus" in msg for msg in debug_messages) == 1
    assert not any("Selected consensus reference" in msg for msg in debug_messages)
    assert not any("Built visit consensus" in msg for msg in debug_messages)
    assert any("Consensus source_idx=0 is the reference row" in msg for msg in debug_messages)
    assert any("keeping rigid transform" in msg for msg in debug_messages)
    assert any("Skipping consensus alignment for source_idx=2" in msg for msg in debug_messages)
    assert any("Consensus aligned source_idx=1" in msg for msg in debug_messages)


def test_register_tumor_consensus_main_debug_logging_keeps_visit_results(
    tmp_path, monkeypatch, caplog
):
    nifti = tmp_path / "scan.nii.gz"
    tumor = tmp_path / "tumor.nii.gz"
    nifti.write_text("x")
    tumor.write_text("x")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [
            {
                "patient_key": "p1",
                "visit_order": 0,
                "phase": "portal",
                "nifti_path": str(nifti),
                "mask_liver_tumor": str(tumor),
            }
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(consensus_module.reg_common, "_load_register_dependencies", lambda: object())
    monkeypatch.setattr(
        consensus_module,
        "build_visit_consensus",
        lambda *args, **kwargs: argparse.Namespace(
            reference_source_idx=0,
            consensus_mask="mask",
            components=[],
            aligned_mask_count=1,
            transform_metadata_by_source_idx={},
        ),
    )
    monkeypatch.setattr(
        consensus_module.reg_common,
        "write_image",
        lambda image, path, **kwargs: (
            Path(path).parent.mkdir(parents=True, exist_ok=True),
            Path(path).write_text("x"),
        ),
    )
    monkeypatch.setattr(consensus_module, "build_longitudinal_audit", lambda **kwargs: [])

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "consensus.csv"),
        components_csv_path=str(tmp_path / "components.csv"),
        audit_csv_path=str(tmp_path / "audit.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        output_dir=str(tmp_path / "registered"),
        organ="liver",
        patient_column="patient_key",
        visit_column="visit_order",
        phase_column="phase",
        image_column="nifti_path",
        organ_mask_column="mask_liver",
        tumor_mask_column="mask_liver_tumor",
        consensus_rule="majority",
        majority_threshold=0.5,
        min_component_voxels=10,
        disable_elastic=False,
        band_mm=15.0,
        bspline_ctrl_spacing_mm=90.0,
        max_centroid_shift_mm=25.0,
        max_total_volume_change_ratio=0.6,
    )

    with caplog.at_level(logging.DEBUG):
        consensus_module.main(args)

    debug_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG]

    assert any("Tumor consensus succeeded for patient=p1 visit=0" in msg for msg in debug_messages)
    assert not any("Processing tumor consensus" in msg for msg in debug_messages)
    assert not any("patient/visit groups for tumor consensus" in msg for msg in debug_messages)
