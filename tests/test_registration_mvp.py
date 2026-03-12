import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imperandi.process.registration import (
    GroupingKeys,
    LongitudinalAuditConfig,
    TumorComponent,
    build_intra_patient_tasks,
    build_longitudinal_audit,
    parse_spacing_csv_value,
)
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
