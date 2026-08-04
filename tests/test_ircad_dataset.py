from __future__ import annotations

import json
from ast import literal_eval
from pathlib import Path

import pandas as pd
import pytest
import yaml

from imperandi.config import config_hash, load_config
from imperandi.io.tables import read_table, table_schema_path
from imperandi.pipeline.defaults import build_default_runner
from imperandi.utils import files as files_module

pytestmark = pytest.mark.slow


def _as_list(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return []
    try:
        parsed = literal_eval(value)
    except (SyntaxError, ValueError):
        return [value]
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    return [value]


def _dataset_relative(path: str, dataset_root: Path) -> str:
    """Make fixture paths comparable when IRCAD_ROOT points outside the repo."""
    normalized = str(path).replace("\\", "/")
    root = dataset_root.resolve().as_posix().rstrip("/")
    if normalized.lower().startswith((root + "/").lower()):
        return normalized[len(root) + 1 :]

    for directory_name in (dataset_root.name, "IRCAD_DICOM", "IRCAD_nifti"):
        marker = f"/{directory_name}/"
        marker_index = normalized.lower().find(marker.lower())
        if marker_index >= 0:
            return normalized[marker_index + len(marker) :]
    return normalized


def _write_v2_config(path: Path, dicom_root: Path, output_root: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "project": {"name": "ircad-slow-test", "profile": "liver_ct_mri"},
                "input": {"sources": [str(dicom_root.resolve())]},
                "output": {
                    "root": str(output_root),
                    "table_format": "csv",
                    "publish_formats": ["csv"],
                },
                "identity": {
                    "source": {
                        "patient_id_columns": ["PatientName"],
                        "namespace_columns": [],
                        "fallback": {"columns": [], "on_missing": "error"},
                    },
                    "normalization": {"case": "preserve"},
                    "canonical": {"strategy": "source"},
                },
                # Exercise the complete v2 orchestration and curation path without
                # requiring optional models or recomputing the checked-in images.
                "phase_prediction": {"enabled": False},
                "conversion": {"enabled": False},
                "segmentation": {"enabled": False, "tasks": []},
                "registration": {"enabled": False},
                "radiomics": {"enabled": False},
                "execution": {"workers": 1, "resume": False},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_ircad_v2_pipeline_matches_reference_metadata(
    tmp_path, ircad_dicom_root, ircad_reference_csv
):
    config_path = _write_v2_config(
        tmp_path / "imperandi.yaml", ircad_dicom_root, tmp_path / "output"
    )
    config = load_config(config_path)
    results = build_default_runner(config).run()

    expected_stages = [
        "01_index",
        "02_identity",
        "03_assemble",
        "04_annotate",
        "05_convert",
        "06_predict_phase",
        "07_resolve_select",
        "08_segment",
        "09_register",
        "10_radiomics",
        "11_publish",
    ]
    assert list(results) == expected_stages

    generated_instances = read_table(
        results["01_index"].artifacts["instances_raw"]
    )
    reference_instances = pd.read_csv(ircad_reference_csv("dicom_index.csv"))
    assert len(generated_instances) == len(reference_instances)
    assert generated_instances["SOPInstanceUID"].is_unique
    assert reference_instances["SOPInstanceUID"].is_unique

    generated_instances = generated_instances.set_index("SOPInstanceUID")
    reference_instances = reference_instances.set_index("SOPInstanceUID")
    assert set(generated_instances.index) == set(reference_instances.index)
    for column in ["StudyInstanceUID", "SeriesInstanceUID", "Modality"]:
        pd.testing.assert_series_equal(
            generated_instances[column].sort_index(),
            reference_instances[column].sort_index(),
            check_dtype=False,
            check_names=False,
            check_index_type=False,
        )

    generated_paths = generated_instances["dicom_path"].map(
        lambda value: _dataset_relative(value, ircad_dicom_root)
    )
    reference_paths = reference_instances["dicom_path"].map(
        lambda value: _dataset_relative(value, ircad_dicom_root)
    )
    pd.testing.assert_series_equal(
        generated_paths.sort_index(),
        reference_paths.sort_index(),
        check_names=False,
        check_index_type=False,
    )

    identified = read_table(results["02_identity"].artifacts["instances"])
    expected_patients = set(reference_instances["PatientName"].dropna().astype(str))
    assert set(identified["patient_id"].dropna().astype(str)) == expected_patients
    assert "patient_key" not in identified.columns
    assert "PatientName" not in identified.columns

    generated_volumes = read_table(results["03_assemble"].artifacts["volumes"])
    reference_volumes = pd.read_csv(ircad_reference_csv("dicom_index_clean.csv"))
    assert len(generated_volumes) == len(reference_volumes)
    assert set(generated_volumes["patient_id"].astype(str)) == expected_patients
    assert "patient_key" not in generated_volumes.columns

    generated_volumes = generated_volumes.set_index("series_id").sort_index()
    reference_volumes = reference_volumes.set_index("series_id").sort_index()
    assert set(generated_volumes.index) == set(reference_volumes.index)
    for column in ["study_id", "Modality"]:
        pd.testing.assert_series_equal(
            generated_volumes[column],
            reference_volumes[column],
            check_dtype=False,
            check_names=False,
            check_index_type=False,
        )
    for column in [
        "Rows",
        "Columns",
        "n_files",
        "volume_length",
        "visit_order",
        "acquisition_order",
    ]:
        pd.testing.assert_series_equal(
            pd.to_numeric(generated_volumes[column], errors="coerce"),
            pd.to_numeric(reference_volumes[column], errors="coerce"),
            check_dtype=False,
            check_exact=False,
            rtol=1e-6,
            atol=1e-6,
            check_names=False,
            check_index_type=False,
        )

    generated_volume_paths = generated_volumes["dicom_path"].map(
        lambda value: sorted(
            _dataset_relative(path, ircad_dicom_root) for path in _as_list(value)
        )
    )
    reference_volume_paths = reference_volumes["dicom_path"].map(
        lambda value: sorted(
            _dataset_relative(path, ircad_dicom_root) for path in _as_list(value)
        )
    )
    pd.testing.assert_series_equal(
        generated_volume_paths,
        reference_volume_paths,
        check_names=False,
        check_index_type=False,
    )

    annotated = read_table(results["04_annotate"].artifacts["volumes_annotated"])
    assert len(annotated) == len(reference_volumes)
    assert set(annotated["curation_modality"]) == {"CT"}
    assert annotated["eligible"].notna().all()

    published = results["11_publish"].artifacts
    assert published["cohort_index"] == published["cohort_index_csv"]
    assert published["cohort_index_csv"].is_file()
    for artifact in published.values():
        assert artifact.is_file()
        assert table_schema_path(artifact).is_file()

    run_dir = config.output.root / "runs" / config_hash(config)[:12]
    run_state = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_state["status"] == "completed"
    assert set(run_state["artifacts"]) >= {"instances", "volumes", "cohort_index"}
    for stage_name in expected_stages:
        stage_state = json.loads(
            (run_dir / stage_name / "stage.json").read_text(encoding="utf-8")
        )
        assert stage_state["status"] == "completed"


def test_ircad_reference_nifti_files_are_readable(
    ircad_nifti_root, ircad_reference_csv
):
    nifti_index = pd.read_csv(ircad_reference_csv("nifti_index.csv"))
    assert "nifti_path" in nifti_index.columns

    paths = []
    for value in nifti_index["nifti_path"].dropna().astype(str):
        relative = _dataset_relative(value, ircad_nifti_root)
        paths.append(ircad_nifti_root / relative)

    missing = [path for path in paths if not path.exists()]
    assert not missing, f"Missing NIfTI files (showing up to 3): {missing[:3]}"
    assert all(files_module.is_valid_nifti(path) for path in paths)
