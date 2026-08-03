import json
from pathlib import Path

import pandas as pd
import yaml

from imperandi.config import config_hash, load_config
from imperandi.io.tables import read_table
from imperandi.pipeline.base import RunContext
from imperandi.pipeline.defaults import build_default_runner
from imperandi.pipeline.stages.imaging import ConvertStage, SegmentStage


def test_metadata_only_pipeline_routes_ct_and_mr_and_publishes(tmp_path):
    source = tmp_path / "instances.csv"
    pd.DataFrame(
        [
            {
                "PatientID": "P1",
                "study_id": "ct-study",
                "series_id": "ct-series",
                "dicom_path": "ct.dcm",
                "Modality": "CT",
                "SeriesDescription": "Abdomen portal venous",
                "ImageType": "ORIGINAL PRIMARY AXIAL",
                "Rows": 512,
                "Columns": 512,
                "SliceThickness": 2.0,
                "StudyDate": "20200101",
            },
            {
                "PatientID": "P1",
                "study_id": "mr-study",
                "series_id": "mr-series",
                "dicom_path": "mr.dcm",
                "Modality": "MR",
                "SeriesDescription": "AX T2 FS BLADE",
                "StudyDate": "20200101",
            },
        ]
    ).to_csv(source, index=False)
    config_path = tmp_path / "imperandi.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "project": {"name": "metadata-test"},
                "input": {"sources": [str(source)]},
                "output": {
                    "root": str(tmp_path / "out"),
                    "table_format": "parquet",
                    "publish_formats": ["parquet", "csv"],
                },
                "identity": {
                    "source": {
                        "patient_id_columns": ["PatientID"],
                        "namespace_columns": [],
                        "fallback": {"on_missing": "error"},
                    },
                    "canonical": {"strategy": "source"},
                },
                "annotations": {
                    "rule_packs": ["builtin:liver_ct", "builtin:liver_mri"],
                    "contextual_strategies": [
                        "art_port",
                        "mask_multiart",
                        "generic_dynamic_volume_order",
                    ],
                },
                "selection": {
                    "required_slots": {
                        "CT": ["CT_PORTAL_VENOUS"],
                        "MR": ["MR_T2"],
                    }
                },
                "conversion": {"enabled": False},
                "segmentation": {"enabled": False},
                "registration": {"enabled": False},
                "radiomics": {"enabled": False},
                "execution": {"workers": 1},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    runner = build_default_runner(config)
    results = runner.run()
    cohort = read_table(results["11_publish"].artifacts["cohort_index"])
    run_dir = config.output.root / "runs" / config_hash(config)[:12]
    run_state = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert set(cohort["clinical_slot"]) == {"CT_PORTAL_VENOUS", "MR_T2"}
    assert set(cohort["patient_id"]) == {"P1"}
    assert (results["11_publish"].artifacts["cohort_index_csv"]).exists()
    assert run_state["status"] == "completed"
    assert run_state["artifacts"]["cohort_index"].endswith("cohort_index.parquet")

    schema_path = (
        results["11_publish"]
        .artifacts["cohort_index"]
        .with_suffix(".parquet.schema.json")
    )
    schema_path.unlink()
    build_default_runner(config).run()
    assert schema_path.exists()

    changed = pd.read_csv(source)
    extra = changed.iloc[[0]].copy()
    extra["PatientID"] = "P2"
    extra["study_id"] = "ct-study-p2"
    extra["series_id"] = "ct-series-p2"
    extra["dicom_path"] = "ct-p2.dcm"
    pd.concat([changed, extra], ignore_index=True).to_csv(source, index=False)

    rerun_results = build_default_runner(config).run()
    rerun_cohort = read_table(rerun_results["11_publish"].artifacts["cohort_index"])
    assert set(rerun_cohort["patient_id"]) == {"P1", "P2"}


def test_unsupported_modality_preserves_an_explicit_exclusion_reason(tmp_path):
    source = tmp_path / "instances.csv"
    pd.DataFrame(
        [
            {
                "PatientID": "P1",
                "study_id": "pet-study",
                "series_id": "pet-series",
                "dicom_path": "pet.dcm",
                "Modality": "PT",
                "SeriesDescription": "PET attenuation",
                "StudyDate": "20200101",
            }
        ]
    ).to_csv(source, index=False)
    rules = tmp_path / "site_rules.yaml"
    rules.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "rules": [
                    {
                        "id": "exclude.pet",
                        "action": "exclude",
                        "reason": "pet_not_in_scope",
                        "priority": 100,
                        "when": {
                            "any": [
                                {
                                    "column": "Modality",
                                    "operator": "eq",
                                    "value": "PT",
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "imperandi.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "project": {"name": "unsupported"},
                "input": {"sources": [str(source)]},
                "output": {"root": str(tmp_path / "out")},
                "identity": {
                    "source": {
                        "patient_id_columns": ["PatientID"],
                        "namespace_columns": [],
                    }
                },
                "annotations": {"rule_packs": [str(rules)]},
            }
        ),
        encoding="utf-8",
    )

    result = build_default_runner(load_config(config_path)).run()
    annotated = read_table(result["04_annotate"].artifacts["volumes_annotated"])

    assert annotated.loc[0, "curation_modality"] == "OTHER"
    assert annotated.loc[0, "exclusion_reason"] == "pet_not_in_scope"
    assert annotated.loc[0, "exclusion_rule_id"] == "exclude.pet"


def test_conversion_uses_patient_key_only_inside_backend_bridge(tmp_path, monkeypatch):
    config_path = tmp_path / "imperandi.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "project": {"name": "convert-bridge"},
                "input": {"sources": ["unused.csv"]},
                "output": {"root": str(tmp_path / "out"), "table_format": "csv"},
                "conversion": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    context = RunContext(config=config, run_dir=tmp_path / "run")
    context.write_table(
        "04_annotate",
        "volumes_shortlist",
        pd.DataFrame(
            {
                "patient_id": ["P-001"],
                "study_id": ["study"],
                "series_id": ["series"],
                "dicom_path": [["scan.dcm"]],
            }
        ),
    )

    def fake_convert(args):
        backend_input = pd.read_csv(args.csv_path[0])
        assert backend_input.loc[0, "patient_key"] == "P-001"
        backend_input["nifti_path"] = "/images/P-001/scan.nii.gz"
        backend_input.to_csv(args.csv_path_out, index=False)

    monkeypatch.setattr("imperandi.process.convert.main", fake_convert)

    result = ConvertStage().run(context)
    converted = read_table(result.artifacts["volumes_converted"])

    assert converted.loc[0, "patient_id"] == "P-001"
    assert "patient_key" not in converted.columns


def test_mixed_modality_segmentation_routes_explicit_backend_tasks(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "imperandi.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "project": {"name": "segment-routing"},
                "input": {"sources": ["unused.csv"]},
                "output": {"root": str(tmp_path / "out"), "table_format": "csv"},
                "segmentation": {
                    "enabled": True,
                    "tasks": [
                        {
                            "id": "liver_ct",
                            "modality": "CT",
                            "task": "total",
                            "output": "liver",
                        },
                        {
                            "id": "liver_mr",
                            "modality": "MR",
                            "task": "total_mr",
                            "output": "liver",
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    context = RunContext(config=config, run_dir=tmp_path / "run")
    context.write_table(
        "07_resolve_select",
        "selected_volumes",
        pd.DataFrame(
            {
                "patient_id": ["P-001", "P-001"],
                "volume_id": ["ct-volume", "mr-volume"],
                "curation_modality": ["CT", "MR"],
                "nifti_path": ["ct.nii.gz", "mr.nii.gz"],
            }
        ),
    )
    routes = {}

    def fake_segment(args):
        backend_config = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        frame = pd.read_csv(args.csv_path)
        modality = frame.loc[0, "curation_modality"]
        tasks = backend_config["segmentation"]["tasks"]
        routes[modality] = [task["task"] for task in tasks]
        frame["mask_liver"] = frame["volume_id"].map(
            lambda volume_id: f"/masks/{volume_id}/liver.nii.gz"
        )
        frame.to_csv(args.csv_path_out, index=False)

    monkeypatch.setattr("imperandi.process.segment.main", fake_segment)

    result = SegmentStage().run(context)
    segmented = read_table(result.artifacts["volumes_segmented"])

    assert routes == {"CT": ["total"], "MR": ["total_mr"]}
    assert set(segmented["volume_id"]) == {"ct-volume", "mr-volume"}
    assert segmented["mask_liver"].notna().all()
