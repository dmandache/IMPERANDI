import json

import pandas as pd
import yaml

from imperandi.config import config_hash, load_config
from imperandi.io.tables import read_table
from imperandi.pipeline.defaults import build_default_runner


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
                "version": 1,
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
