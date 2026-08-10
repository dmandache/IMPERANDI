import sys
from pathlib import Path
import json
import pytest

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imperandi import cli


def test_cli_parse_prefers_flag_paths_over_positionals(tmp_path, capsys):
    root_pos = tmp_path / "root_pos"
    out_pos = tmp_path / "out_pos"
    root_opt = tmp_path / "root_opt"
    out_opt = tmp_path / "out_opt"

    exit_code = cli.main(
        [
            "parse",
            str(root_pos),
            str(out_pos),
            "--root_path",
            str(root_opt),
            "--output_dir",
            str(out_opt),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out.replace("\\\\", "\\")
    assert exit_code == 0
    assert str(root_opt) in output
    assert str(out_opt) in output


def test_cli_clean_accepts_optional_csv_path_only(tmp_path):
    csv_in = tmp_path / "dicom_index.csv"
    csv_in.write_text("patient_key,study_id,series_id\np1,s1,sr1\n")
    csv_out = tmp_path / "dicom_index_clean.csv"

    exit_code = cli.main(
        [
            "clean",
            "--csv_path",
            str(csv_in),
            "--csv_path_out",
            str(csv_out),
            "--dry-run",
        ]
    )

    assert exit_code == 0


def test_cli_clean_accepts_positional_csv_path_and_csv_path_out(tmp_path, capsys):
    csv_in = tmp_path / "dicom_index.csv"
    csv_in.write_text("patient_key,study_id,series_id\np1,s1,sr1\n")
    csv_out = tmp_path / "dicom_index_clean_custom.csv"

    exit_code = cli.main(
        [
            "clean",
            str(csv_in),
            str(csv_out),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out.replace("\\\\", "\\")
    csv_path_out_line = next(
        line for line in output.splitlines() if line.strip().startswith("csv_path_out")
    )
    assert exit_code == 0
    assert str(csv_out) in csv_path_out_line


def test_cli_clean_prefers_flag_csv_path_out_over_positional(tmp_path, capsys):
    csv_in = tmp_path / "dicom_index.csv"
    csv_in.write_text("patient_key,study_id,series_id\np1,s1,sr1\n")
    csv_out_pos = tmp_path / "dicom_index_clean_pos.csv"
    csv_out_opt = tmp_path / "dicom_index_clean_opt.csv"

    exit_code = cli.main(
        [
            "clean",
            str(csv_in),
            str(csv_out_pos),
            "--csv_path_out",
            str(csv_out_opt),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out.replace("\\\\", "\\")
    csv_path_out_line = next(
        line for line in output.splitlines() if line.strip().startswith("csv_path_out")
    )
    assert exit_code == 0
    assert str(csv_out_opt) in csv_path_out_line


def test_cli_ingest_respects_flag_paths(tmp_path, capsys):
    root_opt = tmp_path / "root_opt"
    out_opt = tmp_path / "out_opt"

    exit_code = cli.main(
        [
            "ingest",
            "--root_path",
            str(root_opt),
            "--output_dir",
            str(out_opt),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out.replace("\\\\", "\\")
    assert exit_code == 0
    assert str(root_opt) in output
    assert str(out_opt) in output


def test_cli_phase_accepts_optional_csv_path_only(tmp_path):
    csv_in = tmp_path / "nifti_index.csv"
    csv_in.write_text("nifti_path\n")

    exit_code = cli.main(
        [
            "phase",
            "--csv_path",
            str(csv_in),
            "--dry-run",
        ]
    )

    assert exit_code == 0


def test_cli_phase_accepts_positional_csv_path_and_csv_path_out(tmp_path, capsys):
    csv_in = tmp_path / "nifti_index.csv"
    csv_in.write_text("nifti_path\n")
    csv_out = tmp_path / "nifti_index_phase_custom.csv"

    exit_code = cli.main(
        [
            "phase",
            str(csv_in),
            str(csv_out),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out.replace("\\\\", "\\")
    csv_path_out_line = next(
        line for line in output.splitlines() if line.strip().startswith("csv_path_out")
    )
    assert exit_code == 0
    assert str(csv_out) in csv_path_out_line


def test_cli_phase_prefers_flag_csv_path_out_over_positional(tmp_path, capsys):
    csv_in = tmp_path / "nifti_index.csv"
    csv_in.write_text("nifti_path\n")
    csv_out_pos = tmp_path / "nifti_index_phase_pos.csv"
    csv_out_opt = tmp_path / "nifti_index_phase_opt.csv"

    exit_code = cli.main(
        [
            "phase",
            str(csv_in),
            str(csv_out_pos),
            "--csv_path_out",
            str(csv_out_opt),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out.replace("\\\\", "\\")
    csv_path_out_line = next(
        line for line in output.splitlines() if line.strip().startswith("csv_path_out")
    )
    assert exit_code == 0
    assert str(csv_out_opt) in csv_path_out_line


def test_cli_radiomics_accepts_optional_csv_path_only(tmp_path):
    csv_in = tmp_path / "nifti_index.csv"
    csv_in.write_text("patient_key,phase,nifti_path\n")

    exit_code = cli.main(
        [
            "radiomics",
            "--csv_path",
            str(csv_in),
            "--dry-run",
        ]
    )

    assert exit_code == 0


def test_cli_radiomics_accepts_positional_csv_path_and_csv_path_out(tmp_path, capsys):
    csv_in = tmp_path / "nifti_index.csv"
    csv_in.write_text("patient_key,phase,nifti_path\n")
    csv_out = tmp_path / "nifti_index_radiomics_custom.csv"

    exit_code = cli.main(
        [
            "radiomics",
            str(csv_in),
            str(csv_out),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out.replace("\\\\", "\\")
    csv_path_out_line = next(
        line for line in output.splitlines() if line.strip().startswith("csv_path_out")
    )
    assert exit_code == 0
    assert str(csv_out) in csv_path_out_line


def test_cli_radiomics_prefers_flag_csv_path_out_over_positional(tmp_path, capsys):
    csv_in = tmp_path / "nifti_index.csv"
    csv_in.write_text("patient_key,phase,nifti_path\n")
    csv_out_pos = tmp_path / "nifti_index_radiomics_pos.csv"
    csv_out_opt = tmp_path / "nifti_index_radiomics_opt.csv"

    exit_code = cli.main(
        [
            "radiomics",
            str(csv_in),
            str(csv_out_pos),
            "--csv_path_out",
            str(csv_out_opt),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out.replace("\\\\", "\\")
    csv_path_out_line = next(
        line for line in output.splitlines() if line.strip().startswith("csv_path_out")
    )
    assert exit_code == 0
    assert str(csv_out_opt) in csv_path_out_line


def test_cli_radiomics_accepts_pyradiomics_settings_yaml(tmp_path):
    csv_in = tmp_path / "nifti_index.csv"
    csv_in.write_text("patient_key,phase,nifti_path\n")
    params_yaml = tmp_path / "Params.yaml"
    params_yaml.write_text("setting:\n  binWidth: 25\n")

    exit_code = cli.main(
        [
            "radiomics",
            "--csv_path",
            str(csv_in),
            "--pyradiomics_settings",
            str(params_yaml),
            "--dry-run",
        ]
    )

    assert exit_code == 0


def test_cli_radiomics_accepts_manifest_with_pyradiomics_block(tmp_path):
    csv_in = tmp_path / "nifti_index.csv"
    csv_in.write_text("patient_key,phase,nifti_path\n")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": "tmp",
                "radiomics": {
                    "pyradiomics": {
                        "setting": {"binWidth": 17},
                        "imageType": {"Original": {}},
                    },
                },
            }
        )
    )

    exit_code = cli.main(
        [
            "radiomics",
            "--csv_path",
            str(csv_in),
            "--manifest",
            str(manifest_path),
            "--dry-run",
        ]
    )

    assert exit_code == 0


def test_cli_radiomics_accepts_repeatable_filter_flags(tmp_path, capsys):
    csv_in = tmp_path / "nifti_index.csv"
    csv_in.write_text("patient_key,phase,nifti_path\n")

    exit_code = cli.main(
        [
            "radiomics",
            "--csv_path",
            str(csv_in),
            "--filter",
            "phase=portal,arteriel",
            "--filter",
            "followup_months=0,3",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out.replace("\\\\", "\\")
    filters_line = next(
        line for line in output.splitlines() if line.strip().startswith("filters")
    )
    assert exit_code == 0
    assert "'phase': ['portal', 'arteriel']" in filters_line
    assert "'followup_months': ['0', '3']" in filters_line


def test_cli_parse_accepts_snapshot_flags(tmp_path):
    root_opt = tmp_path / "root_opt"
    out_opt = tmp_path / "out_opt"

    exit_code = cli.main(
        [
            "parse",
            "--root_path",
            str(root_opt),
            "--output_dir",
            str(out_opt),
            "--snapshot_tags",
            "--snapshot_sample_size",
            "200",
            "--snapshot_seed",
            "123",
            "--dry-run",
        ]
    )

    assert exit_code == 0


def test_cli_parse_accepts_canonical_checkpoint_flags(tmp_path):
    root_opt = tmp_path / "root_opt"
    out_opt = tmp_path / "out_opt"

    exit_code = cli.main(
        [
            "parse",
            "--root_path",
            str(root_opt),
            "--output_dir",
            str(out_opt),
            "--checkpoint_every_rows",
            "200",
            "--checkpoint_every_sec",
            "30",
            "--no_resume",
            "--strict_resume",
            "--dry-run",
        ]
    )

    assert exit_code == 0


def test_cli_parse_resume_enabled_by_default(tmp_path, capsys):
    root_opt = tmp_path / "root_opt"
    out_opt = tmp_path / "out_opt"

    exit_code = cli.main(
        [
            "parse",
            "--root_path",
            str(root_opt),
            "--output_dir",
            str(out_opt),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    resume_line = next(
        line for line in output.splitlines() if line.strip().startswith("resume")
    )
    assert exit_code == 0
    assert resume_line.strip().endswith("True")


def test_cli_parse_no_resume_disables_resume(tmp_path, capsys):
    root_opt = tmp_path / "root_opt"
    out_opt = tmp_path / "out_opt"

    exit_code = cli.main(
        [
            "parse",
            "--root_path",
            str(root_opt),
            "--output_dir",
            str(out_opt),
            "--no_resume",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    resume_line = next(
        line for line in output.splitlines() if line.strip().startswith("resume")
    )
    assert exit_code == 0
    assert resume_line.strip().endswith("False")


def test_cli_parse_rejects_removed_resume_flag(tmp_path):
    root_opt = tmp_path / "root_opt"
    out_opt = tmp_path / "out_opt"

    with pytest.raises(SystemExit):
        cli.main(
            [
                "parse",
                "--root_path",
                str(root_opt),
                "--output_dir",
                str(out_opt),
                "--resume",
                "--dry-run",
            ]
        )


def test_cli_parse_rejects_legacy_checkpoint_frequency_flag(tmp_path):
    root_opt = tmp_path / "root_opt"
    out_opt = tmp_path / "out_opt"

    with pytest.raises(SystemExit):
        cli.main(
            [
                "parse",
                "--root_path",
                str(root_opt),
                "--output_dir",
                str(out_opt),
                "--checkpoint_frequency",
                "100",
                "--dry-run",
            ]
        )
