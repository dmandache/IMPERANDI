import sys
from pathlib import Path

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
