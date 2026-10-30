"""Failures remain retryable across finished and legacy volume checkpoints."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from imperandi import cli
from imperandi.extract import phase, radiomics
from imperandi.process import convert, segment
from imperandi.utils.run_state import (
    CheckpointManager,
    build_checkpoint_paths,
    load_state,
)


@pytest.mark.parametrize(
    "case",
    [
        "convert",
        "convert_archive",
        "segment",
        "phase",
        "radiomics",
        "radiomics_missing",
    ],
)
@pytest.mark.parametrize("checkpoint_kind", ["finished", "legacy", "interrupted"])
def test_resume_retries_failures_until_recovered(
    tmp_path, monkeypatch, case, checkpoint_kind
):
    command = case.split("_")[0]
    calls = Counter()
    recovering = False
    rows = []
    source_ids = ["good", "bad"]
    if checkpoint_kind == "interrupted":
        source_ids.append("pending")
    for source_id in source_ids:
        folder = tmp_path / source_id
        folder.mkdir()
        image = folder / "image.nii.gz"
        image.write_text("image")
        rows.append(
            {
                "volume_id": source_id,
                "series_id": source_id,
                "patient_key": source_id,
                "study_id": "study",
                "Modality": "CT",
                "dicom_path": "['image.dcm']",
                "nifti_path": str(image),
            }
        )
    csv_path = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    # Exercise an in-place phase output and separate outputs for other stages.
    output = csv_path if command == "phase" else tmp_path / "output.csv"
    errors = tmp_path / "errors.csv"
    paths = build_checkpoint_paths(output, errors, command)

    def fails(row):
        source_id = row["volume_id"]
        calls[source_id] += 1
        return source_id == "bad" and not recovering

    def fake_convert(frame, output_dir, verbose, num_workers, on_result, **kwargs):
        for idx, row in frame.iterrows():
            if fails(row):
                on_result(idx, None, {"error_message": "temporary failure"}, "failed")
            else:
                on_result(idx, row["nifti_path"], None, "converted")
        return frame, pd.DataFrame()

    def fake_segment(idx, row, *args, **kwargs):
        if fails(row):
            return idx, None, "temporary failure", None
        return idx, str(Path(row["nifti_path"]).parent), None, None

    def fake_phase(idx, row, **kwargs):
        if fails(row):
            return idx, None, "temporary failure"
        return idx, {"totalseg_phase": "portal"}, None

    def fake_features(row, *args, **kwargs):
        if fails(row):
            return {}, [
                (
                    "image path is missing or invalid"
                    if case == "radiomics_missing"
                    else "temporary failure"
                )
            ]
        return {"feature": 1.0}, []

    monkeypatch.setattr(convert, "convert_dicom_to_nifti_parallel", fake_convert)
    monkeypatch.setattr(
        convert,
        "materialize_archive_dicom_paths",
        lambda df, session: (df, pd.DataFrame()),
    )
    if case == "convert_archive":

        def materialize(df, session):
            if not recovering:
                error_rows = df[df["volume_id"] == "bad"].copy()
                calls["bad"] += len(error_rows)
                error_rows["error_message"] = "archive temporarily unavailable"
                return df[df["volume_id"] != "bad"].copy(), error_rows
            return df, pd.DataFrame()

        monkeypatch.setattr(convert, "materialize_archive_dicom_paths", materialize)
    monkeypatch.setattr(
        segment, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment, "process_single_volume", fake_segment)
    monkeypatch.setattr(phase, "process_single_volume", fake_phase)
    monkeypatch.setattr(radiomics, "_load_radiomics_dependencies", lambda: (None, None))
    monkeypatch.setattr(
        radiomics,
        "_create_extractors_for_rows",
        lambda df, *a, **k: (pd.Series("CT", index=df.index), {"CT": None}),
    )
    monkeypatch.setattr(radiomics, "_extract_row_features", fake_features)
    argv = [
        command,
        "--csv_path",
        str(csv_path),
        "--csv_path_out",
        str(output),
        "--error_csv_path",
        str(errors),
        "--checkpoint_every_rows",
        "1",
    ]
    if command in {"convert", "segment"}:
        argv += ["--num_workers", "1"]
    if command == "convert":
        argv += ["--output_dir", str(tmp_path / "nifti")]
    if command == "radiomics":
        argv += ["--skip_filter"]

    if checkpoint_kind == "interrupted":
        original_flush = CheckpointManager.flush

        def interrupt_after_failure(self, **kwargs):
            result = original_flush(self, **kwargs)
            if result and not kwargs["error_df"].empty:
                raise KeyboardInterrupt("interrupted after error checkpoint")
            return result

        with monkeypatch.context() as interruption:
            interruption.setattr(CheckpointManager, "flush", interrupt_after_failure)
            with pytest.raises(KeyboardInterrupt):
                cli.main(argv)
    else:
        assert cli.main(argv) == 0
    assert calls == {"good": 1, "bad": 1}
    state = load_state(paths.state_path)
    assert state["completed_indices"] == ["good"]
    assert set(state["completed_fingerprints"]) == {"good"}
    assert pd.read_csv(paths.error_checkpoint_path)["_source_idx"].tolist() == ["bad"]
    if checkpoint_kind != "interrupted":
        assert errors.exists()

    if checkpoint_kind == "legacy":
        state["completed_indices"].append("bad")
        paths.state_path.write_text(json.dumps(state))

    expected_calls = {"good": 1, "bad": 1}
    expected_completed = {"good"}
    if checkpoint_kind == "interrupted":
        expected_calls["pending"] = 1
        expected_completed.add("pending")

    # Ordinary resume retains errors without treating them as successful work.
    assert cli.main(argv) == 0
    assert calls == expected_calls
    assert pd.read_csv(paths.error_checkpoint_path)["_source_idx"].tolist() == ["bad"]
    assert errors.exists()

    # Toggling retry_failed must not invalidate successful checkpoints.
    argv.append("--retry_failed")
    # A repeated failure must stay pending, without rerunning the successful row.
    assert cli.main(argv) == 0
    assert calls == {**expected_calls, "bad": 2}
    assert set(load_state(paths.state_path)["completed_indices"]) == expected_completed
    assert pd.read_csv(paths.error_checkpoint_path)["_source_idx"].tolist() == ["bad"]

    recovering = True
    assert cli.main(argv) == 0
    assert calls == {**expected_calls, "bad": 3}
    assert set(
        load_state(paths.state_path)["completed_indices"]
    ) == expected_completed | {"bad"}
    assert not paths.error_checkpoint_path.exists()
    assert not errors.exists()
    assert cli.main(argv) == 0
    assert calls == {**expected_calls, "bad": 3}

    # Removing the retry flag after recovery still reuses the same checkpoint.
    argv.remove("--retry_failed")
    assert cli.main(argv) == 0
    assert calls == {**expected_calls, "bad": 3}

    # Explicitly disabling resume reruns every volume.
    # Phase has a separate optimization for already populated predictions.
    fresh_options = (
        ["--no_resume", "--force"] if command == "phase" else ["--no_resume"]
    )
    assert cli.main([*argv, *fresh_options]) == 0
    assert calls == {
        key: count + 1 for key, count in {**expected_calls, "bad": 3}.items()
    }


@pytest.mark.parametrize(
    "command", ["convert", "segment", "phase", "radiomics", "register"]
)
def test_retry_failed_is_an_opt_in_cli_flag(command):
    parser = cli.build_parser()
    assert parser.parse_args([command]).retry_failed is False
    assert parser.parse_args([command, "--retry_failed"]).retry_failed is True


@pytest.mark.parametrize("command", ["convert", "segment", "phase", "radiomics"])
def test_failure_retry_policy_does_not_hide_changed_inputs(tmp_path, command):
    from argparse import Namespace
    from imperandi.utils.run_state import prepare_resume_context

    args = Namespace(resume=True, strict_resume=False)
    output = tmp_path / "output.csv"
    errors = tmp_path / "errors.csv"
    output.write_text("volume_id\ngood\nbad\n")
    fingerprints = {"good": "unchanged", "bad": "old"}
    fingerprint = {"input": "original"}

    def context():
        return prepare_resume_context(
            args=args,
            command=command,
            inputs=None,
            output_path=output,
            error_path=errors,
            input_fingerprint=fingerprint,
            row_fingerprints=fingerprints,
        )

    first = context()
    manager = CheckpointManager(paths=first["paths"], config=first["config"])
    manager.flush(
        main_df=pd.DataFrame({"_source_idx": ["good", "bad"]}),
        error_df=pd.DataFrame({"_source_idx": ["bad"], "error_message": ["failed"]}),
        completed_indices=["good"],
        force=True,
    )
    manager.finalize_state(completed_indices=["good"])
    normal = context()
    assert normal["already_finished"]
    assert normal["deferred_failed_indices"] == {"bad"}

    args.retry_failed = True
    retry = context()
    assert retry["can_resume"]
    assert not retry["already_finished"]
    assert retry["deferred_failed_indices"] == set()
    assert retry["state"]["completed_indices"] == ["good"]
    assert retry["config"].args_hash == first["config"].args_hash

    args.retry_failed = False
    fingerprint = {"input": "changed"}
    fingerprints["bad"] = "new"
    changed = context()
    assert changed["can_partial_resume"]
    assert not changed["already_finished"]
    assert changed["deferred_failed_indices"] == set()
    assert changed["state"]["completed_indices"] == ["good"]

    args.resume = False
    fresh = context()
    assert not fresh["can_resume"]
    assert not fresh["can_partial_resume"]
    assert fresh["deferred_failed_indices"] == set()
