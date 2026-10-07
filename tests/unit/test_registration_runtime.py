"""Registration runtime contracts: resume, manifests, workers, and interruption."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from imperandi.process.registration import register, runtime
from imperandi.process.registration.config import RegistrationConfig
from imperandi.utils.run_state import build_checkpoint_paths

sitk = pytest.importorskip("SimpleITK")


@pytest.fixture
def cohort(tmp_path):
    rows = []
    for n in range(2):
        organ = np.zeros((8, 9, 10), np.uint8)
        organ[2:6, 2:7, 2:8] = 1
        path = tmp_path / f"scan{n}.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(organ), str(path))
        rows.append(
            dict(
                patient_key="001",
                study_id=f"visit{n}",
                Modality="CT",
                phase="PORTAL_VENOUS",
                nifti_path=str(path),
                mask_liver=str(path),
                mask_liver_tumor=str(path),
            )
        )
    source = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    return source


def args_for(source, *flags):
    return register.build_parser().parse_args(
        [str(source), "--timeout_sec", "0", *flags]
    )


def paths_for(source):
    output = source.with_name(source.stem + "_registered.csv")
    errors = output.with_name("register_errors.csv")
    return output, errors, build_checkpoint_paths(output, errors, "register")


def test_manifest_overrides_defaults_and_cli(tmp_path, cohort):
    manifest = tmp_path / "custom.yaml"
    manifest.write_text(
        yaml.safe_dump({"registration": {"method": "majority", "affine": True}})
    )
    args = register.normalize_registration_args(
        args_for(cohort, "--manifest", str(manifest))
    )
    config, _ = register.resolve_config(args)
    assert config.method == "majority" and config.affine
    override = register.normalize_registration_args(
        args_for(
            cohort, "--manifest", str(manifest), "--method", "union", "--no_affine"
        )
    )
    config, _ = register.resolve_config(override)
    assert config.method == "union" and not config.affine
    built_in = register.normalize_registration_args(
        args_for(cohort, "--manifest", "generic")
    )
    assert isinstance(register.resolve_config(built_in)[0], RegistrationConfig)


def test_cli_overrides_manifest_elastic_setting(tmp_path, cohort):
    manifest = tmp_path / "elastic.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {"registration": {"elastic": True, "demons_smoothing_sigma_mm": 2.0}}
        )
    )
    enabled, _ = register.resolve_config(
        register.normalize_registration_args(
            args_for(cohort, "--manifest", str(manifest))
        )
    )
    disabled, _ = register.resolve_config(
        register.normalize_registration_args(
            args_for(
                cohort,
                "--manifest",
                str(manifest),
                "--no_elastic",
                "--demons_smoothing_sigma_mm",
                "1.5",
            )
        )
    )
    assert enabled.elastic is True
    assert enabled.demons_smoothing_sigma_mm == 2.0
    assert disabled.elastic is False
    assert disabled.demons_smoothing_sigma_mm == 1.5


@pytest.mark.parametrize("entry_point", ["cli", "module"])
@pytest.mark.parametrize("override", [False, True])
def test_startup_log_contains_effective_manifest_settings(
    tmp_path, cohort, caplog, entry_point, override
):
    from imperandi import cli

    manifest = tmp_path / "logging.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {"registration": {"method": "majority", "elastic": True, "iterations": 17}}
        )
    )
    flags = ["--manifest", str(manifest), "--dry-run"]
    if override:
        flags += ["--method", "union", "--no_elastic"]
    args = args_for(cohort, *flags)
    with caplog.at_level(logging.INFO):
        if entry_point == "cli":
            cli._handle_register(args)
        else:
            register.main(args)
    records = [
        r
        for r in caplog.records
        if "Running register.py with namespace:" in r.getMessage()
    ]
    assert len(records) == 1
    logged = records[0].args[1]
    assert logged.method == ("union" if override else "majority")
    assert logged.elastic is not override
    assert logged.iterations == 17
    assert logged.min_dice == RegistrationConfig().min_dice
    assert logged.manifest == str(manifest)
    assert logged.csv_path == str(cohort)
    assert not any(name.startswith("_") for name in vars(logged))


def test_default_error_and_qc_filenames_follow_output_directory(cohort, tmp_path):
    output = tmp_path / "custom_result.csv"
    args = register.normalize_registration_args(
        args_for(cohort, "--csv_path_out", str(output))
    )
    assert Path(args.error_csv_path) == tmp_path / "register_errors.csv"
    assert Path(args.qc_csv_path) == tmp_path / "register_qc.csv"


def test_dry_run_no_backend_or_writes(monkeypatch, cohort, tmp_path):
    def unavailable():
        raise AssertionError("Dry run must not load backend")

    monkeypatch.setattr(runtime, "backend", unavailable)
    before = set(tmp_path.rglob("*"))
    register.main(args_for(cohort, "--dry-run"))
    assert set(tmp_path.rglob("*")) == before


def test_finished_run_skips_and_restores_missing_final_csv(monkeypatch, cohort):
    register.main(args_for(cohort))
    output, errors, paths = paths_for(cohort)
    original = output.read_bytes()
    stat = output.stat().st_mtime_ns

    def forbidden(*args, **kwargs):
        raise AssertionError("No completed group should run")
        yield

    monkeypatch.setattr(runtime, "iter_group_results", forbidden)
    register.main(args_for(cohort))
    assert output.stat().st_mtime_ns == stat
    # Recover a deleted final table from checkpoints without processing images.
    output.unlink()
    monkeypatch.setattr(runtime, "iter_group_results", lambda *a, **k: (x for x in ()))
    register.main(args_for(cohort))
    assert output.read_bytes() == original
    assert errors.exists() and paths.state_path.exists()
    assert json.loads(paths.state_path.read_text())["finished"]


def test_registration_behavior_change_invalidates_resume(monkeypatch, cohort):
    current_schema = runtime.REGISTRATION_SCHEMA
    monkeypatch.setattr(runtime, "REGISTRATION_SCHEMA", current_schema - 1)
    register.main(args_for(cohort))
    monkeypatch.setattr(runtime, "REGISTRATION_SCHEMA", current_schema)
    actual = runtime.iter_group_results
    observed = []

    def tracking(groups, *args, **kwargs):
        observed.extend(key for key, _ in groups)
        yield from actual(groups, *args, **kwargs)

    monkeypatch.setattr(runtime, "iter_group_results", tracking)
    register.main(args_for(cohort))
    assert len(observed) == 2


def test_retry_failed_recomputes_recovered_elastic_failure(monkeypatch, cohort):
    from imperandi.process.registration import alignment

    table = pd.read_csv(cohort, dtype=str)
    table["study_id"] = "same-visit"
    table.to_csv(cohort, index=False)
    actual_elastic = alignment.elastic_refine

    def fail_elastic(*args):
        raise ValueError("deliberate elastic failure")

    monkeypatch.setattr(alignment, "elastic_refine", fail_elastic)
    register.main(args_for(cohort, "--elastic"))
    output, errors, _ = paths_for(cohort)
    saved = pd.read_csv(output)
    assert not saved.registration_status.eq("failed").any()
    assert pd.read_csv(errors).empty
    assert (
        saved.registration_stage_details.map(
            lambda value: json.loads(value)["elastic"]["status"] == "failed"
        ).sum()
        == 1
    )
    monkeypatch.setattr(alignment, "elastic_refine", actual_elastic)
    actual_groups = runtime.iter_group_results
    observed = []

    def tracking(groups, *args, **kwargs):
        observed.extend(key for key, _ in groups)
        yield from actual_groups(groups, *args, **kwargs)

    monkeypatch.setattr(runtime, "iter_group_results", tracking)
    register.main(args_for(cohort, "--elastic"))
    assert observed == []
    register.main(args_for(cohort, "--elastic", "--retry_failed"))
    assert len(observed) == 1
    observed.clear()
    # Identical masks now produce an expected no-improvement rejection, which
    # must not be confused with an optimizer or validation failure on retry.
    register.main(args_for(cohort, "--elastic", "--retry_failed"))
    assert observed == []


def test_in_process_group_failure_is_recorded_and_other_groups_continue(
    monkeypatch, cohort
):
    actual = runtime.register_cohort
    previous_threads = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()

    def fail_one(table, *args, **kwargs):
        if table.iloc[0].study_id == "visit0":
            raise ValueError("deliberate group failure")
        return actual(table, *args, **kwargs)

    monkeypatch.setattr(runtime, "register_cohort", fail_one)
    register.main(args_for(cohort))
    output, error_path, _ = paths_for(cohort)
    result = pd.read_csv(output).set_index("study_id")
    assert result.loc["visit0", "registration_status"] == "failed"
    assert result.loc["visit1", "registration_status"] == "reference"
    assert pd.read_csv(error_path).error.tolist() == [
        "ValueError: deliberate group failure"
    ]
    assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == previous_threads


def test_in_process_interrupt_propagates_and_restores_threads(monkeypatch, tmp_path):
    previous = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()

    def interrupt(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(runtime, "register_cohort", interrupt)
    with pytest.raises(KeyboardInterrupt):
        list(
            runtime.iter_group_results(
                [("group", pd.DataFrame())],
                tmp_path,
                RegistrationConfig(),
                timeout_sec=0,
            )
        )
    assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == previous


def test_registration_uses_shared_task_summary(caplog, cohort):
    caplog.set_level(logging.INFO)
    register.main(args_for(cohort))
    assert (
        "Registration summary: 2 total row(s), 2 processed, 2 registered, "
        "0 reused, 0 failed, 2 groups processed"
    ) in caplog.text
    assert "Registration done ✔" in caplog.text

    caplog.clear()
    register.main(args_for(cohort))
    assert (
        "Registration summary: 2 total row(s), 0 processed, 0 registered, "
        "2 reused, 0 failed, 2 groups reused"
    ) in caplog.text


def test_interruption_resumes_only_committed_group(monkeypatch, cohort):
    actual = runtime.iter_group_results

    def interrupted(groups, *args, **kwargs):
        iterator = actual(groups, *args, **kwargs)
        try:
            yield next(iterator)
            raise KeyboardInterrupt()
        finally:
            iterator.close()

    monkeypatch.setattr(runtime, "iter_group_results", interrupted)
    with pytest.raises(KeyboardInterrupt):
        register.main(args_for(cohort, "--checkpoint_every_rows", "100"))
    output, _, paths = paths_for(cohort)
    assert not output.exists()
    state = json.loads(paths.state_path.read_text())
    assert len(state["completed_indices"]) == 1
    assert not state.get("finished", False)
    observed = []

    def tracking(groups, *args, **kwargs):
        observed.extend(key for key, _ in groups)
        yield from actual(groups, *args, **kwargs)

    monkeypatch.setattr(runtime, "iter_group_results", tracking)
    register.main(args_for(cohort))
    assert len(observed) == 1
    assert observed[0] not in state["completed_indices"]
    assert len(pd.read_csv(output)) == 2


@pytest.mark.parametrize("mutation", ["delete", "modify"])
def test_missing_artifact_recomputes_only_affected_group(monkeypatch, cohort, mutation):
    register.main(args_for(cohort))
    output, _, _ = paths_for(cohort)
    first = pd.read_csv(output)
    artifact = Path(first.loc[0, "reg_tumor_native_path"])
    if mutation == "delete":
        artifact.unlink()
    else:
        artifact.write_bytes(artifact.read_bytes() + b"modified")
    actual = runtime.iter_group_results
    observed = []

    def tracking(groups, *args, **kwargs):
        observed.extend(key for key, _ in groups)
        yield from actual(groups, *args, **kwargs)

    monkeypatch.setattr(runtime, "iter_group_results", tracking)
    register.main(args_for(cohort))
    second = pd.read_csv(output)
    assert observed == [first.loc[0, "registration_group_id"]]
    assert (
        second.loc[1, "reg_tumor_native_path"] == first.loc[1, "reg_tumor_native_path"]
    )
    assert Path(second.loc[0, "reg_tumor_native_path"]).is_file()


@pytest.mark.parametrize("change", ["mask", "manifest", "force", "no_resume"])
def test_changed_inputs_and_forced_runs_recompute(
    monkeypatch, cohort, tmp_path, change
):
    manifest = tmp_path / "settings.yaml"
    manifest.write_text("registration:\n  method: anchor\n")
    flags = ["--manifest", str(manifest)]
    register.main(args_for(cohort, *flags))
    if change == "mask":
        row = pd.read_csv(cohort).iloc[0]
        image = sitk.ReadImage(row.mask_liver)
        image[0, 0, 0] = 1
        sitk.WriteImage(image, row.mask_liver)
    elif change == "manifest":
        manifest.write_text("registration:\n  method: union\n")
    else:
        flags += ["--" + change]
    actual = runtime.iter_group_results
    seen = []

    def tracking(groups, *args, **kwargs):
        seen.extend(key for key, _ in groups)
        yield from actual(groups, *args, **kwargs)

    monkeypatch.setattr(runtime, "iter_group_results", tracking)
    register.main(args_for(cohort, *flags))
    assert len(seen) == 2


def test_strict_resume_hashes_referenced_inputs_and_artifacts(cohort):
    register.main(args_for(cohort, "--strict_resume"))
    _, _, paths = paths_for(cohort)
    state = json.loads(paths.state_path.read_text())
    assert len(state["input_fingerprint"]) == 3
    assert all("sha256" in item for item in state["input_fingerprint"])
    assert all(
        "sha256" in item for group in state["artifacts"].values() for item in group
    )


def test_retry_failed_and_preserve_errors(monkeypatch, cohort):
    real = runtime.iter_group_results

    def fail(groups, *args, **kwargs):
        for key, _ in groups:
            yield key, None, "deliberate worker failure"

    monkeypatch.setattr(runtime, "iter_group_results", fail)
    register.main(args_for(cohort))
    output, errors, _ = paths_for(cohort)
    assert len(pd.read_csv(errors)) == 2
    assert pd.read_csv(output).registration_status.eq("failed").all()
    monkeypatch.setattr(runtime, "iter_group_results", real)
    register.main(args_for(cohort))
    assert len(pd.read_csv(errors)) == 2
    register.main(args_for(cohort, "--retry_failed"))
    assert pd.read_csv(errors).empty
    assert pd.read_csv(output).registration_status.eq("reference").all()


def test_real_spawn_workers(cohort):
    register.main(args_for(cohort, "--num_workers", "2", "--timeout_sec", "30"))
    output, errors, _ = paths_for(cohort)
    assert pd.read_csv(errors).empty
    assert pd.read_csv(output).registration_status.eq("reference").all()


def test_hard_timeout_is_recorded(cohort):
    # Startup itself takes longer than this; verify timed-out processes cannot
    # subsequently publish successful rows or leave active children.
    before = {p.pid for p in runtime.mp.active_children()}
    register.main(args_for(cohort, "--timeout_sec", "0.001", "--num_workers", "2"))
    output, errors, _ = paths_for(cohort)
    assert pd.read_csv(output).registration_status.eq("failed").all()
    assert pd.read_csv(errors).error.str.contains("timeout").all()
    assert {p.pid for p in runtime.mp.active_children()} == before


@pytest.mark.parametrize(
    "flag,value",
    [("--num_workers", "0"), ("--timeout_sec", "-1"), ("--checkpoint_every_rows", "0")],
)
def test_runtime_argument_validation(cohort, flag, value):
    with pytest.raises(ValueError):
        register.normalize_registration_args(args_for(cohort, flag, value))


@pytest.mark.parametrize("trigger", ["rows", "time"])
def test_checkpoint_thresholds_commit_complete_groups(monkeypatch, cohort, trigger):
    from imperandi.utils import run_state

    clock = [1000.0]
    monkeypatch.setattr(run_state, "now_epoch", lambda: clock[0])
    # The shared manager reads time.time through now_epoch; its elapsed clock is
    # advanced after the first group, before the parent checks checkpoint limits.
    monkeypatch.setattr(run_state.time, "time", lambda: clock[0])
    real = runtime.iter_group_results
    observed = []

    def tracking(groups, *args, **kwargs):
        iterator = real(groups, *args, **kwargs)
        try:
            first = next(iterator)
            clock[0] += 2
            yield first
            state = json.loads(paths_for(cohort)[2].state_path.read_text())
            observed.append(state["completed_indices"])
            yield from iterator
        finally:
            iterator.close()

    monkeypatch.setattr(runtime, "iter_group_results", tracking)
    flags = (
        ["--checkpoint_every_rows", "1", "--checkpoint_every_sec", "100"]
        if trigger == "rows"
        else ["--checkpoint_every_rows", "100", "--checkpoint_every_sec", "1"]
    )
    register.main(args_for(cohort, *flags))
    assert len(observed) == 1
    assert len(observed[0]) == 1


def test_invalid_manifest_dry_run_does_not_create_outputs(cohort, tmp_path):
    manifest = tmp_path / "invalid.yaml"
    manifest.write_text("registration:\n  unknown_option: true\n")
    with pytest.raises(ValueError, match="Unknown registration"):
        register.main(args_for(cohort, "--manifest", str(manifest), "--dry-run"))
    assert not paths_for(cohort)[0].exists()
    assert not (tmp_path / "registration").exists()


def test_missing_error_checkpoint_reprocesses_instead_of_losing_errors(
    monkeypatch, cohort
):
    def fail(groups, *args, **kwargs):
        for key, _ in groups:
            yield key, None, "failure"

    monkeypatch.setattr(runtime, "iter_group_results", fail)
    register.main(args_for(cohort))
    paths_for(cohort)[2].error_checkpoint_path.unlink()
    seen = []

    def track(groups, *args, **kwargs):
        seen.extend(groups)
        yield from fail(groups, *args, **kwargs)

    monkeypatch.setattr(runtime, "iter_group_results", track)
    register.main(args_for(cohort))
    assert len(seen) == 2
    assert len(pd.read_csv(paths_for(cohort)[1])) == 2


def test_qc_and_canonical_paths_survive_resume_and_qc_restoration(monkeypatch, cohort):
    register.main(args_for(cohort))
    output, _, _ = paths_for(cohort)
    first = pd.read_csv(output)
    qc_path = output.with_name("register_qc.csv")
    qc_before = pd.read_csv(qc_path)
    assert set(qc_before.registration_scan_id) == set(first.registration_scan_id)
    assert first.registration_qc_path.eq(str(qc_path.resolve())).all()
    assert first.mask_liver.equals(first.reg_organ_native_path)
    assert first.mask_liver_tumor.equals(first.reg_tumor_native_path)
    assert first.source_mask_liver.equals(pd.read_csv(cohort).mask_liver)
    qc_path.unlink()

    def no_work(groups, *args, **kwargs):
        assert not groups
        yield from ()

    monkeypatch.setattr(runtime, "iter_group_results", no_work)
    register.main(args_for(cohort))
    second = pd.read_csv(output)
    pd.testing.assert_frame_equal(second, first)
    pd.testing.assert_frame_equal(pd.read_csv(qc_path), qc_before)


def test_worker_failures_appear_in_qc(cohort):
    register.main(args_for(cohort, "--timeout_sec", "0.001"))
    output, _, _ = paths_for(cohort)
    qc = pd.read_csv(output.with_name("register_qc.csv"))
    assert len(qc) == 2
    assert qc.registration_status.eq("failed").all()
    assert qc.errors.str.contains("timeout").all()
    assert qc.dice_rigid.isna().all()


def test_custom_qc_output_and_path_collision(cohort, tmp_path):
    qc = tmp_path / "quality.csv"
    register.main(args_for(cohort, "--qc_csv_path", str(qc)))
    assert len(pd.read_csv(qc)) == 2
    with pytest.raises(ValueError, match="differ"):
        register.main(args_for(cohort, "--qc_csv_path", str(cohort)))


@pytest.mark.parametrize(
    "flag,expected",
    [
        (None, True),
        ("--keep_source_segmentation", True),
        ("--no_keep_source_segmentation", False),
    ],
)
def test_keep_source_segmentation_manifest_and_cli(tmp_path, cohort, flag, expected):
    manifest = tmp_path / "keep.yaml"
    manifest.write_text(
        yaml.safe_dump({"registration": {"keep_source_segmentation": True}})
    )
    flags = ["--manifest", str(manifest)]
    if flag:
        flags.append(flag)
    config, _ = register.resolve_config(
        register.normalize_registration_args(args_for(cohort, *flags))
    )
    assert config.keep_source_segmentation is expected
