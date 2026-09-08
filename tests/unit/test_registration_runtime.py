"""Registration runtime contracts: resume, manifests, workers, and interruption."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from imperandi.process.registration import cli, runner, execution
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
    return cli.build_parser().parse_args([str(source), "--timeout_sec", "0", *flags])


def paths_for(source):
    output = source.with_name(source.stem + "_registered.csv")
    errors = output.with_name(output.stem + "_errors.csv")
    return output, errors, build_checkpoint_paths(output, errors, "register")


def test_manifest_overrides_defaults_and_cli(tmp_path, cohort):
    manifest = tmp_path / "custom.yaml"
    manifest.write_text(
        yaml.safe_dump({"registration": {"method": "majority", "affine": True}})
    )
    args = cli.normalize_registration_args(
        args_for(cohort, "--manifest", str(manifest))
    )
    config, _ = cli.resolve_config(args)
    assert config.method == "majority" and config.affine
    override = cli.normalize_registration_args(
        args_for(
            cohort, "--manifest", str(manifest), "--method", "union", "--no_affine"
        )
    )
    config, _ = cli.resolve_config(override)
    assert config.method == "union" and not config.affine
    built_in = cli.normalize_registration_args(
        args_for(cohort, "--manifest", "generic")
    )
    assert isinstance(cli.resolve_config(built_in)[0], RegistrationConfig)


def test_dry_run_no_backend_or_writes(monkeypatch, cohort, tmp_path):
    def unavailable():
        raise AssertionError("Dry run must not load backend")

    monkeypatch.setattr(runner, "backend", unavailable)
    before = set(tmp_path.rglob("*"))
    cli.main(args_for(cohort, "--dry-run"))
    assert set(tmp_path.rglob("*")) == before


def test_finished_run_skips_and_restores_missing_final_csv(monkeypatch, cohort):
    cli.main(args_for(cohort))
    output, errors, paths = paths_for(cohort)
    original = output.read_bytes()
    stat = output.stat().st_mtime_ns

    def forbidden(*args, **kwargs):
        raise AssertionError("No completed group should run")
        yield

    monkeypatch.setattr(runner, "iter_group_results", forbidden)
    cli.main(args_for(cohort))
    assert output.stat().st_mtime_ns == stat
    # Recover a deleted final table from checkpoints without processing images.
    output.unlink()
    monkeypatch.setattr(runner, "iter_group_results", lambda *a, **k: (x for x in ()))
    cli.main(args_for(cohort))
    assert output.read_bytes() == original
    assert errors.exists() and paths.state_path.exists()
    assert json.loads(paths.state_path.read_text())["finished"]


def test_interruption_resumes_only_committed_group(monkeypatch, cohort):
    actual = runner.iter_group_results

    def interrupted(groups, *args, **kwargs):
        iterator = actual(groups, *args, **kwargs)
        try:
            yield next(iterator)
            raise KeyboardInterrupt()
        finally:
            iterator.close()

    monkeypatch.setattr(runner, "iter_group_results", interrupted)
    with pytest.raises(KeyboardInterrupt):
        cli.main(args_for(cohort, "--checkpoint_every_rows", "100"))
    output, _, paths = paths_for(cohort)
    assert not output.exists()
    state = json.loads(paths.state_path.read_text())
    assert len(state["completed_indices"]) == 1
    assert not state.get("finished", False)
    observed = []

    def tracking(groups, *args, **kwargs):
        observed.extend(key for key, _ in groups)
        yield from actual(groups, *args, **kwargs)

    monkeypatch.setattr(runner, "iter_group_results", tracking)
    cli.main(args_for(cohort))
    assert len(observed) == 1
    assert observed[0] not in state["completed_indices"]
    assert len(pd.read_csv(output)) == 2


@pytest.mark.parametrize("mutation", ["delete", "modify"])
def test_missing_artifact_recomputes_only_affected_group(monkeypatch, cohort, mutation):
    cli.main(args_for(cohort))
    output, _, _ = paths_for(cohort)
    first = pd.read_csv(output)
    artifact = Path(first.loc[0, "reg_tumor_native_path"])
    if mutation == "delete":
        artifact.unlink()
    else:
        artifact.write_bytes(artifact.read_bytes() + b"modified")
    actual = runner.iter_group_results
    observed = []

    def tracking(groups, *args, **kwargs):
        observed.extend(key for key, _ in groups)
        yield from actual(groups, *args, **kwargs)

    monkeypatch.setattr(runner, "iter_group_results", tracking)
    cli.main(args_for(cohort))
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
    cli.main(args_for(cohort, *flags))
    if change == "mask":
        row = pd.read_csv(cohort).iloc[0]
        image = sitk.ReadImage(row.mask_liver)
        image[0, 0, 0] = 1
        sitk.WriteImage(image, row.mask_liver)
    elif change == "manifest":
        manifest.write_text("registration:\n  method: union\n")
    else:
        flags += ["--" + change]
    actual = runner.iter_group_results
    seen = []

    def tracking(groups, *args, **kwargs):
        seen.extend(key for key, _ in groups)
        yield from actual(groups, *args, **kwargs)

    monkeypatch.setattr(runner, "iter_group_results", tracking)
    cli.main(args_for(cohort, *flags))
    assert len(seen) == 2


def test_strict_resume_hashes_referenced_inputs_and_artifacts(cohort):
    cli.main(args_for(cohort, "--strict_resume"))
    _, _, paths = paths_for(cohort)
    state = json.loads(paths.state_path.read_text())
    assert len(state["input_fingerprint"]) == 3
    assert all("sha256" in item for item in state["input_fingerprint"])
    assert all(
        "sha256" in item for group in state["artifacts"].values() for item in group
    )


def test_retry_failed_and_preserve_errors(monkeypatch, cohort):
    real = runner.iter_group_results

    def fail(groups, *args, **kwargs):
        for key, _ in groups:
            yield key, None, "deliberate worker failure"

    monkeypatch.setattr(runner, "iter_group_results", fail)
    cli.main(args_for(cohort))
    output, errors, _ = paths_for(cohort)
    assert len(pd.read_csv(errors)) == 2
    assert pd.read_csv(output).registration_status.eq("failed").all()
    monkeypatch.setattr(runner, "iter_group_results", real)
    cli.main(args_for(cohort))
    assert len(pd.read_csv(errors)) == 2
    cli.main(args_for(cohort, "--retry_failed"))
    assert pd.read_csv(errors).empty
    assert pd.read_csv(output).registration_status.eq("reference").all()


def test_real_spawn_workers(cohort):
    cli.main(args_for(cohort, "--num_workers", "2", "--timeout_sec", "30"))
    output, errors, _ = paths_for(cohort)
    assert pd.read_csv(errors).empty
    assert pd.read_csv(output).registration_status.eq("reference").all()


def test_hard_timeout_is_recorded(cohort):
    # Startup itself takes longer than this; verify timed-out processes cannot
    # subsequently publish successful rows or leave active children.
    before = {p.pid for p in execution.mp.active_children()}
    cli.main(args_for(cohort, "--timeout_sec", "0.001", "--num_workers", "2"))
    output, errors, _ = paths_for(cohort)
    assert pd.read_csv(output).registration_status.eq("failed").all()
    assert pd.read_csv(errors).error.str.contains("timeout").all()
    assert {p.pid for p in execution.mp.active_children()} == before


@pytest.mark.parametrize(
    "flag,value",
    [("--num_workers", "0"), ("--timeout_sec", "-1"), ("--checkpoint_every_rows", "0")],
)
def test_runtime_argument_validation(cohort, flag, value):
    with pytest.raises(ValueError):
        cli.normalize_registration_args(args_for(cohort, flag, value))


@pytest.mark.parametrize("trigger", ["rows", "time"])
def test_checkpoint_thresholds_commit_complete_groups(monkeypatch, cohort, trigger):
    from imperandi.utils import run_state

    clock = [1000.0]
    monkeypatch.setattr(run_state, "now_epoch", lambda: clock[0])
    # The shared manager reads time.time through now_epoch; its elapsed clock is
    # advanced after the first group, before the parent checks checkpoint limits.
    monkeypatch.setattr(run_state.time, "time", lambda: clock[0])
    real = runner.iter_group_results
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

    monkeypatch.setattr(runner, "iter_group_results", tracking)
    flags = (
        ["--checkpoint_every_rows", "1", "--checkpoint_every_sec", "100"]
        if trigger == "rows"
        else ["--checkpoint_every_rows", "100", "--checkpoint_every_sec", "1"]
    )
    cli.main(args_for(cohort, *flags))
    assert len(observed) == 1
    assert len(observed[0]) == 1


def test_invalid_manifest_dry_run_does_not_create_outputs(cohort, tmp_path):
    manifest = tmp_path / "invalid.yaml"
    manifest.write_text("registration:\n  unknown_option: true\n")
    with pytest.raises(ValueError, match="Unknown registration"):
        cli.main(args_for(cohort, "--manifest", str(manifest), "--dry-run"))
    assert not paths_for(cohort)[0].exists()
    assert not (tmp_path / "registration").exists()


def test_missing_error_checkpoint_reprocesses_instead_of_losing_errors(
    monkeypatch, cohort
):
    def fail(groups, *args, **kwargs):
        for key, _ in groups:
            yield key, None, "failure"

    monkeypatch.setattr(runner, "iter_group_results", fail)
    cli.main(args_for(cohort))
    paths_for(cohort)[2].error_checkpoint_path.unlink()
    seen = []

    def track(groups, *args, **kwargs):
        seen.extend(groups)
        yield from fail(groups, *args, **kwargs)

    monkeypatch.setattr(runner, "iter_group_results", track)
    cli.main(args_for(cohort))
    assert len(seen) == 2
    assert len(pd.read_csv(paths_for(cohort)[1])) == 2
