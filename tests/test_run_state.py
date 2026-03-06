import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import pandas as pd

from imperandi.utils.run_state import (
    STATE_SCHEMA_VERSION,
    CheckpointManager,
    build_checkpoint_paths,
    load_state,
    merge_with_existing_output,
    prepare_resume_context,
)


def test_build_checkpoint_paths_contract(tmp_path):
    output = tmp_path / "out.csv"
    err = tmp_path / "errors.csv"
    paths = build_checkpoint_paths(output, err, "convert")
    assert paths.state_path == tmp_path / ".out.convert.state.json"
    assert paths.main_checkpoint_path == tmp_path / ".out.convert.checkpoint.csv"
    assert paths.error_checkpoint_path == tmp_path / ".errors.convert.checkpoint.csv"


def test_prepare_resume_context_requires_new_schema(tmp_path):
    output = tmp_path / "out.csv"
    err = tmp_path / "errors.csv"
    args = argparse.Namespace(
        checkpoint_every_rows=1,
        checkpoint_every_sec=1,
        resume=True,
        strict_resume=False,
    )
    ctx = prepare_resume_context(
        args=args,
        command="phase",
        inputs=[output],
        output_path=output,
        error_path=err,
    )
    assert not ctx["can_resume"]
    assert not ctx["already_finished"]

    manager = CheckpointManager(paths=ctx["paths"], config=ctx["config"])
    df = pd.DataFrame({"a": [1]})
    manager.mark_processed()
    manager.flush(main_df=df, error_df=pd.DataFrame(), completed_indices=[0], force=True)

    ctx2 = prepare_resume_context(
        args=args,
        command="phase",
        inputs=[output],
        output_path=output,
        error_path=err,
    )
    assert ctx2["can_resume"]
    assert not ctx2["already_finished"]


def test_checkpoint_manager_flush_and_finalize(tmp_path):
    output = tmp_path / "out.csv"
    err = tmp_path / "errors.csv"
    args = argparse.Namespace(
        checkpoint_every_rows=2,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )
    ctx = prepare_resume_context(
        args=args,
        command="segment",
        inputs=[output],
        output_path=output,
        error_path=err,
    )
    manager = CheckpointManager(paths=ctx["paths"], config=ctx["config"])
    df = pd.DataFrame({"a": [1], "_source_idx": [0]})
    manager.mark_processed()
    assert not manager.flush(
        main_df=df, error_df=pd.DataFrame(), completed_indices=[0], force=False
    )
    manager.mark_processed()
    assert manager.flush(
        main_df=df, error_df=pd.DataFrame(), completed_indices=[0], force=False
    )
    state = load_state(ctx["paths"].state_path)
    assert state is not None
    assert state["schema_version"] == STATE_SCHEMA_VERSION
    assert state.get("finished") is None

    manager.finalize_state(completed_indices=[0])
    state2 = load_state(ctx["paths"].state_path)
    assert state2 is not None
    assert state2["finished"] is True


def test_prepare_resume_context_ignores_resume_checkpoint_flags_by_default(tmp_path):
    output = tmp_path / "out.csv"
    err = tmp_path / "errors.csv"
    args_first = argparse.Namespace(
        checkpoint_every_rows=1,
        checkpoint_every_sec=1,
        resume=False,
        strict_resume=False,
    )
    ctx_first = prepare_resume_context(
        args=args_first,
        command="parse",
        inputs=[output],
        output_path=output,
        error_path=err,
    )
    manager = CheckpointManager(paths=ctx_first["paths"], config=ctx_first["config"])
    df = pd.DataFrame({"a": [1]})
    manager.mark_processed()
    manager.flush(main_df=df, error_df=pd.DataFrame(), completed_indices=[0], force=True)

    args_second = argparse.Namespace(
        checkpoint_every_rows=999,
        checkpoint_every_sec=999,
        resume=True,
        strict_resume=False,
    )
    ctx_second = prepare_resume_context(
        args=args_second,
        command="parse",
        inputs=[output],
        output_path=output,
        error_path=err,
    )
    assert ctx_second["can_resume"]


def test_prepare_resume_context_marks_already_finished(tmp_path):
    csv_in = tmp_path / "input.csv"
    output = tmp_path / "out.csv"
    err = tmp_path / "errors.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv_in, index=False)
    args = argparse.Namespace(
        checkpoint_every_rows=1,
        checkpoint_every_sec=1,
        resume=True,
        strict_resume=False,
    )
    ctx = prepare_resume_context(
        args=args,
        command="convert",
        inputs=[csv_in],
        output_path=output,
        error_path=err,
    )
    manager = CheckpointManager(paths=ctx["paths"], config=ctx["config"])
    pd.DataFrame({"a": [1]}).to_csv(output, index=False)
    manager.finalize_state(completed_indices=[0])

    resumed = prepare_resume_context(
        args=args,
        command="convert",
        inputs=[csv_in],
        output_path=output,
        error_path=err,
    )
    assert not resumed["can_resume"]
    assert resumed["already_finished"]


def test_prepare_resume_context_finished_without_output_uses_checkpoint_if_available(tmp_path):
    output = tmp_path / "out.csv"
    err = tmp_path / "errors.csv"
    args = argparse.Namespace(
        checkpoint_every_rows=1,
        checkpoint_every_sec=1,
        resume=True,
        strict_resume=False,
    )
    ctx = prepare_resume_context(
        args=args,
        command="convert",
        inputs=[output],
        output_path=output,
        error_path=err,
    )
    manager = CheckpointManager(paths=ctx["paths"], config=ctx["config"])
    manager.mark_processed()
    manager.flush(
        main_df=pd.DataFrame({"a": [1]}),
        error_df=pd.DataFrame(),
        completed_indices=[0],
        force=True,
    )
    manager.finalize_state(completed_indices=[0])

    resumed = prepare_resume_context(
        args=args,
        command="convert",
        inputs=[output],
        output_path=output,
        error_path=err,
    )
    assert resumed["can_resume"]
    assert not resumed["already_finished"]


def test_prepare_resume_context_disables_resume_without_checkpoint(tmp_path):
    output = tmp_path / "out.csv"
    err = tmp_path / "errors.csv"
    args = argparse.Namespace(
        checkpoint_every_rows=1,
        checkpoint_every_sec=1,
        resume=True,
        strict_resume=False,
    )
    ctx = prepare_resume_context(
        args=args,
        command="phase",
        inputs=[output],
        output_path=output,
        error_path=err,
    )
    manager = CheckpointManager(paths=ctx["paths"], config=ctx["config"])
    manager.mark_processed()
    manager.flush(
        main_df=pd.DataFrame({"a": [1]}),
        error_df=pd.DataFrame(),
        completed_indices=[0],
        force=True,
    )

    ctx["paths"].main_checkpoint_path.unlink()
    resumed = prepare_resume_context(
        args=args,
        command="phase",
        inputs=[output],
        output_path=output,
        error_path=err,
    )
    assert not resumed["can_resume"]
    assert not resumed["already_finished"]


def test_merge_with_existing_output_passthrough_when_missing(tmp_path):
    new_df = pd.DataFrame({"nifti_path": ["a.nii.gz"], "x": [1]})
    out = merge_with_existing_output(
        new_df,
        tmp_path / "missing.csv",
        preferred_keys=["nifti_path"],
    )
    pd.testing.assert_frame_equal(out, new_df)


def test_merge_with_existing_output_keyed_merge_preserves_foreign_columns(tmp_path):
    output = tmp_path / "out.csv"
    pd.DataFrame(
        {
            "nifti_path": ["a.nii.gz", "b.nii.gz"],
            "foreign_col": ["keep-a", "keep-b"],
        }
    ).to_csv(output, index=False)
    new_df = pd.DataFrame(
        {
            "nifti_path": ["a.nii.gz", "b.nii.gz"],
            "local_col": [1, 2],
        }
    )

    out = merge_with_existing_output(
        new_df,
        output,
        preferred_keys=["nifti_path"],
    )
    assert "foreign_col" in out.columns
    assert out["foreign_col"].tolist() == ["keep-a", "keep-b"]


def test_merge_with_existing_output_falls_back_to_index_when_lengths_match(tmp_path):
    output = tmp_path / "out.csv"
    pd.DataFrame({"foreign_col": ["a", "b"]}).to_csv(output, index=False)
    new_df = pd.DataFrame({"local_col": [1, 2]})

    out = merge_with_existing_output(
        new_df,
        output,
        preferred_keys=["nifti_path"],
    )
    assert out["foreign_col"].tolist() == ["a", "b"]


def test_merge_with_existing_output_raises_on_unsafe_alignment(tmp_path):
    output = tmp_path / "out.csv"
    pd.DataFrame({"foreign_col": ["a", "b", "c"]}).to_csv(output, index=False)
    new_df = pd.DataFrame({"local_col": [1, 2]})

    try:
        merge_with_existing_output(
            new_df,
            output,
            preferred_keys=["nifti_path"],
            strict=True,
        )
        assert False, "Expected ValueError for unsafe merge"
    except ValueError as exc:
        assert "Cannot safely preserve existing output columns" in str(exc)


def test_merge_with_existing_output_duplicate_key_uses_index_fallback(tmp_path):
    output = tmp_path / "out.csv"
    pd.DataFrame(
        {
            "nifti_path": ["a.nii.gz", "a.nii.gz"],
            "foreign_col": ["keep-1", "keep-2"],
        }
    ).to_csv(output, index=False)
    new_df = pd.DataFrame(
        {
            "nifti_path": ["x.nii.gz", "y.nii.gz"],
            "local_col": [1, 2],
        }
    )

    out = merge_with_existing_output(
        new_df,
        output,
        preferred_keys=["nifti_path"],
    )
    assert out["foreign_col"].tolist() == ["keep-1", "keep-2"]


def test_merge_with_existing_output_unhashable_key_uses_index_fallback(tmp_path):
    output = tmp_path / "out.csv"
    pd.DataFrame({"foreign_col": ["keep-a", "keep-b"]}).to_csv(output, index=False)
    new_df = pd.DataFrame(
        {
            "dicom_path": [["a.dcm", "b.dcm"], ["c.dcm", "d.dcm"]],
            "local_col": [1, 2],
        }
    )

    out = merge_with_existing_output(
        new_df,
        output,
        preferred_keys=["dicom_path"],
    )
    assert out["foreign_col"].tolist() == ["keep-a", "keep-b"]
