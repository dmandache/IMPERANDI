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
