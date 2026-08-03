"""Dependency-aware, resumable pipeline execution."""

from __future__ import annotations

import json
import logging
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import yaml

from imperandi import __version__
from imperandi.config.loader import (
    config_dependency_hashes,
    config_hash,
    resolved_config,
)
from imperandi.config.models import ImperandiConfig
from imperandi.io.tables import table_schema_path

from .base import PipelineStage, RunContext, StageResult

logger = logging.getLogger(__name__)


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for distribution in [
        "pandas",
        "pyarrow",
        "pydicom",
        "dicom2nifti",
        "nibabel",
        "SimpleITK",
        "TotalSegmentator",
        "pyradiomics",
    ]:
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            continue
    return versions


def _atomic_json_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


class PipelineRunner:
    def __init__(self, config: ImperandiConfig, stages: Iterable[PipelineStage]):
        self.config = config
        self.stages = list(stages)
        self._validate_stage_graph()

    def _validate_stage_graph(self) -> None:
        available: set[str] = set()
        names: set[str] = set()
        for stage in self.stages:
            if stage.name in names:
                raise ValueError(f"Duplicate pipeline stage: {stage.name}")
            names.add(stage.name)
            if not stage.enabled(self.config):
                continue
            missing = set(stage.requires) - available
            if missing:
                raise ValueError(
                    f"Stage {stage.name!r} requires unavailable artifacts: "
                    f"{sorted(missing)}"
                )
            available.update(stage.produces)

    def plan(self) -> list[dict]:
        return [stage.describe(self.config) for stage in self.stages]

    def _create_context(self) -> RunContext:
        digest = config_hash(self.config)
        run_dir = self.config.output.root / "runs" / digest[:12]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved_config(self.config), sort_keys=False),
            encoding="utf-8",
        )
        _atomic_json_write(
            run_dir / "environment.json",
            {
                "imperandi_version": __version__,
                "python": sys.version,
                "platform": platform.platform(),
                "config_hash": digest,
                "config_dependencies": config_dependency_hashes(self.config),
                "dependencies": _dependency_versions(),
            },
        )
        context = RunContext(config=self.config, run_dir=run_dir)
        self._update_run_state(
            context,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
            project=self.config.project.name,
            config_hash=digest,
        )
        return context

    @staticmethod
    def _update_run_state(context: RunContext, **updates) -> None:
        path = context.run_dir / "run.json"
        state = {}
        if path.exists():
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                state = {}
        state.update(updates)
        _atomic_json_write(path, state)

    @staticmethod
    def _state_path(context: RunContext, stage: PipelineStage) -> Path:
        return context.stage_dir(stage.name) / "stage.json"

    def _load_completed_result(
        self, context: RunContext, stage: PipelineStage
    ) -> StageResult | None:
        state_path = self._state_path(context, stage)
        if not self.config.execution.resume or not state_path.exists():
            return None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Ignoring unreadable stage state: %s", state_path)
            return None
        if state.get("status") != "completed":
            return None
        if state.get("resume_token") != stage.resume_token(self.config):
            return None
        recorded_artifacts = state.get("artifacts")
        if not isinstance(recorded_artifacts, dict) or not recorded_artifacts:
            return None
        artifacts = {name: Path(path) for name, path in recorded_artifacts.items()}

        def artifact_exists(path: Path) -> bool:
            if not path.exists():
                return False
            if path.suffix.lower() in {".csv", ".parquet"}:
                return table_schema_path(path).exists()
            return True

        if not all(artifact_exists(path) for path in artifacts.values()):
            return None
        context.artifacts.update(artifacts)
        logger.info("Skipping completed stage %s", stage.name)
        return StageResult(artifacts=artifacts, metrics=state.get("metrics", {}))

    def run(self) -> dict[str, StageResult]:
        context = self._create_context()
        results: dict[str, StageResult] = {}
        rerun_downstream = not self.config.execution.resume
        for stage in self.stages:
            if not stage.enabled(self.config):
                logger.info("Skipping disabled stage %s", stage.name)
                continue

            completed = (
                None
                if rerun_downstream
                else self._load_completed_result(context, stage)
            )
            if completed is not None:
                results[stage.name] = completed
                continue

            rerun_downstream = True
            logger.info("Running stage %s", stage.name)
            state_path = self._state_path(context, stage)
            resume_token = stage.resume_token(self.config)
            _atomic_json_write(
                state_path,
                {
                    "status": "running",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "resume_token": resume_token,
                },
            )
            try:
                result = stage.run(context)
            except Exception as exc:
                _atomic_json_write(
                    state_path,
                    {
                        "status": "failed",
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                        "error": str(exc),
                        "resume_token": resume_token,
                    },
                )
                self._update_run_state(
                    context,
                    status="failed",
                    failed_stage=stage.name,
                    failed_at=datetime.now(timezone.utc).isoformat(),
                    error=str(exc),
                )
                raise

            context.artifacts.update(result.artifacts)
            _atomic_json_write(
                state_path,
                {
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "artifacts": {
                        name: str(path) for name, path in result.artifacts.items()
                    },
                    "metrics": result.metrics,
                    "resume_token": resume_token,
                },
            )
            results[stage.name] = result
        self._update_run_state(
            context,
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            artifacts={name: str(path) for name, path in context.artifacts.items()},
        )
        return results
