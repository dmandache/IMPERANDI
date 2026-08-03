"""Shared stage contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from imperandi.config.loader import config_hash
from imperandi.config.models import ImperandiConfig
from imperandi.io.tables import table_suffix, write_table


@dataclass
class StageResult:
    artifacts: dict[str, Path] = field(default_factory=dict)
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    qc_flags: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict[str, int | float | str] = field(default_factory=dict)


@dataclass
class RunContext:
    config: ImperandiConfig
    run_dir: Path
    artifacts: dict[str, Path] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        return config_hash(self.config)

    def stage_dir(self, stage_name: str) -> Path:
        path = self.run_dir / stage_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def table_path(self, stage_name: str, artifact_name: str) -> Path:
        suffix = table_suffix(self.config.output.table_format)
        return self.stage_dir(stage_name) / f"{artifact_name}{suffix}"

    def write_table(
        self, stage_name: str, artifact_name: str, frame: pd.DataFrame
    ) -> Path:
        path = self.table_path(stage_name, artifact_name)
        write_table(frame, path)
        self.artifacts[artifact_name] = path
        return path


class PipelineStage(ABC):
    name: str
    requires: frozenset[str] = frozenset()
    produces: frozenset[str] = frozenset()

    def enabled(self, config: ImperandiConfig) -> bool:
        return True

    def mode(self, config: ImperandiConfig) -> str:
        return "active"

    def resume_token(self, config: ImperandiConfig) -> str | None:
        return None

    @abstractmethod
    def run(self, context: RunContext) -> StageResult:
        raise NotImplementedError

    def describe(self, config: ImperandiConfig) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled(config),
            "mode": self.mode(config),
            "requires": sorted(self.requires),
            "produces": sorted(self.produces),
        }
