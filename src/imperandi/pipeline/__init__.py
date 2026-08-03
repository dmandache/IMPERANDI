"""Pipeline contracts and execution engine."""

from .base import PipelineStage, RunContext, StageResult
from .runner import PipelineRunner

__all__ = ["PipelineRunner", "PipelineStage", "RunContext", "StageResult"]
