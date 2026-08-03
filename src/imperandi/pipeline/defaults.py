"""Default high-level IMPERANDI workflow."""

from __future__ import annotations

from imperandi.config.models import ImperandiConfig

from .runner import PipelineRunner
from .stages import (
    AnnotateStage,
    AssembleStage,
    ConvertStage,
    IdentityStage,
    IndexStage,
    PredictPhaseStage,
    PublishStage,
    RadiomicsStage,
    RegistrationStage,
    ResolveSelectStage,
    SegmentStage,
)


def build_default_runner(config: ImperandiConfig) -> PipelineRunner:
    return PipelineRunner(
        config,
        [
            IndexStage(),
            IdentityStage(),
            AssembleStage(),
            AnnotateStage(),
            ConvertStage(),
            PredictPhaseStage(),
            ResolveSelectStage(),
            SegmentStage(),
            RegistrationStage(),
            RadiomicsStage(),
            PublishStage(),
        ],
    )
