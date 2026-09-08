"""Intra-visit, intra-modal organ registration and tumor consensus.

The public library API consists of :class:`RegistrationConfig` and
:func:`register_cohort`. The ``register`` module implements the CLI, while
``runner`` adds checkpoints and workers around the core ``pipeline``.
SimpleITK is loaded only when image processing starts. Original mask paths are
retained in ``source_mask_*`` columns.
"""

from .config import RegistrationConfig
from .pipeline import register_cohort

__all__ = ["RegistrationConfig", "register_cohort"]
