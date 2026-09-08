"""Intra-visit, intra-modal organ registration and tumor consensus.

SimpleITK is loaded only when processing images. Native input paths are preserved.
"""

from .config import RegistrationConfig
from .pipeline import register_cohort

__all__ = ["RegistrationConfig", "register_cohort"]
