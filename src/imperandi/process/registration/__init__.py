"""Intra-visit, intra-modal organ registration and tumor consensus.

SimpleITK is loaded only when processing images. Original mask paths are retained in source_mask_* columns.
"""

from .config import RegistrationConfig
from .pipeline import register_cohort

__all__ = ["RegistrationConfig", "register_cohort"]
