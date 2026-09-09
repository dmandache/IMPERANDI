"""Intra-visit, intra-modal organ registration and tumor consensus.

The library exposes registration configuration, cohort execution and anatomical
mask QC. ``alignment`` owns mask assessment and pairwise registration; ``cohort``
owns cohort preparation, fusion, and artifact publication. ``reporting`` formats labels,
tables, and logs. ``runtime`` adds checkpoints and workers, ``register`` implements
the CLI, and ``config`` defines settings shared by these modules.
SimpleITK is loaded only when image processing starts. Original mask paths are
retained in ``source_mask_*`` columns.
"""

from .config import RegistrationConfig
from .alignment import OrganMaskQC, assess_organ_mask
from .cohort import register_cohort

__all__ = ["RegistrationConfig", "register_cohort", "OrganMaskQC", "assess_organ_mask"]
