"""Column naming contracts for cohort tables."""

from fnmatch import fnmatchcase

# Require a nonempty logical mask name between the prefix and suffix.
MASK_PATH_COLUMN_PATTERN = "mask_?*_path"


def is_mask_path_column(column: object) -> bool:
    """Identify active mask paths, excluding metrics and source-path backups."""
    return isinstance(column, str) and fnmatchcase(column, MASK_PATH_COLUMN_PATTERN)
