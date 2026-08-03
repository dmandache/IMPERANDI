"""Patient identity extraction, linkage, and pseudonymization."""

from .resolver import (
    IdentityResult,
    resolve_patient_identities,
    validate_identity_crosswalk,
)

__all__ = [
    "IdentityResult",
    "resolve_patient_identities",
    "validate_identity_crosswalk",
]
