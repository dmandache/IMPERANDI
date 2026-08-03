"""Patient identity extraction, linkage, and pseudonymization."""

from .resolver import IdentityResult, resolve_patient_identities

__all__ = ["IdentityResult", "resolve_patient_identities"]
