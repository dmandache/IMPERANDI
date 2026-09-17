"""CT extensions to the canonical CT/MR metadata rules."""

from imperandi.curation import rules as shared
from imperandi.curation.rules import SEP as SEP, token as token

RX_CT_LOCALIZER = shared.RX_LOCALIZER
RX_CT_DERIVED_LOW_VALUE = shared.RX_DERIVED_LOW_VALUE
RX_CT_AXIAL = shared.RX_PLANE_AXIAL
RX_CT_NATIVE = shared.RX_PHASE_NATIVE
RX_CT_PORTAL = shared.RX_PHASE_PORTAL
RX_CT_DELAYED = shared.RX_PHASE_DELAYED

# Angiography identifies the arterial acquisition in CT. In MR, angiography can
# be a separate sequence family, so its name alone must not label a T1 phase.
RX_CT_ARTERIAL = token(shared.RX_PHASE_ARTERIAL, rf"angio(?:graph(?:y|ie))?|ct{SEP}a")
PHASE_RULES = shared.phase_rules(arterial=RX_CT_ARTERIAL)
CT_PHASE_PRIORITY = shared.PHASE_PRIORITY
