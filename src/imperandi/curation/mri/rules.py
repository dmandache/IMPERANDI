"""MR sequence, contrast-agent, dynamic-acquisition, and Dixon extensions.

Generic exclusions, planes, phase vocabulary, timing, and priorities come from
curation.rules. Multiword MR terms use the same token and SEP conventions.
"""

from imperandi.curation import rules as shared
from imperandi.curation.rules import SEP as SEP, token as token

# Shared rules keep their historical names for callers.
RX_LOCALIZER = shared.RX_LOCALIZER
RX_KEY_IMAGES = shared.RX_KEY_IMAGES
RX_SUBTRACTION = shared.RX_SUBTRACTION
RX_MIP_MPR = shared.RX_MIP_MPR
RX_PLANE_AXIAL = shared.RX_PLANE_AXIAL
RX_PLANE_CORONAL = shared.RX_PLANE_CORONAL
RX_PLANE_SAGITTAL = shared.RX_PLANE_SAGITTAL
RX_BREATH_HOLD = shared.RX_BREATH_HOLD
RX_RESP_TRIGGERED = shared.RX_RESP_TRIGGERED
RX_PHASE_ARTERIAL = shared.RX_PHASE_ARTERIAL
RX_PHASE_PORTAL = shared.RX_PHASE_PORTAL
RX_PHASE_DELAYED = shared.RX_PHASE_DELAYED
RX_PHASE_ORDINAL = shared.RX_PHASE_ORDINAL

# Reusable MR families and contrast agents.
GADOLINIUM = r"gd|gad|gado|gadopdc|gadolinium|eovist|primovist|gadoxetate|gadoxetic"
DIXON = rf"(?:[mqe]{SEP})?dixon"
T1_GRE = (
    rf"vibe|lava|e?thrive|twist|grasp|dynava|fspgr|spgr|tfe"
    rf"|twist{SEP}vibe|lava{SEP}flex|{DIXON}|idea(?:l)?|disco"
)
T2_FAST_SPIN = r"tse|fse|frfse"
T2_SINGLE_SHOT = r"haste|e?ssfse|ssh"
T2_MOTION_ROBUST = r"f?blade|propeller|prop"
RX_DIXON_FAT_FRACTION = token(rf"fat{SEP}fraction|fatfrac|pdff|ff|quant")
RX_DIXON_R2STAR = token(rf"r2\*|r2{SEP}star|t2(?:\*|{SEP}star){SEP}map")
RX_QUANT_OR_REPORT = token(
    shared.RX_QUANT_OR_REPORT, RX_DIXON_FAT_FRACTION, RX_DIXON_R2STAR
)

# Sequence families.
RX_SEQUENCE_DWI = token(
    rf"dwi|dw{SEP}epi|diff(?:usion)?|dif|(?:e|d)?adc"
    rf"|apparent{SEP}diffusion{SEP}coefficient"
    rf"|trace|ivim|dti|ep2d|b{SEP}(?:={SEP})?\d{{1,4}}"
)
RX_SEQUENCE_T1 = token(rf"t1(?:{SEP}weighted)?|3d{SEP}t1", T1_GRE)
RX_SEQUENCE_T1_CONTRAST = token(
    GADOLINIUM,
    rf"t1{SEP}(?:{GADOLINIUM}|{shared.CONTRAST}|post|c\+|fs{SEP}post)",
    rf"post{SEP}(?:iv|{GADOLINIUM}|{shared.CONTRAST})",
    rf"(?:post|ce){SEP}t1",
)
RX_SEQUENCE_T2 = token(
    rf"t2(?:{SEP}weighted)?|{T2_FAST_SPIN}|{T2_SINGLE_SHOT}"
    rf"|{T2_MOTION_ROBUST}|spir|spair|aspir|bhte"
)

# Contrast-agent and hepatobiliary phases are specific to MR.
RX_PHASE_NATIVE = token(
    shared.RX_PHASE_NATIVE,
    rf"(?:pr[eé]|sans|non){SEP}(?:{GADOLINIUM})|mas(?:k|que)",
)
RX_PHASE_POST_CONTRAST = token(shared.RX_PHASE_POST_CONTRAST, GADOLINIUM)
RX_PHASE_HEPATOBILIARY = token(
    rf"bili|hepato{SEP}(?:biliary|biliaire)|hbp|bhp"
    rf"|voie{SEP}biliaire|transitionnel"
)
PHASE_RULES = shared.phase_rules(
    native=RX_PHASE_NATIVE,
    extra=(
        shared.PhaseRule(
            "HEPATOBILIARY",
            RX_PHASE_HEPATOBILIARY,
            "hepatobiliary/2h",
            # Existing policy: 20 minutes or 2h through 2h59m59s.
            ((1200, 1200), (7200, 10799)),
        ),
    ),
)

# These profiles describe MR multivolume containers, not a single pure phase.
RX_PHASE_ART_PORT_DYNAMIC = token(
    rf"{shared.ARTERIAL}{SEP}port(?:al(?:e)?)?|art[eé]rio{SEP}portal(?:e)?"
)
RX_PHASE_MASK_MULTIART_DYNAMIC = token(
    rf"mas(?:k|q(?:ue)?){SEP}(?:(?:multi|\d+){SEP})?{shared.ARTERIAL}"
)
RX_PHASE_GENERIC_DYNAMIC = token(
    shared.RX_PHASE_GENERIC_DYNAMIC,
    RX_PHASE_ART_PORT_DYNAMIC,
    RX_PHASE_MASK_MULTIART_DYNAMIC,
    rf"twist|twist{SEP}vibe|grasp",
)
RX_T1_DYNAMIC = token(
    RX_PHASE_GENERIC_DYNAMIC,
    RX_PHASE_ARTERIAL,
    RX_PHASE_PORTAL,
    RX_PHASE_DELAYED,
    RX_PHASE_POST_CONTRAST,
)

# Features used to rank MR diagnostic candidates.
RX_T2_FATSAT = token(rf"fs|fat{SEP}sat|spair|spir|stir|tirm")
RX_T2_MOTION_ROBUST = token(T2_MOTION_ROBUST, r"multivane|radial", RX_RESP_TRIGGERED)
RX_T2_HASTE_SSFSE = token(T2_SINGLE_SHOT, rf"single{SEP}shot")
RX_T2_TSE_FSE = token(T2_FAST_SPIN, rf"sense|te{SEP}\d+|fast{SEP}spin|turbo{SEP}spin")
RX_T2_MRCP_BILIARY = token(r"mrcp|bili|biliary|biliaire|chol|cholangio|cholangi")
RX_T1_3D_GRE = token(r"3d", T1_GRE)

# Dixon components.
RX_DIXON_CONTEXT = token(DIXON, rf"lava{SEP}flex|flex|idea(?:l)?|disco")
RX_DIXON_ALL = token(rf"all(?:{SEP}bh)?|{DIXON}{SEP}all")
RX_DIXON_WATER = token(r"w|water|wat|eau")
RX_DIXON_IN = token(rf"in|ip|in{SEP}phase|phase{SEP}in|eco{SEP}0")
RX_DIXON_OPPOSED = token(
    rf"opp|opposed|out|op|oop|out{SEP}phase|phase{SEP}out|eco{SEP}1"
)
RX_DIXON_FAT = token(r"f|fat|graisse")

# Only MR's extra phase needs a distinct selection priority.
T1_PHASE_PRIORITY = {**shared.PHASE_PRIORITY, "HEPATOBILIARY": 90}

# Explicit phase labels should beat phases inferred from a dynamic container.
T1_PHASE_SOURCE_PRIORITY = {
    "explicit_text": 40,
    "explicit_text_art_port_late_single": 25,
    "explicit_text_art_port_single": 25,
    "explicit_text_mask_multiart_single": 25,
    "ordinal_context": 15,
    "acquisition_order_art_port_late": 10,
    "acquisition_order_art_port": 10,
    "acquisition_order_mask_multiart": 10,
    "acquisition_order_dixon_component": 5,
    "volume_order_art_port_late": 10,
    "volume_order_art_port": 10,
    "volume_order_mask_multiart": 10,
    "volume_order": 0,
    "exam_context": -5,
    "none": -20,
}

DIXON_COMPONENT_PRIORITY = {
    "WATER": 30,
    "IN_PHASE": 18,
    "OPPOSED_PHASE": 15,
    "NOT_DIXON": 0,
    "DIXON_UNKNOWN": 5,
    "DIXON_ALL": -25,
    "FAT": -25,
    "FAT_FRACTION": -150,
    "R2STAR": -150,
}
