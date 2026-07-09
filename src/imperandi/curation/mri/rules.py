"""
Minimal rule set for MRI volume curation.

Keep this file deliberately small and editable. The goal is to make the
classification logic transparent rather than exhaustive.
"""

# Regex helpers ---------------------------------------------------------------
# Separators frequently found in protocol names: space, underscore, dot, plus,
# hyphen, slash. Example: ART-PORT, ART_PORT, art port, mask+multiart.
SEP = r"[\s_.+\-/]*"


def token(pattern: str) -> str:
    """Token-like boundary that treats '_' and '-' as separators."""
    return rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])"


# Generic/non-diagnostic ------------------------------------------------------
RX_LOCALIZER = token(r"locali[sz]er|scout|survey|rep[eè]rage|topogram|calibration|cal\s*body")
RX_KEY_IMAGES = token(
    r"key\s*images?|ko|snapshot|screen\s*save|capture|processed\s*images"
    r"|screen[\s_-]*saves?|images?[\s_-]*cl[eé]s?"
)
RX_SUBTRACTION = token(r"sub|subtraction|soustraction|sous")
RX_MIP_MPR = token(r"mip|mpr|reformat|reconstruction|recon")
RX_QUANT_OR_REPORT = token(r"quant|r2\*|r2star|fat\s*fraction|carto|map|mapping|report|result|dose")


# Sequence families -----------------------------------------------------------
RX_SEQUENCE_DWI = token(
    r"adc|apparent[\s_-]*diffusion[\s_-]*coefficient"
    r"|dwi|dw[\s_-]*epi|dwepi|diff(?:usion)?|dif|ivim|dti"
    r"|rec[\s_-]*b[\s_-]*[0-9]{1,4}"
    r"|b[\s_-]*[0-9]{1,4}"
)

# T1 3D GRE families. Dixon alone is included because liver multiphase T1 is
# frequently named only as VIBE/LAVA/mDIXON/DIXON.
RX_SEQUENCE_T1 = token(
    r"t1|vibe|lava|thrive|ethrive|twist|grasp|dynava|fspgr|spgr|tfe"
    r"|m?dixon|q?dixon|e?dixon|3d\s*t1"
)
RX_SEQUENCE_T1_CONTRAST = token(r"gado|post\s*(?:iv|gado|contrast)|ce\s*t1")

RX_SEQUENCE_T2 = token(
    r"t2|tse|fse|ssfse|haste|blade|propeller|spir|spair|mrcp|bili|bhte"
)


# Perfusion / contrast phase labels ------------------------------------------
RX_PHASE_PRECONTRAST = token(
    r"pre(?:\s*gado|\s*contrast|\s*iv)?|pregado|native|avant"
    r"|sans\s*(?:iv|inj|injection|gado|contraste)"
    r"|ss\s*iv|non\s*(?:injecte|injected|contrast|gado)"
)

RX_PHASE_ART_PORT_DYNAMIC = token(
    rf"art{SEP}port|art{SEP}portal|arterio{SEP}portal"
)
RX_PHASE_MASK_MULTIART_DYNAMIC = token(
    rf"mask{SEP}multi{SEP}art|mask\+multiart|masque{SEP}multi{SEP}art|multi{SEP}art"
)

RX_PHASE_ARTERIAL = token(r"art(?:eriel|erial|[eé]rielle?)?|artery|aorte|hepatic\s*arter")
RX_PHASE_PORTAL = token(r"port(?:al)?|vein|veine|venous|ven|portal\s*venous|vp|pv")

# Put 2h/hepatobiliary before generic delayed in code.
RX_PHASE_HEPATOBILIARY = token(r"hepatobiliary|hbp|bhp|tardif\s*\+?\s*2\s*h|2\s*h|120\s*min")
RX_PHASE_DELAYED = token(
    r"tard(?:if|ive)?|delay(?:ed)?|late|equilibrium|eq|interstitiel"
    r"|3\s*(?:min|mn)|4\s*(?:min|mn)|5\s*(?:min|mn)|10\s*(?:min|mn)"
)
RX_PHASE_GENERIC_DYNAMIC = token(r"dyn|dynamic|perfusion|multi\s*phase|mph|ph\s*\d+|4d")


# Feature extraction ----------------------------------------------------------
RX_PLANE_AXIAL = token(r"ax|axial|tra|trans|transverse")
RX_PLANE_CORONAL = token(r"cor|coronal")
RX_PLANE_SAGITTAL = token(r"sag|sagittal")

RX_T2_FATSAT = token(r"fs|fat\s*sat|fatsat|spair|spir|stir|tirm")
RX_T2_MOTION_ROBUST = token(r"blade|propeller|multivane|radial|pace|navigator|rtr|trigger")
RX_T2_HASTE_SSFSE = token(r"haste|ssfse|single\s*shot")
RX_T2_TSE_FSE = token(r"tse|fse|fast\s*spin|turbo\s*spin")
RX_T2_MRCP_BILIARY = token(r"mrcp|bili|cholangi")

RX_T1_3D_GRE = token(r"3d|vibe|lava|thrive|ethrive|twist|grasp|fspgr|spgr|tfe|dixon|mdixon")
RX_T1_DYNAMIC = token(r"dyn|dynamic|4d|twist|grasp|mph|multi\s*phase|ph\s*\d+|art|port|tardif|gado")
RX_BREATH_HOLD = token(r"bh|apnee|apn[eé]e|breath\s*hold")
RX_RESP_TRIGGERED = token(r"pace|trigger|triggered|resp|rtr|navigator")

# Dixon components ------------------------------------------------------------
RX_DIXON_CONTEXT = token(r"dixon|mdixon|lava\s*flex|flex")
RX_DIXON_ALL = token(r"all|all_bh")
RX_DIXON_WATER = token(r"w|water|eau")
RX_DIXON_IN = token(r"in|ip|inphase|in\s*phase|eco\s*0")
RX_DIXON_OPPOSED = token(r"opp|opposed|out|op|opposed\s*phase|eco\s*1")
RX_DIXON_FAT = token(r"f|fat|graisse")
RX_DIXON_FAT_FRACTION = token(r"fat\s*fraction|ff|quant")
RX_DIXON_R2STAR = token(r"r2\*|r2star")


# Priorities ------------------------------------------------------------------
# These are only selection heuristics, not clinical truth.
T1_PHASE_PRIORITY = {
    "PORTAL_VENOUS": 110,
    "ARTERIAL": 105,
    "DELAYED": 95,
    "HEPATOBILIARY": 90,
    "PRECONTRAST": 80,
    "OTHER": 0,
}

# Explicit phase labels should beat phases inferred from a dynamic container.
T1_PHASE_SOURCE_PRIORITY = {
    "explicit_text": 40,
    "explicit_text_art_port_single": 25,
    "volume_order_art_port": 10,
    "volume_order_mask_multiart": 10,
    "volume_order": 0,
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
