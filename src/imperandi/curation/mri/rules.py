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
RX_LOCALIZER = token(
    r"loc|loca|locali[sz]er|localiser|scout|survey|rep[eè]rage|topogram|calibration|cal\s*body"
)
RX_KEY_IMAGES = token(
    r"key\s*(?:images?|objects?)|key[\s_-]*object[\s_-]*selection"
    r"|ko|kos|kin|kon|snapshot|screen\s*save|capture|processed\s*images?"
    r"|screen[\s_-]*saves?|images?[\s_-]*cl[eé]s?|objets?[\s_-]*cl[eé]s?"
)
RX_SUBTRACTION = token(r"s?sub(?:traction)?|s?soustr(?:ac|a)tion|s?sous")
RX_MIP_MPR = token(r"mip|mpr|reformat|multiplanar[\s_-]*reconstruction|reconstruction|recon")
RX_QUANT_OR_REPORT = token(
    r"quant|r2\*|r2star|fat\s*fraction|carto|map|mapping|reports?|results?|reading|histo"
    r"|iron[\s_-]*reports?|elasto|error|dose"
)


# Sequence families -----------------------------------------------------------
RX_SEQUENCE_DWI = token(
    r"dwi|dw[\s_-]*epi|dwepi|diff|diffusion|dif|adc|apparent[\s_-]*diffusion[\s_-]*coefficient"
    r"|trace|ivim|dti|ep2d|b\s*[=_-]?\s*\d{1,4}"
)

# T1 3D GRE families. Dixon alone is included because liver multiphase T1 is
# frequently named only as VIBE/LAVA/mDIXON/DIXON.
RX_SEQUENCE_T1 = token(
    r"t1|vibe|lava|thrive|ethrive|twist|grasp|dynava|fspgr|spgr|tfe"
    r"|twist[\s_-]*vibe|twistvibe|lava[\s_-]*flex|lavaflex"
    r"|dixon|mdixon|qdixon|edixon|m[\s_-]*dixon|q[\s_-]*dixon|e[\s_-]*dixon"
    r"|ideal|t1[\s_-]*weighted|3d\s*t1"
)
RX_SEQUENCE_T1_CONTRAST = token(
    r"t1[\w\s_/\-+]{0,50}(?:post|gd|gado|gad|gadolinium|contrast|c\+|fs[\s_-]*post)"
    r"|gado|gad|gadolinium|post\s*(?:iv|gado|gad|contrast)|ce\s*t1"
)

RX_SEQUENCE_T2 = token(
    r"t2|t2[\s_-]*weighted|tse|fse|frfse|ssfse|essfse|haste|ssh|blade|fblade"
    r"|propeller|prop|spir|spair|aspir|mrcp|s[\s_-]*mrcp|bili|biliary|biliaire"
    r"|chol|cholangio|cs[\s_-]*bili|bhte"
)


# Perfusion / contrast phase labels ------------------------------------------
RX_PHASE_NATIVE = token(
    r"pre(?:[\s_-]*(?:gado|gad|contrast|iv))?"
    r"|native"
    r"|avant(?:[\s_-]*injection)?"
    r"|sans(?:\s*(?:iv|inj|injection|gado|gad|contraste))?"
    r"|ss\s*(?:iv|i)"
    r"|si"
    r"|non[\s_-]*(?:inject(?:ed|e|ee|é|ée)?|contrast[eé]?|gado|gad)"
    r"|blanc"
    r"|masque"
)

RX_PHASE_ART_PORT_DYNAMIC = token(
    rf"art{SEP}port|art{SEP}portal|arterio{SEP}portal"
)
RX_PHASE_MASK_MULTIART_DYNAMIC = token(
    rf"mask{SEP}multi{SEP}art|masque{SEP}multi{SEP}art"
    rf"|masq(?:ue)?{SEP}multi{SEP}art|masq(?:ue)?{SEP}\d+{SEP}art|multi{SEP}art"
)

RX_PHASE_ARTERIAL = token(
    r"late[\s_-]*arterial|early[\s_-]*arterial|multi[\s_-]*art|multiart"
    r"|multi[\s_-]*art(?:eriel|erial)?|art(?:eriel|erial|[eé]rielle?)?|artériel|artérielle"
    r"|arterial|artery|aorte|hepatic\s*arter|phase[\s_-]*a"
    r"|(?:1[5-9]|[2-4][0-9])[\s_.+\-]*(?:s|sec|secs|second|seconds)"
    r"|0[\s_.+\-]*(?:15|20|25|30|35|40|45)[\s_.+\-]*(?:min|mn)"
)
RX_PHASE_PORTAL = token(
    r"port(?:al)?|porto|porte|portal[\s_-]*venous|portovenous|vein|veine|venous|ven|vp|pv"
    r"|phase[\s_-]*p|veneux|veineux|veineuse|parenchymateux|parenchymal"
    r"|[6-9]0[\s_.+\-]*(?:s|sec|secs|second|seconds)|1[\s_.+\-]*(?:min|mn)"
    r"|1[\s_.+\-]*(?:10|20|30)[\s_.+\-]*(?:min|mn)"
)

# Put 2h/hepatobiliary before generic delayed in code.
RX_PHASE_HEPATOBILIARY = token(
    r"hepato[\s_-]*biliary|hepato[\s_-]*biliaire"
    r"|hbp|bhp|eovist|primovist|gadoxetate|gadoxetic|tardif\s*\+?\s*2\s*h|2\s*h"
    r"|10[\s_-]*(?:min|mn)|15[\s_-]*(?:min|mn)|20[\s_-]*(?:min|mn)|120\s*min"
    r"|hb|hbhr|voie[\s_-]*biliaire|transitionnel"
)
RX_PHASE_DELAYED = token(
    r"tard(?:if|ive)?|delay(?:ed)?|delai|délai|late|equilibrium|equilibre|équilibre|eq|interstitiel"
    r"|phase[\s_-]*d|[2-6]\s*(?:min|mn)"
)
RX_PHASE_ORDINAL = token(r"ph(?:ase)?[\s_-]*([1-9])|ph([1-9])")
RX_PHASE_POST_CONTRAST = token(
    r"gado|gad|gadolinium|contrast|contraste|inject(?:ed|ion|e|ee|é|ée)?|post|c\+"
)
RX_PHASE_GENERIC_DYNAMIC = token(
    r"dyn|dynamic|dynamique|dce|dsc|pwi|perfusion|perf|multi\s*phase|multiphase"
    r"|multiphas(?:e|ic)|multiphasique|mph|m[\s_-]*ph|ph\s*\d+|4d|multi[\s_-]*art"
    r"|multiart|art[\s_-]*port|twist|twistvibe|grasp|bolus|time[\s_-]*resolved"
)


# Feature extraction ----------------------------------------------------------
RX_PLANE_AXIAL = token(r"ax|axial|tra|trans|transverse")
RX_PLANE_CORONAL = token(r"cor|coro|coronal|coronale|ecor")
RX_PLANE_SAGITTAL = token(r"sag|sagi|sagittal|sagittale")

RX_T2_FATSAT = token(r"fs|fat\s*sat|fatsat|spair|spir|stir|tirm")
RX_T2_MOTION_ROBUST = token(r"blade|fblade|propeller|prop|multivane|radial|pace|navigator|rtr|trigger")
RX_T2_HASTE_SSFSE = token(r"haste|ssfse|essfse|ssh|single\s*shot")
RX_T2_TSE_FSE = token(r"tse|fse|frfse|sense|te[\s_-]*\d+|fast\s*spin|turbo\s*spin")
RX_T2_MRCP_BILIARY = token(r"mrcp|bili|biliary|biliaire|chol|cholangio|cholangi")

RX_T1_3D_GRE = token(
    r"3d|vibe|twist[\s_-]*vibe|twistvibe|lava|lava[\s_-]*flex|lavaflex|thrive|ethrive"
    r"|twist|grasp|fspgr|spgr|tfe|dixon|mdixon|m[\s_-]*dixon|qdixon|q[\s_-]*dixon"
    r"|edixon|e[\s_-]*dixon|ideal"
)
RX_T1_DYNAMIC = token(
    r"dyn|dynamic|dynamique|4d|twist|twistvibe|grasp|mph|m[\s_-]*ph|multi\s*phase"
    r"|multiphase|multiphas(?:e|ic)|multiphasique|ph\s*\d+|multi[\s_-]*art|multiart"
    r"|art[\s_+\-/]*port|art|port|tardif|gado|gad|gadopdc|bolus|perf|perfusion|dce"
)
RX_BREATH_HOLD = token(r"bh|mbh|apnee|apn[eé]e|breath\s*hold")
RX_RESP_TRIGGERED = token(r"pace|trigger|triggered|resp|respi|rtr|rt|nav|navigator")

# Dixon components ------------------------------------------------------------
RX_DIXON_CONTEXT = token(
    r"dixon|mdixon|m[\s_-]*dixon|qdixon|q[\s_-]*dixon|edixon|e[\s_-]*dixon"
    r"|lava\s*flex|lavaflex|flex|ideal"
)
RX_DIXON_ALL = token(r"all|all_bh|m[\s_-]*dixon[\s_-]*all|mdixon[\s_-]*all|dixon[\s_-]*all")
RX_DIXON_WATER = token(r"w|water|wat|eau")
RX_DIXON_IN = token(r"in|ip|inphase|in[\s_-]*phase|phase[\s_-]*in|eco\s*0")
RX_DIXON_OPPOSED = token(
    r"opp|opposed|out|op|oop|out[\s_-]*phase|phase[\s_-]*out|eco\s*1"
)
RX_DIXON_FAT = token(r"f|fat|graisse")
RX_DIXON_FAT_FRACTION = token(r"fat[\s_-]*fraction|fatfrac|pdff|ff|quant")
RX_DIXON_R2STAR = token(r"r2\*|r2star|t2\*[\s_-]*map|t2star[\s_-]*map")


# Priorities ------------------------------------------------------------------
# These are only selection heuristics, not clinical truth.
T1_PHASE_PRIORITY = {
    "PORTAL_VENOUS": 110,
    "ARTERIAL": 105,
    "DELAYED": 95,
    "HEPATOBILIARY": 90,
    "NATIVE": 80,
    "OTHER": 0,
}

# Explicit phase labels should beat phases inferred from a dynamic container.
T1_PHASE_SOURCE_PRIORITY = {
    "explicit_text": 40,
    "explicit_text_art_port_single": 25,
    "ordinal_context": 15,
    "acquisition_order_art_port": 10,
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
