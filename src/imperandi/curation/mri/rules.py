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
    r"loc|loca|locali[sz]er|localiser|scout|survey|rep[eè]rage|topogram|calibration|cal\s*body|test|phantom|dummy"
)
RX_KEY_IMAGES = token(
    rf"key{SEP}(?:images?|objects?)"
    rf"|key{SEP}object{SEP}selection"
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
    r"dwi|dw[\s_-]*epi|dwepi|diff|diffusion|dif|(?:e|d)?adc|apparent[\s_-]*diffusion[\s_-]*coefficient"
    r"|trace|ivim|dti|ep2d|b\s*[=_-]?\s*\d{1,4}"
)

# T1 3D GRE families. Dixon alone is included because liver multiphase T1 is
# frequently named only as VIBE/LAVA/mDIXON/DIXON.
RX_SEQUENCE_T1 = token(
    r"t1|vibe|lava|thrive|ethrive|twist|grasp|dynava|fspgr|spgr|tfe"
    r"|twist[\s_-]*vibe|twistvibe|lava[\s_-]*flex|lavaflex"
    r"|dixon|mdixon|qdixon|edixon|m[\s_-]*dixon|q[\s_-]*dixon|e[\s_-]*dixon"
    r"|idea|disco|t1[\s_-]*weighted|3d\s*t1"
)
RX_SEQUENCE_T1_CONTRAST = token(
    r"t1[\w\s_/\-+]{0,50}(?:post|gd|gado|gad|gadolinium|contrast|c\+|fs[\s_-]*post)"
    r"|gado|gad|gadolinium|post\s*(?:iv|gado|gad|contrast)|ce\s*t1"
)
RX_SEQUENCE_T2 = token(
    r"t2(?:[\s_-]*weighted)?"
    r"|tse|fse|frfse|ssfse|essfse"
    r"|haste|ssh"
    r"|blade|fblade"
    r"|propeller|prop"
    r"|spir|spair|aspir"
    r"|bhte"
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
    rf"mas(?:k|q(?:ue)?){SEP}"
    rf"(?:(?:multi|\d+){SEP})?"
    rf"art(?:[ée]ri(?:a|e)l)?"
)

RX_PHASE_ARTERIAL = token(
    r"late[\s_-]*arterial|early[\s_-]*arterial|multi[\s_-]*art|multiart"
    r"|multi[\s_-]*art(?:eriel|erial)?|art(?:eriel|erial|[eé]rielle?)?|artériel|artérielle"
    r"|arterial|artery|aorte|hepatic\s*arter"
    # 20-35 seconds.
    r"|(?:2[0-9]|3[0-5])[\s_.+\-]*(?:s|sec|secs|second|seconds)"
    r"|0[.:,](?:20|25|30|35)[\s_.+\-]*(?:min|mn)"
)
RX_PHASE_PORTAL = token(
    r"port(?:al)?|porto|porte|portovenous"
    r"|portal[\s_-]*venous"
    r"|vein|veine|venous|veneux|veineux|veineuse"
    #r"|parenchymateux|parenchymal"
    r"|phase[\s_-]*p|vp|pv"

    # 60-90 seconds.
    r"|(?<!\d)(?:6\d|7\d|8\d|90)[\s_.+\-]*(?:s|sec(?:ond)?s?)(?!\d)"

    # Exactly 1 minute
    r"|(?<!\d)1[\s_.+\-]*(?:mn|min(?:ute)?s?)(?!\d)"

    # 1:00-1:30 min / 1 min 0-30 s
    r"|(?<!\d)1(?:"
        r"[.:,](?:0?\d|[12]\d|30)[\s_.+\-]*(?:mn|min(?:ute)?s?)"
        r"|"
        r"[\s_.+\-]*(?:mn|min(?:ute)?s?)"
        r"[\s_.+\-]*(?:0?\d|[12]\d|30)[\s_.+\-]*(?:s|sec(?:ond)?s?)"
    r")(?!\d)"
)
# Must be evaluated before generic delayed-phase rules.
RX_PHASE_HEPATOBILIARY = token(
    # Explicit hepatobiliary terminology
    r"hepato[\s_-]*biliary|hepato[\s_-]*biliaire"
    r"|hepatobiliary|hepatobiliaire"
    r"|hbp|bhp"
    r"|voie[\s_-]*biliaire"
    r"|transitionnel"

    # Standalone 20 or 120 minutes. Ten and fifteen minutes are delayed
    # acquisitions, not hepatobiliary phases, in this curation policy.
    r"|(?<![\d:.,_+\-])(?:20|120)"
    r"[\s_.+\-]*(?:mn|min(?:ute)?s?)(?!\w)"

    # 2h / 2 h / 2h30 / 2 h 40 / 2h30min
    r"|(?<!\d)2[\s_.+\-]*h"
    r"(?:[\s_.+\-]*(?:[0-5]?\d)"
    r"(?:[\s_.+\-]*(?:mn|min(?:ute)?s?))?)?"
    r"(?!\w)"
)
RX_PHASE_DELAYED = token(
    # Named delayed phases without a stated duration remain valid. If a
    # duration is stated, it must be covered by the 3-15 minute rule below.
    r"(?:tard(?:if|ive)?|delay(?:ed)?|delai|délai|late|equilibrium|equilibre|équilibre|eq|interstitiel)"
    r"(?![\s_.+\-]*\d+\s*(?:min|mn))"
    r"|phase[\s_-]*d|(?:[3-9]|1[0-5])\s*(?:min|mn)"
)
RX_PHASE_ORDINAL = token(r"ph(?:ase)?[\s_-]*([1-9])|ph([1-9])")
RX_PHASE_POST_CONTRAST = token(
    r"gado|gad|gadolinium|contrast|contraste|inject(?:ed|ion|e|ee|é|ée)?|post|c\+"
    r"|eovist|primovist|gadoxetate|gadoxetic"
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
    r"|lava\s*flex|lavaflex|flex|idea|disco"
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
