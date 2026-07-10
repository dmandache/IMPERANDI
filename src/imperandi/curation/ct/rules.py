"""Rule set for CT volume curation."""

SEP = r"[\s_.+\-/]*"


def token(pattern: str) -> str:
    return rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])"


RX_CT_LOCALIZER = token(r"locali[sz]er|scout|survey|topogram|surview|rep[eè]rage|calibration")
RX_CT_DERIVED_LOW_VALUE = token(
    r"mip|mpr|vr|vrt|volume\s*render|reformat|recon|reconstruction|dose|report|screen|capture|processed|secondary"
)
RX_CT_AXIAL = token(r"ax|axial|tra|trans|transverse")

RX_CT_NATIVE = token(
    r"sans(?:\s*iv|\s*injection|\s*inj|\s*contraste)?"
    r"|non(?:\s*injecte|\s*injected|\s*contrast)"
    r"|native|pre(?:\s*contrast|\s*iv|\s*inj)?|non\s*inject"
)
RX_CT_ARTERIAL = token(r"art(?:eriel|erial|[eé]rielle?)?|artery|aorte|aortic|angio|cta")
RX_CT_PORTAL = token(
    r"portal|porto|portovenous|portal\s*venous|vein|venous|veineux|veineuse|vp|pv|parenchymateux|parenchymal"
)
RX_CT_DELAYED = token(
    r"delay(?:ed)?|delai|d[eé]lai|tardif|tardive|late|equilibrium|equilibre|[eé]quilibre"
    r"|3\s*(?:min|mn)|4\s*(?:min|mn)|5\s*(?:min|mn)|10\s*(?:min|mn)"
)

CT_PHASE_PRIORITY = {
    "PORTAL_VENOUS": 120,
    "ARTERIAL": 105,
    "DELAYED": 90,
    "NATIVE": 80,
    "OTHER": 0,
}
