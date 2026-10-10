"""Canonical CT/MR metadata vocabulary and contrast-phase policy.

All protocol words use token() and all optional protocol separators use SEP.
Phase names take precedence over timing; complete durations are parsed once so
a seconds component cannot be mistaken for a separate acquisition delay.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

SEP_CHARS = r"[\s_.+\-/]"
SEP = rf"{SEP_CHARS}*"


def token(*patterns: str) -> str:
    """Match whole terms, case-insensitively, with Unicode-aware boundaries.

    Underscores are separators, not word characters. Alternatives may themselves
    be token patterns, which lets modality extensions reuse complete shared rules.
    """
    return rf"(?i:(?<![^\W_])(?:{'|'.join(patterns)})(?![^\W_]))"


# Non-diagnostic images and reconstructions, independent of modality.
RX_LOCALIZER = token(
    rf"loc|loca|locali[sz]er|scout|scouts|survey|topogram|tomogramme|surview|rep[eéè]rage"
    rf"|calibration|cal{SEP}body|test|phantom|dummy"
)
RX_KEY_IMAGES = token(
    rf"key{SEP}(?:images?|objects?)|key{SEP}object{SEP}selection"
    rf"|ko|kos|kin|kon|snapshot|screen(?:{SEP}saves?)?|capture"
    rf"|processed(?:{SEP}images?)?|secondary"
    rf"|images?{SEP}cl[eé]s?|objets?{SEP}cl[eé]s?"
)
RX_SUBTRACTION = token(r"s?sub(?:traction)?|s?soustr(?:ac|a)tion|s?sous")
RX_MIP_MPR = token(
    rf"mip|minip|mpr|vr|vrt|volume{SEP}render(?:ing|ed)?"
    rf"|reformat(?:ted)?|multiplanar{SEP}reconstruction|recon(?:struction)?"
)
RX_QUANT_OR_REPORT = token(
    rf"quant|carto|maps?|mapping|reports?|results?|reading|histo"
    rf"|iron{SEP}reports?|elasto|error|dose(?:{SEP}reports?)?"
)
RX_DERIVED_LOW_VALUE = token(
    RX_KEY_IMAGES, RX_SUBTRACTION, RX_MIP_MPR, RX_QUANT_OR_REPORT
)

RX_PLANE_AXIAL = token(r"ax|axi|axial(?:e)?|tra|trans|transvers(?:e|al(?:e)?)")
RX_PLANE_CORONAL = token(r"cor|coro|coronal(?:e)?|ecor")
RX_PLANE_SAGITTAL = token(r"sag|sagi|sagittal(?:e)?")
PLANE_RULES = (
    ("AXIAL", RX_PLANE_AXIAL),
    ("CORONAL", RX_PLANE_CORONAL),
    ("SAGITTAL", RX_PLANE_SAGITTAL),
)
RX_IMAGE_ORIGINAL = token(r"original")
RX_IMAGE_PRIMARY = token(r"primary")
RX_IMAGE_DERIVED = token(r"derived")
RX_BREATH_HOLD = token(rf"bh|mbh|apn[eé]e|breath{SEP}hold")
RX_RESP_TRIGGERED = token(
    rf"pace|trigger(?:ed)?|resp(?:i)?|rtr|rt|nav(?:igator)?"
    rf"|respiratory{SEP}(?:trigger(?:ed)?|gat(?:ed|ing))"
)

# Generic contrast vocabulary. Contrast-agent names belong in modality modules.
INJECTION = r"inj(?:ect(?:ed|ion|e|ee|é|ée)?)?|iv"
CONTRAST = r"contrast(?:ed|[eé]e?)?|c\+"
RX_PHASE_NATIVE = token(
    rf"native|natif|un{SEP}enhanced|non{SEP}enhanced"
    rf"|pr[eé](?:{SEP}(?:{CONTRAST}|{INJECTION}))?"
    rf"|avant(?:{SEP}{INJECTION})?"
    rf"|sans(?:{SEP}(?:iv|{INJECTION}|{CONTRAST}))?"
    rf"|without{SEP}(?:{CONTRAST}|{INJECTION})"
    rf"|non{SEP}(?:{INJECTION}|{CONTRAST})"
    rf"|wo{SEP}(?:{INJECTION}|{CONTRAST})"
    rf"|ss{SEP}(?:iv|i)|si|siv|blanc|c-"
    rf"|w/o(?:{SEP}(?:{INJECTION}|{CONTRAST}))?|wo|w/o"
)
ARTERIAL = r"art(?:erial|[eé]riel(?:le)?)?"
RX_PHASE_ARTERIAL = token(
    rf"{ARTERIAL}|artery|aort(?:e|ic|ique)|hepatic{SEP}arter(?:y|ial)?"
    rf"|(?:early|late|multi){SEP}{ARTERIAL}"
)
RX_PHASE_PORTAL = token(
    rf"port(?:al(?:e)?|o)?|porte|porto{SEP}venous|portal{SEP}venous"
    rf"|vein(?:e)?|venous|veneux|veineux|veineuse|vp|pv"
    rf"|parenchymateux|parenchymal|phase{SEP}p"
)
RX_PHASE_DELAYED = token(
    rf"tard(?:if|ive)?|delay(?:ed)?|d[eé]lai|late|equilibrium|[eé]quilibre"
    rf"|eq|interstit(?:iel|ial)|phase{SEP}d"
)
RX_PHASE_POST_CONTRAST = token(
    rf"post(?:{SEP}(?:{CONTRAST}|{INJECTION}))?"
    rf"|avec(?:{SEP}(?:{CONTRAST}|{INJECTION}))?"
    rf"|with(?:{SEP}(?:{CONTRAST}|{INJECTION}))?"
    rf"|w(?:{SEP}(?:{CONTRAST}|{INJECTION}))?"
    rf"|{CONTRAST}|{INJECTION}|enhanced|c\+|\+c"
)
# Direct mixed labels are group context even without generic native/post words.
# Require a separator or conjunction and whole terms: w/o alone is native.
_MIXED_PHASE_JOIN = rf"(?:{SEP_CHARS}+|{SEP}(?:&|,|and|et){SEP})"
# Short W joins exclude underscores and hyphens to preserve Dixon wo_W/wo-W.
_MIXED_W_JOIN = rf"(?:\s+|{SEP}(?:/|\+|&|,|and|et){SEP})"
_MIXED_WITHOUT = rf"w{SEP}o|without"
RX_PHASE_MIXED_PRE_POST = token(
    rf"(?:w|with){_MIXED_W_JOIN}(?:{_MIXED_WITHOUT})"
    rf"|(?:{_MIXED_WITHOUT}){_MIXED_W_JOIN}(?:w|with)"
    rf"|pr[eé]{_MIXED_PHASE_JOIN}post|post{_MIXED_PHASE_JOIN}pr[eé]"
)
RX_PHASE_ORDINAL = token(rf"ph(?:ase)?{SEP}([1-9])(?!\d)")
DYNAMIC = (
    rf"dyn(?:amic|amique)?|dce|dsc|pwi|perf(?:usion)?"
    rf"|multi{SEP}phas(?:e|ic|ique)|mph|m{SEP}ph|ph{SEP}\d+"
    rf"|4d|multi{SEP}{ARTERIAL}|bolus|time{SEP}resolved"
)
RX_PHASE_GENERIC_DYNAMIC = token(DYNAMIC)

# Units are deliberately separate from SEP: ':' is a clock delimiter, and
# '.' / ',' inside a number are decimal points, never a boundary between times.
SECOND = r"(?:s|sec(?:s|ond(?:e)?s?)?)"
MINUTE = r"(?:mn|min(?:ute)?s?)"
HOUR = r"(?:h|hrs?|hours?|heures?)"
NUMBER = r"\d+(?:[.,]\d+)?"
RX_DURATION = token(
    rf"(?<!\d[.,:])(?:t{SEP})?(?:"
    rf"(?P<clock_minutes>\d+):(?P<clock_seconds>[0-5]\d)(?![.,:]\d)"
    rf"(?:{SEP}{MINUTE})?"
    rf"|(?P<hours>{NUMBER}){SEP}{HOUR}"
    rf"(?:{SEP}(?P<hour_minutes>\d{{1,2}})(?:{SEP}{MINUTE})?"
    rf"(?:{SEP}(?P<hour_seconds>{NUMBER}){SEP}{SECOND})?)?"
    rf"|(?P<minutes>{NUMBER}){SEP}{MINUTE}"
    rf"(?:{SEP}(?P<minute_seconds>{NUMBER}){SEP}{SECOND})?"
    rf"|(?P<seconds>{NUMBER}){SEP}{SECOND}"
    rf")"
)


@dataclass(frozen=True)
class PhaseRule:
    label: str
    pattern: str
    description: str
    time_ranges: tuple[tuple[float, float], ...] = ()

    def accepts_time(self, seconds: float) -> bool:
        return any(start <= seconds <= end for start, end in self.time_ranges)


def phase_rules(
    *,
    native: str = RX_PHASE_NATIVE,
    arterial: str = RX_PHASE_ARTERIAL,
    extra: Sequence[PhaseRule] = (),
) -> tuple[PhaseRule, ...]:
    """Build the common precedence, inserting modality phases after native.

    These timing windows are the project's curation policy, not universal
    acquisition standards. Named delayed phases with an adjacent duration must
    also satisfy the delayed window.
    """
    return (
        PhaseRule("NATIVE", native, "native/non-injected"),
        *extra,
        PhaseRule("PORTAL_VENOUS", RX_PHASE_PORTAL, "portal/venous", ((60, 90),)),
        PhaseRule("ARTERIAL", arterial, "arterial", ((20, 35),)),
        PhaseRule("DELAYED", RX_PHASE_DELAYED, "delayed/tardif", ((180, 900),)),
    )


PHASE_RULES = phase_rules()
PHASE_PRIORITY = {
    "PORTAL_VENOUS": 110,
    "ARTERIAL": 105,
    "DELAYED": 95,
    "NATIVE": 80,
    "OTHER": 0,
}


def _durations(text: str) -> list[tuple[re.Match, float]]:
    result = []
    for match in re.finditer(RX_DURATION, text):
        values = {
            name: float(value.replace(",", ".")) if value else 0.0
            for name, value in match.groupdict().items()
        }
        seconds = (
            values["hours"] * 3600
            + (values["clock_minutes"] + values["hour_minutes"] + values["minutes"])
            * 60
            + values["clock_seconds"]
            + values["hour_seconds"]
            + values["minute_seconds"]
            + values["seconds"]
        )
        result.append((match, seconds))
    return result


def match_plane(text: str) -> str | None:
    return next(
        (label for label, pattern in PLANE_RULES if re.search(pattern, text)), None
    )


def has_post_contrast_text(text: str, post: str = RX_PHASE_POST_CONTRAST) -> bool:
    """Recognize postcontrast words without mistaking a water suffix for 'with'."""
    text = re.sub(r"(?i)(?<=[_-])w(?=$|[\s_.-])", " ", text)
    return bool(re.search(post, text))


def has_pre_post_contrast_text(
    text: str,
    native: str = RX_PHASE_NATIVE,
    post: str = RX_PHASE_POST_CONTRAST,
) -> bool:
    """Recognize explicit mixed labels, then combined native/post evidence.

    For example, ``pre contrast`` is purely native: its ``contrast`` token is
    not independent postcontrast evidence. ``pre and post contrast`` is mixed.
    """
    if re.search(RX_PHASE_MIXED_PRE_POST, text):
        return True
    if not re.search(native, text):
        return False
    return has_post_contrast_text(re.sub(native, " ", text), post)


def has_phase_text_evidence(
    text: str,
    *,
    phase_rules: Sequence[PhaseRule] = PHASE_RULES,
    post: str = RX_PHASE_POST_CONTRAST,
    extra_patterns: Sequence[str] = (),
) -> bool:
    """Recognize the field that owns phase matching and subsequent inference."""
    return bool(
        any(re.search(rule.pattern, text) for rule in phase_rules)
        or has_post_contrast_text(text, post)
        or re.search(RX_PHASE_MIXED_PRE_POST, text)
        or any(
            re.search(pattern, text)
            for pattern in (
                RX_PHASE_GENERIC_DYNAMIC,
                RX_PHASE_ORDINAL,
                RX_DURATION,
                *extra_patterns,
            )
        )
    )


def match_phase(
    text: str,
    rules: Sequence[PhaseRule] = PHASE_RULES,
    *,
    post: str = RX_PHASE_POST_CONTRAST,
) -> tuple[str, str] | None:
    """Match explicit words first, then an unambiguous complete duration.

    Mixed pre/post text supplies group context rather than a pure native label.
    Portal precedes arterial, which precedes generic 'late'. Multiple conflicting
    durations do not identify a pure phase.
    """
    durations = _durations(text)
    for rule in rules:
        if rule.label == "NATIVE" and has_pre_post_contrast_text(
            text, rule.pattern, post
        ):
            continue
        for match in re.finditer(rule.pattern, text):
            if rule.label == "DELAYED" and any(
                duration.start() >= match.end()
                and re.fullmatch(SEP, text[match.end() : duration.start()])
                and not rule.accepts_time(seconds)
                for duration, seconds in durations
            ):
                continue
            return rule.label, rule.description

    timed_matches = set()
    for _, seconds in durations:
        matches = {
            (rule.label, rule.description)
            for rule in rules
            if rule.accepts_time(seconds)
        }
        if not matches:
            return None
        timed_matches.update(matches)
    return next(iter(timed_matches)) if len(timed_matches) == 1 else None
