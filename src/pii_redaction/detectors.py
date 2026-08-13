"""Rule-based PII detection. Pure str -> list[PIIEntity]."""

from __future__ import annotations

import ipaddress
import re
from calendar import month_name
from datetime import date, datetime
from typing import Protocol, runtime_checkable

from pii_redaction.models import (
    PRIORITY_REGEX,
    PRIORITY_VALIDATED,
    PIIEntity,
    PIIType,
    RedactorConfig,
)

# Context words that mean a nearby number is a reference, not PII (PLAN §1.2).
NEGATIVE_CONTEXT: frozenset[str] = frozenset(
    {
        "ticket",
        "order",
        "invoice",
        "receipt",
        "reference",
        "ref",
        "txn",
        "transaction",
        "po",
        "sku",
        "isin",
        "cin",
        "gstin",
        "folio",
        "application",
        "allotment",
        "dp id",
        "client id",
        "section",
        "version",
        "ver",
        "page",
        "regulation",
        "chapter",
        "clause",
        "annexure",
        "schedule",
        "serial",
        "sr",
        "no",
        "number",
        "nos",
    }
)

BIRTH_CUES: frozenset[str] = frozenset(
    {
        "dob",
        "d.o.b",
        "d.o.b.",
        "date of birth",
        "birth date",
        "born",
        "birthday",
        "birthdate",
    }
)

_MONTHS = {m.lower(): i for i, m in enumerate(month_name) if m}
_MONTHS.update({m[:3].lower(): i for m, i in list(_MONTHS.items()) if len(m) >= 3})

# Connectors allowed between a context label and the numeric value.
_LABEL_GAP = re.compile(
    r"^[\s:#\-./]*(?:no\.?|nos\.?|number|num\.?|#)*[\s:#\-./]*$",
    re.IGNORECASE,
)
_NON_DIGIT = re.compile(r"\D")
_YEARISH = re.compile(r"20\d{2}")
_VERSIONISH_PREFIX = re.compile(r"(?:version|section|ver|v)\s*$", re.IGNORECASE)


@runtime_checkable
class Detector(Protocol):
    name: str
    pii_type: PIIType
    priority: int

    def detect(self, text: str, config: RedactorConfig) -> list[PIIEntity]: ...


def preceding_label(text: str, start: int, window: int = 40) -> str | None:
    """Return the nearest context label immediately before ``start``, if any."""
    left = text[max(0, start - window) : start]
    lowered = left.lower()
    for label in sorted(NEGATIVE_CONTEXT | BIRTH_CUES, key=len, reverse=True):
        idx = lowered.rfind(label)
        if idx < 0:
            continue
        # Avoid matching a short label inside a longer word (e.g. "no" in "technology")
        if idx > 0 and lowered[idx - 1].isalnum():
            continue
        after = lowered[idx + len(label) :]
        if _LABEL_GAP.fullmatch(after):
            return label
    return None


def has_birth_cue(text: str, start: int, window: int = 40) -> bool:
    left = text[max(0, start - window) : start].lower()
    return any(cue in left for cue in sorted(BIRTH_CUES, key=len, reverse=True))


def luhn_valid(digits: str) -> bool:
    cleaned = _NON_DIGIT.sub("", digits)
    if not cleaned.isdigit() or not (13 <= len(cleaned) <= 19):
        return False
    total = 0
    for i, ch in enumerate(cleaned[::-1]):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def card_brand_prefix(digits: str) -> str:
    d = _NON_DIGIT.sub("", digits)
    if d.startswith("4"):
        return "visa"
    if d.startswith(("51", "52", "53", "54", "55")) or (
        len(d) >= 4 and 2221 <= int(d[:4]) <= 2720
    ):
        return "mastercard"
    if d.startswith(("34", "37")):
        return "amex"
    if d.startswith(("6011", "65")) or d.startswith("64"):
        return "discover"
    return "unknown"


def ssn_structure_valid(value: str) -> bool:
    digits = _NON_DIGIT.sub("", value)
    if len(digits) != 9:
        return False
    area = int(digits[:3])
    group = int(digits[3:5])
    serial = int(digits[5:])
    if area == 0 or area == 666 or area >= 900:
        return False
    return not (group == 0 or serial == 0)


def parse_calendar_date(value: str) -> date | None:
    value = value.strip()
    patterns = (
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y-%m-%d",
        "%d/%m/%y",
        "%m/%d/%y",
    )
    for fmt in patterns:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", value)
    if m:
        day, month_s, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        month = _MONTHS.get(month_s)
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                return None
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", value)
    if m:
        month_s, day, year = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        month = _MONTHS.get(month_s)
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                return None
    return None


def plausible_dob(d: date, today: date | None = None) -> bool:
    today = today or date.today()
    age = today.year - d.year - ((today.month, today.day) < (d.month, d.day))
    return 0 <= age <= 120 and d.year >= 1900


class RegexDetector:
    """Base class for compiled-pattern detectors with an optional validate hook."""

    name: str
    pii_type: PIIType
    priority: int
    pattern: re.Pattern[str]
    group: str | None = "value"

    def validate(self, match: re.Match[str], text: str, config: RedactorConfig) -> bool:
        return True

    def normalise(self, match: re.Match[str]) -> str | None:
        if self.group and self.group in match.re.groupindex:
            return match.group(self.group)
        return match.group(0)

    def _span(self, match: re.Match[str]) -> tuple[int, int, str]:
        if self.group and self.group in match.re.groupindex:
            return match.start(self.group), match.end(self.group), match.group(self.group)
        return match.start(0), match.end(0), match.group(0)

    def detect(self, text: str, config: RedactorConfig) -> list[PIIEntity]:
        if self.pii_type not in config.enabled_types:
            return []
        results: list[PIIEntity] = []
        for match in self.pattern.finditer(text):
            if not self.validate(match, text, config):
                continue
            if self.normalise(match) is None:
                continue
            start, end, matched = self._span(match)
            results.append(
                PIIEntity(
                    pii_type=self.pii_type,
                    text=matched,
                    start=start,
                    end=end,
                    source=f"regex:{self.name}",
                    confidence=1.0,
                    priority=self.priority,
                )
            )
        return results


_REGISTRY: dict[PIIType, list[Detector]] = {}


def register(detector: Detector) -> Detector:
    """Register a detector *instance* under its PIIType."""
    existing = _REGISTRY.setdefault(detector.pii_type, [])
    for other in existing:
        if other.name == detector.name:
            raise ValueError(
                f"duplicate detector registration: {detector.pii_type}/{detector.name}"
            )
    existing.append(detector)
    return detector


def get_detectors(config: RedactorConfig) -> list[Detector]:
    detectors: list[Detector] = []
    for pii_type, group in _REGISTRY.items():
        if pii_type not in config.enabled_types:
            continue
        detectors.extend(group)
    if config.use_ner:
        # Lazy import keeps `import pii_redaction.detectors` and `--help` spaCy-free.
        from pii_redaction.ner import NERDetector

        ner = NERDetector(
            model_name=config.ner_model,
            confidence_threshold=config.ner_confidence_threshold,
            max_doc_freq=config.ner_max_doc_freq,
        )
        if ner.emits_for(config.enabled_types):
            detectors.append(ner)
    detectors.sort(key=lambda d: (-d.priority, d.name))
    return detectors


def iter_registry() -> list[Detector]:
    out: list[Detector] = []
    for group in _REGISTRY.values():
        out.extend(group)
    return out


def _reject_reference(text: str, start: int, config: RedactorConfig) -> bool:
    if config.redact_reference_numbers:
        return False
    label = preceding_label(text, start)
    return label is not None and label in NEGATIVE_CONTEXT


class EmailDetector(RegexDetector):
    name = "email"
    pii_type = PIIType.EMAIL
    priority = PRIORITY_REGEX
    pattern = re.compile(
        r"(?<![A-Za-z0-9._%+-])"
        r"(?P<value>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
        # Trailing '.' is sentence punctuation, not part of the local/domain charset.
        r"(?![A-Za-z0-9_%+-])"
    )

    def validate(self, match: re.Match[str], text: str, config: RedactorConfig) -> bool:
        value = match.group("value")
        local, _, domain = value.partition("@")
        if not local or not domain or "." not in domain:
            return False
        tld = domain.rsplit(".", 1)[-1]
        if len(tld) < 2:
            return False
        return not (domain.lower() == "localhost" or tld.lower() == "localhost")


class PhoneDetector(RegexDetector):
    name = "phone"
    pii_type = PIIType.PHONE
    priority = PRIORITY_REGEX
    pattern = re.compile(
        r"(?<!\w)"
        r"(?P<value>"
        r"(?:"
        r"(?:\+\s*91[\s\-]*)?(?:\d[\s\-]*){8,12}\d"
        r"|(?:\+?\s*1[\s\-.]*)?\(?\d{3}\)?[\s\-.]+\d{3}[\s\-.]+\d{4}"
        r")"
        r"(?:\s*(?:x|ext\.?)\s*\d{1,5})?"
        r")"
        r"(?!\w)"
    )

    def validate(self, match: re.Match[str], text: str, config: RedactorConfig) -> bool:
        start, _, value = self._span(match)
        if _reject_reference(text, start, config):
            return False
        digits = _NON_DIGIT.sub("", value)
        if not (7 <= len(digits) <= 15):
            return False
        return not bool(_YEARISH.fullmatch(digits))


class SSNDetector(RegexDetector):
    name = "ssn"
    pii_type = PIIType.SSN
    priority = PRIORITY_VALIDATED
    pattern = re.compile(r"(?<!\d)(?P<value>\d{3}[\s\-]\d{2}[\s\-]\d{4})(?!\d)")

    def validate(self, match: re.Match[str], text: str, config: RedactorConfig) -> bool:
        start, _, value = self._span(match)
        if _reject_reference(text, start, config):
            return False
        return ssn_structure_valid(value)


class CreditCardDetector(RegexDetector):
    name = "credit_card"
    pii_type = PIIType.CREDIT_CARD
    priority = PRIORITY_VALIDATED
    pattern = re.compile(r"(?<!\d)(?P<value>(?:\d[ \-]*){12,18}\d)(?!\d)")

    def validate(self, match: re.Match[str], text: str, config: RedactorConfig) -> bool:
        start, _, value = self._span(match)
        if _reject_reference(text, start, config):
            return False
        digits = _NON_DIGIT.sub("", value)
        if not (13 <= len(digits) <= 19):
            return False
        return luhn_valid(digits)


class IPAddressDetector(RegexDetector):
    name = "ip_address"
    pii_type = PIIType.IP_ADDRESS
    priority = PRIORITY_VALIDATED
    pattern = re.compile(
        r"(?<![\w.])(?P<value>"
        r"(?:(?:\d{1,3}\.){3}\d{1,3})"
        r"|(?:[A-Fa-f0-9]{0,4}:){2,7}[A-Fa-f0-9]{0,4}"
        # Allow trailing sentence punctuation ('.') without truncating compressed IPv6.
        r")(?![\w:])"
    )

    def validate(self, match: re.Match[str], text: str, config: RedactorConfig) -> bool:
        start, _, value = self._span(match)
        if _reject_reference(text, start, config):
            return False
        left = text[max(0, start - 16) : start].lower()
        if _VERSIONISH_PREFIX.search(left):
            return False
        try:
            ipaddress.ip_address(value)
        except ValueError:
            return False
        return True


class DOBDetector(RegexDetector):
    name = "dob"
    pii_type = PIIType.DOB
    priority = PRIORITY_VALIDATED
    pattern = re.compile(
        r"(?<!\w)(?P<value>"
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r"|\d{4}-\d{2}-\d{2}"
        r"|\d{1,2}\s+[A-Za-z]+\s+\d{4}"
        r"|[A-Za-z]+\s+\d{1,2},?\s+\d{4}"
        r")(?!\w)"
    )

    def validate(self, match: re.Match[str], text: str, config: RedactorConfig) -> bool:
        start, _, value = self._span(match)
        if not has_birth_cue(text, start):
            return False
        parsed = parse_calendar_date(value)
        if parsed is None:
            return False
        return plausible_dob(parsed)


class AddressDetector(RegexDetector):
    name = "address"
    pii_type = PIIType.ADDRESS
    priority = PRIORITY_REGEX
    pattern = re.compile(
        r"(?P<value>"
        r"\d{1,5}\s+"
        r"[^\n]{3,80}?"
        r"(?:Road|Rd\.?|Street|St\.?|Lane|Ln\.?|Marg|Nagar|Sector|Block|"
        r"Avenue|Ave\.?|Boulevard|Blvd\.?|Floor|Apt\.?|Apartment|"
        r"P\.?\s*O\.?\s*Box)"
        r"[^\n]{0,40}?"
        r"\d{5,6}"
        r")",
        re.IGNORECASE,
    )

    def validate(self, match: re.Match[str], text: str, config: RedactorConfig) -> bool:
        value = match.group("value")
        if len(value) < 10 or "\n" in value:
            return False
        return any(ch.isdigit() for ch in value)


# Register instances (order here is documentation; get_detectors sorts by priority).
register(EmailDetector())
register(PhoneDetector())
register(AddressDetector())
register(SSNDetector())
register(CreditCardDetector())
register(IPAddressDetector())
register(DOBDetector())

# PIITypes that have at least one rule-based detector (FULL_NAME/COMPANY are NER).
RULE_BASED_TYPES: frozenset[PIIType] = frozenset(
    {
        PIIType.EMAIL,
        PIIType.PHONE,
        PIIType.ADDRESS,
        PIIType.SSN,
        PIIType.CREDIT_CARD,
        PIIType.IP_ADDRESS,
        PIIType.DOB,
    }
)

__all__ = [
    "BIRTH_CUES",
    "Detector",
    "NEGATIVE_CONTEXT",
    "RULE_BASED_TYPES",
    "RegexDetector",
    "card_brand_prefix",
    "get_detectors",
    "has_birth_cue",
    "iter_registry",
    "luhn_valid",
    "parse_calendar_date",
    "plausible_dob",
    "preceding_label",
    "register",
    "ssn_structure_valid",
]
