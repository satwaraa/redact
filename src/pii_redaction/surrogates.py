"""Fake value generation and the real→fake consistency map."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from typing import Final

from faker import Faker

from pii_redaction.models import (
    PIIEntity,
    PIIType,
    RedactionError,
    RedactorConfig,
)

_MAX_ATTEMPTS: Final = 48
_HONORIFICS: Final = ("mr.", "mrs.", "ms.", "miss", "dr.", "prof.", "sri", "shri", "smt.")
_COMPANY_SUFFIXES: Final = (
    "pvt. ltd.",
    "pvt ltd",
    "private limited",
    "ltd.",
    "ltd",
    "limited",
    "llc",
    "llp",
    "inc.",
    "inc",
    "corp.",
    "corp",
    "co.",
    "company",
)


class SurrogateCollisionError(RedactionError):
    """Raised when a non-colliding fake cannot be generated within the retry budget."""


def normalise_key(pii_type: PIIType, value: str) -> str:
    """Casefold, collapse whitespace, strip formatting punctuation for map keys."""
    collapsed = re.sub(r"\s+", " ", value.casefold().strip())
    if pii_type in {
        PIIType.PHONE,
        PIIType.SSN,
        PIIType.CREDIT_CARD,
        PIIType.IP_ADDRESS,
    }:
        return re.sub(r"[^a-z0-9:]", "", collapsed)
    if pii_type is PIIType.EMAIL:
        return collapsed
    return re.sub(r"[^\w\s@.]", "", collapsed, flags=re.UNICODE)


def _luhn_checksum_digit(payload: str) -> str:
    total = 0
    # payload is all but check digit; Luhn processes right-to-left with check at end
    for i, ch in enumerate(reversed(payload)):
        n = int(ch)
        # positions from the right starting at 1 for the check digit; payload's
        # rightmost is doubled first
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return str((10 - (total % 10)) % 10)


def _luhn_valid(digits: str) -> bool:
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    body, check = digits[:-1], digits[-1]
    return _luhn_checksum_digit(body) == check


def _reshape_digits(template: str, new_digits: str) -> str:
    chars: list[str] = []
    i = 0
    for ch in template:
        if ch.isdigit():
            if i >= len(new_digits):
                break
            chars.append(new_digits[i])
            i += 1
        else:
            chars.append(ch)
    return "".join(chars)


def _split_honorific(name: str) -> tuple[str | None, str]:
    stripped = name.strip()
    parts = stripped.split(None, 1)
    if len(parts) != 2:
        return None, stripped
    token = parts[0].casefold()
    bare = token.rstrip(".")
    if token in _HONORIFICS or bare in {h.rstrip(".") for h in _HONORIFICS}:
        return parts[0], parts[1]
    return None, stripped


def _name_tokens(name: str) -> list[str]:
    _, rest = _split_honorific(name)
    return [t for t in re.split(r"[^\w]+", rest.casefold(), flags=re.UNICODE) if t]


def _local_forms_from_name(name: str) -> set[str]:
    tokens = _name_tokens(name)
    if not tokens:
        return set()
    first = tokens[0]
    last = tokens[-1] if len(tokens) > 1 else tokens[0]
    forms = {
        first,
        last,
        f"{first}.{last}",
        f"{first}{last}",
        f"{first[0]}.{last}",
        f"{first[0]}{last}",
        f"{first}_{last}",
        "".join(t[0] for t in tokens),
    }
    if len(tokens) > 2:
        forms.add(f"{first}.{tokens[1]}.{last}")
    return forms


def _fake_local_from_name(name: str) -> str:
    tokens = _name_tokens(name)
    if not tokens:
        return "user"
    first = tokens[0]
    last = tokens[-1] if len(tokens) > 1 else tokens[0]
    return f"{first}.{last}"


def _extract_company_suffix(company: str) -> tuple[str, str]:
    lowered = company.casefold()
    for suffix in _COMPANY_SUFFIXES:
        idx = lowered.rfind(suffix)
        if idx >= 0 and idx + len(suffix) == len(lowered):
            return company[:idx].rstrip(), company[idx:]
    return company, ""


def _gen_full_name(original: str, faker: Faker) -> str:
    honorific, _ = _split_honorific(original)
    fake = faker.name()
    _, fake_core = _split_honorific(fake)
    if honorific:
        return f"{honorific} {fake_core}".strip()
    return fake_core


def _gen_phone(original: str, faker: Faker) -> str:
    digits = re.sub(r"\D", "", original)
    if not digits:
        return original
    prefix_len = 0
    stripped = original.lstrip()
    if stripped.startswith("+91"):
        prefix_len = min(2, len(digits))
    elif re.match(r"\+1(?!\d*91)", stripped):
        prefix_len = min(1, len(digits))
    prefix = digits[:prefix_len]
    rest_len = len(digits) - prefix_len
    rest = "".join(str(faker.random_int(0, 9)) for _ in range(rest_len))
    if rest_len and set(rest) <= {"0"}:
        rest = rest[:-1] + "1"
    return _reshape_digits(original, prefix + rest)


def _gen_company(original: str, faker: Faker) -> str:
    _, suffix = _extract_company_suffix(original)
    fake = faker.company()
    fake_base, _ = _extract_company_suffix(fake)
    core = fake_base or fake
    if not suffix:
        return core
    if suffix.startswith((" ", ",")):
        return f"{core.rstrip()}{suffix}"
    return f"{core.rstrip()} {suffix.lstrip()}"


def _gen_address(original: str, faker: Faker) -> str:
    lines = original.splitlines() or [original]
    pin_match = re.search(r"(\d{5,6})\s*$", original)
    pin_len = len(pin_match.group(1)) if pin_match else None
    fake_lines = faker.address().splitlines()
    while len(fake_lines) < len(lines):
        fake_lines.append(faker.street_address())
    fake_lines = fake_lines[: len(lines)]
    if pin_len is not None:
        new_pin = "".join(str(faker.random_int(0, 9)) for _ in range(pin_len))
        if set(new_pin) <= {"0"}:
            new_pin = new_pin[:-1] + "1"
        last = fake_lines[-1]
        if re.search(r"\d{5,6}\s*$", last):
            fake_lines[-1] = re.sub(r"\d{5,6}\s*$", new_pin, last)
        else:
            fake_lines[-1] = f"{last.rstrip()} {new_pin}"
    return "\n".join(fake_lines)


def _gen_ssn(original: str, faker: Faker) -> str:
    area = faker.random_int(1, 899)
    if area == 666:
        area = 665
    group = faker.random_int(1, 99)
    serial = faker.random_int(1, 9999)
    digits = f"{area:03d}{group:02d}{serial:04d}"
    return _reshape_digits(original, digits)


def _card_prefix(digits: str) -> str:
    if digits.startswith("4"):
        return digits[:1]
    if digits.startswith(("34", "37")):
        return digits[:2]
    if len(digits) >= 4 and digits[:2] in {"51", "52", "53", "54", "55"}:
        return digits[:2]
    if len(digits) >= 4 and 2221 <= int(digits[:4]) <= 2720:
        return digits[:4]
    return digits[:6] if len(digits) >= 6 else digits[: max(1, len(digits) - 1)]


def _gen_credit_card(original: str, faker: Faker) -> str:
    digits = re.sub(r"\D", "", original)
    if len(digits) < 13:
        digits = digits.ljust(16, "0")
    prefix = _card_prefix(digits)
    body_len = len(digits) - 1  # exclude check digit
    mid_len = body_len - len(prefix)
    mid = "".join(str(faker.random_int(0, 9)) for _ in range(max(0, mid_len)))
    body = (prefix + mid)[:body_len]
    check = _luhn_checksum_digit(body)
    return _reshape_digits(original, body + check)


def _detect_dob_format(original: str) -> str:
    value = original.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return "%Y-%m-%d"
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", value):
        # Prefer day-first when day > 12, else keep slash layout as %d/%m/%Y
        a, b, _ = value.split("/")
        if int(a) > 12:
            return "%d/%m/%Y"
        if int(b) > 12:
            return "%m/%d/%Y"
        return "%d/%m/%Y"
    if re.fullmatch(r"\d{1,2}-\d{1,2}-\d{4}", value):
        return "%d-%m-%Y"
    if re.fullmatch(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}", value):
        return "%d %B %Y"
    if re.fullmatch(r"[A-Za-z]+\s+\d{1,2},?\s+\d{4}", value):
        return "%B %d, %Y" if "," in value else "%B %d %Y"
    return "%Y-%m-%d"


def _gen_dob(original: str, faker: Faker) -> str:
    fmt = _detect_dob_format(original)
    # Plausible adult DOB
    start = date(1945, 1, 1)
    end = date(2005, 12, 31)
    span = (end - start).days
    chosen = start + timedelta(days=faker.random_int(0, span))
    formatted = chosen.strftime(fmt)
    # Preserve zero-padding style for numeric slash/dash formats loosely
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", original.strip()):
        return chosen.strftime("%d/%m/%Y") if fmt == "%d/%m/%Y" else chosen.strftime(fmt)
    return formatted


def _gen_ip(original: str, faker: Faker) -> str:
    if ":" in original:
        return faker.ipv6()
    parts = original.split(".")
    try:
        first = int(parts[0])
    except (ValueError, IndexError):
        return faker.ipv4()
    if first in {10} or first == 192 or first == 172:
        return faker.ipv4_private()
    return faker.ipv4_public()


def _gen_email(original: str, faker: Faker) -> str:
    local, _, domain = original.partition("@")
    if not domain:
        domain = "example.com"
    # Keep multi-part domains; regenerate local only
    new_local = re.sub(r"[^a-z0-9._+-]", "", faker.user_name().casefold())
    if not new_local:
        new_local = "user"
    # Preserve plus-tag shape if present
    if "+" in local:
        base, _, tag = local.partition("+")
        new_local = f"{new_local}+{re.sub(r'[^a-z0-9]', '', tag.casefold()) or 'tag'}"
    return f"{new_local}@{domain.casefold()}"


Generator = Callable[[str, Faker], str]

GENERATORS: dict[PIIType, Generator] = {
    PIIType.FULL_NAME: _gen_full_name,
    PIIType.EMAIL: _gen_email,
    PIIType.PHONE: _gen_phone,
    PIIType.COMPANY: _gen_company,
    PIIType.ADDRESS: _gen_address,
    PIIType.SSN: _gen_ssn,
    PIIType.CREDIT_CARD: _gen_credit_card,
    PIIType.DOB: _gen_dob,
    PIIType.IP_ADDRESS: _gen_ip,
}


class SurrogateFactory:
    """Seeded, format-preserving surrogate assignment with a consistency map."""

    def __init__(self, config: RedactorConfig) -> None:
        self._config = config
        self._faker = Faker(config.locale)
        self._faker.seed_instance(config.seed)
        self._by_key: dict[tuple[PIIType, str], str] = {}
        self._used_fakes: set[str] = set()
        self._real_values: set[str] = set()
        self._mapping: dict[str, str] = {}
        # casefolded real name -> (original display name, fake name)
        self._name_map: dict[str, tuple[str, str]] = {}

    @property
    def mapping(self) -> Mapping[str, str]:
        return dict(self._mapping)

    def _forbidden(self, candidate: str, original: str) -> bool:
        if candidate == original:
            return True
        reals_cf = {v.casefold() for v in self._real_values}
        used_cf = {v.casefold() for v in self._used_fakes}
        if candidate in self._real_values or candidate.casefold() in reals_cf:
            return True
        return candidate in self._used_fakes or candidate.casefold() in used_cf

    def _generate(self, pii_type: PIIType, original: str) -> str:
        generator = GENERATORS[pii_type]
        for _ in range(_MAX_ATTEMPTS):
            candidate = generator(original, self._faker)
            if not candidate or self._forbidden(candidate, original):
                continue
            if pii_type is PIIType.CREDIT_CARD:
                digits = re.sub(r"\D", "", candidate)
                if not _luhn_valid(digits):
                    continue
            return candidate
        raise SurrogateCollisionError(
            f"could not generate a non-colliding surrogate for {pii_type.value} "
            f"after {_MAX_ATTEMPTS} attempts"
        )

    def _allocate(self, entity: PIIEntity, fake: str | None = None) -> str:
        key = (entity.pii_type, normalise_key(entity.pii_type, entity.text))
        if key in self._by_key:
            return self._by_key[key]
        if fake is not None and not self._forbidden(fake, entity.text):
            value = fake
        else:
            value = self._generate(entity.pii_type, entity.text)
        self._by_key[key] = value
        self._used_fakes.add(value)
        self._mapping[entity.text] = value
        return value

    def _coherent_email(self, email: str) -> str | None:
        local, _, domain = email.partition("@")
        if not domain:
            return None
        base_local = local.casefold().split("+", 1)[0]
        for real_name, fake_name in self._name_map.values():
            if base_local in _local_forms_from_name(real_name):
                new_local = _fake_local_from_name(fake_name)
                if "+" in local:
                    new_local = f"{new_local}+{local.split('+', 1)[1]}"
                return f"{new_local}@{domain}"
        return None

    def assign(self, entities: Sequence[PIIEntity]) -> list[PIIEntity]:
        """Return entities with ``replacement`` set; updates the consistency map."""
        self._real_values = set()
        for entity in entities:
            self._real_values.add(entity.text)
            self._real_values.add(normalise_key(entity.pii_type, entity.text))

        for entity in entities:
            if entity.pii_type is PIIType.FULL_NAME:
                fake = self._allocate(entity)
                self._name_map[entity.text.casefold()] = (entity.text, fake)

        out: list[PIIEntity] = []
        for entity in entities:
            key = (entity.pii_type, normalise_key(entity.pii_type, entity.text))
            if key in self._by_key:
                replacement = self._by_key[key]
                self._mapping.setdefault(entity.text, replacement)
                out.append(entity.with_replacement(replacement))
                continue
            if entity.pii_type is PIIType.EMAIL:
                coherent = self._coherent_email(entity.text)
                replacement = self._allocate(entity, fake=coherent)
            else:
                replacement = self._allocate(entity)
            out.append(entity.with_replacement(replacement))
        return out
