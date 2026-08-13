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

# Types whose surrogates must not embed a real person-name token.
_NAME_LIKE_TYPES: Final = frozenset({PIIType.FULL_NAME, PIIType.COMPANY, PIIType.ADDRESS})

# Public suffixes whose second-to-last label is part of the suffix.
_SECOND_LEVEL_SUFFIXES: Final = frozenset(
    {"co", "com", "net", "org", "gov", "edu", "ac", "in", "res"}
)

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
    """Fallback only. SurrogateFactory rewrites the domain via its domain map."""
    local, _, domain = original.partition("@")
    if not domain:
        domain = "example.com"
    new_local = re.sub(r"[^a-z0-9._+-]", "", faker.user_name().casefold())
    if not new_local:
        new_local = "user"
    # Preserve plus-tag shape if present
    if "+" in local:
        base, _, tag = local.partition("+")
        new_local = f"{new_local}+{re.sub(r'[^a-z0-9]', '', tag.casefold()) or 'tag'}"
    return f"{new_local}@{domain.casefold()}"


def split_host(value: str) -> tuple[str, str, str]:
    """Split a URL/host into (scheme+www prefix, registrable label, tld suffix).

    "https://www.hdfcbank.com" -> ("https://www.", "hdfcbank", ".com")
    "in.mpms.mufg.com"         -> ("", "in.mpms.mufg", ".com")
    Multi-part public suffixes (.co.in, .org.in, .co.uk) stay in the suffix so a
    surrogate keeps the same country/registry shape.
    """
    prefix_match = re.match(r"^(https?://)?(www\.)?", value, flags=re.IGNORECASE)
    prefix = prefix_match.group(0) if prefix_match else ""
    host = value[len(prefix) :]
    labels = host.split(".")
    suffix_len = 1
    if len(labels) >= 3 and labels[-2].lower() in _SECOND_LEVEL_SUFFIXES:
        suffix_len = 2
    core = ".".join(labels[:-suffix_len]) or host
    suffix = "." + ".".join(labels[-suffix_len:]) if len(labels) > suffix_len else ""
    return prefix, core, suffix


def _gen_domain(original: str, faker: Faker) -> str:
    """Fallback only. SurrogateFactory routes DOMAIN through its domain map."""
    prefix, _core, suffix = split_host(original)
    return f"{prefix}{faker.domain_word()}{suffix or '.example'}"


Generator = Callable[[str, Faker], str]

GENERATORS: dict[PIIType, Generator] = {
    PIIType.FULL_NAME: _gen_full_name,
    PIIType.EMAIL: _gen_email,
    PIIType.DOMAIN: _gen_domain,
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
        # casefolded real company -> (original display name, fake company)
        self._company_map: dict[str, tuple[str, str]] = {}
        # casefolded real registrable label -> fake label, shared by website
        # domains and email domains so both forms of a company's domain agree.
        self._domain_map: dict[str, str] = {}
        # Person-name tokens present in the document; no surrogate may contain
        # one (see _contains_real_name_token).
        self._real_name_tokens: set[str] = set()

    @property
    def mapping(self) -> Mapping[str, str]:
        return dict(self._mapping)

    def _contains_real_name_token(self, candidate: str) -> bool:
        """True when a surrogate embeds a real person-name token.

        Faker's en_IN company pool emits names like "Yohannan, Hegde and Patla".
        If "Hegde" is a real surname, that fake company puts it back into the
        document. Equality checks miss this; containment catches it.
        """
        if not self._real_name_tokens:
            return False
        tokens = {t.casefold() for t in re.findall(r"[^\W\d_]{4,}", candidate, re.UNICODE)}
        return bool(tokens & self._real_name_tokens)

    def _forbidden(self, candidate: str, original: str) -> bool:
        if candidate == original:
            return True
        if self._contains_real_name_token(candidate):
            return True
        reals_cf = {v.casefold() for v in self._real_values}
        used_cf = {v.casefold() for v in self._used_fakes}
        if candidate in self._real_values or candidate.casefold() in reals_cf:
            return True
        return candidate in self._used_fakes or candidate.casefold() in used_cf

    def _replace_real_name_tokens(self, value: str) -> str:
        """Swap out any real name token a generator happened to emit.

        Repair, not reject: the company and person pools share a surname list,
        so with hundreds of real names a retry loop just exhausts.
        """
        if not self._real_name_tokens:
            return value

        def _swap(match: re.Match[str]) -> str:
            token = match.group(0)
            if token.casefold() not in self._real_name_tokens:
                return token
            for _ in range(8):
                # word() draws from a vocabulary list, not the surname pool.
                candidate = self._faker.word().capitalize()
                if candidate.casefold() not in self._real_name_tokens:
                    return candidate.upper() if token.isupper() else candidate
            return "Redacted"

        return re.sub(r"[^\W\d_]{4,}", _swap, value, flags=re.UNICODE)

    def _generate(self, pii_type: PIIType, original: str) -> str:
        generator = GENERATORS[pii_type]
        for _ in range(_MAX_ATTEMPTS):
            candidate = generator(original, self._faker)
            if candidate and pii_type in _NAME_LIKE_TYPES:
                candidate = self._replace_real_name_tokens(candidate)
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

    def _derive_domain_label(self, core: str) -> str:
        """Fake registrable label for ``core``, reusing the company surrogate.

        "kshinternational" belongs to "KSH International Limited"; if that
        company is already mapped to "Barad Group", the domain becomes
        "baradgroup" so the redacted document stays internally consistent.
        """
        letters = re.sub(r"[^a-z0-9]", "", core.casefold())
        for real_company, fake_company in self._company_map.values():
            company_letters = re.sub(r"[^a-z0-9]", "", real_company.casefold())
            if not company_letters or not letters:
                continue
            if letters in company_letters or company_letters.startswith(letters):
                base, _ = _extract_company_suffix(fake_company)
                candidate = re.sub(r"[^a-z0-9]", "", (base or fake_company).casefold())
                if candidate and candidate not in self._domain_map.values():
                    return candidate
        for _ in range(_MAX_ATTEMPTS):
            candidate = re.sub(r"[^a-z0-9]", "", self._faker.domain_word().casefold())
            if candidate and candidate not in self._domain_map.values():
                return candidate
        raise SurrogateCollisionError(
            f"could not generate a non-colliding domain label after {_MAX_ATTEMPTS} attempts"
        )

    def _fake_host(self, value: str) -> str:
        """Rewrite a host/URL through the shared domain map, keeping its shape."""
        prefix, core, suffix = split_host(value)
        key = core.casefold()
        fake_core = self._domain_map.get(key)
        if fake_core is None:
            fake_core = self._derive_domain_label(core)
            self._domain_map[key] = fake_core
        return f"{prefix}{fake_core}{suffix}"

    def _email_fake(self, email: str) -> str:
        """Fake local part (name-coherent where possible) plus a mapped domain."""
        _, _, domain = email.partition("@")
        coherent = self._coherent_email(email)
        base = coherent if coherent is not None else _gen_email(email, self._faker)
        new_local = base.partition("@")[0]
        if not domain:
            return base
        return f"{new_local}@{self._fake_host(domain)}"

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
        self._real_name_tokens = {
            token.casefold()
            for entity in entities
            if entity.pii_type is PIIType.FULL_NAME
            for token in re.findall(r"[^\W\d_]{4,}", entity.text, re.UNICODE)
        } - {h.rstrip(".") for h in _HONORIFICS}

        for entity in entities:
            if entity.pii_type is PIIType.FULL_NAME:
                fake = self._allocate(entity)
                self._name_map[entity.text.casefold()] = (entity.text, fake)

        # Companies before domains and emails: a domain surrogate is derived
        # from its company's surrogate, so the company must be allocated first.
        for entity in entities:
            if entity.pii_type is PIIType.COMPANY:
                fake = self._allocate(entity)
                self._company_map[entity.text.casefold()] = (entity.text, fake)

        out: list[PIIEntity] = []
        for entity in entities:
            key = (entity.pii_type, normalise_key(entity.pii_type, entity.text))
            if key in self._by_key:
                replacement = self._by_key[key]
                self._mapping.setdefault(entity.text, replacement)
                out.append(entity.with_replacement(replacement))
                continue
            if entity.pii_type is PIIType.EMAIL:
                replacement = self._allocate(entity, fake=self._email_fake(entity.text))
            elif entity.pii_type is PIIType.DOMAIN:
                replacement = self._allocate(entity, fake=self._fake_host(entity.text))
            else:
                replacement = self._allocate(entity)
            out.append(entity.with_replacement(replacement))
        return out
