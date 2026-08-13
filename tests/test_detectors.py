"""test_detectors — rule layer: positives, negatives, registry, validators."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from pii_redaction.detectors import (
    RULE_BASED_TYPES,
    Detector,
    RegexDetector,
    card_brand_prefix,
    get_detectors,
    has_birth_cue,
    iter_registry,
    luhn_valid,
    parse_calendar_date,
    plausible_dob,
    preceding_label,
    ssn_structure_valid,
)
from pii_redaction.models import (
    PRIORITY_REGEX,
    PRIORITY_VALIDATED,
    PIIType,
    RedactorConfig,
)


def _cfg(**kwargs: object) -> RedactorConfig:
    return replace(RedactorConfig(use_ner=False), **kwargs)  # type: ignore[arg-type]


def _detector_for(pii_type: PIIType) -> Detector:
    for det in iter_registry():
        if det.pii_type is pii_type:
            return det
    raise AssertionError(f"no detector registered for {pii_type}")


def _assert_hit(
    text: str,
    pii_type: PIIType,
    expected: str,
    config: RedactorConfig | None = None,
) -> None:
    config = config or _cfg()
    entities = _detector_for(pii_type).detect(text, config)
    matched = [e for e in entities if e.text == expected]
    assert matched, f"expected {expected!r} in {[e.text for e in entities]} from {text!r}"
    entity = matched[0]
    assert entity.pii_type is pii_type
    assert text[entity.start : entity.end] == entity.text == expected


def _assert_miss(text: str, pii_type: PIIType, config: RedactorConfig | None = None) -> None:
    config = config or _cfg()
    entities = _detector_for(pii_type).detect(text, config)
    assert entities == [], f"expected no {pii_type} in {text!r}, got {[e.text for e in entities]}"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("contact rashhi.patil@gmail.com today", "rashhi.patil@gmail.com"),
        ("mail first.last+tag@sub.domain.co.in please", "first.last+tag@sub.domain.co.in"),
        ("rashhi.patil@gmail.com", "rashhi.patil@gmail.com"),  # start of string
        ("write to a@b.co", "a@b.co"),  # end-ish
    ],
)
def test_email_positives(text: str, expected: str) -> None:
    _assert_hit(text, PIIType.EMAIL, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("call +91 9876543210 now", "+91 9876543210"),
        ("mobile +91-98765-43210", "+91-98765-43210"),
        ("mobile 9876543210 ok", "9876543210"),
        ("US (555) 123-4567 office", "(555) 123-4567"),
        ("line 555-123-4567 x89", "555-123-4567 x89"),
    ],
)
def test_phone_positives(text: str, expected: str) -> None:
    _assert_hit(text, PIIType.PHONE, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ssn 123-45-6789 filed", "123-45-6789"),
        ("ssn 123 45 6789 filed", "123 45 6789"),
    ],
)
def test_ssn_positives(text: str, expected: str) -> None:
    _assert_hit(text, PIIType.SSN, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("card 4111 1111 1111 1111", "4111 1111 1111 1111"),
        ("card 5500-0000-0000-0004", "5500-0000-0000-0004"),
        ("amex 378282246310005", "378282246310005"),
    ],
)
def test_credit_card_positives(text: str, expected: str) -> None:
    _assert_hit(text, PIIType.CREDIT_CARD, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("DOB 12/03/1988", "12/03/1988"),
        ("date of birth 1988-03-12", "1988-03-12"),
        ("born 12 March 1988", "12 March 1988"),
        ("birthday March 12, 1988", "March 12, 1988"),
    ],
)
def test_dob_positives(text: str, expected: str) -> None:
    _assert_hit(text, PIIType.DOB, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("host 192.168.1.1 up", "192.168.1.1"),
        ("net 10.0.0.255 mask", "10.0.0.255"),
        ("v6 2001:db8::1 node", "2001:db8::1"),
        ("v6 2001:db8::1. Next", "2001:db8::1"),
    ],
)
def test_ip_positives(text: str, expected: str) -> None:
    _assert_hit(text, PIIType.IP_ADDRESS, expected)


def test_address_positive() -> None:
    text = "office at 12 MG Road, Bengaluru - 560001 India"
    ents = _detector_for(PIIType.ADDRESS).detect(text, _cfg())
    assert ents, f"no address in {text!r}"
    assert ents[0].pii_type is PIIType.ADDRESS
    assert text[ents[0].start : ents[0].end] == ents[0].text
    assert "12 MG Road" in ents[0].text
    assert "560001" in ents[0].text


def test_address_does_not_span_paragraphs() -> None:
    """DOTALL-style matching used to glue IP tails onto later address lines."""
    text = (
        "Workstation IP: 192.168.1.10 and also 2001:db8::1.\n\n"
        "Home: 12 MG Road, Bengaluru - 560001.\n"
    )
    ents = _detector_for(PIIType.ADDRESS).detect(text, _cfg())
    assert len(ents) == 1
    assert ents[0].text.startswith("12 MG Road")
    assert "\n" not in ents[0].text


@pytest.mark.parametrize(
    "text,reason",
    [
        ("Ticket #1234567890", "support ticket id, not a phone"),
        ("Order 9876543210", "order number masquerading as phone"),
        ("Invoice No. 4111111111111111", "invoice reference, not a phone"),
        (
            "December 2024\n1\n234\n56 next",
            "year/day fragments across newlines are not a phone",
        ),
        (
            "headcount\n100\n200\n300\n400\ntail",
            "table cell digits must not stitch across paragraphs",
        ),
    ],
)
def test_phone_negatives(text: str, reason: str) -> None:
    _ = reason
    _assert_miss(text, PIIType.PHONE)


@pytest.mark.parametrize(
    "text,reason",
    [
        ("card 4111 1111 1111 1112", "fails Luhn"),
        ("Invoice No. 4111111111111111", "invoice reference even if Luhn-valid"),
    ],
)
def test_credit_card_negatives(text: str, reason: str) -> None:
    _ = reason
    _assert_miss(text, PIIType.CREDIT_CARD)


@pytest.mark.parametrize(
    "text,reason",
    [
        ("000-00-0000", "invalid area/group/serial all zero"),
        ("666-12-3456", "forbidden area 666"),
        ("123-00-6789", "group 00 invalid"),
    ],
)
def test_ssn_negatives(text: str, reason: str) -> None:
    _ = reason
    _assert_miss(text, PIIType.SSN)


@pytest.mark.parametrize(
    "text,reason",
    [
        ("999.1.1.1", "octet out of range"),
        ("version 1.2.3.4", "software version, not an IP"),
        ("Section 10.0.0.1", "document section numbering"),
        ("addr :: here", "unspecified IPv6 is noise, not a host"),
        ("bind 0.0.0.0 please", "unspecified IPv4 is not document PII"),
    ],
)
def test_ip_negatives(text: str, reason: str) -> None:
    _ = reason
    _assert_miss(text, PIIType.IP_ADDRESS)


@pytest.mark.parametrize(
    "text,reason",
    [
        ("DOB 31/02/1988", "impossible calendar date"),
        ("meeting on 12/03/1988", "plain date with no birth cue"),
        ("date of birth 01/01/1800", "outside plausible lifespan"),
    ],
)
def test_dob_negatives(text: str, reason: str) -> None:
    _ = reason
    _assert_miss(text, PIIType.DOB)


@pytest.mark.parametrize(
    "text,reason",
    [
        ("see @support", "not an email address"),
        ("user@localhost", "localhost is not a public mailbox"),
    ],
)
def test_email_negatives(text: str, reason: str) -> None:
    _ = reason
    _assert_miss(text, PIIType.EMAIL)


def test_match_at_start_and_end() -> None:
    text = "a@b.co and also end z@y.org"
    ents = _detector_for(PIIType.EMAIL).detect(text, _cfg())
    assert [e.text for e in ents] == ["a@b.co", "z@y.org"]
    assert ents[0].start == 0
    assert ents[-1].end == len(text)


def test_trailing_punctuation_not_swallowed() -> None:
    text = "email a@b.com."
    ents = _detector_for(PIIType.EMAIL).detect(text, _cfg())
    assert len(ents) == 1
    assert ents[0].text == "a@b.com"
    assert text[ents[0].start : ents[0].end] == "a@b.com"


def test_multiple_matches_ascending_and_duplicates() -> None:
    text = "a@b.co mid a@b.co"
    ents = _detector_for(PIIType.EMAIL).detect(text, _cfg())
    assert len(ents) == 2
    assert ents[0].start < ents[1].start
    assert ents[0].text == ents[1].text == "a@b.co"


def test_match_inside_longer_token_rejected() -> None:
    # Digits glued inside an identifier should not be a phone
    _assert_miss("id9876543210xyz", PIIType.PHONE)


def test_luhn_known_vectors() -> None:
    assert luhn_valid("4111111111111111")
    assert luhn_valid("5500000000000004")
    assert luhn_valid("378282246310005")
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("1234")


def test_ssn_structure() -> None:
    assert ssn_structure_valid("123-45-6789")
    assert not ssn_structure_valid("000-12-3456")
    assert not ssn_structure_valid("666-12-3456")
    assert not ssn_structure_valid("900-12-3456")
    assert not ssn_structure_valid("123-00-6789")


def test_calendar_and_plausible_dob() -> None:
    parsed = parse_calendar_date("1988-03-12")
    assert parsed == date(1988, 3, 12)
    assert parse_calendar_date("31/02/1988") is None
    assert plausible_dob(date(1988, 3, 12))
    assert not plausible_dob(date(1800, 1, 1))


def test_card_brand_prefix() -> None:
    assert card_brand_prefix("4111111111111111") == "visa"
    assert card_brand_prefix("5500000000000004") == "mastercard"
    assert card_brand_prefix("378282246310005") == "amex"


def test_redact_reference_numbers_flips_ticket_phone() -> None:
    text = "Ticket #1234567890"
    _assert_miss(text, PIIType.PHONE, _cfg(redact_reference_numbers=False))
    ents = _detector_for(PIIType.PHONE).detect(text, _cfg(redact_reference_numbers=True))
    assert any(e.text.endswith("1234567890") for e in ents)


def test_get_detectors_respects_enabled_types() -> None:
    only_email = _cfg(enabled_types=frozenset({PIIType.EMAIL}))
    names = {d.pii_type for d in get_detectors(only_email)}
    assert names == {PIIType.EMAIL}


def test_preceding_label_window() -> None:
    text = "Invoice No. 4111111111111111"
    start = text.index("4111")
    assert preceding_label(text, start) in {"invoice", "no"}
    assert preceding_label(text, start, window=3) is None


def test_has_birth_cue() -> None:
    assert has_birth_cue("date of birth 12/03/1988", 15)
    assert not has_birth_cue("meeting on 12/03/1988", 11)


def test_registry_covers_rule_based_types_and_protocol() -> None:
    seen: set[PIIType] = set()
    for det in iter_registry():
        assert isinstance(det, Detector)
        assert isinstance(det, RegexDetector)
        assert det.priority in {PRIORITY_REGEX, PRIORITY_VALIDATED}
        assert det.name
        seen.add(det.pii_type)
    assert seen >= RULE_BASED_TYPES
    # FULL_NAME / COMPANY are NER-only — must not claim a rule detector yet
    assert PIIType.FULL_NAME not in seen
    assert PIIType.COMPANY not in seen


def test_validated_detectors_outrank_regex() -> None:
    by_type = {d.pii_type: d for d in iter_registry()}
    assert by_type[PIIType.SSN].priority == PRIORITY_VALIDATED
    assert by_type[PIIType.CREDIT_CARD].priority == PRIORITY_VALIDATED
    assert by_type[PIIType.IP_ADDRESS].priority == PRIORITY_VALIDATED
    assert by_type[PIIType.DOB].priority == PRIORITY_VALIDATED
    assert by_type[PIIType.EMAIL].priority == PRIORITY_REGEX
    assert by_type[PIIType.PHONE].priority == PRIORITY_REGEX
    assert PRIORITY_VALIDATED > PRIORITY_REGEX
