"""test_surrogates — determinism, format preservation, consistency, coherence."""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from pii_redaction.detectors import luhn_valid
from pii_redaction.models import (
    PRIORITY_REGEX,
    PRIORITY_VALIDATED,
    PIIEntity,
    PIIType,
    RedactorConfig,
)
from pii_redaction.surrogates import (
    GENERATORS,
    SurrogateCollisionError,
    SurrogateFactory,
    normalise_key,
)


def _cfg(seed: int = 0) -> RedactorConfig:
    return RedactorConfig(use_ner=False, seed=seed, locale="en_IN")


def _entity(
    text: str,
    pii_type: PIIType,
    *,
    start: int = 0,
    priority: int = PRIORITY_REGEX,
    source: str = "test",
) -> PIIEntity:
    return PIIEntity(
        pii_type=pii_type,
        text=text,
        start=start,
        end=start + len(text),
        source=source,
        priority=priority,
    )


def test_determinism_same_seed() -> None:
    entities = [
        _entity("Rohan Dey", PIIType.FULL_NAME),
        _entity("rohan.dey@gmail.com", PIIType.EMAIL, start=20),
        _entity("+91 98765 43210", PIIType.PHONE, start=50),
    ]
    a = SurrogateFactory(_cfg(0)).assign(entities)
    b = SurrogateFactory(_cfg(0)).assign(entities)
    assert [e.replacement for e in a] == [e.replacement for e in b]


def test_different_seed_changes_values() -> None:
    entities = [_entity("Rohan Dey", PIIType.FULL_NAME)]
    a = SurrogateFactory(_cfg(0)).assign(entities)[0].replacement
    b = SurrogateFactory(_cfg(1)).assign(entities)[0].replacement
    assert a != b


def test_phone_preserves_country_and_grouping() -> None:
    original = "+91 98765 43210"
    factory = SurrogateFactory(_cfg(7))
    fake = factory.assign([_entity(original, PIIType.PHONE)])[0].replacement
    assert fake is not None
    assert fake.startswith("+91 ")
    assert re.fullmatch(r"\+91 \d{5} \d{5}", fake)
    assert fake != original


def test_credit_card_preserves_brand_length_grouping_and_luhn() -> None:
    original = "4111 1111 1111 1111"
    factory = SurrogateFactory(_cfg(3))
    fake = factory.assign(
        [_entity(original, PIIType.CREDIT_CARD, priority=PRIORITY_VALIDATED)]
    )[0].replacement
    assert fake is not None
    assert re.fullmatch(r"\d{4} \d{4} \d{4} \d{4}", fake)
    assert fake[0] == "4"
    assert luhn_valid(fake)
    assert fake != original


def test_dob_preserves_format() -> None:
    cases = [
        ("1988-03-12", r"\d{4}-\d{2}-\d{2}"),
        ("12/03/1988", r"\d{2}/\d{2}/\d{4}"),
        ("12 March 1988", r"\d{1,2} [A-Za-z]+ \d{4}"),
    ]
    factory = SurrogateFactory(_cfg(5))
    for original, pattern in cases:
        fake = factory.assign(
            [_entity(original, PIIType.DOB, priority=PRIORITY_VALIDATED)]
        )[0].replacement
        assert fake is not None
        assert re.fullmatch(pattern, fake), (original, fake)
        assert fake != original


def test_ip_version_and_private_class() -> None:
    factory = SurrogateFactory(_cfg(9))
    v4 = factory.assign([_entity("10.0.0.5", PIIType.IP_ADDRESS)])[0].replacement
    assert v4 is not None and ":" not in v4
    first = int(v4.split(".")[0])
    assert first in {10, 172, 192}
    v6 = factory.assign([_entity("2001:db8::1", PIIType.IP_ADDRESS, start=20)])[0].replacement
    assert v6 is not None and ":" in v6


def test_address_same_line_count() -> None:
    original = "12 MG Road\nBengaluru 560001"
    factory = SurrogateFactory(_cfg(2))
    fake = factory.assign([_entity(original, PIIType.ADDRESS)])[0].replacement
    assert fake is not None
    assert fake.count("\n") == original.count("\n")


def test_company_keeps_legal_suffix() -> None:
    original = "Acme Technologies Pvt Ltd"
    factory = SurrogateFactory(_cfg(4))
    fake = factory.assign([_entity(original, PIIType.COMPANY)])[0].replacement
    assert fake is not None
    assert fake.casefold().endswith("pvt ltd")
    assert fake != original


def test_full_name_keeps_honorific() -> None:
    original = "Mr. Rohan Dey"
    factory = SurrogateFactory(_cfg(6))
    fake = factory.assign([_entity(original, PIIType.FULL_NAME)])[0].replacement
    assert fake is not None
    assert fake.startswith("Mr.")
    assert fake != original


def test_consistency_same_value_same_fake() -> None:
    name = "Rohan Dey"
    entities = [
        _entity(name, PIIType.FULL_NAME, start=0),
        _entity(name, PIIType.FULL_NAME, start=40),
        _entity(name, PIIType.FULL_NAME, start=80),
    ]
    out = SurrogateFactory(_cfg(0)).assign(entities)
    fakes = {e.replacement for e in out}
    assert len(fakes) == 1


def test_phone_formats_share_map_entry() -> None:
    a = _entity("+91-9876543210", PIIType.PHONE, start=0)
    b = _entity("+91 98765 43210", PIIType.PHONE, start=30)
    assert normalise_key(PIIType.PHONE, a.text) == normalise_key(PIIType.PHONE, b.text)
    out = SurrogateFactory(_cfg(0)).assign([a, b])
    assert out[0].replacement == out[1].replacement


def test_same_digits_different_types_do_not_share() -> None:
    # Same digit string as two types must not share a map entry
    phone = _entity("123-45-6789", PIIType.PHONE, start=0)
    ssn = _entity("123-45-6789", PIIType.SSN, start=20, priority=PRIORITY_VALIDATED)
    out = SurrogateFactory(_cfg(0)).assign([phone, ssn])
    assert out[0].replacement != out[1].replacement


def test_distinct_reals_never_share_fake() -> None:
    entities = [
        _entity("Alice Example", PIIType.FULL_NAME, start=0),
        _entity("Bob Example", PIIType.FULL_NAME, start=20),
        _entity("Carol Example", PIIType.FULL_NAME, start=40),
    ]
    out = SurrogateFactory(_cfg(0)).assign(entities)
    fakes = [e.replacement for e in out]
    assert len(fakes) == len(set(fakes))


def test_name_email_coherence() -> None:
    entities = [
        _entity("Rohan Dey", PIIType.FULL_NAME, start=0),
        _entity("rohan.dey@gmail.com", PIIType.EMAIL, start=20),
    ]
    out = SurrogateFactory(_cfg(0)).assign(entities)
    fake_name = out[0].replacement
    fake_email = out[1].replacement
    assert fake_name and fake_email
    local = fake_email.split("@", 1)[0]
    tokens = [t for t in re.split(r"[^\w]+", fake_name.casefold()) if t]
    assert len(tokens) >= 2
    assert local == f"{tokens[0]}.{tokens[-1]}"
    # The domain is replaced too: keeping "@gmail.com" is harmless, but keeping
    # "@kshinternational.com" beside a redacted company name is not, and the
    # rule cannot depend on recognising which domains are identifying.
    assert not fake_email.endswith("@gmail.com")
    assert fake_email.endswith(".com")


def test_independent_email_when_not_linked_to_name() -> None:
    entities = [
        _entity("Rohan Dey", PIIType.FULL_NAME, start=0),
        _entity("support@example.com", PIIType.EMAIL, start=20),
    ]
    out = SurrogateFactory(_cfg(0)).assign(entities)
    assert out[1].replacement is not None
    assert not out[1].replacement.endswith("@example.com")
    assert out[1].replacement.endswith(".com")
    assert "rohan" not in out[1].replacement.casefold()


def test_email_and_website_domains_map_consistently() -> None:
    """The reversibility fix: one real domain gets exactly one fake domain."""
    entities = [
        _entity("KSH International Limited", PIIType.COMPANY, start=0),
        _entity("ipo@kshinternational.com", PIIType.EMAIL, start=40),
        _entity("www.kshinternational.com", PIIType.DOMAIN, start=80),
        _entity("cs@kshinternational.com", PIIType.EMAIL, start=120),
    ]
    out = SurrogateFactory(_cfg(0)).assign(entities)
    email_domain = out[1].replacement.split("@", 1)[1]
    website = out[2].replacement
    second_domain = out[3].replacement.split("@", 1)[1]

    assert email_domain == second_domain
    assert website == f"www.{email_domain}"
    assert "kshinternational" not in " ".join(e.replacement for e in out).casefold()


def test_domain_surrogate_preserves_shape() -> None:
    factory = SurrogateFactory(_cfg(1))
    cases = [
        ("www.example.co.in", r"www\.[a-z0-9]+\.co\.in"),
        ("https://www.example.com", r"https://www\.[a-z0-9]+\.com"),
        ("example.org", r"[a-z0-9]+\.org"),
    ]
    for original, pattern in cases:
        fake = factory.assign([_entity(original, PIIType.DOMAIN)])[0].replacement
        assert fake is not None
        assert re.fullmatch(pattern, fake), (original, fake)
        assert fake != original


def test_no_fake_equals_own_original() -> None:
    samples = [
        _entity("rashhi.patil@gmail.com", PIIType.EMAIL),
        _entity("+91 9876543210", PIIType.PHONE, start=40),
        _entity("4111 1111 1111 1111", PIIType.CREDIT_CARD, start=80),
        _entity("Ada Lovelace", PIIType.FULL_NAME, start=120),
    ]
    out = SurrogateFactory(_cfg(11)).assign(samples)
    for entity in out:
        assert entity.replacement != entity.text


def test_fake_never_equals_any_real_in_document() -> None:
    entities = [
        _entity("Alice One", PIIType.FULL_NAME, start=0),
        _entity("Bob Two", PIIType.FULL_NAME, start=20),
        _entity("Carol Three", PIIType.FULL_NAME, start=40),
    ]
    reals = {e.text for e in entities}
    out = SurrogateFactory(_cfg(0)).assign(entities)
    for entity in out:
        assert entity.replacement not in reals


def test_collision_exhaustion_raises() -> None:
    """A generator that can only emit the original must fail, never leak it."""
    factory = SurrogateFactory(_cfg(0))
    entity = _entity("+91 98765 43210", PIIType.PHONE)

    def always_same(_original: str, _faker: object) -> str:
        return "+91 98765 43210"

    with (
        patch.dict(GENERATORS, {PIIType.PHONE: always_same}),
        pytest.raises(SurrogateCollisionError),
    ):
        factory.assign([entity])


def test_name_like_collision_is_repaired_not_raised() -> None:
    """Name-like types repair a colliding token instead of exhausting retries.

    Faker's en_IN company pool shares a surname list with its person names, so
    on a document with hundreds of real names a reject-and-retry loop fails
    outright. Repair keeps generation total.
    """
    factory = SurrogateFactory(_cfg(0))
    entities = [
        _entity("Hegde Rastogi", PIIType.FULL_NAME, start=0),
        _entity("Hegde Traders Limited", PIIType.COMPANY, start=40),
    ]
    out = factory.assign(entities)
    joined = " ".join(e.replacement or "" for e in out).casefold()
    assert all(e.replacement for e in out)
    assert "hegde" not in joined
    assert "rastogi" not in joined


@pytest.mark.parametrize("pii_type", list(PIIType))
def test_every_type_has_generator(pii_type: PIIType) -> None:
    assert pii_type in GENERATORS


def test_surrogate_never_embeds_a_real_name_token() -> None:
    """Faker's company pool can emit a real surname; containment must be rejected."""
    entities = [
        _entity("Kushal Hegde", PIIType.FULL_NAME, start=0),
        _entity("Acme Industries Limited", PIIType.COMPANY, start=40),
        _entity("Beta Traders Limited", PIIType.COMPANY, start=80),
        _entity("Gamma Exports Limited", PIIType.COMPANY, start=120),
    ]
    out = SurrogateFactory(_cfg(0)).assign(entities)
    joined = " ".join(e.replacement or "" for e in out).casefold()
    assert "hegde" not in joined
    assert "kushal" not in joined
