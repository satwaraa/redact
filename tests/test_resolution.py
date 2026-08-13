"""test_resolution — conflict policy: longest span, priority ties, invariants."""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pii_redaction.models import (
    PRIORITY_NER,
    PRIORITY_REGEX,
    PRIORITY_VALIDATED,
    PIIEntity,
    PIIType,
)
from pii_redaction.resolution import assert_non_overlapping, resolve


def _entity(
    start: int,
    end: int,
    *,
    pii_type: PIIType = PIIType.EMAIL,
    priority: int = PRIORITY_REGEX,
    source: str = "test",
    text: str | None = None,
) -> PIIEntity:
    span = text if text is not None else ("x" * (end - start))
    assert len(span) == end - start
    return PIIEntity(
        pii_type=pii_type,
        text=span,
        start=start,
        end=end,
        source=source,
        priority=priority,
    )


def _assert_invariants(original: Sequence[PIIEntity], resolved: Sequence[PIIEntity]) -> None:
    # ascending by start
    starts = [e.start for e in resolved]
    assert starts == sorted(starts)
    # pairwise non-overlapping
    assert_non_overlapping(resolved)
    # every output entity is an input entity (same object identity)
    original_ids = {id(e) for e in original}
    for entity in resolved:
        assert id(entity) in original_ids
    # idempotent
    assert resolve(resolved) == list(resolved)
    # order-independent
    shuffled = list(original)
    random.Random(0).shuffle(shuffled)
    assert resolve(shuffled) == list(resolved)


# ---------------------------------------------------------------------------
# 1. Interval relations
# ---------------------------------------------------------------------------


def test_disjoint_both_kept() -> None:
    a = _entity(0, 3, source="a")
    b = _entity(5, 8, source="b")
    out = resolve([a, b])
    assert out == [a, b]
    _assert_invariants([a, b], out)


def test_touching_both_kept() -> None:
    a = _entity(0, 3, source="a")
    b = _entity(3, 6, source="b")
    assert not a.overlaps(b)
    out = resolve([a, b])
    assert out == [a, b]
    _assert_invariants([a, b], out)


def test_partial_overlap_longest_wins() -> None:
    short = _entity(0, 4, source="short")  # len 4
    long = _entity(2, 9, source="long")  # len 7 — later start, but longer
    out = resolve([short, long])
    assert out == [long]
    _assert_invariants([short, long], out)


def test_containment_drops_inner() -> None:
    outer = _entity(0, 10, source="outer")
    inner = _entity(2, 5, source="inner")
    out = resolve([outer, inner])
    assert out == [outer]
    _assert_invariants([outer, inner], out)


def test_identical_spans_higher_priority_wins() -> None:
    low = _entity(
        0,
        9,
        text="123456789",
        pii_type=PIIType.PHONE,
        priority=PRIORITY_REGEX,
        source="regex:phone",
    )
    high = _entity(
        0,
        9,
        text="123456789",
        pii_type=PIIType.SSN,
        priority=PRIORITY_VALIDATED,
        source="regex:ssn",
    )
    out = resolve([low, high])
    assert out == [high]
    _assert_invariants([low, high], out)


def test_identical_spans_identical_priority_source_tiebreak() -> None:
    a = _entity(0, 5, text="aaaaa", priority=PRIORITY_REGEX, source="alpha")
    b = _entity(0, 5, text="aaaaa", priority=PRIORITY_REGEX, source="beta")
    # Same length+priority+start → lexicographically smaller source wins
    out = resolve([b, a])
    assert out == [a]
    _assert_invariants([a, b], out)


# ---------------------------------------------------------------------------
# 2. Motivating real collisions
# ---------------------------------------------------------------------------


def test_luhn_card_beats_phone_on_same_digits() -> None:
    digits = "4111111111111111"
    phone = _entity(
        0,
        16,
        text=digits,
        pii_type=PIIType.PHONE,
        priority=PRIORITY_REGEX,
        source="regex:phone",
    )
    card = _entity(
        0,
        16,
        text=digits,
        pii_type=PIIType.CREDIT_CARD,
        priority=PRIORITY_VALIDATED,
        source="regex:credit_card",
    )
    out = resolve([phone, card])
    assert out == [card]
    assert out[0].pii_type is PIIType.CREDIT_CARD
    _assert_invariants([phone, card], out)


def test_ssn_beats_phone_on_same_span() -> None:
    text = "123-45-6789"
    phone = _entity(
        0,
        11,
        text=text,
        pii_type=PIIType.PHONE,
        priority=PRIORITY_REGEX,
        source="regex:phone",
    )
    ssn = _entity(
        0,
        11,
        text=text,
        pii_type=PIIType.SSN,
        priority=PRIORITY_VALIDATED,
        source="regex:ssn",
    )
    out = resolve([phone, ssn])
    assert out == [ssn]
    _assert_invariants([phone, ssn], out)


def test_email_beats_overlapping_ner_person_on_local_part() -> None:
    # "Ada Lovelace" NER span overlapping "Ada" in "Ada@example.com"
    email = _entity(
        0,
        15,
        text="Ada@example.com",
        pii_type=PIIType.EMAIL,
        priority=PRIORITY_REGEX,
        source="regex:email",
    )
    person = _entity(
        0,
        12,
        text="Ada Lovelace",
        pii_type=PIIType.FULL_NAME,
        priority=PRIORITY_NER,
        source="ner:PERSON",
    )
    out = resolve([person, email])
    assert out == [email]
    _assert_invariants([person, email], out)


def test_address_beats_shorter_ner_gpe_overlap() -> None:
    address_text = "12 MG Road Bengaluru 560001xx"
    address = _entity(
        0,
        len(address_text),
        text=address_text,
        pii_type=PIIType.ADDRESS,
        priority=PRIORITY_REGEX,
        source="regex:address",
    )
    gpe = _entity(
        10,
        19,
        text="Bengaluru",
        pii_type=PIIType.ADDRESS,
        priority=PRIORITY_NER,
        source="ner:GPE",
    )
    out = resolve([gpe, address])
    assert out == [address]
    _assert_invariants([gpe, address], out)


def test_empty_and_single() -> None:
    assert resolve([]) == []
    alone = _entity(1, 4, source="only")
    assert resolve([alone]) == [alone]


def test_n_identical_entities_one_survives() -> None:
    clones = [
        _entity(0, 4, text="same", priority=PRIORITY_REGEX, source=f"s{i}")
        for i in range(5)
    ]
    out = resolve(clones)
    assert len(out) == 1
    assert out[0].source == "s0"  # earliest source name wins the tiebreak
    _assert_invariants(clones, out)


def test_mixed_corpus_invariants() -> None:
    entities = [
        _entity(0, 3, source="a"),
        _entity(3, 6, source="b"),  # touching
        _entity(5, 12, source="c"),  # overlaps b, longer
        _entity(20, 25, source="d"),
        _entity(21, 23, source="e"),  # contained in d
        _entity(
            30,
            40,
            text="y" * 10,
            priority=PRIORITY_REGEX,
            source="regex",
        ),
        _entity(
            30,
            40,
            text="y" * 10,
            priority=PRIORITY_VALIDATED,
            source="validated",
        ),
    ]
    out = resolve(entities)
    _assert_invariants(entities, out)
    assert {e.source for e in out} == {"a", "c", "d", "validated"}



def test_assert_non_overlapping_accepts_resolved() -> None:
    entities = [_entity(0, 3), _entity(5, 8)]
    resolved = resolve(entities)
    assert_non_overlapping(resolved)  # does not raise


def test_assert_non_overlapping_raises_without_values() -> None:
    secret = "leak@mail.com"
    left = _entity(0, len(secret), text=secret, source="left")
    right = _entity(5, 5 + len(secret), text=secret, source="right")
    with pytest.raises(ValueError, match="overlapping entities") as exc_info:
        assert_non_overlapping([left, right])
    message = str(exc_info.value)
    assert "0:" in message
    assert secret not in message


@st.composite
def entity_sets(draw: st.DrawFn) -> list[PIIEntity]:
    n = draw(st.integers(min_value=0, max_value=12))
    entities: list[PIIEntity] = []
    for i in range(n):
        start = draw(st.integers(min_value=0, max_value=40))
        length = draw(st.integers(min_value=1, max_value=12))
        end = start + length
        priority = draw(
            st.sampled_from([PRIORITY_NER, PRIORITY_REGEX, PRIORITY_VALIDATED])
        )
        entities.append(
            _entity(
                start,
                end,
                priority=priority,
                source=f"gen-{i}-{priority}",
            )
        )
    return entities


@given(entity_sets())
@settings(max_examples=80, deadline=None)
def test_resolve_invariants_property(entities: list[PIIEntity]) -> None:
    resolved = resolve(entities)
    _assert_invariants(entities, resolved)
