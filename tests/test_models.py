"""test_models — contract invariants for domain types."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from pii_redaction.models import (
    PRIORITY_NER,
    PRIORITY_REGEX,
    PRIORITY_VALIDATED,
    DocumentError,
    LeakDetectedError,
    ModelUnavailableError,
    PIIEntity,
    PIIType,
    RedactionError,
    RedactionResult,
    RedactorConfig,
    assert_consistent,
)


def _entity(**overrides: object) -> PIIEntity:
    base: dict[str, object] = {
        "pii_type": PIIType.EMAIL,
        "text": "a@b.co",
        "start": 0,
        "end": 6,
        "source": "test",
        "confidence": 1.0,
        "priority": PRIORITY_REGEX,
    }
    base.update(overrides)
    return PIIEntity(**base)  # type: ignore[arg-type]


class TestPIIEntityInvariants:
    def test_valid_entity_constructs(self) -> None:
        e = _entity()
        assert e.text == "a@b.co"
        assert e.end - e.start == len(e.text)

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _entity(text="", start=0, end=0)

    def test_negative_start_rejected(self) -> None:
        with pytest.raises(ValueError, match="start"):
            _entity(start=-1, end=5, text="hello")

    def test_end_not_after_start_rejected(self) -> None:
        with pytest.raises(ValueError, match="end"):
            _entity(start=3, end=3, text="x")
        with pytest.raises(ValueError, match="end"):
            _entity(start=5, end=2, text="ab")

    def test_span_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="span length mismatch"):
            _entity(text="abcd", start=0, end=3)

    @pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0])
    def test_confidence_out_of_range_rejected(self, confidence: float) -> None:
        with pytest.raises(ValueError, match="confidence"):
            _entity(confidence=confidence)


class TestImmutability:
    def test_frozen_assignment_raises(self) -> None:
        e = _entity()
        with pytest.raises(FrozenInstanceError):
            e.text = "x"  # type: ignore[misc]

    def test_with_replacement_returns_new_entity(self) -> None:
        original = _entity()
        updated = original.with_replacement("z@z.co")
        assert updated is not original
        assert original.replacement is None
        assert updated.replacement == "z@z.co"
        assert updated.pii_type == original.pii_type
        assert updated.text == original.text
        assert updated.start == original.start
        assert updated.end == original.end
        assert updated.source == original.source
        assert updated.confidence == original.confidence
        assert updated.priority == original.priority

    def test_hashable_and_equal_by_value(self) -> None:
        a = _entity()
        b = _entity()
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1


@pytest.mark.parametrize(
    ("a", "b", "overlaps", "a_contains_b", "b_contains_a"),
    [
        # disjoint
        ((0, 3, "abc"), (5, 8, "def"), False, False, False),
        # touching (end == other.start) — not overlapping; end is exclusive
        ((0, 3, "abc"), (3, 6, "def"), False, False, False),
        # partial overlap a then b
        ((0, 4, "abcd"), (2, 6, "cdef"), True, False, False),
        # partial overlap b then a
        ((2, 6, "cdef"), (0, 4, "abcd"), True, False, False),
        # a contains b
        ((0, 6, "abcdef"), (1, 3, "bc"), True, True, False),
        # b contains a
        ((1, 3, "bc"), (0, 6, "abcdef"), True, False, True),
        # identical — contains() is False by definition (self != other)
        ((0, 3, "abc"), (0, 3, "abc"), True, False, False),
    ],
)
def test_span_algebra(
    a: tuple[int, int, str],
    b: tuple[int, int, str],
    overlaps: bool,
    a_contains_b: bool,
    b_contains_a: bool,
) -> None:
    left = _entity(start=a[0], end=a[1], text=a[2])
    right = _entity(start=b[0], end=b[1], text=b[2])
    assert left.overlaps(right) is overlaps
    assert right.overlaps(left) is overlaps
    assert left.contains(right) is a_contains_b
    assert right.contains(left) is b_contains_a


class TestAssertConsistent:
    def test_passes_when_offsets_match(self) -> None:
        text = "hello a@b.co world"
        entity = _entity(text="a@b.co", start=6, end=12)
        assert_consistent(text, [entity])

    def test_raises_when_offsets_point_elsewhere(self) -> None:
        text = "hello a@b.co world"
        # offsets land on "hello " rather than the email
        entity = _entity(text="a@b.co", start=0, end=6)
        with pytest.raises(ValueError, match="offset inconsistency") as exc_info:
            assert_consistent(text, [entity])
        message = str(exc_info.value)
        assert "0:6" in message
        assert "test" in message
        # D8: never surface the PII value in the error
        assert entity.text not in message


class TestPIIType:
    def test_is_str_at_runtime(self) -> None:
        assert isinstance(PIIType.EMAIL, str)
        assert PIIType.EMAIL == "EMAIL"

    def test_json_round_trip(self) -> None:
        payload = {"type": PIIType.FULL_NAME}
        encoded = json.dumps(payload)
        decoded = json.loads(encoded)
        assert decoded["type"] == "FULL_NAME"
        assert PIIType(decoded["type"]) is PIIType.FULL_NAME

    def test_all_assignment_types_present(self) -> None:
        required = {
            "FULL_NAME",
            "EMAIL",
            "PHONE",
            "COMPANY",
            "ADDRESS",
            "SSN",
            "CREDIT_CARD",
            "DOB",
            "IP_ADDRESS",
        }
        actual = {t.value for t in PIIType}
        assert required <= actual
        # Extensions beyond the assignment minimum are deliberate and listed.
        # DOMAIN: a company name is in scope, and leaving its website domain
        # makes that redaction reversible.
        assert actual - required == {"DOMAIN"}


class TestRedactorConfigAndResult:
    def test_default_enables_every_type(self) -> None:
        config = RedactorConfig.default()
        assert config.enabled_types == frozenset(PIIType)

    def test_config_is_frozen(self) -> None:
        config = RedactorConfig.default()
        with pytest.raises(FrozenInstanceError):
            config.seed = 99  # type: ignore[misc]

    def test_redaction_result_to_dict_is_json_serialisable(self) -> None:
        entity = _entity().with_replacement("x@y.z")
        result = RedactionResult(
            entities=(entity,),
            mapping={"a@b.co": "x@y.z"},
            counts_by_type={PIIType.EMAIL: 1},
            text="redacted",
        )
        payload = result.to_dict()
        # Must not raise
        encoded = json.dumps(payload)
        round_tripped = json.loads(encoded)
        assert round_tripped["counts_by_type"]["EMAIL"] == 1
        assert round_tripped["entities"][0]["pii_type"] == "EMAIL"
        assert round_tripped["mapping"]["a@b.co"] == "x@y.z"


class TestExceptionHierarchy:
    @pytest.mark.parametrize(
        "exc",
        [
            DocumentError("bad doc"),
            ModelUnavailableError("no model"),
            LeakDetectedError(PIIType.SSN, 2),
        ],
    )
    def test_specific_errors_are_redaction_errors(self, exc: RedactionError) -> None:
        assert isinstance(exc, RedactionError)

    def test_leak_message_has_type_and_count_not_values(self) -> None:
        err = LeakDetectedError(PIIType.EMAIL, 3)
        assert "EMAIL" in str(err)
        assert "3" in str(err)
        assert err.pii_type is PIIType.EMAIL
        assert err.count == 3


class TestPriorityConstants:
    def test_ordering(self) -> None:
        assert PRIORITY_VALIDATED > PRIORITY_REGEX > PRIORITY_NER
