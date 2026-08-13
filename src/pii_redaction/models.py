"""Domain types, config, and exceptions. Pure data with enforced invariants."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any


class PIIType(StrEnum):
    FULL_NAME = "FULL_NAME"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    COMPANY = "COMPANY"
    ADDRESS = "ADDRESS"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    DOB = "DOB"
    IP_ADDRESS = "IP_ADDRESS"


PRIORITY_VALIDATED = 30
PRIORITY_REGEX = 20
PRIORITY_NER = 10


@dataclass(frozen=True, slots=True)
class PIIEntity:
    pii_type: PIIType
    text: str
    start: int
    end: int
    source: str
    confidence: float = 1.0
    priority: int = 0
    replacement: str | None = None

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("PIIEntity.text must be non-empty")
        if self.start < 0:
            raise ValueError(f"PIIEntity.start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"PIIEntity.end must be > start, got start={self.start} end={self.end}"
            )
        if self.end - self.start != len(self.text):
            raise ValueError(
                f"PIIEntity span length mismatch: end-start={self.end - self.start} "
                f"!= len(text)={len(self.text)} (offsets {self.start}:{self.end}, "
                f"source={self.source!r})"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"PIIEntity.confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    def overlaps(self, other: PIIEntity) -> bool:
        return self.start < other.end and other.start < self.end

    def contains(self, other: PIIEntity) -> bool:
        return self.start <= other.start and self.end >= other.end and self != other

    def with_replacement(self, value: str) -> PIIEntity:
        return replace(self, replacement=value)

    def __len__(self) -> int:
        return self.end - self.start


def assert_consistent(text: str, entities: Iterable[PIIEntity]) -> None:
    for entity in entities:
        slice_ = text[entity.start : entity.end]
        if slice_ != entity.text:
            raise ValueError(
                f"offset inconsistency at {entity.start}:{entity.end} "
                f"from {entity.source!r}: extracted length {len(slice_)} "
                f"!= entity length {len(entity.text)}"
            )


@dataclass(frozen=True, slots=True)
class RedactionResult:
    entities: tuple[PIIEntity, ...]
    mapping: Mapping[str, str]
    counts_by_type: Mapping[PIIType, int]
    text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [
                {
                    "pii_type": e.pii_type.value,
                    "text": e.text,
                    "start": e.start,
                    "end": e.end,
                    "source": e.source,
                    "confidence": e.confidence,
                    "priority": e.priority,
                    "replacement": e.replacement,
                }
                for e in self.entities
            ],
            "mapping": dict(self.mapping),
            "counts_by_type": {k.value: v for k, v in self.counts_by_type.items()},
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class RedactorConfig:
    enabled_types: frozenset[PIIType] = field(default_factory=lambda: frozenset(PIIType))
    seed: int = 0
    use_ner: bool = True
    ner_model: str = "en_core_web_sm"
    ner_confidence_threshold: float = 0.5
    ner_max_doc_freq: int = 15
    # B8: require model ∩ structural heuristic for FULL_NAME / COMPANY.
    ner_agreement: bool = False
    redact_reference_numbers: bool = False
    locale: str = "en_IN"
    verify_output: bool = True

    @classmethod
    def default(cls) -> RedactorConfig:
        return cls()


class RedactionError(Exception):
    """Base error for the redaction pipeline."""


class DocumentError(RedactionError):
    """Unreadable/unwritable docx, or input/output path collision."""


class ModelUnavailableError(RedactionError):
    """NER backend or model not installed."""


class LeakDetectedError(RedactionError):
    """Original PII survived redaction (D7)."""

    def __init__(self, pii_type: PIIType, count: int) -> None:
        self.pii_type = pii_type
        self.count = count
        super().__init__(
            f"leak detected: {count} occurrence(s) of {pii_type.value} survived redaction"
        )
