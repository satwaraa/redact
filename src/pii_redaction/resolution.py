"""Conflict policy for overlapping PII spans. Pure list-in / list-out."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pii_redaction.models import PIIEntity


def resolve(entities: Iterable[PIIEntity]) -> list[PIIEntity]:
    """Collapse overlapping detections per D4.

    Policy:
      1. Longest span wins.
      2. Equal length → higher ``priority`` wins (validated > regex > NER).
      3. Still tied → earlier ``start``, then ``source`` (deterministic).
      4. Contained / overlapping losers are dropped.

    Output is sorted by start, pairwise non-overlapping, and a subset of the
    input identities (never edited or fabricated).
    """
    ranked = sorted(
        entities,
        key=lambda e: (-len(e), -e.priority, e.start, e.source),
    )
    accepted: list[PIIEntity] = []
    for candidate in ranked:
        if any(candidate.overlaps(prev) for prev in accepted):
            continue
        accepted.append(candidate)
    accepted.sort(key=lambda e: (e.start, e.end, e.source))
    return accepted


def assert_non_overlapping(entities: Sequence[PIIEntity]) -> None:
    ordered = sorted(entities, key=lambda e: (e.start, e.end))
    for left, right in zip(ordered, ordered[1:], strict=False):
        if left.overlaps(right):
            raise ValueError(
                f"overlapping entities at {left.start}:{left.end} and "
                f"{right.start}:{right.end}"
            )
