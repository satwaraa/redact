"""Orchestration: detect → resolve → assign → apply → verify → save."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from pii_redaction.detectors import get_detectors
from pii_redaction.document import DocxDocument, package_corpus
from pii_redaction.models import (
    DocumentError,
    LeakDetectedError,
    PIIEntity,
    PIIType,
    RedactionResult,
    RedactorConfig,
    assert_consistent,
)
from pii_redaction.resolution import resolve
from pii_redaction.surrogates import SurrogateFactory

logger = logging.getLogger(__name__)

# Whole-string leak search only above this length (D7 false-positive guard).
_LEAK_MIN_LEN = 5


def apply_text(text: str, entities: Sequence[PIIEntity]) -> str:
    """Splice replacements into ``text`` right-to-left (non-overlapping spans)."""
    ordered = sorted(entities, key=lambda e: e.start, reverse=True)
    result = text
    for entity in ordered:
        if entity.replacement is None:
            raise DocumentError(
                f"entity at {entity.start}:{entity.end} has no replacement"
            )
        result = result[: entity.start] + entity.replacement + result[entity.end :]
    return result


def expand_occurrences(
    text: str,
    entities: Sequence[PIIEntity],
    *,
    min_len: int = _LEAK_MIN_LEN,
    block_end_for_offset: Callable[[int], int] | None = None,
) -> list[PIIEntity]:
    """Cover every in-scope occurrence of an already-detected value.

    Detectors (especially NER) can miss later repeats of the same string. D7
    treats any surviving original as a hard failure, so once a value is known
    PII, each non-overlapping occurrence must be scheduled for replacement.

    Call this *after* dropping cross-block spans: a longer multi-block hit must
    not claim a position and then be discarded, leaving a shorter in-block
    repeat unreplaced.
    """
    if not entities:
        return []

    existing = list(entities)
    prototypes: dict[str, PIIEntity] = {}
    for entity in existing:
        if len(entity.text) >= min_len:
            prototypes.setdefault(entity.text, entity)

    claimed = [(entity.start, entity.end) for entity in existing]
    extras: list[PIIEntity] = []
    for original, proto in sorted(
        prototypes.items(), key=lambda item: (-len(item[0]), item[0])
    ):
        search_from = 0
        while True:
            start = text.find(original, search_from)
            if start < 0:
                break
            end = start + len(original)
            search_from = start + 1
            if block_end_for_offset is not None:
                try:
                    block_end = block_end_for_offset(start)
                except DocumentError:
                    continue
                if end > block_end:
                    continue
            if any(cs < end and ce > start for cs, ce in claimed):
                continue
            claimed.append((start, end))
            if any(e.start == start and e.end == end for e in existing):
                continue
            extras.append(
                PIIEntity(
                    pii_type=proto.pii_type,
                    text=original,
                    start=start,
                    end=end,
                    source=proto.source,
                    confidence=proto.confidence,
                    priority=proto.priority,
                )
            )

    if extras:
        logger.info("expanded same-text occurrences count=%d", len(extras))
    merged = existing + extras
    merged.sort(key=lambda e: (e.start, e.end, e.source))
    return merged


def _mask_replacements(text: str, entities: Sequence[PIIEntity]) -> str:
    """Blank out known surrogates so whole-string leak search ignores them.

    Short originals can reappear as substrings of unrelated Faker output; that
    is not an unreplaced source span (offset checks cover apply failures).
    """
    masked = text
    replacements = sorted(
        {entity.replacement for entity in entities if entity.replacement},
        key=len,
        reverse=True,
    )
    for replacement in replacements:
        masked = masked.replace(replacement, "\0" * len(replacement))
    return masked


def verify_no_leaks(
    entities: Sequence[PIIEntity],
    rendered: str,
    *,
    original_text: str | None = None,
) -> None:
    """Raise ``LeakDetectedError`` if an original value survived redaction (D7).

    When ``original_text`` is provided, first confirm each assigned span would
    change under ``apply_text`` (offset-level). Then search ``rendered`` for
    surviving originals, but only for values at least ``_LEAK_MIN_LEN`` long so
    short tokens that appear inside unrelated words do not false-positive.
    Known surrogate strings are masked before the whole-string scan.
    """
    if original_text is not None:
        for entity in entities:
            if entity.replacement is None:
                continue
            if original_text[entity.start : entity.end] != entity.text:
                raise LeakDetectedError(entity.pii_type, 1)
            if entity.replacement == entity.text:
                raise LeakDetectedError(entity.pii_type, 1)
        expected = _mask_replacements(apply_text(original_text, entities), entities)
        for entity in entities:
            if entity.replacement is None or len(entity.text) < _LEAK_MIN_LEN:
                continue
            if entity.text in expected:
                raise LeakDetectedError(entity.pii_type, 1)

    searchable = _mask_replacements(rendered, entities)
    leaks: Counter[PIIType] = Counter()
    for entity in entities:
        if entity.replacement is None:
            continue
        if len(entity.text) < _LEAK_MIN_LEN:
            continue
        if entity.text in searchable:
            leaks[entity.pii_type] += 1
            logger.warning(
                "leak candidate type=%s offsets=%d:%d length=%d",
                entity.pii_type.value,
                entity.start,
                entity.end,
                len(entity.text),
            )
    if leaks:
        pii_type, count = leaks.most_common(1)[0]
        raise LeakDetectedError(pii_type, count)


def verify_package_no_leaks(
    entities: Sequence[PIIEntity],
    package_path: Path | str,
) -> None:
    """C1: search every serialized package part for surviving original values."""
    searchable = _mask_replacements(package_corpus(package_path), entities)
    leaks: Counter[PIIType] = Counter()
    for entity in entities:
        if entity.replacement is None or len(entity.text) < _LEAK_MIN_LEN:
            continue
        if entity.text in searchable:
            leaks[entity.pii_type] += 1
            logger.warning(
                "package leak type=%s offsets=%d:%d length=%d",
                entity.pii_type.value,
                entity.start,
                entity.end,
                len(entity.text),
            )
    if leaks:
        pii_type, count = leaks.most_common(1)[0]
        raise LeakDetectedError(pii_type, count)


def collect_rule_values(text: str, config: RedactorConfig) -> dict[str, PIIType]:
    """PII-shaped strings found by rule detectors only (no NER)."""
    rule_cfg = replace(config, use_ner=False, verify_output=False)
    values: dict[str, PIIType] = {}
    for detector in get_detectors(rule_cfg):
        for entity in detector.detect(text, rule_cfg):
            if len(entity.text) >= _LEAK_MIN_LEN:
                values.setdefault(entity.text, entity.pii_type)
    return values


def verify_rule_recall(
    input_text: str,
    output_text: str,
    config: RedactorConfig,
) -> None:
    """C2: rule-shaped values present in both input and output are unreplaced originals.

    Independent of the detection list used for redaction: anything the rules find
    in the source and still find in the result is a recall failure.
    """
    before = collect_rule_values(input_text, config)
    after = collect_rule_values(output_text, config)
    leaks: Counter[PIIType] = Counter()
    for value, pii_type in before.items():
        if value in after:
            leaks[pii_type] += 1
            logger.warning(
                "recall-probe leak type=%s length=%d",
                pii_type.value,
                len(value),
            )
    if leaks:
        pii_type, count = leaks.most_common(1)[0]
        raise LeakDetectedError(pii_type, count)


def _fits_single_block(doc: DocxDocument, entity: PIIEntity) -> bool:
    """True when the entity lies entirely inside one paragraph block."""
    try:
        idx = doc._block_index_for_offset(entity.start)  # noqa: SLF001
    except DocumentError:
        return False
    _, block_end = doc._block_spans[idx]  # noqa: SLF001
    return entity.end <= block_end


class Redactor:
    """Thin pipeline sequencer over detectors, resolution, surrogates, and document."""

    def __init__(self, config: RedactorConfig | None = None) -> None:
        self.config = config if config is not None else RedactorConfig.default()
        self._detectors = get_detectors(self.config)

    def detect(self, text: str) -> list[PIIEntity]:
        entities: list[PIIEntity] = []
        for detector in self._detectors:
            found = detector.detect(text, self.config)
            logger.info(
                "detector %s type=%s count=%d",
                detector.name,
                detector.pii_type.value,
                len(found),
            )
            for entity in found:
                logger.debug(
                    "span type=%s source=%s offsets=%d:%d",
                    entity.pii_type.value,
                    entity.source,
                    entity.start,
                    entity.end,
                )
            entities.extend(found)
        entities.sort(key=lambda e: (e.start, e.end, e.source))
        assert_consistent(text, entities)
        return entities

    def _pipeline(self, text: str) -> tuple[list[PIIEntity], dict[str, str], str]:
        detected = self.detect(text)
        resolved = resolve(detected)
        logger.info(
            "resolve input=%d output=%d",
            len(detected),
            len(resolved),
        )
        expanded = expand_occurrences(text, resolved)
        factory = SurrogateFactory(self.config)
        assigned = factory.assign(expanded)
        logger.info("assigned replacements count=%d", len(assigned))
        redacted = apply_text(text, assigned)
        if self.config.verify_output:
            verify_no_leaks(assigned, redacted, original_text=text)
            verify_rule_recall(text, redacted, self.config)
        return assigned, dict(factory.mapping), redacted

    def redact_text(self, text: str) -> RedactionResult:
        assigned, mapping, redacted = self._pipeline(text)
        counts = Counter(e.pii_type for e in assigned)
        return RedactionResult(
            entities=tuple(assigned),
            mapping=mapping,
            counts_by_type=dict(counts),
            text=redacted,
        )

    def redact_document(
        self,
        src: Path | str,
        dst: Path | str | None = None,
        *,
        dry_run: bool = False,
    ) -> RedactionResult:
        source = Path(src)
        doc = DocxDocument(source)
        text = doc.extract_text()
        logger.info(
            "extracted chars=%d blocks=%d",
            len(text),
            len(doc.blocks),
        )

        detected = self.detect(text)
        resolved = resolve(detected)
        logger.info("resolve input=%d output=%d", len(detected), len(resolved))

        # Document apply cannot splice spans that cross paragraph boundaries
        # (known limitation: multi-block addresses). Drop before expand/assign
        # so a longer multi-block hit cannot suppress an in-block repeat and
        # then vanish, and so leak verification is not asked to prove
        # replacements we will never make.
        applicable = [e for e in resolved if _fits_single_block(doc, e)]
        dropped = len(resolved) - len(applicable)
        if dropped:
            logger.info("dropped cross-block spans count=%d", dropped)

        def _block_end(offset: int) -> int:
            idx = doc._block_index_for_offset(offset)  # noqa: SLF001
            return doc._block_spans[idx][1]  # noqa: SLF001

        expanded = expand_occurrences(
            text, applicable, block_end_for_offset=_block_end
        )

        factory = SurrogateFactory(self.config)
        assigned = factory.assign(expanded)
        logger.info("assigned replacements count=%d", len(assigned))
        mapping = dict(factory.mapping)
        counts = Counter(e.pii_type for e in assigned)
        preview = apply_text(text, assigned)

        if dry_run:
            logger.info("dry-run: skipping apply/save entities=%d", len(assigned))
            return RedactionResult(
                entities=tuple(assigned),
                mapping=mapping,
                counts_by_type=dict(counts),
                text=preview,
            )

        if dst is None:
            raise DocumentError("output path required when not dry-run")

        output = Path(dst)
        doc.apply(assigned)
        rendered = doc.rendered_text()
        if self.config.verify_output:
            verify_no_leaks(assigned, rendered, original_text=text)

        doc.save(output)
        if self.config.verify_output:
            try:
                verify_package_no_leaks(assigned, output)
                verify_rule_recall(
                    package_corpus(source),
                    package_corpus(output),
                    self.config,
                )
            except LeakDetectedError:
                output.unlink(missing_ok=True)
                raise

        logger.info(
            "wrote output entities=%d path_suffix=%s",
            len(assigned),
            output.suffix,
        )
        return RedactionResult(
            entities=tuple(assigned),
            mapping=mapping,
            counts_by_type=dict(counts),
            text=rendered,
        )
