"""Orchestration. Thin by construction — Phase 4: detect, then apply nothing."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from pii_redaction.detectors import get_detectors
from pii_redaction.document import DocxDocument
from pii_redaction.models import (
    DocumentError,
    PIIEntity,
    PIIType,
    RedactionResult,
    RedactorConfig,
    assert_consistent,
)

logger = logging.getLogger(__name__)


class Redactor:
    """Sequence the redaction pipeline.

    Phase 4 runs rule-based detectors and reports counts, but does not yet
    resolve overlaps, assign surrogates, or splice replacements.
    """

    def __init__(self, config: RedactorConfig | None = None) -> None:
        self.config = config if config is not None else RedactorConfig.default()
        self._detectors = get_detectors(self.config)

    def detect(self, text: str) -> list[PIIEntity]:
        entities: list[PIIEntity] = []
        for detector in self._detectors:
            found = detector.detect(text, self.config)
            logger.info(
                "detector %s (%s) matched %d span(s)",
                detector.name,
                detector.pii_type.value,
                len(found),
            )
            entities.extend(found)
        entities.sort(key=lambda e: (e.start, e.end, e.source))
        assert_consistent(text, entities)
        return entities

    def redact_text(self, text: str) -> RedactionResult:
        entities = self.detect(text)
        counts: Counter[PIIType] = Counter(e.pii_type for e in entities)
        return RedactionResult(
            entities=tuple(entities),
            mapping={},
            counts_by_type=dict(counts),
            text=text,
        )

    def redact_document(
        self,
        src: Path | str,
        dst: Path | str | None = None,
        *,
        dry_run: bool = False,
    ) -> RedactionResult:
        """Docx path: load → extract → detect → (no apply yet) → save copy."""
        source = Path(src)
        doc = DocxDocument(source)
        text = doc.extract_text()
        logger.info(
            "extracted %d characters across %d blocks",
            len(text),
            len(doc.blocks),
        )
        result = self.redact_text(text)
        # Phase 4: detections only — no resolve / assign / apply.
        if dry_run:
            logger.info("dry-run: skipping save (%d entities)", len(result.entities))
            return result
        if dst is None:
            raise DocumentError("output path required when not dry-run")
        doc.save(dst)
        logger.info(
            "wrote output unchanged (%d detections, 0 replacements)",
            len(result.entities),
        )
        return result
