"""Orchestration. Thin by construction — Phase 3: load, extract, apply nothing, save."""

from __future__ import annotations

import logging
from pathlib import Path

from pii_redaction.document import DocxDocument
from pii_redaction.models import DocumentError, RedactionResult, RedactorConfig

logger = logging.getLogger(__name__)


class Redactor:
    """Sequence the redaction pipeline. Phase 3 runs with zero detectors."""

    def __init__(self, config: RedactorConfig | None = None) -> None:
        self.config = config if config is not None else RedactorConfig.default()

    def redact_text(self, text: str) -> RedactionResult:
        """Pure text path. With zero detectors this is an identity transform."""
        _ = text  # retained for later detect → resolve → assign stages
        return RedactionResult(
            entities=(),
            mapping={},
            counts_by_type={},
            text=text,
        )

    def redact_document(
        self,
        src: Path | str,
        dst: Path | str | None = None,
        *,
        dry_run: bool = False,
    ) -> RedactionResult:
        """Docx path: load → extract → (no detections) → apply nothing → save."""
        source = Path(src)
        doc = DocxDocument(source)
        text = doc.extract_text()
        logger.info(
            "extracted %d characters across %d blocks",
            len(text),
            len(doc.blocks),
        )
        result = self.redact_text(text)
        # Zero detectors: nothing to resolve, assign, or apply.
        if dry_run:
            logger.info("dry-run: skipping save")
            return result
        if dst is None:
            raise DocumentError("output path required when not dry-run")
        doc.save(dst)
        logger.info("wrote output with 0 replacements")
        return result
