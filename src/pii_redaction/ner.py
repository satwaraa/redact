"""Model-based detection for names, companies, and free-form locations."""

from __future__ import annotations

import logging
import re
from typing import Any

from pii_redaction.models import (
    PRIORITY_NER,
    ModelUnavailableError,
    PIIEntity,
    PIIType,
    RedactorConfig,
)

logger = logging.getLogger(__name__)

_LABEL_MAP: dict[str, PIIType] = {
    "PERSON": PIIType.FULL_NAME,
    "PER": PIIType.FULL_NAME,
    "ORG": PIIType.COMPANY,
    "GPE": PIIType.ADDRESS,
    "LOC": PIIType.ADDRESS,
    "FAC": PIIType.ADDRESS,
    "DATE": PIIType.DOB,
}

# Boilerplate the model over-tags as ORG in prospectus-style documents.
NER_ORG_STOPWORDS: frozenset[str] = frozenset(
    {
        "prospectus",
        "red herring",
        "red herring prospectus",
        "annexure",
        "schedule",
        "the company",
        "company",
        "board of directors",
        "securities",
        "equity shares",
        "bse",
        "nse",
        "sebi",
        "registrar",
        "lead manager",
        "book running lead manager",
        "draft red herring prospectus",
        "drhp",
        "rhp",
    }
)

_BIRTH_CUES: frozenset[str] = frozenset(
    {
        "dob",
        "d.o.b",
        "d.o.b.",
        "date of birth",
        "birth date",
        "born",
        "birthday",
        "birthdate",
    }
)

# Module-level cache: model name → loaded spaCy Language. Never eager-load.
_NLP_CACHE: dict[str, Any] = {}

NER_PII_TYPES: frozenset[PIIType] = frozenset(
    {PIIType.FULL_NAME, PIIType.COMPANY, PIIType.ADDRESS, PIIType.DOB}
)


def _has_birth_cue(text: str, start: int, window: int = 40) -> bool:
    left = text[max(0, start - window) : start].lower()
    return any(cue in left for cue in sorted(_BIRTH_CUES, key=len, reverse=True))


def _load_nlp(model_name: str) -> Any:
    cached = _NLP_CACHE.get(model_name)
    if cached is not None:
        return cached
    try:
        import spacy
    except ImportError as exc:
        raise ModelUnavailableError(
            "spaCy is not installed. Install project deps, then: "
            f"python -m spacy download {model_name}"
        ) from exc
    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise ModelUnavailableError(
            f"spaCy model {model_name!r} is not installed. "
            f"Run: python -m spacy download {model_name}"
        ) from exc
    _NLP_CACHE[model_name] = nlp
    return nlp


def clear_nlp_cache() -> None:
    """Drop cached models (tests only)."""
    _NLP_CACHE.clear()


def iter_text_chunks(text: str, max_chunk_chars: int) -> list[tuple[int, str]]:
    """Split ``text`` on block (``\\n``) boundaries; return (base_offset, chunk)."""
    if not text:
        return []
    if max_chunk_chars <= 0 or len(text) <= max_chunk_chars:
        return [(0, text)]

    blocks = text.split("\n")
    block_starts: list[int] = []
    pos = 0
    for i, block in enumerate(blocks):
        block_starts.append(pos)
        pos += len(block) + (0 if i == len(blocks) - 1 else 1)

    chunks: list[tuple[int, str]] = []
    start_idx = 0
    while start_idx < len(blocks):
        end_idx = start_idx + 1
        while end_idx < len(blocks):
            chunk_start = block_starts[start_idx]
            chunk_end = block_starts[end_idx] + len(blocks[end_idx])
            if chunk_end - chunk_start > max_chunk_chars:
                break
            end_idx += 1
        chunk_start = block_starts[start_idx]
        last = end_idx - 1
        chunk_end = block_starts[last] + len(blocks[last])
        # Keep the joining newline inside this chunk when more blocks follow,
        # so slices remain contiguous and offsets stay aligned with ``text``.
        if last < len(blocks) - 1:
            chunk_end += 1
        chunks.append((chunk_start, text[chunk_start:chunk_end]))
        start_idx = end_idx
    return chunks


class NERDetector:
    """spaCy NER adapter satisfying the Detector protocol (multi-type emitter)."""

    name = "ner:spacy"
    # Representative type for Protocol / sorting; emit set is NER_PII_TYPES.
    pii_type = PIIType.FULL_NAME
    priority = PRIORITY_NER

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        *,
        confidence_threshold: float = 0.5,
        max_chunk_chars: int = 80_000,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_chunk_chars = max_chunk_chars
        # spaCy en_core_web_sm does not emit per-entity scores by default;
        # use a fixed documented confidence rather than inventing a score.
        self._fixed_confidence = 1.0

    @property
    def pii_types(self) -> frozenset[PIIType]:
        return NER_PII_TYPES

    def emits_for(self, enabled: frozenset[PIIType]) -> bool:
        return bool(self.pii_types & enabled)

    def detect(self, text: str, config: RedactorConfig) -> list[PIIEntity]:
        if not self.emits_for(config.enabled_types):
            return []
        if self._fixed_confidence < config.ner_confidence_threshold:
            return []

        nlp = _load_nlp(self.model_name)
        enabled = config.enabled_types
        results: list[PIIEntity] = []
        unmapped: dict[str, int] = {}

        for base, chunk in iter_text_chunks(text, self.max_chunk_chars):
            doc = nlp(chunk)
            for ent in doc.ents:
                label = ent.label_
                pii_type = _LABEL_MAP.get(label)
                if pii_type is None:
                    unmapped[label] = unmapped.get(label, 0) + 1
                    continue
                if pii_type not in enabled:
                    continue

                span_text = ent.text.strip()
                if not span_text or len(span_text) < 2:
                    continue
                if re.fullmatch(r"[\W_]+", span_text, flags=re.UNICODE):
                    continue

                lead = len(ent.text) - len(ent.text.lstrip())
                local_start = ent.start_char + lead
                local_end = local_start + len(span_text)
                abs_start = base + local_start
                abs_end = base + local_end
                if text[abs_start:abs_end] != span_text:
                    continue

                if pii_type is PIIType.DOB and not _has_birth_cue(text, abs_start):
                    continue
                if (
                    pii_type is PIIType.COMPANY
                    and span_text.casefold() in NER_ORG_STOPWORDS
                ):
                    continue

                results.append(
                    PIIEntity(
                        pii_type=pii_type,
                        text=span_text,
                        start=abs_start,
                        end=abs_end,
                        source=self.name,
                        confidence=self._fixed_confidence,
                        priority=self.priority,
                    )
                )

        for label, count in sorted(unmapped.items()):
            logger.debug("ner unmapped label=%s count=%d", label, count)
        results.sort(key=lambda e: (e.start, e.end, e.pii_type.value))
        return results
