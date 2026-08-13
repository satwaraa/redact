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

# Boilerplate the model over-tags as ORG in prospectus-style documents (B3).
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
        "offer",
        "the offer",
        "allotment",
        "bidder",
        "anchor investor",
        "promoter",
        "selling shareholder",
        "book building",
        "book building process",
        "roc",
        "face value",
        "mandate",
        "risk management committee",
        "non-resident",
        "non resident",
        "fresh issue",
        "offer for sale",
        "objects of the offer",
    }
)

# Field labels the model sometimes emits as PERSON/ORG (B6).
NER_FIELD_LABELS: frozenset[str] = frozenset(
    {
        "email",
        "e-mail",
        "e mail",
        "telephone",
        "website",
        "address",
        "contact person",
        "contact",
        "fax",
        "tel",
        "phone",
        "mobile",
        "registered office",
        "corporate office",
        "cin",
        "din",
        "pan",
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

# Person cues near a PERSON span (B4); longer phrases first when matching.
_PERSON_CUES: tuple[str, ...] = tuple(
    sorted(
        {
            "managing director",
            "independent director",
            "non-executive director",
            "company secretary",
            "compliance officer",
            "authorised signatory",
            "authorized signatory",
            "contact person",
            "director",
            "signatory",
            "signature",
            "signed by",
            "chairman",
            "mr.",
            "mrs.",
            "ms.",
            "dr.",
            "shri",
            "smt.",
            "s/o",
            "d/o",
            "w/o",
        },
        key=len,
        reverse=True,
    )
)

_GAZETTEER_BLOCK_CUES = re.compile(
    r"(?i)\b(?:"
    r"director|signator(?:y|ies)?|signed\s+by|managing\s+director|"
    r"independent\s+director|non-executive\s+director|chairman|"
    r"company\s+secretary|compliance\s+officer|"
    r"authorised\s+signatory|authorized\s+signatory"
    r")\b"
)

_TITLE_CASE_NAME = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b"
)

_LEGAL_SUFFIX = re.compile(
    r"(?i)\b(?:"
    r"pvt\.?\s*ltd\.?|private\s+limited|ltd\.?|limited|llp|inc\.?"
    r")\b"
)

_CONTACT_BLOCK_CUES = re.compile(
    r"(?i)\b(?:"
    r"e-?mail|telephone|tel\.?|phone|fax|website|address|"
    r"contact\s+person|registered\s+office|corporate\s+office"
    r")\b"
)

# Reject NER candidates seen more often than this across the document (B2).
DEFAULT_NER_MAX_DOC_FREQ = 15

# Types subject to the document-frequency precision guard.
_FREQ_FILTER_TYPES: frozenset[PIIType] = frozenset(
    {PIIType.FULL_NAME, PIIType.COMPANY, PIIType.ADDRESS}
)

# Module-level cache: model name → loaded spaCy Language. Never eager-load.
_NLP_CACHE: dict[str, Any] = {}

NER_PII_TYPES: frozenset[PIIType] = frozenset(
    {PIIType.FULL_NAME, PIIType.COMPANY, PIIType.ADDRESS, PIIType.DOB}
)


def _normalize_span(text: str) -> str:
    """Casefold, strip, collapse whitespace for stopword / label checks."""
    return re.sub(r"\s+", " ", text.casefold().strip())


def _has_birth_cue(text: str, start: int, window: int = 40) -> bool:
    left = text[max(0, start - window) : start].lower()
    return any(cue in left for cue in sorted(_BIRTH_CUES, key=len, reverse=True))


def _is_field_label(span: str) -> bool:
    """B6: bare field labels are never PII."""
    norm = _normalize_span(span).rstrip(":").strip()
    return norm in NER_FIELD_LABELS


def _is_org_stopword(span: str) -> bool:
    """B3: exact normalised match, or stopword contained as a whole phrase."""
    norm = _normalize_span(span)
    if not norm:
        return False
    if norm in NER_ORG_STOPWORDS:
        return True
    for stop in NER_ORG_STOPWORDS:
        if stop in norm and re.search(
            rf"(?<!\w){re.escape(stop)}(?!\w)", norm, flags=re.UNICODE
        ):
            return True
    return False


def _block_bounds(text: str, offset: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    if end < 0:
        end = len(text)
    return start, end


def _is_heading_block(block: str) -> bool:
    """B5: ALL-CAPS (and short all-caps-like) blocks are section titles."""
    stripped = block.strip()
    if not stripped:
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    if all(c.isupper() for c in letters):
        return True
    # Short standalone title that is entirely stopword lexicon after normalise.
    words = stripped.split()
    if len(words) <= 6 and len(stripped) <= 60 and _is_org_stopword(stripped):
        return True
    return False


def _doc_frequency(text: str, span: str, cache: dict[str, int]) -> int:
    """Count casefolded occurrences of ``span`` in ``text`` (B2)."""
    key = span.casefold()
    cached = cache.get(key)
    if cached is not None:
        return cached
    if not key:
        cache[key] = 0
        return 0
    count = text.casefold().count(key)
    cache[key] = count
    return count


def _title_case_token_count(span: str) -> int:
    """Count Title-Case alphabetic tokens (first upper, rest lower / single upper)."""
    count = 0
    for tok in re.findall(r"[A-Za-z][A-Za-z'’\-]*", span):
        if len(tok) == 1:
            if tok.isupper():
                count += 1
            continue
        if tok[0].isupper() and tok[1:].islower():
            count += 1
    return count


def _has_person_cue(text: str, start: int, end: int, window: int = 48) -> bool:
    """True when a person cue sits near the span (B4)."""
    left = text[max(0, start - window) : start]
    right = text[end : min(len(text), end + window)]
    region = f"{left} {right}".casefold()
    return any(cue in region for cue in _PERSON_CUES)


def build_person_gazetteer(text: str) -> frozenset[str]:
    """Names harvested from director / signatory lines (B4 gazetteer)."""
    names: set[str] = set()
    for block in text.split("\n"):
        if not _GAZETTEER_BLOCK_CUES.search(block):
            continue
        for match in _TITLE_CASE_NAME.finditer(block):
            name = match.group(1)
            if _is_org_stopword(name) or _is_field_label(name):
                continue
            if _title_case_token_count(name) < 2:
                continue
            names.add(name.casefold())
    return frozenset(names)


def _has_legal_suffix(span: str) -> bool:
    """ORG legal-form suffix (B4)."""
    return _LEGAL_SUFFIX.search(span) is not None


def _in_contact_block(text: str, start: int, *, radius: int = 1) -> bool:
    """True when this block or a neighbour looks like a contact block (B4)."""
    block_start, _ = _block_bounds(text, start)
    idx = text.count("\n", 0, block_start)
    blocks = text.split("\n")
    if idx >= len(blocks):
        return False
    lo = max(0, idx - radius)
    hi = min(len(blocks), idx + radius + 1)
    return _CONTACT_BLOCK_CUES.search("\n".join(blocks[lo:hi])) is not None


def _person_positive(
    text: str,
    span: str,
    start: int,
    end: int,
    gazetteer: frozenset[str],
) -> bool:
    """B4: PERSON needs title-case tokens, a nearby cue, or gazetteer membership."""
    if _title_case_token_count(span) >= 2:
        return True
    if _has_person_cue(text, start, end):
        return True
    return span.casefold() in gazetteer


def _org_positive(text: str, span: str, start: int) -> bool:
    """B4: ORG needs a legal suffix or a contact-block context."""
    if _has_legal_suffix(span):
        return True
    return _in_contact_block(text, start)


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


def clip_at_newlines(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split ``text[start:end]`` on ``\\n``; return stripped in-block ``(start, end)``.

    NER often tags spans that cross the block join used by ``extract_text()``.
    Those spans win resolution on length, then cannot be applied. Clipping keeps
    the in-block pieces so they remain usable and stop smothering shorter hits.
    """
    if start < 0 or end < start or end > len(text):
        return []
    span = text[start:end]
    if "\n" not in span:
        return [(start, end)] if span else []

    pieces: list[tuple[int, int]] = []
    offset = start
    for part in span.split("\n"):
        if part:
            lead = len(part) - len(part.lstrip())
            stripped = part.strip()
            if stripped:
                piece_start = offset + lead
                pieces.append((piece_start, piece_start + len(stripped)))
        offset += len(part) + 1
    return pieces


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
        max_doc_freq: int = DEFAULT_NER_MAX_DOC_FREQ,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_chunk_chars = max_chunk_chars
        self.max_doc_freq = max_doc_freq
        # spaCy en_core_web_sm does not emit per-entity scores by default;
        # use a fixed documented confidence rather than inventing a score.
        self._fixed_confidence = 1.0

    @property
    def pii_types(self) -> frozenset[PIIType]:
        return NER_PII_TYPES

    def emits_for(self, enabled: frozenset[PIIType]) -> bool:
        return bool(self.pii_types & enabled)

    def _accept_piece(
        self,
        text: str,
        pii_type: PIIType,
        piece: str,
        piece_start: int,
        freq_cache: dict[str, int],
        *,
        max_doc_freq: int,
        gazetteer: frozenset[str],
    ) -> bool:
        if len(piece) < 2:
            return False
        if re.fullmatch(r"[\W_]+", piece, flags=re.UNICODE):
            return False
        if _is_field_label(piece):
            return False
        block_start, block_end = _block_bounds(text, piece_start)
        if _is_heading_block(text[block_start:block_end]):
            return False
        if pii_type is PIIType.DOB and not _has_birth_cue(text, piece_start):
            return False
        if pii_type is PIIType.COMPANY and _is_org_stopword(piece):
            return False
        if pii_type is PIIType.FULL_NAME and not _person_positive(
            text, piece, piece_start, piece_start + len(piece), gazetteer
        ):
            return False
        if pii_type is PIIType.COMPANY and not _org_positive(text, piece, piece_start):
            return False
        if pii_type in _FREQ_FILTER_TYPES:
            if _doc_frequency(text, piece, freq_cache) > max_doc_freq:
                return False
        return True

    def detect(self, text: str, config: RedactorConfig) -> list[PIIEntity]:
        if not self.emits_for(config.enabled_types):
            return []
        if self._fixed_confidence < config.ner_confidence_threshold:
            return []

        nlp = _load_nlp(self.model_name)
        enabled = config.enabled_types
        results: list[PIIEntity] = []
        unmapped: dict[str, int] = {}
        freq_cache: dict[str, int] = {}
        max_doc_freq = config.ner_max_doc_freq
        gazetteer = build_person_gazetteer(text)

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

                for piece_start, piece_end in clip_at_newlines(text, abs_start, abs_end):
                    piece = text[piece_start:piece_end]
                    if not self._accept_piece(
                        text,
                        pii_type,
                        piece,
                        piece_start,
                        freq_cache,
                        max_doc_freq=max_doc_freq,
                        gazetteer=gazetteer,
                    ):
                        continue
                    results.append(
                        PIIEntity(
                            pii_type=pii_type,
                            text=piece,
                            start=piece_start,
                            end=piece_end,
                            source=self.name,
                            confidence=self._fixed_confidence,
                            priority=self.priority,
                        )
                    )

        for label, count in sorted(unmapped.items()):
            logger.debug("ner unmapped label=%s count=%d", label, count)
        results.sort(key=lambda e: (e.start, e.end, e.pii_type.value))
        return results
