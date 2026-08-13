"""Model-based detection for names, companies, and free-form locations."""

from __future__ import annotations

import logging
import re
from collections import Counter
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

# Lexical precision guards, in place of a document-frequency threshold.
#
# Frequency was measured and rejected: in an IPO prospectus the promoter family
# is named on nearly every page, so the most sensitive names are also the most
# frequent strings. On a hand-labelled sample of 120 NER detections, document
# frequency scored AUC 0.588 as a false-positive signal (0.50 = no signal), and
# a threshold of 15 discarded every promoter name in this document while still
# leaving 65% of survivors mis-tagged.
#
# These two rules were measured on the same sample and do separate:
#   determiner prefix     kills 22 false positives, 0 true positives
#   all-tokens-common     kills 43 false positives, 2 true positives
# Combined: recall 94.1%, precision 49.2% (baseline 28.3%), rejecting 43.3% of
# NER detections document-wide and 0 of 111 gold person-name detections.
_DETERMINERS: frozenset[str] = frozenset(
    {"the", "our", "such", "each", "any", "this"}
)

# A token must appear lowercase at least this often to count as common English.
_COMMON_TOKEN_MIN_COUNT = 2

# Types subject to the lexical precision guards.
_LEXICAL_FILTER_TYPES: frozenset[PIIType] = frozenset(
    {PIIType.FULL_NAME, PIIType.COMPANY, PIIType.ADDRESS}
)

_ALPHA_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
_LOWER_TOKEN = re.compile(r"(?<![A-Za-z'’\-])[a-z][a-z'’\-]+")

# Email local parts, domains and URLs are lowercase by convention and are not
# prose. Counting them would make "ananya.krishnan@example.com" evidence that
# "ananya" is ordinary vocabulary, when it is evidence of the opposite.
_NON_PROSE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+|https?://\S+|www\.\S+")

# Module-level cache: model name → loaded spaCy Language. Never eager-load.
_NLP_CACHE: dict[str, Any] = {}

NER_PII_TYPES: frozenset[PIIType] = frozenset(
    {PIIType.FULL_NAME, PIIType.COMPANY, PIIType.ADDRESS, PIIType.DOB}
)


# Unicode spaces Word emits inside table cells and between name parts. Each is
# a single character, so translating them preserves every offset exactly — the
# model sees ordinary spaces while entity text still comes from the original.
_SPACE_TRANSLATION = str.maketrans(
    {
        " ": " ",  # no-break space — "Robert Aragon" in Word tables
        " ": " ",  # figure space
        " ": " ",  # narrow no-break space
        " ": " ",  # thin space
        " ": " ",  # hair space
        " ": " ",  # en space
        " ": " ",  # em space
    }
)


def normalise_spaces(text: str) -> str:
    """Map exotic Unicode spaces to U+0020 without changing length or offsets."""
    return text.translate(_SPACE_TRANSLATION)


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
    return len(words) <= 6 and len(stripped) <= 60 and _is_org_stopword(stripped)


def build_lowercase_vocabulary(
    text: str, *, min_count: int = _COMMON_TOKEN_MIN_COUNT
) -> frozenset[str]:
    """Tokens that appear in lowercase across the document.

    A word the document also writes in lowercase is ordinary vocabulary, not a
    proper noun. "equity", "shares" and "committee" appear lowercase; "Hegde"
    and "Malvadkar" never do. This is the document telling us its own lexicon,
    which is why it transfers to documents we have not seen.
    """
    prose = _NON_PROSE.sub(" ", text)
    counts: Counter[str] = Counter(_LOWER_TOKEN.findall(prose))
    return frozenset(tok for tok, n in counts.items() if n >= min_count)


def _starts_with_determiner(span: str) -> bool:
    """Reject "the Offer", "our Company" — a name does not start with these."""
    tokens = _ALPHA_TOKEN.findall(span)
    return bool(tokens) and tokens[0].casefold() in _DETERMINERS


def _all_tokens_common(span: str, vocabulary: frozenset[str]) -> bool:
    """True when every alphabetic token also appears lowercase in the document.

    "Equity Shares" and "Risk Management Committee" are entirely ordinary
    vocabulary; a real name contains at least one token the document never
    writes lowercase.
    """
    tokens = _ALPHA_TOKEN.findall(span)
    if not tokens:
        return False
    return all(tok.casefold() in vocabulary for tok in tokens)


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


def _person_rule_agree(span: str) -> bool:
    """B8 structural heuristic for PERSON: looks like a multi-token name."""
    return _title_case_token_count(span) >= 2


def _org_rule_agree(span: str) -> bool:
    """B8 structural heuristic for ORG: carries a legal-form suffix."""
    return _has_legal_suffix(span)


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
        require_agreement: bool = False,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_chunk_chars = max_chunk_chars
        self.require_agreement = require_agreement
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
        *,
        vocabulary: frozenset[str],
        gazetteer: frozenset[str],
        require_agreement: bool,
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
        # A phrase from the document's defined-term lexicon is not a person
        # either: "Promoter Selling Shareholder" is title-cased throughout a
        # prospectus and spaCy tags it PERSON as readily as ORG.
        if pii_type in (PIIType.COMPANY, PIIType.FULL_NAME) and _is_org_stopword(piece):
            return False
        if pii_type is PIIType.FULL_NAME and not _person_positive(
            text, piece, piece_start, piece_start + len(piece), gazetteer
        ):
            return False
        if pii_type is PIIType.COMPANY and not _org_positive(text, piece, piece_start):
            return False
        # B8: model already proposed the span; also require the structural rule.
        if require_agreement:
            if pii_type is PIIType.FULL_NAME and not _person_rule_agree(piece):
                return False
            if pii_type is PIIType.COMPANY and not _org_rule_agree(piece):
                return False
        if pii_type in _LEXICAL_FILTER_TYPES:
            if _starts_with_determiner(piece):
                return False
            if _all_tokens_common(piece, vocabulary):
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
        require_agreement = config.ner_agreement
        normalised = normalise_spaces(text)
        gazetteer = build_person_gazetteer(normalised)
        vocabulary = build_lowercase_vocabulary(normalised)

        for base, chunk in iter_text_chunks(normalised, self.max_chunk_chars):
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
                # The model reads normalised text; entities must carry the
                # ORIGINAL slice so text[start:end] == entity.text still holds.
                if normalised[abs_start:abs_end] != span_text:
                    continue
                span_text = text[abs_start:abs_end]

                for piece_start, piece_end in clip_at_newlines(text, abs_start, abs_end):
                    piece = text[piece_start:piece_end]
                    if not self._accept_piece(
                        text,
                        pii_type,
                        piece,
                        piece_start,
                        vocabulary=vocabulary,
                        gazetteer=gazetteer,
                        require_agreement=require_agreement,
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
