"""test_ner — lazy spaCy adapter, chunk offsets, --no-ner stays fast."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from pii_redaction.detectors import get_detectors
from pii_redaction.models import (
    PRIORITY_NER,
    ModelUnavailableError,
    PIIType,
    RedactorConfig,
)
from pii_redaction.ner import (
    NERDetector,
    clear_nlp_cache,
    clip_at_newlines,
    iter_text_chunks,
    _is_field_label,
    _is_heading_block,
    _is_org_stopword,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_nlp_cache()
    yield
    clear_nlp_cache()


def _cfg(**kwargs: object) -> RedactorConfig:
    base: dict[str, object] = {
        "use_ner": True,
        "ner_model": "en_core_web_sm",
        "ner_confidence_threshold": 0.5,
    }
    base.update(kwargs)
    return RedactorConfig(**base)  # type: ignore[arg-type]


class _FakeSpan:
    def __init__(self, text: str, start_char: int, label: str) -> None:
        self.text = text
        self.start_char = start_char
        self.end_char = start_char + len(text)
        self.label_ = label


class _FakeDoc:
    def __init__(self, ents: list[_FakeSpan]) -> None:
        self.ents = ents


def _fake_nlp_factory(ents_by_chunk: dict[str, list[_FakeSpan]]) -> Any:
    def _nlp(chunk: str) -> _FakeDoc:
        return _FakeDoc(ents_by_chunk.get(chunk, []))

    return _nlp


def test_iter_text_chunks_preserves_absolute_offsets() -> None:
    text = "alpha block\nbeta block here\ngamma"
    chunks = iter_text_chunks(text, max_chunk_chars=20)
    assert len(chunks) >= 2
    for base, chunk in chunks:
        assert text[base : base + len(chunk)] == chunk
    # Chunks are contiguous non-overlapping slices covering the whole string
    assert chunks[0][0] == 0
    assert chunks[-1][0] + len(chunks[-1][1]) == len(text)
    for i in range(len(chunks) - 1):
        assert chunks[i][0] + len(chunks[i][1]) == chunks[i + 1][0]


def test_clip_at_newlines_noop_without_newline() -> None:
    text = "Acme Technologies Pvt Ltd"
    assert clip_at_newlines(text, 0, len(text)) == [(0, len(text))]


def test_clip_at_newlines_splits_and_strips() -> None:
    text = "prefix Acme Corp\n  Headquarters end"
    start = text.index("Acme")
    end = text.index("end")
    pieces = clip_at_newlines(text, start, end)
    assert [(text[s:e], s, e) for s, e in pieces] == [
        ("Acme Corp", start, start + len("Acme Corp")),
        ("Headquarters", text.index("Headquarters"), text.index("Headquarters") + len("Headquarters")),
    ]
    assert all("\n" not in text[s:e] for s, e in pieces)


def test_cross_block_ner_span_is_clipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """B1: a span that straddles ``\\n`` must become in-block pieces."""
    text = "Acme Technologies Pvt Ltd is\nHeadquarters listed"
    cross = text  # whole string as one ORG, including the join newline
    assert "\n" in cross

    def _nlp(chunk: str) -> _FakeDoc:
        return _FakeDoc([_FakeSpan(cross, 0, "ORG")])

    monkeypatch.setattr("pii_redaction.ner._load_nlp", lambda _name: _nlp)
    entities = NERDetector().detect(text, _cfg())
    assert entities
    assert all("\n" not in e.text for e in entities)
    assert all(text[e.start : e.end] == e.text for e in entities)
    # B4: only the piece with a legal suffix survives
    assert [e.text for e in entities] == ["Acme Technologies Pvt Ltd is"]


def test_cross_block_clip_drops_whitespace_only_piece(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Acme Corp Ltd\n   \nBeta Ltd"
    cross = text

    def _nlp(chunk: str) -> _FakeDoc:
        return _FakeDoc([_FakeSpan(cross, 0, "ORG")])

    monkeypatch.setattr("pii_redaction.ner._load_nlp", lambda _name: _nlp)
    entities = NERDetector().detect(text, _cfg())
    assert [e.text for e in entities] == ["Acme Corp Ltd", "Beta Ltd"]


def test_chunk_boundary_entity_gets_absolute_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "first paragraph only\nAda Lovelace wrote notes"
    # Force a tiny budget so the second block is its own chunk
    chunks = iter_text_chunks(text, max_chunk_chars=10)
    assert len(chunks) >= 2
    second_base, second_chunk = next(c for c in chunks if "Ada" in c[1])
    local = second_chunk.index("Ada Lovelace")
    fake_ents = {
        second_chunk: [_FakeSpan("Ada Lovelace", local, "PERSON")],
    }
    monkeypatch.setattr(
        "pii_redaction.ner._load_nlp",
        lambda _name: _fake_nlp_factory(fake_ents),
    )
    det = NERDetector(max_chunk_chars=10)
    entities = det.detect(text, _cfg())
    assert len(entities) == 1
    ent = entities[0]
    assert ent.pii_type is PIIType.FULL_NAME
    assert ent.start == second_base + local
    assert text[ent.start : ent.end] == "Ada Lovelace"
    assert ent.priority == PRIORITY_NER
    assert ent.source == "ner:spacy"


def test_date_requires_birth_cue(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "meeting on 12 March 1988 and DOB 12 March 1988"
    # One DATE span at the birth-cued occurrence only would be ideal; feed both
    meeting = text.index("12 March 1988")
    dob = text.rindex("12 March 1988")

    def _nlp(chunk: str) -> _FakeDoc:
        return _FakeDoc(
            [
                _FakeSpan("12 March 1988", meeting, "DATE"),
                _FakeSpan("12 March 1988", dob, "DATE"),
            ]
        )

    monkeypatch.setattr("pii_redaction.ner._load_nlp", lambda _name: _nlp)
    entities = NERDetector().detect(text, _cfg())
    assert len(entities) == 1
    assert entities[0].pii_type is PIIType.DOB
    assert entities[0].start == dob


def test_org_stopwords_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "see the Prospectus and Acme Technologies Pvt Ltd here"

    def _nlp(chunk: str) -> _FakeDoc:
        return _FakeDoc(
            [
                _FakeSpan("Prospectus", text.index("Prospectus"), "ORG"),
                _FakeSpan(
                    "Acme Technologies Pvt Ltd",
                    text.index("Acme Technologies Pvt Ltd"),
                    "ORG",
                ),
            ]
        )

    monkeypatch.setattr("pii_redaction.ner._load_nlp", lambda _name: _nlp)
    entities = NERDetector().detect(text, _cfg())
    assert [e.text for e in entities] == ["Acme Technologies Pvt Ltd"]
    assert entities[0].pii_type is PIIType.COMPANY


def test_org_stopword_containment_and_normalise() -> None:
    assert _is_org_stopword("the Offer")
    assert _is_org_stopword("Risk Management Committee")
    assert _is_org_stopword("Equity Shares")
    assert not _is_org_stopword("Acme Corp")
    assert not _is_org_stopword("Link Intime India Private Limited")


def test_org_stopword_containment_drops_span(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "see the Offer and Acme Components Private Limited"

    def _nlp(chunk: str) -> _FakeDoc:
        return _FakeDoc(
            [
                _FakeSpan("the Offer", text.index("the Offer"), "ORG"),
                _FakeSpan(
                    "Acme Components Private Limited",
                    text.index("Acme Components Private Limited"),
                    "ORG",
                ),
            ]
        )

    monkeypatch.setattr("pii_redaction.ner._load_nlp", lambda _name: _nlp)
    entities = NERDetector().detect(text, _cfg())
    assert [e.text for e in entities] == ["Acme Components Private Limited"]


def test_doc_freq_filter_rejects_boilerplate(monkeypatch: pytest.MonkeyPatch) -> None:
    # Phrase not on the stopword list, but repeated above the default threshold.
    phrase = "Zorp Widget Limited"
    text = "\n".join([f"{phrase} appears here"] * 16 + ["Acme Corp Ltd once"])

    def _nlp(chunk: str) -> _FakeDoc:
        ents = []
        start = 0
        while True:
            i = chunk.find(phrase, start)
            if i < 0:
                break
            ents.append(_FakeSpan(phrase, i, "ORG"))
            start = i + 1
        j = chunk.find("Acme Corp Ltd")
        if j >= 0:
            ents.append(_FakeSpan("Acme Corp Ltd", j, "ORG"))
        return _FakeDoc(ents)

    monkeypatch.setattr("pii_redaction.ner._load_nlp", lambda _name: _nlp)
    entities = NERDetector().detect(text, _cfg())
    assert [e.text for e in entities] == ["Acme Corp Ltd"]


def test_heading_block_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "REGISTRAR TO THE OFFER\nAda Lovelace signed"

    def _nlp(chunk: str) -> _FakeDoc:
        return _FakeDoc(
            [
                _FakeSpan(
                    "REGISTRAR TO THE OFFER",
                    text.index("REGISTRAR TO THE OFFER"),
                    "ORG",
                ),
                _FakeSpan("Ada Lovelace", text.index("Ada Lovelace"), "PERSON"),
            ]
        )

    monkeypatch.setattr("pii_redaction.ner._load_nlp", lambda _name: _nlp)
    entities = NERDetector().detect(text, _cfg())
    assert [e.text for e in entities] == ["Ada Lovelace"]
    assert entities[0].pii_type is PIIType.FULL_NAME


def test_field_label_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Email: Ada Lovelace"

    def _nlp(chunk: str) -> _FakeDoc:
        return _FakeDoc(
            [
                _FakeSpan("Email", text.index("Email"), "PERSON"),
                _FakeSpan("Ada Lovelace", text.index("Ada Lovelace"), "PERSON"),
            ]
        )

    monkeypatch.setattr("pii_redaction.ner._load_nlp", lambda _name: _nlp)
    entities = NERDetector().detect(text, _cfg())
    assert [e.text for e in entities] == ["Ada Lovelace"]
    assert _is_field_label("Email")
    assert _is_field_label("Contact Person")
    assert _is_heading_block("REGISTRAR TO THE OFFER")
    assert not _is_heading_block("Ada Lovelace")


def test_agreement_rejects_cue_only_person(monkeypatch: pytest.MonkeyPatch) -> None:
    """B8: single-token name with a cue passes B4 but fails structural agreement."""
    text = "Director: Meera reviewed the draft"

    def _nlp(chunk: str) -> _FakeDoc:
        return _FakeDoc([_FakeSpan("Meera", text.index("Meera"), "PERSON")])

    monkeypatch.setattr("pii_redaction.ner._load_nlp", lambda _name: _nlp)
    assert [e.text for e in NERDetector().detect(text, _cfg())] == ["Meera"]
    assert (
        NERDetector(require_agreement=True).detect(text, _cfg(ner_agreement=True)) == []
    )


def test_agreement_rejects_contact_block_org_without_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Zorp Holdings\nEmail: desk@zorp.example"

    def _nlp(chunk: str) -> _FakeDoc:
        return _FakeDoc(
            [_FakeSpan("Zorp Holdings", text.index("Zorp Holdings"), "ORG")]
        )

    monkeypatch.setattr("pii_redaction.ner._load_nlp", lambda _name: _nlp)
    assert [e.text for e in NERDetector().detect(text, _cfg())] == ["Zorp Holdings"]
    assert (
        NERDetector(require_agreement=True).detect(text, _cfg(ner_agreement=True)) == []
    )


def test_agreement_keeps_legal_suffix_org(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Acme Components Private Limited filed today"

    def _nlp(chunk: str) -> _FakeDoc:
        return _FakeDoc(
            [
                _FakeSpan(
                    "Acme Components Private Limited",
                    text.index("Acme Components Private Limited"),
                    "ORG",
                )
            ]
        )

    monkeypatch.setattr("pii_redaction.ner._load_nlp", lambda _name: _nlp)
    entities = NERDetector(require_agreement=True).detect(
        text, _cfg(ner_agreement=True)
    )
    assert [e.text for e in entities] == ["Acme Components Private Limited"]


def test_person_requires_positive_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Solo met Ada Lovelace near the desk"

    def _nlp(chunk: str) -> _FakeDoc:
        return _FakeDoc(
            [
                _FakeSpan("Solo", text.index("Solo"), "PERSON"),
                _FakeSpan("Ada Lovelace", text.index("Ada Lovelace"), "PERSON"),
            ]
        )

    monkeypatch.setattr("pii_redaction.ner._load_nlp", lambda _name: _nlp)
    entities = NERDetector().detect(text, _cfg())
    assert [e.text for e in entities] == ["Ada Lovelace"]


def test_person_cue_allows_single_token(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Director: Meera reviewed the draft"

    def _nlp(chunk: str) -> _FakeDoc:
        return _FakeDoc([_FakeSpan("Meera", text.index("Meera"), "PERSON")])

    monkeypatch.setattr("pii_redaction.ner._load_nlp", lambda _name: _nlp)
    entities = NERDetector().detect(text, _cfg())
    assert [e.text for e in entities] == ["Meera"]


def test_person_gazetteer_from_director_lines() -> None:
    from pii_redaction.ner import build_person_gazetteer

    text = (
        "Rajesh Kumar Sharma, Managing Director\n"
        "Sneha Patel, Independent Director\n"
        "Unrelated line without cues\n"
    )
    gaz = build_person_gazetteer(text)
    assert "rajesh kumar sharma" in gaz
    assert "sneha patel" in gaz
    assert "unrelated line without cues" not in gaz


def test_org_requires_legal_suffix_or_contact_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = (
        "Zorp Holdings\n"
        "Email: desk@zorp.example\n"
        "Unrelated paragraph filler\n"
        "Soft Balloons nearby"
    )

    def _nlp(chunk: str) -> _FakeDoc:
        return _FakeDoc(
            [
                _FakeSpan("Zorp Holdings", text.index("Zorp Holdings"), "ORG"),
                _FakeSpan("Soft Balloons", text.index("Soft Balloons"), "ORG"),
            ]
        )

    monkeypatch.setattr("pii_redaction.ner._load_nlp", lambda _name: _nlp)
    entities = NERDetector().detect(text, _cfg())
    assert [e.text for e in entities] == ["Zorp Holdings"]


def test_model_unavailable_names_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_nlp_cache()

    def _missing(_name: str) -> Any:
        raise OSError("Can't find model")

    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(load=_missing))
    with pytest.raises(ModelUnavailableError, match="spacy download") as exc_info:
        NERDetector(model_name="en_core_web_sm").detect("Ada", _cfg())
    assert "en_core_web_sm" in str(exc_info.value)


def test_get_detectors_includes_ner_only_when_enabled() -> None:
    with_ner = get_detectors(_cfg(use_ner=True))
    without = get_detectors(_cfg(use_ner=False))
    assert any(d.name == "ner:spacy" for d in with_ner)
    assert all(d.name != "ner:spacy" for d in without)


def test_no_ner_path_does_not_load_spacy(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = {"called": False}

    def _load(_name: str) -> Any:
        loaded["called"] = True
        raise AssertionError("should not load")

    monkeypatch.setattr("pii_redaction.ner._load_nlp", _load)
    from pii_redaction.redactor import Redactor

    result = Redactor(_cfg(use_ner=False)).redact_text("Ada Lovelace met Bob")
    assert loaded["called"] is False
    # Without NER, plain names are not rule-detected
    assert all(e.pii_type is not PIIType.FULL_NAME for e in result.entities)


def test_help_still_avoids_spacy(capsys: pytest.CaptureFixture[str]) -> None:
    from pii_redaction.cli import main

    sys.modules.pop("spacy", None)
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    assert "spacy" not in sys.modules
    assert "usage:" in capsys.readouterr().out.lower()
