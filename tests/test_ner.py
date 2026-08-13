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
    iter_text_chunks,
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
    text = "see the Prospectus and Acme Corp here"

    def _nlp(chunk: str) -> _FakeDoc:
        return _FakeDoc(
            [
                _FakeSpan("Prospectus", text.index("Prospectus"), "ORG"),
                _FakeSpan("Acme Corp", text.index("Acme Corp"), "ORG"),
            ]
        )

    monkeypatch.setattr("pii_redaction.ner._load_nlp", lambda _name: _nlp)
    entities = NERDetector().detect(text, _cfg())
    assert [e.text for e in entities] == ["Acme Corp"]
    assert entities[0].pii_type is PIIType.COMPANY


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
