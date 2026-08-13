"""test_redactor — full pipeline: resolve, assign, apply, verify, logging."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from docx import Document

from pii_redaction.document import DocxDocument
from pii_redaction.models import (
    PRIORITY_REGEX,
    LeakDetectedError,
    PIIEntity,
    PIIType,
    RedactorConfig,
)
from pii_redaction.redactor import Redactor, apply_text, verify_no_leaks


def _write_simple_docx(path: Path, text: str) -> Path:
    doc = Document()
    body = doc.element.body
    for child in list(body):
        if child.tag.endswith("}sectPr"):
            continue
        body.remove(child)
    doc.add_paragraph(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    return path


def _cfg(**kwargs: object) -> RedactorConfig:
    base = {"use_ner": False, "seed": 0, "verify_output": True}
    base.update(kwargs)
    return RedactorConfig(**base)  # type: ignore[arg-type]


def test_redact_text_replaces_pii_and_keeps_non_pii() -> None:
    text = "Contact rashhi.patil@gmail.com or +91 9876543210 today"
    result = Redactor(_cfg()).redact_text(text)
    assert result.text is not None
    assert "rashhi.patil@gmail.com" not in result.text
    assert "+91 9876543210" not in result.text
    assert "Contact " in result.text
    assert " today" in result.text
    assert result.entities
    assert all(e.replacement for e in result.entities)
    assert dict(result.mapping)
    assert PIIType.EMAIL in result.counts_by_type
    assert PIIType.PHONE in result.counts_by_type


def test_assignment_example_regression() -> None:
    text = "Rashi Patil / rashhi.patil@gmail.com / +91 9876543210"
    result = Redactor(_cfg()).redact_text(text)
    assert result.text is not None
    assert "rashhi.patil@gmail.com" not in result.text
    assert "+91 9876543210" not in result.text


def test_no_pii_unchanged() -> None:
    for sample in ("", "   ", "hello world", "no pii here"):
        result = Redactor(_cfg()).redact_text(sample)
        assert result.text == sample
        assert result.entities == ()
        assert dict(result.mapping) == {}


def test_text_and_docx_paths_agree(tmp_path: Path) -> None:
    text = "mail a@b.co and call +91 9876543210 please"
    redactor = Redactor(_cfg(seed=42))
    text_result = redactor.redact_text(text)

    src = _write_simple_docx(tmp_path / "in.docx", text)
    dst = tmp_path / "out.docx"
    doc_result = Redactor(_cfg(seed=42)).redact_document(src, dst)

    assert text_result.text == DocxDocument(dst).extract_text()
    assert {e.pii_type for e in text_result.entities} == {
        e.pii_type for e in doc_result.entities
    }


def test_document_dry_run_writes_nothing(tmp_path: Path) -> None:
    src = _write_simple_docx(tmp_path / "in.docx", "a@b.co")
    dst = tmp_path / "out.docx"
    result = Redactor(_cfg()).redact_document(src, dst, dry_run=True)
    assert not dst.exists()
    assert any(e.pii_type is PIIType.EMAIL for e in result.entities)
    assert all(e.replacement for e in result.entities)


def test_leak_when_apply_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = _write_simple_docx(tmp_path / "in.docx", "secret@mail.com lives here")
    dst = tmp_path / "out.docx"

    def _noop_apply(self: DocxDocument, entities: object) -> None:  # noqa: ARG001
        return None

    monkeypatch.setattr(DocxDocument, "apply", _noop_apply)
    with pytest.raises(LeakDetectedError) as exc_info:
        Redactor(_cfg()).redact_document(src, dst)
    assert "EMAIL" in str(exc_info.value)
    assert "secret@mail.com" not in str(exc_info.value)
    assert not dst.exists()


def test_verify_output_false_skips_leak_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _write_simple_docx(tmp_path / "in.docx", "secret@mail.com")
    dst = tmp_path / "out.docx"
    monkeypatch.setattr(DocxDocument, "apply", lambda self, entities: None)
    result = Redactor(_cfg(verify_output=False)).redact_document(src, dst)
    assert dst.exists()
    assert result.entities


def test_short_surname_false_positive_guard() -> None:
    # "Ann" is below the leak length threshold and may survive inside "Anniversary"
    entities = [
        PIIEntity(
            pii_type=PIIType.FULL_NAME,
            text="Ann",
            start=0,
            end=3,
            source="test",
            priority=PRIORITY_REGEX,
            replacement="Zoe",
        )
    ]
    rendered = "Anniversary party"
    verify_no_leaks(entities, rendered, original_text=None)


def test_determinism_two_runs(tmp_path: Path) -> None:
    text = "email a@b.co phone +91 9876543210"
    a = Redactor(_cfg(seed=7)).redact_text(text)
    b = Redactor(_cfg(seed=7)).redact_text(text)
    assert a.text == b.text
    assert [e.replacement for e in a.entities] == [e.replacement for e in b.entities]


def test_logging_never_contains_pii_values(caplog: pytest.LogCaptureFixture) -> None:
    text = "reach rashhi.patil@gmail.com or +91 9876543210"
    with caplog.at_level(logging.DEBUG, logger="pii_redaction"):
        Redactor(_cfg()).redact_text(text)
    blob = " ".join(rec.getMessage() for rec in caplog.records)
    assert "rashhi.patil@gmail.com" not in blob
    assert "+91 9876543210" not in blob
    assert "9876543210" not in blob


def test_apply_text_right_to_left() -> None:
    text = "aa@b.co and cc@d.co"
    e1 = PIIEntity(
        pii_type=PIIType.EMAIL,
        text="aa@b.co",
        start=0,
        end=7,
        source="t",
        priority=PRIORITY_REGEX,
        replacement="ONE",
    )
    e2 = PIIEntity(
        pii_type=PIIType.EMAIL,
        text="cc@d.co",
        start=12,
        end=19,
        source="t",
        priority=PRIORITY_REGEX,
        replacement="TWO",
    )
    assert apply_text(text, [e1, e2]) == "ONE and TWO"
