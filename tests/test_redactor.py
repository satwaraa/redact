"""test_redactor — Phase 4: detectors wired, still no replacements."""

from __future__ import annotations

from pathlib import Path

from docx import Document

from pii_redaction.document import DocxDocument
from pii_redaction.models import PIIType, RedactorConfig
from pii_redaction.redactor import Redactor


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


class TestRedactTextPhase4:
    def test_identity_when_no_pii(self) -> None:
        redactor = Redactor(RedactorConfig(use_ner=False))
        for sample in ("", "   ", "hello world", "no pii here"):
            result = redactor.redact_text(sample)
            assert result.text == sample
            assert result.entities == ()
            assert dict(result.mapping) == {}
            assert dict(result.counts_by_type) == {}

    def test_detects_without_replacing(self) -> None:
        text = "Contact rashhi.patil@gmail.com or +91 9876543210"
        redactor = Redactor(RedactorConfig(use_ner=False))
        result = redactor.redact_text(text)
        assert result.text == text  # no splicing yet
        assert dict(result.mapping) == {}
        types = {e.pii_type for e in result.entities}
        assert PIIType.EMAIL in types
        assert PIIType.PHONE in types
        assert result.counts_by_type[PIIType.EMAIL] >= 1
        assert result.counts_by_type[PIIType.PHONE] >= 1
        for entity in result.entities:
            assert text[entity.start : entity.end] == entity.text
            assert entity.replacement is None


class TestRedactDocumentPhase4:
    def test_save_unchanged_but_reports_detections(self, tmp_path: Path) -> None:
        src = _write_simple_docx(tmp_path / "in.docx", "mail a@b.co please")
        dst = tmp_path / "out.docx"
        redactor = Redactor(RedactorConfig(use_ner=False, seed=0))
        result = redactor.redact_document(src, dst)

        assert dst.exists()
        assert any(e.pii_type is PIIType.EMAIL for e in result.entities)
        assert DocxDocument(dst).extract_text() == DocxDocument(src).extract_text()

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        src = _write_simple_docx(tmp_path / "in.docx", "a@b.co")
        dst = tmp_path / "out.docx"
        redactor = Redactor(RedactorConfig(use_ner=False))
        result = redactor.redact_document(src, dst, dry_run=True)
        assert not dst.exists()
        assert any(e.pii_type is PIIType.EMAIL for e in result.entities)
