"""test_redactor — Phase 3 pipeline skeleton (zero detectors)."""

from __future__ import annotations

from pathlib import Path

from docx import Document

from pii_redaction.document import DocxDocument
from pii_redaction.models import RedactorConfig
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


class TestRedactTextPhase3:
    def test_identity_for_empty_and_plain_text(self) -> None:
        redactor = Redactor(RedactorConfig(use_ner=False))
        for sample in ("", "   ", "hello world", "no pii here 123"):
            result = redactor.redact_text(sample)
            assert result.text == sample
            assert result.entities == ()
            assert dict(result.mapping) == {}
            assert dict(result.counts_by_type) == {}


class TestRedactDocumentPhase3:
    def test_load_extract_apply_nothing_save(self, tmp_path: Path) -> None:
        src = _write_simple_docx(tmp_path / "in.docx", "alpha beta gamma")
        dst = tmp_path / "out.docx"
        redactor = Redactor(RedactorConfig(use_ner=False, seed=0))
        result = redactor.redact_document(src, dst)

        assert dst.exists()
        assert result.entities == ()
        assert dict(result.counts_by_type) == {}

        original = DocxDocument(src).extract_text()
        copied = DocxDocument(dst).extract_text()
        assert copied == original

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        src = _write_simple_docx(tmp_path / "in.docx", "content")
        dst = tmp_path / "out.docx"
        redactor = Redactor(RedactorConfig(use_ner=False))
        result = redactor.redact_document(src, dst, dry_run=True)
        assert not dst.exists()
        assert result.entities == ()
