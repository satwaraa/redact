"""test_cli — Phase 3 argument handling, exit codes, dump-text."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from docx import Document

from pii_redaction.cli import build_parser, config_from_args, main
from pii_redaction.models import PIIType


def _write_simple_docx(path: Path, text: str = "hello") -> Path:
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


class TestExitCodes:
    def test_success(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        src = _write_simple_docx(tmp_path / "in.docx", "plain text")
        out = tmp_path / "out.docx"
        code = main([str(src), "-o", str(out), "--no-ner"])
        assert code == 0
        assert out.exists()
        captured = capsys.readouterr()
        assert "total: 0" in captured.out
        assert "plain text" not in captured.out  # D8: no values in summary

    def test_missing_input(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        missing = tmp_path / "nope.docx"
        code = main([str(missing), "-o", str(tmp_path / "out.docx")])
        assert code == 1
        err = capsys.readouterr().err
        assert "not found" in err
        assert "Traceback" not in err

    def test_non_docx_input(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        bad = tmp_path / "x.docx"
        bad.write_text("not a docx", encoding="utf-8")
        code = main([str(bad), "-o", str(tmp_path / "out.docx")])
        assert code == 1
        assert "cannot open" in capsys.readouterr().err

    def test_output_equals_input(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        src = _write_simple_docx(tmp_path / "same.docx")
        code = main([str(src), "-o", str(tmp_path / "same.docx")])
        assert code == 1
        assert "refusing to overwrite" in capsys.readouterr().err

    def test_unknown_types(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        src = _write_simple_docx(tmp_path / "in.docx")
        code = main([str(src), "-o", str(tmp_path / "out.docx"), "--types", "NOT_A_TYPE"])
        assert code == 1
        assert "unknown PII type" in capsys.readouterr().err

    def test_missing_output_without_dump_or_dry_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        src = _write_simple_docx(tmp_path / "in.docx")
        code = main([str(src)])
        assert code == 1
        assert "--output is required" in capsys.readouterr().err


class TestDumpText:
    def test_prints_extraction_without_output_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        src = _write_simple_docx(tmp_path / "in.docx", "visible extraction")
        out = tmp_path / "out.docx"
        code = main([str(src), "--dump-text"])
        assert code == 0
        assert not out.exists()
        assert "visible extraction" in capsys.readouterr().out


class TestDryRun:
    def test_writes_no_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        src = _write_simple_docx(tmp_path / "in.docx", "content")
        out = tmp_path / "out.docx"
        code = main([str(src), "--dry-run", "-o", str(out), "--no-ner"])
        assert code == 0
        assert not out.exists()
        assert "total: 0" in capsys.readouterr().out

    def test_reports_counts_for_pii(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        src = _write_simple_docx(tmp_path / "in.docx", "mail a@b.co please")
        out = tmp_path / "out.docx"
        code = main([str(src), "--dry-run", "--no-ner", "--seed", "0"])
        assert code == 0
        assert not out.exists()
        captured = capsys.readouterr().out
        assert "EMAIL" in captured
        assert "a@b.co" not in captured


class TestReportOptIn:
    def test_report_only_when_requested(self, tmp_path: Path) -> None:
        src = _write_simple_docx(tmp_path / "in.docx", "a@b.co")
        out = tmp_path / "out.docx"
        report = tmp_path / "r.json"
        code = main(
            [str(src), "-o", str(out), "--no-ner", "--seed", "0", "--report", str(report)]
        )
        assert code == 0
        assert report.exists()
        body = report.read_text(encoding="utf-8")
        assert "mapping" in body
        assert "version" in body

    def test_no_report_without_flag(self, tmp_path: Path) -> None:
        src = _write_simple_docx(tmp_path / "in.docx", "a@b.co")
        out = tmp_path / "out.docx"
        code = main([str(src), "-o", str(out), "--no-ner"])
        assert code == 0
        assert list(tmp_path.glob("*.json")) == []


class TestFlagWiring:
    def test_config_from_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "in.docx",
                "-o",
                "out.docx",
                "--seed",
                "42",
                "--no-ner",
                "--types",
                "EMAIL,PHONE",
                "--redact-reference-numbers",
                "--ner-model",
                "en_core_web_md",
                "--no-verify",
            ]
        )
        config = config_from_args(args)
        assert config.seed == 42
        assert config.use_ner is False
        assert config.enabled_types == frozenset({PIIType.EMAIL, PIIType.PHONE})
        assert config.redact_reference_numbers is True
        assert config.ner_model == "en_core_web_md"
        assert config.verify_output is False


class TestHelpLaziness:
    def test_help_does_not_import_spacy(self, capsys: pytest.CaptureFixture[str]) -> None:
        sys.modules.pop("spacy", None)
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        assert "spacy" not in sys.modules
        assert "usage:" in capsys.readouterr().out.lower()
