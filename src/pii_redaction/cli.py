"""Argument parsing, logging setup, exit codes, and human summary."""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path

from pii_redaction.document import DocxDocument
from pii_redaction.models import (
    DocumentError,
    LeakDetectedError,
    ModelUnavailableError,
    PIIType,
    RedactionError,
    RedactionResult,
    RedactorConfig,
)
from pii_redaction.redactor import Redactor

EXIT_OK = 0
EXIT_BAD_INPUT = 1
EXIT_MODEL_UNAVAILABLE = 2
EXIT_LEAK = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="redact",
        description="Replace PII in a .docx with realistic surrogates.",
    )
    parser.add_argument("input", type=Path, help="source .docx")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="destination .docx (required unless --dump-text or --dry-run)",
    )
    parser.add_argument(
        "--types",
        default=None,
        help="comma-separated PIIType subset (default: all)",
    )
    parser.add_argument(
        "--no-ner",
        action="store_true",
        help="rule-based detection only (no spaCy)",
    )
    parser.add_argument(
        "--ner-model",
        default=None,
        help="override the spaCy model name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="surrogate RNG seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--redact-reference-numbers",
        action="store_true",
        help="treat ticket/order/invoice-style numbers as PII",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="write entities + mapping JSON (re-identification key; opt-in only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the pipeline and print a summary; write no .docx",
    )
    parser.add_argument(
        "--dump-text",
        action="store_true",
        help="print the extracted text and exit (no redaction, no output file)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip leak verification (unsafe; debugging only)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log verbosity (repeatable)",
    )
    return parser


def _configure_logging(verbosity: int) -> None:
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _parse_types(raw: str | None) -> frozenset[PIIType]:
    if raw is None or raw.strip() == "":
        return frozenset(PIIType)
    selected: set[PIIType] = set()
    for part in raw.split(","):
        name = part.strip()
        if not name:
            continue
        try:
            selected.add(PIIType[name] if name in PIIType.__members__ else PIIType(name))
        except ValueError as exc:
            known = ", ".join(t.value for t in PIIType)
            raise DocumentError(f"unknown PII type {name!r}; expected one of: {known}") from exc
    if not selected:
            raise DocumentError("--types selected no PII types")
    return frozenset(selected)


def config_from_args(args: argparse.Namespace) -> RedactorConfig:
    return RedactorConfig(
        enabled_types=_parse_types(args.types),
        seed=args.seed,
        use_ner=not args.no_ner,
        ner_model=args.ner_model or RedactorConfig.default().ner_model,
        redact_reference_numbers=args.redact_reference_numbers,
        verify_output=not args.no_verify,
    )


def _print_summary(result: RedactionResult, output: Path | None, elapsed_s: float) -> None:
    total = sum(result.counts_by_type.values())
    if result.counts_by_type:
        for pii_type in sorted(result.counts_by_type, key=lambda t: t.value):
            print(f"  {pii_type.value}: {result.counts_by_type[pii_type]}")
    else:
        print("  (no replacements)")
    print(f"total: {total}")
    if output is not None:
        print(f"output: {output}")
    print(f"elapsed: {elapsed_s:.3f}s")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _configure_logging(args.verbose)

    try:
        input_path = args.input.resolve()
        if not input_path.exists():
            raise DocumentError(f"input not found: {input_path}")

        if args.dump_text:
            text = DocxDocument(input_path).extract_text()
            sys.stdout.write(text)
            if text and not text.endswith("\n"):
                sys.stdout.write("\n")
            return EXIT_OK

        if args.output is None and not args.dry_run:
            raise DocumentError("--output is required unless --dump-text or --dry-run")

        output_path = args.output.resolve() if args.output is not None else None
        if output_path is not None and output_path == input_path:
            raise DocumentError(f"refusing to overwrite input path: {output_path}")

        config = config_from_args(args)
        redactor = Redactor(config)
        started = time.perf_counter()
        result = redactor.redact_document(
            input_path,
            output_path,
            dry_run=args.dry_run,
        )
        elapsed = time.perf_counter() - started
        _print_summary(result, None if args.dry_run else output_path, elapsed)

        if args.report is not None:
            import json

            from pii_redaction import __version__

            payload = {
                "version": __version__,
                "seed": config.seed,
                **result.to_dict(),
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        return EXIT_OK

    except ModelUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        if args.verbose >= 2:
            traceback.print_exc()
        return EXIT_MODEL_UNAVAILABLE
    except LeakDetectedError as exc:
        print(str(exc), file=sys.stderr)
        if args.verbose >= 2:
            traceback.print_exc()
        return EXIT_LEAK
    except RedactionError as exc:
        print(str(exc), file=sys.stderr)
        if args.verbose >= 2:
            traceback.print_exc()
        return EXIT_BAD_INPUT
    except Exception as exc:
        # Unexpected failures are still bad input/runtime from the CLI's perspective
        # until later phases introduce finer codes; never silently succeed.
        print(str(exc), file=sys.stderr)
        if args.verbose >= 2:
            traceback.print_exc()
        return EXIT_BAD_INPUT


if __name__ == "__main__":
    raise SystemExit(main())
