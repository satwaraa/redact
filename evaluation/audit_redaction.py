"""Audit a redacted .docx against its source. No ground truth required.

    uv run python evaluation/audit_redaction.py data/x.docx out/x.redacted.docx

Prints a report and exits 1 if anything the probes consider PII survived, so it
can gate a batch:

    for f in inbox/*.docx; do
      uv run redact "$f" -o "out/$(basename "$f")" &&
      uv run python evaluation/audit_redaction.py "$f" "out/$(basename "$f")" || echo "FAIL $f"
    done

Two independent views, deliberately kept separate:

  applied   what the tool's own rule detectors find in the source, checked for
            survival in the output. Answers "did it replace what it found?"
  probe     regex/structure patterns written here, owned by nobody in the
            pipeline. Answers "what did it never see?": this is the one that
            catches detector gaps, so it is the number to trust.

A probe hit is a candidate, not proof: these patterns are deliberately broad and
will flag some non-PII. Read the listed values before treating them as leaks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from pii_redaction.detectors import (  # noqa: E402
    card_scheme_match,
    get_detectors,
    luhn_valid,
    ssn_structure_valid,
)
from pii_redaction.document import DocxDocument  # noqa: E402
from pii_redaction.models import RedactorConfig  # noqa: E402

# Structure-independent probes. Broad on purpose: a false alarm costs a glance,
# a missed category costs a disclosure.
_TITLE = r"[A-Z][a-zA-Z'’.\-]+(?:[ \t][A-Z][a-zA-Z'’.\-]+){1,4}"
PROBES: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "url": re.compile(r"(?:https?://|www\.)[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "card": re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
    "ipv4": re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])"),
    "phone_intl": re.compile(r"\+\d{1,3}[ \-]?\d[\d \-]{6,13}\d"),
    "labelled_person": re.compile(
        rf"(?:Contact\s+Person|Compliance\s+Officer|Reported\s+by|Attn|Name)\s*:?\s*({_TITLE})"
    ),
    "honorific_person": re.compile(rf"(?:Mr|Mrs|Ms|Dr|Shri|Smt|Prof)\.?\s+({_TITLE})"),
    # The trailing guard stops "Inc" matching inside "Income"/"Incentive",
    # which reported "Gross National Disposable Inc" as a surviving company.
    "legal_entity": re.compile(
        rf"({_TITLE}[ \t]+"
        r"(?:Private[ \t]+Limited|Limited|Ltd\.?|LLP|Inc\.?|GmbH|Pvt\.?[ \t]*Ltd\.?))"
        r"(?![A-Za-z])"
    ),
}

# Both views collapse into these plain-language buckets for the headline score.
SCORE_CATEGORIES: dict[str, str] = {
    "FULL_NAME": "Names",
    "labelled_person": "Names",
    "honorific_person": "Names",
    "EMAIL": "Email addresses",
    "email": "Email addresses",
    "PHONE": "Phone numbers",
    "phone_intl": "Phone numbers",
    "COMPANY": "Company names",
    "legal_entity": "Company names",
    "ADDRESS": "Postal addresses",
    "SSN": "SSNs",
    "ssn": "SSNs",
    "CREDIT_CARD": "Card numbers",
    "card": "Card numbers",
    "DOB": "Dates of birth",
    "IP_ADDRESS": "IP addresses",
    "ipv4": "IP addresses",
    "DOMAIN": "Websites",
    "url": "Websites",
}

# Over-redaction check: a capitalised word this common in prose is vocabulary,
# not PII, and should still be there afterwards.
_COMMON_WORD = re.compile(r"(?<![\w.@])[A-Z][a-z]{3,}(?![\w@])")
_OVER_REDACTION_MIN_COUNT = 12
_OVER_REDACTION_KEEP_RATIO = 0.25

_LABEL_TOKENS = {
    "website", "telephone", "tel", "email", "e-mail", "fax", "address", "name",
    "number", "registration", "contact", "person", "designation", "date",
}


def _clean_name(value: str) -> str:
    """Trim a captured name at the first label token."""
    out: list[str] = []
    for token in re.split(r"[\s\xa0]+", value.strip()):
        if not token or token.casefold().strip(".:") in _LABEL_TOKENS:
            break
        out.append(token)
    return " ".join(out)


def _probe_values(text: str, name: str, pattern: re.Pattern[str]) -> set[str]:
    values: set[str] = set()
    for match in pattern.finditer(text):
        value = match.group(1) if pattern.groups else match.group(0)
        if name.endswith("person"):
            value = _clean_name(value)
            if len(value.split()) < 2:
                continue
        value = value.strip()
        if len(value) >= 4:
            values.add(value)
    return values


def _package_text(path: Path) -> str:
    """Every XML part concatenated: catches field codes, headers, rels, metadata."""
    with zipfile.ZipFile(path) as archive:
        return "\n".join(
            archive.read(n).decode("utf-8", "ignore")
            for n in archive.namelist()
            if n.endswith((".xml", ".rels"))
        )


def _fmt(values: Iterable[str], limit: int) -> str:
    listed = sorted(values)
    head = ", ".join(repr(v) for v in listed[:limit])
    extra = len(listed) - limit
    return head + (f" … +{extra} more" if extra > 0 else "")


def audit(source: Path, redacted: Path, *, limit: int = 8) -> dict[str, Any]:
    src, red = DocxDocument(source), DocxDocument(redacted)
    src_text, red_text = src.extract_text(), red.extract_text()
    src_blocks, red_blocks = list(src.blocks), list(red.blocks)
    red_package = _package_text(redacted)

    report: dict[str, Any] = {
        "source": str(source),
        "redacted": str(redacted),
        "structure": {
            "blocks": [len(src_blocks), len(red_blocks)],
            "blocks_match": len(src_blocks) == len(red_blocks),
            "chars": [len(src_text), len(red_text)],
            "tables": [len(src._doc.tables), len(red._doc.tables)],  # noqa: SLF001
            "parts": [
                len(zipfile.ZipFile(source).namelist()),
                len(zipfile.ZipFile(redacted).namelist()),
            ],
        },
        "applied": {},
        "probe": {},
        "surrogate_quality": {},
        "metadata": {},
    }

    if len(src_blocks) == len(red_blocks):
        changed = sum(1 for a, b in zip(src_blocks, red_blocks, strict=False) if a.text != b.text)
        report["structure"]["blocks_changed"] = changed
        report["structure"]["churn_pct"] = round(100 * changed / max(1, len(src_blocks)), 1)

    # View 1: did the tool replace what its own rules found?
    config = RedactorConfig(use_ner=False)
    found: dict[str, set[str]] = {}
    for detector in get_detectors(config):
        for entity in detector.detect(src_text, config):
            found.setdefault(entity.pii_type.value, set()).add(entity.text)
    for pii_type, values in sorted(found.items()):
        survivors = {v for v in values if v in red_text}
        report["applied"][pii_type] = {
            "found": len(values),
            "survived": len(survivors),
            "values": sorted(survivors)[:limit],
        }

    # View 2: what did the tool never see? Checks visible text AND the package.
    for name, pattern in PROBES.items():
        values = _probe_values(src_text, name, pattern)
        if not values:
            continue
        visible = {v for v in values if v in red_text}
        buried = {v for v in values if v not in visible and v in red_package}
        report["probe"][name] = {
            "in_source": len(values),
            "survived_visible": len(visible),
            "survived_package_only": len(buried),
            "values": sorted(visible | buried)[:limit],
        }

    # Score: merge both views into plain categories and count what is gone.
    # Values are deduplicated per category so an email seen by both the rules
    # and the probe counts once.
    candidates: dict[str, set[str]] = {}
    for pii_type, values in found.items():
        candidates.setdefault(SCORE_CATEGORIES.get(pii_type, pii_type), set()).update(values)
    for name, pattern in PROBES.items():
        values = _probe_values(src_text, name, pattern)
        if values:
            candidates.setdefault(SCORE_CATEGORIES.get(name, name), set()).update(values)

    scored: dict[str, dict[str, Any]] = {}
    total = removed_total = 0
    for label, values in candidates.items():
        remaining = sorted(v for v in values if v in red_text or v in red_package)
        removed = len(values) - len(remaining)
        total += len(values)
        removed_total += removed
        scored[label] = {
            "candidates": len(values),
            "removed": removed,
            "remaining": len(remaining),
            "pct": round(100 * removed / len(values), 1) if values else None,
            "values": remaining[:limit],
        }
    report["score"] = {
        "categories": scored,
        "candidates": total,
        "removed": removed_total,
        "remaining": total - removed_total,
        "pct": round(100 * removed_total / total, 1) if total else None,
    }

    # Over-redaction: frequent capitalised words that largely vanished.
    # A country, a state or a defined term appears many times in running prose
    # and should survive redaction. If "India" occurs 97 times in the source and
    # 0 times in the output, something classified a place name as PII. Recall
    # probes cannot see this class of error, because nothing leaked.
    src_counts = Counter(_COMMON_WORD.findall(src_text))
    red_counts = Counter(_COMMON_WORD.findall(red_text))
    vanished = []
    for word, count in src_counts.most_common():
        if count < _OVER_REDACTION_MIN_COUNT:
            break
        kept = red_counts.get(word, 0)
        if kept <= count * _OVER_REDACTION_KEEP_RATIO:
            vanished.append({"word": word, "in_source": count, "in_output": kept})
    report["over_redaction"] = {
        "checked_words": sum(1 for c in src_counts.values() if c >= _OVER_REDACTION_MIN_COUNT),
        "vanished": vanished[:limit],
        "vanished_total": len(vanished),
    }

    # Surrogate quality on whatever the output now contains.
    out_cards = {re.sub(r"\D", "", c) for c in PROBES["card"].findall(red_text)}
    out_cards = {c for c in out_cards if 13 <= len(c) <= 19}
    out_ssns = set(PROBES["ssn"].findall(red_text))
    if out_cards:
        report["surrogate_quality"]["cards"] = {
            "total": len(out_cards),
            "fail_luhn": sum(1 for c in out_cards if not luhn_valid(c)),
            "bad_scheme": sum(1 for c in out_cards if not card_scheme_match(c)),
        }
    if out_ssns:
        report["surrogate_quality"]["ssns"] = {
            "total": len(out_ssns),
            "structurally_invalid": sum(1 for s in out_ssns if not ssn_structure_valid(s)),
        }

    with zipfile.ZipFile(redacted) as archive:
        if "docProps/core.xml" in archive.namelist():
            core = archive.read("docProps/core.xml").decode("utf-8", "ignore")
            report["metadata"] = {
                tag: value
                for tag, value in re.findall(
                    r"<(dc:creator|cp:lastModifiedBy)>(.*?)</\1>", core
                )
                if value.strip()
            }

    applied_leaks = sum(v["survived"] for v in report["applied"].values())
    probe_leaks = sum(
        v["survived_visible"] + v["survived_package_only"] for v in report["probe"].values()
    )
    report["totals"] = {
        "applied_leaks": applied_leaks,
        "probe_leaks": probe_leaks,
        "metadata_fields_present": len(report["metadata"]),
        "clean": applied_leaks == 0 and probe_leaks == 0 and not report["metadata"],
    }
    return report


def render(report: dict[str, Any], *, limit: int = 8) -> str:
    out: list[str] = []
    score = report["score"]
    out.append(f"SOURCE   {report['source']}")
    out.append(f"REDACTED {report['redacted']}")
    out.append("")
    out.append("=" * 66)
    if score["pct"] is None:
        out.append("REDACTION SCORE   no PII found in the source: nothing to score")
    else:
        out.append(
            f"REDACTION SCORE   {score['pct']}%   "
            f"({score['removed']} of {score['candidates']} PII values removed)"
        )
        if score["remaining"]:
            out.append(f"                  {score['remaining']} still present in the output")
    out.append("=" * 66)

    if score["categories"]:
        out.append("")
        out.append("BY TYPE            removed / found")
        for label, row in sorted(
            score["categories"].items(), key=lambda kv: (kv[1]["pct"] or 0, kv[0])
        ):
            bar = "ok  " if row["remaining"] == 0 else "MISS"
            out.append(
                f"  {label:18} {row['removed']:4} / {row['candidates']:<4}"
                f" {row['pct']:5.1f}%  {bar}"
            )
            if row["values"]:
                out.append(f"      still there: {_fmt(row['values'], limit)}")

    out.append("")
    out.append("HOW TO READ THIS")
    out.append("  The score is the share of PII this audit can see that was removed.")
    out.append("  It is not accuracy against a hand-labelled answer key: it cannot")
    out.append("  count PII that neither the tool nor these patterns recognise, so")
    out.append("  treat it as a floor. Anything under 'still there' is a candidate -")
    out.append("  the patterns are broad, so read the values before acting.")

    s = report["structure"]
    out.append("")
    out.append("STRUCTURE (did the file survive intact?)")
    flag = "MATCH" if s["blocks_match"] else "MISMATCH: splicing altered the layout"
    out.append(f"  blocks {s['blocks'][0]} -> {s['blocks'][1]}  {flag}")
    out.append(f"  chars  {s['chars'][0]} -> {s['chars'][1]}  ({s['chars'][1]-s['chars'][0]:+d})")
    out.append(
        f"  tables {s['tables'][0]} -> {s['tables'][1]}"
        f"   parts {s['parts'][0]} -> {s['parts'][1]}"
    )
    if "blocks_changed" in s:
        out.append(f"  blocks changed {s['blocks_changed']} ({s['churn_pct']}%)")

    out.append("")
    out.append("DETAIL 1: values the tool's own rules found, and whether they went")
    if not report["applied"]:
        out.append("  (no rule-based detections)")
    for pii_type, row in sorted(report["applied"].items()):
        mark = "LEAK" if row["survived"] else "ok"
        line = f"  {pii_type:12} found {row['found']:4}  survived {row['survived']:4}  {mark}"
        out.append(line)
        if row["values"]:
            out.append(f"      {_fmt(row['values'], limit)}")

    out.append("")
    out.append("DETAIL 2: independent patterns; catches what detection never saw")
    if not report["probe"]:
        out.append("  (no probe matches in source)")
    for name, row in report["probe"].items():
        total = row["survived_visible"] + row["survived_package_only"]
        mark = "LEAK" if total else "ok"
        out.append(
            f"  {name:18} in source {row['in_source']:4}  survived {total:4}"
            f" (visible {row['survived_visible']},"
            f" package-only {row['survived_package_only']})  {mark}"
        )
        if row["values"]:
            out.append(f"      {_fmt(row['values'], limit)}")

    over = report.get("over_redaction", {})
    if over.get("vanished"):
        out.append("")
        out.append("OVER-REDACTION: common words that mostly disappeared")
        for row in over["vanished"]:
            out.append(
                f"  {row['word']:20} {row['in_source']:4} in source"
                f" -> {row['in_output']:4} in output"
            )
        if over["vanished_total"] > len(over["vanished"]):
            out.append(f"  … +{over['vanished_total'] - len(over['vanished'])} more")
        out.append(
            "  Candidates only: a frequently-named real person belongs here too."
        )
        out.append(
            "  What does not is ordinary vocabulary: a country, a state, a"
            " defined term."
        )

    if report["surrogate_quality"]:
        out.append("")
        out.append("FAKE VALUE QUALITY: do the replacements look real?")
        cards = report["surrogate_quality"].get("cards")
        if cards:
            out.append(
                f"  cards {cards['total']:4}  fail Luhn {cards['fail_luhn']}"
                f"  impossible scheme {cards['bad_scheme']}"
            )
        ssns = report["surrogate_quality"].get("ssns")
        if ssns:
            out.append(
                f"  ssns  {ssns['total']:4}  structurally invalid {ssns['structurally_invalid']}"
            )

    out.append("")
    if report["metadata"]:
        out.append(f"METADATA  identity fields still set: {report['metadata']}")
    else:
        out.append("METADATA  clean")

    t = report["totals"]
    out.append("")
    verdict = "CLEAN" if t["clean"] else "LEAKS FOUND"
    out.append(
        f"VERDICT  {verdict}   applied leaks {t['applied_leaks']}, probe leaks {t['probe_leaks']}"
    )
    if not t["clean"]:
        out.append("  Probe hits are candidates: read the values above before acting.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="original .docx")
    parser.add_argument("redacted", type=Path, help="redacted .docx")
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    parser.add_argument("--limit", type=int, default=8, help="values listed per row")
    args = parser.parse_args(argv)

    for path in (args.source, args.redacted):
        if not path.is_file():
            print(f"no such file: {path}", file=sys.stderr)
            return 2

    report = audit(args.source, args.redacted, limit=args.limit)
    print(render(report, limit=args.limit))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON written to {args.json}")
    return 0 if report["totals"]["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
