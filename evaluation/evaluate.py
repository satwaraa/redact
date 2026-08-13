"""Score predictions against hand-labelled ground truth (deliverable #4)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pii_redaction
from pii_redaction.document import DocxDocument
from pii_redaction.models import (
    PRIORITY_REGEX,
    PIIEntity,
    PIIType,
    RedactorConfig,
)
from pii_redaction.redactor import Redactor
from pii_redaction.resolution import resolve

MatchPolicy = Literal["exact", "relaxed"]

EXTRACTION_VERSION = "1"


@dataclass(frozen=True, slots=True)
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    type_confusion: int = 0

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return None if denom == 0 else self.tp / denom

    @property
    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return None if denom == 0 else self.tp / denom

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "type_confusion": self.type_confusion,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass
class MatchResult:
    policy: MatchPolicy
    pairs: list[tuple[PIIEntity, PIIEntity]] = field(default_factory=list)
    type_confusions: list[tuple[PIIEntity, PIIEntity]] = field(default_factory=list)
    false_positives: list[PIIEntity] = field(default_factory=list)
    false_negatives: list[PIIEntity] = field(default_factory=list)


def extraction_fingerprint(text: str, *, version: str = EXTRACTION_VERSION) -> str:
    digest = hashlib.sha256(f"{version}\n{text}".encode()).hexdigest()
    return f"sha256:{digest}"


def load_ground_truth(path: Path | str) -> dict[str, Any]:
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "entities" not in raw:
        raise ValueError(f"ground truth missing entities: {path}")
    return raw


def entities_from_truth(data: dict[str, Any]) -> list[PIIEntity]:
    out: list[PIIEntity] = []
    for row in data["entities"]:
        text = row["text"]
        start = int(row["start"])
        end = int(row["end"])
        out.append(
            PIIEntity(
                pii_type=PIIType(row["pii_type"]),
                text=text,
                start=start,
                end=end,
                source=f"truth:{row.get('id', 'unknown')}",
                priority=PRIORITY_REGEX,
            )
        )
    return out


def assert_truth_consistent(text: str, data: dict[str, Any]) -> None:
    expected = data.get("extraction_fingerprint")
    actual = extraction_fingerprint(
        text, version=str(data.get("extraction_version", EXTRACTION_VERSION))
    )
    if expected and expected != actual:
        raise SystemExit(
            "extraction fingerprint mismatch: ground truth was labelled against a "
            f"different string\n  truth: {expected}\n  now:   {actual}\n"
            "Re-extract with DocxDocument.extract_text() and relabel offsets."
        )
    for row in data["entities"]:
        start, end, span = int(row["start"]), int(row["end"]), row["text"]
        if text[start:end] != span:
            raise SystemExit(
                f"truth entity {row.get('id')!r} offset mismatch: "
                f"extracted[{start}:{end}] != labelled text"
            )


def _overlap_len(a: PIIEntity, b: PIIEntity) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def match(
    truth: Sequence[PIIEntity],
    predicted: Sequence[PIIEntity],
    policy: MatchPolicy,
) -> MatchResult:
    """Greedy one-to-one matching: largest overlap first; each span used once."""
    result = MatchResult(policy=policy)
    candidates: list[tuple[int, int, int, int]] = []
    # (overlap, -type_mismatch, truth_idx, pred_idx)
    for ti, t in enumerate(truth):
        for pi, p in enumerate(predicted):
            if policy == "exact":
                if t.start == p.start and t.end == p.end:
                    overlap = len(t)
                else:
                    continue
            else:
                overlap = _overlap_len(t, p)
                if overlap == 0:
                    continue
            type_ok = 1 if t.pii_type is p.pii_type else 0
            candidates.append((overlap, type_ok, ti, pi))

    candidates.sort(key=lambda c: (-c[0], -c[1], c[2], c[3]))
    used_t: set[int] = set()
    used_p: set[int] = set()
    for _overlap, type_ok, ti, pi in candidates:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        if type_ok:
            result.pairs.append((truth[ti], predicted[pi]))
        else:
            result.type_confusions.append((truth[ti], predicted[pi]))

    for ti, t in enumerate(truth):
        if ti not in used_t:
            result.false_negatives.append(t)
    for pi, p in enumerate(predicted):
        if pi not in used_p:
            result.false_positives.append(p)
    return result


def score(match_result: MatchResult) -> dict[str, Metrics]:
    """Per-type metrics; type confusions counted separately from FP/FN."""
    by_type: dict[PIIType, Metrics] = {t: Metrics() for t in PIIType}
    for truth_e, _pred in match_result.pairs:
        m = by_type[truth_e.pii_type]
        by_type[truth_e.pii_type] = Metrics(
            tp=m.tp + 1, fp=m.fp, fn=m.fn, type_confusion=m.type_confusion
        )
    for fp in match_result.false_positives:
        m = by_type[fp.pii_type]
        by_type[fp.pii_type] = Metrics(
            tp=m.tp, fp=m.fp + 1, fn=m.fn, type_confusion=m.type_confusion
        )
    for fn in match_result.false_negatives:
        m = by_type[fn.pii_type]
        by_type[fn.pii_type] = Metrics(
            tp=m.tp, fp=m.fp, fn=m.fn + 1, type_confusion=m.type_confusion
        )
    for truth_e, pred in match_result.type_confusions:
        # Charge confusion on the truth type (and note pred type in report lists)
        m = by_type[truth_e.pii_type]
        by_type[truth_e.pii_type] = Metrics(
            tp=m.tp, fp=m.fp, fn=m.fn, type_confusion=m.type_confusion + 1
        )
        _ = pred

    per_type = {t.value: by_type[t] for t in PIIType}
    micro = Metrics(
        tp=sum(m.tp for m in by_type.values()),
        fp=sum(m.fp for m in by_type.values()),
        fn=sum(m.fn for m in by_type.values()),
        type_confusion=sum(m.type_confusion for m in by_type.values()),
    )
    # Macro: mean over types that have support (tp+fp+fn+confusion > 0)
    supported = [
        m
        for m in by_type.values()
        if (m.tp + m.fp + m.fn + m.type_confusion) > 0
    ]
    if supported:
        macro = Metrics(
            tp=sum(m.tp for m in supported),
            fp=sum(m.fp for m in supported),
            fn=sum(m.fn for m in supported),
            type_confusion=sum(m.type_confusion for m in supported),
        )
        # Store macro as averaged rates via a synthetic Metrics using float fields —
        # report layer will average precision/recall explicitly.
        per_type["micro"] = micro
        per_type["macro"] = macro
    else:
        per_type["micro"] = micro
        per_type["macro"] = Metrics()
    return per_type


def token_accuracy(text: str, truth: Sequence[PIIEntity], predicted: Sequence[PIIEntity]) -> float:
    """Weak token-level accuracy: fraction of chars with correct PII/non-PII label.

    True negatives are unbounded for span extraction; this char-level proxy is
    defined only so the assignment's "accuracy" request has an explicit, weak
    answer rather than a silent omission.
    """
    if not text:
        return 1.0
    truth_mask = [False] * len(text)
    pred_mask = [False] * len(text)
    for e in truth:
        for i in range(e.start, min(e.end, len(text))):
            truth_mask[i] = True
    for e in predicted:
        for i in range(e.start, min(e.end, len(text))):
            pred_mask[i] = True
    correct = sum(1 for a, b in zip(truth_mask, pred_mask, strict=True) if a == b)
    return correct / len(text)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def render_markdown(
    scores_exact: dict[str, Metrics],
    scores_relaxed: dict[str, Metrics],
    *,
    accuracy: float,
    sample: dict[str, Any],
    seed: int,
    version: str,
    fingerprint: str,
) -> str:
    lines: list[str] = []
    lines.append("# Evaluation report")
    lines.append("")
    lines.append(f"- generated: `{datetime.now(UTC).isoformat()}`")
    lines.append(f"- pii_redaction: `{version}`")
    lines.append(f"- seed: `{seed}`")
    lines.append(f"- extraction_fingerprint: `{fingerprint}`")
    if sample.get("is_sample"):
        lines.append(
            f"- sample: {sample.get('description', 'representative sample')} "
            f"({sample.get('rationale', '')})"
        )
    else:
        lines.append("- sample: full labelled set")
    lines.append("")
    lines.append(
        "Matching: **exact** = same offsets + type; **relaxed** = any overlap + "
        "same type. Greedy one-to-one. Type confusions are separate from FP/FN."
    )
    lines.append("")
    lines.append(
        f"Token-level accuracy (weak): **{accuracy:.3f}** — fraction of characters "
        "whose PII/non-PII status matches. This is the weakest of the four numbers: "
        "true negatives dominate and span boundary errors barely move it."
    )
    lines.append("")
    lines.append("## Micro / macro (relaxed)")
    lines.append("")
    lines.append("| aggregate | precision | recall | F1 | TP | FP | FN | type-conf |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in ("micro", "macro"):
        m = scores_relaxed[key]
        if key == "macro":
            # Average rates over supported types
            supported = []
            for t in PIIType:
                m = scores_relaxed[t.value]
                if (m.tp + m.fp + m.fn) > 0:
                    supported.append(m)
            if not supported:
                lines.append(f"| {key} | — | — | — | 0 | 0 | 0 | 0 |")
                continue
            precs = [x.precision for x in supported if x.precision is not None]
            recs = [x.recall for x in supported if x.recall is not None]
            f1s = [x.f1 for x in supported if x.f1 is not None]
            p = sum(precs) / len(precs) if precs else None
            r = sum(recs) / len(recs) if recs else None
            f = sum(f1s) / len(f1s) if f1s else None
            lines.append(
                f"| {key} | {_fmt(p)} | {_fmt(r)} | {_fmt(f)} | "
                f"{m.tp} | {m.fp} | {m.fn} | {m.type_confusion} |"
            )
        else:
            lines.append(
                f"| {key} | {_fmt(m.precision)} | {_fmt(m.recall)} | {_fmt(m.f1)} | "
                f"{m.tp} | {m.fp} | {m.fn} | {m.type_confusion} |"
            )
    lines.append("")
    lines.append("## Per-type (relaxed)")
    lines.append("")
    lines.append("| type | precision | recall | F1 | TP | FP | FN | type-conf |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for t in PIIType:
        m = scores_relaxed[t.value]
        support = m.tp + m.fp + m.fn + m.type_confusion
        if support == 0:
            lines.append(f"| {t.value} | — | — | — | 0 | 0 | 0 | 0 |")
        else:
            lines.append(
                f"| {t.value} | {_fmt(m.precision)} | {_fmt(m.recall)} | {_fmt(m.f1)} | "
                f"{m.tp} | {m.fp} | {m.fn} | {m.type_confusion} |"
            )
    lines.append("")
    lines.append("## Exact-match micro")
    lines.append("")
    em = scores_exact["micro"]
    lines.append(
        f"precision={_fmt(em.precision)} recall={_fmt(em.recall)} "
        f"F1={_fmt(em.f1)} (TP={em.tp} FP={em.fp} FN={em.fn})"
    )
    lines.append("")
    return "\n".join(lines)


def _context(text: str, entity: PIIEntity, window: int = 40) -> str:
    lo = max(0, entity.start - window)
    hi = min(len(text), entity.end + window)
    return text[lo:hi].replace("\n", " ")


def error_analysis_md(
    text: str,
    relaxed: MatchResult,
) -> str:
    lines = ["## Error analysis (relaxed)", ""]
    lines.append("### False negatives (missed PII)")
    lines.append("")
    if not relaxed.false_negatives:
        lines.append("_None._")
    for e in relaxed.false_negatives:
        lines.append(
            f"- `{e.pii_type.value}` @{e.start}:{e.end} `{e.text!r}` … "
            f"context: …{_context(text, e)}…"
        )
    lines.append("")
    lines.append("### False positives (over-redaction)")
    lines.append("")
    if not relaxed.false_positives:
        lines.append("_None._")
    for e in relaxed.false_positives:
        lines.append(
            f"- `{e.pii_type.value}` @{e.start}:{e.end} `{e.text!r}` … "
            f"context: …{_context(text, e)}…"
        )
    lines.append("")
    lines.append("### Type confusions")
    lines.append("")
    if not relaxed.type_confusions:
        lines.append("_None._")
    for t, p in relaxed.type_confusions:
        lines.append(
            f"- truth `{t.pii_type.value}` vs pred `{p.pii_type.value}` "
            f"@{t.start}:{t.end} `{t.text!r}`"
        )
    lines.append("")
    return "\n".join(lines)


def predict_entities(text: str, *, seed: int, use_ner: bool) -> list[PIIEntity]:
    config = RedactorConfig(seed=seed, use_ner=use_ner, verify_output=False)
    redactor = Redactor(config)
    detected = redactor.detect(text)
    return resolve(detected)


def run_evaluation(
    text: str,
    truth_data: dict[str, Any],
    *,
    seed: int,
    use_ner: bool,
) -> dict[str, Any]:
    assert_truth_consistent(text, truth_data)
    truth = entities_from_truth(truth_data)
    predicted = predict_entities(text, seed=seed, use_ner=use_ner)
    exact = match(truth, predicted, "exact")
    relaxed = match(truth, predicted, "relaxed")
    scores_exact = score(exact)
    scores_relaxed = score(relaxed)
    acc = token_accuracy(text, truth, predicted)
    fingerprint = extraction_fingerprint(
        text, version=str(truth_data.get("extraction_version", EXTRACTION_VERSION))
    )
    sample = truth_data.get("sample", {})
    md = render_markdown(
        scores_exact,
        scores_relaxed,
        accuracy=acc,
        sample=sample,
        seed=seed,
        version=pii_redaction.__version__,
        fingerprint=fingerprint,
    )
    md += "\n" + error_analysis_md(text, relaxed)
    payload = {
        "version": pii_redaction.__version__,
        "seed": seed,
        "use_ner": use_ner,
        "extraction_fingerprint": fingerprint,
        "sample": sample,
        "accuracy_token": acc,
        "exact": {k: v.as_dict() for k, v in scores_exact.items()},
        "relaxed": {k: v.as_dict() for k, v in scores_relaxed.items()},
        "counts": {
            "truth": len(truth),
            "predicted": len(predicted),
            "exact_tp": scores_exact["micro"].tp,
            "relaxed_tp": scores_relaxed["micro"].tp,
            "type_confusions": scores_relaxed["micro"].type_confusion,
        },
    }
    return {"markdown": md, "json": payload}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate PII detection against ground truth")
    p.add_argument(
        "--corpus",
        type=Path,
        default=Path("evaluation/sample_corpus.json"),
        help=(
            "labelled corpus with embedded text (default: small synthetic; "
            "use evaluation/pages_sample_corpus.json for the D1 6-page sample)"
        ),
    )
    p.add_argument("--input", type=Path, default=None, help="optional .docx to evaluate")
    p.add_argument(
        "--truth",
        type=Path,
        default=None,
        help="ground truth JSON (required with --input)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-ner", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/report.md"),
        help="markdown report path (gitignored by default)",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=Path("evaluation/report.json"),
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.input is not None:
        if args.truth is None:
            print("--truth is required when --input is set", file=sys.stderr)
            return 1
        if not args.input.exists():
            print(f"input not found: {args.input}", file=sys.stderr)
            return 1
        if not args.truth.exists():
            print(f"truth not found: {args.truth}", file=sys.stderr)
            return 1
        text = DocxDocument(args.input).extract_text()
        truth_data = load_ground_truth(args.truth)
    else:
        if not args.corpus.exists():
            print(f"corpus not found: {args.corpus}", file=sys.stderr)
            return 1
        corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
        text = corpus["extracted_text"]
        truth_data = corpus

    result = run_evaluation(
        text,
        truth_data,
        seed=args.seed,
        use_ner=not args.no_ner,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result["markdown"], encoding="utf-8")
    args.json_out.write_text(
        json.dumps(result["json"], indent=2) + "\n", encoding="utf-8"
    )
    # Print the headline table section to stdout for copy/paste into README
    print(result["markdown"].split("## Error analysis")[0].rstrip())
    print(f"\nwrote {args.out}")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
