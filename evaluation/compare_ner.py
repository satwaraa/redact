"""B7/B8: compare spaCy models and optional agreement on a labelled corpus.

Prints a small table of relaxed micro metrics. Does not change defaults; that
decision belongs in EVALUATION.md after reading the numbers.

Usage:
  uv run python evaluation/compare_ner.py \\
    --corpus evaluation/pages_sample_corpus.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.evaluate import load_ground_truth, run_evaluation
from pii_redaction.models import ModelUnavailableError


def _micro(result: dict[str, Any]) -> dict[str, Any]:
    return result["json"]["relaxed"]["micro"]


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def compare(
    text: str,
    truth: dict[str, Any],
    *,
    seed: int,
    models: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs: list[tuple[str, bool, bool, str]] = [
        ("rules-only", False, False, "en_core_web_sm"),
    ]
    for model in models:
        configs.append((f"{model}", True, False, model))
        configs.append((f"{model}+agreement", True, True, model))

    for label, use_ner, agreement, model in configs:
        try:
            result = run_evaluation(
                text,
                truth,
                seed=seed,
                use_ner=use_ner,
                ner_model=model,
                ner_agreement=agreement,
            )
        except ModelUnavailableError as exc:
            rows.append(
                {
                    "label": label,
                    "error": str(exc).split("\n")[0],
                }
            )
            continue
        micro = _micro(result)
        rows.append(
            {
                "label": label,
                "precision": micro["precision"],
                "recall": micro["recall"],
                "f1": micro["f1"],
                "tp": micro["tp"],
                "fp": micro["fp"],
                "fn": micro["fn"],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("evaluation/pages_sample_corpus.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--models",
        default="en_core_web_sm,en_core_web_trf",
        help="comma-separated spaCy model names to compare",
    )
    args = parser.parse_args(argv)

    if not args.corpus.exists():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return 1

    data = load_ground_truth(args.corpus)
    text = data["extracted_text"]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = compare(text, data, seed=args.seed, models=models)

    print("| config | precision | recall | F1 | TP | FP | FN |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if "error" in row:
            print(f"| {row['label']} | _unavailable_ | — | — | — | — | — |")
            print(f"  # {row['error']}", file=sys.stderr)
            continue
        print(
            f"| {row['label']} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | "
            f"{_fmt(row['f1'])} | {row['tp']} | {row['fp']} | {row['fn']} |"
        )
    print()
    print(json.dumps({"seed": args.seed, "corpus": str(args.corpus), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
