"""D3: fail if labelled-corpus metrics fall below ratcheted floors.

Usage:
  uv run python evaluation/assert_baselines.py
  uv run python evaluation/assert_baselines.py --baselines evaluation/baselines.json
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

from evaluation.evaluate import load_ground_truth, run_evaluation  # noqa: E402
from pii_redaction.models import ModelUnavailableError  # noqa: E402


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _check_floor(
    *,
    label: str,
    actual: float | None,
    minimum: float,
    failures: list[str],
) -> None:
    if actual is None:
        failures.append(f"{label}: metric undefined (got n/a), need ≥ {minimum:.3f}")
        return
    if actual + 1e-12 < minimum:
        failures.append(f"{label}: {_fmt(actual)} < min {minimum:.3f}")


def assert_suite(suite: dict[str, Any], *, seed: int, policy: str) -> list[str]:
    corpus_path = Path(suite["corpus"])
    if not corpus_path.is_absolute():
        corpus_path = _ROOT / corpus_path
    data = load_ground_truth(corpus_path)
    text = data["extracted_text"]
    use_ner = bool(suite.get("use_ner", False))
    try:
        result = run_evaluation(
            text,
            data,
            seed=seed,
            use_ner=use_ner,
            ner_model=suite.get("ner_model") or "en_core_web_sm",
            ner_agreement=bool(suite.get("ner_agreement", False)),
        )
    except ModelUnavailableError as exc:
        return [f"{suite['id']}: model unavailable: {exc}"]

    scores = result["json"][policy]
    micro = scores["micro"]
    failures: list[str] = []
    suite_id = suite["id"]
    _check_floor(
        label=f"{suite_id} micro precision",
        actual=micro.get("precision"),
        minimum=float(suite["min_micro_precision"]),
        failures=failures,
    )
    _check_floor(
        label=f"{suite_id} micro recall",
        actual=micro.get("recall"),
        minimum=float(suite["min_micro_recall"]),
        failures=failures,
    )
    _check_floor(
        label=f"{suite_id} micro F1",
        actual=micro.get("f1"),
        minimum=float(suite["min_micro_f1"]),
        failures=failures,
    )

    for pii_type, floors in (suite.get("per_type") or {}).items():
        metrics = scores.get(pii_type) or {}
        if "min_precision" in floors:
            _check_floor(
                label=f"{suite_id} {pii_type} precision",
                actual=metrics.get("precision"),
                minimum=float(floors["min_precision"]),
                failures=failures,
            )
        if "min_recall" in floors:
            _check_floor(
                label=f"{suite_id} {pii_type} recall",
                actual=metrics.get("recall"),
                minimum=float(floors["min_recall"]),
                failures=failures,
            )

    print(
        f"{suite_id}: P={_fmt(micro.get('precision'))} "
        f"R={_fmt(micro.get('recall'))} F1={_fmt(micro.get('f1'))} "
        f"TP={micro.get('tp')} FP={micro.get('fp')} FN={micro.get('fn')}"
    )
    return failures


def load_baselines(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "suites" not in raw:
        raise ValueError(f"baselines missing suites: {path}")
    return raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baselines",
        type=Path,
        default=_ROOT / "evaluation" / "baselines.json",
    )
    args = parser.parse_args(argv)
    if not args.baselines.exists():
        print(f"baselines not found: {args.baselines}", file=sys.stderr)
        return 1

    cfg = load_baselines(args.baselines)
    seed = int(cfg.get("seed", 0))
    policy = str(cfg.get("policy", "relaxed"))
    all_failures: list[str] = []
    for suite in cfg["suites"]:
        all_failures.extend(assert_suite(suite, seed=seed, policy=policy))

    if all_failures:
        print("baseline regressions:", file=sys.stderr)
        for line in all_failures:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print("all baseline floors met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
