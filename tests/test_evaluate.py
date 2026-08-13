"""test_evaluate — matcher/scoring unit tests with hand-computed expectations."""

from __future__ import annotations

from evaluation.evaluate import (
    extraction_fingerprint,
    match,
    score,
    token_accuracy,
)

from pii_redaction.models import PRIORITY_REGEX, PRIORITY_VALIDATED, PIIEntity, PIIType


def _e(
    text: str,
    start: int,
    pii_type: PIIType,
    *,
    source: str = "t",
    priority: int = PRIORITY_REGEX,
) -> PIIEntity:
    return PIIEntity(
        pii_type=pii_type,
        text=text,
        start=start,
        end=start + len(text),
        source=source,
        priority=priority,
    )


def test_exact_vs_relaxed_on_same_input() -> None:
    truth = [_e("Rohan Dey", 0, PIIType.FULL_NAME)]
    # Prediction includes honorific — relaxed hits, exact misses
    pred = [_e("Mr. Rohan Dey", 0, PIIType.FULL_NAME)]
    # Adjust: truth at offset 4 inside prediction
    truth = [_e("Rohan Dey", 4, PIIType.FULL_NAME)]
    pred = [_e("Mr. Rohan Dey", 0, PIIType.FULL_NAME)]
    exact = match(truth, pred, "exact")
    relaxed = match(truth, pred, "relaxed")
    assert exact.pairs == []
    assert len(exact.false_negatives) == 1
    assert len(exact.false_positives) == 1
    assert len(relaxed.pairs) == 1
    assert relaxed.false_negatives == []
    assert relaxed.false_positives == []


def test_one_to_one_consumption() -> None:
    truth = [_e("aaaa", 0, PIIType.EMAIL)]
    preds = [
        _e("aaaa", 0, PIIType.EMAIL, source="a"),
        _e("aaa", 0, PIIType.EMAIL, source="b"),
        _e("aa", 0, PIIType.EMAIL, source="c"),
    ]
    relaxed = match(truth, preds, "relaxed")
    assert len(relaxed.pairs) == 1
    assert len(relaxed.false_positives) == 2


def test_type_confusion_separate_from_fp_fn() -> None:
    truth = [_e("123-45-6789", 0, PIIType.SSN)]
    pred = [
        _e(
            "123-45-6789",
            0,
            PIIType.PHONE,
            priority=PRIORITY_VALIDATED,
            source="phone",
        )
    ]
    result = match(truth, pred, "exact")
    assert result.pairs == []
    assert len(result.type_confusions) == 1
    assert result.false_positives == []
    assert result.false_negatives == []
    metrics = score(result)
    assert metrics["SSN"].type_confusion == 1
    assert metrics["SSN"].fp == 0
    assert metrics["SSN"].fn == 0
    assert metrics["PHONE"].fp == 0  # consumed as confusion, not FP


def test_zero_support_metrics_are_none() -> None:
    result = match([], [], "exact")
    metrics = score(result)
    assert metrics["EMAIL"].precision is None
    assert metrics["EMAIL"].recall is None
    assert metrics["micro"].precision is None


def test_known_metrics_hand_computed() -> None:
    # Truth: two emails. Pred: one exact email + one spurious phone.
    truth = [
        _e("a@b.co", 0, PIIType.EMAIL),
        _e("c@d.co", 10, PIIType.EMAIL),
    ]
    pred = [
        _e("a@b.co", 0, PIIType.EMAIL),
        _e("9999999999", 20, PIIType.PHONE),
    ]
    result = match(truth, pred, "exact")
    metrics = score(result)
    # EMAIL: TP=1 FP=0 FN=1 → P=1.0 R=0.5
    assert metrics["EMAIL"].tp == 1
    assert metrics["EMAIL"].fn == 1
    assert metrics["EMAIL"].fp == 0
    assert metrics["EMAIL"].precision == 1.0
    assert metrics["EMAIL"].recall == 0.5
    # PHONE: FP=1
    assert metrics["PHONE"].fp == 1
    assert metrics["micro"].tp == 1
    assert metrics["micro"].fp == 1
    assert metrics["micro"].fn == 1
    assert metrics["micro"].precision == 0.5
    assert metrics["micro"].recall == 0.5


def test_token_accuracy_weak_definition() -> None:
    text = "aaaa bbbb"
    truth = [_e("aaaa", 0, PIIType.EMAIL)]
    pred = [_e("aaaa", 0, PIIType.EMAIL)]
    assert token_accuracy(text, truth, pred) == 1.0
    pred_bad = [_e("bbbb", 5, PIIType.EMAIL)]
    # 4 chars wrong on each side of disagreement overlapping differently
    acc = token_accuracy(text, truth, pred_bad)
    assert 0.0 < acc < 1.0


def test_fingerprint_stable() -> None:
    a = extraction_fingerprint("hello")
    b = extraction_fingerprint("hello")
    c = extraction_fingerprint("hello!")
    assert a == b
    assert a != c
    assert a.startswith("sha256:")


def test_pages_sample_corpus_is_consistent() -> None:
    """D1 corpus: fingerprint + offsets must stay aligned with extracted_text."""
    import json
    from pathlib import Path

    from evaluation.evaluate import assert_truth_consistent, load_ground_truth

    path = Path("evaluation/pages_sample_corpus.json")
    data = load_ground_truth(path)
    text = data["extracted_text"]
    assert_truth_consistent(text, data)
    sample = data["sample"]
    assert sample["is_sample"] is True
    assert sample["pages"] == 6
    assert text.count("PAGE ") == 6
    types = {row["pii_type"] for row in data["entities"]}
    assert types == {"FULL_NAME", "EMAIL", "PHONE", "COMPANY", "ADDRESS", "DOMAIN"}
    # Sanity: committed file still matches its own fingerprint field
    assert data["extraction_fingerprint"] == extraction_fingerprint(text)
    # Reload via Path to ensure JSON is well-formed on disk
    assert json.loads(path.read_text(encoding="utf-8"))["entities"]
