"""test_baselines — D3 CI ratchet over labelled corpora."""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.assert_baselines import assert_suite, load_baselines

from pii_redaction.models import ModelUnavailableError
from pii_redaction.ner import _load_nlp, clear_nlp_cache

_ROOT = Path(__file__).resolve().parents[1]
_BASELINES = _ROOT / "evaluation" / "baselines.json"


def _spacy_model_available(name: str) -> bool:
    clear_nlp_cache()
    try:
        _load_nlp(name)
    except ModelUnavailableError:
        return False
    return True


def test_baseline_file_well_formed() -> None:
    cfg = load_baselines(_BASELINES)
    assert cfg["policy"] == "relaxed"
    assert cfg["suites"]
    ids = {suite["id"] for suite in cfg["suites"]}
    assert "sample_corpus_rules" in ids
    assert "pages_sample_ner_sm" in ids


def test_sample_corpus_rules_baseline() -> None:
    cfg = load_baselines(_BASELINES)
    suite = next(s for s in cfg["suites"] if s["id"] == "sample_corpus_rules")
    failures = assert_suite(suite, seed=int(cfg["seed"]), policy=str(cfg["policy"]))
    assert failures == []


def test_pages_sample_ner_sm_baseline() -> None:
    if not _spacy_model_available("en_core_web_sm"):
        pytest.skip("en_core_web_sm not installed")
    cfg = load_baselines(_BASELINES)
    suite = next(s for s in cfg["suites"] if s["id"] == "pages_sample_ner_sm")
    failures = assert_suite(suite, seed=int(cfg["seed"]), policy=str(cfg["policy"]))
    assert failures == []
