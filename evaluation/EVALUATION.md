# Evaluation

I ran the harness once with a fixed seed and pasted the numbers here so you
don't need the gitignored report files. To regenerate:

```bash
uv run python evaluation/evaluate.py --no-ner --seed 0
```

That dumps `evaluation/report.md` and `report.json` (both gitignored). Numbers
below are from `pii_redaction` 0.1.0 with that command.

## What I'm measuring

This is span extraction, not "is this doc sensitive yes/no". The detector runs
the same detect → resolve path as the actual redactor, then I compare predicted
spans against labelled spans in `evaluation/sample_corpus.json`.

The corpus is a short synthetic paragraph with all nine `PIIType`s labelled. I
went synthetic because the prospectus attachment doesn't cleanly cover SSN /
card / IP / DOB, and I didn't want real PII in the repo. Offsets are against the
embedded `extracted_text`. There's also an extraction fingerprint in the truth
file — if someone changes the extractor and shifts every character, the harness
won't quietly score it as correct.

For a real `.docx` you'd label against `DocxDocument.extract_text()`, keep the
fingerprint, and pass `--input` / `--truth`.

## How matching works

`evaluate.py` does greedy one-to-one matching between predicted and truth spans:

- **exact** — same start/end offsets and same type
- **relaxed** — any character overlap and same type

Biggest overlap wins. Each span gets used at most once. If something overlaps
but has the wrong type, that's a type confusion (separate from FP/FN) — you
found it, just called it the wrong thing.

**Precision** = of the spans I predicted, how many were actually right.
**Recall** = of the labelled spans, how many I found.
**F1** = harmonic mean of those two.

Micro pools TP/FP/FN across types first, then computes P/R/F1. Macro averages
per-type scores over types that have any support. Types with zero predictions
get undefined precision (shown as `—`); they still hurt recall when missed.

Accuracy is a bit awkward for spans. The harness reports a weak character-level
one: fraction of characters where "is this PII?" agrees between truth and
prediction. Most characters are non-PII, so true negatives inflate it. Treat it
as the softest number.

## Setup for this run

- Corpus: `evaluation/sample_corpus.json`
- Command: `uv run python evaluation/evaluate.py --no-ner --seed 0`
- NER off, seed `0`
- Fingerprint: `sha256:bf8e51a7550d48387161a26058dec9dc4dfd1552801851d717d4feb708742e1a`

I used `--no-ner` on purpose so reviewers can re-run without downloading spaCy.
The misses below (name / company) are exactly what regex-only mode shows.

## Numbers (relaxed)

| Metric | Value |
|---|---:|
| Accuracy (char-level) | 0.922 |
| Precision (micro) | 1.000 |
| Recall (micro) | 0.833 |
| F1 (micro) | 0.909 |
| TP / FP / FN | 10 / 0 / 2 |

Macro (relaxed): precision 1.000, recall 0.778, F1 1.000 — F1 only averaged over
types where F1 is defined.

Exact match landed the same as relaxed on this sample: P=1.000, R=0.833,
F1=0.909 (TP=10, FP=0, FN=2). Offsets lined up; nothing was "almost" overlapping.

### Per type

| type | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| FULL_NAME | — | 0.000 | — | 0 | 0 | 1 |
| EMAIL | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 |
| PHONE | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 |
| COMPANY | — | 0.000 | — | 0 | 0 | 1 |
| ADDRESS | 1.000 | 1.000 | 1.000 | 1 | 0 | 0 |
| SSN | 1.000 | 1.000 | 1.000 | 1 | 0 | 0 |
| CREDIT_CARD | 1.000 | 1.000 | 1.000 | 1 | 0 | 0 |
| DOB | 1.000 | 1.000 | 1.000 | 1 | 0 | 0 |
| IP_ADDRESS | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 |

## What went wrong

Two false negatives, both expected without NER:

1. `FULL_NAME` — `Rashi Patil`. No regex for person names; you'd need NER.
2. `COMPANY` — `Acme Technologies Pvt Ltd`. Same deal.

No false positives on this corpus with `--no-ner`. No type confusions either.

So the recall gap isn't a flaky offset bug — it's just that names and companies
aren't something the regex detectors try to catch. Turning NER on can pick those
up, but then you have a model download and whatever label noise spaCy brings.
I left the snapshot on `--no-ner` so the numbers stay easy to re-check.
