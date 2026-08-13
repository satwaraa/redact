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

## D1 — pages sample (precision-tuning corpus)

`evaluation/pages_sample_corpus.json` is the labelled sample for Workstream B.
It is a **synthetic 6-page** prospectus-style excerpt covering the types that
actually appear in `data/prospectus.docx`: `FULL_NAME`, `EMAIL`, `PHONE`,
`COMPANY`, `ADDRESS`. Types the prospectus lacks (SSN / card / IP / DOB) stay
on the small `sample_corpus.json` only.

Sampling decisions recorded in the JSON `sample` + `span_conventions` blocks:

- Every occurrence is labelled, including repeats across pages.
- Labels were written independently of detector output.
- Issuer name is labelled `COMPANY`; document dates are not `DOB`.
- Boilerplate (`Equity Shares`, `the Offer`, `Board of Directors`, …) is
  deliberately left unlabelled so B2/B3 false-positive lists have signal.
- Real prospectus PII remains gitignored (`data/`, `evaluation/ground_truth.json`).

Re-run:

```bash
uv run python evaluation/evaluate.py \
  --corpus evaluation/pages_sample_corpus.json --no-ner --seed 0
```

## B7 / B8 — model upgrade and agreement (measured)

Compared on `evaluation/pages_sample_corpus.json`, seed `0`, relaxed micro
match, with the B1–B6 filter chain already in place.

| config | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| rules-only | 1.000 | 0.417 | 0.588 | 20 | 0 | 28 |
| `en_core_web_sm` | 0.950 | 0.826 | 0.884 | 38 | 2 | 8 |
| `en_core_web_sm` + agreement | 1.000 | 0.809 | 0.894 | 38 | 0 | 9 |
| `en_core_web_trf` | unavailable here | — | — | — | — | — |

**B7.** `en_core_web_trf` could not be installed in this environment (pulls
`torch` + NVIDIA wheels; download failed with disk quota). Default stays
`en_core_web_sm`. Re-measure when the model is available:

```bash
python -m spacy download en_core_web_trf
PYTHONPATH=. uv run python evaluation/compare_ner.py \
  --corpus evaluation/pages_sample_corpus.json
```

**B8.** Two-signal agreement (`--ner-agreement`): FULL_NAME needs ≥2 Title-Case
tokens; COMPANY needs a legal suffix. On this sample it clears the last two
false positives and lifts F1 slightly (0.884 → 0.894) while cutting recall by
~0.017. That is a precision trade, not a free upgrade — left **opt-in**, default
off.

Defaults after B7/B8: `ner_model=en_core_web_sm`, `ner_agreement=False`.

## D3 — CI metric ratchet

`evaluation/baselines.json` holds floor values for:

1. `sample_corpus_rules` — small synthetic corpus, `--no-ner` (always cheap)
2. `pages_sample_ner_sm` — D1 pages sample with default `en_core_web_sm`

CI downloads `en_core_web_sm` and runs:

```bash
uv run python evaluation/assert_baselines.py
```

Floors sit slightly under the measured B7/B8 numbers so ordinary float noise
does not flake, but a change that buys recall by wrecking precision (or drops
micro recall below ~0.80 on the pages sample) fails the job. Raise the floors
when a real improvement lands; do not lower them to green-wash a regression.
