# Evaluation report

How I measured this PII redaction tool, what it scores, and where it fails.

## Summary

| | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| Rules only, all nine PII types | 0.974 | 1.000 | 0.917 | 0.957 |
| Rules + NER, prospectus-style pages | 0.983 | 0.979 | 0.959 | 0.969 |

On the actual assignment document, an independent whole-file check finds
**159 of 160 PII values removed (99.4%)**, and the single remaining flag is a
false alarm rather than a leak.

Both runs use seed 0 and are reproducible with one command.

## What I measured against

Redaction has no natural answer key, so I hand-labelled two corpora.

**`sample_corpus.json`** is a short document containing all nine required PII
types. The prospectus attachment has no SSNs, credit cards, IP addresses or
dates of birth, so without this corpus four of the nine types would go
completely unmeasured. Rules only, so it runs without downloading a model.

**`pages_sample_corpus.json`** is six pages written in the style of the
prospectus: cover page, registrar and banker contact blocks, a board-of-directors
table, and a signature page. It covers the types that actually appear in the real
document (names, companies, emails, phones, addresses, websites), including
values that repeat across pages.

Both corpora are **synthetic**, and that is a deliberate trade. Labelling the
real prospectus would mean committing a JSON file full of real names, emails and
phone numbers to the repository, the exact disclosure this tool exists to
prevent. The cost is that these numbers measure the detectors rather than this
specific document, which is why the whole-file check further down exists.

Labelling rules I followed:

- Every occurrence is labelled, including repeats. Recall is per instance, not
  per unique value.
- Labels were written before looking at detector output, so the tool is not
  being scored against itself.
- Boilerplate such as "Equity Shares", "the Offer" and "Board of Directors" is left
  unlabelled on purpose, so over-redaction shows up as a false positive.
- The issuing company's own name counts as a company. Ordinary document dates
  are not dates of birth.

Each corpus stores a hash of the text it was labelled against. If extraction ever
changes, scoring refuses to run rather than reporting confident nonsense against
shifted offsets.

## How matching works

A prediction and a label match when they refer to the same value, but "same" has
two reasonable definitions that answer different questions, so I report both.

**Relaxed**: the spans overlap and the type agrees. This asks *did you find the
PII?*

**Exact**: identical start and end offsets, same type. This asks *did you draw
the boundary exactly where I did?*

Matching is greedy and one-to-one: candidate pairs are sorted by overlap size and
each label and each prediction is consumed once. Without that, three overlapping
predictions could all "match" a single label and inflate precision.

Type confusions (right span, wrong type) are counted separately rather than
folded into the false positive and false negative totals. "Found it but called it
a phone instead of an SSN" is a different engineering problem from "never saw
it", and merging them hides which one you have.

**On "accuracy":** it is ill-defined for span extraction, because true negatives
are unbounded, since every character correctly left alone counts as one. I report it
at token level: the share of tokens whose redacted / not-redacted status is
correct. It reads high by construction, since most of a document is not PII, so
it is the least informative of the four numbers here. Precision and recall carry
the real signal.

## Results

### Rules only, all nine types

Relaxed: precision 1.000, recall 0.917, F1 0.957 (11 true positives, 0 false
positives, 1 false negative). Token accuracy 0.974.

Exact: precision 0.909, recall 0.833, F1 0.870. The drop is one span-boundary
disagreement, not a missed value. The tool takes a slightly wider slice than my
label did.

| type | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| FULL_NAME | n/a | 0.000 | n/a | 0 | 0 | 1 |
| EMAIL | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 |
| PHONE | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 |
| COMPANY | 1.000 | 1.000 | 1.000 | 1 | 0 | 0 |
| ADDRESS | 1.000 | 1.000 | 1.000 | 1 | 0 | 0 |
| SSN | 1.000 | 1.000 | 1.000 | 1 | 0 | 0 |
| CREDIT_CARD | 1.000 | 1.000 | 1.000 | 1 | 0 | 0 |
| DOB | 1.000 | 1.000 | 1.000 | 1 | 0 | 0 |
| IP_ADDRESS | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 |

### Rules + NER, prospectus-style pages

Relaxed: precision 0.979, recall 0.959, F1 0.969 (47 true positives, 1 false
positive, 2 false negatives). Token accuracy 0.983.

Exact: precision 0.896, recall 0.878, F1 0.887, again a boundary gap rather than
missed values.

| type | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| FULL_NAME | 0.933 | 0.875 | 0.903 | 14 | 1 | 2 |
| EMAIL | 1.000 | 1.000 | 1.000 | 9 | 0 | 0 |
| PHONE | 1.000 | 1.000 | 1.000 | 7 | 0 | 0 |
| COMPANY | 1.000 | 1.000 | 1.000 | 10 | 0 | 0 |
| ADDRESS | 1.000 | 1.000 | 1.000 | 6 | 0 | 0 |
| DOMAIN | 1.000 | 1.000 | 1.000 | 1 | 0 | 0 |

## Errors

**The one rules-only miss** is a person's name. There is no regex for a bare
name, so that category needs the model, which is what the second configuration
measures.

**Company names are caught without the model.** A legal suffix ("Pvt Ltd",
"Limited", "LLP") is unambiguous evidence, and a rule reads it more reliably
than spaCy does. On the real prospectus spaCy returned no span at all for
"Bajaj Finance Limited" or "Precision Wires India Limited"; adding that rule took
company detection from 34 of 44 to 54 of 54 on the real document.

**Names remain the weakest type**, at 0.933 precision and 0.875 recall. Two
misses and one false positive, all in ordinary prose rather than in the
structured contact blocks where a label such as "Contact Person:" gives the rule
layer something to anchor on.

**No type confusions** occurred in either run.

## Whole-document check, without an answer key

Because the corpora are synthetic, they cannot tell you what happened to the real
attachment. `audit_redaction.py` closes that gap. It re-scans the redacted output
using patterns written independently of the detection pipeline, inspects the
entire `.docx` package rather than just the visible text, and needs no labels:

```bash
uv run python evaluation/audit_redaction.py data/prospectus.docx out/redacted.docx
```

On the shipped output it reports **99.4%, 159 of 160 values removed**, with
companies, emails, phone numbers, addresses and websites all at 100%. The single flag is
`Anchor Investor Pay`, a fragment of the phrase "Anchor Investor Pay-in Date"
rather than a person's name.

Reading the whole package matters. Twenty-seven of the prospectus's email
addresses live inside Word HYPERLINK field codes, which `Paragraph.text` cannot
see. A leak check reading through that same API would have declared them clean.

The audit also reports **over-redaction**: common words that largely disappeared
between input and output. No recall metric can detect this, because nothing
leaks when a place name is destroyed. That check is what caught the word "India"
being rewritten as a postal address 150 times.

## What these numbers do not cover

- **Corpus size.** Twelve and forty-nine labelled instances. Enough to catch
  systematic failures, not enough for tight confidence intervals.
- **Synthetic text.** Real documents are messier than anything I wrote.
- **One document type.** Tested separately on a support-ticket log, company
  detection fell to 1 of 4, because it leans on legal suffixes and contact-block
  structure that a ticket log does not have.
- **Scanned content.** No OCR, so text inside images is invisible.

## Reproducing

```bash
uv sync

# rules only, all nine types
uv run python evaluation/evaluate.py --no-ner --seed 0

# rules + NER, prospectus-style pages
uv run python evaluation/evaluate.py \
  --corpus evaluation/pages_sample_corpus.json --seed 0

# whole-document check on the real attachment
uv run python evaluation/audit_redaction.py data/prospectus.docx out/redacted.docx
```

Every number above comes from seed 0. `evaluation/baselines.json` holds these
metrics as CI floors, so a change that trades precision for recall fails the
build instead of quietly shipping.
