# redact

Reads a `.docx`, finds PII, and writes another `.docx` where those values are
replaced with realistic fakes — same shape, so Luhn-valid cards stay Luhn-valid
and `+91` numbers stay `+91`. Not blacked out, not deleted.

```
- Contact Person: Hitesh Ramani              - E-mail: ksh.ipo@nuvama.com
+ Contact Person: Kashish Yogi               + E-mail: sudiksha@kantmolestiae.com

- Telephone: +91 81081 14949                 - Website: www.kshinternational.com
+ Telephone: +91 38640 14004                 + Website: www.chaudharigroup.com
```

The same real value always maps to the same fake, and a company's website and
email domains map together, so the redacted document still reads as one coherent
prospectus rather than noise.

## Quick start

```bash
uv sync                                    # installs the spaCy model too
uv run redact data/prospectus.docx -o out/prospectus.redacted.docx --seed 0
```

Put the provided prospectus at `data/prospectus.docx` — the source document is
gitignored because it holds real PII. The redacted output ships in this repo at
[`out/prospectus.redacted.docx`](out/prospectus.redacted.docx).

Useful flags: `--no-ner` (rules only, fast), `--dry-run`, `--types`, `--seed`,
`--redact-reference-numbers`, `--report`.

## Results

Measured on a hand-labelled synthetic corpus, seed 0, relaxed span matching:

| | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| rules only (`sample_corpus`, all 9 types) | 0.922 | 1.000 | 0.917 | 0.957 |
| rules + NER (`pages_sample`, 6 pages) | — | 0.979 | 0.959 | 0.969 |

Method, per-type tables and error analysis:
[`evaluation/EVALUATION.md`](evaluation/EVALUATION.md).

The corpora are synthetic on purpose — labelling the real prospectus would mean
committing its PII to this repo. That makes the numbers a fair measure of the
detectors but not of this exact document, so there is a second check that needs
no answer key:

```bash
uv run python evaluation/audit_redaction.py data/prospectus.docx out/prospectus.redacted.docx
```

It re-scans the output with independent patterns and reports what survived. On
the prospectus it scores **98.8% (158 of 160 PII values removed)** and exits
non-zero when anything is left, so it can gate a batch run. It also reports
over-redaction, which no recall metric can see: if a common word like "India"
occurs 150 times in the source and 0 times in the output, something classified
ordinary vocabulary as PII.

## Approach

Rules plus validators for the types with hard structure — email, phone, SSN,
Luhn-checked cards, IP, DOB, postal address, website domain, plus companies
carrying a legal suffix and people introduced by a label ("Contact Person:").
spaCy `en_core_web_sm` covers what those rules cannot express.
Overlapping hits resolve by longest span, then detector priority, so a
checksum-backed match beats a plain regex on the same characters.

Two things were less obvious than expected:

**Word hides text outside paragraphs.** 27 of the prospectus's email addresses
live in `w:instrText` HYPERLINK field codes that `python-docx`'s `Paragraph.text`
cannot see. Extraction covers field codes, text boxes, every header/footer part,
document metadata and relationship targets — and the leak check reads the raw
package, not the same API, because a check that shares its subject's blind spot
cannot audit it.

**Frequency is not a boilerplate signal.** Filtering NER candidates that repeat
often looked obvious, and measured at AUC 0.588 — essentially nothing. In an IPO
prospectus the promoter family is named on nearly every page, so the most
frequent strings are also the most sensitive; a threshold of 15 discarded every
promoter name. Two lexical rules replaced it: reject spans starting with a
determiner, and reject spans whose every token also appears lowercase elsewhere
in the document. That second rule is the document describing its own vocabulary,
so it transfers to documents this was never tuned on.

## Tradeoffs and known errors

**Reference numbers are not PII by default.** "Ticket #12345" and "Order 887"
are left alone; `--redact-reference-numbers` flips it.

**Only birth-cued dates are treated as dates of birth.** A prospectus is dense
with issue and board dates. This trades recall for precision deliberately.

**False negatives on the prospectus** (from the audit above): `Kishan Rastogi`
survives in one of its four occurrences — three sit under a `Contact Person:`
label and were caught, the fourth appears bare in a table. Companies are now
found by legal suffix as well as by the model, which took them from 34/44 to
54/54 on this document; a company written without a suffix is still a miss, and
on a support-ticket log that cost 3 of 4.

**False positives.** The audit flags `Gross National Disposable Inc` as a
surviving company; it is a financial term, not a company. The probe patterns are
broad on purpose — a false alarm costs a glance, a missed category costs a
disclosure. Over-redaction in the document itself runs at 10.2% of blocks
touched; boilerplate like "the Offer", "Equity Shares" and "ASBA Bidders" is
preserved, as are place names in prose.

**Place names are not addresses.** spaCy tags every location as `GPE` and an
earlier version mapped that straight to `ADDRESS`, so a bare "India" in a
sentence about production capacity became a postal address — and because
occurrence expansion propagates any accepted value document-wide, that rewrote
all 150 occurrences of the word. An NER location span now has to carry a street
number or PIN itself; the full address is the rule detector's job.

**Out of scope:** addresses spanning paragraph boundaries, scanned images (no
OCR), and non-Indian/US identifier formats such as Aadhaar, PAN or IBAN.

## Adding a new PII type

Three places, and the tests enforce the third:

1. Add a member to `PIIType`
2. Register a `Detector` (pattern plus a `validate()` hook)
3. Register a surrogate generator

The surrogate tests are parametrised over `PIIType`, so a new member fails the
suite until its generator exists.

## Development

```bash
uv run pytest          # 289 tests
uv run ruff check
uv run mypy src
uv run python evaluation/assert_baselines.py    # metric floors, also run in CI
```
