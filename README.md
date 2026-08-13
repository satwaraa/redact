# redact

Reads a `.docx`, finds PII, and writes another `.docx` where those values are
replaced with realistic fakes (same shape: Luhn-valid cards, grouped phones, and
so on). Not blacked out, not deleted.

## Approach

Detection is mostly regex plus validators (Luhn for cards, basic SSN / IP /
phone shape checks). Overlapping hits are resolved by longest span, then source
priority (checksum-backed rules beat plain regex). Optional spaCy NER
(`en_core_web_sm`) can pick up names and company names; default CLI runs can
skip it with `--no-ner`. Surrogates are seeded so the same real value maps to
the same fake inside one document.

## Tradeoffs

Rule-based types (email, phone, SSN, card, DOB, address, IP) are precise on the
synthetic eval corpus but miss anything that needs language understanding
names and orgs without NER are false negatives. NER helps those, but it can
also over-tag prospectus boilerplate and Indian names are a weak spot for
`en_core_web_sm`. Reference numbers (ticket/order style) are left alone unless
you pass `--redact-reference-numbers`. Addresses that cross paragraph boundaries
can be detected on flat text and then dropped at apply. Text boxes, footnotes,
and scans are out of scope.

Metrics and how they were measured: see [`evaluation/EVALUATION.md`](evaluation/EVALUATION.md).

## Quick start

```bash
uv sync
uv run python -m spacy download en_core_web_sm   # only if you want NER
uv run redact data/prospectus.docx -o out/redacted.docx --seed 0
```

Useful: `--no-ner`, `--dry-run`, `--types`, `--seed`. Regenerate metrics with
`uv run python evaluation/evaluate.py --no-ner --seed 0`.
