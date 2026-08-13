# PII Redaction Tool

Reads a `.docx`, detects personally identifiable information, and writes a
redacted `.docx` in which every PII value is replaced by a realistic,
format-preserving fake — not masked, not deleted.

## Install

```
uv sync
uv run python -m spacy download en_core_web_sm
```

## Usage

```
uv run redact data/prospectus.docx -o out/redacted.docx --seed 42
```

| Flag | Purpose |
|---|---|
| `--dry-run` | Run the pipeline and print a summary; write no file. |
| `--dump-text` | Print the extracted text and exit — useful when a detector appears to miss something. |
| `--no-ner` | Rule-based detection only; fast and fully deterministic. |
| `--types` | Comma-separated subset of PII types to redact. |
| `--redact-reference-numbers` | Treat ticket/order/invoice numbers as PII. |
| `--report PATH` | Write the entity list and real→fake mapping as JSON. |
| `--seed` | Surrogate RNG seed, for reproducible output. |

Evaluation:

```
uv run python evaluation/evaluate.py --input data/prospectus.docx \
    --truth evaluation/ground_truth.json --out evaluation/report.md
```

## Development

```
uv run pytest
uv run ruff check
uv run mypy src
```
