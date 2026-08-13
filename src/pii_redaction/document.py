"""Word .docx adapter: flatten, extract, splice runs, save."""

from __future__ import annotations

import bisect
from collections.abc import Iterator, Sequence
from pathlib import Path

from docx import Document as _open_docx
from docx.document import Document as _Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.text.run import Run

from pii_redaction.models import DocumentError, PIIEntity, assert_consistent


class DocxDocument:
    """Load a .docx, expose flat paragraph blocks, splice replacements into runs."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path).resolve()
        try:
            self._doc: _Document = _open_docx(str(self._path))
        except Exception as exc:  # noqa: BLE001 — wrap as DocumentError
            raise DocumentError(f"cannot open docx: {self._path}") from exc
        self._blocks: list[Paragraph] = list(self._iter_paragraphs())
        self._block_spans: list[tuple[int, int]] = []
        self._cached_text: str | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def blocks(self) -> Sequence[Paragraph]:
        return self._blocks

    def _iter_paragraphs(self) -> Iterator[Paragraph]:
        yield from self._walk_container(self._doc)
        seen_header_ids: set[int] = set()
        seen_footer_ids: set[int] = set()
        for section in self._doc.sections:
            header = section.header
            header_id = id(header._element)  # noqa: SLF001
            if header_id not in seen_header_ids:
                seen_header_ids.add(header_id)
                if not _is_linked_to_previous(header):
                    yield from self._iter_nonempty_hf(header)
            footer = section.footer
            footer_id = id(footer._element)  # noqa: SLF001
            if footer_id not in seen_footer_ids:
                seen_footer_ids.add(footer_id)
                if not _is_linked_to_previous(footer):
                    yield from self._iter_nonempty_hf(footer)

    def _iter_nonempty_hf(self, header_or_footer: object) -> Iterator[Paragraph]:
        """Yield header/footer paragraphs only when the part has real text.

        python-docx often materialises an empty unlinked header/footer on save;
        including those would append spurious empty blocks and shift offsets.
        """
        paragraphs = list(self._walk_container(header_or_footer))
        if not any(p.text for p in paragraphs):
            return
        yield from paragraphs

    def _walk_container(self, container: object) -> Iterator[Paragraph]:
        iter_inner = getattr(container, "iter_inner_content", None)
        if iter_inner is None:
            paragraphs = getattr(container, "paragraphs", None)
            if paragraphs is not None:
                yield from paragraphs
            tables = getattr(container, "tables", None)
            if tables is not None:
                for table in tables:
                    yield from self._walk_table(table)
            return
        for item in iter_inner():
            if isinstance(item, Paragraph):
                yield item
            elif isinstance(item, Table):
                yield from self._walk_table(item)

    def _walk_table(self, table: Table) -> Iterator[Paragraph]:
        seen_cells: set[int] = set()
        for row in table.rows:
            for cell in row.cells:
                cell_id = id(cell._tc)  # noqa: SLF001
                if cell_id in seen_cells:
                    continue
                seen_cells.add(cell_id)
                yield from self._walk_container(cell)

    def extract_text(self) -> str:
        if self._cached_text is not None:
            return self._cached_text
        parts: list[str] = []
        spans: list[tuple[int, int]] = []
        offset = 0
        for block in self._blocks:
            text = block.text
            start = offset
            end = offset + len(text)
            spans.append((start, end))
            parts.append(text)
            offset = end + 1  # +1 for the "\n" join separator
        joined = "\n".join(parts)
        self._block_spans = spans
        self._cached_text = joined
        return joined

    def _block_index_for_offset(self, offset: int) -> int:
        if not self._block_spans:
            self.extract_text()
        if not self._block_spans:
            raise DocumentError(f"offset {offset} outside document text")
        starts = [s for s, _ in self._block_spans]
        idx = bisect.bisect_right(starts, offset) - 1
        if idx < 0 or idx >= len(self._block_spans):
            raise DocumentError(f"offset {offset} outside document text")
        start, end = self._block_spans[idx]
        # end is exclusive; offset == end is the "\n" separator (or an empty block)
        if offset < start or offset >= end:
            raise DocumentError(f"offset {offset} is on a block separator")
        return idx

    def apply(self, entities: Sequence[PIIEntity]) -> None:
        _assert_non_overlapping(entities)
        text = self.extract_text()
        assert_consistent(text, entities)
        by_block: dict[int, list[tuple[int, int, str]]] = {}
        for entity in entities:
            if entity.replacement is None:
                raise DocumentError(
                    f"entity at {entity.start}:{entity.end} has no replacement"
                )
            block_idx = self._block_index_for_offset(entity.start)
            block_start, block_end = self._block_spans[block_idx]
            if entity.end > block_end:
                raise DocumentError(
                    f"entity at {entity.start}:{entity.end} spans multiple blocks"
                )
            local_start = entity.start - block_start
            local_end = entity.end - block_start
            by_block.setdefault(block_idx, []).append(
                (local_start, local_end, entity.replacement)
            )
        for block_idx, spans in by_block.items():
            paragraph = self._blocks[block_idx]
            for local_start, local_end, replacement in sorted(
                spans, key=lambda s: s[0], reverse=True
            ):
                _splice_paragraph(paragraph, local_start, local_end, replacement)
        self._cached_text = None
        self._block_spans = []

    def rendered_text(self) -> str:
        return "\n".join(block.text for block in self._blocks)

    def save(self, path: Path | str) -> None:
        out = Path(path).resolve()
        if out == self._path:
            raise DocumentError(f"refusing to overwrite input path: {out}")
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            self._doc.save(str(out))
        except DocumentError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DocumentError(f"cannot save docx: {out}") from exc


def _assert_non_overlapping(entities: Sequence[PIIEntity]) -> None:
    ordered = sorted(entities, key=lambda e: e.start)
    for left, right in zip(ordered, ordered[1:], strict=False):
        if left.overlaps(right):
            raise DocumentError(
                f"overlapping entities at {left.start}:{left.end} and "
                f"{right.start}:{right.end}"
            )


def _is_linked_to_previous(header_or_footer: object) -> bool:
    is_linked = getattr(header_or_footer, "is_linked_to_previous", None)
    if callable(is_linked):
        return bool(is_linked())
    if isinstance(is_linked, bool):
        return is_linked
    return False


def _run_text(run: Run) -> str:
    return run.text or ""


def _splice_paragraph(
    paragraph: Paragraph, local_start: int, local_end: int, replacement: str
) -> None:
    """Replace paragraph[local_start:local_end] across runs; keep formatting outside."""
    runs = list(paragraph.runs)
    if not runs:
        paragraph.add_run(replacement)
        return

    run_spans: list[tuple[Run, int, int]] = []
    offset = 0
    for run in runs:
        text = _run_text(run)
        run_spans.append((run, offset, offset + len(text)))
        offset += len(text)

    total = offset
    if local_start < 0 or local_end > total or local_start >= local_end:
        raise DocumentError(
            f"local span {local_start}:{local_end} out of range for paragraph "
            f"length {total}"
        )

    covered = [
        (run, rs, re)
        for run, rs, re in run_spans
        if rs < local_end and re > local_start
    ]
    if not covered:
        raise DocumentError(f"no runs cover local span {local_start}:{local_end}")

    first_run, first_rs, _ = covered[0]
    last_run, last_rs, _ = covered[-1]
    first_prefix = _run_text(first_run)[: local_start - first_rs]
    last_suffix = _run_text(last_run)[local_end - last_rs :]

    if first_run is last_run:
        first_run.text = first_prefix + replacement + last_suffix
        return

    first_run.text = first_prefix + replacement
    for run, _, _ in covered[1:-1]:
        run.text = ""
    last_run.text = last_suffix
