"""Word .docx adapter: flatten, extract, splice runs, save."""

from __future__ import annotations

import bisect
import re
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from docx import Document as _open_docx
from docx.document import Document as _Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.text.run import Run

from pii_redaction.models import DocumentError, PIIEntity, assert_consistent

_STORY_PART_RE = re.compile(r"/word/(document|header\d+|footer\d+)\.xml$")


def iter_package_texts(path: Path | str) -> Iterator[tuple[str, str]]:
    """Yield ``(part_name, text)`` for every part in a serialized .docx package."""
    package = Path(path)
    try:
        with zipfile.ZipFile(package) as zf:
            for name in zf.namelist():
                raw = zf.read(name)
                yield name, raw.decode("utf-8", errors="ignore")
    except zipfile.BadZipFile as exc:
        raise DocumentError(f"cannot read docx package: {package}") from exc


def package_corpus(path: Path | str) -> str:
    """Concatenate every package part's text (for package-level leak probes)."""
    return "\n".join(text for _, text in iter_package_texts(path))


@runtime_checkable
class TextBlock(Protocol):
    @property
    def text(self) -> str: ...

    def replace_span(self, start: int, end: int, replacement: str) -> None: ...


class ParagraphBlock:
    """Paragraph story content; splice goes through python-docx runs."""

    __slots__ = ("_paragraph",)

    def __init__(self, paragraph: Paragraph) -> None:
        self._paragraph = paragraph

    @property
    def text(self) -> str:
        return self._paragraph.text

    @property
    def runs(self) -> list[Run]:
        return list(self._paragraph.runs)

    @property
    def paragraph(self) -> Paragraph:
        return self._paragraph

    def replace_span(self, start: int, end: int, replacement: str) -> None:
        _splice_paragraph(self._paragraph, start, end, replacement)


class InstrTextBlock:
    """One ``w:instrText`` node as its own block (PLAN A1)."""

    __slots__ = ("_element",)

    def __init__(self, element: object) -> None:
        self._element = element

    @property
    def text(self) -> str:
        return getattr(self._element, "text", None) or ""

    def replace_span(self, start: int, end: int, replacement: str) -> None:
        current = self.text
        if start < 0 or end > len(current) or start >= end:
            raise DocumentError(
                f"local span {start}:{end} out of range for instrText length "
                f"{len(current)}"
            )
        self._element.text = current[:start] + replacement + current[end:]


def _iter_story_parts(doc: _Document) -> list[object]:
    """Document part first, then every header/footer part (PLAN A3)."""
    main = doc.part
    extras: list[object] = []
    seen: set[int] = {id(main)}
    for part in main.package.parts:
        if id(part) in seen:
            continue
        if _STORY_PART_RE.search(str(part.partname)):
            extras.append(part)
            seen.add(id(part))
    extras.sort(key=lambda p: str(p.partname))
    return [main, *extras]


def _iter_part_paragraphs(part: object) -> Iterator[Paragraph]:
    """All ``w:p`` in the part, including those under ``w:txbxContent`` (A2)."""
    element = getattr(part, "element", None)
    if element is None:
        return
    for p_el in element.iter(qn("w:p")):
        yield Paragraph(p_el, part)


def _iter_part_instr_texts(part: object) -> Iterator[object]:
    element = getattr(part, "element", None)
    if element is None:
        return
    yield from element.iter(qn("w:instrText"))


class DocxDocument:
    """Load a .docx, expose flat text blocks, splice replacements, save."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path).resolve()
        try:
            self._doc: _Document = _open_docx(str(self._path))
        except Exception as exc:  # noqa: BLE001 — wrap as DocumentError
            raise DocumentError(f"cannot open docx: {self._path}") from exc
        self._blocks: list[TextBlock] = list(self._collect_blocks())
        self._block_spans: list[tuple[int, int]] = []
        self._cached_text: str | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def blocks(self) -> Sequence[TextBlock]:
        return self._blocks

    def _collect_blocks(self) -> Iterator[TextBlock]:
        """Paragraphs from every story part, then each instrText as its own block."""
        seen_p: set[int] = set()
        seen_instr: set[int] = set()
        instr_blocks: list[InstrTextBlock] = []

        for part in _iter_story_parts(self._doc):
            for paragraph in _iter_part_paragraphs(part):
                pid = id(paragraph._element)  # noqa: SLF001
                if pid in seen_p:
                    continue
                seen_p.add(pid)
                yield ParagraphBlock(paragraph)
            for instr in _iter_part_instr_texts(part):
                iid = id(instr)
                if iid in seen_instr:
                    continue
                seen_instr.add(iid)
                instr_blocks.append(InstrTextBlock(instr))

        yield from instr_blocks

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
            block = self._blocks[block_idx]
            for local_start, local_end, replacement in sorted(
                spans, key=lambda s: s[0], reverse=True
            ):
                block.replace_span(local_start, local_end, replacement)
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
