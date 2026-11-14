from __future__ import annotations

import logging
import re
from html.parser import HTMLParser

import cv2
import numpy as np

from oxr.utils.table import _TableParser, _expand_spans, strip_table_code_fence


logger = logging.getLogger(__name__)

TABLE_CHUNK_MAX_HEIGHT = 620
TABLE_CHUNK_MIN_HEIGHT = 96


class _TableValidationParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.counts = {tag: 0 for tag in ("table", "tr", "td", "th")}
        self.open_counts = {tag: 0 for tag in self.counts}
        self.current_cell: list[str] | None = None
        self.cells: list[str] = []
        self.invalid = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.counts:
            self.counts[tag] += 1
            self.open_counts[tag] += 1
            if tag in {"td", "th"}:
                if self.current_cell is not None:
                    self.invalid = True
                self.current_cell = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.counts:
            self.open_counts[tag] -= 1
            if self.open_counts[tag] < 0:
                self.invalid = True
            if tag in {"td", "th"} and self.current_cell is not None:
                self.cells.append(" ".join("".join(self.current_cell).split()))
                self.current_cell = None

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)


def table_row_widths(text: str) -> list[int]:
    """Return logical column counts after expanding row and column spans."""
    cleaned = strip_table_code_fence(text)
    table_parser = _TableParser()
    try:
        table_parser.feed(cleaned)
    except Exception:
        return []
    return [len(row) for row in _expand_spans(table_parser.rows) if row]


def expected_table_columns(text: str) -> int | None:
    """Infer a stable logical column count when most rows agree."""
    widths = table_row_widths(text)
    if len(widths) < 2:
        return None
    counts = {width: widths.count(width) for width in set(widths)}
    expected, votes = max(counts.items(), key=lambda item: item[1])
    return expected if votes / len(widths) >= 0.6 else None


def is_balanced_table_html(text: str) -> bool:
    """Check that output is a usable, closed table without judging its row widths."""
    cleaned = strip_table_code_fence(text)
    if re.search(
        r"&lt;\s*/?\s*(?:table|thead|tbody|tfoot|tr|td|th)\b",
        cleaned,
        flags=re.I,
    ):
        return False
    parser = _TableValidationParser()
    try:
        parser.feed(cleaned)
        parser.close()
    except Exception:
        return False
    if parser.invalid or parser.counts["table"] != 1 or parser.counts["tr"] == 0:
        return False
    if parser.counts["td"] + parser.counts["th"] == 0 or any(parser.open_counts.values()):
        return False
    repeated = 0
    previous = None
    for cell in parser.cells:
        if cell and cell == previous:
            repeated += 1
            if repeated >= 48:
                return False
        else:
            repeated = 0
        previous = cell
    return True


def is_complete_table_html(text: str) -> bool:
    """Check syntax and logical column consistency after expanding spans."""
    if not is_balanced_table_html(text):
        return False
    widths = set(table_row_widths(text))
    return len(widths) <= 1


def repair_table_html_prefix(text: str) -> str | None:
    """Restore missing leading table/row/cell tags when the model starts mid-table."""
    cleaned = strip_table_code_fence(text).strip()
    tags = list(re.finditer(r"</?(table|tr|td|th)\b[^>]*>", cleaned, flags=re.I))
    if not tags:
        return None

    def first_tag(pattern: str):
        return re.search(pattern, cleaned, flags=re.I)

    table_open = first_tag(r"<table\b[^>]*>")
    table_close = first_tag(r"</table\s*>")
    has_table = table_open is not None or table_close is not None
    if not has_table:
        return None

    prefix_table = table_open is None or (
        table_close is not None and table_close.start() < table_open.start()
    )
    row_open = first_tag(r"<tr\b[^>]*>")
    row_close = first_tag(r"</tr\s*>")
    prefix_row = row_close is not None and (
        row_open is None or row_close.start() < row_open.start()
    )
    cell_open = first_tag(r"<(?:td|th)\b[^>]*>")
    cell_close = first_tag(r"</(?:td|th)\s*>")
    prefix_cell = cell_close is not None and (
        cell_open is None or cell_close.start() < cell_open.start()
    )
    if not (prefix_table or prefix_row or prefix_cell):
        return None

    candidate = cleaned
    if prefix_table:
        candidate = "<table>" + candidate
    table_start = re.search(r"<table\b[^>]*>", candidate, flags=re.I)
    if table_start is None:
        return None
    leading = ("<tr>" if prefix_row else "") + ("<td>" if prefix_cell else "")
    candidate = candidate[:table_start.end()] + leading + candidate[table_start.end():]
    return candidate if is_balanced_table_html(candidate) else None


def _horizontal_rules(image: np.ndarray) -> list[int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    coverage = np.mean(gray < 205, axis=1)
    candidates = np.flatnonzero(coverage >= 0.35).tolist()
    groups: list[list[int]] = []
    for y in candidates:
        if not groups or y > groups[-1][-1] + 2:
            groups.append([y])
        else:
            groups[-1].append(y)
    return [round(sum(group) / len(group)) for group in groups if len(group) <= 8]


def split_table_image(
    image: np.ndarray,
    max_height: int = TABLE_CHUNK_MAX_HEIGHT,
) -> list[np.ndarray]:
    """Split a tall table near row boundaries, preferring visible grid rules."""
    return [chunk for chunk, _, _ in split_table_image_with_bounds(image, max_height)]


def split_table_image_with_bounds(
    image: np.ndarray,
    max_height: int = TABLE_CHUNK_MAX_HEIGHT,
) -> list[tuple[np.ndarray, int, int]]:
    """Return table chunks with their source y-coordinate ranges."""
    height = image.shape[0]
    if height <= max_height:
        return [(image, 0, height)]
    rules = _horizontal_rules(image)
    chunks: list[tuple[np.ndarray, int, int]] = []
    start = 0
    while height - start > max_height:
        min_chunk_margin = min(TABLE_CHUNK_MIN_HEIGHT, max(36, max_height // 3))
        low = start + min_chunk_margin
        # The cut row is included in both neighboring chunks, so the cut
        # index must be at most max_height - 1 to keep every chunk bounded.
        min_tail = max(36, min(TABLE_CHUNK_MIN_HEIGHT, max_height // 3))
        high = min(height - 1, start + max_height - 1, height - min_tail)
        candidates = [y for y in rules if low <= y <= high]
        if candidates:
            cut = min(candidates, key=lambda y: abs(y - high))
        else:
            # Borderless tables have no full-width rule to cut on. Find the
            # quietest horizontal band in the safe range, where row spacing
            # usually leaves a natural boundary. A small overlap preserves
            # glyphs touching the boundary without duplicating full rows.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            ink = np.mean(gray < 205, axis=1)
            valley_low = max(start + min_chunk_margin, start + 24)
            valley_high = min(high, start + max_height - 24)
            if valley_low >= valley_high:
                break
            window = ink[valley_low:valley_high]
            if window.size == 0:
                break
            valley = valley_low + int(np.argmin(window))
            # Avoid cutting through dense text. Even when there is no perfectly
            # blank row, require a meaningful local reduction in ink density.
            local = ink[max(start, valley - 24):min(height, valley + 25)]
            if not local.size or float(ink[valley]) > max(0.04, float(np.percentile(local, 35))):
                break
            cut = valley
        if cut <= start or cut >= height:
            break
        overlap = min(3, max(0, cut - start - 1))
        end = cut + overlap + 1
        chunks.append((image[start:end].copy(), start, end))
        start = max(start + 1, cut - overlap)
    if start < height:
        chunks.append((image[start:height].copy(), start, height))
    return chunks if len(chunks) > 1 else [(image, 0, height)]


def _row_signature(row_html: str) -> tuple[str, ...]:
    """Return normalized cell text for a single HTML table row."""
    parser = _TableParser()
    parser.feed("<table>" + row_html + "</table>")
    if not parser.rows:
        return ()
    return tuple(cell.text.casefold() for cell in parser.rows[0])


def _rows_from_header(text: str) -> list[str]:
    """Find explicit header rows without assuming a fixed header height."""
    cleaned = strip_table_code_fence(text)
    thead = re.search(r"<thead\b[^>]*>(.*?)</thead\s*>", cleaned, flags=re.I | re.S)
    if thead:
        return re.findall(r"<tr\b[^>]*>.*?</tr\s*>", thead.group(1), flags=re.I | re.S)

    rows = re.findall(r"<tr\b[^>]*>.*?</tr\s*>", cleaned, flags=re.I | re.S)
    header_rows = []
    for row in rows:
        if re.search(r"<th\b", row, flags=re.I):
            header_rows.append(row)
        elif header_rows:
            break
    return header_rows


def extract_table_header_schema(text: str) -> str | None:
    """Convert detected header cells and spans into a compact textual schema."""
    rows = _rows_from_header(text)
    if not rows:
        return None
    lines = []
    for index, row in enumerate(rows, start=1):
        parser = _TableParser()
        parser.feed("<table>" + row + "</table>")
        cells = parser.rows[0] if parser.rows else []
        if not cells:
            continue
        labels = []
        for cell in cells:
            label = cell.text or "(blank)"
            spans = []
            if cell.colspan > 1:
                spans.append(f"colspan={cell.colspan}")
            if cell.rowspan > 1:
                spans.append(f"rowspan={cell.rowspan}")
            labels.append(label + (f" [{', '.join(spans)}]" if spans else ""))
        lines.append(f"Header row {index}: " + " | ".join(labels))
    return "\n".join(lines) or None


def combine_table_chunks(chunks: list[str]) -> str:
    """Merge table segments and remove only explicitly identified header rows."""
    chunk_rows = [
        re.findall(r"<tr\b[^>]*>.*?</tr\s*>", strip_table_code_fence(chunk), flags=re.I | re.S)
        for chunk in chunks
    ]
    header_signatures = set()
    if chunk_rows:
        for row in _rows_from_header(chunks[0]):
            signature = _row_signature(row)
            if signature:
                header_signatures.add(signature)
        # Some model responses use <td> for a header despite the prompt. If a
        # leading row repeats at the start of every recognized segment, treat
        # it as a header row too, regardless of its HTML tag.
        first_rows = chunk_rows[0]
        for index, row in enumerate(first_rows[:3]):
            signature = _row_signature(row)
            if signature and all(
                len(rows) > index and _row_signature(rows[index]) == signature
                for rows in chunk_rows[1:]
            ):
                header_signatures.add(signature)

    rows = list(chunk_rows[0]) if chunk_rows else []
    removed_headers = 0
    removed_separators = 0
    for rows_in_chunk in chunk_rows[1:]:
        remaining = list(rows_in_chunk)
        while remaining and _row_signature(remaining[0]) in header_signatures:
            remaining.pop(0)
            removed_headers += 1
        cleaned_rows = []
        for index, row in enumerate(remaining):
            parser = _TableParser()
            parser.feed("<table>" + row + "</table>")
            cells = parser.rows[0] if parser.rows else []
            is_blank_spacer = len(cells) == 1 and cells[0].colspan > 1 and not cells[0].text
            next_cells = []
            if index + 1 < len(remaining):
                next_signature = _row_signature(remaining[index + 1])
                next_cells = list(next_signature[:1])
            is_repeated_row_number = (
                len(cells) == 1 and cells[0].colspan > 1 and bool(cells[0].text)
                and next_cells and cells[0].text.casefold() == next_cells[0]
            )
            if is_blank_spacer or is_repeated_row_number:
                removed_separators += 1
                continue
            cleaned_rows.append(row)
        rows.extend(cleaned_rows)

    if removed_headers:
        logger.info("Removed %d repeated table header rows from segmented output.", removed_headers)
    if removed_separators:
        logger.info("Removed %d empty or repeated-number separator rows from segmented output.", removed_separators)
    if not rows:
        raise ValueError("No complete rows were returned by table recognition.")
    return "<table>\n" + "\n".join(rows) + "\n</table>"
