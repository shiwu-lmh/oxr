import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import List, Optional


@dataclass
class _TableCell:
    text: str
    rowspan: int = 1
    colspan: int = 1


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: List[List[_TableCell]] = []
        self._table_depth = 0
        self._current_row: Optional[List[_TableCell]] = None
        self._current_cell: Optional[dict] = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._table_depth += 1
            return
        if self._table_depth == 0:
            return
        if tag == "tr":
            self._current_row = []
            return
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell is None:
            attr_map = {key.lower(): value for key, value in attrs}
            self._current_cell = {
                "text": [],
                "rowspan": _positive_int(attr_map.get("rowspan")),
                "colspan": _positive_int(attr_map.get("colspan")),
            }
            return
        if tag == "br" and self._current_cell is not None:
            self._current_cell["text"].append(" ")
        if tag == "img" and self._current_cell is not None:
            src = dict(attrs).get("src")
            if src:
                self._current_cell["text"].append(f"![]({src})")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_cell is not None and self._current_row is not None:
            self._current_row.append(
                _TableCell(
                    text=_normalize_cell_text("".join(self._current_cell["text"])),
                    rowspan=self._current_cell["rowspan"],
                    colspan=self._current_cell["colspan"],
                )
            )
            self._current_cell = None
            return
        if tag == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None
            return
        if tag == "table" and self._table_depth:
            self._table_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell["text"].append(data)


def _positive_int(value: Optional[str]) -> int:
    try:
        parsed = int(value or "1")
    except ValueError:
        return 1
    return max(parsed, 1)


def _normalize_cell_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_table_code_fence(text: str) -> str:
    text = text.strip()
    match = re.search(r"```html\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if "```" in text:
        match = re.search(r"```(?:.*?)\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


_TABLE_TAG = re.compile(r"</?\s*table\b[^>]*>", re.IGNORECASE)
_TABLE_ROW = re.compile(r"<tr\b[^>]*>.*?</tr\s*>", re.IGNORECASE | re.DOTALL)
_TABLE_CELL = re.compile(
    r"<(?P<tag>td|th)\b(?P<attrs>[^>]*)>(?P<content>.*?)</(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_TAG = re.compile(r"<[^>]+>")


def _visible_cell(cell_html: str) -> str:
    return " ".join(html.unescape(_HTML_TAG.sub(" ", cell_html)).split())


def _repair_spilled_last_column_header(table: str, suffix: str) -> str:
    """Restore a truncated last header when spilled values duplicate its column."""
    lines = [line.strip() for line in suffix.splitlines() if line.strip()]
    if len(lines) < 4 or any("<" in line or ">" in line for line in lines):
        return table
    spilled_header, spilled_values = lines[0], lines[1:]
    rows = list(_TABLE_ROW.finditer(table))
    if not rows:
        return table

    first_row = rows[0]
    header_cells = list(_TABLE_CELL.finditer(first_row.group()))
    if not header_cells:
        return table
    last_header = header_cells[-1]
    header_text = _visible_cell(last_header.group("content"))
    if (
        not header_text
        or len(spilled_header) <= len(header_text)
        or not spilled_header.startswith(header_text)
    ):
        return table

    last_column_values = []
    for row in rows:
        cells = list(_TABLE_CELL.finditer(row.group()))
        if cells:
            last_column_values.append(_visible_cell(cells[-1].group("content")))
    if not any(
        last_column_values[index : index + len(spilled_values)] == spilled_values
        for index in range(len(last_column_values) - len(spilled_values) + 1)
    ):
        return table

    repaired_cell = (
        f"<{last_header.group('tag')}{last_header.group('attrs')}>"
        f"{html.escape(spilled_header)}"
        f"</{last_header.group('tag')}>"
    )
    row_text = first_row.group()
    repaired_row = row_text[: last_header.start()] + repaired_cell + row_text[last_header.end() :]
    return table[: first_row.start()] + repaired_row + table[first_row.end() :]


def keep_complete_html_tables(text: str) -> str:
    """Keep complete HTML table blocks and discard model text after them.

    If a repeated detached column follows a table and its values exactly match
    a contiguous run in the table's last column, restore a visibly truncated
    final header before removing that duplicate spillover.
    """
    table_spans: list[tuple[int, int, str]] = []
    depth = 0
    start = None
    for match in _TABLE_TAG.finditer(text):
        is_closing = match.group().lstrip().startswith("</")
        if not is_closing:
            if depth == 0:
                start = match.start()
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0 and start is not None:
                table_spans.append((start, match.end(), text[start : match.end()]))
                start = None

    if not table_spans:
        return text.strip()

    cleaned_tables = [table for _, _, table in table_spans]
    final_end = table_spans[-1][1]
    if final_end < len(text):
        cleaned_tables[-1] = _repair_spilled_last_column_header(
            cleaned_tables[-1], text[final_end:]
        )
    return "\n".join(cleaned_tables)


def _expand_spans(rows: List[List[_TableCell]]) -> List[List[str]]:
    expanded: List[List[str]] = []
    pending: dict[int, tuple[str, int]] = {}

    for source_row in rows:
        row: List[str] = []
        col_idx = 0

        def fill_pending_cell() -> bool:
            nonlocal col_idx
            if col_idx not in pending:
                return False
            text, remaining = pending[col_idx]
            row.append(text)
            if remaining > 1:
                pending[col_idx] = (text, remaining - 1)
            else:
                del pending[col_idx]
            col_idx += 1
            return True

        for cell in source_row:
            while fill_pending_cell():
                pass
            for _ in range(cell.colspan):
                row.append(cell.text)
                if cell.rowspan > 1:
                    pending[col_idx] = (cell.text, cell.rowspan - 1)
                col_idx += 1

        while pending and col_idx <= max(pending):
            if not fill_pending_cell():
                row.append("")
                col_idx += 1

        expanded.append(row)

    return expanded


def _escape_markdown_cell(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|")


def html_table_to_markdown(html: str) -> str:
    """Convert an HTML table to Markdown, expanding rowspans and colspans."""
    cleaned = strip_table_code_fence(html)
    parser = _TableParser()
    parser.feed(cleaned)

    rows = _expand_spans(parser.rows)
    if not rows:
        return _normalize_cell_text(re.sub(r"<[^>]+>", " ", cleaned))

    width = max(len(row) for row in rows)
    padded_rows = [row + [""] * (width - len(row)) for row in rows]
    escaped_rows = [
        [_escape_markdown_cell(_normalize_cell_text(cell)) for cell in row]
        for row in padded_rows
    ]

    header = escaped_rows[0]
    separator = ["---"] * width
    body = escaped_rows[1:]
    markdown_rows = [header, separator, *body]
    return "\n".join(f"| {' | '.join(row)} |" for row in markdown_rows)
