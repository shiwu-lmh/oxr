import html
import re

from oxr.pipeline.models import Block, ContentFormat
from oxr.pipeline.postprocess.repetition_postprocess import postprocess_repetition_loops
from oxr.utils.table import (
    html_table_to_markdown,
    keep_complete_html_tables,
    strip_table_code_fence,
)


_TABLE_ROW = re.compile(r"(?P<open><tr\b[^>]*>)(?P<body>.*?)</tr\s*>", re.I | re.S)
_TABLE_CELL = re.compile(r"<(?P<tag>td|th)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)\s*>", re.I | re.S)
_RESULT_HEADER = re.compile(r"结果|测定值|检验结果|检测结果|result|value", re.I)
_REFERENCE_HEADER = re.compile(r"参考(?:区间|范围)|正常范围|reference", re.I)
_FLAG_HEADER = re.compile(r"异常|标志|提示|flag", re.I)
_ARROW_ONLY = re.compile(r"^[↑↓↗↘⇧⇩]+$")


def _cell_text(cell_match: re.Match[str]) -> str:
    text = re.sub(r"<[^>]*>", "", cell_match.group("body"))
    return html.unescape(text).strip()


def _simple_cells(row_body: str) -> list[re.Match[str]] | None:
    cells = list(_TABLE_CELL.finditer(row_body))
    if not cells or "".join(cell.group(0) for cell in cells).strip() != row_body.strip():
        return None
    if any(
        re.search(r"\b(?:rowspan|colspan)\s*=", cell.group("attrs"), re.I)
        or re.search(r"<(?!/?br\b)[a-z][^>]*>", cell.group("body"), re.I)
        for cell in cells
    ):
        return None
    return cells


def _merge_abnormal_arrow_cells(table: str) -> str:
    """Attach standalone abnormal arrows to result values before table export."""
    rows = list(_TABLE_ROW.finditer(table))
    header_index = None
    header_cells = None
    result_column = None
    for index, row in enumerate(rows):
        cells = _simple_cells(row.group("body"))
        if not cells:
            continue
        labels = [_cell_text(cell) for cell in cells]
        result_column = next((
            position for position, label in enumerate(labels)
            if _RESULT_HEADER.search(label)
        ), None)
        if result_column is None or result_column + 1 >= len(labels):
            continue
        next_header = labels[result_column + 1]
        if _FLAG_HEADER.search(next_header):
            continue
        if next_header and not _REFERENCE_HEADER.search(next_header):
            continue
        header_index, header_cells = index, cells
        break

    if header_index is None or header_cells is None or result_column is None:
        return table
    flag_column = result_column + 1
    expected_width = len(header_cells)
    replacements: dict[int, str] = {}
    for index, row in enumerate(rows):
        if index <= header_index:
            continue
        cells = _simple_cells(row.group("body"))
        if not cells or len(cells) <= flag_column:
            continue
        if len(cells) not in {expected_width, expected_width + 1}:
            continue
        if not _ARROW_ONLY.fullmatch(_cell_text(cells[flag_column])):
            continue

        previous = cells[result_column]
        arrow = _cell_text(cells[flag_column])
        updated_previous = (
            f"<{previous.group('tag')}{previous.group('attrs')}>"
            f"{previous.group('body').rstrip()} {arrow}"
            f"</{previous.group('tag')}>"
        )
        new_cells = [cell.group(0) for cell in cells]
        new_cells[result_column] = updated_previous
        del new_cells[flag_column]
        while len(new_cells) < expected_width:
            new_cells.append("<td></td>")
        new_body = "".join(new_cells)
        replacements[index] = row.group("open") + new_body + "</tr>"

    if not replacements:
        return table
    result = []
    cursor = 0
    for index, row in enumerate(rows):
        result.append(table[cursor:row.start()])
        result.append(replacements.get(index, row.group(0)))
        cursor = row.end()
    result.append(table[cursor:])
    return "".join(result)


def postprocess_table(block: Block, raw_text: str, output_format: str = "html") -> Block:
    """Clean up table block output to the requested table format."""
    text = strip_table_code_fence(raw_text)
    text = _merge_abnormal_arrow_cells(text)
    text = postprocess_repetition_loops(text, preserve_html=True)
    text = keep_complete_html_tables(text)
    if output_format == "markdown":
        block.content = html_table_to_markdown(text)
        block.format = ContentFormat.MARKDOWN
        return block

    block.content = text.strip()
    block.format = ContentFormat.HTML
    return block
