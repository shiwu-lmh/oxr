"""Quality checks for model-generated tables before they are exported."""

from __future__ import annotations

import html
import re
from datetime import date

from oxr.utils.table import _TableParser, strip_table_code_fence


_DATE = re.compile(r"(?<!\d)(20\d{2})[./-](\d{1,2})[./-](\d{1,2})(?!\d)")
_NUMBER = re.compile(r"(?<![\w.])(-?\d+(?:[.,]\d+)?)")


def _rows(text: str) -> list[list[str]]:
    if "<tr" not in text.casefold() and "|" in text:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if cells and all(re.fullmatch(r"[:\-\s]+", cell or "-") for cell in cells):
                continue
            rows.append(cells)
        return rows
    parser = _TableParser()
    try:
        parser.feed(strip_table_code_fence(text))
    except Exception:
        return []
    return [[cell.text.strip() for cell in row] for row in parser.rows]


def _parse_date(match: re.Match[str]) -> date | None:
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _date_conflicts_with_context(table_text: str, context_text: str) -> bool:
    plain = html.unescape(re.sub(r"<[^>]*>", " ", table_text))
    baseline = re.search(
        r"基线(?:期)?\s*[:：]?\s*(?:检查日期\s*[:：]?\s*)?"
        r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2})",
        plain,
    )
    if not baseline:
        return False
    baseline_date = _parse_date(baseline)
    if baseline_date is None:
        return False

    context_dates = [
        parsed
        for match in _DATE.finditer(context_text)
        if (parsed := _parse_date(match)) is not None
    ]
    if not context_dates:
        return False
    return min(abs((baseline_date - other).days) for other in context_dates) > 730


def table_quality_issues(
    text: str,
    image_height: int,
    context_text: str = "",
) -> list[str]:
    """Flag table outputs whose syntax is valid but whose content is implausible."""
    rows = _rows(text)
    issues: list[str] = []
    if not rows:
        return ["no_table_rows"]

    # A crop cannot physically contain thousands of distinct visual rows.
    # Use a generous row density limit to avoid flagging compact printed tables.
    max_rows = max(120, image_height // 3)
    if len(rows) > max_rows:
        issues.append("row_count_exceeds_image_capacity")

    signatures = [tuple(cell.casefold() for cell in row) for row in rows]
    longest_run = 1
    current_run = 1
    for previous, current in zip(signatures, signatures[1:]):
        if current and current == previous:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
    nonempty_single_cell_rows = [
        row[0].casefold()
        for row in rows
        if len(row) == 1 and row[0].strip()
    ]
    if longest_run >= 12:
        issues.append("consecutive_duplicate_rows")
    if (
        len(nonempty_single_cell_rows) >= 20
        and len(nonempty_single_cell_rows) / len(rows) >= 0.2
    ):
        issues.append("single_cell_row_explosion")

    # For RECIST target-lesion tables, compare the entered total against the
    # measured values in the target rows. This catches lost leading digits and
    # shifted numeric cells without changing any values by inference.
    target_sizes: list[float] = []
    reported_total: float | None = None
    for row in rows:
        cells = [re.sub(r"\s+", " ", cell).strip() for cell in row]
        row_text = " ".join(cells)
        if "靶病灶直径总和" in row_text:
            total_label_end = row_text.find("靶病灶直径总和") + len("靶病灶直径总和")
            match = _NUMBER.search(row_text[total_label_end:].replace("_", " "))
            if match:
                reported_total = float(match.group(1).replace(",", "."))
            continue
        if len(cells) >= 3 and cells[0] in {"1", "2", "3", "4", "5"}:
            measurement = cells[2].replace("_", " ")
            match = _NUMBER.search(measurement)
            if match:
                target_sizes.append(float(match.group(1).replace(",", ".")))
    if reported_total is not None and len(target_sizes) >= 2:
        calculated = sum(target_sizes)
        if abs(calculated - reported_total) > max(0.15, calculated * 0.002):
            issues.append("target_diameter_sum_mismatch")

    if context_text and _date_conflicts_with_context(text, context_text):
        issues.append("baseline_date_conflicts_with_adjacent_reports")

    return issues
