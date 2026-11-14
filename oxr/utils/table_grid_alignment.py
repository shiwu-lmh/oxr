"""Align model HTML table spans to visible grid lines when row mapping is safe."""

from __future__ import annotations

import html as html_lib
import logging
import re

import cv2
import numpy as np


logger = logging.getLogger(__name__)

_TABLE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)
_TABLE_BODY = re.compile(
    r"(?P<open><table\b[^>]*>)(?P<body>.*?)(?P<close></table\s*>)",
    re.IGNORECASE | re.DOTALL,
)
_ROW = re.compile(r"<tr\b[^>]*>.*?</tr\s*>", re.IGNORECASE | re.DOTALL)
_CELL = re.compile(
    r"<(?P<tag>td|th)\b(?P<attrs>[^>]*)>(?P<content>.*?)</(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_SPAN = re.compile(r"\bcolspan\s*=\s*(?:['\"]\s*(\d+)\s*['\"]|(\d+))", re.I)
_ROWSPAN = re.compile(r"\browspan\s*=\s*(?:['\"]\s*(\d+)\s*['\"]|(\d+))", re.I)


def _attr_int(pattern: re.Pattern, attrs: str) -> int:
    match = pattern.search(attrs)
    if not match:
        return 1
    return max(1, int(match.group(1) or match.group(2)))


def _cluster(indices: np.ndarray, gap: int = 2) -> list[int]:
    clusters: list[list[int]] = []
    for value in indices.tolist():
        value = int(value)
        if not clusters or value > clusters[-1][-1] + gap:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [round(sum(cluster) / len(cluster)) for cluster in clusters]


def _grid_binary(image: np.ndarray) -> np.ndarray | None:
    if image is None or image.size == 0 or image.ndim not in (2, 3):
        return None
    if image.ndim == 3:
        if image.shape[2] != 3:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if min(gray.shape) < 16:
        return None
    background = float(np.percentile(gray, 90))
    if background < 180:
        return None
    threshold = int(np.clip(background - 35, 200, 235))
    return cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]


def _row_grid_lines(binary: np.ndarray, row_count: int) -> list[int] | None:
    height, width = binary.shape
    kernel_width = max(8, int(width * 0.35))
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)),
    )
    coverage = np.count_nonzero(horizontal, axis=1)
    candidates = _cluster(np.flatnonzero(coverage >= width * 0.25))
    candidates = [y for y in candidates if y > 2]

    # A scrollbar or image edge can look like an extra bottom rule. Drop only
    # that edge when the remaining rules map one-to-one to the model's rows.
    if len(candidates) == row_count + 1 and candidates[-1] >= height - 8:
        candidates.pop()
    if len(candidates) != row_count:
        return None
    return candidates


def _column_grid_lines(binary: np.ndarray) -> list[int] | None:
    height, width = binary.shape
    kernel_height = max(8, int(height * 0.1))
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height)),
    )
    coverage = np.count_nonzero(vertical, axis=0)
    candidates = _cluster(np.flatnonzero(coverage >= max(8, height * 0.4)))
    if len(candidates) < 3:
        return None
    return candidates


def _row_vertical_lines(
    binary: np.ndarray,
    top: int,
    bottom: int,
    column_lines: list[int],
) -> set[int]:
    height, width = binary.shape
    top = max(0, top)
    bottom = min(height - 1, bottom)
    if bottom - top < 4:
        return set()
    row = binary[top : bottom + 1]
    row_height = row.shape[0]
    kernel_height = max(4, min(31, int(row_height * 0.5)))
    vertical = cv2.morphologyEx(
        row,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height)),
    )
    coverage = np.count_nonzero(vertical, axis=0)
    detected = _cluster(np.flatnonzero(coverage >= max(3, row_height * 0.35)))
    tolerance = max(2, round(width * 0.0015))
    return {
        index
        for index, x in enumerate(column_lines)
        if any(abs(x - candidate) <= tolerance for candidate in detected)
    }


def _replace_colspan(open_tag: str, attrs: str, span: int) -> str:
    if span == 1:
        new_attrs = _SPAN.sub("", attrs).rstrip()
    else:
        match = _SPAN.search(attrs)
        if match:
            new_attrs = attrs[: match.start()] + f'colspan="{span}"' + attrs[match.end() :]
        else:
            new_attrs = attrs + f' colspan="{span}"'
    tag_match = re.match(r"<(?P<tag>td|th)\b", open_tag, re.I)
    if not tag_match:
        return open_tag
    return f"<{tag_match.group('tag')}{new_attrs}>"


def _edge_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return directional contrast masks that work on colored table fills."""
    if image is None or image.size == 0 or image.ndim not in (2, 3):
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if min(gray.shape) < 16:
        return None

    # A global dark-pixel threshold treats tinted header/background fills as
    # ink. Local directional contrast isolates the ruled edges instead.
    horizontal = np.zeros_like(gray)
    vertical = np.zeros_like(gray)
    horizontal[1:-1, :] = (
        cv2.absdiff(gray[2:, :], gray[:-2, :]) >= 5
    ).astype(np.uint8) * 255
    vertical[:, 1:-1] = (
        cv2.absdiff(gray[:, 2:], gray[:, :-2]) >= 5
    ).astype(np.uint8) * 255
    return horizontal, vertical


def _header_band_height(horizontal_edges: np.ndarray) -> int | None:
    height, width = horizontal_edges.shape
    horizontal = cv2.morphologyEx(
        horizontal_edges,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(8, int(width * 0.35)), 1)
        ),
    )
    coverage = np.count_nonzero(horizontal, axis=1)
    lines = _cluster(np.flatnonzero(coverage >= width * 0.25))
    lines = [y for y in lines if y > 4]
    if len(lines) < 4:
        return None
    # Repeated body-row spacing estimates the height of the first header tier.
    gaps = np.diff(lines)
    gaps = gaps[(gaps >= 8) & (gaps <= height * 0.1)]
    if not len(gaps):
        return None
    return max(8, int(np.median(gaps)))


def _first_header_grid_spans(
    image: np.ndarray, table_html: str, column_lines: list[int]
) -> list[int] | None:
    rows = list(_ROW.finditer(table_html))
    if len(rows) < 2:
        return None
    cells = list(_CELL.finditer(rows[0].group()))
    if not cells or not any(
        _attr_int(_SPAN, cell.group("attrs")) > 1 for cell in cells
    ) or not any(_attr_int(_ROWSPAN, cell.group("attrs")) > 1 for cell in cells):
        return None

    masks = _edge_masks(image)
    if masks is None:
        return None
    horizontal_edges, vertical_edges = masks
    band_height = _header_band_height(horizontal_edges)
    if band_height is None:
        return None

    band = vertical_edges[: min(band_height, vertical_edges.shape[0])]
    present: list[int] = []
    tolerance = 2
    for index, x in enumerate(column_lines):
        x0, x1 = max(0, x - tolerance), min(band.shape[1], x + tolerance + 1)
        support = np.count_nonzero(band[:, x0:x1]) / max(1, band.shape[0] * (x1 - x0))
        if support >= 0.55:
            present.append(index)

    # In the top tier, merged group headers suppress their interior rules.
    # Each remaining boundary must correspond to one OCR header cell edge.
    if (
        len(present) != len(cells) + 1
        or present[0] != 0
        or present[-1] != len(column_lines) - 1
    ):
        return None
    spans = [right - left for left, right in zip(present, present[1:])]
    if not spans or any(span < 1 for span in spans):
        return None
    return spans


def _rowspan_intervals(row_html: str) -> list[tuple[int, int]]:
    occupied: set[int] = set()
    cursor = 0
    for cell in _CELL.finditer(row_html):
        attrs = cell.group("attrs")
        span = _attr_int(_SPAN, attrs)
        rowspan = _attr_int(_ROWSPAN, attrs)
        if rowspan > 1:
            occupied.update(range(cursor, cursor + span))
        cursor += span
    if not occupied:
        return []
    intervals: list[tuple[int, int]] = []
    first = previous = None
    for column in sorted(occupied):
        if first is None:
            first = previous = column
        elif column == previous + 1:
            previous = column
        else:
            intervals.append((first, previous + 1))
            first = previous = column
    intervals.append((first, previous + 1))
    return intervals


def _free_intervals(row_html: str) -> list[tuple[int, int]]:
    cells = list(_CELL.finditer(row_html))
    if not cells:
        return []
    occupied = _rowspan_intervals(row_html)
    width = sum(_attr_int(_SPAN, cell.group("attrs")) for cell in cells)
    cursor = 0
    intervals: list[tuple[int, int]] = []
    for start, end in occupied:
        if start > cursor:
            intervals.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < width:
        intervals.append((cursor, width))
    return intervals


def _add_missing_header_cells(
    parent_before: str, parent_after: str, child_row: str
) -> str | None:
    old_intervals = _free_intervals(parent_before)
    new_intervals = _free_intervals(parent_after)
    child_cells = list(_CELL.finditer(child_row))
    if (
        not old_intervals
        or len(old_intervals) != len(new_intervals)
        or not child_cells
        or any(_attr_int(_SPAN, cell.group("attrs")) != 1 for cell in child_cells)
        or any(_attr_int(_ROWSPAN, cell.group("attrs")) != 1 for cell in child_cells)
    ):
        return None

    source_counts = [end - start for start, end in old_intervals]
    # The final group consumes any recognized overflow caused by a short
    # colspan in the original top tier. Earlier boundaries are anchored by
    # rowspan cells and therefore have an unambiguous position.
    remaining = len(child_cells)
    for index in range(len(source_counts)):
        if index == len(source_counts) - 1:
            source_counts[index] = remaining
        else:
            source_counts[index] = min(source_counts[index], remaining)
            remaining -= source_counts[index]

    if sum(source_counts) != len(child_cells):
        return None
    targets = [end - start for start, end in new_intervals]
    if any(source > target for source, target in zip(source_counts, targets)):
        return None
    deficits = [target - source for source, target in zip(source_counts, targets)]
    if not any(deficits):
        return child_row

    rebuilt: list[str] = []
    cell_index = 0
    for count, deficit in zip(source_counts, deficits):
        segment = child_cells[cell_index : cell_index + count]
        split_cell: int | None = None
        split_parts: tuple[str, str] | None = None
        if deficit:
            if deficit != 1 or not segment:
                return None
            text_lengths = [
                len(html_lib.unescape(re.sub(r"<[^>]+>", "", cell.group("content"))).strip())
                for cell in segment
            ]
            longest = max(text_lengths)
            if (
                longest < 2
                or text_lengths.count(longest) != 1
                or longest < 1.5 * (sum(text_lengths) / len(text_lengths))
            ):
                # Without coordinates for each recognized heading, don't guess
                # where a missing subcell belongs within this group.
                return None
            split_cell = text_lengths.index(longest)
            raw_content = segment[split_cell].group("content")
            if re.search(r"<[^>]+>", raw_content):
                return None
            parts = re.split(r"\s+", raw_content.strip(), maxsplit=1)
            if len(parts) != 2 or not all(part.strip() for part in parts):
                # A blank would conceal the unresolved heading. Only split a
                # fused OCR cell when the source output has a clear separator.
                return None
            split_parts = (parts[0], parts[1])

        for index, cell in enumerate(segment):
            if split_cell == index and split_parts is not None:
                whole = cell.group()
                open_tag = whole[: whole.find(">") + 1]
                close_tag = whole[whole.rfind("<") :]
                rebuilt.append(open_tag + split_parts[0] + close_tag)
                rebuilt.append(open_tag + split_parts[1] + close_tag)
            else:
                rebuilt.append(cell.group())
        cell_index += count
    return child_row[: child_cells[0].start()] + "".join(rebuilt) + child_row[child_cells[-1].end() :]


def _align_multilevel_header(image: np.ndarray, table_html: str, column_lines: list[int]) -> str:
    rows = list(_ROW.finditer(table_html))
    if len(rows) < 2:
        return table_html
    parent_before = rows[0].group()
    cells = list(_CELL.finditer(parent_before))
    spans = _first_header_grid_spans(image, table_html, column_lines)
    if spans is None:
        return table_html

    rebuilt_cells: list[str] = []
    for cell, span in zip(cells, spans):
        whole = cell.group()
        open_tag = whole[: whole.find(">") + 1]
        open_tag = _replace_colspan(open_tag, cell.group("attrs"), span)
        rebuilt_cells.append(open_tag + cell.group("content") + whole[whole.rfind("<") :])
    parent_after = parent_before[: cells[0].start()] + "".join(rebuilt_cells) + parent_before[cells[-1].end() :]
    child_after = _add_missing_header_cells(parent_before, parent_after, rows[1].group())
    if child_after is None:
        child_after = rows[1].group()
    new_body = table_html
    for row_match, replacement in reversed(
        [(rows[0], parent_after), (rows[1], child_after)]
    ):
        new_body = new_body[: row_match.start()] + replacement + new_body[row_match.end() :]
    return new_body


def _align_row(
    row_html: str,
    top: int,
    bottom: int,
    binary: np.ndarray,
    column_lines: list[int],
    pending_rowspans: dict[int, int],
) -> tuple[str, int, int, dict[int, int]]:
    matches = list(_CELL.finditer(row_html))
    if not matches:
        return row_html, 0, 0, pending_rowspans

    present_lines = _row_vertical_lines(binary, top, bottom, column_lines)
    cursor = 0
    changed = 0
    rebuilt_cells: list[str] = []
    new_rowspans: dict[int, int] = {}

    for match in matches:
        while cursor in pending_rowspans:
            cursor += 1

        attrs = match.group("attrs")
        span = _attr_int(_SPAN, attrs)
        rowspan = _attr_int(_ROWSPAN, attrs)
        corrected_span = span

        # A visible vertical rule inside a model-merged cell disproves that
        # merge. Keep its text in the first physical cell; let following cells
        # fall back onto their grid columns.
        if rowspan == 1 and span > 1 and cursor + span < len(column_lines):
            for offset in range(1, span):
                boundary_index = cursor + offset
                if boundary_index in present_lines:
                    corrected_span = offset
                    break

        open_tag = match.group(0)[: match.group(0).find(">") + 1]
        close_tag = match.group(0)[match.group(0).rfind("<") :]
        if corrected_span != span:
            changed += 1
            open_tag = _replace_colspan(open_tag, attrs, corrected_span)
        rebuilt_cells.append(open_tag + match.group("content") + close_tag)

        if rowspan > 1:
            for column in range(cursor, cursor + corrected_span):
                new_rowspans[column] = rowspan - 1
        cursor += corrected_span

    row_width = max(cursor, max(pending_rowspans, default=-1) + 1)
    next_rowspans = {
        column: remaining - 1
        for column, remaining in pending_rowspans.items()
        if remaining > 1
    }
    next_rowspans.update(new_rowspans)
    if changed and row_width < len(column_lines) - 1:
        rebuilt_cells.extend("<td></td>" for _ in range(len(column_lines) - 1 - row_width))

    if not changed:
        return row_html, 0, row_width, next_rowspans
    rebuilt_row = row_html[: matches[0].start()] + "".join(rebuilt_cells) + row_html[matches[-1].end() :]
    return rebuilt_row, changed, row_width, next_rowspans


def _cell_has_text(match: re.Match) -> bool:
    content = match.group("content")
    if re.search(r"<(?:img|svg|math)\b", content, re.I):
        return True
    visible = re.sub(r"<[^>]+>", "", content)
    return bool(html_lib.unescape(visible).replace("\xa0", " ").strip())


def _rebuild_row_from_grid(
    row_html: str,
    top: int,
    bottom: int,
    gray: np.ndarray,
    binary: np.ndarray,
    column_lines: list[int],
    inherited_rowspans: dict[int, int],
) -> str | None:
    source_cells = list(_CELL.finditer(row_html))
    if not source_cells or any(
        _attr_int(_ROWSPAN, cell.group("attrs")) > 1 for cell in source_cells
    ):
        return None

    present = _row_vertical_lines(binary, top, bottom, column_lines)
    last_boundary = len(column_lines) - 1
    if column_lines[0] <= 4:
        present.add(0)
    if binary.shape[1] - 1 - column_lines[-1] <= 4:
        present.add(last_boundary)
    if 0 not in present or last_boundary not in present:
        return None
    boundaries = sorted(present)
    groups = list(zip(boundaries, boundaries[1:]))

    inherited = set(inherited_rowspans)
    visible_groups: list[tuple[int, int, bool]] = []
    row_height = max(1, bottom - top)
    y_margin = max(2, round(row_height * 0.18))
    for start, end in groups:
        columns = set(range(start, end))
        if columns & inherited:
            if not columns <= inherited:
                return None
            continue

        x0, x1 = column_lines[start], column_lines[end]
        x_margin = max(2, round((x1 - x0) * 0.08))
        y0 = min(gray.shape[0], top + y_margin)
        y1 = max(y0, min(gray.shape[0], bottom - y_margin))
        ix0 = min(gray.shape[1], x0 + x_margin)
        ix1 = max(ix0, min(gray.shape[1], x1 - x_margin))
        crop = gray[y0:y1, ix0:ix1]
        has_ink = bool(crop.size and np.count_nonzero(crop < 180) >= 4)
        visible_groups.append((start, end, has_ink))

    text_cells = [cell for cell in source_cells if _cell_has_text(cell)]
    ink_group_count = sum(has_ink for _, _, has_ink in visible_groups)
    if ink_group_count == len(text_cells) + 1 and visible_groups:
        first_start, first_end, first_has_ink = visible_groups[0]
        widths = [
            column_lines[end] - column_lines[start]
            for start, end, _ in visible_groups
        ]
        typical_width = float(np.median(widths[1:] or widths))
        first_cell_is_empty = not _cell_has_text(source_cells[0])
        if (
            first_has_ink
            and first_cell_is_empty
            and widths[0] <= typical_width * 0.45
        ):
            # Selection boxes and similar marks in a narrow leading column
            # can have pixels even when the recognizer emits an empty cell.
            # Keep that grid slot empty and map recognized text to later cells.
            visible_groups[0] = (first_start, first_end, False)
            ink_group_count -= 1

    if ink_group_count != len(text_cells):
        return None

    rebuilt: list[str] = []
    text_index = 0
    for start, end, has_ink in visible_groups:
        span = end - start
        if has_ink:
            template = text_cells[text_index]
            text_index += 1
            whole = template.group()
            open_tag = whole[: whole.find(">") + 1]
            open_tag = _replace_colspan(open_tag, template.group("attrs"), span)
            close_tag = whole[whole.rfind("<") :]
            rebuilt.append(open_tag + template.group("content") + close_tag)
        elif span == 1:
            rebuilt.append("<td></td>")
        else:
            rebuilt.append(f'<td colspan="{span}"></td>')

    return row_html[: source_cells[0].start()] + "".join(rebuilt) + row_html[source_cells[-1].end() :]


def align_html_table_to_grid(image: np.ndarray, html: str) -> str:
    """Correct HTML colspan only when source grid lines contradict the merge.

    Alignment is skipped unless every model row maps one-to-one to a detected
    horizontal table band and the image contains a stable vertical grid. This
    keeps uncertain or unruled tables unchanged.
    """
    tables = list(_TABLE.finditer(html or ""))
    if len(tables) != 1:
        return html
    match = _TABLE_BODY.fullmatch(tables[0].group())
    if not match:
        return html

    table_html = match.group()
    rows = list(_ROW.finditer(match.group("body")))
    if not rows:
        return html
    binary = _grid_binary(image)
    edge_masks = _edge_masks(image)
    if binary is None and edge_masks is None:
        return html

    # Prefer a directional edge grid when its line count agrees with the
    # repeated body-row structure. This avoids treating colored cell fills as
    # ink while keeping the existing threshold detector for ordinary scans.
    expected_columns = max(
        (
            sum(_attr_int(_SPAN, cell.group("attrs")) for cell in _CELL.finditer(row.group()))
            for row in rows[2:]
        ),
        default=0,
    )
    column_lines = _column_grid_lines(binary) if binary is not None else None
    edge_vertical = edge_masks[1] if edge_masks is not None else None
    edge_columns = _column_grid_lines(edge_vertical) if edge_vertical is not None else None
    if edge_columns is not None and (
        column_lines is None
        or (expected_columns and len(edge_columns) == expected_columns + 1
            and len(column_lines) != expected_columns + 1)
    ):
        column_lines = edge_columns

    if column_lines is None:
        return html

    aligned_table_html = _align_multilevel_header(image, table_html, column_lines)
    if aligned_table_html != table_html:
        html = html[: tables[0].start()] + aligned_table_html + html[tables[0].end() :]
        table_html = aligned_table_html
        match = _TABLE_BODY.fullmatch(table_html)
        if not match:
            return html
        rows = list(_ROW.finditer(match.group("body")))

    if binary is None:
        return html
    row_lines = _row_grid_lines(binary, len(rows))
    if row_lines is None or column_lines is None:
        logger.debug(
            "Skipped table grid alignment: detected rows/columns do not map reliably to model HTML."
        )
        return html

    rebuilt_rows: list[str] = []
    repaired_spans = 0
    rebuilt_row_count = 0
    previous_y = 0
    total_columns = len(column_lines) - 1
    pending_rowspans: dict[int, int] = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    for row_match, bottom_y in zip(rows, row_lines):
        original_row = row_match.group()
        inherited_before = dict(pending_rowspans)
        rebuilt, changed, width, pending_rowspans = _align_row(
            original_row,
            previous_y,
            bottom_y,
            binary,
            column_lines,
            pending_rowspans,
        )
        if changed and width > total_columns:
            logger.debug("Skipped table grid alignment for an over-wide row.")
            return html

        if not inherited_before and not any(
            _attr_int(_ROWSPAN, cell.group("attrs")) > 1
            for cell in _CELL.finditer(rebuilt)
        ):
            # Reconstruct only rows whose visible text maps one-to-one to ink
            # in the detected grid cells.
            aligned_row = _rebuild_row_from_grid(
                rebuilt,
                previous_y,
                bottom_y,
                gray,
                binary,
                column_lines,
                inherited_before,
            )
            if aligned_row is not None:
                if aligned_row != rebuilt:
                    rebuilt_row_count += 1
                rebuilt = aligned_row

        rebuilt_rows.append(rebuilt)
        repaired_spans += changed
        previous_y = bottom_y

    if not repaired_spans and not rebuilt_row_count:
        return html

    new_body = match.group("body")
    for row_match, rebuilt in reversed(list(zip(rows, rebuilt_rows))):
        new_body = new_body[: row_match.start()] + rebuilt + new_body[row_match.end() :]
    logger.info(
        "Applied table grid alignment: repaired %s span(s), rebuilt %s row(s).",
        repaired_spans,
        rebuilt_row_count,
    )
    return html[: tables[0].start()] + match.group("open") + new_body + match.group("close") + html[tables[0].end() :]
