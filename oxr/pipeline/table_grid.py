"""Recover physical cells of a ruled table before transcribing their contents."""

from dataclasses import dataclass
import html
import re
from collections.abc import Iterator

import cv2
import numpy as np

from oxr.utils.table import _TableParser, strip_table_code_fence


@dataclass(frozen=True)
class GridCell:
    row: int
    column: int
    rowspan: int
    colspan: int
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class TableGrid:
    rows: int
    columns: int
    header_rows: int
    cells: tuple[GridCell, ...]


def _line_positions(coverage: np.ndarray, threshold: float) -> list[int]:
    hits = np.flatnonzero(coverage >= threshold)
    groups = np.split(hits, np.flatnonzero(np.diff(hits) > 2) + 1)
    return [round(float(group.mean())) for group in groups if 0 < len(group) <= 9]


def detect_table_grid(image: np.ndarray) -> TableGrid | None:
    """Use long strokes and their local gaps to find rectangular merged cells."""
    height, width = image.shape[:2]
    if min(height, width) < 48:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Local contrast also finds thin rules on colored or shaded cells.
    contrast = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((7, 7), np.uint8))
    ink = (contrast >= 6).astype(np.uint8)
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(16, height // 45), 1), np.uint8))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(16, width // 65)), np.uint8))
    xs = _line_positions(vertical.mean(axis=0), 0.55)
    strong_horizontal = cv2.morphologyEx(
        (contrast >= 10).astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, max(16, width // 65)), np.uint8),
    )
    ys = _line_positions(strong_horizontal.mean(axis=1), 0.09)
    horizontal_support = horizontal.mean(axis=1)
    rows, columns = len(ys) - 1, len(xs) - 1
    if rows < 2 or columns < 2 or rows * columns > 8000:
        return None
    if (
        max(xs[0], width - 1 - xs[-1]) > max(6, min(24, width * 0.025))
        or max(ys[0], height - 1 - ys[-1]) > max(6, min(24, height * 0.025))
        or min(np.diff(xs)) < 9 or min(np.diff(ys)) < 9
    ):
        return None

    def separator(
        mask: np.ndarray, start: int, end: int, position: int, *, long_rule: bool = False,
    ) -> bool | None:
        sample = mask[start + 2:end - 1, max(0, position - 2):position + 3]
        along = sample.any(axis=1)
        coverage = float(along.mean())
        edge = max(2, min(7, len(along) // 3))
        left, right = along[:edge].mean(), along[-edge:].mean()
        if coverage >= 0.62 and min(left, right) >= 0.5 or (
            long_rule and coverage >= 0.76 and max(left, right) >= 0.5
        ):
            return True
        if coverage <= 0.20 or max(left, right) <= 0.30:
            return False
        return None

    parents = list(range(rows * columns))
    separated: list[tuple[int, int]] = []

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    # Join adjacent slots only where the separating rule is confidently absent.
    for r in range(rows):
        for c in range(columns + 1):
            present = separator(vertical, ys[r], ys[r + 1], xs[c])
            if present is None or (not present and c in (0, columns)):
                return None
            if not present:
                parents[root(r * columns + c)] = root(r * columns + c - 1)
            elif 0 < c < columns:
                separated.append((r * columns + c - 1, r * columns + c))
    for r in range(rows + 1):
        support = horizontal_support[max(0, ys[r] - 2):ys[r] + 3].max()
        clipped_outer = (r == 0 and ys[r] <= 1 or r == rows and ys[r] >= height - 2) and support >= 0.85
        for c in range(columns):
            # A screenshot can end on a well-supported rule with one
            # clipped segment. The image boundary closes that last cell.
            present = True if clipped_outer else separator(
                horizontal.T, xs[c], xs[c + 1], ys[r], long_rule=support >= 0.85,
            )
            if present is None or (not present and r in (0, rows)):
                return None
            if not present:
                parents[root(r * columns + c)] = root((r - 1) * columns + c)
            elif 0 < r < rows:
                separated.append(((r - 1) * columns + c, r * columns + c))

    if any(root(a) == root(b) for a, b in separated):
        return None
    groups: dict[int, list[tuple[int, int]]] = {}
    for index in range(rows * columns):
        groups.setdefault(root(index), []).append(divmod(index, columns))
    cells = []
    for slots in groups.values():
        rr, cc = zip(*slots)
        r, c, bottom, right = min(rr), min(cc), max(rr) + 1, max(cc) + 1
        if (bottom - r) * (right - c) != len(slots):
            return None
        cells.append(GridCell(r, c, bottom - r, right - c, (xs[c], ys[r], xs[right], ys[bottom])))
    cells.sort(key=lambda cell: (cell.row, cell.column))

    top = [cell for cell in cells if cell.row == 0]
    header_rows = max(cell.rowspan for cell in top)
    if (
        not 1 < header_rows < rows
        or not any(cell.colspan > 1 and cell.rowspan == 1 for cell in top)
        or any(cell.row < header_rows < cell.row + cell.rowspan for cell in cells)
    ):
        header_rows = 0
    return TableGrid(rows, columns, header_rows, tuple(cells))


def render_table_grid(grid: TableGrid, values: dict[int, str]) -> str:
    """Build HTML from image positions; never shift values to fill missing IDs."""
    if values.keys() != set(range(1, len(grid.cells) + 1)):
        raise ValueError("Incomplete table cell transcription.")
    rows: list[list[str]] = [[] for _ in range(grid.rows)]
    for index, cell in enumerate(grid.cells, 1):
        tag = "th" if cell.row < grid.header_rows else "td"
        attributes = ""
        if cell.rowspan > 1:
            attributes += f' rowspan="{cell.rowspan}"'
        if cell.colspan > 1:
            attributes += f' colspan="{cell.colspan}"'
        rows[cell.row].append(f"<{tag}{attributes}>{html.escape(values[index])}</{tag}>")
    markup = ["<tr>" + "".join(row) + "</tr>" for row in rows]
    header = "<thead>\n" + "\n".join(markup[:grid.header_rows]) + "\n</thead>\n" if grid.header_rows else ""
    return "<table>\n" + header + "<tbody>\n" + "\n".join(markup[grid.header_rows:]) + "\n</tbody>\n</table>"


def parse_indexed_cells(text: str, ids: tuple[int, ...]) -> dict[int, str]:
    """Recover ID/value pairs from the model's native table output."""
    expected = set(ids)
    # OCR may change letter case or insert a leading zero; the numeric ID
    # still has to match an expected cell exactly.
    marker = re.compile(r"K(\d{1,8})", re.I | re.ASCII)
    parser = _TableParser()
    parser.feed(strip_table_code_fence(text))
    values: dict[int, str] = {}
    seen: set[int] = set()
    for row in parser.rows:
        for position, label in enumerate(row):
            match = marker.fullmatch(label.text)
            if not match:
                continue
            index = int(match[1])
            if index not in expected or index in seen:
                raise ValueError("Unknown or duplicate table cell ID.")
            seen.add(index)
            if label.rowspan != 1 or label.colspan != 1 or position + 1 >= len(row):
                continue
            value = row[position + 1]
            if value.rowspan == value.colspan == 1 and not marker.fullmatch(value.text):
                values[index] = value.text
    return values


def make_cell_batches(
    image: np.ndarray,
    grid: TableGrid,
    ids: tuple[int, ...],
    *,
    enlarge_ids: tuple[int, ...] = (),
) -> Iterator[tuple[tuple[int, ...], np.ndarray]]:
    """Pack isolated, labeled cell crops into bounded-size vision requests."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    emphasized = set(enlarge_ids)
    pairs = min(5, (len(ids) + 1) // 2)
    pair_width, id_width, height_limit = 320, 76, 1400
    if not pairs:
        return
    width = pairs * pair_width
    canvas = np.full((height_limit + 1, width + 1), 255, np.uint8)
    y = 0
    batch: list[int] = []
    for start in range(0, len(ids), pairs):
        row_ids = ids[start:start + pairs]
        crops = []
        for index in row_ids:
            x1, y1, x2, y2 = grid.cells[index - 1].box
            crop = gray[y1 + 2:y2 - 1, x1 + 2:x2 - 1]
            h, w = crop.shape
            max_scale = 3.4 if index in emphasized else 1.65
            max_height = 300 if index in emphasized else 220
            scale = min(max_scale, (pair_width - id_width - 12) / w, max_height / h)
            crops.append(cv2.resize(
                crop, (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
            ))
        row_height = max(34, max(crop.shape[0] for crop in crops) + 8)
        if batch and (y + row_height > height_limit or len(batch) + len(row_ids) > 90):
            yield tuple(batch), canvas[:y + 1].copy()
            canvas.fill(255)
            y, batch = 0, []
        bottom = y + row_height
        cv2.line(canvas, (0, y), (width, y), 0, 2)
        cv2.line(canvas, (0, bottom), (width, bottom), 0, 2)
        for col in range(pairs):
            x = col * pair_width
            cv2.line(canvas, (x, y), (x, bottom), 0, 2)
            cv2.line(canvas, (x + id_width, y), (x + id_width, bottom), 0, 2)
            if col >= len(crops):
                continue
            crop = crops[col]
            h, w = crop.shape
            top = y + (row_height - h) // 2
            canvas[top:top + h, x + id_width + 6:x + id_width + 6 + w] = crop
            cv2.putText(canvas, f"K{row_ids[col]:04}", (x + 5, y + row_height // 2 + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, .58, 0, 2, cv2.LINE_AA)
        cv2.line(canvas, (width, y), (width, bottom), 0, 2)
        y = bottom
        batch.extend(row_ids)
    if batch:
        yield tuple(batch), canvas[:y + 1].copy()
