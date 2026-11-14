"""Content-preserving image preparation shared by every table OCR request."""

from __future__ import annotations

import re

import cv2
import numpy as np


_TARGET_TEXT_HEIGHT = 22
_MAX_UPSCALE = 2.0
_MAX_PREPARED_DIMENSION = 4096
_MIN_BORDER = 8
_MAX_BORDER = 28
_TABLE_CHUNK_ROWS = 8
_MIN_SPLIT_ROWS = 18
_HTML_TABLE = re.compile(r"<table\b[^>]*>(.*?)</table\s*>", re.IGNORECASE | re.DOTALL)
_HTML_ROW = re.compile(r"<tr\b[^>]*>.*?</tr\s*>", re.IGNORECASE | re.DOTALL)


def _estimate_glyph_height(gray: np.ndarray) -> float | None:
    """Estimate printed glyph height from connected dark components."""
    height, width = gray.shape[:2]
    if height < 8 or width < 8:
        return None

    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates: list[int] = []
    max_component_height = max(20, int(height * 0.2))
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        del x, y
        if (
            5 <= component_height <= max_component_height
            and 2 <= component_width <= max(24, component_height * 4)
            and 5 <= area <= max(3000, component_height * component_width)
        ):
            candidates.append(int(component_height))

    if len(candidates) < 8:
        return None
    return float(np.median(candidates))


def _enhance_luminance(image: np.ndarray) -> np.ndarray:
    """Apply mild local contrast enhancement without binarizing table lines."""
    if image.ndim == 2:
        return cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lightness, a_channel, b_channel = cv2.split(lab)
    enhanced = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(
        cv2.merge((enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR
    )


def prepare_table_image(image: np.ndarray) -> np.ndarray:
    """Enhance a table crop while preserving all rows, columns, and cell content.

    The operation is adaptive: tiny text is enlarged toward a readable pixel
    height, already large text is left at its original size, and a white border
    prevents characters or table rules touching the crop edge from being cut.
    No grid lines or image regions are removed.
    """
    if image is None or image.size == 0:
        raise ValueError("A non-empty table image is required.")
    if image.ndim not in (2, 3):
        raise ValueError("Table image must have two or three dimensions.")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("Table image must be grayscale or have three BGR channels.")

    prepared = _enhance_luminance(image)
    gray = prepared if prepared.ndim == 2 else cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
    glyph_height = _estimate_glyph_height(gray)
    scale = 1.0
    if glyph_height is not None and glyph_height < _TARGET_TEXT_HEIGHT:
        scale = min(_MAX_UPSCALE, _TARGET_TEXT_HEIGHT / glyph_height)
        max_dimension = max(prepared.shape[:2])
        if max_dimension * scale > _MAX_PREPARED_DIMENSION:
            scale = max(1.0, _MAX_PREPARED_DIMENSION / max_dimension)

    if scale > 1.01:
        height, width = prepared.shape[:2]
        prepared = cv2.resize(
            prepared,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    border = min(
        _MAX_BORDER,
        max(_MIN_BORDER, round((glyph_height or 16) * scale * 0.6)),
    )
    return cv2.copyMakeBorder(
        prepared,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=255 if prepared.ndim == 2 else (255, 255, 255),
    )


def split_long_table_image(image: np.ndarray) -> list[np.ndarray]:
    """Split clearly ruled, long tables into non-overlapping row-aligned crops.

    The first crop keeps the table heading; later crops begin and end on detected
    horizontal rules. Tables without enough reliable rules remain unchanged.
    """
    if image is None or image.size == 0:
        raise ValueError("A non-empty table image is required.")

    height, width = image.shape[:2]
    if height < 240 or width < 32:
        return [image]

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background_level = float(np.percentile(gray, 90))
    if background_level < 200:
        return [image]

    threshold = int(np.clip(background_level - 10, 200, 250))
    binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    kernel_width = max(8, int(width * 0.35))
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)),
    )
    row_coverage = np.count_nonzero(horizontal, axis=1)
    line_rows = np.flatnonzero(row_coverage >= max(20, width * 0.25))
    if not len(line_rows):
        return [image]

    lines: list[int] = []
    cluster: list[int] = [int(line_rows[0])]
    for row in line_rows[1:]:
        row = int(row)
        if row <= cluster[-1] + 2:
            cluster.append(row)
        else:
            lines.append(round(sum(cluster) / len(cluster)))
            cluster = [row]
    lines.append(round(sum(cluster) / len(cluster)))

    # Ignore border duplicates and rules too close to represent a row boundary.
    boundaries: list[int] = []
    for line in lines:
        if line < 4 or line >= height - 6:
            continue
        if not boundaries or line - boundaries[-1] >= 10:
            boundaries.append(line)
        elif row_coverage[line] > row_coverage[boundaries[-1]]:
            boundaries[-1] = line

    if len(boundaries) < _MIN_SPLIT_ROWS + 2:
        return [image]

    # Skip the top heading area when choosing body chunk boundaries. It remains
    # in the first image; later images contain only their own source rows.
    anchor_end = boundaries[2]
    body_boundaries = boundaries[3:] + [height]
    if anchor_end < 24 or len(body_boundaries) < _MIN_SPLIT_ROWS:
        return [image]

    ranges: list[tuple[int, int]] = []
    start = anchor_end
    for index in range(_TABLE_CHUNK_ROWS - 1, len(body_boundaries), _TABLE_CHUNK_ROWS):
        end = body_boundaries[index]
        if end - start >= 16:
            ranges.append((start, end))
            start = end
    if height - start >= 16:
        ranges.append((start, height))
    if len(ranges) < 2:
        return [image]

    chunks = [image[: ranges[0][1]]]
    for top, bottom in ranges[1:]:
        chunks.append(image[top:bottom])
    return chunks


def merge_table_chunk_outputs(outputs: list[str]) -> tuple[str, int] | None:
    """Merge non-overlapping chunk rows in source order."""
    parsed: list[tuple[str, list[str]]] = []
    for output in outputs:
        match = _HTML_TABLE.search(output)
        if match is None:
            return None
        opening_end = output.find(">", match.start()) + 1
        opening = output[match.start() : opening_end]
        rows = _HTML_ROW.findall(match.group(1))
        if not rows:
            return None
        parsed.append((opening, rows))

    merged_rows = list(parsed[0][1])
    for _, rows in parsed[1:]:
        merged_rows.extend(rows)
    return parsed[0][0] + "".join(merged_rows) + "</table>", len(merged_rows)
