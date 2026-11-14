import cv2
import numpy as np


def _align(image, html):
    try:
        from oxr.utils.table_grid_alignment import align_html_table_to_grid
    except ImportError:
        return None
    return align_html_table_to_grid(image, html)


def _grid_image(columns: int, *, row_height: int = 50) -> np.ndarray:
    image = np.full((row_height + 1, columns * 100 + 1, 3), 255, dtype=np.uint8)
    for x in range(0, columns * 100 + 1, 100):
        cv2.line(image, (x, 0), (x, row_height), (130, 130, 130), 1)
    cv2.line(image, (0, 0), (columns * 100, 0), (130, 130, 130), 1)
    cv2.line(image, (0, row_height), (columns * 100, row_height), (130, 130, 130), 1)
    return image


def test_splits_model_colspan_when_source_grid_has_a_cell_boundary():
    image = _grid_image(3)
    html = "<table><tr><td>A</td><td colspan='2'>B</td></tr></table>"

    aligned = _align(image, html)

    assert aligned is not None
    assert "colspan='2'" not in aligned
    assert "<td>B</td><td></td>" in aligned


def test_keeps_colspan_when_source_grid_has_no_internal_boundary():
    image = np.full((101, 301, 3), 255, dtype=np.uint8)
    for x in (0, 100, 200, 300):
        cv2.line(image, (x, 0), (x, 50), (130, 130, 130), 1)
    for x in (0, 100, 300):
        cv2.line(image, (x, 50), (x, 100), (130, 130, 130), 1)
    for y in (0, 50, 100):
        cv2.line(image, (0, y), (300, y), (130, 130, 130), 1)
    html = (
        "<table><tr><td>A</td><td>B</td><td>C</td></tr>"
        "<tr><td>A</td><td colspan='2'>merged</td></tr></table>"
    )

    aligned = _align(image, html)

    assert aligned is not None
    assert "colspan='2'" in aligned


def test_restores_omitted_blank_cell_from_visible_text_positions():
    image = _grid_image(3)
    cv2.putText(image, "A", (25, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(image, "B", (225, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    html = "<table><tr><td>A</td><td>B</td></tr></table>"

    aligned = _align(image, html)

    assert aligned is not None
    assert "<td>A</td><td></td><td>B</td>" in aligned


def test_leaves_table_unchanged_when_rows_do_not_map_to_image_grid():
    image = _grid_image(3)
    html = "<table><tr><td>A</td></tr><tr><td>B</td></tr></table>"

    assert _align(image, html) == html


def test_ignores_extra_bottom_rule_when_matching_rows_to_image():
    image = np.full((60, 301, 3), 255, dtype=np.uint8)
    for x in (0, 100, 200, 300):
        cv2.line(image, (x, 0), (x, 50), (130, 130, 130), 1)
    for y in (0, 50, 57):
        cv2.line(image, (0, y), (300, y), (130, 130, 130), 1)
    cv2.putText(image, "A", (25, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    html = "<table><tr><td>A</td></tr></table>"

    aligned = _align(image, html)

    assert aligned is not None
    assert "<td>A</td><td></td><td></td>" in aligned
