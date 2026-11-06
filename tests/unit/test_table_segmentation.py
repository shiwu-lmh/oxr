import numpy as np

from oxr.pipeline.table_segmentation import (
    TABLE_CHUNK_MAX_HEIGHT,
    expected_table_columns,
    is_balanced_table_html,
    repair_table_html_prefix,
    split_table_image,
    table_row_widths,
)


def test_split_table_image_keeps_a_table_intact_when_no_safe_rule_exists():
    image = np.full((TABLE_CHUNK_MAX_HEIGHT + 1, 80, 3), 255, dtype=np.uint8)
    image[TABLE_CHUNK_MAX_HEIGHT, :, :] = 0

    parts = split_table_image(image)

    assert len(parts) == 1
    assert parts[0].shape == image.shape


def test_repair_table_html_with_missing_leading_container_tags():
    raw = (
        "Applicant</td></tr>"
        "<tr><td colspan='2'>Privacy</td></tr>"
        "<tr><td colspan='2'>Contact</td></tr></table>"
    )

    repaired = repair_table_html_prefix(raw)

    assert repaired is not None
    assert is_balanced_table_html(repaired)
    assert table_row_widths(repaired) == [1, 2, 2]
    assert expected_table_columns(repaired) == 2


def test_escaped_table_markup_is_not_accepted_as_a_complete_table():
    escaped = (
        "<table><tbody><tr><td>"
        "&lt;table&gt;&lt;tr&gt;&lt;td&gt;all rows&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;"
        "</td></tr></tbody></table>"
    )

    assert not is_balanced_table_html(escaped)
