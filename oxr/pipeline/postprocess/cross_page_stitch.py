"""Conservative generic stitching for paragraph and table continuations."""

import logging
import re

from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.table_segmentation import (
    combine_table_chunks,
    extract_table_header_schema,
    table_row_widths,
)
from oxr.pipeline.postprocess.exact_text_dedupe import dedupe_repeated_text_regions


logger = logging.getLogger(__name__)
_END_PUNCTUATION = set("。！？.!?；;:：”’\"'）)]}】》」』")
_LIST_START = re.compile(r"^\s*(?:[-*+]\s|\d+(?:\.\d+)*[、.)）]\s*|[一二三四五六七八九十]+[、.])")


def _edge_block(blocks: list[Block], reverse: bool = False) -> Block | None:
    candidates = reversed(blocks) if reverse else iter(blocks)
    return next((block for block in candidates if block.content.strip()), None)


def _continues_paragraph(previous: Block, following: Block) -> bool:
    left = previous.content.rstrip()
    right = following.content.lstrip()
    if not left or not right or left[-1] in _END_PUNCTUATION:
        return False
    if _LIST_START.match(right) or following.type == BlockType.TITLE:
        return False
    # A lowercase Latin continuation or Chinese text without a sentence stop is
    # a useful generic signal; avoid joining isolated short labels.
    if len(left) < 24 or len(right) < 12:
        return False
    return previous.type == following.type == BlockType.TEXT


def stitch_markdown_pages(
    pages: list[list[Block]], page_heights: list[int] | None = None,
    *, dedupe_text: bool = False,
) -> list[list[Block]]:
    """Clean exact text repeats and stitch continuations in copied page blocks.

    Text deduplication is limited to blocks whose boxes reach the bottom and
    top bands of adjacent pages. Table continuation remains on its existing
    schema and row-width path.
    """
    result = [[Block(**vars(block)) for block in page] for page in pages]
    for page_index in range(1, len(result)):
        previous_page, current_page = result[page_index - 1], result[page_index]
        previous = _edge_block(previous_page, reverse=True)
        following = _edge_block(current_page)
        if previous is None or following is None:
            continue

        if previous.type == following.type == BlockType.TABLE:
            left_schema = extract_table_header_schema(previous.content)
            right_schema = extract_table_header_schema(following.content)
            left_widths, right_widths = table_row_widths(previous.content), table_row_widths(following.content)
            if left_schema and left_schema == right_schema and left_widths and right_widths and max(left_widths) == max(right_widths):
                try:
                    previous.content = combine_table_chunks([previous.content, following.content])
                except ValueError:
                    continue
                previous.format = ContentFormat.HTML
                current_page.remove(following)
                logger.info("Stitched matching table continuation across pages %d and %d.", page_index, page_index + 1)
                continue

        if dedupe_text and page_heights and len(page_heights) == len(result):
            previous_height = page_heights[page_index - 1]
            current_height = page_heights[page_index]
            has_bottom_anchor = any(
                block.type == BlockType.TEXT
                and block.bbox[3] >= 0.85 * previous_height
                for block in previous_page
            )
            has_top_anchor = any(
                block.type == BlockType.TEXT
                and block.bbox[1] <= 0.15 * current_height
                for block in current_page
            )
            if previous_height > 0 and current_height > 0 and has_bottom_anchor and has_top_anchor:
                previous_edge = [
                    block for block in previous_page
                    if block.type == BlockType.TEXT
                    and block.bbox[3] >= 0.4 * previous_height
                ]
                current_edge = [
                    block for block in current_page
                    if block.type == BlockType.TEXT
                    and block.bbox[1] <= 0.2 * current_height
                ]
                for following in current_edge:
                    updated = following.content
                    for previous in previous_edge:
                        updated = dedupe_repeated_text_regions(previous.content, updated)
                        if not updated:
                            break
                    if updated != following.content:
                        following.content = updated
                        if not updated.strip():
                            current_page.remove(following)
                        logger.info(
                            "Removed exact repeated text at page boundary %d/%d.",
                            page_index, page_index + 1,
                        )

        if following not in current_page:
            continue

        if _continues_paragraph(previous, following):
            previous.content = previous.content.rstrip() + " " + following.content.lstrip()
            current_page.remove(following)
            logger.info("Stitched a continued paragraph across pages %d and %d.", page_index, page_index + 1)

    return result
