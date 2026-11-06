"""Conservative generic stitching for paragraph and table continuations."""

import logging
import re

from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.table_segmentation import (
    combine_table_chunks,
    extract_table_header_schema,
    table_row_widths,
)


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


def stitch_markdown_pages(pages: list[list[Block]]) -> list[list[Block]]:
    """Merge only high-confidence page-edge continuations in a copy of blocks."""
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

        if _continues_paragraph(previous, following):
            previous.content = previous.content.rstrip() + " " + following.content.lstrip()
            current_page.remove(following)
            logger.info("Stitched a continued paragraph across pages %d and %d.", page_index, page_index + 1)

    return result
