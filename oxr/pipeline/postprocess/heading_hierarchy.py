"""Infer Markdown heading depth from numbering and title geometry."""

import re

from oxr.pipeline.models import Block, BlockType


_NUMBERED_DEPTH = re.compile(r"^\s*(\d+(?:\.\d+){0,5})(?:[.、)）\s])")
_CHAPTER = re.compile(r"^\s*第[一二三四五六七八九十百零〇\d]+[章节篇部]")
_CHINESE_SECTION = re.compile(r"^\s*[一二三四五六七八九十]+[、.]")
_CHINESE_SUBSECTION = re.compile(r"^\s*[（(][一二三四五六七八九十\d]+[）)]")


def assign_heading_levels(pages: list[list[Block]]) -> None:
    """Set title levels using explicit numbering first, then relative title size."""
    titles = [block for page in pages for block in page if block.type == BlockType.TITLE]
    if not titles:
        return

    heights = sorted({max(1, round(block.bbox[3] - block.bbox[1])) for block in titles}, reverse=True)
    height_levels: dict[int, int] = {}
    previous_height = None
    current_level = 0
    for height in heights:
        if previous_height is None or height < previous_height * 0.88:
            current_level = min(6, current_level + 1)
        height_levels[height] = max(1, current_level)
        previous_height = height

    for block in titles:
        text = block.content.strip()
        numbered = _NUMBERED_DEPTH.match(text)
        if numbered:
            block.heading_level = min(6, numbered.group(1).count(".") + 1)
        elif _CHAPTER.match(text):
            block.heading_level = 1
        elif _CHINESE_SECTION.match(text):
            block.heading_level = 2
        elif _CHINESE_SUBSECTION.match(text):
            block.heading_level = 3
        else:
            height = max(1, round(block.bbox[3] - block.bbox[1]))
            block.heading_level = height_levels[height]
