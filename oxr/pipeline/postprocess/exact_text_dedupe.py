"""Remove exact cross-category duplicates of recognized textual blocks."""

import math
import re
from collections import Counter, defaultdict

from oxr.pipeline.models import Block, BlockType


_TEXTUAL_TYPES = frozenset({BlockType.TEXT, BlockType.CAPTION, BlockType.TITLE})
_CONTAINED_TEXT_TYPES = frozenset({BlockType.TEXT})


def _normalized_text(content: str) -> str:
    return re.sub(r"\s+", "", content)


def _inside(child: Block, parent: Block) -> bool:
    x1, y1, x2, y2 = child.bbox
    px1, py1, px2, py2 = parent.bbox
    child_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if child_area <= 0:
        return False
    intersection = max(0.0, min(x2, px2) - max(x1, px1)) * max(
        0.0, min(y2, py2) - max(y1, py1)
    )
    return intersection / child_area >= 0.95


def dedupe_contained_text_blocks(blocks: list[Block]) -> list[Block]:
    """Drop a text block only when its full text is repeated inside a larger box."""
    removed: set[int] = set()
    for child_pos, child in enumerate(blocks):
        if child.type not in _CONTAINED_TEXT_TYPES or not isinstance(child.content, str):
            continue
        child_text = _normalized_text(child.content)
        if len(child_text) < 5:
            continue
        for parent_pos, parent in enumerate(blocks):
            if child_pos == parent_pos or parent.type not in _CONTAINED_TEXT_TYPES:
                continue
            if parent_pos in removed or not isinstance(parent.content, str):
                continue
            if len(_normalized_text(parent.content)) <= len(child_text):
                continue
            if not _inside(child, parent):
                continue
            if child.format != parent.format or child.text_before or child.text_after:
                continue
            if child.generated_role or parent.generated_role:
                continue
            if child_text in _normalized_text(parent.content):
                removed.add(child_pos)
                break
    if not removed:
        return blocks
    return [block for position, block in enumerate(blocks) if position not in removed]


def dedupe_repeated_text_regions(previous: str, following: str, *, min_chars: int = 30) -> str:
    """Remove exact full sentence/paragraph repeats from the start-side text.

    Matching complete sentence units avoids fuzzy similarity decisions that
    could collapse clinical values such as medication doses or dates.
    """
    if not previous or not following:
        return following
    previous_numbers = tuple(re.findall(r"\d+(?:[./:-]\d+)*", previous))
    following_numbers = tuple(re.findall(r"\d+(?:[./:-]\d+)*", following))
    if previous_numbers != following_numbers:
        return following
    previous_units = Counter(
        _normalized_text(match.group(0))
        for match in re.finditer(r"[^。！？!?；;\n]+[。！？!?；;]?", previous)
    )
    matches = list(re.finditer(r"[^。！？!?；;\n]+[。！？!?；;]?", following))
    remove_spans = []
    for match in matches:
        normalized = _normalized_text(match.group(0))
        if len(normalized) < min_chars or previous_units[normalized] < 1:
            continue
        previous_units[normalized] -= 1
        remove_spans.append(match.span())
    if not remove_spans:
        return following
    pieces = []
    cursor = 0
    for start, end in remove_spans:
        pieces.append(following[cursor:start])
        cursor = end
    pieces.append(following[cursor:])
    return "".join(pieces).strip()


def dedupe_exact_textual_blocks(blocks: list[Block]) -> list[Block]:
    """Keep the highest-confidence block without changing its fields or order.

    Require identical geometry, original text and assembly fields, one block
    per category, and finite confidence for every member of the group.
    """
    groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for position, block in enumerate(blocks):
        if block.type not in _TEXTUAL_TYPES:
            continue
        if not isinstance(block.content, str) or not block.content.strip():
            continue
        box = block.bbox
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in box):
            continue
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        key = (
            tuple(box), block.content, block.format,
            block.text_before, block.text_after, block.generated_role,
        )
        groups[key].append(position)

    removed: set[int] = set()
    for positions in groups.values():
        categories = {blocks[position].type for position in positions}
        if len(categories) < 2 or len(categories) != len(positions):
            continue
        scores = [blocks[position].layout_score for position in positions]
        if any(
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(score)
            for score in scores
        ):
            continue
        winner = max(
            positions,
            key=lambda position: (blocks[position].layout_score, -position),
        )
        removed.update(position for position in positions if position != winner)

    if not removed:
        return blocks
    return [block for position, block in enumerate(blocks) if position not in removed]
