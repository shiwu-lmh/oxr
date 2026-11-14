"""Conservative detection and removal of content clipped by page boundaries."""

import logging

import cv2
import numpy as np

from oxr.pipeline.models import Block, BlockType
from oxr.pipeline.table_segmentation import _horizontal_rules


logger = logging.getLogger(__name__)
EDGE_TOLERANCE_PX = 2
EDGE_INK_RATIO = 0.035
MIN_TABLE_HEIGHT = 48


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _ink_ratio(strip: np.ndarray) -> float:
    if strip.size == 0:
        return 0.0
    return float(np.mean(_gray(strip) < 210))


def is_clipped_at_page_edge(block: Block, page_image: np.ndarray) -> bool:
    """Detect non-table blocks whose visible ink is cut by the page boundary."""
    height, width = page_image.shape[:2]
    x1, y1, x2, y2 = (round(value) for value in block.bbox)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x1 >= x2 or y1 >= y2:
        return True

    crop = page_image[y1:y2, x1:x2]
    edge_width = min(3, crop.shape[1])
    edge_height = min(3, crop.shape[0])
    checks = []
    if x1 <= EDGE_TOLERANCE_PX:
        checks.append(_ink_ratio(crop[:, :edge_width]))
    if x2 >= width - EDGE_TOLERANCE_PX:
        checks.append(_ink_ratio(crop[:, -edge_width:]))
    if y1 <= EDGE_TOLERANCE_PX:
        checks.append(_ink_ratio(crop[:edge_height, :]))
    if y2 >= height - EDGE_TOLERANCE_PX:
        checks.append(_ink_ratio(crop[-edge_height:, :]))
    return any(ratio >= EDGE_INK_RATIO for ratio in checks)


def trim_partial_table_edges(
    crop: np.ndarray,
    block: Block,
    page_image: np.ndarray,
) -> tuple[np.ndarray | None, bool]:
    """Trim an incomplete first/last table row when the table meets a page edge."""
    height, width = page_image.shape[:2]
    y1, y2 = round(block.bbox[1]), round(block.bbox[3])
    touches_top = y1 <= EDGE_TOLERANCE_PX
    touches_bottom = y2 >= height - EDGE_TOLERANCE_PX
    if not touches_top and not touches_bottom:
        return crop, False

    rules = _horizontal_rules(crop)
    start, end = 0, crop.shape[0]
    changed = False
    if touches_top and not any(rule <= EDGE_TOLERANCE_PX for rule in rules):
        next_rule = next((rule for rule in rules if rule >= 8), None)
        if next_rule is None:
            return None, True
        start = next_rule
        changed = True
    if touches_bottom and not any(rule >= crop.shape[0] - 1 - EDGE_TOLERANCE_PX for rule in rules):
        previous_rule = next((rule for rule in reversed(rules) if rule <= crop.shape[0] - 8), None)
        if previous_rule is None:
            return None, True
        end = previous_rule + 1
        changed = True

    if end - start < MIN_TABLE_HEIGHT:
        return None, changed
    if changed:
        original_y1 = block.bbox[1]
        block.bbox[1] = original_y1 + start
        block.bbox[3] = block.bbox[1] + (end - start)
        logger.info(
            "Trimmed clipped table edge rows (top=%s, bottom=%s).",
            touches_top and start > 0,
            touches_bottom and end < crop.shape[0],
        )
        return crop[start:end].copy(), True
    return crop, False


def skip_clipped_content(block: Block) -> Block:
    logger.info("Skipping clipped %s block %s at the page boundary.", block.type.value, block.idx)
    block.content = ""
    return block
