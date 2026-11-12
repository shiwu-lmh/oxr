import logging
import re
from oxr.pipeline.models import Block, ContentFormat
from oxr.pipeline.postprocess.formula_postprocess import move_formula_numbering_outside_math


logger = logging.getLogger(__name__)
_NUMBER_TOKEN = re.compile(r"(?<!\d)\d{1,4}(?!\d)")
_LIST_PREFIX = re.compile(r"^(\s*(?:\d+[.、)]|[-*+])\s*)")
_NUMBER_SEPARATORS = frozenset("、，,;； \t")
_MIN_PATHOLOGICAL_SEQUENCE = 64
_MIN_PATHOLOGICAL_LINE_LENGTH = 500


def _has_pathological_ascending_run(line: str) -> bool:
    """Detect an implausibly long inline run of consecutive list integers."""
    matches = list(_NUMBER_TOKEN.finditer(line))
    run_length = 1
    for previous, current in zip(matches, matches[1:]):
        separator = line[previous.end():current.start()]
        consecutive = (
            separator
            and all(char in _NUMBER_SEPARATORS for char in separator)
            and int(current.group()) == int(previous.group()) + 1
        )
        run_length = run_length + 1 if consecutive else 1
        if run_length >= _MIN_PATHOLOGICAL_SEQUENCE:
            return True
    return False


def _suppress_pathological_number_lines(text: str, block: Block) -> str:
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if (
            len(line) < _MIN_PATHOLOGICAL_LINE_LENGTH
            or not _has_pathological_ascending_run(line)
        ):
            continue
        prefix = _LIST_PREFIX.match(line)
        marker = "[疑似连续数字识别噪声，已省略，请核对原图]"
        lines[index] = (prefix.group(1) if prefix else "") + marker
        logger.warning(
            "Suppressed a pathological ascending-number run in text block %s; "
            "the source line should be checked against the page image.",
            block.idx,
        )
    return "\n".join(lines)


def postprocess_text(block: Block, raw_text: str) -> Block:
    """Clean up text block output."""
    text = raw_text.strip()
    # Remove some excessive whitespaces
    text = re.sub(r'\n+', '\n', text)
    text = _suppress_pathological_number_lines(text, block)
    text = move_formula_numbering_outside_math(text)
    block.content = text
    block.format = ContentFormat.TEXT
    return block
