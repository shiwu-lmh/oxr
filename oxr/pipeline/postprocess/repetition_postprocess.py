"""Conservatively flag and shorten obvious model repetition loops."""

from __future__ import annotations

import re


_HTML_PARTS = re.compile(r"(<[^>]*>)")
_SPACED_CHAR_RUN = re.compile(r"(?P<char>[^\s])(?:\s*(?P=char)){11,}", re.UNICODE)
_MAX_PATTERN_LENGTH = 64
_MIN_REPEATS = 6
_MIN_SINGLE_CHAR_REPEATS = 12
_KEEP_REPEATS = 3


def _repetition_at(text: str, start: int) -> tuple[int, int] | None:
    """Return (pattern length, repeat count) for a strong loop at start."""
    remaining = len(text) - start
    best: tuple[int, int] | None = None
    max_pattern = min(_MAX_PATTERN_LENGTH, remaining // _MIN_REPEATS)
    for pattern_length in range(1, max_pattern + 1):
        pattern = text[start : start + pattern_length]
        if not any(not character.isspace() for character in pattern):
            continue
        if text[start + pattern_length : start + pattern_length * 2] != pattern:
            continue

        repeats = 2
        cursor = start + pattern_length * 2
        while cursor + pattern_length <= len(text) and text[cursor : cursor + pattern_length] == pattern:
            repeats += 1
            cursor += pattern_length

        minimum = _MIN_SINGLE_CHAR_REPEATS if pattern_length == 1 else _MIN_REPEATS
        if repeats >= minimum and (best is None or repeats * pattern_length > best[0] * best[1]):
            best = (pattern_length, repeats)
    return best


def _clean_plain_text(text: str) -> str:
    # OCR can emit repeated characters separated by spaces or line wrapping.
    # Collapse that run to a few copies, regardless of the repeated character.
    text = _SPACED_CHAR_RUN.sub(
        lambda match: match.group("char") * _KEEP_REPEATS,
        text,
    )

    result: list[str] = []
    cursor = 0
    while cursor < len(text):
        repeated = _repetition_at(text, cursor)
        if repeated is None:
            result.append(text[cursor])
            cursor += 1
            continue

        pattern_length, repeats = repeated
        result.append(text[cursor : cursor + pattern_length * _KEEP_REPEATS])
        cursor += pattern_length * repeats
    return "".join(result)


def postprocess_repetition_loops(text: str, *, preserve_html: bool = False) -> str:
    """Shorten long, exact repeated substrings to a few copies.

    Table markup is split from its text payload so HTML tags remain intact. A
    repeated sequence is only shortened inside one text block/cell; identical
    values in neighboring table cells or rows are never compared.
    """
    if not text:
        return text
    if not preserve_html:
        return _clean_plain_text(text)

    return "".join(
        part if part.startswith("<") and part.endswith(">") else _clean_plain_text(part)
        for part in _HTML_PARTS.split(text)
    )
