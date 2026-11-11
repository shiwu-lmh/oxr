import math
from collections.abc import Mapping, Sequence
from numbers import Real


_WATERMARK_GUIDANCE = """Ignore decorative watermarks and translucent background marks when transcribing. Preserve meaningful stamps and printed document text. Never guess characters fully covered by a watermark; leave unclear text unfilled."""

PROMPT_TEXT_RECOGNITION = f"""Transcribe the visible text character by character in reading order. Preserve the source wording, original characters, punctuation, line breaks, numbering, names, dates, and numbers. Before answering, visually recheck ambiguous glyphs against the image and use nearby context only to distinguish characters that are visibly present. Do not correct grammar, normalize wording, infer missing content, or silently replace uncertain characters. Mark unreadable text as [无法辨认]. Output only the transcription. {_WATERMARK_GUIDANCE}"""

PROMPT_FORMULA_RECOGNITION = f"""Recognize the formula of the image into KaTex format. {_WATERMARK_GUIDANCE}"""

PROMPT_TABLE_RECOGNITION = f"""Transcribe the visible table into HTML. The crop may be rotated; read it in the natural upright orientation. Include legible handwritten entries as well as printed text. Identify all actual column-header rows and place them in <thead>, using <th> cells; place data rows in <tbody> with <td> cells. Preserve every visible row and column in order, including empty cells and rowspan/colspan. Never omit an empty cell or shift later values. Before answering, in this same response, visually cross-check every cell against the image character by character, especially short Chinese terms, names, digits, decimal points, comparison signs, and measurement units. Preserve the printed glyphs exactly; do not normalize or silently correct them based on context. Mark unreadable text as [无法辨认]. Do not infer missing text or add rows. Return one complete balanced <table> using literal HTML tags; never HTML-escape tags such as <table> or put the whole table markup inside a cell. {_WATERMARK_GUIDANCE}"""

PROMPT_TABLE_SEGMENT_RECOGNITION = f"""Transcribe only the data rows visible in this table segment. The crop may be rotated; read it upright and include legible handwritten entries. The textual header schema below describes the original table's columns; use it to preserve column order and merged-column structure, but do not output any header row. Include an empty cell for every blank grid cell so later values do not shift. Before answering, in this same response, visually cross-check every cell against the image character by character, especially short Chinese terms, names, digits, decimal points, comparison signs, and measurement units. Preserve the printed glyphs exactly; do not normalize or silently correct them based on context. Do not copy rows from other segments, invent rows, or repeat rows. Return one complete balanced <table> containing only this segment's <tbody> rows, using literal HTML tags; never HTML-escape tags or put the whole table markup inside a cell. {_WATERMARK_GUIDANCE}

Header schema:
{{header_schema}}"""

PROMPT_TABLE_TEXT_RECOGNITION = f"""Transcribe visible printed words and values in this table, in reading order. Output one line per table row and separate cells with a tab. Preserve blank cells as [空白]. Before answering, in this same response, visually cross-check every visible cell character by character, especially short Chinese terms, names, digits, decimal points, comparison signs, and measurement units. Preserve the printed glyphs exactly; do not normalize or silently correct them based on context. If you cannot finish every row in one response, still return the rows you can read in order; do not return an error message. Mark only unreadable text as [无法辨认]. Do not summarize, infer, or explain. If the table is split across images, transcribe only rows visible in this image. {_WATERMARK_GUIDANCE}"""

PROMPT_TABLE_CELL_RECOGNITION = f"""Each visible row contains labeled table-cell crops. Read each crop independently and pair its transcription with the K identifier printed beside it. Return one HTML table row containing exactly two <td> cells per crop, in any order: first the unchanged identifier such as K0001, then the cell transcription. For a visibly empty cell, return an empty second cell. Do not omit identifiers, merge neighboring cells, or move text between crops. Preserve every visible character exactly. For blood-pressure values, preserve every digit and slash exactly as printed; do not add, remove, or infer digits or separators. Mark only unreadable text as [无法辨认]. Return only one complete balanced HTML table, with literal tags."""

PROMPT_CODE_RECOGNITION = f"""Recognize the code of the image into markdown code format. {_WATERMARK_GUIDANCE}"""

PROMPT_LAYOUT_ANALYSIS = """Detect the layout of the input image, layout categories include [Caption,Footnote,Formula,Text,Page-footer,Page-header,Picture,Title,Table,Stamp,Chemical Structure,Code Block]. Box format is [x1,y1,x2,y2] and output box with reading order."""

PROMPT_WHOLE_PAGE_RECOGNITION = f"""Recognize the whole page content of the image. {_WATERMARK_GUIDANCE}"""

PROMPT_CHEMICAL_STRUCTURE_RECOGNITION = f"""Recognize the chemical structure of the image into SMILES. {_WATERMARK_GUIDANCE}"""

PROMPT_READING_ORDER = """
Given the detected layout boxes below, output the reading order as a whitespace-separated permutation of 0-based box IDs only.

Boxes:
{boxes}

Output only the box IDs."""


def _validate_coordinate(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field} must contain only finite numbers")
    return float(value)


def _format_bbox(
    bbox: Sequence[object],
    *,
    bbox_type: str,
    image_size: tuple[int, int],
) -> str:
    if len(bbox) != 4:
        raise ValueError("each reading-order bbox must contain exactly four coordinates")

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    if bbox_type == "real":
        scale_x, scale_y = 1000 / width, 1000 / height
    elif bbox_type == "norm1":
        scale_x = scale_y = 1000
    else:
        raise ValueError("objects.bbox_type must be 'real' or 'norm1'")

    coordinates = [
        int(round(_validate_coordinate(value, field="objects.bbox") * scale))
        for value, scale in zip(bbox, (scale_x, scale_y, scale_x, scale_y))
    ]
    return (
        f"<|box_start|>({coordinates[0]},{coordinates[1]}),"
        f"({coordinates[2]},{coordinates[3]})<|box_end|>"
    )


def build_reading_order_prompt(
    objects: Mapping[str, object],
    *,
    image_size: tuple[int, int],
) -> str:
    """Render ms-swift grounding objects into the reading-order prompt."""
    if not isinstance(objects, Mapping):
        raise ValueError("reading_order requires an objects JSON object")

    refs = objects.get("ref")
    bboxes = objects.get("bbox")
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
        raise ValueError("objects.ref must be a list")
    if not isinstance(bboxes, Sequence) or isinstance(bboxes, (str, bytes)):
        raise ValueError("objects.bbox must be a list")
    if len(refs) != len(bboxes):
        raise ValueError("objects.ref and objects.bbox must have the same length")

    bbox_type = objects.get("bbox_type") or "real"
    if not isinstance(bbox_type, str):
        raise ValueError("objects.bbox_type must be 'real' or 'norm1'")

    lines = []
    for index, (ref, bbox) in enumerate(zip(refs, bboxes)):
        if not isinstance(ref, str):
            raise ValueError("objects.ref must contain only strings")
        if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)):
            raise ValueError("objects.bbox must contain coordinate lists")
        lines.append(
            f"{index} | <|object_ref_start|>{ref}<|object_ref_end|> | "
            f"{_format_bbox(bbox, bbox_type=bbox_type, image_size=image_size)}"
        )

    return PROMPT_READING_ORDER.format(boxes="\n".join(lines))
