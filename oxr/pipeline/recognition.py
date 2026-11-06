import asyncio
import html
import logging
import math
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np

from oxr.config.settings import settings
from oxr.model.client import model_client
from oxr.pipeline.concurrency import gather_cancel_on_error
from oxr.pipeline.errors import PipelineError, format_error_message, get_error_status_code
from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.postprocess.chemical_structure_postprocess import (
    postprocess_chemical_structure,
)
from oxr.pipeline.postprocess.code_postprocess import postprocess_code
from oxr.pipeline.postprocess.formula_postprocess import postprocess_formula
from oxr.pipeline.postprocess.picture_postprocess import postprocess_picture
from oxr.pipeline.postprocess.table_postprocess import postprocess_table
from oxr.pipeline.postprocess.table_images import (
    group_table_images, mark_table_images, prepare_table_images,
    restore_table_images, unchanged_block,
)
from oxr.pipeline.postprocess.text_postprocess import postprocess_text
from oxr.pipeline.table_segmentation import (
    combine_table_chunks,
    expected_table_columns, extract_table_header_schema,
    is_balanced_table_html, is_complete_table_html,
    repair_table_html_prefix, split_table_image, table_row_widths,
    split_table_image_with_bounds,
)
from oxr.utils.image import crop_image, resize_image
from oxr.utils.table import _TableParser, strip_table_code_fence

logger = logging.getLogger(__name__)

LENGTH_RECOVERY_REPETITION_PENALTY = 1.1
TABLE_LENGTH_RECOVERY_SCALE = 0.75
ZERO_OUTPUT_MIN_TOKENS = 128


def _is_zero_only_output(text: str) -> bool:
    tokens = text.split()
    return len(tokens) >= ZERO_OUTPUT_MIN_TOKENS and all(token == "0" for token in tokens)


def _is_zero_dominated_output(text: str) -> bool:
    """Catch zero loops wrapped in Markdown fences or a short generated prefix."""
    body = re.sub(r"(?m)^\s*`{3,}[^\n]*$", " ", text)
    tokens = re.findall(r"\S+", body)
    zero_count = sum(token.strip(".,;:!?()[]{}") == "0" for token in tokens)
    return (
        zero_count >= ZERO_OUTPUT_MIN_TOKENS
        and zero_count / max(1, len(tokens)) >= 0.95
    )


def _finish_reason(output: str) -> str:
    return str(getattr(output, "finish_reason", "unknown"))


def _length_recovery_options(
    model_options: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    recovered = dict(model_options or {})
    current_penalty = float(recovered.get("repetition_penalty", 1.0))
    recovered["repetition_penalty"] = max(
        current_penalty,
        LENGTH_RECOVERY_REPETITION_PENALTY,
    )
    return recovered


async def _request_with_retries(
    request: Callable[[], Awaitable[str]],
    *,
    block: Block,
    attempts: int,
    timeout: int,
) -> str:
    for attempt in range(attempts):
        try:
            return await request()
        except Exception as error:
            if attempt == attempts - 1:
                message = format_error_message(error, f"Timeout after {timeout}s")
                raise PipelineError(
                    f"Recognition failed for block {block.idx} after {attempts} attempts: {message}",
                    status_code=get_error_status_code(error, default=503),
                ) from error
            await asyncio.sleep(1)

    raise AssertionError("Recognition retry loop exited unexpectedly.")


async def _request_table_with_budget(
    request: Callable[[], Awaitable[str]],
    *,
    block: Block,
    attempts: int,
    timeout: int,
    budget: dict[str, int],
) -> str:
    """Allow one initial request per chunk plus a small structural retry pool."""
    max_calls = budget["segments"] + budget["retries"]
    if budget["used"] >= max_calls:
        raise PipelineError(
            f"Adaptive table request budget ({max_calls}) reached for block {block.idx}.",
            status_code=502,
            public_message="表格分段识别额度已用完。",
        )
    budget["used"] += 1
    return await _request_with_retries(
        request, block=block, attempts=attempts, timeout=timeout,
    )


async def _request_block(
    block: Block,
    crop_img: np.ndarray,
    sem: asyncio.Semaphore,
    model_options: Optional[Dict[str, Any]],
) -> str:
    async with sem:
        if block.type == BlockType.TABLE:
            return await model_client.recognize_table(crop_img, model_options)
        if block.type == BlockType.FORMULA:
            return await model_client.recognize_formula(crop_img, model_options)
        if block.type == BlockType.CODE_BLOCK:
            # Layout classifiers can label dense printed content (for example,
            # medical measurements) as code. The code-specific prompt can then
            # trigger repetitive token loops. Use general OCR for detected code
            # regions; postprocess_code still preserves the Markdown code format.
            return await model_client.recognize_text(crop_img, model_options)
        if block.type == BlockType.CHEMICAL_STRUCTURE:
            return await model_client.recognize_chemical_structure(
                crop_img,
                model_options,
            )
        return await model_client.recognize_text(crop_img, model_options)


async def _recognize_table_image(
    block: Block,
    image: np.ndarray,
    sem: asyncio.Semaphore,
    model_options: Optional[Dict[str, Any]],
    *,
    depth: int = 0,
    header_schema: Optional[str] = None,
    request_budget: Optional[dict[str, int]] = None,
    preserve_image_tokens: bool = False,
) -> str:
    """Recognize the full table crop before splitting incomplete model output."""
    config = settings.oxr_model
    if request_budget is None:
        request_budget = {
            "used": 0,
            "segments": 1,
            "retries": 1,
            "retries_used": 0,
        }

    async with sem:
        async def recognize_chunk() -> str:
            if header_schema is None:
                return await model_client.recognize_table(image, model_options)
            return await model_client.recognize_table(
                image, model_options, header_schema=header_schema,
            )

        raw = await _request_table_with_budget(
            recognize_chunk,
            block=block,
            attempts=config.retry_times,
            timeout=config.timeout,
            budget=request_budget,
        )

    if _finish_reason(raw) != "length" and is_complete_table_html(raw):
        widths = table_row_widths(raw)
        if len(set(widths)) <= 1:
            return raw
    if _finish_reason(raw) != "length" and not is_balanced_table_html(raw):
        repaired = repair_table_html_prefix(raw)
        if repaired is not None:
            logger.warning(
                "Table block %s response omitted leading HTML tags; repairing its envelope.",
                block.idx,
            )
            raw = repaired
            if is_complete_table_html(raw):
                return raw

    # A ragged row is often caused by the model omitting an empty cell. Give it
    # the dominant logical width and ask it to realign the same crop once.
    expected_columns = expected_table_columns(raw)
    if (
        expected_columns
        and _finish_reason(raw) != "length"
        and request_budget["retries_used"] < request_budget["retries"]
    ):
        request_budget["retries_used"] += 1
        logger.warning(
            "Table block %s has inconsistent row widths; retrying with %d logical columns.",
            block.idx, expected_columns,
        )
        async with sem:
            retried = await _request_table_with_budget(
                lambda: model_client.recognize_table(
                    image, model_options,
                    header_schema=header_schema,
                    expected_columns=expected_columns,
                ),
                block=block,
                attempts=config.retry_times,
                timeout=config.timeout,
                budget=request_budget,
            )
        if _finish_reason(retried) != "length" and not is_balanced_table_html(retried):
            repaired = repair_table_html_prefix(retried)
            if repaired is not None:
                logger.warning(
                    "Table block %s retry had missing leading HTML tags; repairing its envelope.",
                    block.idx,
                )
                retried = repaired
        if _finish_reason(retried) != "length" and is_complete_table_html(retried):
            return retried
        if not is_balanced_table_html(raw):
            raw = retried

    should_split = _finish_reason(raw) == "length" or not is_balanced_table_html(raw)
    if depth < 12 and should_split and image.shape[0] > TABLE_MIN_SPLIT_HEIGHT:
        target_height = min(
            TABLE_RECOGNITION_CHUNK_HEIGHT, max(TABLE_MIN_SPLIT_HEIGHT, image.shape[0] // 2)
        )
        smaller = split_table_image_with_bounds(image, max_height=target_height)
        if len(smaller) > 1:
            logger.warning("Table block %s returned incomplete HTML; retrying as %d smaller chunks.", block.idx, len(smaller))
            # Count the newly created children as baseline work; the failed
            # parent request has already been charged.
            request_budget["segments"] += len(smaller)
            request_budget["retries"] = min(
                request_budget["segments"],
                max(1, math.ceil(request_budget["segments"] * 0.2)),
            )
            results = []
            active_schema = header_schema
            for part, _, _ in smaller:
                recognized = await _recognize_table_image(
                    block, part, sem, model_options, depth=depth + 1,
                    header_schema=active_schema,
                    request_budget=request_budget,
                    preserve_image_tokens=preserve_image_tokens,
                )
                results.append(recognized)
                if active_schema is None:
                    active_schema = extract_table_header_schema(recognized)
            return combine_table_chunks(results)

    # Do not fail the whole document over a small row-width disagreement after
    # schema-guided retries. Preserve closed rows for export and log the widths
    # so the user can still review the table instead of losing every page.
    if _finish_reason(raw) != "length" and is_balanced_table_html(raw):
        widths = table_row_widths(raw)
        logger.warning(
            "Table block %s remains ragged after retries (logical row widths=%s); exporting best-effort rows.",
            block.idx, sorted(set(widths)),
        )
        return raw

    if depth and not preserve_image_tokens:
        logger.warning("Table block %s has an unreadable HTML slice; reading that slice as text.", block.idx)
        return await _recognize_table_text_fallback(block, image, sem, model_options)

    raise PipelineError(
        f"Table recognition returned incomplete HTML for block {block.idx}.",
        status_code=502,
        public_message="表格分块识别仍未得到完整结构。请重试；未识别内容不会以图片代替。",
    )


TABLE_MIN_SPLIT_HEIGHT = 96
TABLE_RECOGNITION_CHUNK_HEIGHT = 360

async def _recognize_table_text_fallback(
    block: Block,
    image: np.ndarray,
    sem: asyncio.Semaphore,
    model_options: Optional[Dict[str, Any]],
) -> str:
    """Use the same vision model to preserve table text if HTML OCR fails."""
    config = settings.oxr_model

    async def recognize_piece(piece: np.ndarray, depth: int = 0) -> List[str]:
        async with sem:
            output = await _request_with_retries(
                lambda: model_client.recognize_table_as_text(piece, model_options),
                block=block,
                attempts=config.retry_times,
                timeout=config.timeout,
            )
        if output.strip() and _finish_reason(output) != "length":
            return [output.strip()]

        height = piece.shape[0]
        if depth >= 8 or height <= TABLE_MIN_SPLIT_HEIGHT:
            if output.strip():
                logger.warning(
                    "Table block %s text OCR remains truncated at the minimum crop; keeping the partial model output.",
                    block.idx,
                )
                return [output.strip()]
            raise PipelineError(
                f"Text fallback for table block {block.idx} returned no content.",
                status_code=502,
                public_message="表格识别未返回内容。",
            )

        target_height = max(TABLE_MIN_SPLIT_HEIGHT, height // 2)
        smaller = split_table_image(piece, max_height=target_height)
        if len(smaller) <= 1:
            midpoint = height // 2
            overlap = min(12, max(1, height // 16))
            smaller = [
                piece[:midpoint + overlap].copy(),
                piece[midpoint - overlap:].copy(),
            ]
        logger.warning(
            "Table block %s text OCR was truncated; splitting %dpx segment into %d smaller crops.",
            block.idx, height, len(smaller),
        )
        results: List[str] = []
        for child in smaller:
            results.extend(await recognize_piece(child, depth + 1))
        return results

    chunks = split_table_image(image, max_height=TABLE_RECOGNITION_CHUNK_HEIGHT)
    responses: List[str] = []
    for chunk in chunks:
        responses.extend(await recognize_piece(chunk))

    table_responses: List[str] = []
    for response in responses:
        candidate = strip_table_code_fence(response).strip()
        decoded = html.unescape(candidate)
        recognized_table = next((
            value for value in (candidate, decoded)
            if is_balanced_table_html(value)
        ), None)
        if recognized_table is not None:
            table_responses.append(recognized_table)
    if len(table_responses) == len(responses) and table_responses:
        return (
            table_responses[0]
            if len(table_responses) == 1
            else combine_table_chunks(table_responses)
        )

    lines: List[str] = []
    for response in responses:
        candidate = html.unescape(strip_table_code_fence(response)).strip()
        if is_balanced_table_html(candidate):
            parser = _TableParser()
            parser.feed(candidate)
            lines.extend("\t".join(cell.text for cell in row) for row in parser.rows)
        else:
            lines.extend(candidate.splitlines())
    rows = "".join(
        "<tr>" + "".join(
            "<td>" + html.escape(cell) + "</td>"
            for cell in line.split("\t")
        ) + "</tr>"
        for line in lines
    )
    if not rows:
        raise PipelineError(
            f"Text fallback for table block {block.idx} returned no rows.",
            status_code=502,
            public_message="表格识别失败，文本补充识别没有返回内容。",
        )
    return "<table><tbody>" + rows + "</tbody></table>"


async def recognize_block(
    block: Block,
    page_image: np.ndarray,
    sem: asyncio.Semaphore,
    file_name: str,
    page_num: int,
    model_options: Optional[Dict[str, Any]] = None,
    element_formats: Optional[Dict[str, str]] = None,
    table_images: Optional[List[Block]] = None,
) -> Block:
    crop_img = crop_image(page_image, block.bbox)
    element_formats = element_formats or {}

    if block.type == BlockType.PICTURE:
        # No OCR for picture, just save it
        return await asyncio.to_thread(postprocess_picture, block, crop_img, file_name, page_num)
    if (
        block.type == BlockType.CHEMICAL_STRUCTURE
        and element_formats.get("cs") == "image"
    ):
        return await asyncio.to_thread(postprocess_picture, block, crop_img, file_name, page_num)

    sources: Dict[int, str] = {}
    if table_images:
        try:
            crop_img, sources = await asyncio.to_thread(prepare_table_images, block, table_images, page_image)
        except ValueError as error:
            logger.warning("Table %s image placeholders skipped: %s; using table text OCR.", block.idx, error)
            mark_table_images(table_images)
            raw_out = await _recognize_table_text_fallback(
                block, crop_img, sem, model_options,
            )
            return postprocess_table(block, raw_out, element_formats.get("table", "html"))

    config = settings.oxr_model
    try:
        if block.type == BlockType.TABLE:
            raw_out = await _recognize_table_image(
                block, crop_img, sem, model_options,
                preserve_image_tokens=bool(sources),
            )
        else:
            raw_out = await _request_with_retries(
                lambda: _request_block(block, crop_img, sem, model_options),
                block=block,
                attempts=config.retry_times,
                timeout=config.timeout,
            )
    except (PipelineError, ValueError) as error:
        if block.type != BlockType.TABLE:
            raise
        logger.error("Table %s failed after adaptive row splitting: %s; trying same-model text OCR.", block.idx, error)
        if table_images:
            mark_table_images(table_images)
        raw_out = await _recognize_table_text_fallback(
            block, crop_img, sem, model_options,
        )
        return postprocess_table(block, raw_out, element_formats.get("table", "html"))

    if block.type == BlockType.CODE_BLOCK and (
        _is_zero_only_output(raw_out)
        or _is_zero_dominated_output(raw_out)
        or _finish_reason(raw_out) == "length"
    ):
        logger.warning(
            "Code block %s returned a zero-dominated or truncated response; retrying with text recognition.",
            block.idx,
        )
        fallback_options = _length_recovery_options(model_options)
        try:
            async with sem:
                fallback_out = await _request_with_retries(
                    lambda: model_client.recognize_text(crop_img, fallback_options),
                    block=block,
                    attempts=config.retry_times,
                    timeout=config.timeout,
                )
        except PipelineError as error:
            logger.warning(
                "Text fallback failed for code block %s (%s); dropping malformed output.",
                block.idx,
                type(error).__name__,
            )
            fallback_out = ""

        if (
            not fallback_out
            or _is_zero_only_output(fallback_out)
            or _is_zero_dominated_output(fallback_out)
            or _finish_reason(fallback_out) == "length"
        ):
            logger.warning(
                "Code block %s remained zero-dominated or truncated after text fallback; dropping it.",
                block.idx,
            )
            block.content = ""
            block.format = ContentFormat.TEXT
            block.type = BlockType.TEXT
            return block

        # The text prompt recovered ordinary document content; don't wrap it as code.
        block.type = BlockType.TEXT
        raw_out = fallback_out

    if _is_zero_only_output(raw_out):
        zero_count = len(raw_out.split())
        retry_options = _length_recovery_options(model_options)
        logger.warning(
            "Recognition returned a repeated-zero block (tokens=%d); retrying with repetition penalty %.1f.",
            zero_count,
            retry_options["repetition_penalty"],
        )
        try:
            retried_out = await _request_with_retries(
                lambda: _request_block(block, crop_img, sem, retry_options),
                block=block,
                attempts=config.retry_times,
                timeout=config.timeout,
            )
        except PipelineError as error:
            logger.warning(
                "Repeated-zero retry failed for block %s (%s); dropping the malformed block.",
                block.idx,
                type(error).__name__,
            )
            retried_out = ""
        if _is_zero_only_output(retried_out):
            logger.warning(
                "Repeated-zero retry remained malformed for block %s; dropping the block.",
                block.idx,
            )
            block.content = ""
            block.format = ContentFormat.TEXT
            return block
        raw_out = retried_out

    if _finish_reason(raw_out) == "length":
        recovery_image: Optional[np.ndarray] = None
        recovery_options: Optional[Dict[str, Any]] = None
        recovery_method: Optional[str] = None
        if block.type == BlockType.TABLE:
            recovery_image = resize_image(crop_img, TABLE_LENGTH_RECOVERY_SCALE)
            recovery_options = model_options
            recovery_method = f"input scale {TABLE_LENGTH_RECOVERY_SCALE}"
        elif block.type in {
            BlockType.TEXT,
            BlockType.TITLE,
            BlockType.CAPTION,
            BlockType.FOOTNOTE,
            BlockType.PAGE_HEADER,
            BlockType.PAGE_FOOTER,
            BlockType.STAMP,
        }:
            recovery_image = crop_img
            recovery_options = _length_recovery_options(model_options)
            recovery_method = (
                "repetition penalty "
                f"{recovery_options['repetition_penalty']}"
            )
        else:
            logger.warning(
                "Recognition output reached the token limit for block %s (%s); "
                "no recovery is defined, preserving the truncated model output.",
                block.idx,
                block.type.value,
            )

        if recovery_method is not None and recovery_image is not None:
            logger.warning(
                "Recognition output reached the token limit for block %s (%s); "
                "retrying with %s.",
                block.idx,
                block.type.value,
                recovery_method,
            )
            try:
                raw_out = await _request_with_retries(
                    lambda: _request_block(block, recovery_image, sem, recovery_options),
                    block=block,
                    attempts=config.retry_times,
                    timeout=config.timeout,
                )
            except PipelineError:
                if block.type != BlockType.TABLE:
                    raise
                logger.error("Table %s truncated response could not be recovered.", block.idx)
                if table_images:
                    mark_table_images(table_images)
                raw_out = await _recognize_table_text_fallback(
                    block, crop_img, sem, model_options,
                )
                return postprocess_table(block, raw_out, element_formats.get("table", "html"))
            if _finish_reason(raw_out) == "length":
                logger.warning(
                    "Recognition output remained truncated for block %s (%s) "
                    "after length recovery with %s; preserving the truncated "
                    "model output.",
                    block.idx,
                    block.type.value,
                    recovery_method,
                )

    if block.type == BlockType.TABLE:
        if table_images:
            restored = restore_table_images(raw_out, sources)
            mark_table_images(table_images)
            if restored is None or _finish_reason(raw_out) == "length":
                logger.warning("Table %s has incomplete image IDs/output; retrying table text OCR.", block.idx)
                raw_out = await _recognize_table_text_fallback(
                    block, crop_img, sem, model_options,
                )
                return postprocess_table(block, raw_out, element_formats.get("table", "html"))
            raw_out = restored
        return postprocess_table(
            block,
            raw_out,
            element_formats.get("table", "html"),
        )
    if block.type == BlockType.FORMULA:
        return postprocess_formula(block, raw_out)
    if block.type == BlockType.CODE_BLOCK:
        return postprocess_code(block, raw_out)
    if block.type == BlockType.CHEMICAL_STRUCTURE:
        return postprocess_chemical_structure(block, raw_out)
    return postprocess_text(block, raw_out)


async def recognize_pages(
    pages_blocks: List[List[Block]],
    pages_images: List[np.ndarray],
    file_name: str,
    semaphore: asyncio.Semaphore,
    model_options: Optional[Dict[str, Any]] = None,
    element_formats: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Callable[[dict[str, int]], Awaitable[None]]] = None,
) -> List[List[Block]]:
    """Run recognition for all blocks across all pages concurrently."""
    if len(pages_blocks) != len(pages_images):
        raise ValueError("Page block and image counts must match")
    tasks = []
    block_total = sum(len(blocks) for blocks in pages_blocks)
    block_completed = 0
    page_remaining = [len(blocks) for blocks in pages_blocks]
    page_completed = sum(1 for remaining in page_remaining if remaining == 0)
    progress_lock = asyncio.Lock()

    async def track_progress(page_idx: int, task):
        nonlocal block_completed, page_completed
        result = await task
        if progress_callback is not None:
            async with progress_lock:
                block_completed += 1
                page_remaining[page_idx] -= 1
                if page_remaining[page_idx] == 0:
                    page_completed += 1
                await progress_callback({
                    "block_completed": block_completed,
                    "block_total": block_total,
                    "pages_completed": page_completed,
                    "page_total": len(pages_blocks),
                })
        return result

    if progress_callback is not None and block_total == 0:
        await progress_callback({
            "block_completed": 0,
            "block_total": 0,
            "pages_completed": len(pages_blocks),
            "page_total": len(pages_blocks),
        })
    # keep track of which task belongs to which page and block index
    # tasks structure: (page_idx, block_idx, task)
    for page_idx, (blocks, image) in enumerate(zip(pages_blocks, pages_images)):
        groups = (
            group_table_images(
                blocks, settings.pipeline.table_image_min_score,
                keep_outer_tables=settings.pipeline.full_overlap_dedupe,
            )
            if settings.pipeline.table_image_placeholders else {}
        )
        nested = {id(picture) for pictures in groups.values() for picture in pictures}
        for block_idx, block in enumerate(blocks):
            if id(block) in nested:
                task = unchanged_block(block)
            else:
                kwargs = {"table_images": groups[block_idx]} if block_idx in groups else {}
                task = recognize_block(
                    block, image, semaphore, file_name, page_idx,
                    model_options, element_formats, **kwargs,
                )
            if progress_callback is not None:
                task = track_progress(page_idx, task)
            tasks.append((page_idx, block_idx, task))

    # Run all tasks
    gathered_results = await gather_cancel_on_error(*(t[2] for t in tasks))

    # Reconstruct pages_blocks
    final_pages_blocks: List[List[Optional[Block]]] = [
        [None] * len(page_blocks) for page_blocks in pages_blocks
    ]
    for (page_idx, block_idx, _), block_result in zip(tasks, gathered_results):
        final_pages_blocks[page_idx][block_idx] = block_result

    return [[block for block in page if block is not None] for page in final_pages_blocks]
