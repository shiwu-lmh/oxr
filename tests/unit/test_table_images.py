import asyncio
import base64

import cv2
import numpy as np
import pytest

from oxr.config.settings import settings
from oxr.model.client import ModelOutput
from oxr.pipeline import orchestrator
from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.postprocess.table_images import (
    TABLE_IMAGE_ROLE,
    group_table_images,
    prepare_table_images,
    restore_table_images,
)
from oxr.pipeline.recognition import recognize_block, recognize_pages
from oxr.utils.table import html_table_to_markdown


def block(idx, kind, box, score=0.9):
    return Block(idx, kind, box, "", ContentFormat.TEXT, layout_score=score)


def scene():
    image = np.full((180, 260, 3), 255, dtype=np.uint8)
    image[40:85, 135:235] = [10, 70, 130]
    image[100:145, 135:235] = [40, 100, 160]
    table = block(8, BlockType.TABLE, [10.4, 10.4, 250.2, 170.2])
    first = block(23, BlockType.PICTURE, [135, 40, 235, 85])
    second = block(41, BlockType.PICTURE, [135, 100, 235, 145])
    return image, table, first, second


def test_group_uses_confidence_smallest_parent_and_geometry_order():
    _, table, first, second = scene()
    outer = block(99, BlockType.TABLE, [0, 0, 260, 180])
    low = block(32, BlockType.PICTURE, [20, 20, 120, 50], 0.26)
    unknown = block(33, BlockType.PICTURE, [20, 60, 120, 90], None)
    crossing = block(34, BlockType.PICTURE, [245, 160, 265, 175])
    same_box = block(35, BlockType.PICTURE, table.bbox)
    groups = group_table_images(
        [second, outer, low, table, first, unknown, crossing, same_box], 0.5
    )
    assert groups[3] == [first, second]
    assert groups[1] == [same_box]
    assert low not in sum(groups.values(), [])
    assert unknown not in sum(groups.values(), [])
    assert crossing not in sum(groups.values(), [])


def test_mask_keeps_original_image_and_correct_rounded_crops():
    image, table, first, second = scene()
    original = image.copy()
    masked, sources = prepare_table_images(table, [first, second], image)
    np.testing.assert_array_equal(image, original)
    assert masked.shape == (160, 240, 3)
    assert set(sources) == {1, 2}
    restored = cv2.imdecode(
        np.frombuffer(base64.b64decode(sources[1].split(",")[1]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    np.testing.assert_array_equal(restored, image[36:89, 131:239])
    assert np.any(np.all(masked == [255, 0, 0], axis=2))
    assert not sources[1] == sources[2]


def test_full_overlap_mode_associates_images_with_surviving_outer_table():
    _, table, first, second = scene()
    outer = block(99, BlockType.TABLE, [0, 0, 260, 180])
    assert group_table_images(
        [table, first, outer, second], 0.5, keep_outer_tables=True
    ) == {2: [first, second]}


@pytest.mark.parametrize("variant", ["tiny", "overlap"])
def test_ambiguous_or_tiny_masks_fail_before_recognition(variant):
    image, table, first, second = scene()
    if variant == "tiny":
        first.bbox = [20, 20, 25, 25]
    else:
        second.bbox = first.bbox
    with pytest.raises(ValueError):
        prepare_table_images(table, [first, second], image)


@pytest.mark.parametrize(
    "tokens",
    [("<tit>2", "<tit>1"), ("&lt;tit>2", "&lt;tit&gt;1"), ("<TIT>2", "<TIT>1")],
)
def test_restore_maps_by_id_even_when_order_changes_and_preserves_text(tokens):
    text = f"<table><tr><td>A {tokens[0]} B {tokens[1]}</td><td>x|x≠0</td></tr></table>"
    restored = restore_table_images(
        "```html\n" + text + "\n```",
        {1: "data:image/png;base64,first", 2: "data:image/png;base64,second"},
    )
    assert restored.index("base64,second") < restored.index("base64,first")
    assert "x|x≠0" in restored
    assert restored.count("<img ") == 2


@pytest.mark.parametrize(
    "text",
    [
        "<table><tr><td><tit>1</td></tr></table>",
        "<table><tr><td><tit>1 <tit>1</td></tr></table>",
        "<table><tr><td><tit>1 <tit>3</td></tr></table>",
        "<table><tr><td><tit>1 <tit></td></tr></table>",
        "<table><tr><td><tit>1 <tit>2",
        "<tit>1<table><tr><td><tit>2</td></tr></table>",
        '<table><tr><td title="<tit>1"><tit>2</td></tr></table>',
        "<table><tr><td><tit>1 <tit>2 <tit>3</td></tr></table>",
    ],
)
def test_incomplete_duplicate_unknown_or_outside_cell_ids_are_rejected(text):
    assert restore_table_images(text, {1: "first", 2: "second"}) is None


def test_markdown_keeps_multiple_inline_images_in_merged_cell():
    html = restore_table_images(
        '<table><tr><td colspan="2">A <tit>1 B <tit>2</td></tr></table>',
        {1: "data:image/png;base64,first", 2: "data:image/png;base64,second"},
    )
    markdown = html_table_to_markdown(html)
    assert markdown.count("![](data:image/png;base64,first)") == 2
    assert markdown.count("![](data:image/png;base64,second)") == 2


def test_restore_preserves_math_comparisons_before_later_image_cells():
    text = "<table><tr><td>0<a&lt;1</td><td>&lt;tit>1</td></tr><tr><td>x<0</td><td>&lt;tit>2</td></tr></table>"
    restored = restore_table_images(text, {1: "first", 2: "second"})
    assert restored is not None
    assert "0&lt;a&lt;1" in restored and "x&lt;0" in restored
    assert restored.count("<img ") == 2
    markdown = html_table_to_markdown(restored)
    assert "0<a<1" in markdown and "x<0" in markdown


def test_recognition_preserves_ro_index_space_without_extra_picture_files(
    monkeypatch, tmp_path
):
    image, table, first, second = scene()
    calls = []

    async def table_request(_image, _options=None):
        calls.append("table")
        return "<table><tr><td><tit>2</td><td><tit>1</td></tr></table>"

    monkeypatch.setattr(settings.pipeline, "table_image_placeholders", True)
    monkeypatch.setattr(settings.pipeline, "image_output_dir", str(tmp_path))
    monkeypatch.setattr(settings.oxr_model, "retry_times", 1)
    monkeypatch.setattr(
        "oxr.pipeline.recognition.model_client.recognize_table", table_request
    )
    result = asyncio.run(
        recognize_pages(
            [[second, table, first]], [image], "demo.png", asyncio.Semaphore(1)
        )
    )
    assert result[0] == [second, table, first]
    assert table.content.count("<img ") == 2
    assert first.generated_role == second.generated_role == TABLE_IMAGE_ROLE
    assert calls == ["table"]
    assert not list(tmp_path.iterdir())


def test_table_recognition_uses_same_model_text_fallback_after_structure_fails(monkeypatch):
    image, table, _, _ = scene()
    calls = []

    async def fail_table(*_args, **_kwargs):
        raise ValueError("table structure could not be recovered")

    async def read_table_as_text(_image, _options=None):
        calls.append("same-model-table-text")
        return "事件：眼痛\t1级：轻微\t2级：中度"

    monkeypatch.setattr(
        "oxr.pipeline.recognition._recognize_table_image", fail_table
    )
    monkeypatch.setattr(
        "oxr.pipeline.recognition.model_client.recognize_table_as_text",
        read_table_as_text,
        raising=False,
    )
    result = asyncio.run(
        recognize_block(table, image, asyncio.Semaphore(1), "demo.png", 0)
    )

    assert calls == ["same-model-table-text"]
    assert "眼痛" in result.content and "轻微" in result.content and "中度" in result.content
    assert result.content.count("<td>") == 3
    assert "该表格区域仍有内容未能识别" not in result.content


def test_tall_table_keeps_full_context_when_first_response_is_complete(monkeypatch):
    image = np.full((520, 260, 3), 255, dtype=np.uint8)
    table = block(8, BlockType.TABLE, [10, 10, 250, 510])
    heights = []

    async def read_table(crop, _options=None, **_kwargs):
        heights.append(crop.shape[0])
        return "<table><tr><td>眼痛</td></tr></table>"

    monkeypatch.setattr("oxr.pipeline.recognition.model_client.recognize_table", read_table)

    result = asyncio.run(recognize_block(table, image, asyncio.Semaphore(1), "demo.png", 0))

    assert heights == [500]
    assert result.content.count("眼痛") == 1


def test_visible_text_at_page_boundary_is_sent_to_model(monkeypatch):
    image = np.full((100, 120, 3), 255, dtype=np.uint8)
    image[:3, 10:110] = 0
    text_block = block(8, BlockType.TEXT, [10, 0, 110, 20])

    async def read_text(crop, _options=None):
        np.testing.assert_array_equal(crop, image[:20, 10:110])
        return "Visible text"

    monkeypatch.setattr("oxr.pipeline.recognition.model_client.recognize_text", read_text)

    result = asyncio.run(recognize_block(text_block, image, asyncio.Semaphore(1), "demo.png", 0))

    assert result.content == "Visible text"


def test_failed_table_slice_does_not_replace_structured_sibling(monkeypatch):
    image, table, _, _ = scene()
    text_heights = []

    async def read_table(crop, _options=None, **_kwargs):
        if crop.shape[0] == 160:
            return ModelOutput("<table>", finish_reason="length")
        if crop.shape[0] == 79:
            return '<table><tr><td colspan="2">保留原合并格</td></tr></table>'
        return "broken structure"

    def split(crop, max_height):
        return [(crop[:79], 0, 79), (crop[79:], 79, 160)]

    async def read_text(crop, _options=None):
        text_heights.append(crop.shape[0])
        return "眼痛\t轻微"

    monkeypatch.setattr("oxr.pipeline.recognition.model_client.recognize_table", read_table)
    monkeypatch.setattr("oxr.pipeline.recognition.model_client.recognize_table_as_text", read_text)
    monkeypatch.setattr("oxr.pipeline.recognition.split_table_image_with_bounds", split)

    result = asyncio.run(recognize_block(table, image, asyncio.Semaphore(1), "demo.png", 0))

    assert text_heights == [81]
    assert 'colspan="2">保留原合并格' in result.content
    assert "眼痛" in result.content


def test_truncated_table_text_fallback_splits_into_smaller_model_requests(monkeypatch):
    image, table, _, _ = scene()
    heights = []

    async def fail_table(*_args, **_kwargs):
        raise ValueError("table structure could not be recovered")

    async def read_table_as_text(crop, _options=None):
        heights.append(crop.shape[0])
        if crop.shape[0] > 100:
            return ModelOutput("partial", finish_reason="length")
        return "识别行"

    def split_text(_image, max_height):
        if max_height >= 300:
            return [_image]
        return [_image[:80], _image[80:]]

    monkeypatch.setattr("oxr.pipeline.recognition._recognize_table_image", fail_table)
    monkeypatch.setattr("oxr.pipeline.recognition.split_table_image", split_text)
    monkeypatch.setattr(
        "oxr.pipeline.recognition.model_client.recognize_table_as_text",
        read_table_as_text,
    )
    result = asyncio.run(
        recognize_block(table, image, asyncio.Semaphore(1), "demo.png", 0)
    )

    assert heights == [160, 80, 80]
    assert result.content.count("识别行") == 2


def test_table_text_fallback_restores_html_returned_as_escaped_text(monkeypatch):
    image, table, _, _ = scene()

    async def fail_table(*_args, **_kwargs):
        raise ValueError("table structure could not be recovered")

    async def html_as_text(_image, _options=None):
        return (
            "&lt;table&gt;&lt;tr&gt;&lt;td&gt;事件&lt;/td&gt;"
            "&lt;td&gt;分级1&lt;/td&gt;&lt;/tr&gt;"
            "&lt;tr&gt;&lt;td&gt;眼痛&lt;/td&gt;&lt;td&gt;轻微&lt;/td&gt;"
            "&lt;/tr&gt;&lt;/table&gt;"
        )

    monkeypatch.setattr("oxr.pipeline.recognition._recognize_table_image", fail_table)
    monkeypatch.setattr(
        "oxr.pipeline.recognition.model_client.recognize_table_as_text", html_as_text
    )
    result = asyncio.run(
        recognize_block(table, image, asyncio.Semaphore(1), "demo.png", 0)
    )

    assert result.content.count("<table") == 1
    assert result.content.count("<tr>") == 2
    assert "&lt;table&gt;" not in result.content
    assert "眼痛" in result.content and "轻微" in result.content


def test_truncated_table_text_fallback_returns_partial_model_text(monkeypatch):
    image, table, _, _ = scene()
    heights = []

    async def fail_table(*_args, **_kwargs):
        raise ValueError("table structure could not be recovered")

    async def truncated_text(crop, _options=None):
        heights.append(crop.shape[0])
        return ModelOutput("已识别到的单元格内容", finish_reason="length")

    def split_in_half(crop, max_height):
        if max_height >= 300:
            return [crop]
        return [crop[:80], crop[80:]]

    monkeypatch.setattr("oxr.pipeline.recognition._recognize_table_image", fail_table)
    monkeypatch.setattr("oxr.pipeline.recognition.split_table_image", split_in_half)
    monkeypatch.setattr(
        "oxr.pipeline.recognition.model_client.recognize_table_as_text",
        truncated_text,
    )
    result = asyncio.run(
        recognize_block(table, image, asyncio.Semaphore(1), "demo.png", 0)
    )

    assert heights == [160, 80, 80]
    assert result.type == BlockType.TABLE
    assert result.content.count("已识别到的单元格内容") == 2
    assert "该表格区域仍有内容未能识别" not in result.content


@pytest.mark.parametrize(
    "failure", ["missing", "length", "request", "retry_request", "tiny"]
)
def test_recognition_fallback_ocr_preserves_table_text(monkeypatch, failure):
    image, table, first, second = scene()
    calls = []
    text_calls = []

    async def table_request(_image, _options=None):
        calls.append(_image.shape)
        if failure == "request" or failure == "retry_request" and len(calls) == 2:
            raise RuntimeError("test failure")
        if failure in {"length", "retry_request"}:
            return ModelOutput(
                "<table><tr><td><tit>1 <tit>2</td></tr></table>", finish_reason="length"
            )
        return "<table><tr><td><tit>1</td></tr></table>"

    async def text_request(_image, _options=None):
        text_calls.append("text")
        return "补充识别文本"

    if failure == "tiny":
        first.bbox = [20, 20, 25, 25]
    monkeypatch.setattr(settings.oxr_model, "retry_times", 1)
    monkeypatch.setattr(
        "oxr.pipeline.recognition.model_client.recognize_table", table_request
    )
    monkeypatch.setattr(
        "oxr.pipeline.recognition.model_client.recognize_table_as_text", text_request
    )
    result = asyncio.run(
        recognize_block(
            table,
            image,
            asyncio.Semaphore(1),
            "demo.png",
            0,
            table_images=[first, second],
        )
    )
    assert result.type == BlockType.TABLE
    assert result.format == ContentFormat.HTML
    assert "补充识别文本" in result.content
    assert "该表格区域仍有内容未能识别" not in result.content
    assert first.generated_role == second.generated_role == TABLE_IMAGE_ROLE
    assert text_calls == ["text"]
    assert len(calls) == ({"tiny": 0, "length": 2, "retry_request": 2}.get(failure, 1))


@pytest.mark.parametrize("provides_ro", [True, False])
@pytest.mark.parametrize("table_format", ["html", "markdown"])
@pytest.mark.parametrize("full_overlap", [True, False])
def test_full_document_restores_images_before_overlap_cleanup(
    monkeypatch, provides_ro, table_format, full_overlap
):
    image, table, first, second = scene()

    async def layout(*_):
        return [[second, table, first]]

    async def order(*_):
        return [[2, 1, 0]]

    async def table_request(*_):
        return "<table><tr><td><tit>1 <tit>2</td></tr></table>"

    monkeypatch.setattr(orchestrator, "_decode_document_images", lambda *_: [image])
    monkeypatch.setattr(orchestrator, "analyze_layouts", layout)
    monkeypatch.setattr(orchestrator, "resolve_reading_orders", order)
    monkeypatch.setattr(
        settings.layout_analyze_model, "provides_reading_order", provides_ro
    )
    monkeypatch.setattr(settings.pipeline, "table_image_placeholders", True)
    monkeypatch.setattr(settings.pipeline, "full_overlap_dedupe", full_overlap)
    monkeypatch.setattr(settings.oxr_model, "retry_times", 1)
    monkeypatch.setattr(
        "oxr.pipeline.recognition.model_client.recognize_table", table_request
    )
    output = asyncio.run(
        orchestrator.process_document(
            b"input",
            "demo.png",
            ["json", "markdown"],
            element_formats={"table": table_format},
        )
    )
    blocks = output["outputs"]["json"]["pages"][0]["blocks"]
    assert len(blocks) == 1 and blocks[0]["type"] == "Table"
    image_marker = "<img " if table_format == "html" else "![](data:image/png;base64,"
    assert blocks[0]["format"] == table_format
    assert blocks[0]["content"].count(image_marker) == 2
    assert output["outputs"]["markdown"].count(image_marker) == 2


def test_feature_off_keeps_unmasked_table_request_and_standalone_picture(
    monkeypatch, tmp_path
):
    image, table, first, _ = scene()
    seen = []

    async def table_request(crop, *_):
        seen.append(crop.copy())
        return "<table><tr><td>plain</td></tr></table>"

    monkeypatch.setattr(settings.pipeline, "table_image_placeholders", False)
    monkeypatch.setattr(settings.pipeline, "image_output_dir", str(tmp_path))
    monkeypatch.setattr(
        "oxr.pipeline.recognition.model_client.recognize_table", table_request
    )
    result = asyncio.run(
        recognize_pages([[table, first]], [image], "demo.png", asyncio.Semaphore(1))
    )
    np.testing.assert_array_equal(seen[0], image[10:170, 10:250])
    assert first.generated_role == ""
    assert first.content.startswith("![](")
    assert len(result[0]) == 2
