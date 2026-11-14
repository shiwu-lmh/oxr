"""The OXR Online Demo's upload, preview, and result endpoints."""

import asyncio
from contextlib import suppress
import logging
import math
from pathlib import Path
import time

import cv2
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from oxr.pipeline import orchestrator
from oxr.pipeline.errors import get_public_error_message
from oxr.server.demo_store import DemoStore


router = APIRouter(tags=["OXR Online Demo"])
LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"code": status, "message": message, "data": None}, status_code=status)


def _response(data: dict) -> dict:
    return {"code": 0, "message": "success", "data": data}



class _TableRowsParser:
    def __init__(self):
        from html.parser import HTMLParser

        class Parser(HTMLParser):
            def __init__(self, outer):
                super().__init__(convert_charrefs=True)
                self.outer = outer
                self.in_cell = False

            def handle_starttag(self, tag, attrs):
                if tag == "tr":
                    self.outer.current_row = []
                elif tag in {"td", "th"}:
                    self.in_cell = True
                    self.outer.current_cell = ""

            def handle_data(self, data):
                if self.in_cell:
                    self.outer.current_cell += data

            def handle_endtag(self, tag):
                if tag in {"td", "th"} and self.in_cell:
                    self.outer.current_row.append(" ".join(self.outer.current_cell.split()))
                    self.in_cell = False
                elif tag == "tr" and self.outer.current_row:
                    self.outer.rows.append(self.outer.current_row)
                    self.outer.current_row = []

        self.rows = []
        self.current_row = []
        self.current_cell = ""
        self.parser = Parser(self)

    def feed(self, text):
        self.parser.feed(text)


def _markdown_docx(markdown: str, title: str):
    from io import BytesIO
    from html import unescape
    from html.parser import HTMLParser
    import re
    from docx import Document
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor

    class PlainTextParser(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts = []

        def handle_data(self, data):
            self.parts.append(data)

    def plain(text):
        parser = PlainTextParser()
        parser.feed(text)
        return " ".join("".join(parser.parts).split())

    def inline(text):
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", text)
        text = plain(text)
        for marker in ("**", "__", "~~", chr(96), "*"):
            text = text.replace(marker, "")
        return unescape(text).strip()

    def markdown_row(line):
        return [inline(cell.replace(r"\|", "|")) for cell in line.strip().strip("|").split("|")]

    def add_table(rows):
        if not rows:
            return
        width = max(len(row) for row in rows)
        table = document.add_table(rows=0, cols=width)
        table.style = "Table Grid"
        table.autofit = True
        for row_index, row in enumerate(rows):
            cells = table.add_row().cells
            for column, value in enumerate(row + [""] * (width - len(row))):
                cells[column].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                cells[column].text = value
                for paragraph in cells[column].paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(9)
                        if row_index == 0:
                            run.bold = True
        document.add_paragraph()

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    normal = document.styles["Normal"]
    normal.font.name = "宋体"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", "宋体"
    )
    for style_name, size in (("Title", 18), ("Heading 1", 16), ("Heading 2", 14), ("Heading 3", 12)):
        style = document.styles[style_name]
        style.font.name = "黑体"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(31, 78, 121)
        style._element.rPr.rFonts.set(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", "黑体"
        )
    if title:
        heading = document.add_paragraph(style="Title")
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        heading.add_run(title)

    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    index = 0
    in_code = False
    code_lines = []
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith(chr(96) * 3):
            if in_code:
                paragraph = document.add_paragraph()
                run = paragraph.add_run("\n".join(code_lines))
                run.font.name = "Consolas"
                run.font.size = Pt(9)
                code_lines = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code_lines.append(lines[index])
            index += 1
            continue
        if not stripped:
            index += 1
            continue
        if "<table" in stripped.lower():
            html_table = [stripped]
            while "</table>" not in html_table[-1].lower() and index + 1 < len(lines):
                index += 1
                html_table.append(lines[index].strip())
                if not lines[index].strip() and "</table>" not in "".join(html_table).lower():
                    break
            parser = _TableRowsParser()
            parser.feed("\n".join(html_table))
            add_table([[inline(cell) for cell in row] for row in parser.rows])
            index += 1
            continue
        if stripped.startswith("|") and "|" in stripped[1:]:
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = markdown_row(lines[index])
                if row and not all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in row):
                    rows.append(row)
                index += 1
            add_table(rows)
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            document.add_paragraph(
                inline(heading.group(2)), style=f"Heading {min(len(heading.group(1)), 3)}"
            )
        elif re.match(r"^[-*+]\s+", stripped):
            document.add_paragraph(
                inline(re.sub(r"^[-*+]\s+", "", stripped)), style="List Bullet"
            )
        elif re.match(r"^\d+[.)]\s+", stripped):
            document.add_paragraph(
                inline(re.sub(r"^\d+[.)]\s+", "", stripped)), style="List Number"
            )
        elif stripped.startswith(">"):
            quote = document.add_paragraph(style="Quote")
            quote.add_run(inline(stripped.lstrip("> ")))
        elif stripped not in {"---", "***", "___"}:
            document.add_paragraph(inline(stripped))
        index += 1
    if in_code and code_lines:
        paragraph = document.add_paragraph()
        run = paragraph.add_run("\n".join(code_lines))
        run.font.name = "Consolas"
        run.font.size = Pt(9)
    output = BytesIO()
    document.save(output)
    output.seek(0)
    return output


@router.post("/export/docx")
async def export_markdown_docx(request: Request):
    try:
        payload = await request.json()
    except (ValueError, TypeError):
        return _error("Word 导出请求格式无效。", 400)
    if not isinstance(payload, dict):
        return _error("Word 导出请求格式无效。", 400)
    markdown = payload.get("markdown")
    file_name = payload.get("file_name", "document")
    if not isinstance(markdown, str) or not markdown.strip():
        return _error("没有可导出的 Markdown 内容。", 422)
    if len(markdown.encode("utf-8")) > 20 * 1024 * 1024:
        return _error("识别结果过大，无法导出 Word。", 413)
    safe_name = Path(str(file_name).replace("\\", "/")).name
    stem = Path(safe_name).stem or "document"
    try:
        output = await asyncio.to_thread(_markdown_docx, markdown, stem)
    except ImportError:
        LOGGER.exception("python-docx is unavailable")
        return _error("Word 导出依赖未安装，请安装项目依赖后重启服务。", 503)
    except Exception as exc:
        LOGGER.warning("Word export failed (error_type=%s)", type(exc).__name__)
        return _error("Word 文档生成失败。", 500)
    from urllib.parse import quote
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(stem + '-OXR.docx')}"},
    )


def _decode_and_publish(store: DemoStore, job_id: str, data: bytes, file_name: str):
    images = orchestrator._decode_document_images(data, file_name)
    directory = store.job_dir(job_id) / "images"
    pages = []
    assets = []
    for page_num, image in enumerate(images):
        asset_id = f"page-{page_num:04d}.png"
        if not cv2.imwrite(str(directory / asset_id), image):
            raise OSError("Unable to create the page preview. Temporary storage may be full.")
        store.ensure_capacity()
        height, width = image.shape[:2]
        pages.append({
            "page_num": page_num,
            "page_size": {"w": width, "h": height},
            "image_url": f"/v1/demo/jobs/{job_id}/assets/{asset_id}",
        })
        assets.append(asset_id)
    store.update(job_id, status="processing", pages=pages, assets=assets)
    return images


async def _process_job(
    store: DemoStore, job_id: str, data: bytes, file_name: str,
    model_options: dict[str, float] | None = None,
    keep_header_footer: bool = True,
) -> None:
    decode_task = None
    publish_task = None
    try:
        started_at = time.perf_counter()
        store.update(job_id, status="rendering")
        decode_task = asyncio.create_task(asyncio.to_thread(_decode_and_publish, store, job_id, data, file_name))
        images = await asyncio.shield(decode_task)
        result = await orchestrator.process_decoded_document(
            images,
            file_name,
            ["markdown", "json"],
            keep_header_footer=keep_header_footer,
            model_options=model_options,
            demo=True,
            asset_dir=store.job_dir(job_id) / "images",
            asset_url_prefix=f"/v1/demo/jobs/{job_id}/assets/",
            started_at=started_at,
        )
        publish_task = asyncio.create_task(asyncio.to_thread(store.publish_result, job_id, result))
        await asyncio.shield(publish_task)
    except asyncio.CancelledError:
        # Threads cannot be cancelled: finish publication before writing failed,
        # so a late decoder cannot resurrect a stopped job as processing.
        if decode_task is not None:
            await asyncio.gather(decode_task, return_exceptions=True)
        if publish_task is not None:
            await asyncio.gather(publish_task, return_exceptions=True)
        with suppress(KeyError):
            if store.get(job_id)["status"] != "success":
                store.update(job_id, status="failed", error="The server stopped while processing this document.")
        raise
    except Exception as error:
        LOGGER.warning(
            "Demo processing failed (job_id=%s, error_type=%s)", job_id, type(error).__name__,
        )
        if isinstance(error, OSError):
            with suppress(OSError):
                store.discard_payload(job_id)
        public_message = (
            "Temporary storage is unavailable or full. Try again later."
            if isinstance(error, OSError) else get_public_error_message(error)
        )
        with suppress(KeyError, OSError):
            store.update(
                job_id, status="failed", error=public_message,
                **({"pages": [], "assets": []} if isinstance(error, OSError) else {}),
            )


async def _maintain_jobs(app) -> None:
    while True:
        await asyncio.sleep(30)
        # Keep active work alive even when recognition lasts longer than the TTL.
        for job_id in tuple(app.state.demo_tasks):
            with suppress(KeyError, FileNotFoundError):
                app.state.demo_store.keep_alive(job_id)
        await asyncio.to_thread(app.state.demo_store.cleanup)


async def start_demo(app) -> None:
    app.state.demo_store = DemoStore()
    app.state.demo_tasks = {}
    await asyncio.to_thread(app.state.demo_store.cleanup)
    app.state.demo_maintenance = asyncio.create_task(_maintain_jobs(app))


async def stop_demo(app) -> None:
    app.state.demo_maintenance.cancel()
    tasks = list(app.state.demo_tasks.values())
    for task in tasks:
        task.cancel()
    await asyncio.gather(app.state.demo_maintenance, *tasks, return_exceptions=True)


# OXR Online Demo: independent of the existing SoMark-compatible parse API.
@router.post("/demo/jobs")
async def submit_demo_job(
    request: Request,
    file: UploadFile = File(...),
    temperature: str = Form("0"),
    top_p: str = Form("1"),
    repetition_penalty: str = Form("1"),
    keep_header_footer: bool = Form(True),
):
    # Read the raw form as well: FastAPI substitutes defaults for empty fields,
    # but explicitly blank model parameters must be rejected rather than reset.
    form = await request.form()
    model_options = {}
    for name, default, minimum, maximum in (
        ("temperature", temperature, 0.0, 2.0),
        ("top_p", top_p, 0.0, 1.0),
        ("repetition_penalty", repetition_penalty, 0.1, 2.0),
    ):
        try:
            value = float(form.get(name, default))
        except (TypeError, ValueError):
            await file.close()
            return _error(f"{name} must be a number between {minimum:g} and {maximum:g}.", 422)
        if not math.isfinite(value) or not minimum <= value <= maximum:
            await file.close()
            return _error(f"{name} must be a finite number between {minimum:g} and {maximum:g}.", 422)
        model_options[name] = value
    file_name = Path((file.filename or "").replace("\\", "/")).name
    if Path(file_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        await file.close()
        return _error("Upload a PDF or a supported image file.", 415)
    data = bytearray()
    try:
        while chunk := await file.read(1024 * 1024):
            data.extend(chunk)
            if len(data) > MAX_UPLOAD_BYTES:
                return _error("The file exceeds the 100 MiB upload limit.", 413)
    finally:
        await file.close()
    if not data:
        return _error("The uploaded file is empty.", 422)
    store = request.app.state.demo_store
    try:
        manifest = await asyncio.to_thread(store.create, file_name)
    except OSError:
        return _error("The demo's temporary storage is unavailable or full. Try again later.", 503)
    job_id = manifest["id"]
    task = asyncio.create_task(_process_job(
        store, job_id, bytes(data), file_name, model_options, keep_header_footer,
    ))
    request.app.state.demo_tasks[job_id] = task
    task.add_done_callback(lambda _: request.app.state.demo_tasks.pop(job_id, None))
    return _response({"id": job_id, "status": "queued"})


@router.get("/demo/jobs/{job_id}")
async def demo_job_status(request: Request, job_id: str):
    try:
        manifest = request.app.state.demo_store.get(job_id)
    except KeyError:
        return _error("This document is unavailable or has expired. Upload it again.", 404)
    return _response({key: value for key, value in manifest.items() if key not in {"assets", "expires_at"}})


@router.get("/demo/jobs/{job_id}/result")
async def demo_job_result(request: Request, job_id: str):
    try:
        result = await asyncio.to_thread(request.app.state.demo_store.result, job_id)
    except (KeyError, FileNotFoundError):
        return _error("This document is unavailable or has expired. Upload it again.", 404)
    except ValueError:
        return _error("The document is not ready yet.", 409)
    return _response(result)


@router.get("/demo/jobs/{job_id}/assets/{asset_id}")
async def demo_job_asset(request: Request, job_id: str, asset_id: str):
    try:
        path = request.app.state.demo_store.asset(job_id, asset_id)
    except KeyError:
        return _error("The image is unavailable or has expired.", 404)
    return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})
