"""PDF2Word 法务字段提取接口（端到端、内存处理版）。

2026-07-21：批量 results 的每个文件对象新增顶层 document_type 字段。
2026-07-23：新增 visual=2 前端交接结构；visual=0/1 与提取逻辑保持不变。

处理链路：
    上传 PDF / 图片
        -> 复用现有 web_server.py 中已经初始化的 PP-Structure predictor
        -> 直接读取 PP-Structure 的版面识别结果
        -> 调用 legal_ocr_core.py 提取法务字段
        -> 返回 JSON

本文件刻意不调用 ``convert_info_docx_multi_page``，因此不会生成、保存或重新读取
Word 文档。OCR 模型、模型路径和推理参数全部沿用现有 ``web_server.py``，本接口
不再建立另一套 OCR 配置。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import traceback
import uuid
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import fitz
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# 严格复用 PDF2Word 页面现有的 OCR 实例、锁和图像预处理方法。
# 导入 web_server 时，其全局 predictor 会按照原项目参数初始化一次。
try:
    from .web_server import ocr_lock, predictor, redink_remover
except ImportError:  # 允许在当前目录直接执行：python interfaces-tmp.py
    from web_server import ocr_lock, predictor, redink_remover

try:
    from .legal_ocr_core import (
        SourceDocument,
        TextLine,
        api_schema,
        build_visual2_content,
        extract_legal_document,
        normalize_text,
    )
except ImportError:
    from legal_ocr_core import (
        SourceDocument,
        TextLine,
        api_schema,
        build_visual2_content,
        extract_legal_document,
        normalize_text,
    )

try:
    from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
except ImportError as exc:  # 运行目录或 PYTHONPATH 配置错误时给出明确提示
    raise ImportError(
        "无法导入 ppstructure.recovery.recovery_to_doc.sorted_layout_boxes；"
        "请按照原 PDF2Word 项目的方式从 PaddleOCR 项目根目录启动。"
    ) from exc


# ---------------------------------------------------------------------------
# Flask 配置
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.json.ensure_ascii = False
CORS(app)

MAX_CONTENT_MB = int(os.getenv("LEGAL_EXTRACT_MAX_CONTENT_MB", "80"))
HOST = os.getenv("LEGAL_EXTRACT_HOST", "0.0.0.0")
PORT = int(os.getenv("LEGAL_EXTRACT_PORT", "8006"))
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ---------------------------------------------------------------------------
# 通用辅助函数
# ---------------------------------------------------------------------------


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _safe_filename(filename: str) -> str:
    """只保留文件名，避免将客户端路径带入响应或日志。"""
    name = Path(str(filename or "")).name.strip()
    return name or "uploaded_file"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bbox_to_list(value: Any) -> Optional[List[float]]:
    """兼容 PP-Structure 中 [x1,y1,x2,y2] 和四点坐标两种形式。"""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None

    if arr.size == 4:
        return [float(item) for item in arr.reshape(-1).tolist()]

    if arr.ndim == 2 and arr.shape[1] >= 2:
        xs = arr[:, 0]
        ys = arr[:, 1]
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    return None


def _mean_confidence(values: Iterable[Any]) -> Optional[float]:
    scores: List[float] = []
    for value in values:
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= score <= 1.0:
            scores.append(score)
    if not scores:
        return None
    return sum(scores) / len(scores)


class _TableHTMLTextParser(HTMLParser):
    """从 PP-Structure 表格 HTML 中读取单元格文本，不引入额外依赖。"""

    def __init__(self) -> None:
        super().__init__()
        self._in_cell = False
        self._cell_parts: List[str] = []
        self._current_row: List[str] = []
        self.rows: List[List[str]] = []

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"}:
            self._in_cell = True
            self._cell_parts = []
        elif tag == "br" and self._in_cell:
            self._cell_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"}:
            value = normalize_text("".join(self._cell_parts))
            self._current_row.append(value)
            self._cell_parts = []
            self._in_cell = False
        elif tag == "tr":
            values = [value for value in self._current_row if value]
            if values:
                self.rows.append(values)
            self._current_row = []


def _table_rows_from_html(html: str) -> List[str]:
    if not html:
        return []
    parser = _TableHTMLTextParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # 表格 HTML 偶有不完整标签；失败时保守地去标签，不影响其他区域。
        plain = normalize_text(re.sub(r"<[^>]+>", " ", html))
        return [plain] if plain else []
    return [" | ".join(row) for row in parser.rows if row]


def _extract_text_items(region_res: Any) -> List[Tuple[str, Optional[float], Optional[List[float]]]]:
    """将一个 PP-Structure 区域转换为文本项。

    返回元素：``(text, confidence, bbox)``。
    """
    items: List[Tuple[str, Optional[float], Optional[List[float]]]] = []

    if isinstance(region_res, dict):
        html = region_res.get("html")
        if isinstance(html, str):
            for row_text in _table_rows_from_html(html):
                items.append((row_text, None, None))

        # 兼容某些表格模型返回的 rec_res / text / texts。
        for key in ("rec_res", "texts", "text"):
            value = region_res.get(key)
            if value is None:
                continue
            items.extend(_extract_text_items(value))
        return items

    if isinstance(region_res, str):
        text = normalize_text(region_res)
        if text:
            items.append((text, None, None))
        return items

    if isinstance(region_res, tuple):
        # 常见 OCR 元素形态：(text, confidence)
        if region_res and isinstance(region_res[0], str):
            text = normalize_text(region_res[0])
            confidence = None
            if len(region_res) > 1:
                confidence = _mean_confidence([region_res[1]])
            if text:
                items.append((text, confidence, None))
            return items
        for value in region_res:
            items.extend(_extract_text_items(value))
        return items

    if not isinstance(region_res, list):
        return items

    for value in region_res:
        if isinstance(value, dict):
            text = value.get("text") or value.get("transcription")
            if text is not None:
                cleaned = normalize_text(str(text))
                if cleaned:
                    confidence = _mean_confidence(
                        [value.get("confidence"), value.get("score")]
                    )
                    bbox = _bbox_to_list(
                        value.get("text_region")
                        or value.get("bbox")
                        or value.get("points")
                    )
                    items.append((cleaned, confidence, bbox))
                continue
            items.extend(_extract_text_items(value))
            continue

        if isinstance(value, (list, tuple, str)):
            items.extend(_extract_text_items(value))

    return items


def _source_from_structure_results(
    *,
    filename: str,
    extension: str,
    page_results: List[Dict[str, Any]],
    file_hash: str,
) -> SourceDocument:
    """读取将要传给 Word 恢复模块的同一份 PP-Structure 结果。"""
    lines: List[TextLine] = []
    text_parts: List[str] = []

    for page_item in page_results:
        page_index = int(page_item["page_index"])
        for region_index, region in enumerate(page_item["res"]):
            region_type = str(region.get("type") or "unknown")
            region_bbox = _bbox_to_list(region.get("bbox"))
            text_items = _extract_text_items(region.get("res"))

            for item_index, (text, confidence, item_bbox) in enumerate(text_items):
                if not text:
                    continue
                lines.append(
                    TextLine(
                        text=text,
                        page=page_index + 1,
                        paragraph=len(lines),
                        bbox=item_bbox or region_bbox,
                        confidence=confidence,
                        block_type=region_type,
                    )
                )
                text_parts.append(text)

    text = "\n".join(text_parts).strip()
    warnings: List[str] = []
    if not text:
        warnings.append("PP-Structure 已完成推理，但未读取到可用于字段提取的文本。")

    return SourceDocument(
        filename=filename,
        extension=extension,
        text=text,
        lines=lines,
        extraction_method="ppstructure_memory",
        page_count=len(page_results),
        sha256=file_hash,
        warnings=warnings,
    )


def _source_from_pdf_text(
    *,
    filename: str,
    extension: str,
    pdf: fitz.Document,
    file_hash: str,
) -> SourceDocument:
    """读取文字版 PDF，并按页面视觉坐标重建阅读顺序。

    PDF 内部对象顺序并不一定等于页面上的阅读顺序。若直接遍历
    ``page.get_text("dict")`` 返回的 block，视觉上位于“事实与理由”
    之前的某一条诉讼请求，可能在对象流中排到该标题之后，进而造成
    段落串栏。这里先收集整页文本行，再统一按 ``(y0, x0)`` 排序。
    """
    lines: List[TextLine] = []
    text_parts: List[str] = []

    for page_index in range(pdf.page_count):
        page = pdf[page_index]
        try:
            page_dict = page.get_text("dict", sort=True)
        except TypeError:
            # 兼容较旧的 PyMuPDF；后续仍会自行按 bbox 排序。
            page_dict = page.get_text("dict")

        page_lines: List[Tuple[float, float, int, str, Optional[List[float]]]] = []
        sequence = 0
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            block_bbox = _bbox_to_list(block.get("bbox"))
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = normalize_text(
                    "".join(str(span.get("text", "")) for span in spans)
                )
                if not line_text:
                    continue
                bbox = _bbox_to_list(line.get("bbox")) or block_bbox
                if bbox:
                    x0, y0 = float(bbox[0]), float(bbox[1])
                else:
                    # 没有坐标时保持原始相对次序，并排在有坐标文本之后。
                    x0, y0 = float(sequence), float("inf")
                page_lines.append((y0, x0, sequence, line_text, bbox))
                sequence += 1

        page_lines.sort(key=lambda item: (item[0], item[1], item[2]))
        for _, _, _, line_text, bbox in page_lines:
            lines.append(
                TextLine(
                    text=line_text,
                    page=page_index + 1,
                    paragraph=len(lines),
                    bbox=bbox,
                    confidence=None,
                    block_type="pdf_text",
                )
            )
            text_parts.append(line_text)

    return SourceDocument(
        filename=filename,
        extension=extension,
        text="\n".join(text_parts).strip(),
        lines=lines,
        extraction_method="pdf_text_memory_visual_order",
        page_count=pdf.page_count,
        sha256=file_hash,
    )


def _images_from_upload(
    data: bytes,
    extension: str,
    remove_red_seal: bool,
) -> List[np.ndarray]:
    """完全沿用原接口的 PDF 2 倍渲染与 OpenCV 图片读取方式。"""
    images: List[np.ndarray] = []

    if extension == ".pdf":
        with fitz.open(stream=data, filetype="pdf") as pdf:
            for page_index in range(pdf.page_count):
                page = pdf[page_index]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image_array = np.frombuffer(pix.tobytes(), np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                if remove_red_seal:
                    image = redink_remover(image)
                images.append(image)
        return images

    image_array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is not None:
        if remove_red_seal:
            image = redink_remover(image)
        images.append(image)
    return images


def _run_existing_ppstructure(images: List[np.ndarray]) -> List[Dict[str, Any]]:
    """调用 web_server.py 中的同一个 predictor，不进行绘图和 Word 恢复。"""
    page_results: List[Dict[str, Any]] = []

    with ocr_lock:
        for page_index, image in enumerate(images):
            result, _ = predictor(image)
            _, width, _ = image.shape
            result = sorted_layout_boxes(result, width)
            for region in result:
                region["page_index"] = page_index
            page_results.append(
                {
                    "res": result,
                    "page_index": page_index,
                }
            )

    return page_results


def _process_uploaded_file(
    *,
    data: bytes,
    filename: str,
    force_ocr: bool,
    remove_red_seal: bool,
    document_type_hint: Optional[str],
    include_raw_text: bool,
    include_layout_text: bool,
) -> Dict[str, Any]:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"不支持的文件类型：{extension or '无扩展名'}；支持：{supported}")
    if not data:
        raise ValueError("上传文件为空")

    file_hash = _sha256(data)
    source: Optional[SourceDocument] = None

    # 与原 web_server.py 一致：PDF 在未 force_ocr 时先检查文字层；只要存在
    # 可提取文本，就走直接分支，不再运行 PP-Structure。
    if extension == ".pdf" and not force_ocr:
        with fitz.open(stream=data, filetype="pdf") as pdf:
            text_count = sum(len(pdf[page].get_text().strip()) for page in range(pdf.page_count))
            if text_count > 0:
                source = _source_from_pdf_text(
                    filename=filename,
                    extension=extension,
                    pdf=pdf,
                    file_hash=file_hash,
                )

    if source is None:
        images = _images_from_upload(data, extension, remove_red_seal)
        if not images:
            raise ValueError("无法读取上传的图片或 PDF")
        page_results = _run_existing_ppstructure(images)
        source = _source_from_structure_results(
            filename=filename,
            extension=extension,
            page_results=page_results,
            file_hash=file_hash,
        )

    result = extract_legal_document(
        source,
        document_type_hint=document_type_hint,
    )
    result["processing"] = {
        "mode": "end_to_end_memory",
        "word_generated": False,
        "word_read_back": False,
        "method": source.extraction_method,
        "page_count": source.page_count,
    }
    if source.warnings:
        result.setdefault("warnings", []).extend(source.warnings)
    if include_raw_text:
        result["raw_text"] = source.text
    if include_layout_text:
        result["layout_text_lines"] = [line.to_dict() for line in source.lines]
    return result



# ---------------------------------------------------------------------------
# 返回结果辅助函数
# ---------------------------------------------------------------------------


def _get_document_type(result: Dict[str, Any]) -> str:
    """从完整法务提取结果中读取统一的文书类型名称。"""
    classification = result.get("classification") or {}
    patch = result.get("frontend_patch") or {}
    return str(
        classification.get("document_type_name")
        or classification.get("page_name")
        or classification.get("document_type")
        or patch.get("page_name")
        or patch.get("document_type")
        or "未识别"
    )


# ---------------------------------------------------------------------------
# 简化 JSON 返回
# ---------------------------------------------------------------------------


def _build_key_value_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """仅保留人工测试和前端联调所需的关键 key-value。

    不返回全文、bbox、置信度、证据位置、内部字段对象等详细信息。
    不生成 HTML。
    """
    classification = result.get("classification") or {}
    patch = result.get("frontend_patch") or {}

    review_required: Dict[str, Any] = {}
    for label, detail in (patch.get("review_required") or {}).items():
        if isinstance(detail, dict) and "value" in detail:
            review_required[str(label)] = detail.get("value")
        else:
            review_required[str(label)] = detail

    blocked: Dict[str, str] = {}
    for detail in patch.get("blocked") or []:
        if isinstance(detail, dict):
            label = detail.get("frontend_label") or detail.get("key")
            if label:
                blocked[str(label)] = str(detail.get("reason") or "禁止自动填充")
        elif detail:
            blocked[str(detail)] = "禁止自动填充"

    return {
        "document": {
            "document_type": _get_document_type(result),
            "page_code": classification.get("page_code") or patch.get("page_code"),
            "page_name": classification.get("page_name") or patch.get("page_name"),
            "stage": classification.get("stage_name") or classification.get("stage"),
        },
        "safe_autofill": patch.get("safe_autofill") or {},
        "review_required": review_required,
        "blocked": blocked,
    }


def _build_visual_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """将完整批量结果转换成按文件分组、按等级区分的简化 JSON。"""
    simplified_results: List[Dict[str, Any]] = []

    for item in payload.get("results", []):
        simplified: Dict[str, Any] = {
            "index": item.get("index"),
            "filename": item.get("filename"),
            "document_type": item.get("document_type"),
            "success": bool(item.get("success")),
        }
        if item.get("success"):
            simplified.update(_build_key_value_result(item.get("result") or {}))
        else:
            simplified["error"] = item.get("error") or "处理失败"
            if item.get("error_type"):
                simplified["error_type"] = item.get("error_type")
        simplified_results.append(simplified)

    return {
        "task_id": payload.get("task_id"),
        "success": payload.get("success"),
        "visual": 1,
        "total": payload.get("total"),
        "success_count": payload.get("success_count"),
        "failure_count": payload.get("failure_count"),
        "results": simplified_results,
    }


def _ordered_json_response(payload: Dict[str, Any], status_code: int) -> Any:
    """仅用于 visual=2，确保 JSON 键顺序不被 Flask 默认排序打乱。"""
    return app.response_class(
        response=json.dumps(payload, ensure_ascii=False, sort_keys=False),
        status=status_code,
        mimetype="application/json",
    )


def _build_visual2_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """构造面向前端交接的有序 JSON；每个文件对象严格只保留六个键。"""
    handoff_results: List[Dict[str, Any]] = []

    for item in payload.get("results", []):
        result = item.get("result") or {}
        classification = result.get("classification") or {}
        patch = result.get("frontend_patch") or {}
        success = bool(item.get("success"))

        if success:
            document_type = _get_document_type(result)
            page_name = classification.get("page_name") or patch.get("page_name") or ""
            stage = classification.get("stage_name") or classification.get("stage") or ""
            content = build_visual2_content(result)
        else:
            document_type = item.get("document_type") or ""
            page_name = ""
            stage = ""
            content = {
                "处理错误": {
                    "value": item.get("error") or "处理失败",
                    "conf_level": "review_required",
                }
            }

        # Insertion order is the API contract requested for visual=2.
        handoff_results.append(
            {
                "filename": item.get("filename"),
                "document_type": document_type,
                "page_name": page_name,
                "stage": stage,
                "index": item.get("index"),
                "content": content,
            }
        )

    return {
        "task_id": payload.get("task_id"),
        "success": payload.get("success"),
        "visual": 2,
        "total": payload.get("total"),
        "success_count": payload.get("success_count"),
        "failure_count": payload.get("failure_count"),
        "results": handoff_results,
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> Any:
    return jsonify(
        {
            "service": "pdf2word-legal-field-extract",
            "mode": "end_to_end_memory",
            "description": (
                "上传 PDF/图片后复用现有 PDF2Word PP-Structure 流程，"
                "直接从内存识别结果提取法务字段，不生成 Word。"
            ),
            "endpoint": "POST /api/v1/legal/extract",
        }
    )


@app.get("/api/v1/health")
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "mode": "end_to_end_memory",
            "predictor_reused_from": "web_server.py",
            "word_generated": False,
        }
    )


@app.get("/api/v1/schemas")
def schemas() -> Any:
    return jsonify(api_schema())


@app.post("/api/v1/legal/extract")
def extract_legal_fields() -> Any:
    """端到端上传接口（支持多文件）。

    multipart/form-data：
      - files: 必填，支持多个 PDF / 图片；
      - file: 兼容旧调用，单文件；
      - force_ocr: 可选，默认 false；
      - remove_red_seal: 可选，默认 false；
      - document_type: 可选，文种提示；
      - include_raw_text: 可选，默认 false；
      - include_layout_text: 可选，默认 false；
      - visual: 可选，默认 0。visual=0 返回完整 JSON；visual=1 返回
        按等级分组的测试版简化 JSON；visual=2 返回按字段报告顺序组织的
        前端交接 JSON。
    """
    storages = [storage for storage in request.files.getlist("files") if storage and storage.filename]
    if not storages:
        single = request.files.get("file")
        if single is not None and single.filename:
            storages = [single]

    if not storages:
        return jsonify({"error": "未收到文件；multipart 字段名请使用 files（多文件）或 file（单文件）"}), 400

    force_ocr = _as_bool(request.form.get("force_ocr"), False)
    remove_red_seal = _as_bool(request.form.get("remove_red_seal"), False)
    document_type_hint = request.form.get("document_type") or None
    include_raw_text = _as_bool(request.form.get("include_raw_text"), False)
    include_layout_text = _as_bool(request.form.get("include_layout_text"), False)

    results: List[Dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    has_server_error = False

    for index, storage in enumerate(storages):
        filename = _safe_filename(storage.filename)
        data = storage.stream.read()
        item: Dict[str, Any] = {
            "index": index,
            "filename": filename,
            # 多文件返回中始终保留该键；成功后会替换为实际识别类型。
            "document_type": document_type_hint,
        }
        try:
            result = _process_uploaded_file(
                data=data,
                filename=filename,
                force_ocr=force_ocr,
                remove_red_seal=remove_red_seal,
                document_type_hint=document_type_hint,
                include_raw_text=include_raw_text,
                include_layout_text=include_layout_text,
            )
            item["document_type"] = _get_document_type(result)
            item["success"] = True
            item["result"] = result
            success_count += 1
        except ValueError as exc:
            item["success"] = False
            item["error"] = str(exc)
            item["error_type"] = "value_error"
            failure_count += 1
        except Exception as exc:
            print(traceback.format_exc())
            item["success"] = False
            item["error"] = str(exc)
            item["error_type"] = "internal_error"
            failure_count += 1
            has_server_error = True
        results.append(item)

    status_code = 200
    if success_count == 0:
        status_code = 500 if has_server_error else 400

    payload: Dict[str, Any] = {
        "task_id": str(uuid.uuid4()),
        "success": success_count > 0,
        "total": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "results": results,
    }

    # visual=0：保持原有完整 JSON，不改变任何字段。
    # visual=1：保持原有按等级组织的简化 key-value JSON。
    # visual=2：仅重排现有识别结果，输出面向前端交接的有序字段结构。
    visual_mode = str(request.form.get("visual") or "0").strip()
    if visual_mode == "2":
        return _ordered_json_response(_build_visual2_payload(payload), status_code)
    if _as_bool(visual_mode, False):
        return jsonify(_build_visual_payload(payload)), status_code

    return jsonify(payload), status_code


# 保留旧接口名，方便原临时调用方迁移；请求参数与新接口完全相同。
@app.post("/autofill_from_uploads")
def compatibility_extract() -> Any:
    return extract_legal_fields()


@app.errorhandler(413)
def file_too_large(_error: Exception) -> Any:
    return jsonify({"success": False, "error": f"请求体超过 {MAX_CONTENT_MB} MB"}), 413


if __name__ == "__main__":
    print(f"Legal extraction API: http://127.0.0.1:{PORT}")
    print("OCR predictor: reused from web_server.py")
    print("DOCX generation: disabled")
    app.run(host=HOST, port=PORT, debug=False)
