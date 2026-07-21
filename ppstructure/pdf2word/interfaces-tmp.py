"""从 PDF 转 Word 最终结果中提取法务字段的轻量接口。

重要说明：
- 本文件不初始化 PaddleOCR、PP-Structure 或任何新的 OCR 模型；
- PDF/图片到 Word 的转换继续完全使用现有 ``web_server.py`` 的 ``/upload`` 流程；
- 本接口只读取 ``web_output`` 中已经生成的最终 DOCX，再提取结构化字段。

典型调用顺序：
1. 调用现有 ``POST /upload``，得到返回值中的 ``filename``；
2. 调用本服务 ``POST /api/v1/extract/from-output``，请求体：
   ``{"filename": "xxxxxxxx_ocr.docx"}``；
3. 将返回的 ``frontend_patch`` 交给外部前端回填。

也支持直接上传已经生成好的 Word：
``POST /api/v1/extract/from-docx``，multipart 字段名为 ``files``。
"""

from __future__ import annotations

import json
import os
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

try:
    from .legal_ocr_core import api_schema, extract_legal_document, read_docx_source
except ImportError:  # 允许在当前目录直接执行 python interfaces-tmp.py
    from legal_ocr_core import api_schema, extract_legal_document, read_docx_source


BASE_DIR = Path(__file__).resolve().parent
WEB_OUTPUT_DIR = Path(
    os.getenv("PDF2WORD_OUTPUT_DIR", str(BASE_DIR / "web_output"))
).resolve()
UPLOAD_DIR = Path(
    os.getenv("DOCX_EXTRACT_UPLOAD_DIR", str(BASE_DIR / "field_extract_uploads"))
).resolve()
RESULT_DIR = Path(
    os.getenv("DOCX_EXTRACT_RESULT_DIR", str(BASE_DIR / "field_extract_results"))
).resolve()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONTENT_MB = int(os.getenv("DOCX_EXTRACT_MAX_CONTENT_MB", "80"))
MAX_FILES = int(os.getenv("DOCX_EXTRACT_MAX_FILES", "20"))
HOST = os.getenv("DOCX_EXTRACT_HOST", "0.0.0.0")
PORT = int(os.getenv("DOCX_EXTRACT_PORT", "8006"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
app.json.ensure_ascii = False
CORS(app)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _safe_docx_name(filename: str) -> str:
    """只接受 basename 形式的 DOCX 文件名，阻止目录穿越。"""
    raw = str(filename or "").strip()
    if not raw:
        raise ValueError("filename 不能为空")
    if Path(raw).name != raw:
        raise ValueError("filename 只能是文件名，不能包含目录")
    if Path(raw).suffix.lower() != ".docx":
        raise ValueError("仅支持读取 PDF 转 Word 最终生成的 .docx 文件")
    return raw


def _resolve_output_docx(filename: str) -> Path:
    safe_name = _safe_docx_name(filename)
    path = (WEB_OUTPUT_DIR / safe_name).resolve()
    if path.parent != WEB_OUTPUT_DIR:
        raise ValueError("非法文件路径")
    if not path.is_file():
        raise FileNotFoundError(f"在 web_output 中未找到文件：{safe_name}")
    return path


def _extract_one(
    docx_path: Path,
    display_name: Optional[str] = None,
    document_type_hint: Optional[str] = None,
    include_raw_text: bool = False,
) -> Dict[str, Any]:
    source = read_docx_source(docx_path, filename=display_name or docx_path.name)
    result = extract_legal_document(source, document_type_hint=document_type_hint)
    if include_raw_text:
        result["raw_text"] = source.text
    return result


def _save_result(task_id: str, payload: Dict[str, Any]) -> str:
    filename = f"{task_id}_field_extract.json"
    path = RESULT_DIR / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return filename


def _finish_payload(task_id: str, source: str, results: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "task_id": task_id,
        "source": source,
        "total": len(results),
        "results": results,
    }
    saved_file = _save_result(task_id, payload)
    payload["saved_file"] = saved_file
    payload["download_url"] = f"/download/{saved_file}"
    return payload


@app.get("/")
def root():
    return jsonify(
        {
            "service": "pdf2word-final-docx-field-extractor",
            "description": "只读取现有 PDF 转 Word 最终 DOCX，不执行 OCR。",
            "pdf2word_output_dir": str(WEB_OUTPUT_DIR),
            "endpoints": [
                "GET /api/v1/health",
                "GET /api/v1/schemas",
                "POST /api/v1/extract/from-output",
                "POST /api/v1/extract/from-docx",
                "POST /autofill_from_uploads",
                "GET /download/<filename>",
            ],
        }
    )


@app.get("/api/v1/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "mode": "final_docx_only",
            "ocr_initialized": False,
            "pdf2word_output_dir": str(WEB_OUTPUT_DIR),
            "output_dir_exists": WEB_OUTPUT_DIR.is_dir(),
        }
    )


@app.get("/api/v1/schemas")
def schemas():
    return jsonify(api_schema())


@app.post("/api/v1/extract/from-output")
def extract_from_output():
    """读取现有 ``web_output`` 中的一个或多个最终 DOCX。

    JSON 示例：
        {"filename": "abcd_材料_ocr.docx"}

    或：
        {"filenames": ["a.docx", "b.docx"], "include_raw_text": false}
    """
    body = request.get_json(silent=True) or {}
    filenames = body.get("filenames")
    if filenames is None:
        one = body.get("filename") or body.get("docx_filename")
        filenames = [one] if one else []
    if not isinstance(filenames, list):
        return jsonify({"error": "filenames 必须是字符串数组"}), 400
    filenames = [str(item).strip() for item in filenames if str(item or "").strip()]
    if not filenames:
        return jsonify({"error": "请提供 filename 或 filenames"}), 400
    if len(filenames) > MAX_FILES:
        return jsonify({"error": f"一次最多处理 {MAX_FILES} 个 Word 文件"}), 400

    task_id = str(uuid.uuid4())[:8]
    include_raw_text = _as_bool(body.get("include_raw_text"), False)
    document_type_hint = body.get("document_type")
    results: Dict[str, Any] = {}

    for filename in filenames:
        try:
            path = _resolve_output_docx(filename)
            results[filename] = _extract_one(
                path,
                display_name=filename,
                document_type_hint=document_type_hint,
                include_raw_text=include_raw_text,
            )
        except Exception as exc:
            results[filename] = {"error": str(exc)}

    return jsonify(_finish_payload(task_id, "web_output", results))


@app.post("/api/v1/extract/from-docx")
def extract_from_docx_upload():
    """直接上传一个或多个已经由 PDF 转 Word 生成的 DOCX。"""
    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        one = request.files.get("file")
        uploaded_files = [one] if one else []
    uploaded_files = [item for item in uploaded_files if item and item.filename]
    if not uploaded_files:
        return jsonify({"error": "未收到 Word 文件；multipart 字段名应为 files"}), 400
    if len(uploaded_files) > MAX_FILES:
        return jsonify({"error": f"一次最多处理 {MAX_FILES} 个 Word 文件"}), 400

    task_id = str(uuid.uuid4())[:8]
    batch_dir = UPLOAD_DIR / task_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    include_raw_text = _as_bool(request.form.get("include_raw_text"), False)
    document_type_hint = request.form.get("document_type") or None
    results: Dict[str, Any] = {}

    for storage in uploaded_files:
        original_name = str(storage.filename)
        try:
            if Path(original_name).suffix.lower() != ".docx":
                raise ValueError("本接口只接收 PDF 转 Word 最终导出的 .docx 文件")
            safe_name = secure_filename(original_name)
            if not safe_name.lower().endswith(".docx"):
                safe_name = f"{uuid.uuid4().hex[:8]}.docx"
            saved_path = batch_dir / safe_name
            storage.save(saved_path)
            results[original_name] = _extract_one(
                saved_path,
                display_name=original_name,
                document_type_hint=document_type_hint,
                include_raw_text=include_raw_text,
            )
        except Exception as exc:
            results[original_name] = {"error": str(exc)}

    return jsonify(_finish_payload(task_id, "uploaded_docx", results))


@app.post("/autofill_from_uploads")
def compatibility_autofill_from_uploads():
    """保留旧接口名；行为等同于 ``/api/v1/extract/from-docx``。"""
    return extract_from_docx_upload()


@app.get("/download/<filename>")
def download(filename: str):
    return send_from_directory(RESULT_DIR, filename, as_attachment=True)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": f"请求体超过 {MAX_CONTENT_MB} MB"}), 413


@app.errorhandler(Exception)
def unhandled_error(exc: Exception):
    return (
        jsonify(
            {
                "error": str(exc),
                "traceback": traceback.format_exc()
                if app.debug
                else None,
            }
        ),
        500,
    )


if __name__ == "__main__":
    print(f"Field extraction API: http://127.0.0.1:{PORT}")
    print("Mode: final DOCX only; no OCR model is initialized here.")
    print(f"Reading PDF2Word outputs from: {WEB_OUTPUT_DIR}")
    app.run(host=HOST, port=PORT, debug=False)
