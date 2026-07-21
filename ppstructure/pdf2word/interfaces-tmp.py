import io
import json
import os
import re
import traceback
import uuid
from difflib import SequenceMatcher

import fitz
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 修复：exported_frontend 只写一次
FRONTEND_DIR = os.path.join(BASE_DIR, "exported_frontend")  
DEFAULT_INPUT_DIR = os.path.join(BASE_DIR, "OCR识别文书")
UPLOAD_DIR = os.path.join(BASE_DIR, "autofill_uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "autofill_output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 前端 tab 文件夹映射
TAB_FOLDERS = {
    "民事起诉状": "001_民事起诉状",
    "传票": "002_传票",
    "举证通知书": "003_举证通知书",
    "管辖权异议": "004_管辖权异议",
    "证据 - 财产保全": "005_证据_-_财产保全",
    "答辩状": "006_答辩状",
    "民事上诉状": "007_民事上诉状",
    "案件判决": "008_案件判决",
    "执行": "009_执行",
    "原始附件": "010_原始附件",
}

# 文件名关键词 -> tab 映射
FILENAME_TAB_RULES = [
    ("民事起诉状", ["起诉状", "起诉"]),
    ("传票", ["传票", "开庭传票"]),
    ("举证通知书", ["举证通知书", "举证"]),
    ("管辖权异议", ["管辖权异议", "异议申请"]),
    ("证据 - 财产保全", ["财产保全", "证据保全", "保全申请"]),
    ("答辩状", ["答辩状", "答辩"]),
    ("民事上诉状", ["上诉状", "上诉"]),
    ("案件判决", ["判决书", "判决", "裁定书", "裁定"]),
    ("执行", ["执行", "执行申请"]),
]

SUPPORTED_EXTS = {"pdf", "docx", "jpg", "jpeg", "png", "bmp", "tif", "tiff"}

app = Flask(__name__)
CORS(app)

# ====== 直接导入 web_server-test.py 的资源（不用动态加载）======
import sys
sys.path.insert(0, BASE_DIR)
try:
    from ppstructure.predict_system import StructureSystem
    from ppstructure.utility import parse_args
    from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
    import cv2
    import numpy as np
    import copy
    import threading
    
    def redink_remover(img):
        image = copy.deepcopy(img)
        blue_c, green_c, red_c = cv2.split(image)
        result_img = np.expand_dims(red_c, axis=2)
        result_img = np.concatenate((result_img, result_img, result_img), axis=-1)
        return result_img
    
    # 初始化 OCR 引擎（只初始化一次）
    def init_ocr_engine():
        root = os.path.abspath(os.path.join(BASE_DIR, '../../'))
        args = parse_args()
        args.use_gpu = True
        args.table_max_len = 488
        args.ocr = True
        args.recovery = True
        args.layout_model_dir = r'/data/tensorflow/kath/OCR/models/pdf2word/layout/429/best_model/infer/picodet_2026_pdf2word'
        args.det_model_dir = os.path.join(root, "inference", "cn_PP-OCRv3_det_infer")
        args.rec_model_dir = os.path.join(root, "inference", "cn_PP-OCRv3_rec_infer")
        args.table_model_dir = os.path.join(root, "inference", "table_new")
        args.rec_char_dict_path = os.path.join(root, "ppocr", "utils", "ppocr_keys_v1.txt")
        args.layout_dict_path = os.path.join(root, "ppocr", "utils", "dict", "layout_dict", "pdf2word_dict.txt")
        args.layout_score_threshold = 0.6
        args.table_char_dict_path = os.path.join(root, "ppocr", "utils", "dict", "table_structure_dict_ch.txt")
        return StructureSystem(args), threading.Lock()
    
    predictor, ocr_lock = init_ocr_engine()
    OCR_AVAILABLE = True
except Exception as e:
    print(f"OCR engine initialization failed: {e}")
    predictor = None
    ocr_lock = None
    OCR_AVAILABLE = False


def _normalize_for_match(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"[\s\u3000]+", "", text)
    text = re.sub(r"[：:，,。；;（）()\[\]【】""\"'\-_/]", "", text)
    return text


def _guess_tab_by_filename(filename):
    name = str(filename or "")
    for tab, keywords in FILENAME_TAB_RULES:
        if any(k in name for k in keywords):
            return tab
    return "原始附件"


def _extract_labels_from_html(html):
    labels = re.findall(r'n-form-item-label__text">([^<]{1,80})<', html)
    sections = re.findall(r'>(诉讼请求|事实与理由|答辩意见|判决信息|案件分析|备注|执行进度|证据信息)<', html)
    labels.extend(sections)
    
    result = []
    seen = set()
    for item in labels:
        item = re.sub(r"\s+", "", item)
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _load_frontend_label_map():
    """加载前端各 tab 的字段标签"""
    tab_label_map = {}
    for tab_name, folder in TAB_FOLDERS.items():
        page_path = os.path.join(FRONTEND_DIR, folder, "page.html")
        if not os.path.exists(page_path):
            print(f"Warning: Frontend page not found: {page_path}")
            tab_label_map[tab_name] = []
            continue
        try:
            with open(page_path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
            labels = _extract_labels_from_html(html)
            tab_label_map[tab_name] = labels
            print(f"Loaded {len(labels)} labels for {tab_name}")
        except Exception as e:
            print(f"Error loading labels for {tab_name}: {e}")
            tab_label_map[tab_name] = []
    return tab_label_map


FRONTEND_LABELS = _load_frontend_label_map()


def _extract_text_from_pdf(file_path):
    """从 PDF 提取全文本"""
    try:
        with fitz.open(file_path) as pdf:
            texts = [pdf[pg].get_text("text") or "" for pg in range(pdf.page_count)]
            return "\n".join(texts)
    except Exception as e:
        print(f"PDF text extraction error: {e}")
        return ""


def _extract_via_ocr(file_path, ext, remove_red_seal=False):
    """走 OCR 提取（需要 predictor 和 ocr_lock）"""
    if not OCR_AVAILABLE:
        return {}, ""
    
    imgs = []
    try:
        if ext == "pdf":
            with fitz.open(file_path) as pdf:
                for pg in range(pdf.page_count):
                    pix = pdf[pg].get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_np = np.frombuffer(pix.tobytes(), np.uint8)
                    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                    if img is not None:
                        if remove_red_seal:
                            img = redink_remover(img)
                        imgs.append(img)
        else:
            img = cv2.imread(file_path)
            if img is not None:
                if remove_red_seal:
                    img = redink_remover(img)
                imgs.append(img)
    except Exception as e:
        print(f"Image loading error: {e}")
        return {}, ""
    
    if not imgs:
        return {}, ""
    
    record = {}
    all_text = []
    
    with ocr_lock:
        for img in imgs:
            try:
                res, _ = predictor(img)
                h, w, _ = img.shape
                res_sorted = sorted_layout_boxes(res, w)
                
                for region in res_sorted:
                    # 提取文本
                    if region.get("type", "").lower() == "text":
                        if isinstance(region.get("res"), list):
                            for item in region["res"]:
                                if isinstance(item, dict) and "text" in item:
                                    all_text.append(item["text"])
                    
                    # 提取表格键值对
                    if region.get("type", "").lower() == "table":
                        r = region.get("res")
                        if isinstance(r, dict) and r.get("html"):
                            try:
                                import pandas as pd
                                dfs = pd.read_html(io.StringIO(r["html"]), header=None)
                                if dfs:
                                    tbl_df = dfs[0]
                                    for _, row_s in tbl_df.iterrows():
                                        vals = ['' if str(v) == 'nan' else str(v).strip() for v in row_s]
                                        i = 0
                                        while i + 1 < len(vals):
                                            k, v = vals[i], vals[i + 1]
                                            if k:
                                                k_clean = k.strip().rstrip('：:')
                                                if k_clean:
                                                    record[k_clean] = v
                                            i += 2
                            except Exception as e:
                                print(f"Table parsing error: {e}")
            except Exception as e:
                print(f"OCR page error: {e}")
    
    full_text = "\n".join(all_text)
    return record, full_text


def _parse_key_values_from_text(text):
    """从纯文本中正则提取关键字段"""
    result = {}
    
    m = re.search(r"([\u4e00-\u9fa5]{2,40}人民法院)", text)
    if m:
        result["受理法院"] = m.group(1)
    
    m = re.search(r"([（(]\d{4}[）)][^\n]{2,40}?号)", text)
    if m:
        result["案件编号"] = m.group(1)
    
    m = re.search(r"原告[人]?[：:\s]*([\u4e00-\u9fa5A-Za-z·]{2,30})", text)
    if m:
        result["原告"] = m.group(1)
    
    m = re.search(r"被告[人]?[：:\s]*([\u4e00-\u9fa5A-Za-z·]{2,40})", text)
    if m:
        result["被告"] = m.group(1)
    
    m = re.search(r"(?:诉讼标的额|标的额|诉讼金额)[：:\s]*([0-9,.]{1,20})", text)
    if m:
        result["诉讼标的额"] = m.group(1)
    
    m = re.search(r"判决如下：:", text, re.S)
    if m:
        result["判决结果"] = re.sub(r"\s+", "", m.group(1))[:200]
    
    return result


def _best_label_match(record_key, labels):
    """找到最匹配的前端标签"""
    key_norm = _normalize_for_match(record_key)
    if not key_norm:
        return None, 0.0
    
    best_label = None
    best_score = 0.0
    for label in labels:
        label_norm = _normalize_for_match(label)
        if not label_norm:
            continue
        
        if key_norm == label_norm:
            return label, 1.0
        if key_norm in label_norm or label_norm in key_norm:
            score = 0.92
        else:
            score = SequenceMatcher(None, key_norm, label_norm).ratio()
        
        if score > best_score:
            best_score = score
            best_label = label
    
    return best_label, best_score


def _map_record_to_frontend(tab_name, record):
    """将提取的字段映射到前端标签"""
    labels = FRONTEND_LABELS.get(tab_name, [])
    mapped = {}
    unmatched = {}
    
    for k, v in record.items():
        if k == "文件名":
            continue
        value = str(v or "").strip()
        if not value:
            continue
        
        label, score = _best_label_match(k, labels)
        if label and score >= 0.68:
            if label not in mapped or len(value) > len(mapped[label]):
                mapped[label] = value
        else:
            unmatched[k] = value
    
    return mapped, unmatched


def _process_single_file(file_path, filename, remove_red_seal=False):
    """处理单个文件，返回按文档名组织的结果"""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTS:
        return {
            filename: {
                "错误": f"不支持的文件类型: {ext}"
            }
        }
    
    tab_name = _guess_tab_by_filename(filename)
    
    # 1. 先尝试纯文本提取（PDF 文字层）
    text = ""
    record = {}
    if ext == "pdf":
        text = _extract_text_from_pdf(file_path)
        if text.strip():
            record = _parse_key_values_from_text(text)
    
    # 2. 如果纯文本提取不够，走 OCR
    if not record and OCR_AVAILABLE:
        ocr_record, ocr_text = _extract_via_ocr(file_path, ext, remove_red_seal)
        record.update(ocr_record)
        if ocr_text:
            text_kv = _parse_key_values_from_text(ocr_text)
            for k, v in text_kv.items():
                if k not in record:
                    record[k] = v
    
    # 3. 映射到前端字段
    mapped, unmatched = _map_record_to_frontend(tab_name, record)
    
    # 4. 按文档名组织输出
    return {
        filename: {
            "文书类型": tab_name,
            "前端字段": mapped,
            "未匹配字段": unmatched,
            "提取方式": "OCR" if OCR_AVAILABLE else "文本",
            "原始记录条目数": len(record)
        }
    }


def _save_batch_json(task_id, payload):
    out_name = f"{task_id}_autofill.json"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_name


@app.route("/")
def root():
    return jsonify({
        "service": "pdf2word-autofill",
        "ocr_available": OCR_AVAILABLE,
        "frontend_tabs_loaded": len(FRONTEND_LABELS),
        "endpoints": [
            "POST /autofill_from_folder",
            "POST /autofill_from_uploads",
            "GET /download/<filename>",
        ],
    })


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route("/autofill_from_folder", methods=["POST"])
def autofill_from_folder():
    body = request.get_json(silent=True) or {}
    input_path = request.args.get("input_folder") or body.get("input_folder", DEFAULT_INPUT_DIR)
    remove_red_seal = request.args.get("remove_red_seal", "").lower() == "true" or bool(body.get("remove_red_seal", False))
    
    input_path = input_path.strip('"').strip("'").strip()
    task_id = str(uuid.uuid4())[:8]
    all_results = {}
    
    try:
        if os.path.isfile(input_path):
            filename = os.path.basename(input_path)
            if "." in filename:
                file_result = _process_single_file(input_path, filename, remove_red_seal)
                all_results.update(file_result)
        
        elif os.path.isdir(input_path):
            for filename in sorted(os.listdir(input_path)):
                file_path = os.path.join(input_path, filename)
                if not os.path.isfile(file_path):
                    continue
                if "." not in filename:
                    continue
                ext = filename.rsplit(".", 1)[-1].lower()
                if ext not in SUPPORTED_EXTS:
                    continue
                
                try:
                    file_result = _process_single_file(file_path, filename, remove_red_seal)
                    all_results.update(file_result)
                except Exception as exc:
                    all_results[filename] = {"错误": str(exc)}
        else:
            return jsonify({"error": f"路径不存在: {input_path}"}), 400
        
        payload = {
            "task_id": task_id,
            "source": "folder" if os.path.isdir(input_path) else "single_file",
            "input_path": input_path,
            "total": len(all_results),
            "results": all_results
        }
        
        out_name = _save_batch_json(task_id, payload)
        return jsonify({
            "task_id": task_id,
            "total": len(all_results),
            "download_url": f"/download/{out_name}",
            "saved_file": out_name,
            "results": all_results
        })
    
    except Exception as exc:
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


@app.route("/autofill_from_uploads", methods=["POST"])
def autofill_from_uploads():
    uploaded_files = request.files.getlist("files")
    remove_red_seal = request.form.get("remove_red_seal", "false").lower() == "true"
    
    if not uploaded_files:
        return jsonify({"error": "No files uploaded"}), 400
    
    task_id = str(uuid.uuid4())[:8]
    batch_dir = os.path.join(UPLOAD_DIR, task_id)
    os.makedirs(batch_dir, exist_ok=True)
    
    all_results = {}
    
    try:
        for file in uploaded_files:
            if not file or not file.filename:
                continue
            filename = file.filename
            if "." not in filename:
                continue
            
            ext = filename.rsplit(".", 1)[-1].lower()
            if ext not in SUPPORTED_EXTS:
                all_results[filename] = {"错误": f"不支持的文件类型: {ext}"}
                continue
            
            safe_name = filename.replace("/", "_").replace("\\", "_")
            saved_path = os.path.join(batch_dir, safe_name)
            file.save(saved_path)
            
            try:
                file_result = _process_single_file(saved_path, safe_name, remove_red_seal)
                all_results.update(file_result)
            except Exception as exc:
                all_results[safe_name] = {"错误": str(exc)}
        
        payload = {
            "task_id": task_id,
            "source": "upload",
            "total": len(all_results),
            "results": all_results
        }
        
        out_name = _save_batch_json(task_id, payload)
        return jsonify({
            "task_id": task_id,
            "total": len(all_results),
            "download_url": f"/download/{out_name}",
            "saved_file": out_name,
            "results": all_results
        })
    
    except Exception as exc:
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    print(f"Flask server starting at http://localhost:8006")
    print(f"OCR Available: {OCR_AVAILABLE}")
    print(f"Frontend tabs loaded: {len(FRONTEND_LABELS)}")
    for tab, labels in FRONTEND_LABELS.items():
        print(f"  {tab}: {len(labels)} 个字段")
    app.run(host="0.0.0.0", port=8006, debug=False)