def _normalize_match_text(s):
    """用于相似度比较：去空白和常见标点，降低 OCR 断字/标点噪声影响。"""
    s = str(s or '').strip().lower()
    if not s:
        return ''
    s = re.sub(r'[\s\u3000]+', '', s)
    s = re.sub(r'[：:，,。；;（）()\[\]【】“”"\'\-_/]', '', s)
    return s


def _is_similar_or_contains(text, target, threshold=0.9):
    """匹配规则：完全包含，或相似度 >= threshold。"""
    t1 = _normalize_match_text(text)
    t2 = _normalize_match_text(target)
    if not t1 or not t2:
        return False
    if t2 in t1 or t1 in t2:
        return True
    return SequenceMatcher(None, t1, t2).ratio() >= threshold


def _collect_text_fragments(obj):
    """从 OCR region 的嵌套结构中尽量提取文本片段。"""
    out = []
    if obj is None:
        return out
    if isinstance(obj, str):
        txt = obj.strip()
        if txt:
            out.append(txt)
        return out
    if isinstance(obj, dict):
        txt = obj.get('text')
        if isinstance(txt, str) and txt.strip():
            out.append(txt.strip())
        for v in obj.values():
            out.extend(_collect_text_fragments(v))
        return out
    if isinstance(obj, (list, tuple)):
        for item in obj:
            out.extend(_collect_text_fragments(item))
    return out


def _fill_handler_fields_from_text(text, result):
    """从自由文本中兜底提取经办人字段。"""
    if not text:
        return
    if not result['经办人姓名']:
        m = re.search(r'经办人(?:姓名)?[：:\s]*([\u4e00-\u9fa5A-Za-z·]{2,30})', text)
        if m:
            result['经办人姓名'] = m.group(1).strip()
    if not result['经办人证件号码']:
        m = re.search(r'经办人(?:证件号码|身份证号码|证件号)?[：:\s]*([0-9Xx*]{6,25})', text)
        if not m:
            m = re.search(r'(?:证件号码|身份证号码|证件号)[：:\s]*([0-9Xx*]{6,25})', text)
        if m:
            result['经办人证件号码'] = m.group(1).strip()


def _fill_legal_fields_from_text(text, result):
    """从自由文本中兜底提取法定代表人字段。"""
    if not text:
        return
    # print('[text for legal fields]\n', text)
    if not result['法定代表人姓名']:
        m = re.search(
            r'姓名[：:\s_]*([\u4e00-\u9fa5A-Za-z·]{2,30})[\s\S]{0,100}?(?:身份证号码|证件号码|证件号)[：:\s_]*[0-9Xx*]{6,25}[\s\S]{0,100}?系本单位的法定代表人',
            text
        )
        
    
        if not m:
            m = re.search(r'法定代表人(?:姓名)?[：:\s]*([\u4e00-\u9fa5A-Za-z·]{2,5}身份证)', text)
         
        if not m:
            m = re.search(r'姓名[：:\s]*([\u4e00-\u9fa5A-Za-z·]{2,5}身份证)', text)
            
        if m:
            result['法定代表人姓名'] = m.group(1).strip() #[:-3]
    if not result['法定代表人身份证号码']:
        m = re.search(r'(?:法定代表人)?(?:身份证号码|证件号码|证件号)[：:\s]*([0-9Xx*]{6,25})', text)
        if m:
            result['法定代表人身份证号码'] = m.group(1).strip()


def extract_digital_cert_fields_via_ocr(input_path, ext, filename, remove_red_seal=False):
    """
    针对电子签章平台数字证书申请表，OCR所有表格和文本区域，抽取5个目标字段。
    返回 dict: {文件名, 经办人姓名, 经办人证件号码, 法定代表人姓名, 法定代表人身份证号码}
    """
    result = {
        '文件名': filename,
        '经办人姓名': '',
        '经办人证件号码': '',
        '法定代表人姓名': '',
        '法定代表人身份证号码': ''
    }

    handler_section = '申请单位信息（必填项）'
    legal_section = '申请单位法定代表人授权委托及单位声明（必填项，此项如涂改，请在涂改处加盖公章）'
    legal_end_marker = '系本单位的法定代表人'

    imgs = []
    if ext == 'pdf':
        try:
            with fitz.open(input_path) as pdf:
                # 业务要求：上传 PDF 时仅在第一页提取目标字段
                if pdf.page_count > 0:
                    pix = pdf[0].get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_np = np.frombuffer(pix.tobytes(), np.uint8)
                    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                    if img is not None:
                        if remove_red_seal:
                            img = redink_remover(img)
                        imgs.append(img)
        except Exception as e:
            print(f'extract_digital_cert_fields_via_ocr: PDF read error: {e}')
    else:
        img = cv2.imread(input_path)
        if img is not None:
            if remove_red_seal:
                img = redink_remover(img)
            imgs.append(img)

    with ocr_lock:
        for img in imgs:
            try:
                res, _ = predictor(img)
                _, w, _ = img.shape
                res_sorted = sorted_layout_boxes(res, w)

                non_table_texts = []

                for region in res_sorted:
                    region_type = str(region.get('type', '')).lower()

                    if region_type != 'table':
                        segs = _collect_text_fragments(region.get('res'))
                        for seg in segs:
                            if seg and seg != 'nan':
                                non_table_texts.append(seg)
                        continue

                    r = region.get('res')
                    if not (isinstance(r, dict) and r.get('html')):
                        continue
                    try:
                        raw_dfs = pd.read_html(io.StringIO(r['html']), header=None)
                        if not raw_dfs:
                            continue
                        tbl_df = raw_dfs[0]
                    except Exception:
                        continue

                    if tbl_df is None or tbl_df.empty:
                        continue

                    flat_cells = tbl_df.astype(str).values.flatten().tolist()
                    flat_cells = [c.strip() for c in flat_cells if c and str(c).strip() != 'nan']
                    table_text = ' '.join(flat_cells)

                    section = None
                    if (
                        _is_similar_or_contains(table_text, handler_section)
                        or any(_is_similar_or_contains(c, handler_section) for c in flat_cells)
                    ):
                        section = 'handler'
                    elif (
                        _is_similar_or_contains(table_text, legal_section)
                        or any(_is_similar_or_contains(c, legal_section) for c in flat_cells)
                    ):
                        section = 'legal'

                    if section == 'handler':
                        for _, row_s in tbl_df.iterrows():
                            vals = ['' if str(v) == 'nan' else str(v).strip() for v in row_s]
                            i = 0
                            while i + 1 < len(vals):
                                k, v = vals[i], vals[i + 1]
                                k_norm = _norm_key(k)
                                if '经办人姓名' in k_norm and not result['经办人姓名']:
                                    result['经办人姓名'] = v
                                elif '经办人证件号码' in k_norm and not result['经办人证件号码']:
                                    result['经办人证件号码'] = v
                                i += 2
                    elif section == 'legal':
                        for _, row_s in tbl_df.iterrows():
                            vals = ['' if str(v) == 'nan' else str(v).strip() for v in row_s]
                            i = 0
                            while i + 1 < len(vals):
                                k, v = vals[i], vals[i + 1]
                                k_norm = _norm_key(k)
                                if (
                                    ('法定代表人姓名' in k_norm or k_norm == '姓名')
                                    and not result['法定代表人姓名']
                                ):
                                    result['法定代表人姓名'] = v
                                elif (
                                    ('法定代表人身份证号码' in k_norm or '身份证号码' in k_norm)
                                    and not result['法定代表人身份证号码']
                                ):
                                    result['法定代表人身份证号码'] = v
                                i += 2

                # 文本区兜底：法律声明区可能不是表格
                if non_table_texts:
                    # 去掉连续重复文本，降低 OCR 重复输出干扰
                    compact_texts = []
                    for txt in non_table_texts:
                        if not compact_texts or compact_texts[-1] != txt:
                            compact_texts.append(txt)

                    full_text = ' '.join(compact_texts)
                    # 经办人姓名/证件号码只从表格提取，不走文本兜底

                    start_idx = -1
                    end_idx = -1
                    for i, txt in enumerate(compact_texts):
                        if _is_similar_or_contains(txt, legal_section):
                            start_idx = i
                            break
                    if start_idx >= 0:
                        for j in range(start_idx, len(compact_texts)):
                            if _is_similar_or_contains(compact_texts[j], legal_end_marker):
                                end_idx = j
                                break
                        if end_idx < 0:
                            legal_text = ' '.join(compact_texts[start_idx:])
                        else:
                            legal_text = ' '.join(compact_texts[start_idx:end_idx + 1])
                        _fill_legal_fields_from_text(legal_text, result)
                    else:
                        _fill_legal_fields_from_text(full_text, result)

            except Exception as e:
                print(f'extract_digital_cert_fields_via_ocr: page error: {e}')

    return result
import os
import sys
import uuid
import traceback
import threading
import io
import re
from difflib import SequenceMatcher
import cv2
import numpy as np
import fitz
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from pdf2docx.converter import Converter
import copy

def redink_remover(img):
    image = copy.deepcopy(img)
    blue_c, green_c, red_c = cv2.split(image)
    result_img = np.expand_dims( red_c, axis=2)
    result_img = np.concatenate((result_img, result_img, result_img), axis=-1)
    return result_img

# 设置项目根目录
file_path = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(file_path, '../../'))
sys.path.append(file_path)
sys.path.insert(0, root)

# 复用原有的预测器和逻辑
import copy
from ppstructure.predict_system import StructureSystem
from ppstructure.utility import parse_args, draw_structure_result
from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx_multi_page
 
app = Flask(__name__)
CORS(app) # 允许 Vue 跨域调用

# 配置
UPLOAD_FOLDER = os.path.join(file_path, 'web_uploads')
OUTPUT_FOLDER = os.path.join(file_path, 'web_output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 初始化 OCR 引擎
# 注意：这里路径参考了 pdf2word.py 中的硬编码路径
def init_ocr():
    args = parse_args()
    args.use_gpu = True
    args.table_max_len = 488
    args.ocr = True
    args.recovery = True
    # 布局模型路径
    args.layout_model_dir = r'/data/tensorflow/kath/Service/pdf2word/models/pdf2wordlayout' 
    args.layout_model_dir=r'/data/tensorflow/kath/OCR/models/pdf2word/layout/429/best_model/infer/picodet_2026_pdf2word'
    # 基础模型路径
    args.det_model_dir = os.path.join(root, "inference", "cn_PP-OCRv3_det_infer")
    args.rec_model_dir = os.path.join(root, "inference", "cn_PP-OCRv3_rec_infer")
    # args.table_model_dir = os.path.join(root, "inference", "cn_ppstructure_mobile_v2.0_SLANet_infer")
    args.table_model_dir = os.path.join(root, "inference", "table_new")
    
    # 字典路径
    args.rec_char_dict_path = os.path.join(root, "ppocr", "utils", "ppocr_keys_v1.txt")
    # args.layout_dict_path = os.path.join(root, "ppocr", "utils", "dict", "layout_dict", "layout_cdla_dict.txt")
    args.layout_dict_path = os.path.join(root, "ppocr", "utils", "dict", "layout_dict", "pdf2word_dict.txt")
    args.layout_score_threshold =0.6
    args.table_char_dict_path = os.path.join(root, "ppocr", "utils", "dict", "table_structure_dict_ch.txt")
    print('\n\n[OCR args]\n', args)
    return StructureSystem(args)

# 全局初始化一次，避免每次请求重载
predictor = init_ocr()
# PaddlePaddle 推理引擎非线程安全，同一时刻只允许一个请求使用 predictor
ocr_lock = threading.Lock()

@app.route('/')
def redirect_to_index():
    from flask import redirect
    return redirect('/index')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    keep_reference = request.form.get('keep_reference', 'true').lower() == 'true'
    # 开发测试选项：强制走完整 OCR 流程，跳过 pdf2docx 直接转换
    force_ocr = request.form.get('force_ocr', 'false').lower() == 'true'
    remove_red_seal = request.form.get('remove_red_seal', 'false').lower() == 'true'
    
    task_id = str(uuid.uuid4())
    filename = file.filename
    ext = filename.split('.')[-1].lower()
    original_basename = filename.rsplit('.', 1)[0]
    # 用 task_id 前8位作为唯一前缀，避免多用户同名文件互相覆盖
    img_basename = f"{task_id[:8]}_{original_basename}"

    # 保存原始文件
    input_path = os.path.join(UPLOAD_FOLDER, f"{task_id}.{ext}")
    file.save(input_path)

    try:
        # ----------------------------------------------------------------
        # 阶段1：对 PDF 优先尝试 pdf2docx 直接转换（除非强制 OCR）
        # ----------------------------------------------------------------
        if ext == 'pdf' and not force_ocr:
            try:
                with fitz.open(input_path) as pdf:
                    text_count = sum(
                        len(pdf[pg].get_text().strip())
                        for pg in range(pdf.page_count)
                    )
                
                if text_count > 0:
                    # 文字版 PDF → 直接用 pdf2docx 转换
                    print(f"文字版PDF（文字量={text_count}），使用 pdf2docx 直接转换")
                    docx_filename = f"{img_basename}.docx"
                    docx_path = os.path.join(OUTPUT_FOLDER, docx_filename)
                    cv = Converter(input_path)
                    cv.convert(docx_path)
                    cv.close()
                    print(f"直接转换完成: {docx_path}")
                    return jsonify({
                        "task_id": task_id,
                        "filename": docx_filename,
                        "download_url": f"/download/{docx_filename}",
                        "method": "direct"
                    })
                else:
                    # 扫描版 PDF（无文字层）→ 回退至 OCR
                    print("扫描版PDF（无文字层），自动回退至 OCR 流程")
            except Exception as e:
                print(f"pdf2docx 直接转换失败，回退至 OCR 流程: {e}")
                print(traceback.format_exc())

        # ----------------------------------------------------------------
        # 阶段2：OCR 流程（扫描版PDF / 图片 / 强制OCR）
        # ----------------------------------------------------------------
        imgs = []
        if ext == 'pdf':
            with fitz.open(input_path) as pdf:
                for pg in range(pdf.page_count):
                    page = pdf[pg]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_np = np.frombuffer(pix.tobytes(), np.uint8)
                    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                    if img is not None:
                        if remove_red_seal:
                            img = redink_remover(img)
                        imgs.append(img)
        else:
            # 图片处理
            file.seek(0)  # 重新读取
            img_np = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if img is not None:
                if remove_red_seal:
                    img = redink_remover(img)
                imgs.append(img)

        if not imgs:
            return jsonify({"error": "Failed to read image/pdf"}), 400

        # OCR 推理引擎非线程安全，加锁保证同一时刻只有一个请求在跑推理
        with ocr_lock:
            all_res = []
            vis_font_path = os.path.join(root, "doc", "fonts", "simfang.ttf")
            img_dir = os.path.normpath(os.path.join(OUTPUT_FOLDER, img_basename))
            os.makedirs(img_dir, exist_ok=True)

            for i, img in enumerate(imgs):
                res, _ = predictor(img)
                h, w, _ = img.shape
                
                # 格式转换与画图保存
                res_for_draw = []
                for region in res:
                    region_copy = copy.deepcopy(region)
                    if isinstance(region_copy.get('res'), list):
                        for item in region_copy['res']:
                            if isinstance(item, dict) and 'text_region' in item:
                                text_region = item['text_region']
                                if isinstance(text_region, list) and len(text_region) == 4:
                                    x1, y1, x2, y2 = text_region
                                    item['text_region'] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    res_for_draw.append(region_copy)
                
                draw_img = draw_structure_result(img, res_for_draw, vis_font_path)
                img_save_path = os.path.normpath(os.path.join(img_dir, f'show_{i}.jpg'))
                
                if res != []:
                    is_success, im_buf_arr = cv2.imencode(".jpg", draw_img)
                    if is_success:
                        im_buf_arr.tofile(img_save_path)
                    print('\n\n[image saved at ]', img_save_path)

                res = sorted_layout_boxes(res, w)
                for region in res:
                    region['page_index'] = i
                all_res.append({'img': img, 'res': res, 'page_index': i})

            # convert_info_docx_multi_page 生成 {img_basename}_ocr.docx
            convert_info_docx_multi_page(all_res, OUTPUT_FOLDER, img_basename, keep_reference)
        
        expected_filename = f"{img_basename}_ocr.docx"
        return jsonify({
            "task_id": task_id,
            "filename": expected_filename,
            "download_url": f"/download/{expected_filename}",
            "method": "ocr"
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


# ----------------------------------------------------------------
# 表格提取辅助函数
# ----------------------------------------------------------------

def _dedup_columns(headers):
    """对重复列名自动加后缀，避免 DataFrame 列名冲突（合并单元格会产生重复）。"""
    seen = {}
    result = []
    for h in headers:
        h = str(h).strip()
        if h in seen:
            seen[h] += 1
            result.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            result.append(h)
    return result
 

def _unique_cells(row):
    """python-docx 行去重——合并单元格在 row.cells 里会重复出现，按 _tc 对象去重。"""
    seen_tc = set()
    result = []
    for cell in row.cells:
        tc_id = id(cell._tc)
        if tc_id not in seen_tc:
            seen_tc.add(tc_id)
            result.append(cell.text.strip())
    return result


def _norm_key(k):
    """去掉字段名尾部的全角/半角冒号/分号，统一列名。"""
    k = str(k).strip()
    while k.endswith('：') or k.endswith(':') or k.endswith('；') or k.endswith(';'):
        k = k[:-1].rstrip()
    return k


def _parse_kv_row(left, right):
    """
    解析两列表格的一行，返回 (key, value) 或 None（跳过）。
    特殊情况：right 为空 且 left 含冒号或分号且不在末尾
    → OCR/表格切分错误，按第一个匹配标点拆分 left 自身。
    """
    left = (left or '').strip()
    right = (right or '').strip()
    if not left:
        return None
        
    match = re.search(r'[：:；;]', left)
    if right == '' and match and match.end() < len(left):
        idx = match.start()
        key = _norm_key(left[:idx])
        val = left[idx + 1:].strip()
    else:
        key = _norm_key(left)
        val = right
    return (key, val) if key else None


# 合同电子签章：特殊表单提取（Excel）Schema 对齐配置
CONTRACT_ESIGN_SCHEMA_NO_ADDRESS = [
    "收件人姓名",
    "收件人电话",
    "收件人电子邮箱",
    "名称",
    "纳税人识别号",
    "地址",
    "电话",
    "银行开户行",
    "银行账号",
]

CONTRACT_ESIGN_SCHEMA_WITH_ADDRESS = [
    "收件人姓名",
    "收件人电话",
    "收件人地址",
    "收件人电子邮箱",
    "名称",
    "纳税人识别号",
    "地址",
    "电话",
    "银行开户行",
    "银行账号",
]

CONTRACT_ESIGN_SCHEMAS = [
    CONTRACT_ESIGN_SCHEMA_WITH_ADDRESS,
    CONTRACT_ESIGN_SCHEMA_NO_ADDRESS,
]


def _normalize_key_for_match(text):
    if not text:
        return ""
    text = re.sub(r'\s+', '', str(text))
    text = text.replace(':', '：')
    while text.endswith('：'):
        text = text[:-1]
    text = re.sub(r'[\*\-_．\.#@、，,（）\(\)]', '', text)
    return text


def _schema_key_similarity(raw_text, schema_key):
    raw_norm = _normalize_key_for_match(raw_text)
    key_norm = _normalize_key_for_match(schema_key)
    if not raw_norm or not key_norm:
        return 0.0

    score = SequenceMatcher(None, raw_norm, key_norm).ratio()
    if key_norm in raw_norm:
        score += 0.25
    elif raw_norm in key_norm:
        score += 0.1
    return min(1.0, score)


def _best_schema_key_match(raw_text, schema):
    best_key = None
    best_score = -1.0
    for key in schema:
        score = _schema_key_similarity(raw_text, key)
        if score > best_score:
            best_score = score
            best_key = key
    return best_key, best_score


def _split_suffix_after_colon(raw_left, expected_key, schema=None):
    if ':' not in raw_left and '：' not in raw_left:
        return None

    idx = raw_left.find(':')
    idx_cn = raw_left.find('：')
    if idx == -1:
        split_idx = idx_cn
    elif idx_cn == -1:
        split_idx = idx
    else:
        split_idx = min(idx, idx_cn)

    key_part = raw_left[:split_idx].strip()
    suffix = raw_left[split_idx + 1:].strip()
    if not (1 <= len(suffix) <= 4):
        return None

    sim_score = _schema_key_similarity(key_part, expected_key)
    similar_enough = sim_score >= 0.55
    if not similar_enough and schema:
        _, best_score = _best_schema_key_match(key_part, schema)
        similar_enough = best_score >= 0.55

    if similar_enough:
        return expected_key, suffix
    return None


def _previous_value_completion_score(prev_value, prefix):
    if not prefix:
        return 0.0
    combined = (prev_value or "") + prefix
    high_suffixes = (
        "支行", "分行", "银行", "公司", "有限公司", "号", "室", "楼", "层",
        "省", "市", "区", "县", "镇"
    )
    medium_prefixes = ("行", "司", "号", "室", "楼", "层", "省", "市", "区")

    if combined.endswith(high_suffixes):
        return 1.0
    if prefix in medium_prefixes and len(prefix) <= 2:
        return 0.6
    return 0.1


def _is_probable_value_fragment(fragment):
    if not fragment or len(fragment) > 4:
        return False
    if ':' in fragment or '：' in fragment:
        return False

    known_fragments = (
        "号", "室", "楼", "层", "行", "司", "公司", "支行", "分行",
        "省", "市", "区", "县", "镇"
    )
    if fragment in known_fragments:
        return True
    if re.match(r'^[a-zA-Z0-9]+$', fragment):
        return True
    if re.match(r'^\d+[号室楼层区]$', fragment):
        return True
    return False


def _choose_contract_esign_schema_from_pairs(kv_pairs):
    if not kv_pairs:
        return CONTRACT_ESIGN_SCHEMA_WITH_ADDRESS

    score_with = 0.0
    score_no = 0.0
    has_address = False
    has_email = False
    has_phone = False

    for pair in kv_pairs:
        rk = pair.get('raw_key', '')
        best_with_sim = max(_schema_key_similarity(rk, sk) for sk in CONTRACT_ESIGN_SCHEMA_WITH_ADDRESS)
        best_no_sim = max(_schema_key_similarity(rk, sk) for sk in CONTRACT_ESIGN_SCHEMA_NO_ADDRESS)
        score_with += best_with_sim
        score_no += best_no_sim

        if _schema_key_similarity(rk, "收件人地址") >= 0.7:
            has_address = True
        if _schema_key_similarity(rk, "收件人电子邮箱") >= 0.7:
            has_email = True
        if _schema_key_similarity(rk, "收件人电话") >= 0.7:
            has_phone = True

    if has_address:
        score_with += 5.0
    elif has_email and has_phone:
        score_no += 2.0

    return CONTRACT_ESIGN_SCHEMA_WITH_ADDRESS if score_with >= score_no else CONTRACT_ESIGN_SCHEMA_NO_ADDRESS


def _align_kv_pairs_to_schema(kv_pairs, schema):
    aligned_keys = []
    schema_pos = 0

    for pair in kv_pairs:
        rk = pair.get('raw_key', '')

        best_suffix_key = None
        best_suffix_idx = -1
        best_suffix_score = -1.0
        for i in range(schema_pos, len(schema)):
            sk = schema[i]
            score = _schema_key_similarity(rk, sk)
            if score > best_suffix_score:
                best_suffix_score = score
                best_suffix_key = sk
                best_suffix_idx = i

        if best_suffix_score >= 0.55:
            aligned_keys.append(best_suffix_key)
            schema_pos = best_suffix_idx + 1
            continue

        best_global_key = None
        best_global_idx = -1
        best_global_score = -1.0
        for i, sk in enumerate(schema):
            score = _schema_key_similarity(rk, sk)
            if score > best_global_score:
                best_global_score = score
                best_global_key = sk
                best_global_idx = i

        if best_global_score >= 0.55:
            aligned_keys.append(best_global_key)
            schema_pos = best_global_idx + 1
        else:
            aligned_keys.append(None)

    return aligned_keys


def _repair_contract_esign_record(record, max_contam_len=4):
    """
    对合同电子签章记录做强 schema 对齐，仅修 key；
    value 仅在 prefix/suffix 串行污染迁移时改动。
    """
    try:
        if not isinstance(record, dict) or not record:
            return record

        pairs = []
        for k, v in record.items():
            if k == '文件名':
                continue
            pairs.append({'raw_key': str(k), 'raw_value': '' if v is None else str(v)})

        if not pairs:
            return record

        schema = _choose_contract_esign_schema_from_pairs(pairs)
        aligned_keys = _align_kv_pairs_to_schema(pairs, schema)

        keys = [p['raw_key'] for p in pairs]
        values = [p['raw_value'] for p in pairs]

        for i, raw_left in enumerate(keys):
            expected_key = aligned_keys[i]
            if not expected_key:
                continue

            raw_right = values[i]

            # D1 后缀污染：收件人地址：号 -> key=收件人地址, value+=号
            split_res = _split_suffix_after_colon(raw_left, expected_key, schema=schema)
            if split_res:
                _, suffix = split_res
                if len(suffix) <= max_contam_len and _is_probable_value_fragment(suffix):
                    keys[i] = expected_key
                    values[i] = raw_right + suffix
                    print(f'[contract_esign_schema_repair] suffix moved: raw_key="{raw_left}", expected="{expected_key}", moved="{suffix}"')
                    continue

            # D2 前缀污染：行 银行账号： -> prefix 回上一行 value
            # if i > 0:
            #     prev_value = values[i - 1]
            #     for prefix_len in range(1, max_contam_len + 1):
            #         if len(raw_left) <= prefix_len:
            #             break
            #         prefix = raw_left[:prefix_len].strip()
            #         rest_key = raw_left[prefix_len:].strip()
            #         if not prefix:
            #             continue
            #         if _schema_key_similarity(rest_key, expected_key) >= 0.65:
            #             score = _previous_value_completion_score(prev_value, prefix)
            #             if score >= 0.6:
            #                 values[i - 1] = prev_value + prefix
            #                 keys[i] = expected_key
            #                 print(f'[contract_esign_schema_repair] prefix moved: raw_key="{raw_left}", expected="{expected_key}", moved="{prefix}"')
            #                 break
            # D2 前缀污染：只在 raw_left 中存在空格分隔时才尝试修复
            # 例： "行 银行账号：" -> prefix="行", rest_key="银行账号："
            if i > 0 and re.search(r'\s+', raw_left):
                prev_value = values[i - 1]

                parts = re.split(r'\s+', raw_left.strip(), maxsplit=1)
                if len(parts) == 2:
                    prefix, rest_key = parts[0].strip(), parts[1].strip()

                    if 0 < len(prefix) <= max_contam_len:
                        if _schema_key_similarity(rest_key, expected_key) >= 0.65:
                            score = _previous_value_completion_score(prev_value, prefix)
                            if score >= 0.6:
                                values[i - 1] = prev_value + prefix
                                keys[i] = expected_key
                                print(f'[contract_esign_schema_repair] prefix moved: raw_key="{raw_left}", expected="{expected_key}", moved="{prefix}"')

            # D3 标题归正
            sim = _schema_key_similarity(raw_left, expected_key)
            raw_norm = _normalize_key_for_match(raw_left)
            expected_norm = _normalize_key_for_match(expected_key)
            # 原始 key 本身已经是目标字段，不能再尝试把它前几个字拆走
            if raw_norm == expected_norm:
                keys[i] = expected_key
                continue
            if sim >= 0.55 or (expected_norm in raw_norm) or (raw_norm in expected_norm and raw_norm):
                if keys[i] != expected_key:
                    print(f'[contract_esign_schema_repair] key canonicalized: raw_key="{raw_left}", expected="{expected_key}"')
                keys[i] = expected_key

        repaired = {}
        if '文件名' in record:
            repaired['文件名'] = record['文件名']

        for k, v in zip(keys, values):
            nk = _norm_key(k)
            if not nk:
                continue
            if nk not in repaired:
                repaired[nk] = v
            elif (not repaired[nk]) and v:
                repaired[nk] = v

        # for k, v in record.items():
        #     if k not in repaired:
        #         repaired[k] = v

        return repaired
    except Exception as e:
        print(f'[contract_esign_schema_repair] Error: {e}')
        return record


def extract_record_from_docx(docx_path):
    """
    从 Word 文档提取键值对表格，所有表格合并为一个 dict（= Excel 一行）。
    只处理恰好两列的表格：左列为 key，右列为 value。
    """
    from docx import Document as DocxDoc

    record = {}
    try:
        doc = DocxDoc(docx_path)
        for tbl_idx, tbl in enumerate(doc.tables):
            if len(tbl.columns) != 2:
                print(f'[docx] tbl={tbl_idx} 跳过（列数={len(tbl.columns)}，非2列）')
                continue
            for row_idx, row in enumerate(tbl.rows):
                cells = _unique_cells(row)
                print(f'[docx] tbl={tbl_idx} row={row_idx} cells={cells}')
                left  = cells[0] if len(cells) > 0 else ''
                right = cells[1] if len(cells) > 1 else ''
                kv = _parse_kv_row(left, right)
                if kv:
                    record[kv[0]] = kv[1]
    except Exception as e:
        print(f'extract_record_from_docx error ({docx_path}): {e}')
        print(traceback.format_exc())
    print(f'[docx] merged record keys({len(record)}): {list(record.keys())}')
    return record


def _parse_html_table_to_df(html_str):
    """将 OCR 生成的 HTML 表格字符串解析为 DataFrame（作为 OCR 回退路径使用）。"""
    try:
        dfs = pd.read_html(io.StringIO(html_str), header=0)
        if dfs:
            df = dfs[0]
            df.columns = _dedup_columns(df.columns.tolist())
            return df
    except Exception:
        pass
    try:
        from bs4 import BeautifulSoup as BS
        soup = BS(html_str, 'html.parser')
        table = soup.find('table')
        if not table:
            return None
        rows = [[cell.get_text(strip=True) for cell in tr.find_all(['th', 'td'])]
                for tr in table.find_all('tr')]
        rows = [r for r in rows if r]
        if len(rows) < 2:
            return None
        headers = _dedup_columns(rows[0])
        data = []
        for row in rows[1:]:
            padded = row + [''] * (len(headers) - len(row))
            data.append(padded[:len(headers)])
        return pd.DataFrame(data, columns=headers)
    except Exception as e:
        print(f'_parse_html_table_to_df error: {e}')
        return None


def extract_record_via_ocr(input_path, ext, remove_red_seal=False):
    """扫描版 PDF / 图片的兜底方案：运行 OCR 提取表格，返回合并后的 dict。"""
    imgs = []
    if ext == 'pdf':
        try:
            with fitz.open(input_path) as pdf:
                for pg in range(pdf.page_count):
                    pix = pdf[pg].get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_np = np.frombuffer(pix.tobytes(), np.uint8)
                    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                    if img is not None:
                        if remove_red_seal:
                            img = redink_remover(img)
                        imgs.append(img)
        except Exception as e:
            print(f'extract_record_via_ocr: PDF read error: {e}')
    else:
        img = cv2.imread(input_path)
        if img is not None:
            if remove_red_seal:
                img = redink_remover(img)
            imgs.append(img)

    record = {}
    with ocr_lock:
        for img in imgs:
            try:
                res, _ = predictor(img)
                h, w, _ = img.shape
                res_sorted = sorted_layout_boxes(res, w)
                for region in res_sorted:
                    if region['type'].lower() != 'table':
                        continue
                    r = region.get('res')
                    if not (isinstance(r, dict) and r.get('html')):
                        continue
                    # 用 header=None 保证第一行不被当列名吞掉；按列对(0,1)(2,3)...扫描支持多列宽表
                    try:
                        raw_dfs = pd.read_html(io.StringIO(r['html']), header=None)
                        if not raw_dfs:
                            continue
                        tbl_df = raw_dfs[0]
                    except Exception:
                        continue
                    if tbl_df is None or tbl_df.empty:
                        continue
                    for _, row_s in tbl_df.iterrows():
                        vals = ['' if str(v) == 'nan' else str(v).strip() for v in row_s]
                        # 每两列为一组 (key, value)，支持2列/4列/6列…宽表
                        i = 0
                        while i + 1 < len(vals):
                            kv = _parse_kv_row(vals[i], vals[i + 1])
                            if kv:
                                record[kv[0]] = kv[1]
                            i += 2
            except Exception as e:
                print(f'extract_record_via_ocr: page error: {e}')
    return record


def extract_record_from_file(input_path, ext, filename='', remove_red_seal=False):
    """
    统一入口：根据文件类型提取键值对，返回 **(dict, used_ocr: bool)**。
      .docx  → python-docx 直接读（used_ocr=False）
      .pdf   → 有文字层时用 pdfplumber（used_ocr=False），否则 OCR（used_ocr=True）
      图片   → OCR（used_ocr=True）
    """
    print(f'[extract_record] {filename} ext={ext}')

    if ext == 'docx':
        rec = extract_record_from_docx(input_path)
        print(f'[extract_record] docx → {len(rec)} 个字段')
        return rec, False

    if ext == 'pdf':
        try:
            with fitz.open(input_path) as pdf:
                text_count = sum(len(pdf[pg].get_text().strip()) for pg in range(pdf.page_count))
        except Exception:
            text_count = 0

        if text_count > 0:
            record = {}
            try:
                import pdfplumber
                with pdfplumber.open(input_path) as pdf:
                    for page in pdf.pages:
                        for tbl in (page.extract_tables() or []):
                            if not tbl or not tbl[0] or len(tbl[0]) < 2:
                                continue
                            for row in tbl:
                                # 每两列为一组 (key, value)，支持 2/4/6 列宽表
                                i = 0
                                while i + 1 < len(row):
                                    kv = _parse_kv_row(
                                        str(row[i] or '').strip(),
                                        str(row[i + 1] or '').strip()
                                    )
                                    if kv:
                                        record[kv[0]] = kv[1]
                                    i += 2
                print(f'[extract_record] pdfplumber → {len(record)} 个字段')
                if record:
                    return record, False
            except ImportError:
                pass
            except Exception as e:
                print(f'[extract_record] pdfplumber error: {e}')

            # pdfplumber 提取为空，但文字层存在：用 pdf2docx 转临时 docx 再提取
            print(f'[extract_record] pdfplumber 未提取到表格，尝试 pdf2docx 路径')
            try:
                import tempfile
                from pdf2docx.converter import Converter as _Pdf2DocxConverter
                tmp_fd, tmp_docx_path = tempfile.mkstemp(suffix='.docx')
                os.close(tmp_fd)
                try:
                    _cv = _Pdf2DocxConverter(input_path)
                    _cv.convert(tmp_docx_path)
                    _cv.close()
                    rec = extract_record_from_docx(tmp_docx_path)
                    print(f'[extract_record] pdf2docx fallback → {len(rec)} 个字段')
                    if rec:
                        return rec, False
                finally:
                    try:
                        os.unlink(tmp_docx_path)
                    except Exception:
                        pass
            except Exception as e:
                print(f'[extract_record] pdf2docx fallback error: {e}')

        rec = extract_record_via_ocr(input_path, ext, remove_red_seal=remove_red_seal)
        print(f'[extract_record] OCR fallback → {len(rec)} 个字段')
        return rec, True

    # 图片
    rec = extract_record_via_ocr(input_path, ext, remove_red_seal=remove_red_seal)
    print(f'[extract_record] OCR image → {len(rec)} 个字段')
    return rec, True


@app.route('/extract_special_form', methods=['POST'])
def extract_special_form():
    form_type = request.form.get('form_type', 'contract_esign')
    remove_red_seal = request.form.get('remove_red_seal', 'false').lower() == 'true'
    uploaded_files = request.files.getlist('files')

    if not uploaded_files:
        return jsonify({'error': 'No files uploaded'}), 400

    task_id = str(uuid.uuid4())
    batch_id = task_id[:8]

    try:
        if form_type == 'contract_esign':
            # ── 批量键值对提取：每个文件 → 一个 dict → Excel 里一行 ──────
            all_records = []   # 每个元素是一个 dict（一份文件 = 一行）

            for file in uploaded_files:
                if not file.filename:
                    continue
                filename = file.filename
                ext = filename.rsplit('.', 1)[-1].lower()
                input_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_{filename}")
                file.save(input_path)
                print(f'[extract] Processing: {filename}')

                try:
                    record, used_ocr = extract_record_from_file(input_path, ext, filename, remove_red_seal=remove_red_seal)
                    if record:
                        # 去除提取值中的所有空格
                        for k in record.keys():
                            if isinstance(record[k], str):
                                record[k] = re.sub(r'\s+', '', record[k])

                        if used_ocr:
                            # OCR 结果才需要：schema 对齐与前后缀串行污染修复
                            print('\n走了ocr~~~\n')
                            record = _repair_contract_esign_record(record)
                                
                        # --- 针对合同电子签章后处理：分离错误合并在姓名后面的电话 ---
                        for k in list(record.keys()):
                            if '收件人姓名' in k:
                                target_p_key = k.replace('姓名', '电话')
                                # 如果对应的收件人电话没提取到或为空
                                if not record.get(target_p_key, ''):
                                    name_val = record[k]
                                    # 匹配尾部的一串数字及可能混杂的特殊字符/英文字母/错别字“一”
                                    m = re.search(r'([0-9a-zA-Z\-_+()（）*#@,:：，。一]+)$', name_val)
                                    if m:
                                        tail_str = m.group(1)
                                        # 如果捕获的尾部中实际数字个数 >= 7则确认为电话
                                        if len(re.findall(r'\d', tail_str)) >= 7:
                                            record[target_p_key] = tail_str
                                            record[k] = name_val[:m.start()]
                        # ----------------------------------------------------------

                        # 把文件名作为第一列插入
                        record = {'文件名': filename, **record}
                        all_records.append(record)
                        print(f'[extract] {filename}: {len(record) - 1} 个字段')
                    else:
                        print(f'[extract] {filename}: 未提取到任何字段')
                except Exception as e:
                    print(f'[extract] Error for {filename}: {e}')
                    print(traceback.format_exc())

            if not all_records:
                return jsonify({'error': '未能从上传文件中识别到任何键值对字段'}), 400

            # 每个 dict 变一行，列名取所有 dict 键的并集，缺值补空
            merged_df = pd.DataFrame(all_records).fillna('')

            # ── 列名后处理：去掉尾部冒号，合并同类项 ──────────────────
            merged_df.columns = [_norm_key(c) for c in merged_df.columns]

            # 若规范化后出现重复列名，按行合并（取第一个非空值）
            if merged_df.columns.duplicated().any():
                unique_cols = list(dict.fromkeys(merged_df.columns))
                rows = []
                for _, row in merged_df.iterrows():
                    merged_row = {}
                    for col, val in zip(merged_df.columns, row):
                        if col not in merged_row:
                            merged_row[col] = val
                        elif merged_row[col] == '' and val != '':
                            merged_row[col] = val  # 用非空值补上
                    rows.append(merged_row)
                merged_df = pd.DataFrame(rows, columns=unique_cols).fillna('')
            # ────────────────────────────────────────────────────────────

            excel_filename = f"{batch_id}_tables.xlsx"
            excel_path = os.path.join(OUTPUT_FOLDER, excel_filename)
            merged_df.to_excel(excel_path, index=False)
            print(f'[extract] Excel saved: {excel_path}  ({len(all_records)} 行 × {len(merged_df.columns)} 列)')

            return jsonify({
                'task_id': task_id,
                'filename': excel_filename,
                'download_url': f'/download/{excel_filename}'
            })


        elif form_type == 'digital_cert_application':
            # ── 电子签章平台数字证书申请表特殊字段批量提取 ──
            all_records = []
            for file in uploaded_files:
                if not file.filename:
                    continue
                filename = file.filename
                ext = filename.rsplit('.', 1)[-1].lower()
                input_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_{filename}")
                file.save(input_path)
                # print(f'[extract_digital_cert] Processing: {filename}')
                try:
                    record = extract_digital_cert_fields_via_ocr(input_path, ext, filename, remove_red_seal=remove_red_seal)
                    # 去除提取值中的所有空格（保留原始文件名）
                    for k in record.keys():
                        if k != '文件名' and isinstance(record[k], str):
                            record[k] = re.sub(r'\s+', '', record[k])
                    all_records.append(record)
                    # print(f'[extract_digital_cert] {filename}: {record}')
                except Exception as e:
                    print(f'[extract_digital_cert] Error for {filename}: {e}')
                    print(traceback.format_exc())

            if not all_records:
                return jsonify({'error': '未能从上传文件中识别到任何目标字段'}), 400

            merged_df = pd.DataFrame(all_records).fillna('')
            excel_filename = f"{batch_id}_digital_cert.xlsx"
            excel_path = os.path.join(OUTPUT_FOLDER, excel_filename)
            merged_df.to_excel(excel_path, index=False)
            print(f'[extract_digital_cert] Excel saved: {excel_path}  ({len(all_records)} 行 × {len(merged_df.columns)} 列)')

            return jsonify({
                'task_id': task_id,
                'filename': excel_filename,
                'download_url': f'/download/{excel_filename}'
            })

        else:
            return jsonify({'error': f'Unknown form_type: {form_type}'}), 400

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Flask server starting at http://localhost:8006")
    app.run(host='0.0.0.0', port=8006, debug=False)
