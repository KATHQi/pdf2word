#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
只使用 table 输出文件生成 Paddle 标注格式的：
1. fileState.txt
2. Label.txt

输入：
- batch_results.json   (必需)
- batch_results.csv    (可选，仅用于校验 basename / image_file)
- show.html            (可选，仅用于校验，不参与主逻辑)

输出的 Label.txt 格式：
image_path \t [{"transcription": "...", "points": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]], "difficult": false}]

设计说明：
- points: 由 batch_results.json 中的 table_bbox / bbox / table_box / cell_bbox 并集得到
- transcription:
    1) 优先从 html 中提取“可见文字”
    2) 自动清洗 colspan / rowspan / border 等结构噪声
    3) 若清洗后无文字，则置为空字符串

注意：
- 对你当前这批文件，html 基本是坏的，boxes/rec_res 为空，因此很多 transcription 可能为空。

python convert_table_outputs_to_paddle.py \
  --table_json batch_results.json \
  --table_csv batch_results.csv \
  --show_html show.html \
  --out_label Label.txt \
  --out_state fileState.txt

如果你想强制让 transcription 全部为空，直接这样跑：
python convert_table_outputs_to_paddle.py \
  --table_json batch_results.json \
  --table_csv batch_results.csv \
  --show_html show.html \
  --out_label Label.txt \
  --out_state fileState.txt \
  --force_empty_transcription


"""

import os
import re
import csv
import json
import html as ihtml
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Optional[str]):
    if not path or not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def basename_norm(p: str) -> str:
    return os.path.basename(str(p)).lower().strip()


def points_to_rect(points: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def union_rect(rects: List[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not rects:
        return None
    x1 = min(r[0] for r in rects)
    y1 = min(r[1] for r in rects)
    x2 = max(r[2] for r in rects)
    y2 = max(r[3] for r in rects)
    return x1, y1, x2, y2


def rect_to_points(rect: Tuple[float, float, float, float]) -> List[List[int]]:
    x1, y1, x2, y2 = rect
    return [
        [int(round(x1)), int(round(y1))],
        [int(round(x2)), int(round(y1))],
        [int(round(x2)), int(round(y2))],
        [int(round(x1)), int(round(y2))]
    ]


def get_table_rect(item: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    # 1) 优先用显式 table bbox
    for k in ("table_bbox", "bbox", "table_box"):
        v = item.get(k)
        if not v:
            continue
        if isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
            x1, y1, x2, y2 = map(float, v)
            return x1, y1, x2, y2
        if isinstance(v, list) and len(v) >= 4 and isinstance(v[0], (list, tuple)):
            return points_to_rect(v)

    # 2) 否则退化为 cell_bbox 并集
    cell_bbox = item.get("cell_bbox", [])
    rects = []
    for box in cell_bbox:
        if isinstance(box, list) and len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
            rects.append(tuple(map(float, box)))
        elif isinstance(box, list) and len(box) >= 4 and isinstance(box[0], (list, tuple)):
            rects.append(points_to_rect(box))
    return union_rect(rects)


def clean_html_to_text(raw_html: str) -> str:
    """
    从 table html 提取可见文本，同时尽量去掉结构噪声。
    当前 batch_results.json 里的 html 很多是坏的，所以这里做强清洗。
    """
    if not raw_html:
        return ""

    s = raw_html

    # 去 script/style
    s = re.sub(r"<script.*?>.*?</script>", " ", s, flags=re.I | re.S)
    s = re.sub(r"<style.*?>.*?</style>", " ", s, flags=re.I | re.S)

    # 对换行更友好的标签先替换
    s = re.sub(r"</tr\s*>", "\n", s, flags=re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</p\s*>", "\n", s, flags=re.I)
    s = re.sub(r"</div\s*>", "\n", s, flags=re.I)
    s = re.sub(r"</td\s*>", "\t", s, flags=re.I)
    s = re.sub(r"</th\s*>", "\t", s, flags=re.I)

    # 去标签
    s = re.sub(r"<[^>]+>", " ", s)

    # HTML entity
    s = ihtml.unescape(s)

    # 去掉纯结构噪声
    s = re.sub(r'\b(?:colspan|rowspan|border|width|height|cellspacing|cellpadding)\s*=\s*"?[^"\s>]+"?', " ", s, flags=re.I)

    # 去掉孤立的 table/thead/tbody/tr/td/th 等词
    s = re.sub(r"\b(?:html|body|table|thead|tbody|tr|td|th)\b", " ", s, flags=re.I)

    # 去掉纯符号片段
    s = re.sub(r"[<>=\"/]+", " ", s)

    # 按行清洗
    lines = []
    for line in s.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            continue

        # 若整行几乎全是噪声关键词，则丢弃
        tmp = re.sub(r"\b(?:colspan|rowspan)\b", "", line, flags=re.I).strip()
        if not tmp:
            continue

        # 如果整行不含中英文数字，通常也没意义
        if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", line):
            continue

        lines.append(line)

    # 去重相邻重复行
    dedup = []
    for line in lines:
        if not dedup or dedup[-1] != line:
            dedup.append(line)

    text = "\n".join(dedup).strip()

    # 如果清洗后仍只剩结构噪声，则置空
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text):
        return ""
    stripped = re.sub(r"\b(?:colspan|rowspan)\b", "", text, flags=re.I)
    stripped = re.sub(r'[\s"=0-9]+', "", stripped)
    if not stripped:
        return ""

    return text


def choose_image_path(item: Dict[str, Any], csv_map: Dict[str, Dict[str, str]], prefer: str = "json") -> str:
    image_file = item.get("image_file", "")
    basename = item.get("basename", "") or os.path.basename(image_file)

    if prefer == "csv":
        row = csv_map.get(basename_norm(basename))
        if row and row.get("image_file"):
            return row["image_file"]
    return image_file or basename


def build_csv_map(csv_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    m = {}
    for row in csv_rows:
        b = row.get("basename") or os.path.basename(row.get("image_file", ""))
        if b:
            m[basename_norm(b)] = row
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table_json", required=True, help="batch_results.json")
    ap.add_argument("--table_csv", default="", help="batch_results.csv，可选")
    ap.add_argument("--show_html", default="", help="show.html，可选，仅用于存在性校验")
    ap.add_argument("--out_label", required=True, help="输出 Label.txt")
    ap.add_argument("--out_state", required=True, help="输出 fileState.txt")
    ap.add_argument("--image_dir", default=r"D:\kath-workfile\PDF2WORD\Dataset\table\img1", help="可选：自定义图片所在目录路径（例如 D:\\kath-workfile\\PDF2WORD\\Dataset\\table\\img1）")
    ap.add_argument("--prefer_path_from", default="json", choices=["json", "csv"], help="输出路径优先来源")
    ap.add_argument("--force_empty_transcription", action="store_true", help="强制 transcription 为空，不尝试从 html 提取文字")
    args = ap.parse_args()

    items = read_json(args.table_json)
    csv_rows = read_csv_rows(args.table_csv)
    csv_map = build_csv_map(csv_rows)

    # show.html 只做存在性检查，不参与主逻辑
    if args.show_html and not Path(args.show_html).exists():
        raise FileNotFoundError(f"show.html 不存在: {args.show_html}")

    label_lines = []
    state_lines = []

    total = 0
    no_rect = 0
    empty_text = 0

    for item in items:
        total += 1
        raw_image_path = choose_image_path(item, csv_map, prefer=args.prefer_path_from)
        filename = os.path.basename(raw_image_path)
        
        # 处理路径：
        # state_path: 完整路径 (image_dir + filename)
        # label_path: 最后一层目录 + / + filename
        if args.image_dir:
            state_path = os.path.join(args.image_dir, filename)
            last_dir = os.path.basename(os.path.normpath(args.image_dir))
            label_path = f"{last_dir}/{filename}"
        else:
            state_path = raw_image_path
            label_path = raw_image_path

        # 优先使用 OCR 检测框 (boxes)，包含真实的文字区域，数量远多于 cell_bbox
        # 若 boxes 为空则退回到 cell_bbox
        ocr_boxes = item.get("boxes", [])
        rec_res_list = item.get("rec_res", [])
        source_boxes = ocr_boxes if ocr_boxes else item.get("cell_bbox", [])

        objs = []
        for i, box in enumerate(source_boxes):
            if isinstance(box, list) and len(box) == 4:
                rect = tuple(map(float, box))
                points = rect_to_points(rect)

                transcription = ""
                if i < len(rec_res_list):
                    entry = rec_res_list[i]
                    transcription = entry[0] if isinstance(entry, (list, tuple)) else str(entry)

                objs.append({
                    "transcription": transcription,
                    "points": points,
                    "difficult": False
                })
        print(f"处理图片 {filename}: boxes={len(ocr_boxes)}, cell_bbox={len(item.get('cell_bbox',[]))}, 输出={len(objs)} 个框")

        if not objs:
            no_rect += 1
            continue

        label_lines.append(f"{label_path}\t{json.dumps(objs, ensure_ascii=False)}")
        state_lines.append(f"{state_path}\t1")

    Path(args.out_label).write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    Path(args.out_state).write_text("\n".join(state_lines) + ("\n" if state_lines else ""), encoding="utf-8")

    print("=" * 80)
    print("转换完成")
    print(f"总样本数              : {total}")
    print(f"成功输出样本数        : {len(label_lines)}")
    print(f"缺少可用表格框数      : {no_rect}")
    print(f"transcription 为空数  : {empty_text}")
    print(f"输出 Label.txt        : {args.out_label}")
    print(f"输出 fileState.txt    : {args.out_state}")
    print("=" * 80)


if __name__ == "__main__":
    main()