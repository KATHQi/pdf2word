#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成 TableMaster / PubTabNet 风格的数据集格式：
每一行是一个包含 filename, split, html(structure, cells) 的 JSON 对象。
"""

import os
import json
import argparse
import re
from pathlib import Path

def parse_html_to_tokens(html_str):
    """
    将 HTML 拆分为结构化的 tokens。
    例如: <thead><tr><td>text</td></tr></thead>
    变为: ['<thead>', '<tr>', '<td>', '</td>', '</tr>', '</thead>']
    """
    if not html_str:
        return []
    
    # 按照标签和非标签内容拆分
    tokens = re.findall(r'<[^>]+>|[^<]+', html_str)
    refined_tokens = []
    
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if t.startswith('<'):
            # 处理带属性的标签，如 <td rowspan="2"> 统一转为 <td>
            tag_match = re.search(r'</?([a-zA-Z1-6]+)', t)
            if tag_match:
                tag_name = tag_match.group(1)
                # 过滤掉非表格核心结构的标签 (可选)
                if tag_name.lower() in ['html', 'body', 'table']:
                    continue
                
                # 构造标准化的 token
                if t.startswith('</'):
                    refined_tokens.append(f'</{tag_name}>')
                else:
                    refined_tokens.append(f'<{tag_name}>')
        else:
            # 文本内容不计入 structure tokens
            pass
            
    return refined_tokens

def extract_cells_and_text(item):
    """
    从 predict_table 的输出中提取 cells 列表。
    """
    cells = []
    rec_res = item.get("rec_res", [])
    boxes = item.get("boxes", [])
    
    # 确保 boxes 和 rec_res 长度匹配
    for i in range(len(boxes)):
        text = ""
        if i < len(rec_res):
            # PaddleOCR 的 rec_res 格式通常是 [text, score]
            entry = rec_res[i]
            text = entry[0] if isinstance(entry, (list, tuple)) else str(entry)
        
        # 将文本拆分为字符 tokens
        char_tokens = list(text)
        
        # 处理坐标 [x0, y0, x1, y1]
        box = boxes[i]
        if len(box) == 4:
            # 矩形框格式 [x_min, y_min, x_max, y_max]
            bbox = [int(v) for v in box]
        elif len(box) >= 8:
            # 可能是点坐标格式 [x1, y1, x2, y2, x3, y3, x4, y4]
            if isinstance(box[0], (int, float)):
                xs = [box[j] for j in range(0,len(box),2)]
                ys = [box[j] for j in range(1,len(box),2)]
            else:
                # 可能是 [[x1,y1], [x2,y2]...]
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
            bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        else:
            bbox = [0, 0, 0, 0]
            
        cells.append({
            "tokens": char_tokens,
            "bbox": bbox
        })
    return cells

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table_json", required=True, help="输入的 batch_results.json 路径")
    ap.add_argument("--output", required=True, help="输出的 json 路径")
    ap.add_argument("--split", default="train", help="数据集划分标记: train, val, etc.")
    args = ap.parse_args()

    if not os.path.exists(args.table_json):
        print(f"Error: 找不到输入文件 {args.table_json}")
        return

    # 加载 JSON 数据
    with open(args.table_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 按照要求的格式转换
    dataset = []
    for idx, item in enumerate(data):
        filename = item.get("basename") or os.path.basename(item.get("image_file", ""))
        
        # 1. 提取 HTML 结构 tokens
        structure_tokens = parse_html_to_tokens(item.get("html", ""))
        
        # 2. 提取单元格文本及其坐标
        cells = extract_cells_and_text(item)
        
        # 3. 构造单条数据
        record = {
            "filename": filename,
            "split": args.split,
            "imgid": idx,
            "html": {
                "structure": {"tokens": structure_tokens},
                "cells": cells
            }
        }
        dataset.append(record)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"成功转换 {len(dataset)} 条数据。")
    print(f"输出文件: {args.output}")

if __name__ == "__main__":
    main()
