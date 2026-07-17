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
    例如: <html><body><table><tr><td>text</td></tr></table></body></html>
    变为: ['<html>', '<body>', '<table>', '<tr>', '<td>', '</td>', '</tr>', '</table>', '</body>', '</html>']
    """
    if not html_str:
        return []
    # 提取所有标签，包括自闭合标签和带属性的标签
    tokens = re.findall(r'<[^>]+>|[^<]+', html_str)
    refined_tokens = []
    for t in tokens:
        t = t.strip()
        if not t: continue
        if t.startswith('<'):
            # 统一清理属性，只保留标签名，如 <td rowspan="2"> 变为 <td>
            tag_name = re.findall(r'</?[a-zA-Z1-6]+', t)[0] + '>'
            refined_tokens.append(tag_name)
        else:
            # 文本内容在 tokens 列表中通常被结构化模型忽略，结构只存标签
            pass 
    return refined_tokens

def extract_cells_and_text(item):
    """
    从 predict_table 的输出中尝试还原 cells 列表。
    注意：由于 TableSystem 默认输出的是对齐后的 HTML，
    这里尝试将每项 OCR 结果及其坐标对应起来。
    """
    cells = []
    rec_res = item.get("rec_res", [])
    boxes = item.get("boxes", [])
    
    # 遍历 OCR 结果
    for i in range(len(boxes)):
        text = ""
        if i < len(rec_res):
            text = rec_res[i][0] if isinstance(rec_res[i], (list, tuple)) else str(rec_res[i])
        
        # 将文本拆分为字符 tokens (如 'Paddle' -> ['P','a','d','d','l','e'])
        char_tokens = list(text)
        
        # 转换坐标格式 [x0, y0, x1, y1]
        box = boxes[i]
        if len(box) == 4:
            bbox = [int(v) for v in box]
        else:
            # 如果是 8 点坐标，取外接矩形
            xs = [p[0] for p in box] if isinstance(box[0], list) else [box[i] for i in range(0,len(box),2)]
            ys = [p[1] for p in box] if isinstance(box[0], list) else [box[i] for i in range(1,len(box),2)]
            bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            
        cells.append({
            "tokens": char_tokens,
            "bbox": bbox
        })
    return cells

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入的 batch_results.json 路径")
    ap.add_argument("--output", required=True, help="输出的训练集 jsonl 路径")
    ap.add_argument("--split", default="train", help="数据集划分: train 或 val")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        return

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_lines = []
    for idx, item in enumerate(data):
        filename = item.get("basename") or os.path.basename(item.get("image_file", ""))
        
        # 1. 提取 HTML 结构 tokens
        structure_tokens = parse_html_to_tokens(item.get("html", ""))
        
        # 2. 提取单元格文本及坐标
        cells = extract_cells_and_text(item)
        
        # 3. 组装格式
        record = {
            "filename": filename,
            "split": args.split,
            "imgid": idx,
            "html": {
                "structure": {"tokens": structure_tokens},
                "cells": cells
            }
        }
        output_lines.append(json.dumps(record, ensure_ascii=False))

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines) + "\n")

    print(f"成功转换 {len(output_lines)} 条数据，已保存至: {args.output}")

if __name__ == "__main__":
    main()