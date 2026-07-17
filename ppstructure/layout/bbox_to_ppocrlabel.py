#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将Layout模型bbox.json预测结果转换为PPOCRLabel的Label.txt和fileState.txt格式

bbox.json格式（PicoDetPostProcess输出）：
[
    {
        "image_id": 81,
        "category_id": 0,
        "file_name": "merged_pdf_p0020.png",
        "bbox": [x, y, w, h],
        "box_": [category_id, score, x1, y1, x2, y2],
        "score": 0.8...
    },
    ...
]

PPOCRLabel格式：
Label.txt: image_path\t[{"transcription": "label_name", "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]}]
fileState.txt: image_name\t0 (0表示未标注需要标注，1表示已标注)
"""

import json
import os
import argparse
import sys
from pathlib import Path
from collections import defaultdict

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list

logger = get_logger()

# Layout类别名称映射 (对应模型配置的label_list顺序)
LAYOUT_LABEL_NAMES = {
    0: "Text",              # 文本
    1: "Title",             # 标题
    2: "Figure",            # 图片
    3: "Figure caption",    # 图片说明
    4: "Table",             # 表格
    5: "Table caption",     # 表格说明
    6: "Header",            # 页眉
    7: "Footer",            # 页脚
    8: "Reference",         # 参考文献
    9: "Equation"           # 公式
}


def load_bbox_json(json_file):
    """加载bbox.json文件"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Failed to load {json_file}: {e}")
        return []


def convert_bbox_format(bbox, box_info=None):
    """
    转换bbox格式
    
    输入：
    - bbox: [x, y, w, h] (x, y是左上角坐标，w, h是宽高)
    - box_info: [category_id, score, x1, y1, x2, y2]
    
    输出：
    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] (四个顶点，顺序：左上、右上、右下、左下)
    """
    if box_info and len(box_info) >= 6:
        # 使用box_中的坐标 (已经是x1, y1, x2, y2格式)
        x1, y1, x2, y2 = box_info[2], box_info[3], box_info[4], box_info[5]
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        # 转换bbox [x, y, w, h] 为 [x1, y1, x2, y2]
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        x1, y1, x2, y2 = x, y, x + w, y + h
    else:
        return None
    
    # 转换为四个顶点 (PPOCRLabel格式)
    points = [
        [int(x1), int(y1)],  # 左上
        [int(x2), int(y1)],  # 右上
        [int(x2), int(y2)],  # 右下
        [int(x1), int(y2)]   # 左下
    ]
    return points


def get_label_name(category_id):
    """获取标签名称"""
    return LAYOUT_LABEL_NAMES.get(category_id, f"category_{category_id}")


def process_bbox_json(json_file, output_label_txt, output_filestate_txt, min_score=0.0, image_dir=None):
    """
    处理bbox.json文件，生成Label.txt和fileState.txt
    
    参数：
    - json_file: bbox.json文件路径
    - output_label_txt: 输出的Label.txt路径
    - output_filestate_txt: 输出的fileState.txt路径
    - min_score: 最小置信度阈值
    - image_dir: 图像目录（用于验证图像是否存在）
    """
    
    # 加载json数据
    bbox_data = load_bbox_json(json_file)
    if not bbox_data:
        logger.error("No data found in bbox.json")
        return False
    
    logger.info(f"Loaded {len(bbox_data)} bbox entries")
    
    # 按图像文件名分组
    image_groups = defaultdict(list)
    for item in bbox_data:
        file_name = item.get("file_name", "unknown")
        score = item.get("score", 1.0)
        
        # 过滤低置信度的预测
        if score < min_score:
            continue
        
        image_groups[file_name].append(item)
    
    logger.info(f"Processing {len(image_groups)} images")
    
    # 生成Label.txt
    label_list = []
    filestate_list = []
    
    for file_name, items in sorted(image_groups.items()):
        # Label.txt使用相对路径 (img/xxx.png格式，斜杠为/)
        relative_image_path = f"img/{file_name}".replace('\\', '/')
        
        # fileState.txt使用绝对路径
        if image_dir:
            absolute_image_path = os.path.abspath(os.path.join(image_dir, file_name))
        else:
            absolute_image_path = os.path.abspath(file_name)
        
        # 构造PPOCRLabel标注格式
        annotations = []
        for item in items:
            bbox = item.get("bbox", [])
            box_info = item.get("box_", None)
            category_id = item.get("category_id", 0)
            score = item.get("score", 1.0)
            
            points = convert_bbox_format(bbox, box_info)
            if points is None:
                logger.warning(f"Invalid bbox for {file_name}: {bbox}")
                continue
            
            label_name = get_label_name(category_id)
            
            annotation = {
                "transcription": label_name,
                "points": points,
                "key_cls": category_id,
                "score": round(float(score), 4)
            }
            annotations.append(annotation)
        
        if annotations:
            # 格式：relative_image_path\t[json_data]
            json_str = json.dumps(annotations, ensure_ascii=False)
            label_line = f"{relative_image_path}\t{json_str}"
            label_list.append(label_line)
            
            # fileState.txt格式：absolute_image_path\t0 (0表示未标注，1表示已标注)
            # 这里我们标记为0（未标注），用户可以在PPOCRLabel中标注后改为1
            filestate_line = f"{absolute_image_path}\t0"
            filestate_list.append(filestate_line)
    
    # 写入Label.txt
    os.makedirs(os.path.dirname(output_label_txt), exist_ok=True)
    try:
        with open(output_label_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(label_list))
        logger.info(f"Created Label.txt with {len(label_list)} entries: {output_label_txt}")
    except Exception as e:
        logger.error(f"Failed to write Label.txt: {e}")
        return False
    
    # 写入fileState.txt
    os.makedirs(os.path.dirname(output_filestate_txt), exist_ok=True)
    try:
        with open(output_filestate_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(filestate_list))
        logger.info(f"Created fileState.txt with {len(filestate_list)} entries: {output_filestate_txt}")
    except Exception as e:
        logger.error(f"Failed to write fileState.txt: {e}")
        return False
    
    logger.info(f"Conversion completed successfully!")
    logger.info(f"  - Total images: {len(image_groups)}")
    logger.info(f"  - Total annotations: {sum(len(items) for items in image_groups.values())}")
    
    return True


def main(args):
    """主函数"""
    
    json_file = args.json_file
    output_dir = args.output_dir
    min_score = args.min_score
    image_dir = args.image_dir
    
    if not os.path.exists(json_file):
        logger.error(f"bbox.json file not found: {json_file}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出文件路径
    output_label_txt = os.path.join(output_dir, "Label.txt")
    output_filestate_txt = os.path.join(output_dir, "fileState.txt")
    
    # 处理
    success = process_bbox_json(
        json_file,
        output_label_txt,
        output_filestate_txt,
        min_score=min_score,
        image_dir=image_dir
    )
    
    if success:
        logger.info(f"\n✓ Conversion successful!")
        logger.info(f"You can now open the Label.txt in PPOCRLabel:")
        logger.info(f"  python PPOCRLabel.py --labelfile {output_label_txt}")
    else:
        logger.error("Conversion failed!")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Convert bbox.json to PPOCRLabel format (Label.txt and fileState.txt)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--json_file',
        type=str,
        required=True,
        help='Path to bbox.json file from layout prediction'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./ppocrlabel_output',
        help='Output directory for Label.txt and fileState.txt'
    )
    
    parser.add_argument(
        '--min_score',
        type=float,
        default=0.0,
        help='Minimum confidence score threshold for filtering predictions'
    )
    
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Original image directory (optional, for constructing full image paths)'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
