#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将Layout模型预测结果转换为PPOCRLabel标注格式
PPOCRLabel标注格式：json字符串，包含图像中的所有标注框和对应的标签

使用方式：
python convert_to_ppocrlabel.py --pred_dir ./layout_results --image_dir ./images --output_dir ./ppocrlabel_annotations
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list

logger = get_logger()


def load_layout_predictions(pred_file):
    """
    加载layout预测结果
    结果格式应为：[{'bbox': [x1, y1, x2, y2], 'label': 'text', 'score': 0.95}, ...]
    """
    if not os.path.exists(pred_file):
        return []
    
    try:
        with open(pred_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return []
            # 尝试解析JSON格式
            result = json.loads(content)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                # 如果是字典格式，可能包含'regions'或类似的键
                if 'regions' in result:
                    return result['regions']
                else:
                    return result
            return []
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse {pred_file}: {e}")
        return []


def bbox_to_ppocrlabel_format(bbox, label, score=None):
    """
    将预测框转换为PPOCRLabel格式
    
    PPOCRLabel格式：
    {
        "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],  # 四个点的坐标（左上、右上、右下、左下）
        "text": "label_name"
    }
    
    参数：
    bbox: [x1, y1, x2, y2] 或 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    label: 标签名称
    score: 置信度（可选）
    """
    
    # 处理不同格式的bbox
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        if isinstance(bbox[0], (list, tuple)):
            # 已经是点的格式
            points = bbox
        else:
            # [x1, y1, x2, y2] 格式，转换为四个点
            x1, y1, x2, y2 = bbox
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    else:
        logger.warning(f"Invalid bbox format: {bbox}")
        return None
    
    ppocrlabel_item = {
        "bbox": points,
        "text": str(label)
    }
    
    if score is not None:
        ppocrlabel_item["score"] = float(score)
    
    return ppocrlabel_item


def convert_single_image(image_path, pred_results, min_score_threshold=0.0):
    """
    转换单张图像的预测结果
    
    参数：
    image_path: 图像路径
    pred_results: 预测结果列表
    min_score_threshold: 最低置信度阈值
    
    返回：
    符合PPOCRLabel格式的标注数据
    """
    
    # 过滤低置信度的预测结果
    filtered_results = []
    for item in pred_results:
        score = item.get('score', 1.0)
        if score >= min_score_threshold:
            filtered_results.append(item)
    
    ppocrlabel_data = []
    
    for item in filtered_results:
        bbox = item.get('bbox')
        label = item.get('label') or item.get('text', 'unknown')
        score = item.get('score', 1.0)
        
        ppocrlabel_item = bbox_to_ppocrlabel_format(bbox, label, score)
        if ppocrlabel_item:
            ppocrlabel_data.append(ppocrlabel_item)
    
    return ppocrlabel_data


def save_ppocrlabel_annotation(output_file, annotation_data):
    """
    保存PPOCRLabel格式的标注数据
    PPOCRLabel txt格式：一行一个样本，格式为 "image_path\t{json_annotation}"
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json_str = json.dumps(annotation_data, ensure_ascii=False)
            f.write(json_str + '\n')
        logger.info(f"Saved annotation to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {output_file}: {e}")
        return False


def create_ppocrlabel_txt(annotation_dir, image_dir, output_txt_path):
    """
    创建PPOCRLabel所需的txt文件
    格式：image_path\tjson_annotation
    
    参数：
    annotation_dir: 标注结果所在目录
    image_dir: 原始图像所在目录
    output_txt_path: 输出txt文件路径
    """
    
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    
    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
        for annotation_file in os.listdir(annotation_dir):
            if annotation_file.endswith('.json'):
                image_name = annotation_file.replace('.json', '')
                
                # 尝试找到对应的图像文件
                image_path = None
                for ext in ['.jpg', '.png', '.bmp', '.jpeg']:
                    potential_path = os.path.join(image_dir, image_name + ext)
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break
                
                if image_path is None:
                    logger.warning(f"Cannot find image for {annotation_file}")
                    continue
                
                # 读取标注文件
                annotation_path = os.path.join(annotation_dir, annotation_file)
                try:
                    with open(annotation_path, 'r', encoding='utf-8') as f:
                        annotation_data = f.read().strip()
                    
                    # 写入txt文件，格式为 "image_path\tjson"
                    # 注意：image_path可以是相对路径或绝对路径
                    relative_image_path = os.path.relpath(image_path)
                    txt_file.write(f"{relative_image_path}\t{annotation_data}\n")
                    
                except Exception as e:
                    logger.error(f"Error processing {annotation_file}: {e}")
    
    logger.info(f"Created PPOCRLabel txt file: {output_txt_path}")


def main(args):
    """主函数"""
    
    pred_dir = args.pred_dir
    image_dir = args.image_dir
    output_dir = args.output_dir
    min_score = args.min_score
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = get_image_file_list(image_dir)
    
    if not image_files:
        logger.error(f"No images found in {image_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images")
    
    converted_count = 0
    empty_count = 0
    
    for image_file in image_files:
        # 获取图像文件名（不含扩展名）
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        
        # 构造预测结果文件路径（假设预测结果以json格式保存）
        # 你需要根据实际的预测结果保存格式调整这里
        pred_file = os.path.join(pred_dir, image_name + '.json')
        
        # 如果预测结果文件不存在，尝试其他可能的位置
        if not os.path.exists(pred_file):
            # 可能保存在同名txt文件中
            pred_file = os.path.join(pred_dir, image_name + '.txt')
        
        if not os.path.exists(pred_file):
            logger.warning(f"Prediction file not found for {image_name}")
            continue
        
        # 加载预测结果
        pred_results = load_layout_predictions(pred_file)
        
        if not pred_results:
            logger.info(f"No predictions for {image_name}")
            empty_count += 1
            pred_results = []
        
        # 转换为PPOCRLabel格式
        ppocrlabel_data = convert_single_image(image_file, pred_results, min_score)
        
        # 保存标注结果
        output_file = os.path.join(output_dir, image_name + '.json')
        save_ppocrlabel_annotation(output_file, ppocrlabel_data)
        
        converted_count += 1
    
    logger.info(f"Conversion completed!")
    logger.info(f"Total converted: {converted_count}, Empty predictions: {empty_count}")
    
    # 创建PPOCRLabel txt文件（用于导入PPOCRLabel）
    if args.create_txt:
        txt_output = os.path.join(output_dir, 'ppocrlabel_annotations.txt')
        create_ppocrlabel_txt(output_dir, image_dir, txt_output)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Convert Layout predictions to PPOCRLabel format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--pred_dir',
        type=str,
        required=True,
        help='Directory containing layout prediction results'
    )
    
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='Directory containing original images'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./ppocrlabel_annotations',
        help='Output directory for PPOCRLabel format annotations'
    )
    
    parser.add_argument(
        '--min_score',
        type=float,
        default=0.0,
        help='Minimum confidence score threshold'
    )
    
    parser.add_argument(
        '--create_txt',
        action='store_true',
        default=True,
        help='Create txt file for PPOCRLabel import'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
