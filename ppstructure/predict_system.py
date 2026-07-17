# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import json
import numpy as np
import time
import logging
from copy import deepcopy

from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from ppocr.utils.visual import draw_ser_results, draw_re_results
from tools.infer.predict_system import TextSystem
from ppstructure.layout.predict_layout import LayoutPredictor
from ppstructure.table.predict_table import TableSystem, to_excel
from ppstructure.utility import parse_args, draw_structure_result

logger = get_logger()


def remove_red_seal(img):
    """
    去除图像中的红色印章
    返回去除红章后的图像
    """
    try:
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 定义红色的HSV范围（红色在HSV中分两段）
        # 红色范围1: 0-10
        lower_red1 = np.array([0, 43, 46])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        # 红色范围2: 156-180
        lower_red2 = np.array([156, 43, 46])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # 合并两个红色mask
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学操作，去除噪点
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # 将红色区域替换为白色
        result = img.copy()
        result[red_mask > 0] = [255, 255, 255]
        
        logger.info("Red seal removal completed")
        return result
    except Exception as e:
        logger.warning(f"Failed to remove red seal: {e}, using original image")
        return img


def save_image_with_chinese_path(img, save_path):
    """
    保存图像到中文路径，规避cv2.imwrite的中文路径问题
    使用PIL库来处理中文路径的图像保存
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 方法1：先用cv2.imwrite尝试保存（如果路径中没有中文会成功）
        result = cv2.imwrite(save_path, img)
        if result:
            logger.info(f"Image saved successfully with cv2: {save_path}")
            return True
        
        # 方法2：如果cv2失败，使用PIL保存（支持中文路径）
        logger.info(f"cv2.imwrite failed, using PIL for: {save_path}")
        from PIL import Image as PILImage
        
        # 将OpenCV的BGR格式转换为PIL的RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        pil_img.save(save_path, 'JPEG', quality=95)
        logger.info(f"Image saved successfully with PIL: {save_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving image to {save_path}: {type(e).__name__}: {e}")
        return False


def calculate_iou(box1, box2):
    """
    计算两个边界框的 IoU（交并比）
    box1, box2 格式：[x1, y1, x2, y2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou

def filter_contained_layout_regions(layout_res, containment_threshold=0.95):
    """
    过滤同类型的layout区域：如果A区域>=threshold包含B区域，则去掉B区域。
    支持传递性：如果C包含在B中，B包含在A中，最终只保留A。
    """
    if len(layout_res) <= 1:
        return layout_res

    # 按类型分组
    from collections import defaultdict
    groups = defaultdict(list)
    no_bbox_regions = []
    for idx, region in enumerate(layout_res):
        if region['bbox'] is None:
            no_bbox_regions.append(region)
        else:
            groups[region['label'].lower()].append((idx, region))

    removed_indices = set()

    for label, regions in groups.items():
        n = len(regions)
        if n <= 1:
            continue

        # 计算B区域被A区域包含的比例（B的面积中有多少在A内）
        # 如果 >= threshold，说明A包含B
        contain_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                bbox_a = regions[i][1]['bbox']
                bbox_b = regions[j][1]['bbox']

                # 计算A包含B的比例 = intersection / area_B
                x1_inter = max(bbox_a[0], bbox_b[0])
                y1_inter = max(bbox_a[1], bbox_b[1])
                x2_inter = min(bbox_a[2], bbox_b[2])
                y2_inter = min(bbox_a[3], bbox_b[3])

                if x2_inter <= x1_inter or y2_inter <= y1_inter:
                    continue

                inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
                if area_b > 0:
                    contain_matrix[i][j] = inter_area / area_b

        # 对每个区域j，如果存在某个区域i使得i包含j（contain_matrix[i][j] >= threshold），
        # 则标记j为被包含。传递性通过面积排序保证：大区域不会被小区域包含。
        # 按面积从大到小排序，大的优先作为"容器"
        area_order = sorted(range(n), key=lambda k: (
            (regions[k][1]['bbox'][2] - regions[k][1]['bbox'][0]) *
            (regions[k][1]['bbox'][3] - regions[k][1]['bbox'][1])
        ), reverse=True)

        local_removed = set()
        for rank_i, i in enumerate(area_order):
            if i in local_removed:
                continue
            for j in area_order[rank_i + 1:]:
                if j in local_removed:
                    continue
                if contain_matrix[i][j] >= containment_threshold:
                    local_removed.add(j)
                    removed_indices.add(regions[j][0])

    # 构建过滤后的结果
    filtered = [r for idx, r in enumerate(layout_res) if idx not in removed_indices and r['bbox'] is not None]
    filtered = no_bbox_regions + filtered

    return filtered

def match_det_to_layout(det_boxes, layout_regions, iou_threshold=0.3):
    """
    将 det 检测结果与 layout 分析结果进行匹配
    
    输入：
        det_boxes: List[box]，格式为 [x1, y1, x2, y2]
        layout_regions: List[Dict]，包含 bbox 和 label
        iou_threshold: float，IoU 阈值
    
    输出：
        matched_layout: Dict，key 为 layout 的 bbox，value 为属于该 layout 的 det_boxes 列表
        unmatched_det: List，未被匹配的 det_boxes
    """
    matched_layout = {}
    for region in layout_regions:
        if region['bbox'] is not None:
            matched_layout[tuple(region['bbox'])] = []
    
    matched_det_indices = set()
    
    # 对每个 det box，找到最合适的 layout region
    for det_idx, det_box in enumerate(det_boxes):
        best_layout_idx = -1
        best_score = -1
        
        for layout_idx, layout_region in enumerate(layout_regions):
            layout_bbox = layout_region['bbox']
            if layout_bbox is None:
                continue
            
            # 对于table区域，优先使用"包含关系"而非IoU
            # 检查det_box是否主要位于layout_bbox内
            x1_inter = max(det_box[0], layout_bbox[0])
            y1_inter = max(det_box[1], layout_bbox[1])
            x2_inter = min(det_box[2], layout_bbox[2])
            y2_inter = min(det_box[3], layout_bbox[3])
            
            if x2_inter < x1_inter or y2_inter < y1_inter:
                # 没有交集
                continue
            
            # 计算交集面积和det框面积的比例（重叠率）
            inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
            
            # 如果det框大部分在layout内，则匹配（阈值0.5表示至少50%重叠）
            overlap_ratio = inter_area / det_area if det_area > 0 else 0
            
            if overlap_ratio >= 0.5:  # det至少50%在layout内
                if overlap_ratio > best_score:
                    best_score = overlap_ratio
                    best_layout_idx = layout_idx
        
        # 如果找到了匹配的 layout，记录这个 det box
        if best_layout_idx >= 0:
            layout_key = tuple(layout_regions[best_layout_idx]['bbox'])
            matched_layout[layout_key].append(det_idx)
            matched_det_indices.add(det_idx)
    
    # 收集未被匹配的 det boxes
    unmatched_det = [
        (idx, det_boxes[idx]) 
        for idx in range(len(det_boxes)) 
        if idx not in matched_det_indices
    ]
    
    return matched_layout, unmatched_det


class StructureSystem(object):
    def __init__(self, args):
        self.mode = args.mode
        self.recovery = args.recovery

        self.image_orientation_predictor = None
        if args.image_orientation:
            import paddleclas
            self.image_orientation_predictor = paddleclas.PaddleClas(
                model_name="text_image_orientation")

        if self.mode == 'structure':
            if not args.show_log:
                logger.setLevel(logging.INFO)
            if args.layout == False and args.ocr == True:
                args.ocr = False
                logger.warning(
                    "When args.layout is false, args.ocr is automatically set to false"
                )
            args.drop_score = 0
            
            # ========== 调整 det 参数以提高召回率，尽可能多地检测文字 ==========
            args.det_db_thresh = 0.2      # 降低置信度阈值，检测更多文字（默认0.3）
            args.det_db_box_thresh = 0.4  # 降低边界框过滤阈值（默认0.6）
            args.det_db_unclip_ratio = 1.8  # 增加边界框扩展比例，得到更大的框（默认1.5）
            logger.info(f"Det参数调整: thresh={args.det_db_thresh}, box_thresh={args.det_db_box_thresh}, unclip_ratio={args.det_db_unclip_ratio}")
            
            # init model
            self.layout_predictor = None
            self.text_system = None
            self.table_system = None
            if args.layout:
                self.layout_predictor = LayoutPredictor(args)
                if args.ocr:
                    self.text_system = TextSystem(args)
            if args.table:
                if self.text_system is not None:
                    self.table_system = TableSystem(
                        args, self.text_system.text_detector,
                        self.text_system.text_recognizer)
                else:
                    self.table_system = TableSystem(args)

        elif self.mode == 'kie':
            from ppstructure.kie.predict_kie_token_ser_re import SerRePredictor
            self.kie_predictor = SerRePredictor(args)

    def __call__(self, img, return_ocr_result_in_table=False, img_idx=0):
        # # 转换为灰度图（黑白，不保留色彩）
        # if len(img.shape) == 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     # 转回3通道以兼容后续处理
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        time_dict = {
            'image_orientation': 0,
            'layout': 0,
            'table': 0,
            'table_match': 0,
            'det': 0,
            'rec': 0,
            'kie': 0,
            'all': 0
        }
        start = time.time()
        if self.image_orientation_predictor is not None:
            tic = time.time()
            cls_result = self.image_orientation_predictor.predict(
                input_data=img)
            cls_res = next(cls_result)
            angle = cls_res[0]['label_names'][0]
            cv_rotate_code = {
                '90': cv2.ROTATE_90_COUNTERCLOCKWISE,
                '180': cv2.ROTATE_180,
                '270': cv2.ROTATE_90_CLOCKWISE
            }
            if angle in cv_rotate_code:
                img = cv2.rotate(img, cv_rotate_code[angle])
            toc = time.time()
            time_dict['image_orientation'] = toc - tic
        if self.mode == 'structure':
            ori_im = img.copy()
            h, w = ori_im.shape[:2]
            
            # ========== 第 1 步：先做 layout 分析 ==========
            if self.layout_predictor is not None:
                layout_res, elapse = self.layout_predictor(img)
                time_dict['layout'] += elapse
            else:
                layout_res = [dict(bbox=None, label='table')]
            
            logger.info(f"Layout detected {len(layout_res)} regions")
            
            # ========== 第 1.5 步：过滤同类型的重叠layout区域 ==========
            layout_res = filter_contained_layout_regions(layout_res, containment_threshold=0.95)
            logger.info(f"After filtering contained regions: {len(layout_res)} regions remain")
            
            # ========== 第 2 步：对 reference 区域去红章，直接替换到ori_im1上 ==========
            ori_im1 = ori_im.copy()  # 深拷贝原图
            reference_cleaned_imgs = {}  # 存储去红章后的reference区域图像（用于保存）
            
            for region in layout_res:
                if region['label'].lower() == 'reference' and region['bbox'] is not None:
                    x1, y1, x2, y2 = region['bbox']
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 提取reference区域
                    roi_img = ori_im[y1:y2, x1:x2, :]
                    
                    # 去红章
                    cleaned_roi = remove_red_seal(roi_img)
                    
                    # 直接替换到ori_im1上
                    ori_im1[y1:y2, x1:x2, :] = cleaned_roi
                    
                    # 保存去红章后的图像（用于后续保存到文件）
                    reference_cleaned_imgs[tuple(region['bbox'])] = cleaned_roi
                    
                    logger.info(f"Reference region detected and cleaned: bbox={region['bbox']}")
            
            # ========== 第 3 步：对处理后的ori_im1做一次det和rec ==========
            all_det_boxes = []
            all_rec_results = []
            
            if self.text_system is not None:
                tic = time.time()
                
                # 对处理后的图像（reference区域已去红章）做一次det+rec
                filter_boxes, filter_rec_res, ocr_time_dict = self.text_system(ori_im1)
                time_dict['det'] += ocr_time_dict['det']
                time_dict['rec'] += ocr_time_dict['rec']
                
                # 转换为 bbox 格式 [x1, y1, x2, y2]
                for box, rec_res in zip(filter_boxes, filter_rec_res):
                    x_coords = [point[0] for point in box]
                    y_coords = [point[1] for point in box]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    
                    all_det_boxes.append([x1, y1, x2, y2])
                    all_rec_results.append(rec_res)
                
                toc = time.time()
                
                logger.info(f"Det found {len(all_det_boxes)} text regions (reference regions processed with red seal removed)")
            
            # ========== 第 4 步：利用匹配逻辑将 det 结果与 layout 关联 ==========
            res_list = []
            used_det_indices = set()  # 追踪哪些 det 已被 table 系统处理过
            
            if len(all_det_boxes) > 0:
                # 使用改进的匹配算法：基于"包含关系"而非IoU
                # 判断det框是否主要位于某个layout区域内（至少50%重叠）
                matched_layout, unmatched_det = match_det_to_layout(
                    all_det_boxes, 
                    layout_res
                )
                
                logger.info(f"Matched {sum(len(v) for v in matched_layout.values())} det to layout regions")
                logger.info(f"Unmatched det: {len(unmatched_det)}")
            else:
                matched_layout = {}
                unmatched_det = []
            
            # ========== 处理 layout 中已匹配的区域 ==========
            for region_idx, region in enumerate(layout_res):
                print('[debug]',region['label'])
                res = ''
                layout_key = tuple(region['bbox']) if region['bbox'] is not None else None
                
                if region['bbox'] is not None:
                    x1, y1, x2, y2 = region['bbox']
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    roi_img = ori_im[y1:y2, x1:x2, :]
                else:
                    x1, y1, x2, y2 = 0, 0, w, h
                    roi_img = ori_im
                
                if region['label'] == 'table':
                    # 表格处理：尝试复用全图的 det/rec 结果
                    res = None
                    use_reused_ocr = False
                    
                    if layout_key is not None and layout_key in matched_layout and len(matched_layout[layout_key]) > 0:
                        try:
                            # 有匹配的 det，尝试用这些结果进行表格结构匹配
                            det_indices = matched_layout[layout_key]
                            print('[debug] det_indices ',len(det_indices ))
                            # 获取 table 的坐标用于坐标映射
                            table_x1, table_y1, table_x2, table_y2 = region['bbox']
                            
                            # 将全图坐标的 det boxes 和 rec 结果映射到 roi 坐标系
                            roi_det_boxes = []
                            roi_rec_res = []
                            
                            for det_idx in det_indices:
                                box = all_det_boxes[det_idx]  # [x1, y1, x2, y2]
                                rec_tuple = all_rec_results[det_idx]  # (rec_str, rec_conf)
                                
                                # 转换坐标到 roi 坐标系 - 保持 [x1,y1,x2,y2] 格式
                                roi_x1 = box[0] - table_x1
                                roi_y1 = box[1] - table_y1
                                roi_x2 = box[2] - table_x1
                                roi_y2 = box[3] - table_y1
                                
                                roi_box = [roi_x1, roi_y1, roi_x2, roi_y2]
                                
                                roi_det_boxes.append(roi_box)
                                roi_rec_res.append(rec_tuple)  # 保持 (text, conf) 格式
                            
                            # 调用 table_system 的结构识别和匹配
                            if self.table_system is not None and len(roi_det_boxes) > 0:
                                # logger.info(f'[Table复用] 使用 {len(roi_det_boxes)} 个已识别的det结果')
                                
                                tic = time.time()
                                structure_res, elapse = self.table_system._structure(roi_img)
                                time_dict['table'] += elapse
                                
                                # ===== DEBUG: 保存 structure_res =====
                                try:
                                    debug_dir = os.path.join('./tmp', 'table_debug')
                                    os.makedirs(debug_dir, exist_ok=True)
                                    sr_tokens, sr_bboxes = structure_res
                                    structure_res_data = {
                                        'tokens': sr_tokens if isinstance(sr_tokens, list) else (sr_tokens.tolist() if hasattr(sr_tokens, 'tolist') else str(sr_tokens)),
                                        'cell_bboxes': [cb.tolist() if hasattr(cb, 'tolist') else (cb if isinstance(cb, list) else list(cb)) for cb in sr_bboxes]
                                    }
                                    sr_save_path = os.path.join(debug_dir, f'structure_res_{img_idx}_region{region_idx}.json')
                                    with open(sr_save_path, 'w', encoding='utf-8') as _f:
                                        json.dump(structure_res_data, _f, ensure_ascii=False, indent=2)
                                    logger.info(f'[TableDebug] 已保存 structure_res: {sr_save_path}')
                                except Exception as _se:
                                    logger.warning(f'[TableDebug] 保存 structure_res 失败: {_se}')
                                # ===== DEBUG END =====
                                
                                # ===== DEBUG: 保存 structure_res cell bbox 和 roi_det_boxes 可视化 =====
                                try:
                                    debug_img = roi_img.copy()
                                    # 画 structure_res 的 cell bbox（蓝色）
                                    _, cell_bboxes = structure_res
                                    for cb in cell_bboxes:
                                        if len(cb) == 8:
                                            pts = np.array(cb, dtype=np.int32).reshape(4, 2)
                                            cv2.polylines(debug_img, [pts], True, (255, 0, 0), 2)
                                        elif len(cb) == 4:
                                            cx1, cy1, cx2, cy2 = int(cb[0]), int(cb[1]), int(cb[2]), int(cb[3])
                                            cv2.rectangle(debug_img, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)
                                    # 画 roi_det_boxes（红色）+ 文字
                                    for bi, (rb, rtxt) in enumerate(zip(roi_det_boxes, roi_rec_res)):
                                        rx1, ry1, rx2, ry2 = int(rb[0]), int(rb[1]), int(rb[2]), int(rb[3])
                                        cv2.rectangle(debug_img, (rx1, ry1), (rx2, ry2), (0, 0, 255), 1)
                                        cv2.putText(debug_img, str(rtxt[0])[:10], (rx1, max(ry1-2, 0)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                                    debug_dir = os.path.join('./tmp', 'table_debug')
                                    os.makedirs(debug_dir, exist_ok=True)
                                    debug_path = os.path.join(debug_dir, f'table_{img_idx}_region{region_idx}.jpg')
                                    debug_path2 = os.path.join(debug_dir, f'table-orri_{img_idx}_region{region_idx}.jpg')
                                    cv2.imwrite(debug_path, debug_img)
                                    cv2.imwrite(debug_path2, roi_img)
                                    logger.info(f'[TableDebug] 已保存可视化: {debug_path} | cell_bbox数={len(cell_bboxes)}, roi_det数={len(roi_det_boxes)}')
                                except Exception as _de:
                                    logger.warning(f'[TableDebug] 可视化失败: {_de}')
                                # ===== DEBUG END =====
                                
                                # 转换为 numpy 数组
                                roi_det_boxes_np = np.array(roi_det_boxes)
                                
                                # ===== DEBUG: 保存 roi_det_boxes_np =====
                                try:
                                    _det_save_path = os.path.join(os.path.join('./tmp', 'table_debug'), f'roi_det_boxes_{img_idx}_region{region_idx}.json')
                                    _det_data = {
                                        'roi_det_boxes': roi_det_boxes_np.tolist(),
                                        'roi_rec_res': [(str(t), float(c)) for t, c in roi_rec_res]
                                    }
                                    with open(_det_save_path, 'w', encoding='utf-8') as _f:
                                        json.dump(_det_data, _f, ensure_ascii=False, indent=2)
                                    logger.info(f'[TableDebug] 已保存 roi_det_boxes: {_det_save_path}')
                                except Exception as _de2:
                                    logger.warning(f'[TableDebug] 保存 roi_det_boxes 失败: {_de2}')
                                # ===== DEBUG END =====
                                
                                # 使用我们的 det/rec 结果进行匹配
                                tic = time.time()
                                pred_html = self.table_system.match(structure_res, roi_det_boxes_np, roi_rec_res)
                                toc = time.time()
                                time_dict['table_match'] += toc - tic
                                
                                # ===== DEBUG: 保存 pred_html =====
                                try:
                                    _html_save_path = os.path.join(os.path.join('./tmp', 'table_debug'), f'pred_html_{img_idx}_region{region_idx}.html')
                                    with open(_html_save_path, 'w', encoding='utf-8') as _f:
                                        _f.write(pred_html if pred_html else '')
                                    logger.info(f'[TableDebug] 已保存 pred_html: {_html_save_path}')
                                except Exception as _he:
                                    logger.warning(f'[TableDebug] 保存 pred_html 失败: {_he}')
                                # ===== DEBUG END =====
                                
                                # 检查 HTML 中是否有实际文字内容
                                import re as _re
                                html_text_content = _re.sub(r'<[^>]+>', '', pred_html).strip()
                                # 去掉 colspan/rowspan 属性噪音（结构 token 漏到文本里的情况）
                                real_text = _re.sub(r'(colspan|rowspan)="\d+"', '', html_text_content).strip()
                                real_text = _re.sub(r'\s+', '', real_text)
                                logger.info(f'[Table复用] html长度={len(pred_html)}, 实际文字长度={len(real_text)}, 预览={repr(real_text[:80])}')
                                
                                empty_html_patterns = ('', '<html><body><table></table></body></html>')
                                if not pred_html or pred_html.strip() in empty_html_patterns or not real_text:
                                    logger.warning('[Table复用] 单元格内容为空（OCR未匹配到结构），回退到完整table_system')
                                    res = None
                                else:
                                    res = {'html': pred_html}
                                    use_reused_ocr = True
                                    # 标记这些 det 为已使用
                                    for det_idx in det_indices:
                                        used_det_indices.add(det_idx)
                        except Exception as e:
                            logger.warning(f'[Table复用失败] {e}, 使用完整table_system')
                            res = None
                    
                    # 如果复用失败或没有匹配的det，调用完整的 table_system
                    if res is None and self.table_system is not None:
                        logger.info('[Table] 使用完整table_system识别')
                        res, table_time_dict = self.table_system(
                            roi_img, return_ocr_result_in_table)
                        time_dict['table'] += table_time_dict['table']
                        time_dict['table_match'] += table_time_dict['match']
                        time_dict['det'] += table_time_dict['det']
                        time_dict['rec'] += table_time_dict['rec']
                        
                        # 如果使用了完整table_system，也要标记匹配的det为已使用
                        if layout_key is not None and layout_key in matched_layout:
                            for det_idx in matched_layout[layout_key]:
                                used_det_indices.add(det_idx)
                    
                    if res is None:
                        res = []
                else:
                    # 文字处理：使用匹配的 det 结果
                    res = []
                    if layout_key is not None and layout_key in matched_layout:
                        det_indices = matched_layout[layout_key]
                        style_token = [
                            '<strike>', '<strike>', '<sup>', '</sub>', '<b>',
                            '</b>', '<sub>', '</sup>', '<overline>',
                            '</overline>', '<underline>', '</underline>', '<i>',
                            '</i>'
                        ]
                        
                        for det_idx in det_indices:
                            box = all_det_boxes[det_idx]
                            rec_str, rec_conf = all_rec_results[det_idx]
                            
                            # 移除样式标签
                            for token in style_token:
                                if token in rec_str:
                                    rec_str = rec_str.replace(token, '')
                            
                            res.append({
                                'text': rec_str,
                                'confidence': float(rec_conf),
                                'text_region': box
                            })
                
                res_list.append({
                    'type': region['label'].lower(),
                    'bbox': [x1, y1, x2, y2],
                    'img': roi_img,
                    'res': res,
                    'img_idx': img_idx
                })
            
            # ========== 处理未匹配的 det（属于没被识别到的区域） ==========
            # 过滤掉已被 table 系统处理过的 det
            unmatched_det_filtered = [
                (idx, box) for idx, box in unmatched_det 
                if idx not in used_det_indices
            ]
            
            if len(unmatched_det_filtered) > 0:
                logger.info(f"Adding {len(unmatched_det_filtered)} unmatched text regions as separate text areas")
                
                style_token = [
                    '<strike>', '<strike>', '<sup>', '</sub>', '<b>',
                    '</b>', '<sub>', '</sup>', '<overline>',
                    '</overline>', '<underline>', '</underline>', '<i>',
                    '</i>'
                ]
                
                for det_idx, det_box in unmatched_det_filtered:
                    x1, y1, x2, y2 = det_box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 确保坐标在图像范围内
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    roi_img = ori_im[y1:y2, x1:x2, :]
                    
                    rec_str, rec_conf = all_rec_results[det_idx]
                    
                    # 移除样式标签
                    for token in style_token:
                        if token in rec_str:
                            rec_str = rec_str.replace(token, '')
                    
                    res_list.append({
                        'type': 'text',
                        'bbox': [x1, y1, x2, y2],
                        'img': roi_img,
                        'res': [{
                            'text': rec_str,
                            'confidence': float(rec_conf),
                            'text_region': det_box
                        }],
                        'img_idx': img_idx
                    })
            
            end = time.time()
            time_dict['all'] = end - start
            return res_list, time_dict
        elif self.mode == 'kie':
            re_res, elapse = self.kie_predictor(img)
            time_dict['kie'] = elapse
            time_dict['all'] = elapse
            return re_res[0], time_dict
        return None, None


def save_structure_res(res, save_folder, img_name, img_idx=0):
    excel_save_folder = os.path.join(save_folder, img_name)
    os.makedirs(excel_save_folder, exist_ok=True)
    res_cp = deepcopy(res)
    
    # 自定义 JSON 编码器，支持 numpy 类型
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return super().default(obj)
    
    # save res
    with open(
            os.path.join(excel_save_folder, 'res_{}.txt'.format(img_idx)),
            'w',
            encoding='utf8') as f:
        for region in res_cp:
            roi_img = region.pop('img')
            
            # 转换 res 中的 numpy 类型
            if isinstance(region['res'], list):
                for item in region['res']:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if isinstance(value, (np.float32, np.float64)):
                                item[key] = float(value)
                            elif isinstance(value, (np.int32, np.int64)):
                                item[key] = int(value)
                            elif isinstance(value, np.ndarray):
                                item[key] = value.tolist()
            
            # 转换 bbox 中的值
            if isinstance(region['bbox'], list):
                region['bbox'] = [int(x) if isinstance(x, (float, np.floating, np.integer)) else x for x in region['bbox']]
            
            f.write('{}\n'.format(json.dumps(region, cls=NumpyEncoder, ensure_ascii=False)))

            if region['type'].lower() == 'table' and isinstance(region['res'], dict) and 'html' in region['res']:
                excel_path = os.path.join(
                    excel_save_folder,
                    '{}_{}.xlsx'.format(region['bbox'], img_idx))
                to_excel(region['res']['html'], excel_path)
            elif region['type'].lower() == 'figure' or region['type'].lower() == 'reference':
                img_path = os.path.join(
                    excel_save_folder,
                    '{}_{}.jpg'.format(region['bbox'], img_idx))
                save_image_with_chinese_path(roi_img, img_path)


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list
    image_file_list = image_file_list[args.process_id::args.total_process_num]

    if not args.use_pdf2docx_api:
        structure_sys = StructureSystem(args)
        save_folder = os.path.join(args.output, structure_sys.mode)
        os.makedirs(save_folder, exist_ok=True)
    img_num = len(image_file_list)

    for i, image_file in enumerate(image_file_list):
        logger.info("[{}/{}] {}".format(i, img_num, image_file))
        img, flag_gif, flag_pdf = check_and_read(image_file)
        img_name = os.path.basename(image_file).split('.')[0]

        if args.recovery and args.use_pdf2docx_api and flag_pdf:
            from pdf2docx.converter import Converter
            os.makedirs(args.output, exist_ok=True)
            docx_file = os.path.join(args.output,
                                     '{}_api.docx'.format(img_name))
            cv = Converter(image_file)
            cv.convert(docx_file)
            cv.close()
            logger.info('docx save to {}'.format(docx_file))
            continue

        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)

        if not flag_pdf:
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            imgs = img

        all_res = []
        for index, img in enumerate(imgs):
            res, time_dict = structure_sys(img, img_idx=index)
            img_save_path = os.path.join(save_folder, img_name,
                                         'show_{}.jpg'.format(index))
            os.makedirs(os.path.join(save_folder, img_name), exist_ok=True)
            if structure_sys.mode == 'structure' and res != []:
                draw_img = draw_structure_result(img, res, args.vis_font_path)
                save_structure_res(res, save_folder, img_name, index)
            elif structure_sys.mode == 'kie':
                if structure_sys.kie_predictor.predictor is not None:
                    draw_img = draw_re_results(
                        img, res, font_path=args.vis_font_path)
                else:
                    draw_img = draw_ser_results(
                        img, res, font_path=args.vis_font_path)

                with open(
                        os.path.join(save_folder, img_name,
                                     'res_{}_kie.txt'.format(index)),
                        'w',
                        encoding='utf8') as f:
                    res_str = '{}\t{}\n'.format(
                        image_file,
                        json.dumps(
                            {
                                "ocr_info": res
                            }, ensure_ascii=False))
                    f.write(res_str)
            if res != []:
                cv2.imwrite(img_save_path, draw_img)
                logger.info('result save to {}'.format(img_save_path))
            if args.recovery and res != []:
                from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx
                h, w, _ = img.shape
                res = sorted_layout_boxes(res, w)
                all_res += res

        if args.recovery and all_res != []:
            try:
                convert_info_docx(img, all_res, save_folder, img_name)
            except Exception as ex:
                logger.error("error in layout recovery image:{}, err msg: {}".
                             format(image_file, ex))
                continue
        logger.info("Predict time : {:.3f}s".format(time_dict['all']))


if __name__ == "__main__":
    args = parse_args()
    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        main(args)
