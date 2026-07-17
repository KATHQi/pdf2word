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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import time

import tools.infer.utility as utility
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppstructure.utility import parse_args

logger = get_logger()


def preprocess(img, target_size=[800, 608]):
    """PaddleDetection style preprocessing"""
    # Resize
    img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    
    # NormalizeImage
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    img = (img - mean) / std
    
    # Permute (HWC -> CHW)
    img = img.transpose((2, 0, 1))
    
    return img


class LayoutPredictor(object):
    def __init__(self, args):
        self.score_threshold = args.layout_score_threshold
        
        # Load layout labels
        with open(args.layout_dict_path, 'r', encoding='utf-8') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'layout', logger)
        
        # Get all input tensors
        input_names = self.predictor.get_input_names()
        print(f"Model input names: {input_names}")
        self.input_handles = {}
        for name in input_names:
            self.input_handles[name] = self.predictor.get_input_handle(name)

    def __call__(self, img):
        ori_im = img.copy()
        
        # Use manual preprocessing (PaddleDetection style)
        preprocessed_img = preprocess(img, target_size=[800, 608])
        
        if preprocessed_img is None:
            return None, 0

        preprocessed_img = np.expand_dims(preprocessed_img, axis=0).astype(np.float32)
        
        # Calculate scale_factor (since we don't keep ratio, scale_factor is [1.0, 1.0])
        scale_factor = np.array([[1.0, 1.0]], dtype=np.float32)

        starttime = time.time()

        # Set all inputs
        self.input_handles['image'].copy_from_cpu(preprocessed_img)
        self.input_handles['scale_factor'].copy_from_cpu(scale_factor)
        
        self.predictor.run()

        # Get outputs - the model outputs NMS results directly
        # Output 0: detection boxes [N, 6] where each row is [class_id, score, x1, y1, x2, y2]
        # Output 1: number of boxes [1]
        np_boxes = self.output_tensors[0].copy_to_cpu()
        np_boxes_num = self.output_tensors[1].copy_to_cpu()
        
        # Process results: filter by score threshold and format output
        results = []
        for box in np_boxes:
            class_id = int(box[0])
            score = float(box[1])
            if score < self.score_threshold:
                continue
            
            # Get label name
            label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
            
            # bbox coordinates [x1, y1, x2, y2]
            bbox = box[2:6].tolist()
            
            results.append({
                'bbox': bbox,
                'label': label,
                'score': score
            })
        
        elapse = time.time() - starttime
        return results, elapse


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    layout_predictor = LayoutPredictor(args)
    count = 0
    total_time = 0

    os.makedirs(args.output, exist_ok=True)
    
    # Save results to bbox.json
    all_results = []
    
    for image_file in image_file_list:
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue

        layout_res, elapse = layout_predictor(img)

        logger.info("result: {}".format(layout_res))
        
        # Save result
        result_dict = {
            'image_path': image_file,
            'layout_result': layout_res
        }
        all_results.append(result_dict)

        if count > 0:
            total_time += elapse
        count += 1
        logger.info("Predict time of {}: {}".format(image_file, elapse))
    
    # Save to bbox.json
    import json
    bbox_file = os.path.join(args.output, 'bbox.json')
    with open(bbox_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info("Results saved to {}".format(bbox_file))
    
    if count > 1:
        logger.info("Avg Time: {}".format(total_time / (count - 1)))


if __name__ == "__main__":
    main(parse_args())
