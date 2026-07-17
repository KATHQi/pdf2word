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
from paddle.inference import Config
from paddle.inference import create_predictor

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


def create_layout_predictor_config(args):
    """Create predictor using Paddle Inference API"""
    model_dir = args.layout_model_dir
    
    # Try to find model files
    infer_model = os.path.join(model_dir, 'model.pdmodel')
    infer_params = os.path.join(model_dir, 'model.pdiparams')
    
    if not os.path.exists(infer_model):
        infer_model = os.path.join(model_dir, 'inference.pdmodel')
        infer_params = os.path.join(model_dir, 'inference.pdiparams')
        if not os.path.exists(infer_model):
            raise ValueError(
                f"Cannot find any inference model in dir: {model_dir}")
    
    logger.info(f"Loading layout model from: {model_dir}")
    logger.info(f"  Model file: {infer_model}")
    logger.info(f"  Params file: {infer_params}")
    
    # Check file sizes and MD5
    import hashlib
    def get_file_md5(filepath):
        if not os.path.exists(filepath):
            return "File not found"
        md5_hash = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    
    model_size = os.path.getsize(infer_model) / (1024*1024)  # MB
    params_size = os.path.getsize(infer_params) / (1024*1024)  # MB
    model_md5 = get_file_md5(infer_model)
    params_md5 = get_file_md5(infer_params)
    
    logger.info(f"  Model file size: {model_size:.2f} MB")
    logger.info(f"  Params file size: {params_size:.2f} MB")
    logger.info(f"  Model MD5: {model_md5}")
    logger.info(f"  Params MD5: {params_md5}")
    
    # Create config
    config = Config(infer_model, infer_params)
    
    # Print Paddle version
    import paddle
    logger.info(f"Paddle version: {paddle.__version__}")
    try:
        from paddle import inference
        logger.info(f"Paddle Inference module: {inference.__file__}")
    except:
        logger.warning("Cannot get Paddle Inference module info")
    
    # Set device
    use_gpu = args.use_gpu if hasattr(args, 'use_gpu') else True
    logger.info(f"Using GPU: {use_gpu}")
    
    if use_gpu:
        config.enable_use_gpu(200, 0)
        config.switch_ir_optim(True)
        logger.info("GPU enabled with 200MB memory, IR optimization ON")
    else:
        config.disable_gpu()
        cpu_threads = args.cpu_threads if hasattr(args, 'cpu_threads') else 10
        config.set_cpu_math_library_num_threads(cpu_threads)
        logger.info(f"CPU mode with {cpu_threads} threads")
        
        enable_mkldnn = args.enable_mkldnn if hasattr(args, 'enable_mkldnn') else False
        if enable_mkldnn:
            try:
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
                logger.info("MKLDNN enabled")
            except Exception as e:
                logger.warning(
                    f"The current environment does not support `mkldnn`: {e}")
    
    # Disable print log when predict
    config.disable_glog_info()
    # Enable shared memory
    config.enable_memory_optim()
    # Disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    
    logger.info("Config settings:")
    logger.info(f"  - glog_info disabled: True")
    logger.info(f"  - memory_optim enabled: True")
    logger.info(f"  - use_feed_fetch_ops: False")
    
    # Create predictor
    predictor = create_predictor(config)
    
    # Get and print input/output info
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    
    logger.info(f"Predictor created successfully")
    logger.info(f"  Input names: {input_names}")
    logger.info(f"  Output names: {output_names}")
    logger.info(f"  Number of inputs: {len(input_names)}")
    logger.info(f"  Number of outputs: {len(output_names)}")
    
    # Get detailed input shape info
    for name in input_names:
        input_handle = predictor.get_input_handle(name)
        try:
            shape = input_handle.shape()
            logger.info(f"  Input '{name}' shape: {shape}")
        except Exception as e:
            logger.warning(f"  Cannot get shape for input '{name}': {e}")
    
    return predictor


class LayoutPredictor(object):
    def __init__(self, args):
        self.score_threshold = args.layout_score_threshold
        
        # Load layout labels
        with open(args.layout_dict_path, 'r', encoding='utf-8') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
        # Create predictor using Paddle Inference API
        self.predictor = create_layout_predictor_config(args)
        
        # Get input/output names
        input_names = self.predictor.get_input_names()
        output_names = self.predictor.get_output_names()
        
        logger.info(f"Model input names: {input_names}")
        logger.info(f"Model output names: {output_names}")
        
        # Get input handles
        self.input_handles = {}
        for name in input_names:
            self.input_handles[name] = self.predictor.get_input_handle(name)
        
        # Get output handles
        self.output_handles = []
        for name in output_names:
            self.output_handles.append(self.predictor.get_output_handle(name))

    def __call__(self, img):
        ori_im = img.copy()
        
        # Get original image size
        ori_h, ori_w = img.shape[:2]
        target_size = [800, 608]
        
        # Use manual preprocessing (PaddleDetection style)
        preprocessed_img = preprocess(img, target_size=target_size)
        
        if preprocessed_img is None:
            return None, 0

        preprocessed_img = np.expand_dims(preprocessed_img, axis=0).astype(np.float32)
        
        # Calculate scale factors for coordinate restoration
        scale_w = ori_w / target_size[1]  # original_width / resized_width
        scale_h = ori_h / target_size[0]  # original_height / resized_height
        
        starttime = time.time()

        # Set inputs - check if model needs scale_factor
        self.input_handles['image'].copy_from_cpu(preprocessed_img)
        
        # Only set scale_factor if the model requires it
        if 'scale_factor' in self.input_handles:
            # This scale_factor is for the model's internal use
            scale_factor = np.array([[1.0, 1.0]], dtype=np.float32)
            self.input_handles['scale_factor'].copy_from_cpu(scale_factor)
        
        # Run inference
        self.predictor.run()

        # Get outputs - the model outputs NMS results directly
        # Output 0: detection boxes [N, 6] where each row is [class_id, score, x1, y1, x2, y2]
        # Output 1: number of boxes [1]
        np_boxes = self.output_handles[0].copy_to_cpu()
        np_boxes_num = self.output_handles[1].copy_to_cpu() if len(self.output_handles) > 1 else None
        
        # Process results: filter by score threshold and format output
        results = []
        for box in np_boxes:
            class_id = int(box[0])
            score = float(box[1])
            if score < self.score_threshold:
                continue
            
            # Get label name
            label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
            
            # bbox coordinates [x1, y1, x2, y2] - in resized image coordinates
            # Need to scale back to original image coordinates
            x1, y1, x2, y2 = box[2:6]
            
            # Restore coordinates to original image size
            x1_ori = x1 * scale_w
            y1_ori = y1 * scale_h
            x2_ori = x2 * scale_w
            y2_ori = y2 * scale_h
            
            bbox = [x1_ori, y1_ori, x2_ori, y2_ori]
            
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
    # import json
    # bbox_file = os.path.join(args.output, 'bbox.json')
    # with open(bbox_file, 'w', encoding='utf-8') as f:
    #     json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # logger.info("Results saved to {}".format(bbox_file))
    
    if count > 1:
        logger.info("Avg Time: {}".format(total_time / (count - 1)))


if __name__ == "__main__":
    main(parse_args())
