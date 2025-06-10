#!/usr/bin/env python3
"""
Meter Reading Extraction Web Application
A complete pipeline for extracting readings from industrial meter displays

Pipeline:
1. Upload image -> YOLO detection -> Crop meter region
2. Crop -> DeepLabV3+ segmentation -> Generate masks
3. Masks -> Reading extraction -> Final result

Author: AI Assistant
Date: 2024
"""

import gradio as gr
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import os
import sys
from typing import Tuple, Optional, List, Dict, Any

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))
from scripts.extract_meter_reading import MeterReader

# Import ONNX runtime for segmentation
import onnxruntime as ort


class MeterDetector:
    """YOLO-based meter detection"""
    
    def __init__(self, model_path: str):
        """Initialize detector with trained model"""
        self.model = YOLO(model_path)
        
    def detect_meters(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect meters in image
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold
            
        Returns:
            List of detection results with bounding boxes
        """
        results = self.model(image, conf=conf_threshold)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': 'meter'
                    })
        
        return detections
    
    def crop_meter(self, image: np.ndarray, bbox: List[int], padding: int = 20) -> np.ndarray:
        """
        Crop meter region from image
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Padding around bounding box
            
        Returns:
            Cropped meter image
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        return image[y1:y2, x1:x2]


class MeterSegmentor:
    """ONNX-based meter segmentation"""
    
    def __init__(self, model_path: str, device: str = 'cpu', post_process_config: Dict = None):
        """Initialize segmentor with ONNX model"""
        self.device = device
        self.session = self._load_onnx_model(model_path)
        
        # 后处理配置
        self.post_process_config = post_process_config or {
            'remove_noise': True,           # 是否去除噪声
            'keep_largest_component': False, # 是否只保留最大连通域
            'pointer_erosion': 1,           # 指针腐蚀迭代次数
            'scale_erosion': 3,             # 刻度腐蚀迭代次数
            'fill_holes': False,             # 是否填充小洞
            'connect_scale_lines': False     # 是否连接断裂的刻度线
        }
        
    def _load_onnx_model(self, model_path: str):
        """Load ONNX segmentation model"""
        # Check if ONNX file exists
        onnx_path = model_path.replace('.pth', '.onnx')
        if not os.path.exists(onnx_path):
            # Try the exported directory
            onnx_path = "models/segmentation/segmentation_model.onnx"
        
        if os.path.exists(onnx_path):
            try:
                # Configure ONNX Runtime providers
                providers = []
                if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.append('CUDAExecutionProvider')
                elif self.device == 'mps':
                    # ONNX Runtime doesn't support MPS directly, use CPU
                    providers.append('CPUExecutionProvider')
                else:
                    providers.append('CPUExecutionProvider')
                
                # Create inference session
                session = ort.InferenceSession(onnx_path, providers=providers)
                print(f"✅ Loaded ONNX model from: {onnx_path}")
                print(f"📊 Input shape: {session.get_inputs()[0].shape}")
                print(f"📊 Output shape: {session.get_outputs()[0].shape}")
                print(f"🔧 Providers: {session.get_providers()}")
                
                return session
                
            except Exception as e:
                print(f"❌ Error loading ONNX model: {e}")
                return None
        else:
            print(f"⚠️  ONNX model not found at: {onnx_path}")
            return None
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Preprocess image for ONNX inference"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image_resized = cv2.resize(image_rgb, target_size)
        
        # Normalize to [0, 1] and then apply ImageNet normalization
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_normalized - mean) / std
        
        # Convert to NCHW format
        image_tensor = np.transpose(image_normalized, (2, 0, 1))
        image_batch = np.expand_dims(image_tensor, axis=0)
        
        return image_batch.astype(np.float32)
    
    def post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        对分割结果进行后处理，去除离群点和优化边界
        
        Args:
            mask: 原始分割掩码
            
        Returns:
            处理后的分割掩码
        """
        if not any(self.post_process_config.values()):
            return mask  # 如果所有后处理都关闭，直接返回原掩码
            
        processed_mask = mask.copy()
        config = self.post_process_config
        
        # 为每个类别分别处理
        for class_id in [1, 2]:  # 指针和刻度
            if class_id not in mask:
                continue
                
            # 提取当前类别的掩码
            class_mask = (mask == class_id).astype(np.uint8)
            
            if np.sum(class_mask) == 0:
                continue
            
            # 1. 去除小的离群点 - 开运算（先腐蚀后膨胀）
            if config['remove_noise']:
                kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel_small)
            
            # 2. 连通域分析，保留最大的连通区域
            if config['keep_largest_component']:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
                
                if num_labels > 1:  # 有连通域（除了背景）
                    # 找到最大的连通域（排除背景标签0）
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    class_mask = (labels == largest_label).astype(np.uint8)
            
            # 3. 根据类别进行特定处理
            if class_id == 1:  # 指针
                # 指针需要细化，使用较小的腐蚀核
                if config['pointer_erosion'] > 0:
                    kernel_pointer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                    class_mask = cv2.erode(class_mask, kernel_pointer, iterations=config['pointer_erosion'])
                
            elif class_id == 2:  # 刻度
                # 刻度需要更多腐蚀来收缩边界，防止外移
                if config['scale_erosion'] > 0:
                    kernel_scale = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    class_mask = cv2.erode(class_mask, kernel_scale, iterations=config['scale_erosion'])
                
                # 对刻度进行额外的形态学闭运算，连接断裂的刻度线
                if config['connect_scale_lines']:
                    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                    class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel_close)
            
            # 4. 填充小洞
            if config['fill_holes']:
                kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel_fill)
            
            # 将处理后的类别掩码放回原掩码
            processed_mask[mask == class_id] = 0  # 先清除原来的
            processed_mask[class_mask == 1] = class_id  # 再放入处理后的
        
        return processed_mask
    
    def segment_meter(self, image: np.ndarray) -> np.ndarray:
        """
        Segment meter image into classes using ONNX
        
        Args:
            image: Input meter image (BGR format)
            
        Returns:
            Segmentation mask with class labels
        """
        if self.session is None:
            print("⚠️  No ONNX model loaded, returning dummy mask")
            # Return a dummy mask with some basic segmentation
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            # Create a simple circular mask as placeholder
            center = (w//2, h//2)
            radius = min(w, h) // 4
            cv2.circle(mask, center, radius, 1, -1)  # pointer region
            cv2.circle(mask, center, radius + 20, 2, 10)  # scale region
            return mask
        
        original_size = image.shape[:2]
        
        # Preprocess
        input_data = self.preprocess_image(image)
        
        # Get input name
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        try:
            outputs = self.session.run(None, {input_name: input_data})
            output = outputs[0]  # First output
            
            # Get predictions (argmax along channel dimension)
            predictions = np.argmax(output, axis=1).squeeze(0).astype(np.uint8)
            
            # Resize back to original size
            mask = cv2.resize(predictions, 
                             (original_size[1], original_size[0]), 
                             interpolation=cv2.INTER_NEAREST)
            
            # 后处理：去除离群点和优化边界
            mask = self.post_process_mask(mask)
            
            return mask
            
        except Exception as e:
            print(f"❌ ONNX inference error: {e}")
            # Return dummy mask on error
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            return mask


class DigitDetector:
    """YOLO-based digit detection for LCD displays"""
    
    def __init__(self, model_path: str):
        """Initialize digit detector with trained model"""
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"✅ Loaded digit detection model: {model_path}")
        else:
            print(f"⚠️ Digit model not found at {model_path}, using base YOLO")
            self.model = YOLO("yolov10n.pt")
        
        # Define class names for digits
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'point']
        
    def detect_digits(self, image: np.ndarray, conf_threshold: float = 0.3) -> List[Dict]:
        """
        Detect digits in LCD display image
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold for detection
            
        Returns:
            List of detection results with digit information
        """
        results = self.model(image, conf=conf_threshold)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # Get class name
                    if hasattr(r, 'names') and cls in r.names:
                        class_name = r.names[cls]
                    elif cls < len(self.class_names):
                        class_name = self.class_names[cls]
                    else:
                        class_name = str(cls)
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': cls,
                        'class': class_name,
                        'center_x': (x1 + x2) / 2,
                        'center_y': (y1 + y2) / 2
                    })
        
        return detections
    
    def filter_duplicate_detections(self, detections: List[Dict], 
                                  overlap_threshold: float = 0.7,
                                  distance_threshold: float = 30) -> List[Dict]:
        """
        Filter out duplicate detections that are too close to each other
        
        Args:
            detections: List of detection results
            overlap_threshold: IoU threshold for considering detections as duplicates
            distance_threshold: Distance threshold in pixels
            
        Returns:
            Filtered detections list
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        
        for det in detections:
            is_duplicate = False
            
            for existing in filtered:
                # 对于相同的数字类别，检查是否过于接近
                if det['class'] == existing['class']:
                    # 小数点单独处理，允许多个小数点候选，后续会合并
                    if det['class'] == 'point':
                        # 小数点的距离阈值更严格
                        dx = det['center_x'] - existing['center_x']
                        dy = det['center_y'] - existing['center_y']
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        # 小数点距离很近才认为是重复
                        if distance < distance_threshold * 0.5:  # 更严格的距离要求
                            is_duplicate = True
                            break
                    else:
                        # 对于数字，检查距离和重叠
                        dx = det['center_x'] - existing['center_x']
                        dy = det['center_y'] - existing['center_y']
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        # Calculate IoU
                        iou = self._calculate_iou(det['bbox'], existing['bbox'])
                        
                        # 如果距离很近或重叠很大，认为是重复
                        if distance < distance_threshold or iou > overlap_threshold:
                            is_duplicate = True
                            break
                
                # 额外检查：如果是不同数字但位置几乎重叠，可能是误识别
                elif det['class'] != 'point' and existing['class'] != 'point':
                    iou = self._calculate_iou(det['bbox'], existing['bbox'])
                    # 如果两个不同数字的重叠度很高，保留置信度更高的
                    if iou > 0.8:  # 高重叠阈值
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(det)
        
        return filtered
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def extract_reading(self, detections: List[Dict]) -> str:
        """
        Extract the complete reading from detected digits with intelligent grouping
        
        Args:
            detections: List of filtered detections
            
        Returns:
            String representation of the reading
        """
        if not detections:
            return ""
        
        # 按x坐标排序（从左到右）
        sorted_detections = sorted(detections, key=lambda x: x['center_x'])
        
        # 进一步处理数字序列，处理重复和分组
        final_reading = self._process_digit_sequence(sorted_detections)
        
        return final_reading
    
    def _process_digit_sequence(self, sorted_detections: List[Dict]) -> str:
        """
        处理数字序列，智能分组并构建最终读数
        
        Args:
            sorted_detections: 按x坐标排序的检测结果
            
        Returns:
            处理后的读数字符串
        """
        if not sorted_detections:
            return ""
        
        # 分析数字的位置分布，确定数字组
        digit_groups = self._group_digits_by_position(sorted_detections)
        
        # 为每个组构建读数
        readings = []
        for group in digit_groups:
            group_reading = self._construct_group_reading(group)
            if group_reading:
                readings.append(group_reading)
        
        # 合并所有读数（用空格分隔多个独立的数字）
        if len(readings) == 1:
            return readings[0]
        else:
            return " ".join(readings)
    
    def _group_digits_by_position(self, detections: List[Dict]) -> List[List[Dict]]:
        """
        根据位置将数字分组，识别不同的数字显示区域
        
        Args:
            detections: 排序后的检测结果
            
        Returns:
            数字组列表
        """
        if not detections:
            return []
        
        groups = []
        current_group = [detections[0]]
        
        # 计算平均字符宽度和间距作为分组依据
        widths = [det['bbox'][2] - det['bbox'][0] for det in detections]
        avg_width = np.mean(widths) if widths else 50
        
        for i in range(1, len(detections)):
            prev_det = detections[i-1]
            curr_det = detections[i]
            
            # 计算两个检测框之间的间距
            prev_right = prev_det['bbox'][2]
            curr_left = curr_det['bbox'][0]
            gap = curr_left - prev_right
            
            # 如果间距大于平均宽度的1.5倍，认为是新的数字组
            # 或者如果是小数点，间距更小也可能是新组
            gap_threshold = avg_width * 1.5
            if prev_det['class'] == 'point' or curr_det['class'] == 'point':
                gap_threshold = avg_width * 0.8  # 小数点附近的阈值更小
            
            if gap > gap_threshold:
                groups.append(current_group)
                current_group = [curr_det]
            else:
                current_group.append(curr_det)
        
        groups.append(current_group)
        return groups
    
    def _construct_group_reading(self, group: List[Dict]) -> str:
        """
        为单个数字组构建读数，处理重复数字
        
        Args:
            group: 单个组的检测结果
            
        Returns:
            该组的读数字符串
        """
        if not group:
            return ""
        
        # 再次按x坐标精确排序
        group = sorted(group, key=lambda x: x['center_x'])
        
        # 处理位置相近的重复数字
        filtered_group = self._remove_positional_duplicates(group)
        
        # 构建读数字符串
        reading_parts = []
        for det in filtered_group:
            if det['class'] == 'point':
                reading_parts.append(".")
            else:
                reading_parts.append(str(det['class']))
        
        reading = "".join(reading_parts)
        
        # 清理读数
        reading = self._validate_reading(reading)
        
        return reading
    
    def _remove_positional_duplicates(self, group: List[Dict]) -> List[Dict]:
        """
        移除位置相近的重复数字，保留置信度最高的
        
        Args:
            group: 数字组
            
        Returns:
            去重后的数字组
        """
        if len(group) <= 1:
            return group
        
        # 按置信度降序排序
        group = sorted(group, key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for det in group:
            is_duplicate = False
            
            for existing in filtered:
                # 检查是否为位置相近的相同类别
                if det['class'] == existing['class']:
                    # 计算中心距离
                    dx = det['center_x'] - existing['center_x']
                    dy = det['center_y'] - existing['center_y']
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # 计算平均框大小作为距离阈值
                    avg_size = np.mean([
                        det['bbox'][2] - det['bbox'][0],
                        det['bbox'][3] - det['bbox'][1],
                        existing['bbox'][2] - existing['bbox'][0],
                        existing['bbox'][3] - existing['bbox'][1]
                    ])
                    
                    # 如果距离小于平均尺寸的0.5倍，认为是重复
                    if distance < avg_size * 0.5:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(det)
        
        # 重新按x坐标排序
        return sorted(filtered, key=lambda x: x['center_x'])
    
    def _validate_reading(self, reading: str) -> str:
        """
        验证并清理读数字符串
        
        Args:
            reading: 原始读数字符串
            
        Returns:
            清理后的读数字符串
        """
        if not reading:
            return "0"
        
        # 移除连续的小数点
        while '..' in reading:
            reading = reading.replace('..', '.')
        
        # 移除开头和结尾的小数点
        reading = reading.strip('.')
        
        # 如果为空，返回"0"
        if not reading:
            return "0"
        
        # 处理多个小数点的情况，只保留第一个
        if reading.count('.') > 1:
            parts = reading.split('.')
            if parts[0]:  # 如果第一部分不为空
                reading = parts[0] + '.' + ''.join(parts[1:])
            else:  # 如果第一部分为空，取第二部分作为整数
                reading = ''.join(parts[1:])
                if '.' in reading:  # 如果还有小数点
                    sub_parts = reading.split('.')
                    reading = sub_parts[0] + '.' + ''.join(sub_parts[1:])
        
        # 确保不以小数点开头
        if reading.startswith('.'):
            reading = '0' + reading
        
        # 移除末尾的小数点
        if reading.endswith('.'):
            reading = reading[:-1]
        
        # 验证是否为有效数字格式
        try:
            float(reading)
        except ValueError:
            # 如果不是有效数字，尝试提取纯数字
            digits_only = ''.join([c for c in reading if c.isdigit() or c == '.'])
            if digits_only and digits_only != '.':
                reading = digits_only
                # 再次清理
                if reading.count('.') > 1:
                    parts = reading.split('.')
                    reading = parts[0] + '.' + ''.join(parts[1:])
            else:
                reading = "0"
        
        return reading
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Visualize digit detections on image
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Image with visualization
        """
        vis_img = image.copy()
        
        # Define colors for different classes
        colors = {
            'point': (0, 0, 255),  # Red for decimal point
            'digit': (0, 255, 0)   # Green for digits
        }
        
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            
            # Choose color
            color = colors['point'] if class_name == 'point' else colors['digit']
            
            # Draw bounding box
            cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(vis_img, 
                         (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), 
                         color, -1)
            
            # Label text
            cv2.putText(vis_img, label, 
                       (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw center point
            center = (int(det['center_x']), int(det['center_y']))
            cv2.circle(vis_img, center, 3, color, -1)
        
        return vis_img


class MeterReadingApp:
    """Complete meter reading application"""
    
    def __init__(self):
        """Initialize the application"""
        # Model paths
        self.detection_model_path = "models/detection/detection_model.pt"
        self.segmentation_model_path = "models/segmentation/segmentation_model.onnx"
        
        # Fallback to base models if trained models not available
        if not os.path.exists(self.detection_model_path):
            self.detection_model_path = "yolov10n.pt"
            print("Using base YOLOv10 model (not trained on meters)")
        
        # Initialize components
        self.detector = MeterDetector(self.detection_model_path)
        
        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.segmentor = MeterSegmentor(self.segmentation_model_path, self.device)
        self.reader = MeterReader(scale_range=(0.0, 1.6), debug=False)
    
    def process_image(self, image: np.ndarray, conf_threshold: float = 0.5, 
                     scale_min: float = 0.0, scale_max: float = 1.6) -> Dict[str, Any]:
        """
        Complete processing pipeline for meter reading
        
        Args:
            image: Input image
            conf_threshold: Detection confidence threshold
            scale_min: Minimum scale value
            scale_max: Maximum scale value
            
        Returns:
            Dictionary with all results and visualizations
        """
        results = {
            'success': False,
            'error': None,
            'detections': [],
            'readings': [],
            'visualizations': {}
        }
        
        try:
            # Step 1: Detection
            detections = self.detector.detect_meters(image, conf_threshold)
            results['detections'] = detections
            
            if not detections:
                results['error'] = "No meters detected in the image"
                return results
            
            # Process each detected meter
            for i, detection in enumerate(detections):
                try:
                    # Step 2: Crop meter region
                    cropped_meter = self.detector.crop_meter(image, detection['bbox'])
                    
                    # Step 3: Segmentation
                    segmentation_mask = self.segmentor.segment_meter(cropped_meter)
                    
                    # Step 4: Reading extraction
                    self.reader.scale_beginning = scale_min
                    self.reader.scale_end = scale_max
                    reading = self.reader.process_single_meter(cropped_meter, segmentation_mask)
                    
                    if reading is not None:
                        results['readings'].append({
                            'meter_id': i,
                            'reading': reading,
                            'confidence': detection['confidence'],
                            'bbox': detection['bbox']
                        })
                    
                    # Generate visualizations
                    vis_detection = self._visualize_detection(image, [detection])
                    vis_crop = cropped_meter
                    vis_segmentation = self._visualize_segmentation(cropped_meter, segmentation_mask)
                    vis_result = self._visualize_reading_result(cropped_meter, segmentation_mask, reading)
                    
                    results['visualizations'][f'meter_{i}'] = {
                        'detection': vis_detection,
                        'crop': vis_crop,
                        'segmentation': vis_segmentation,
                        'result': vis_result
                    }
                    
                except Exception as e:
                    print(f"Error processing meter {i}: {e}")
                    continue
            
            results['success'] = len(results['readings']) > 0
            if not results['success']:
                results['error'] = "Failed to extract readings from detected meters"
                
        except Exception as e:
            results['error'] = f"Processing error: {str(e)}"
            
        return results
    
    def _visualize_detection(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Visualize detection results"""
        vis_img = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            conf = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Draw label
            label = f"Meter: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_img, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), (0, 255, 0), -1)
            cv2.putText(vis_img, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return vis_img
    
    def _visualize_segmentation(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Visualize segmentation results"""
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 1] = [0, 0, 255]    # Pointer - Red
        colored_mask[mask == 2] = [0, 255, 0]    # Scale - Green
        
        # Blend with original image
        alpha = 0.6
        vis_img = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
        
        # Add statistics text
        pointer_pixels = np.sum(mask == 1)
        scale_pixels = np.sum(mask == 2)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        # Add text overlay with statistics
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        
        text_lines = [
            f"Pointer: {pointer_pixels} px ({pointer_pixels/total_pixels*100:.1f}%)",
            f"Scale: {scale_pixels} px ({scale_pixels/total_pixels*100:.1f}%)",
            f"Post-processed: Cleaned noise & boundaries"
        ]
        
        y_offset = 15
        for i, text in enumerate(text_lines):
            y_pos = y_offset + i * 15
            # Add background rectangle for better readability
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            cv2.rectangle(vis_img, (5, y_pos - 12), (text_size[0] + 10, y_pos + 3), (0, 0, 0), -1)
            cv2.putText(vis_img, text, (8, y_pos), font, font_scale, (255, 255, 255), thickness)
        
        return vis_img
    
    def _visualize_reading_result(self, image: np.ndarray, mask: np.ndarray, reading: Optional[float]) -> np.ndarray:
        """Visualize final reading result"""
        if reading is None:
            return image
        
        # Use the reader's visualization method
        try:
            # Extract components for visualization
            pointer_mask = self.reader.threshold_by_category(mask, 1)
            scale_mask = self.reader.threshold_by_category(mask, 2)
            
            # Find components
            scale_locations = self.reader.get_scale_locations(scale_mask)
            center = self.reader.get_center_location(image)
            pointer_locations = self.reader.get_pointer_locations(pointer_mask, center) if center else None
            
            if all([scale_locations, center, pointer_locations]):
                vis_img = self.reader.visualize_result(image, scale_locations, pointer_locations, center, reading)
                return vis_img
        except:
            pass
        
        # Fallback: simple text overlay
        vis_img = image.copy()
        text = f"Reading: {reading:.3f}"
        cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return vis_img


class DigitReadingApp:
    """LCD digit reading application"""
    
    def __init__(self):
        """Initialize the digit reading application"""
        # Model path for digit detection
        self.digit_model_path = "models/detection/digits_model.pt"
        
        # Initialize digit detector
        self.digit_detector = DigitDetector(self.digit_model_path)
        print(f"Digit reading app initialized with model: {self.digit_model_path}")
    
    def process_digit_image(self, image: np.ndarray, conf_threshold: float = 0.3,
                          overlap_threshold: float = 0.7, 
                          distance_threshold: float = 30) -> Dict[str, Any]:
        """
        Process LCD digit image and extract reading
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Detection confidence threshold
            overlap_threshold: IoU threshold for duplicate filtering
            distance_threshold: Distance threshold for duplicate filtering
            
        Returns:
            Dictionary with processing results
        """
        results = {
            'success': False,
            'reading': '',
            'raw_detections': [],
            'filtered_detections': [],
            'error': None
        }
        
        try:
            # Step 1: Detect all digits
            raw_detections = self.digit_detector.detect_digits(image, conf_threshold)
            results['raw_detections'] = raw_detections
            
            if not raw_detections:
                results['error'] = "No digits detected in the image"
                return results
            
            # Step 2: Filter duplicate detections
            filtered_detections = self.digit_detector.filter_duplicate_detections(
                raw_detections, overlap_threshold, distance_threshold)
            results['filtered_detections'] = filtered_detections
            
            # Step 3: Extract reading
            reading = self.digit_detector.extract_reading(filtered_detections)
            results['reading'] = reading
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = f"Processing error: {str(e)}"
        
        return results


def create_gradio_interface():
    """Create Gradio interface with tabs for different functionalities"""
    
    # Initialize apps
    meter_app = MeterReadingApp()
    digit_app = DigitReadingApp()
    
    def process_uploaded_image(image, conf_threshold, scale_min, scale_max):
        """Process uploaded image and return results for meter reading"""
        if image is None:
            return None, None, None, None, "Please upload an image"
        
        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process image
        results = meter_app.process_image(image_cv, conf_threshold, scale_min, scale_max)
        
        if not results['success']:
            error_msg = results.get('error', 'Unknown error occurred')
            return None, None, None, None, error_msg
        
        # Prepare outputs
        summary_text = f"Found {len(results['readings'])} meter(s)\n\n"
        for reading in results['readings']:
            summary_text += f"Meter {reading['meter_id']}: {reading['reading']:.3f} (conf: {reading['confidence']:.2f})\n"
        
        # Get visualizations for first meter
        if results['visualizations']:
            first_meter = list(results['visualizations'].keys())[0]
            vis = results['visualizations'][first_meter]
            
            # Convert BGR to RGB for display
            detection_img = cv2.cvtColor(vis['detection'], cv2.COLOR_BGR2RGB)
            crop_img = cv2.cvtColor(vis['crop'], cv2.COLOR_BGR2RGB)
            segmentation_img = cv2.cvtColor(vis['segmentation'], cv2.COLOR_BGR2RGB)
            result_img = cv2.cvtColor(vis['result'], cv2.COLOR_BGR2RGB)
            
            return detection_img, crop_img, segmentation_img, result_img, summary_text
        
        return None, None, None, None, summary_text
    
    def process_digit_image(image, conf_threshold, overlap_threshold, distance_threshold):
        """Process LCD digit image and return results"""
        if image is None:
            return None, None, "Please upload an image"
        
        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process image
        results = digit_app.process_digit_image(image_cv, conf_threshold, 
                                              overlap_threshold, distance_threshold)
        
        if not results['success']:
            error_msg = results.get('error', 'Unknown error occurred')
            return None, None, error_msg
        
        # Prepare visualizations
        raw_vis = digit_app.digit_detector.visualize_detections(
            image_cv, results['raw_detections'])
        filtered_vis = digit_app.digit_detector.visualize_detections(
            image_cv, results['filtered_detections'])
        
        # Convert BGR to RGB for display
        raw_vis_rgb = cv2.cvtColor(raw_vis, cv2.COLOR_BGR2RGB)
        filtered_vis_rgb = cv2.cvtColor(filtered_vis, cv2.COLOR_BGR2RGB)
        
        # 创建详细的处理信息
        raw_count = len(results['raw_detections'])
        filtered_count = len(results['filtered_detections'])
        removed_count = raw_count - filtered_count
        
        # 分析检测到的数字类别
        detected_classes = {}
        for det in results['filtered_detections']:
            class_name = det['class']
            if class_name not in detected_classes:
                detected_classes[class_name] = 0
            detected_classes[class_name] += 1
        
        class_info = ", ".join([f"{k}: {v}" for k, v in detected_classes.items()])
        
        # 构建详细信息
        summary_text = f"📱 LCD读数识别结果\n"
        summary_text += f"{'='*40}\n\n"
        
        summary_text += f"✅ 最终读数: **{results['reading']}**\n\n"
        
        summary_text += f"📊 检测统计:\n"
        summary_text += f"  🔍 原始检测: {raw_count} 个数字\n"
        summary_text += f"  🎯 过滤后: {filtered_count} 个数字\n"
        summary_text += f"  🗑️ 移除重复: {removed_count} 个\n\n"
        
        if class_info:
            summary_text += f"📋 检测类别统计:\n"
            summary_text += f"  {class_info}\n\n"
        
        summary_text += f"⚙️ 处理参数:\n"
        summary_text += f"  置信度阈值: {conf_threshold}\n"
        summary_text += f"  重叠阈值: {overlap_threshold}\n"
        summary_text += f"  距离阈值: {distance_threshold} 像素\n\n"
        
        if results['filtered_detections']:
            summary_text += "🔍 最终检测详情 (从左到右):\n"
            sorted_dets = sorted(results['filtered_detections'], key=lambda x: x['center_x'])
            for i, det in enumerate(sorted_dets):
                pos_x = int(det['center_x'])
                pos_y = int(det['center_y'])
                summary_text += f"  {i+1}. '{det['class']}' (置信度: {det['confidence']:.3f}, 位置: {pos_x},{pos_y})\n"
        
        # 如果有被移除的检测，显示一些信息
        if removed_count > 0:
            summary_text += f"\n⚠️ 移除的重复检测:\n"
            removed_dets = [det for det in results['raw_detections'] 
                          if det not in results['filtered_detections']]
            for i, det in enumerate(removed_dets[:5]):  # 最多显示5个
                summary_text += f"  - '{det['class']}' (置信度: {det['confidence']:.3f})\n"
            if len(removed_dets) > 5:
                summary_text += f"  ... 以及其他 {len(removed_dets)-5} 个重复检测\n"
        
        return raw_vis_rgb, filtered_vis_rgb, summary_text
    
    # Create interface with tabs
    with gr.Blocks(title="Meter Reading Extraction", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🔧 Industrial Meter & LCD Display Reading System
        
        AI-powered system for extracting readings from industrial meters and LCD displays.
        """)
        
        with gr.Tabs():
            # Tab 1: Traditional Meter Reading
            with gr.TabItem("🔧 Industrial Meters"):
                gr.Markdown("""
                ## Industrial Meter Reading Extraction
                
                Upload an image containing industrial meters to automatically extract readings using AI.
                
                **Pipeline:** Detection → Cropping → Segmentation → Reading Extraction
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Input section
                        gr.Markdown("### 📤 Input")
                        meter_image_input = gr.Image(type="pil", label="Upload Meter Image")
                        
                        with gr.Row():
                            meter_conf_threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.1, 
                                                           label="Detection Confidence")
                            
                        with gr.Row():
                            scale_min = gr.Number(value=0.0, label="Scale Min Value")
                            scale_max = gr.Number(value=1.6, label="Scale Max Value")
                        
                        meter_process_btn = gr.Button("🚀 Extract Readings", variant="primary", size="lg")
                        
                        # Results summary
                        gr.Markdown("### 📊 Results")
                        meter_results_text = gr.Textbox(label="Summary", lines=5, interactive=False)
                    
                    with gr.Column(scale=2):
                        # Visualization section
                        gr.Markdown("### 👁️ Process Visualization")
                        
                        with gr.Row():
                            meter_detection_output = gr.Image(label="1. Detection Results")
                            meter_crop_output = gr.Image(label="2. Cropped Meter")
                        
                        with gr.Row():
                            meter_segmentation_output = gr.Image(label="3. Segmentation Masks")
                            meter_result_output = gr.Image(label="4. Final Reading")
                
                # Event handlers for meter tab
                meter_process_btn.click(
                    fn=process_uploaded_image,
                    inputs=[meter_image_input, meter_conf_threshold, scale_min, scale_max],
                    outputs=[meter_detection_output, meter_crop_output, 
                           meter_segmentation_output, meter_result_output, meter_results_text]
                )
                
                # Examples for meter
                gr.Markdown("### 📋 Usage Instructions")
                gr.Markdown("""
                1. **Upload Image**: Choose an image containing industrial meters
                2. **Adjust Settings**: 
                   - Detection Confidence: Higher values = more strict detection
                   - Scale Range: Set the min/max values of your meter scale
                3. **Process**: Click "Extract Readings" to run the complete pipeline
                4. **View Results**: Check the visualization and summary
                
                **Supported Formats**: JPG, PNG, BMP
                **Best Results**: Clear, well-lit images with visible meter faces
                """)
            
            # Tab 2: LCD Digit Reading
            with gr.TabItem("📱 LCD Display Reading"):
                gr.Markdown("""
                ## LCD Digital Display Reading
                
                Upload an image containing LCD digital displays to automatically extract numeric readings.
                
                **Features:** 
                - Detects digits 0-9 and decimal points
                - Filters duplicate detections
                - Reads from left to right (high to low digit)
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Input section
                        gr.Markdown("### 📤 Input")
                        digit_image_input = gr.Image(type="pil", label="Upload LCD Display Image")
                        
                        gr.Markdown("### ⚙️ Detection Settings")
                        digit_conf_threshold = gr.Slider(0.1, 0.9, value=0.3, step=0.05, 
                                                       label="Detection Confidence")
                        
                        gr.Markdown("### 🔧 Duplicate Filtering")
                        overlap_threshold = gr.Slider(0.1, 1.0, value=0.7, step=0.05,
                                                    label="Overlap Threshold (IoU)")
                        distance_threshold = gr.Slider(5, 100, value=30, step=5,
                                                     label="Distance Threshold (pixels)")
                        
                        digit_process_btn = gr.Button("🔍 Extract Reading", variant="primary", size="lg")
                        
                        # Results summary
                        gr.Markdown("### 📊 Results")
                        digit_results_text = gr.Textbox(label="Detection Summary", lines=8, interactive=False)
                    
                    with gr.Column(scale=2):
                        # Visualization section
                        gr.Markdown("### 👁️ Detection Visualization")
                        
                        with gr.Row():
                            raw_detection_output = gr.Image(label="Raw Detections")
                            filtered_detection_output = gr.Image(label="Filtered Detections")
                
                # Event handlers for digit tab
                digit_process_btn.click(
                    fn=process_digit_image,
                    inputs=[digit_image_input, digit_conf_threshold, overlap_threshold, distance_threshold],
                    outputs=[raw_detection_output, filtered_detection_output, digit_results_text]
                )
                
                # Examples for digit reading
                gr.Markdown("### 📋 Usage Instructions")
                gr.Markdown("""
                1. **Upload Image**: Choose an image containing LCD digital displays
                2. **Adjust Detection Settings**: 
                   - Detection Confidence: Lower values detect more digits but may include false positives
                3. **Configure Duplicate Filtering**:
                   - Overlap Threshold: Higher values are more strict about overlapping detections
                   - Distance Threshold: Minimum distance between same digits to be considered separate
                4. **Process**: Click "Extract Reading" to analyze the display
                5. **View Results**: Check both raw and filtered detections
                
                **Tips**:
                - For clear displays: Use higher confidence (0.5-0.7)
                - For blurry/poor quality: Use lower confidence (0.2-0.4)
                - Adjust distance threshold based on digit spacing in your displays
                
                **Supported**: Numbers 0-9, decimal points, multi-digit readings
                """)
    
    return interface


def main():
    """Main function to launch the application"""
    print("Initializing Meter Reading Extraction App...")
    
    # Create and launch interface
    interface = create_gradio_interface()
    
    print("Launching Gradio interface...")
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        debug=True
    )


if __name__ == "__main__":
    main() 