#!/usr/bin/env python3
"""
Meter Reading Extraction Web Application
A complete pipeline for extracting readings from industrial meter displays

Pipeline:
1. Upload image -> YOLO detection -> Crop meter region
2. Crop -> SegFormer segmentation -> Generate masks
3. Masks -> Reading extraction -> Final result

Author: Chijiang
Date: 2025-11-21
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

# Import meter reading module
from scripts.meter_reading import MeterReader

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
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = r.names.get(cls_id, 'unknown')

                    # Only keep gauge detections (class 0: gauge, class 1: gauges)
                    # Skip class 2 (numbers) as they are not meter regions
                    if cls_id in [0, 1] or 'gauge' in cls_name.lower():
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': conf,
                            'class': cls_name
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
    """ONNX-based meter segmentation using SegFormer"""

    # Default input size for SegFormer model
    DEFAULT_INPUT_SIZE = 512

    def __init__(self, model_path: str, device: str = 'cpu', post_process_config: Dict = None, input_size: int = None):
        """Initialize segmentor with ONNX model

        Args:
            model_path: Path to ONNX model file
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            post_process_config: Configuration for post-processing masks
            input_size: Input image size for the model (default: auto-detect or 512)
        """
        self.device = device
        self.input_size = input_size  # Will be set during model loading if None
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
        """Load ONNX segmentation model (SegFormer or DeepLabV3+)"""
        # Check model paths in order of priority
        # 1. SegFormer model (preferred)
        # 2. Original model path
        # 3. Legacy DeepLabV3+ model

        search_paths = [
            "models/segmentation/segformer_meter.onnx",  # SegFormer model
            model_path,  # Provided path
            model_path.replace('.pth', '.onnx'),  # Convert .pth to .onnx
            "models/segmentation/segmentation_model.onnx",  # Legacy DeepLabV3+
        ]

        onnx_path = None
        for path in search_paths:
            if os.path.exists(path):
                onnx_path = path
                break

        if onnx_path is None:
            print(f"⚠️  No ONNX model found. Searched paths:")
            for path in search_paths:
                print(f"   - {path}")
            return None

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

            # Auto-detect input size from model
            input_shape = session.get_inputs()[0].shape
            if len(input_shape) >= 4 and isinstance(input_shape[2], int):
                detected_size = input_shape[2]
                if self.input_size is None:
                    self.input_size = detected_size
                print(f"✅ Loaded SegFormer ONNX model from: {onnx_path}")
            else:
                if self.input_size is None:
                    self.input_size = self.DEFAULT_INPUT_SIZE
                print(f"✅ Loaded ONNX model from: {onnx_path}")

            print(f"📊 Input shape: {input_shape}")
            print(f"📊 Output shape: {session.get_outputs()[0].shape}")
            print(f"📐 Using input size: {self.input_size}x{self.input_size}")
            print(f"🔧 Providers: {session.get_providers()}")

            return session

        except Exception as e:
            print(f"❌ Error loading ONNX model: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Preprocess image for ONNX inference (SegFormer compatible)

        Args:
            image: Input image in BGR format
            target_size: Target size (height, width). If None, use self.input_size

        Returns:
            Preprocessed image tensor in NCHW format
        """
        if target_size is None:
            target_size = (self.input_size, self.input_size)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image
        image_resized = cv2.resize(image_rgb, target_size)

        # Normalize to [0, 1] and then apply ImageNet normalization
        image_normalized = image_resized.astype(np.float32) / 255.0

        # ImageNet normalization (same for both DeepLabV3+ and SegFormer)
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


class MeterReadingApp:
    """Complete meter reading application"""

    def __init__(self):
        """Initialize the application"""
        # Model paths - use newly trained YOLOv11 model
        self.detection_model_path = "models/detection/yolo11_meter.pt"
        # Prefer SegFormer model, fallback to legacy DeepLabV3+
        self.segmentation_model_path = "models/segmentation/segformer_meter.onnx"

        # Fallback to base models if trained models not available
        if not os.path.exists(self.detection_model_path):
            # Try alternative paths
            alt_paths = [
                "models/detection/detection_model.pt",
                "yolo11m.pt",
                "yolov10n.pt"
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    self.detection_model_path = alt_path
                    print(f"Using fallback model: {alt_path}")
                    break
            else:
                print("Warning: No detection model found")
        
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
        Complete processing pipeline
        
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

                    # Filter out invalid crops (too small or wrong aspect ratio)
                    h, w = cropped_meter.shape[:2]
                    min_size = 100  # Minimum dimension
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 999

                    if w < min_size or h < min_size:
                        print(f"\n⏭️ [SKIP] Meter {i}: Too small ({w}x{h}), min required: {min_size}")
                        continue

                    if aspect_ratio > 3.0:
                        print(f"\n⏭️ [SKIP] Meter {i}: Bad aspect ratio ({aspect_ratio:.2f}), likely not a gauge")
                        continue
                    
                    # Step 3: Segmentation
                    segmentation_mask = self.segmentor.segment_meter(cropped_meter)

                    # Debug logging for segmentation mask
                    print(f"\n📊 [DEBUG] Meter {i} Segmentation Analysis:")
                    print(f"   - Cropped image shape: {cropped_meter.shape}")
                    print(f"   - Mask shape: {segmentation_mask.shape}")
                    print(f"   - Mask unique values: {np.unique(segmentation_mask)}")
                    print(f"   - Background (0) pixels: {np.sum(segmentation_mask == 0)}")
                    print(f"   - Pointer (1) pixels: {np.sum(segmentation_mask == 1)}")
                    print(f"   - Scale (2) pixels: {np.sum(segmentation_mask == 2)}")

                    # Check if scale mask has valid pixels
                    scale_mask = (segmentation_mask == 2).astype(np.uint8) * 255
                    if np.sum(scale_mask) > 0:
                        # Find scale bounding box
                        coords = np.where(scale_mask > 0)
                        print(f"   - Scale region: rows [{coords[0].min()}-{coords[0].max()}], cols [{coords[1].min()}-{coords[1].max()}]")
                    else:
                        print(f"   ⚠️ WARNING: No scale pixels detected in segmentation mask!")

                    # Check pointer mask
                    pointer_mask = (segmentation_mask == 1).astype(np.uint8) * 255
                    if np.sum(pointer_mask) > 0:
                        coords = np.where(pointer_mask > 0)
                        print(f"   - Pointer region: rows [{coords[0].min()}-{coords[0].max()}], cols [{coords[1].min()}-{coords[1].max()}]")
                    else:
                        print(f"   ⚠️ WARNING: No pointer pixels detected in segmentation mask!")

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


def create_gradio_interface():
    """Create Gradio interface"""
    
    # Initialize app
    app = MeterReadingApp()
    
    def process_uploaded_image(image, conf_threshold, scale_min, scale_max):
        """Process uploaded image and return results"""
        if image is None:
            return None, None, None, None, "Please upload an image"
        
        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process image
        results = app.process_image(image_cv, conf_threshold, scale_min, scale_max)
        
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
    
    # Create interface
    with gr.Blocks(title="Meter Reading Extraction", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🔧 Industrial Meter Reading Extraction
        
        Upload an image containing industrial meters to automatically extract readings using AI.
        
        **Pipeline:** Detection → Cropping → Segmentation → Reading Extraction
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("## 📤 Input")
                image_input = gr.Image(type="pil", label="Upload Meter Image")
                
                with gr.Row():
                    conf_threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.1, 
                                             label="Detection Confidence")
                    
                with gr.Row():
                    scale_min = gr.Number(value=0.0, label="Scale Min Value")
                    scale_max = gr.Number(value=1.6, label="Scale Max Value")
                
                process_btn = gr.Button("🚀 Extract Readings", variant="primary", size="lg")
                
                # Results summary
                gr.Markdown("## 📊 Results")
                results_text = gr.Textbox(label="Summary", lines=5, interactive=False)
            
            with gr.Column(scale=2):
                # Visualization section
                gr.Markdown("## 👁️ Process Visualization")
                
                with gr.Row():
                    detection_output = gr.Image(label="1. Detection Results")
                    crop_output = gr.Image(label="2. Cropped Meter")
                
                with gr.Row():
                    segmentation_output = gr.Image(label="3. Segmentation Masks")
                    result_output = gr.Image(label="4. Final Reading")
        
        # Event handlers
        process_btn.click(
            fn=process_uploaded_image,
            inputs=[image_input, conf_threshold, scale_min, scale_max],
            outputs=[detection_output, crop_output, segmentation_output, result_output, results_text]
        )
        
        # Examples
        gr.Markdown("## 📋 Usage Instructions")
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
    
    return interface


