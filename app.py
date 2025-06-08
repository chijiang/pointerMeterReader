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
import torch.nn.functional as F
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os
import sys
from typing import Tuple, Optional, List, Dict, Any
import json
from PIL import Image
import io
import base64

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