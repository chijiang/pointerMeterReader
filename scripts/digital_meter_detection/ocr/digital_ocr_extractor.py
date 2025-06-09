#!/usr/bin/env python3
"""
液晶数字OCR提取脚本

使用EasyOCR从增强后的液晶屏图像中提取数字内容。
支持多种图像格式和批量处理，专门优化数字识别。

依赖安装:
    pip install easyocr pillow

功能特性：
1. 高精度数字识别
2. 多种OCR引擎支持
3. 结果过滤和验证
4. 批量处理能力
5. 详细的识别报告

作者: chijiang
日期: 2025-06-09
"""

import os
import sys
import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import warnings
import re

warnings.filterwarnings('ignore')

# 尝试导入OCR库
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

class DigitalOCRExtractor:
    """液晶数字OCR提取器"""
    
    def __init__(self, ocr_engine: str = "easyocr", languages: List[str] = None):
        """
        初始化OCR提取器
        
        Args:
            ocr_engine: OCR引擎类型 ("easyocr", "paddleocr", "tesseract")
            languages: 支持的语言列表，默认为['en']
        """
        self.project_root = self._get_project_root()
        self.ocr_engine = ocr_engine
        self.languages = languages or ['en']
        
        # 初始化OCR引擎
        self.reader = self._initialize_ocr_engine()
        
        # 设置输出目录
        self.setup_output_dirs()
        
        print(f"🔤 数字OCR提取器已初始化")
        print(f"📁 项目根目录: {self.project_root}")
        print(f"🔧 OCR引擎: {self.ocr_engine}")
        print(f"🌐 支持语言: {self.languages}")
        print(f"📊 输出目录: {self.output_dir}")
    
    def _get_project_root(self) -> Path:
        """智能获取项目根目录"""
        current_dir = Path.cwd()
        if current_dir.name == "ocr":
            return current_dir.parent.parent.parent
        elif current_dir.name == "digital_meter_detection":
            return current_dir.parent.parent
        elif current_dir.name == "scripts":
            return current_dir.parent
        else:
            return current_dir
    
    def _initialize_ocr_engine(self):
        """初始化OCR引擎"""
        if self.ocr_engine == "easyocr":
            if not EASYOCR_AVAILABLE:
                raise ImportError("EasyOCR not installed. Run: pip install easyocr")
            
            print("🔧 初始化EasyOCR...")
            # 使用GPU加速（如果可用）
            reader = easyocr.Reader(self.languages, gpu=True)
            print("✅ EasyOCR初始化完成")
            return reader
        
        elif self.ocr_engine == "paddleocr":
            if not PADDLEOCR_AVAILABLE:
                raise ImportError("PaddleOCR not installed. Run: pip install paddleocr")
            
            print("🔧 初始化PaddleOCR...")
            # 使用英文+数字模式，关闭方向分类器加速识别
            reader = PaddleOCR(use_angle_cls=False, lang='en')
            print("✅ PaddleOCR初始化完成")
            return reader
        
        elif self.ocr_engine == "tesseract":
            try:
                import pytesseract
                print("🔧 使用Tesseract OCR")
                return None  # Tesseract不需要预加载
            except ImportError:
                raise ImportError("Tesseract not installed. Run: pip install pytesseract")
        
        else:
            raise ValueError(f"不支持的OCR引擎: {self.ocr_engine}")
    
    def setup_output_dirs(self):
        """设置输出目录"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 输出目录结构
        self.output_root = self.project_root / "outputs" / "digital_ocr"
        self.output_dir = self.output_root / f"ocr_results_{self.timestamp}"
        self.results_dir = self.output_dir / "extracted_text"
        self.analysis_dir = self.output_dir / "analysis"
        self.visualization_dir = self.output_dir / "visualization"
        
        # 创建目录
        for dir_path in [self.results_dir, self.analysis_dir, self.visualization_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """为OCR预处理图像"""
        # 如果是彩色图像，转换为灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 确保文字是黑色，背景是白色
        # 检查图像的亮度分布
        mean_intensity = np.mean(gray)
        
        # 如果背景较暗（液晶屏通常是深色背景），反转图像
        if mean_intensity < 127:
            gray = cv2.bitwise_not(gray)
        
        # 调整大小以提高OCR准确性（至少300 DPI）
        height, width = gray.shape
        if height < 100 or width < 300:
            scale_factor = max(300/width, 100/height, 2.0)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # 进一步去噪
        gray = cv2.medianBlur(gray, 3)
        
        return gray
    
    def extract_with_paddleocr(self, image: np.ndarray) -> List[Dict]:
        """使用PaddleOCR提取文字"""
        # 预处理图像
        processed_image = self.preprocess_for_ocr(image)
        
        # PaddleOCR识别
        results = self.reader.ocr(processed_image, cls=False)
        
        # 格式化结果
        formatted_results = []
        if results and results[0]:  # PaddleOCR返回格式: [[[bbox], (text, confidence)]]
            for line in results[0]:
                bbox, (text, confidence) = line
                
                # 只保留数字和小数点、负号
                clean_text = re.sub(r'[^0-9.\-]', '', text)
                
                if clean_text:  # 只保留包含数字的结果
                    formatted_results.append({
                        'text': text,
                        'clean_text': clean_text,
                        'confidence': float(confidence),
                        'bbox': bbox,
                        'engine': 'paddleocr'
                    })
        
        return formatted_results
    
    def extract_with_easyocr(self, image: np.ndarray) -> List[Dict]:
        """使用EasyOCR提取文字"""
        # 预处理图像
        processed_image = self.preprocess_for_ocr(image)
        
        # EasyOCR识别
        results = self.reader.readtext(processed_image)
        
        # 格式化结果
        formatted_results = []
        for (bbox, text, confidence) in results:
            # 只保留数字和小数点、负号
            clean_text = re.sub(r'[^0-9.\-]', '', text)
            
            if clean_text:  # 只保留包含数字的结果
                formatted_results.append({
                    'text': text,
                    'clean_text': clean_text,
                    'confidence': float(confidence),
                    'bbox': bbox,
                    'engine': 'easyocr'
                })
        
        return formatted_results
    
    def extract_with_tesseract(self, image: np.ndarray) -> List[Dict]:
        """使用Tesseract提取文字"""
        try:
            import pytesseract
        except ImportError:
            raise ImportError("Tesseract not available")
        
        # 预处理图像
        processed_image = self.preprocess_for_ocr(image)
        
        # Tesseract配置 - 专门用于数字识别
        config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.-'
        
        # 获取详细结果
        data = pytesseract.image_to_data(processed_image, config=config, output_type=pytesseract.Output.DICT)
        
        formatted_results = []
        n_boxes = len(data['level'])
        
        for i in range(n_boxes):
            confidence = int(data['conf'][i])
            text = data['text'][i].strip()
            
            if confidence > 30 and text:  # 过滤低置信度结果
                # 清理文本
                clean_text = re.sub(r'[^0-9.\-]', '', text)
                
                if clean_text:
                    # 构建边界框
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bbox = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                    
                    formatted_results.append({
                        'text': text,
                        'clean_text': clean_text,
                        'confidence': confidence / 100.0,
                        'bbox': bbox,
                        'engine': 'tesseract'
                    })
        
        return formatted_results
    
    def validate_digital_reading(self, text: str) -> Dict:
        """验证数字读数"""
        validation_result = {
            'is_valid': False,
            'value': None,
            'format_type': 'unknown',
            'issues': []
        }
        
        # 清理文本
        clean_text = re.sub(r'[^0-9.\-]', '', text.strip())
        
        if not clean_text:
            validation_result['issues'].append('无数字内容')
            return validation_result
        
        try:
            # 尝试转换为数字
            if '.' in clean_text:
                value = float(clean_text)
                validation_result['format_type'] = 'decimal'
            else:
                value = int(clean_text)
                validation_result['format_type'] = 'integer'
            
            validation_result['value'] = value
            validation_result['is_valid'] = True
            
            # 检查合理性
            if abs(value) > 999999:
                validation_result['issues'].append('数值过大')
            
            if '.' in clean_text and len(clean_text.split('.')[1]) > 3:
                validation_result['issues'].append('小数位过多')
                
        except ValueError:
            validation_result['issues'].append('无法转换为数字')
        
        return validation_result
    
    def extract_from_image(self, image: np.ndarray) -> Dict:
        """从单张图像提取数字"""
        results = {
            'raw_results': [],
            'validated_results': [],
            'best_result': None,
            'extraction_summary': {}
        }
        
        # 使用选定的OCR引擎
        if self.ocr_engine == "easyocr":
            raw_results = self.extract_with_easyocr(image)
        elif self.ocr_engine == "paddleocr":
            raw_results = self.extract_with_paddleocr(image)
        elif self.ocr_engine == "tesseract":
            raw_results = self.extract_with_tesseract(image)
        else:
            raise ValueError(f"不支持的OCR引擎: {self.ocr_engine}")
        
        results['raw_results'] = raw_results
        
        # 验证每个结果
        validated_results = []
        for result in raw_results:
            validation = self.validate_digital_reading(result['clean_text'])
            result.update(validation)
            validated_results.append(result)
        
        results['validated_results'] = validated_results
        
        # 选择最佳结果
        valid_results = [r for r in validated_results if r['is_valid']]
        if valid_results:
            # 按置信度排序
            best_result = max(valid_results, key=lambda x: x['confidence'])
            results['best_result'] = best_result
        
        # 生成摘要
        results['extraction_summary'] = {
            'total_detections': len(raw_results),
            'valid_detections': len(valid_results),
            'best_confidence': results['best_result']['confidence'] if results['best_result'] else 0,
            'extracted_value': results['best_result']['value'] if results['best_result'] else None
        }
        
        return results
    
    def process_single_image(self, image_path: Union[str, Path]) -> Dict:
        """处理单张图像"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        print(f"🔤 OCR提取: {image_path.name}")
        
        # 加载图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 执行OCR提取
        ocr_results = self.extract_from_image(image)
        
        # 保存结果
        image_name = image_path.stem
        
        # 保存OCR结果为JSON
        results_file = self.results_dir / f"{image_name}_ocr_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # 处理numpy数组以便JSON序列化
            serializable_results = self._make_json_serializable(ocr_results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 创建可视化
        self.create_ocr_visualization(image, ocr_results, 
                                    self.visualization_dir / f"{image_name}_ocr_visualization.png")
        
        return {
            'image_path': str(image_path),
            'image_name': image_name,
            'results_file': str(results_file),
            'extracted_value': ocr_results['extraction_summary']['extracted_value'],
            'confidence': ocr_results['extraction_summary']['best_confidence'],
            'total_detections': ocr_results['extraction_summary']['total_detections'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _make_json_serializable(self, obj):
        """使对象可序列化为JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def create_ocr_visualization(self, image: np.ndarray, ocr_results: Dict, output_path: Path):
        """创建OCR结果可视化"""
        # 创建图像副本用于绘制
        vis_image = image.copy()
        
        # 绘制所有检测到的文本框
        for result in ocr_results['validated_results']:
            bbox = result['bbox']
            confidence = result['confidence']
            text = result['clean_text']
            is_valid = result['is_valid']
            
            # 转换bbox为整数坐标
            points = np.array(bbox, dtype=np.int32)
            
            # 根据有效性选择颜色
            color = (0, 255, 0) if is_valid else (0, 0, 255)  # 绿色=有效，红色=无效
            
            # 绘制边界框
            cv2.polylines(vis_image, [points], True, color, 2)
            
            # 绘制文本和置信度
            x, y = points[0]
            label = f"{text} ({confidence:.2f})"
            cv2.putText(vis_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 创建matplotlib图形
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原图
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # OCR结果
        axes[1].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title('OCR Detection Results')
        axes[1].axis('off')
        
        # 添加结果摘要
        summary = ocr_results['extraction_summary']
        best_result = ocr_results['best_result']
        
        summary_text = f"Detections: {summary['total_detections']}\n"
        summary_text += f"Valid: {summary['valid_detections']}\n"
        if best_result:
            summary_text += f"Best Value: {best_result['value']}\n"
            summary_text += f"Confidence: {best_result['confidence']:.3f}"
        
        plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def process_batch(self, input_path: Union[str, Path]) -> List[Dict]:
        """批量处理图像"""
        input_path = Path(input_path)
        
        # 获取图像文件列表
        if input_path.is_file():
            image_files = [input_path]
        elif input_path.is_dir():
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                image_files.extend(input_path.glob(ext))
                image_files.extend(input_path.glob(ext.upper()))
        else:
            raise ValueError(f"输入路径无效: {input_path}")
        
        if not image_files:
            raise ValueError(f"未找到图像文件: {input_path}")
        
        print(f"🔤 开始批量OCR，共 {len(image_files)} 张图像")
        
        all_results = []
        
        # 处理每张图像
        for image_file in tqdm(image_files, desc="OCR提取"):
            try:
                result = self.process_single_image(image_file)
                all_results.append(result)
            except Exception as e:
                print(f"⚠️  处理 {image_file.name} 时出错: {e}")
                continue
        
        return all_results
    
    def save_results(self, all_results: List[Dict]):
        """保存处理结果"""
        print("\n💾 保存OCR结果...")
        
        # 保存详细结果
        results_file = self.output_dir / "ocr_extraction_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 生成总结报告
        successful_extractions = [r for r in all_results if r['extracted_value'] is not None]
        
        summary = {
            'total_images': len(all_results),
            'successful_extractions': len(successful_extractions),
            'success_rate': len(successful_extractions) / len(all_results) if all_results else 0,
            'average_confidence': np.mean([r['confidence'] for r in successful_extractions]) if successful_extractions else 0,
            'extracted_values': [r['extracted_value'] for r in successful_extractions],
            'ocr_engine': self.ocr_engine,
            'languages': self.languages,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = self.output_dir / "ocr_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self._generate_markdown_report(summary, all_results)
        
        print(f"✅ 结果保存完成:")
        print(f"  - 详细结果: {results_file}")
        print(f"  - OCR总结: {summary_file}")
    
    def _generate_markdown_report(self, summary: Dict, all_results: List[Dict]):
        """生成Markdown格式报告"""
        md_content = f"""# Digital OCR Extraction Report

## Processing Summary
- **Total Images**: {summary['total_images']}
- **Successfully Extracted**: {summary['successful_extractions']}
- **Success Rate**: {summary['success_rate']:.1%}
- **Average Confidence**: {summary['average_confidence']:.3f}
- **Processing Time**: {summary['timestamp'][:19].replace('T', ' ')}

## OCR Configuration
- **Engine**: {summary['ocr_engine']}
- **Languages**: {', '.join(summary['languages'])}

## Extracted Values
"""
        
        if summary['extracted_values']:
            md_content += "| Image | Extracted Value | Confidence |\n"
            md_content += "|-------|----------------|------------|\n"
            
            for result in all_results:
                if result['extracted_value'] is not None:
                    md_content += f"| {result['image_name']} | {result['extracted_value']} | {result['confidence']:.3f} |\n"
        else:
            md_content += "No values successfully extracted.\n"
        
        md_content += f"""

## Technical Details
- **Preprocessing**: Image scaling, noise reduction, contrast enhancement
- **Text Filtering**: Only digits, decimal points, and negative signs retained
- **Validation**: Automatic format checking and range validation

## Output Files
- **Results**: `extracted_text/` - Individual OCR results in JSON format
- **Visualizations**: `visualization/` - OCR detection visualizations
- **Analysis**: `analysis/` - Summary reports and statistics

---
*Report generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        md_file = self.output_dir / "ocr_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='液晶数字OCR提取脚本')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像文件或目录路径')
    parser.add_argument('--engine', type=str, default='easyocr',
                       choices=['easyocr', 'paddleocr', 'tesseract'],
                       help='OCR引擎选择 (默认: easyocr)')
    parser.add_argument('--languages', type=str, nargs='+', default=['en'],
                       help='支持的语言列表 (默认: en)')
    parser.add_argument('--output', type=str,
                       help='自定义输出目录')
    
    args = parser.parse_args()
    
    try:
        # 检查依赖
        if args.engine == 'easyocr' and not EASYOCR_AVAILABLE:
            print("❌ EasyOCR未安装。请运行: pip install easyocr")
            sys.exit(1)
        elif args.engine == 'paddleocr' and not PADDLEOCR_AVAILABLE:
            print("❌ PaddleOCR未安装。请运行: pip install paddleocr")
            sys.exit(1)

        
        if not PILLOW_AVAILABLE:
            print("❌ Pillow未安装。请运行: pip install pillow")
            sys.exit(1)
        
        # 创建OCR提取器
        extractor = DigitalOCRExtractor(ocr_engine=args.engine, languages=args.languages)
        
        # 如果指定了自定义输出目录
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            extractor.output_dir = output_dir
            extractor.results_dir = output_dir / "extracted_text"
            extractor.analysis_dir = output_dir / "analysis"
            extractor.visualization_dir = output_dir / "visualization"
            
            for dir_path in [extractor.results_dir, extractor.analysis_dir, extractor.visualization_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # 批量处理
        results = extractor.process_batch(args.input)
        
        # 保存结果
        extractor.save_results(results)
        
        # 输出统计信息
        successful = len([r for r in results if r['extracted_value'] is not None])
        print(f"\n🎉 OCR提取完成!")
        print(f"📊 处理图像: {len(results)} 张")
        print(f"✅ 成功提取: {successful} 张")
        print(f"📈 成功率: {successful/len(results):.1%}")
        print(f"📁 结果目录: {extractor.output_dir}")
        print(f"🔧 OCR引擎: {args.engine}")
        
    except Exception as e:
        print(f"❌ OCR提取过程出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 