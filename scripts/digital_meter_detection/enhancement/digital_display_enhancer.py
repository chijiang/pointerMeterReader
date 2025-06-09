#!/usr/bin/env python3
"""
液晶数字屏幕增强脚本

专门用于增强液晶屏显示的数字，解决反光、对比度低、显示不清晰等问题。
包含多种图像处理技术来提取和增强数字信息。

功能特性：
1. 反光去除和光照均衡
2. 对比度和清晰度增强
3. 数字区域分割和提取
4. 多种增强算法组合
5. 结果对比和可视化

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

warnings.filterwarnings('ignore')

class DigitalDisplayEnhancer:
    """液晶数字显示增强器"""
    
    def __init__(self, output_dir: None | Path = None):
        """初始化增强器"""
        self.project_root = self._get_project_root()
        self.setup_output_dirs(output_dir)
        
        print("🎨 液晶数字显示增强器已初始化")
        print(f"📁 项目根目录: {self.project_root}")
        print(f"📊 输出目录: {self.output_dir}")
    
    def _get_project_root(self) -> Path:
        """智能获取项目根目录"""
        current_dir = Path.cwd()
        if current_dir.name == "enhancement":
            return current_dir.parent.parent.parent
        elif current_dir.name == "digital_meter_detection":
            return current_dir.parent.parent
        elif current_dir.name == "scripts":
            return current_dir.parent
        else:
            return current_dir
    
    def setup_output_dirs(self, output_dir: Path | None):
        """设置输出目录"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 输出目录结构
        if output_dir is None:
            self.output_root = self.project_root / "outputs" / "digital_enhancement"
        else:
            self.output_root = output_dir
        self.output_dir = self.output_root / f"enhancement_{self.timestamp}"
        self.original_dir = self.output_dir / "1_original"
        self.enhanced_dir = self.output_dir / "2_enhanced" 
        self.comparison_dir = self.output_dir / "3_comparison"
        self.analysis_dir = self.output_dir / "4_analysis"
        
        # 创建目录
        for dir_path in [self.original_dir, self.enhanced_dir, self.comparison_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def remove_glare_and_reflections(self, image: np.ndarray) -> np.ndarray:
        """去除反光和反射"""
        # 转换为LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # 检测高亮区域（反光）
        _, glare_mask = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)
        
        # 使用形态学操作优化mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel)
        
        # 对反光区域进行修复
        if np.sum(glare_mask) > 0:
            result = cv2.inpaint(image, glare_mask, 3, cv2.INPAINT_TELEA)
        else:
            result = image.copy()
        
        return result
    
    def normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """光照归一化"""
        # 转换为灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 使用高斯滤波估计背景光照
        blurred = cv2.GaussianBlur(gray, (51, 51), 0)
        
        # 计算归一化后的图像
        normalized = np.zeros_like(gray, dtype=np.float32)
        normalized = cv2.divide(gray.astype(np.float32), blurred.astype(np.float32) + 1e-6)
        
        # 将结果缩放到0-255范围
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        
        # 如果原图是彩色的，应用到所有通道
        if len(image.shape) == 3:
            result = image.copy().astype(np.float32)
            for i in range(3):
                channel = image[:, :, i].astype(np.float32)
                blurred_channel = cv2.GaussianBlur(channel, (51, 51), 0)
                result[:, :, i] = cv2.divide(channel, blurred_channel + 1e-6)
            
            result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
            result = result.astype(np.uint8)
        else:
            result = normalized
        
        return result
    
    def enhance_contrast_clahe(self, image: np.ndarray) -> np.ndarray:
        """使用CLAHE增强对比度"""
        if len(image.shape) == 3:
            # 转换为LAB颜色空间
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # 对L通道应用CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            # 重新组合
            lab[:, :, 0] = l_channel
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # 灰度图像直接应用CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            result = clahe.apply(image)
        
        return result
    
    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """图像锐化"""
        # 使用拉普拉斯算子进行锐化
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 拉普拉斯锐化核
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        sharpened = cv2.filter2D(gray, -1, kernel)
        
        # 如果原图是彩色的，将锐化效果应用到原图
        if len(image.shape) == 3:
            # 将锐化后的灰度图转换为三通道
            sharpened_3ch = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            # 混合原图和锐化图
            result = cv2.addWeighted(image, 0.7, sharpened_3ch, 0.3, 0)
        else:
            result = sharpened
        
        return result
    
    def extract_digital_region(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """提取数字显示区域"""
        # 转换为灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 使用多种阈值方法
        # 1. Otsu阈值
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. 自适应阈值
        thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
        
        # 3. 反向阈值（对于暗背景亮文字）
        _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 形态学操作清理噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        thresh_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
        thresh_adaptive = cv2.morphologyEx(thresh_adaptive, cv2.MORPH_CLOSE, kernel)
        thresh_inv = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, kernel)
        
        # 组合多种阈值结果
        combined = cv2.bitwise_or(thresh_otsu, thresh_adaptive)
        combined = cv2.bitwise_or(combined, thresh_inv)
        
        return combined, gray
    
    def filter_digit_contours(self, binary_image: np.ndarray) -> List[Tuple]:
        """过滤出数字轮廓"""
        # 查找轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤条件
        height, width = binary_image.shape
        min_area = (height * width) * 0.001  # 最小面积
        max_area = (height * width) * 0.3    # 最大面积
        min_aspect_ratio = 0.1
        max_aspect_ratio = 3.0
        
        digit_contours = []
        
        for contour in contours:
            # 计算轮廓属性
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            
            # 计算边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue
            
            # 计算填充率
            rect_area = w * h
            fill_ratio = area / rect_area if rect_area > 0 else 0
            
            if fill_ratio < 0.1:  # 过滤掉太稀疏的轮廓
                continue
            
            digit_contours.append((contour, x, y, w, h, area))
        
        # 按x坐标排序（从左到右）
        digit_contours.sort(key=lambda x: x[1])
        
        return digit_contours
    
    def enhance_single_image(self, image: np.ndarray, method: str = "comprehensive") -> Dict:
        """增强单张图像"""
        results = {}
        
        # 保存原图
        results['original'] = image.copy()
        
        if method == "comprehensive" or method == "all":
            # 综合增强流程
            
            # 步骤1: 去除反光
            step1 = self.remove_glare_and_reflections(image)
            results['step1_deglare'] = step1
            
            # 步骤2: 光照归一化
            step2 = self.normalize_illumination(step1)
            results['step2_illumination'] = step2
            
            # 步骤3: 对比度增强
            step3 = self.enhance_contrast_clahe(step2)
            results['step3_contrast'] = step3
            
            # 步骤4: 图像锐化
            step4 = self.sharpen_image(step3)
            results['step4_sharpen'] = step4
            
            # 步骤5: 数字区域提取
            binary, gray = self.extract_digital_region(step4)
            results['step5_binary'] = binary
            results['step5_gray'] = gray
            
            # 步骤6: 轮廓过滤
            digit_contours = self.filter_digit_contours(binary)
            results['digit_contours'] = digit_contours
            
            # 最终结果
            results['final'] = step4
            
        elif method == "deglare_only":
            # 仅去反光
            enhanced = self.remove_glare_and_reflections(image)
            results['final'] = enhanced
            
        elif method == "contrast_only":
            # 仅对比度增强
            enhanced = self.enhance_contrast_clahe(image)
            results['final'] = enhanced
            
        elif method == "sharpen_only":
            # 仅锐化
            enhanced = self.sharpen_image(image)
            results['final'] = enhanced
        
        return results
    
    def create_comparison_visualization(self, results: Dict, output_path: Path):
        """创建对比可视化"""
        # 创建对比图
        if 'step1_deglare' in results:
            # 详细步骤对比
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle('Digital Display Enhancement Process', fontsize=16, fontweight='bold')
            
            # 原图
            axes[0, 0].imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # 去反光
            axes[0, 1].imshow(cv2.cvtColor(results['step1_deglare'], cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title('Glare Removal')
            axes[0, 1].axis('off')
            
            # 光照归一化
            if len(results['step2_illumination'].shape) == 3:
                axes[0, 2].imshow(cv2.cvtColor(results['step2_illumination'], cv2.COLOR_BGR2RGB))
            else:
                axes[0, 2].imshow(results['step2_illumination'], cmap='gray')
            axes[0, 2].set_title('Illumination Normalization')
            axes[0, 2].axis('off')
            
            # 对比度增强
            if len(results['step3_contrast'].shape) == 3:
                axes[0, 3].imshow(cv2.cvtColor(results['step3_contrast'], cv2.COLOR_BGR2RGB))
            else:
                axes[0, 3].imshow(results['step3_contrast'], cmap='gray')
            axes[0, 3].set_title('Contrast Enhancement')
            axes[0, 3].axis('off')
            
            # 图像锐化
            if len(results['step4_sharpen'].shape) == 3:
                axes[1, 0].imshow(cv2.cvtColor(results['step4_sharpen'], cv2.COLOR_BGR2RGB))
            else:
                axes[1, 0].imshow(results['step4_sharpen'], cmap='gray')
            axes[1, 0].set_title('Image Sharpening')
            axes[1, 0].axis('off')
            
            # 二值化
            axes[1, 1].imshow(results['step5_binary'], cmap='gray')
            axes[1, 1].set_title('Digit Extraction')
            axes[1, 1].axis('off')
            
            # 轮廓检测
            contour_img = results['original'].copy()
            if 'digit_contours' in results:
                for contour_data in results['digit_contours']:
                    contour, x, y, w, h, area = contour_data
                    cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            axes[1, 2].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
            axes[1, 2].set_title('Digit Localization')
            axes[1, 2].axis('off')
            
            # 最终结果
            if len(results['final'].shape) == 3:
                axes[1, 3].imshow(cv2.cvtColor(results['final'], cv2.COLOR_BGR2RGB))
            else:
                axes[1, 3].imshow(results['final'], cmap='gray')
            axes[1, 3].set_title('Final Result')
            axes[1, 3].axis('off')
            
        else:
            # 简单前后对比
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle('Digital Display Enhancement Comparison', fontsize=16, fontweight='bold')
            
            # 原图
            axes[0].imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # 增强后
            if len(results['final'].shape) == 3:
                axes[1].imshow(cv2.cvtColor(results['final'], cv2.COLOR_BGR2RGB))
            else:
                axes[1].imshow(results['final'], cmap='gray')
            axes[1].set_title('Enhanced Image')
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def process_single_image(self, image_path: Union[str, Path], 
                           method: str = "comprehensive") -> Dict:
        """处理单张图像"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        print(f"🎨 增强图像: {image_path.name}")
        
        # 加载图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 执行增强
        results = self.enhance_single_image(image, method)
        
        # 保存结果
        image_name = image_path.stem
        
        # 保存原图
        original_path = self.original_dir / f"{image_name}_original.jpg"
        cv2.imwrite(str(original_path), results['original'])
        
        # 保存增强后的图像
        enhanced_path = self.enhanced_dir / f"{image_name}_enhanced.jpg"
        cv2.imwrite(str(enhanced_path), results['final'])
        
        # 保存二值化结果（如果有）
        if 'step5_binary' in results:
            binary_path = self.enhanced_dir / f"{image_name}_binary.jpg"
            cv2.imwrite(str(binary_path), results['step5_binary'])
        
        # 创建对比可视化
        comparison_path = self.comparison_dir / f"{image_name}_comparison.png"
        self.create_comparison_visualization(results, comparison_path)
        
        return {
            'image_path': str(image_path),
            'image_name': image_name,
            'original_path': str(original_path),
            'enhanced_path': str(enhanced_path),
            'comparison_path': str(comparison_path),
            'method': method,
            'digit_count': len(results.get('digit_contours', [])),
            'timestamp': datetime.now().isoformat()
        }
    
    def process_batch(self, input_path: Union[str, Path], 
                     method: str = "comprehensive") -> List[Dict]:
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
        
        print(f"🎨 开始批量增强，共 {len(image_files)} 张图像")
        
        all_results = []
        
        # 处理每张图像
        for image_file in tqdm(image_files, desc="增强图像"):
            try:
                result = self.process_single_image(image_file, method)
                all_results.append(result)
            except Exception as e:
                print(f"⚠️  处理 {image_file.name} 时出错: {e}")
                continue
        
        return all_results
    
    def save_results(self, all_results: List[Dict]):
        """保存处理结果"""
        print("\n💾 保存增强结果...")
        
        # 保存详细结果
        results_file = self.output_dir / "enhancement_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 生成总结报告
        summary = {
            'total_images': len(all_results),
            'successful_enhancements': len(all_results),
            'output_directories': {
                'original': str(self.original_dir),
                'enhanced': str(self.enhanced_dir),
                'comparison': str(self.comparison_dir),
                'analysis': str(self.analysis_dir)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = self.output_dir / "enhancement_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self._generate_markdown_report(summary, all_results)
        
        print(f"✅ 结果保存完成:")
        print(f"  - 详细结果: {results_file}")
        print(f"  - 增强总结: {summary_file}")
    
    def _generate_markdown_report(self, summary: Dict, all_results: List[Dict]):
        """生成Markdown格式报告"""
        md_content = f"""# Digital Display Enhancement Report

## Processing Summary
- **Total Images**: {summary['total_images']}
- **Successfully Enhanced**: {summary['successful_enhancements']}
- **Processing Time**: {summary['timestamp'][:19].replace('T', ' ')}

## Output Directories
- **Original Images**: `1_original/`
- **Enhanced Images**: `2_enhanced/`
- **Comparison Images**: `3_comparison/`
- **Analysis Results**: `4_analysis/`

## Processing Methods
This enhancement uses a comprehensive processing pipeline:

1. **Glare Removal**: Detect and repair high-brightness reflection areas
2. **Illumination Normalization**: Balance image illumination distribution
3. **Contrast Enhancement**: Use CLAHE algorithm to enhance local contrast
4. **Image Sharpening**: Use Laplacian operator to enhance edges
5. **Digit Extraction**: Multi-threshold methods to extract digit regions
6. **Contour Filtering**: Filter digit contours based on geometric features

## Technical Features
- **Multi-step Processing**: Specialized optimization for LCD display characteristics
- **Reflection Handling**: Effectively remove LCD display reflection issues
- **Contrast Optimization**: Enhance contrast between digits and background
- **Edge Enhancement**: Improve digit boundary clarity

## File Description
Each image generates the following files:
- `*_original.jpg`: Original image
- `*_enhanced.jpg`: Final enhanced image
- `*_binary.jpg`: Digit binarized image
- `*_comparison.png`: Processing steps comparison chart

---
*Report generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        md_file = self.output_dir / "enhancement_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='液晶数字屏幕增强脚本')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像文件或目录路径')
    parser.add_argument('--method', type=str, default='comprehensive',
                       choices=['comprehensive', 'deglare_only', 'contrast_only', 'sharpen_only'],
                       help='增强方法 (默认: comprehensive)')
    parser.add_argument('--output', type=str, default='outputs/digital_enhancement',
                       help='自定义输出目录')
    
    args = parser.parse_args()
    
    try:
        
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = None
        
        # 创建增强器
        enhancer = DigitalDisplayEnhancer(output_dir=output_dir)

        # 如果指定了自定义输出目录
        if output_dir:
            enhancer.output_dir = output_dir
            enhancer.original_dir = output_dir / "1_original"
            enhancer.enhanced_dir = output_dir / "2_enhanced"
            enhancer.comparison_dir = output_dir / "3_comparison"
            enhancer.analysis_dir = output_dir / "4_analysis"
            
            for dir_path in [enhancer.original_dir, enhancer.enhanced_dir, 
                           enhancer.comparison_dir, enhancer.analysis_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
        
        
        
        # 批量处理
        results = enhancer.process_batch(args.input, args.method)
        
        # 保存结果
        enhancer.save_results(results)
        
        # 输出统计信息
        print(f"\n🎉 增强完成!")
        print(f"📊 处理图像: {len(results)} 张")
        print(f"📁 结果目录: {enhancer.output_dir}")
        print(f"🎨 增强方法: {args.method}")
        
    except Exception as e:
        print(f"❌ 增强过程出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 