#!/usr/bin/env python3
"""
液晶数字OCR演示脚本

演示如何使用OCR提取器从增强后的液晶屏图像中提取数字。
包含从单张图像到批量处理的完整演示。

使用示例:
    python demo_ocr.py
    
作者: chijiang
日期: 2025-06-09
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from scripts.digital_meter_detection.ocr.digital_ocr_extractor import DigitalOCRExtractor
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保OCR提取器脚本存在")
    sys.exit(1)

def create_sample_digital_images():
    """创建一些示例数字图像用于OCR测试"""
    print("🔢 Creating sample digital images for OCR testing...")
    
    # 创建示例目录
    sample_dir = project_root / "data" / "ocr_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 示例1: 清晰的数字显示
    img1 = np.zeros((80, 200, 3), dtype=np.uint8)
    img1[:] = (240, 240, 240)  # 浅灰背景
    cv2.putText(img1, "123.45", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.imwrite(str(sample_dir / "clear_digits.jpg"), img1)
    
    # 示例2: 不同格式的数字
    img2 = np.zeros((80, 250, 3), dtype=np.uint8)
    img2[:] = (250, 250, 250)  # 白色背景
    cv2.putText(img2, "987.321", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2)
    cv2.imwrite(str(sample_dir / "decimal_digits.jpg"), img2)
    
    # 示例3: 整数
    img3 = np.zeros((60, 150, 3), dtype=np.uint8)
    img3[:] = (245, 245, 245)  # 几乎白色背景
    cv2.putText(img3, "42567", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.imwrite(str(sample_dir / "integer_digits.jpg"), img3)
    
    # 示例4: 带负号的数字
    img4 = np.zeros((70, 180, 3), dtype=np.uint8)
    img4[:] = (235, 235, 235)  # 浅色背景
    cv2.putText(img4, "-15.67", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.imwrite(str(sample_dir / "negative_digits.jpg"), img4)
    
    # 示例5: 模拟低质量图像
    img5 = np.zeros((70, 200, 3), dtype=np.uint8)
    img5[:] = (220, 220, 220)  # 中等灰度背景
    cv2.putText(img5, "0.0098", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 1)
    # 添加一些噪声
    noise = np.random.randint(0, 30, img5.shape, dtype=np.uint8)
    img5 = cv2.add(img5, noise)
    cv2.imwrite(str(sample_dir / "noisy_digits.jpg"), img5)
    
    print(f"✅ Sample digital images created in: {sample_dir}")
    return sample_dir

def demo_single_ocr():
    """演示单张图像OCR"""
    print("\n🔤 Demo: Single Image OCR Extraction")
    print("=" * 50)
    
    # 创建示例图像
    sample_dir = create_sample_digital_images()
    
    # 选择一个测试图像
    test_image = sample_dir / "clear_digits.jpg"
    
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    try:
        # 创建OCR提取器
        extractor = DigitalOCRExtractor(ocr_engine="easyocr")
        
        # 处理图像
        print(f"🔤 Processing: {test_image.name}")
        result = extractor.process_single_image(test_image)
        
        print(f"✅ OCR extraction completed!")
        print(f"📁 Results saved to: {extractor.output_dir}")
        print(f"🔢 Extracted value: {result['extracted_value']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print(f"🎯 Total detections: {result['total_detections']}")
        
        return result
        
    except ImportError as e:
        print(f"❌ OCR library not available: {e}")
        print("Please install EasyOCR: pip install easyocr")
        return None

def demo_batch_ocr():
    """演示批量OCR提取"""
    print("\n📦 Demo: Batch OCR Extraction")
    print("=" * 50)
    
    # 创建示例图像
    sample_dir = create_sample_digital_images()
    
    try:
        # 创建OCR提取器
        extractor = DigitalOCRExtractor(ocr_engine="easyocr")
        
        # 批量处理
        print(f"🔤 Processing all images in: {sample_dir}")
        results = extractor.process_batch(sample_dir)
        
        # 保存结果
        extractor.save_results(results)
        
        print(f"✅ Batch OCR extraction completed!")
        print(f"📊 Processed {len(results)} images")
        print(f"📁 Results saved to: {extractor.output_dir}")
        
        # 显示提取的数值
        successful_extractions = [r for r in results if r['extracted_value'] is not None]
        print(f"🎯 Successfully extracted {len(successful_extractions)} values:")
        
        for result in successful_extractions:
            print(f"  - {result['image_name']}: {result['extracted_value']} (confidence: {result['confidence']:.3f})")
        
        return results
        
    except ImportError as e:
        print(f"❌ OCR library not available: {e}")
        print("Please install EasyOCR: pip install easyocr")
        return None

def demo_different_engines():
    """演示不同OCR引擎的效果"""
    print("\n🔬 Demo: Different OCR Engines Comparison")
    print("=" * 50)
    
    # 创建示例图像
    sample_dir = create_sample_digital_images()
    test_image = sample_dir / "clear_digits.jpg"
    
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    engines = ["easyocr", "paddleocr", "tesseract"]
    
    for engine in engines:
        print(f"\n🔧 Testing OCR engine: {engine}")
        
        try:
            # 创建OCR提取器
            extractor = DigitalOCRExtractor(ocr_engine=engine)
            
            # 处理图像
            result = extractor.process_single_image(test_image)
            
            print(f"  ✅ Engine '{engine}' completed")
            print(f"  📁 Results: {extractor.output_dir}")
            print(f"  🔢 Value: {result['extracted_value']}")
            print(f"  📊 Confidence: {result['confidence']:.3f}")
            
        except ImportError as e:
            print(f"  ❌ Engine '{engine}' not available: {e}")
        except Exception as e:
            print(f"  ❌ Error with engine '{engine}': {e}")

def demo_enhanced_images():
    """演示对增强图像的OCR"""
    print("\n🎨 Demo: OCR on Enhanced Images")
    print("=" * 50)
    
    # 检查是否有增强后的图像
    enhanced_dir = project_root / "outputs" / "digital_enhancement"
    
    if not enhanced_dir.exists():
        print("⚠️  No enhanced images found in outputs/digital_enhancement/")
        print("   Running OCR on sample images instead...")
        demo_batch_ocr()
        return
    
    # 查找最新的增强结果
    enhancement_dirs = list(enhanced_dir.glob("enhancement_*"))
    if not enhancement_dirs:
        print("⚠️  No enhancement results found")
        print("   Running OCR on sample images instead...")
        demo_batch_ocr()
        return
    
    # 使用最新的增强结果
    latest_enhancement = max(enhancement_dirs, key=lambda x: x.name)
    enhanced_images_dir = latest_enhancement / "2_enhanced"
    
    if not enhanced_images_dir.exists() or not any(enhanced_images_dir.glob("*.jpg")):
        print("⚠️  No enhanced images found")
        print("   Running OCR on sample images instead...")
        demo_batch_ocr()
        return
    
    print(f"📂 Found enhanced images: {enhanced_images_dir}")
    
    try:
        # 创建OCR提取器
        extractor = DigitalOCRExtractor(ocr_engine="easyocr")
        
        # 只处理前3张增强图像作为演示
        image_files = list(enhanced_images_dir.glob("*_enhanced.jpg"))[:3]
        
        if not image_files:
            print("❌ No enhanced JPG images found")
            return
        
        print(f"🔤 Processing {len(image_files)} enhanced images...")
        
        results = []
        for image_file in image_files:
            try:
                result = extractor.process_single_image(image_file)
                results.append(result)
                print(f"  ✅ Processed: {image_file.name}")
                print(f"     Value: {result['extracted_value']}, Confidence: {result['confidence']:.3f}")
            except Exception as e:
                print(f"  ❌ Error processing {image_file.name}: {e}")
        
        # 保存结果
        if results:
            extractor.save_results(results)
            print(f"\n✅ Enhanced images OCR completed!")
            print(f"📊 Successfully processed: {len(results)} images")
            print(f"📁 Results saved to: {extractor.output_dir}")
        
    except ImportError as e:
        print(f"❌ OCR library not available: {e}")
        print("Please install EasyOCR: pip install easyocr")

def interactive_demo():
    """交互式演示菜单"""
    while True:
        print("\n" + "="*60)
        print("🔤 Digital OCR Extraction Demo")
        print("="*60)
        print("1. Single Image OCR Demo")
        print("2. Batch OCR Demo") 
        print("3. Different OCR Engines Comparison")
        print("4. OCR on Enhanced Images")
        print("5. View Project Structure")
        print("6. Check Dependencies")
        print("0. Exit")
        print("-"*60)
        
        choice = input("👆 Select an option (0-6): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            demo_single_ocr()
        elif choice == "2":
            demo_batch_ocr()
        elif choice == "3":
            demo_different_engines()
        elif choice == "4":
            demo_enhanced_images()
        elif choice == "5":
            show_project_structure()
        elif choice == "6":
            check_dependencies()
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\n📝 Press Enter to continue...")

def show_project_structure():
    """显示项目结构"""
    print("\n📁 Project Structure:")
    print("=" * 50)
    
    structure = """
pointMeterDetection/
├── data/
│   ├── digital_meters/          # Real dataset (JPG images)
│   ├── enhancement_samples/     # Generated enhancement samples
│   └── ocr_samples/             # Generated OCR test samples
├── outputs/
│   ├── digital_enhancement/     # Enhancement results
│   └── digital_ocr/             # OCR extraction results
│       └── ocr_results_YYYYMMDD_HHMMSS/
│           ├── extracted_text/  # Individual OCR results (JSON)
│           ├── visualization/   # OCR detection visualizations
│           └── analysis/        # Summary reports
└── scripts/
    └── digital_meter_detection/
        ├── enhancement/         # Image enhancement scripts
        └── ocr/                 # OCR extraction scripts
            ├── digital_ocr_extractor.py  # Main OCR script
            └── demo_ocr.py               # This demo script
    """
    
    print(structure)
    
    # 检查关键目录
    data_dir = project_root / "data" / "digital_meters"
    outputs_dir = project_root / "outputs"
    enhanced_dir = project_root / "outputs" / "digital_enhancement"
    
    print("\n📊 Directory Status:")
    print(f"  Real dataset: {'✅ Found' if data_dir.exists() and any(data_dir.glob('*.jpg')) else '❌ Not found'}")
    print(f"  Outputs: {'✅ Ready' if outputs_dir.exists() else '🔧 Will be created'}")
    print(f"  Enhanced images: {'✅ Available' if enhanced_dir.exists() and any(enhanced_dir.glob('*/2_enhanced/*.jpg')) else '❌ Not found'}")

def check_dependencies():
    """检查依赖项"""
    print("\n🔧 Checking Dependencies:")
    print("=" * 50)
    
    dependencies = {
        "OpenCV": ("cv2", "pip install opencv-python"),
        "NumPy": ("numpy", "pip install numpy"),
        "Matplotlib": ("matplotlib", "pip install matplotlib"),
        "EasyOCR": ("easyocr", "pip install easyocr"),
        "Pillow": ("PIL", "pip install pillow"),
        "Tesseract": ("pytesseract", "pip install pytesseract"),
        "tqdm": ("tqdm", "pip install tqdm")
    }
    
    for name, (module, install_cmd) in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name}: Available")
        except ImportError:
            print(f"❌ {name}: Not installed - {install_cmd}")
    
    print("\n📝 Recommended OCR engines:")
    print("  - EasyOCR: Best for general use, good accuracy")
    print("  - Tesseract: Fast, good for clear text")
    
    print("\n💡 Installation tips:")
    print("  - EasyOCR: Requires internet connection for first-time model download")
    print("  - Tesseract: May require additional system installation")

def main():
    """主函数"""
    print("🔤 Digital OCR Extraction Demo")
    print("Extract numerical values from LCD digital displays")
    print("with high accuracy using advanced OCR techniques.")
    
    # 启动交互式演示
    interactive_demo()

if __name__ == "__main__":
    main() 