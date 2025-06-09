#!/usr/bin/env python3
"""
液晶数字显示增强演示脚本

这个脚本提供一个简单的方式来测试液晶屏数字增强功能，
包含从单张图像到批量处理的完整演示。

使用示例:
    python demo_enhancement.py
    
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

from scripts.digital_meter_detection.enhancement.digital_display_enhancer import DigitalDisplayEnhancer

def create_sample_images():
    """创建一些示例图像用于测试"""
    print("🎨 Creating sample images for testing...")
    
    # 创建示例目录
    sample_dir = project_root / "data" / "enhancement_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 示例1: 模拟液晶屏数字显示（清晰）
    img1 = np.zeros((100, 300, 3), dtype=np.uint8)
    img1[:] = (30, 30, 30)  # 深灰背景
    
    # 绘制数字区域
    cv2.rectangle(img1, (20, 20), (280, 80), (0, 50, 0), -1)  # 液晶屏背景
    cv2.putText(img1, "123.45", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    
    cv2.imwrite(str(sample_dir / "clear_display.jpg"), img1)
    
    # 示例2: 模拟反光问题
    img2 = img1.copy()
    # 添加反光效果
    overlay = np.zeros_like(img2)
    cv2.ellipse(overlay, (150, 50), (80, 30), 0, 0, 360, (255, 255, 255), -1)
    img2 = cv2.addWeighted(img2, 0.7, overlay, 0.3, 0)
    
    cv2.imwrite(str(sample_dir / "glare_display.jpg"), img2)
    
    # 示例3: 模拟对比度低的问题
    img3 = np.zeros((100, 300, 3), dtype=np.uint8)
    img3[:] = (60, 60, 60)  # 中灰背景
    cv2.rectangle(img3, (20, 20), (280, 80), (50, 70, 50), -1)
    cv2.putText(img3, "987.65", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 120, 80), 2)
    
    cv2.imwrite(str(sample_dir / "low_contrast_display.jpg"), img3)
    
    # 示例4: 模拟模糊显示
    img4 = img1.copy()
    img4 = cv2.GaussianBlur(img4, (5, 5), 0)
    
    cv2.imwrite(str(sample_dir / "blurry_display.jpg"), img4)
    
    print(f"✅ Sample images created in: {sample_dir}")
    return sample_dir

def demo_single_image_enhancement():
    """演示单张图像增强"""
    print("\n🔍 Demo: Single Image Enhancement")
    print("=" * 50)
    
    # 创建示例图像
    sample_dir = create_sample_images()
    
    # 选择一个测试图像
    test_image = sample_dir / "glare_display.jpg"
    
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    # 创建增强器
    enhancer = DigitalDisplayEnhancer()
    
    # 处理图像
    print(f"🎨 Processing: {test_image.name}")
    result = enhancer.process_single_image(test_image, method="comprehensive")
    
    print(f"✅ Enhancement completed!")
    print(f"📁 Results saved to: {enhancer.output_dir}")
    print(f"🎯 Detected digits: {result['digit_count']}")
    
    return result

def demo_batch_enhancement():
    """演示批量图像增强"""
    print("\n📦 Demo: Batch Image Enhancement")
    print("=" * 50)
    
    # 创建示例图像
    sample_dir = create_sample_images()
    
    # 创建增强器
    enhancer = DigitalDisplayEnhancer()
    
    # 批量处理
    print(f"🎨 Processing all images in: {sample_dir}")
    results = enhancer.process_batch(sample_dir, method="comprehensive")
    
    # 保存结果
    enhancer.save_results(results)
    
    print(f"✅ Batch enhancement completed!")
    print(f"📊 Processed {len(results)} images")
    print(f"📁 Results saved to: {enhancer.output_dir}")
    
    # 显示统计信息
    total_digits = sum(result['digit_count'] for result in results)
    print(f"🎯 Total detected digits: {total_digits}")
    
    return results

def demo_different_methods():
    """演示不同的增强方法"""
    print("\n🔬 Demo: Different Enhancement Methods")
    print("=" * 50)
    
    # 创建示例图像
    sample_dir = create_sample_images()
    test_image = sample_dir / "glare_display.jpg"
    
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    methods = ["comprehensive", "deglare_only", "contrast_only", "sharpen_only"]
    
    for method in methods:
        print(f"\n🎨 Testing method: {method}")
        
        # 创建增强器
        enhancer = DigitalDisplayEnhancer()
        
        # 处理图像
        result = enhancer.process_single_image(test_image, method=method)
        
        print(f"  ✅ Method '{method}' completed")
        print(f"  📁 Results: {enhancer.output_dir}")
        print(f"  🎯 Digits: {result['digit_count']}")

def demo_real_data():
    """演示使用真实数据"""
    print("\n📷 Demo: Real Dataset Enhancement")
    print("=" * 50)
    
    # 检查是否有真实数据
    real_data_dir = project_root / "data" / "digital_meters"
    
    if not real_data_dir.exists() or not any(real_data_dir.glob("*.jpg")):
        print("⚠️  No real dataset found in data/digital_meters/")
        print("   Creating sample data instead...")
        demo_batch_enhancement()
        return
    
    # 使用真实数据
    print(f"📂 Found real dataset: {real_data_dir}")
    
    # 创建增强器
    enhancer = DigitalDisplayEnhancer()
    
    # 只处理前5张图像作为演示
    image_files = list(real_data_dir.glob("*.jpg"))[:5]
    
    if not image_files:
        print("❌ No JPG images found in the dataset")
        return
    
    print(f"🎨 Processing {len(image_files)} real images...")
    
    results = []
    for image_file in image_files:
        try:
            result = enhancer.process_single_image(image_file, method="comprehensive")
            results.append(result)
            print(f"  ✅ Processed: {image_file.name}")
        except Exception as e:
            print(f"  ❌ Error processing {image_file.name}: {e}")
    
    # 保存结果
    if results:
        enhancer.save_results(results)
        print(f"\n✅ Real data enhancement completed!")
        print(f"📊 Successfully processed: {len(results)} images")
        print(f"📁 Results saved to: {enhancer.output_dir}")

def interactive_demo():
    """交互式演示菜单"""
    while True:
        print("\n" + "="*60)
        print("🎨 Digital Display Enhancement Demo")
        print("="*60)
        print("1. Single Image Enhancement Demo")
        print("2. Batch Enhancement Demo") 
        print("3. Different Methods Comparison")
        print("4. Real Dataset Enhancement")
        print("5. View Project Structure")
        print("0. Exit")
        print("-"*60)
        
        choice = input("👆 Select an option (0-5): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            demo_single_image_enhancement()
        elif choice == "2":
            demo_batch_enhancement()
        elif choice == "3":
            demo_different_methods()
        elif choice == "4":
            demo_real_data()
        elif choice == "5":
            show_project_structure()
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
│   └── enhancement_samples/     # Generated sample images
├── outputs/
│   └── digital_enhancement/     # Enhancement results
│       └── enhancement_YYYYMMDD_HHMMSS/
│           ├── 1_original/      # Original images
│           ├── 2_enhanced/      # Enhanced images
│           ├── 3_comparison/    # Step-by-step comparisons
│           └── 4_analysis/      # Analysis reports
└── scripts/
    └── digital_meter_detection/
        └── enhancement/
            ├── digital_display_enhancer.py  # Main enhancement script
            └── demo_enhancement.py          # This demo script
    """
    
    print(structure)
    
    # 检查关键目录
    data_dir = project_root / "data" / "digital_meters"
    outputs_dir = project_root / "outputs"
    
    print("\n📊 Directory Status:")
    print(f"  Real dataset: {'✅ Found' if data_dir.exists() and any(data_dir.glob('*.jpg')) else '❌ Not found'}")
    print(f"  Outputs: {'✅ Ready' if outputs_dir.exists() else '🔧 Will be created'}")

def main():
    """主函数"""
    print("🎨 Digital Display Enhancement Demo")
    print("Specialized for LCD digit enhancement with glare removal")
    print("and contrast improvement capabilities.")
    
    # 检查依赖
    try:
        import cv2
        import matplotlib.pyplot as plt
        import numpy as np
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages: pip install opencv-python matplotlib numpy")
        sys.exit(1)
    
    # 启动交互式演示
    interactive_demo()

if __name__ == "__main__":
    main() 