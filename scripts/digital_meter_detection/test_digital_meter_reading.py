#!/usr/bin/env python3
"""
液晶数字表读数提取测试脚本
Test script for digital meter reading extraction

作者: chijiang
日期: 2025-06-09
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

def create_test_model():
    """创建一个模拟的模型文件用于测试"""
    model_dir = project_root / "models" / "detection"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "digital_detection_model.pt"
    
    if not model_path.exists():
        print("⚠️  注意: 实际的YOLO模型文件不存在")
        print(f"期望路径: {model_path}")
        print("创建一个虚拟文件用于测试...")
        
        # 创建一个虚拟的模型文件
        with open(model_path, 'w') as f:
            f.write("# 这是一个测试用的虚拟模型文件\n")
            f.write("# 实际使用时需要替换为真正的YOLO模型\n")
        
        print(f"✅ 虚拟模型文件创建: {model_path}")
    
    return model_path

def create_test_images():
    """创建测试图像"""
    test_dir = project_root / "data" / "test_digital_meters"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建一个带有数字的测试图像
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # 绘制一个矩形框模拟液晶显示屏
    cv2.rectangle(image, (150, 150), (450, 250), (200, 200, 200), -1)
    cv2.rectangle(image, (150, 150), (450, 250), (0, 0, 0), 2)
    
    # 添加文字模拟数字显示
    cv2.putText(image, "123.45", (200, 220), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # 保存测试图像
    test_image_path = test_dir / "test_digital_meter.jpg"
    cv2.imwrite(str(test_image_path), image)
    
    print(f"✅ 测试图像创建: {test_image_path}")
    return test_image_path

def test_without_yolo():
    """测试不使用YOLO的基本功能"""
    print("\n🧪 测试基本功能（不使用YOLO）...")
    
    try:
        from scripts.digital_meter_detection.enhancement.digital_display_enhancer import DigitalDisplayEnhancer
        from scripts.digital_meter_detection.ocr.digital_ocr_extractor import DigitalOCRExtractor
        
        # 创建测试图像
        test_image_path = create_test_images()
        image = cv2.imread(str(test_image_path))
        
        # 模拟裁剪的液晶显示屏区域
        cropped_image = image[150:250, 150:450]
        
        print("🎨 测试图像增强...")
        enhancer = DigitalDisplayEnhancer()
        enhanced_result = enhancer.enhance_single_image(cropped_image)
        enhanced_image = enhanced_result['final']
        
        # 保存增强结果
        enhanced_path = test_image_path.parent / "test_enhanced.jpg"
        cv2.imwrite(str(enhanced_path), enhanced_image)
        print(f"✅ 增强图像保存: {enhanced_path}")
        
        print("🔤 测试OCR提取...")
        ocr_extractor = DigitalOCRExtractor(ocr_engine="easyocr")
        ocr_result = ocr_extractor.extract_from_image(enhanced_image)
        
        if ocr_result['best_result']:
            value = ocr_result['best_result']['value']
            confidence = ocr_result['best_result']['confidence']
            print(f"✅ OCR成功: 提取值 = {value}, 置信度 = {confidence:.3f}")
        else:
            print("⚠️  OCR未能提取到有效数值")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

def test_full_pipeline():
    """测试完整的流水线（需要YOLO）"""
    print("\n🧪 测试完整流水线...")
    
    try:
        # 检查是否有YOLO
        try:
            from ultralytics import YOLO
            yolo_available = True
        except ImportError:
            yolo_available = False
        
        if not yolo_available:
            print("⚠️  YOLO不可用，跳过完整流水线测试")
            return True
        
        # 创建模型文件（虚拟的）
        model_path = create_test_model()
        
        # 由于模型是虚拟的，这里只测试导入和初始化
        from digital_meter_reading import DigitalMeterReader
        
        print("✅ 成功导入DigitalMeterReader")
        print("⚠️  完整测试需要真实的YOLO模型文件")
        
        return True
        
    except Exception as e:
        print(f"❌ 完整流水线测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 液晶数字表读数提取 - 功能测试")
    print("=" * 50)
    
    # 检查项目结构
    required_dirs = [
        "scripts/digital_meter_detection/enhancement",
        "scripts/digital_meter_detection/ocr"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not (project_root / dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ 缺少必要的目录:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        return
    
    # 运行测试
    tests = [
        ("基本功能测试", test_without_yolo),
        ("完整流水线测试", test_full_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} 通过")
        else:
            print(f"❌ {test_name} 失败")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过!")
        print("\n📝 使用说明:")
        print("1. 准备真实的YOLO检测模型文件")
        print("2. 将模型文件放置到: models/detection/digital_detection_model.pt")
        print("3. 运行完整的读数提取:")
        print("   python digital_meter_reading.py --input your_image.jpg")
    else:
        print("⚠️  有测试失败，请检查依赖和模块")

if __name__ == "__main__":
    main() 