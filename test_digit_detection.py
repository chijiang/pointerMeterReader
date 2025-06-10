#!/usr/bin/env python3
"""
测试LCD数字检测功能

这个脚本用于测试新添加的数字检测功能是否正常工作。
"""

import cv2
import numpy as np
from app import DigitDetector, DigitReadingApp
import os

def test_digit_detector():
    """测试数字检测器基本功能"""
    print("🧪 测试数字检测器...")
    
    # 创建一个简单的测试图像
    test_image = np.zeros((100, 300, 3), dtype=np.uint8)
    test_image.fill(255)  # 白色背景
    
    # 在图像上写一些数字
    cv2.putText(test_image, "123.45", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # 保存测试图像
    cv2.imwrite("test_digits.jpg", test_image)
    print("✅ 创建测试图像: test_digits.jpg")
    
    # 初始化检测器
    try:
        detector = DigitDetector("models/detection/digits_model.pt")
        print("✅ 数字检测器初始化成功")
    except Exception as e:
        print(f"❌ 数字检测器初始化失败: {e}")
        return False
    
    # 测试检测功能
    try:
        detections = detector.detect_digits(test_image, conf_threshold=0.1)
        print(f"✅ 检测功能正常，检测到 {len(detections)} 个对象")
        
        if detections:
            for i, det in enumerate(detections):
                print(f"  检测 {i+1}: {det['class']} (置信度: {det['confidence']:.3f})")
    except Exception as e:
        print(f"❌ 检测功能失败: {e}")
        return False
    
    return True

def test_digit_app():
    """测试数字读取应用"""
    print("\n🧪 测试数字读取应用...")
    
    try:
        app = DigitReadingApp()
        print("✅ 数字读取应用初始化成功")
    except Exception as e:
        print(f"❌ 数字读取应用初始化失败: {e}")
        return False
    
    # 创建测试图像
    test_image = np.zeros((100, 400, 3), dtype=np.uint8)
    test_image.fill(255)
    cv2.putText(test_image, "987.65", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    try:
        results = app.process_digit_image(test_image)
        print("✅ 图像处理功能正常")
        print(f"  成功: {results['success']}")
        print(f"  读数: {results['reading']}")
        print(f"  原始检测数量: {len(results['raw_detections'])}")
        print(f"  过滤后检测数量: {len(results['filtered_detections'])}")
        
        if results['error']:
            print(f"  错误信息: {results['error']}")
            
    except Exception as e:
        print(f"❌ 图像处理失败: {e}")
        return False
    
    return True

def test_gradio_components():
    """测试Gradio组件是否正常导入"""
    print("\n🧪 测试Gradio组件...")
    
    try:
        import gradio as gr
        print("✅ Gradio导入成功")
        
        # 测试创建接口
        from app import create_gradio_interface
        print("✅ 界面创建函数导入成功")
        
    except Exception as e:
        print(f"❌ Gradio组件测试失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始测试LCD数字检测功能\n")
    print("=" * 50)
    
    tests = [
        ("数字检测器", test_digit_detector),
        ("数字读取应用", test_digit_app),
        ("Gradio组件", test_gradio_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 测试: {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"✅ {test_name} 测试通过")
                passed += 1
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！LCD数字检测功能可以正常使用。")
        print("\n💡 下一步:")
        print("1. 训练您自己的数字检测模型:")
        print("   python scripts/start_digits_training.py")
        print("2. 启动应用:")
        print("   python app.py")
    else:
        print("⚠️ 部分测试失败，请检查错误信息并修复问题。")
    
    # 清理测试文件
    if os.path.exists("test_digits.jpg"):
        os.remove("test_digits.jpg")
        print("\n🧹 已清理测试文件")

if __name__ == "__main__":
    main() 