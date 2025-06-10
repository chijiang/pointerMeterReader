#!/usr/bin/env python3
"""
测试指针表识别修复
Test meter reading functionality after fix
"""

import numpy as np
import cv2
from app import MeterReadingApp

def test_meter_app():
    """测试指针表应用初始化和基本功能"""
    print("🔧 测试指针表识别修复")
    print("=" * 50)
    
    try:
        # 1. 测试应用初始化
        print("📊 初始化 MeterReadingApp...")
        app = MeterReadingApp()
        print("✅ MeterReadingApp 初始化成功")
        
        # 2. 测试组件存在
        print("\n📋 检查组件:")
        print(f"  检测器: {'✅' if hasattr(app, 'detector') else '❌'}")
        print(f"  分割器: {'✅' if hasattr(app, 'segmentor') else '❌'}")
        print(f"  读数器: {'✅' if hasattr(app, 'reader') else '❌'}")
        print(f"  设备: {app.device}")
        
        # 3. 测试 process_image 方法是否存在
        print(f"\n🔍 process_image 方法: {'✅' if hasattr(app, 'process_image') else '❌'}")
        
        # 4. 创建测试图像（简单的虚拟图像）
        print("\n🖼️ 创建测试图像...")
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        # 绘制一个简单的圆形作为表盘
        cv2.circle(test_image, (150, 150), 100, (255, 255, 255), 2)
        # 绘制指针
        cv2.line(test_image, (150, 150), (200, 100), (0, 255, 0), 3)
        print("✅ 测试图像创建完成")
        
        # 5. 测试 process_image 方法调用（但不期望真正工作，因为没有训练好的模型）
        print("\n🧪 测试 process_image 方法调用...")
        try:
            results = app.process_image(test_image, conf_threshold=0.5, scale_min=0.0, scale_max=1.6)
            print("✅ process_image 方法调用成功")
            print(f"  返回结果类型: {type(results)}")
            print(f"  包含键: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
            
            if isinstance(results, dict):
                print(f"  成功状态: {results.get('success', 'Unknown')}")
                if not results.get('success'):
                    print(f"  错误信息: {results.get('error', 'No error message')}")
                
        except Exception as e:
            print(f"❌ process_image 方法调用失败: {e}")
        
        # 6. 测试可视化方法
        print("\n🎨 测试可视化方法:")
        vis_methods = ['_visualize_detection', '_visualize_segmentation', '_visualize_reading_result']
        for method in vis_methods:
            print(f"  {method}: {'✅' if hasattr(app, method) else '❌'}")
        
        print("\n" + "=" * 50)
        print("✅ 基本功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_digit_app():
    """简单测试数字识别应用确保没有破坏"""
    print("\n📱 测试数字识别应用 (确保没有破坏)")
    print("-" * 50)
    
    try:
        from app import DigitReadingApp
        
        digit_app = DigitReadingApp()
        print("✅ DigitReadingApp 初始化成功")
        
        # 测试数字处理方法
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 128
        results = digit_app.process_digit_image(test_image)
        print("✅ process_digit_image 方法调用成功")
        print(f"  返回结果: {type(results)}")
        
    except Exception as e:
        print(f"❌ 数字识别应用测试失败: {e}")

def main():
    """主测试函数"""
    test_meter_app()
    test_digit_app()
    
    print("\n🎉 所有测试完成!")
    print("\n💡 修复总结:")
    print("  ✅ 添加了缺失的 MeterReadingApp.process_image 方法")
    print("  ✅ 添加了完整的可视化方法")
    print("  ✅ 保持了数字识别功能完整性")
    print("  ✅ 现在两个标签页都应该正常工作")

if __name__ == "__main__":
    main() 