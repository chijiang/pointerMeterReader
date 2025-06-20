#!/usr/bin/env python3
"""
LED颜色识别演示脚本

快速演示LED颜色识别功能的使用方法。
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

def create_demo_led_image():
    """创建一个演示用的LED图像"""
    # 创建黑色背景图像
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # 添加标题
    cv2.putText(img, "LED Status Panel Demo", (150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 绘制LED指示灯
    led_positions = [
        (150, 100, "Power", (0, 255, 0)),      # 绿色 - 电源
        (300, 100, "Network", (0, 255, 0)),    # 绿色 - 网络
        (450, 100, "Status", (0, 255, 0)),     # 绿色 - 状态
        
        (150, 200, "Error", (0, 0, 255)),      # 红色 - 错误
        (300, 200, "Warning", (0, 255, 255)), # 黄色 - 警告
        (450, 200, "Alarm", (0, 0, 255)),     # 红色 - 警报
        
        (150, 300, "Ready", (0, 255, 0)),      # 绿色 - 就绪
        (300, 300, "Standby", (0, 255, 255)), # 黄色 - 待机
        (450, 300, "Fault", (0, 0, 255)),     # 红色 - 故障
    ]
    
    for x, y, label, color in led_positions:
        # 绘制LED圆圈
        cv2.circle(img, (x, y), 15, color, -1)
        cv2.circle(img, (x, y), 15, (255, 255, 255), 2)  # 白色边框
        
        # 添加标签
        cv2.putText(img, label, (x-30, y+35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return img

def demo_led_detection():
    """演示LED颜色检测功能"""
    print("🚦 LED颜色识别演示")
    print("=" * 50)
    
    try:
        # 导入LED检测类
        from app import LEDColorApp
        
        # 创建演示图像
        print("📷 创建演示LED面板图像...")
        demo_image = create_demo_led_image()
        
        # 保存演示图像
        demo_path = "demo_led_panel.jpg"
        cv2.imwrite(demo_path, demo_image)
        print(f"✅ 演示图像已保存: {demo_path}")
        
        # 初始化LED检测应用
        print("\n🔧 初始化LED颜色检测应用...")
        led_app = LEDColorApp()
        
        # 处理图像
        print("🔍 开始LED颜色检测...")
        results = led_app.process_led_image(demo_image, conf_threshold=0.1)
        
        # 显示结果
        print("\n📊 检测结果:")
        print(f"   成功: {results['success']}")
        
        if results['success']:
            analysis = results['analysis']
            detections = results['detections']
            
            print(f"   检测到的LED数量: {len(detections)}")
            print(f"   系统状态: {analysis['status_summary']}")
            
            # 详细统计
            print(f"\n📈 颜色统计:")
            for color in ['green', 'red', 'yellow']:
                count = analysis['color_counts'][color]
                percentage = analysis['color_percentages'][color]
                print(f"   {color.upper()}: {count} 个 ({percentage:.1f}%)")
            
            # 生成可视化
            print(f"\n🎨 生成可视化结果...")
            
            # 检测可视化
            detection_vis = led_app.led_detector.visualize_detections(
                demo_image, detections, True, True)
            cv2.imwrite("demo_led_detections.jpg", detection_vis)
            print(f"   检测可视化已保存: demo_led_detections.jpg")
            
            # 状态覆盖
            status_overlay = led_app.led_detector.create_status_overlay(
                demo_image, analysis)
            cv2.imwrite("demo_led_status_overlay.jpg", status_overlay)
            print(f"   状态覆盖已保存: demo_led_status_overlay.jpg")
            
            # 操作建议
            print(f"\n💡 操作建议:")
            red_count = analysis['color_counts']['red']
            yellow_count = analysis['color_counts']['yellow']
            green_count = analysis['color_counts']['green']
            
            if red_count > 0:
                print(f"   🚨 发现 {red_count} 个红色LED，建议立即检查系统状态")
            if yellow_count > 0:
                print(f"   ⚠️ 发现 {yellow_count} 个黄色LED，建议关注系统警告")
            if green_count > 0 and red_count == 0 and yellow_count == 0:
                print(f"   ✅ 系统正常，所有LED均为绿色状态")
        else:
            print(f"   错误: {results.get('error', '未知错误')}")
        
        print(f"\n🎉 演示完成！")
        print(f"\n📋 生成的文件:")
        print(f"   - demo_led_panel.jpg (原始演示图像)")
        print(f"   - demo_led_detections.jpg (检测结果)")
        print(f"   - demo_led_status_overlay.jpg (状态分析)")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def demo_app_integration():
    """演示应用集成"""
    print(f"\n🌐 Web应用集成演示")
    print("=" * 30)
    
    print("📝 要在Web应用中使用LED颜色识别:")
    print("   1. 启动应用: python app.py")
    print("   2. 访问: http://localhost:7860")
    print("   3. 点击 '🚦 LED Status Detection' 标签页")
    print("   4. 上传包含LED的图像")
    print("   5. 调整检测参数")
    print("   6. 点击 '🚦 Analyze LED Status' 开始分析")
    
    print(f"\n🔧 模型训练:")
    print("   如果需要训练自定义模型:")
    print("   1. 准备数据: data/led_color/")
    print("   2. 运行训练: python run_led_color_training.py")
    print("   3. 部署模型: cp outputs/.../best.pt models/detection/led_color_model.pt")

def main():
    """主函数"""
    print("🚀 LED颜色识别系统演示")
    print("=" * 60)
    
    # 检查是否存在训练好的模型
    model_path = Path("models/detection/led_color_model.pt")
    if model_path.exists():
        print("✅ 发现训练好的LED颜色识别模型")
        demo_led_detection()
    else:
        print("⚠️ 未发现训练好的模型，将使用基础YOLO模型")
        print("💡 要获得最佳效果，请先训练LED颜色识别模型:")
        print("   python run_led_color_training.py")
        print()
        
        # 仍然可以演示基本功能
        demo_led_detection()
    
    demo_app_integration()
    
    print(f"\n🎓 演示总结:")
    print("   ✅ LED颜色识别功能已成功集成到应用中")
    print("   ✅ 支持绿色、红色、黄色LED检测")
    print("   ✅ 提供详细的状态分析和操作建议")
    print("   ✅ 包含完整的Web界面")

if __name__ == "__main__":
    main() 