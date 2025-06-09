#!/usr/bin/env python3
"""
液晶数字表检测系统 - 便捷启动脚本

这个脚本提供了一个简单的命令行界面来运行各种液晶数字表检测功能。
"""

import os
import sys
import subprocess
from pathlib import Path

def get_project_root():
    """获取项目根目录"""
    current_dir = Path(__file__).parent
    # 从 scripts/digital_meter_detection/ 回到项目根目录
    return current_dir.parent.parent

def print_banner():
    """打印横幅"""
    print("=" * 60)
    print("🔍 液晶数字表检测系统")
    print("=" * 60)
    print()

def print_menu():
    """打印菜单"""
    print("请选择要执行的功能：")
    print()
    print("1. 📊 验证数据集")
    print("2. 🚀 训练模型（完整训练 - 200轮）")
    print("3. ⚡ 训练模型（快速演示 - 20轮）")
    print("4. 🎯 模型推理（需要提供模型路径）")
    print("5. 🎨 液晶屏数字增强演示")
    print("6. 🔤 OCR数字提取演示")
    print("7. 🎬 完整演示流程")
    print("8. 🔢 完整读数提取（检测+增强+OCR）")
    print("9. 🧪 功能测试")
    print("10. ❓ 查看帮助")
    print("0. 🚪 退出")
    print()

def run_command(command, description=""):
    """执行命令"""
    if description:
        print(f"▶️  {description}")
    
    print(f"💻 执行命令: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, cwd=get_project_root())
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
        return False
    except Exception as e:
        print(f"❌ 命令执行失败: {e}")
        return False

def validate_dataset():
    """验证数据集"""
    project_root = get_project_root()
    script_path = project_root / "scripts" / "digital_meter_detection" / "validation" / "validate_digital_meter_dataset.py"
    
    command = f"python {script_path} --dataset data/digital_meters"
    return run_command(command, "验证液晶数字表数据集")

def train_full_model():
    """训练完整模型"""
    project_root = get_project_root()
    script_path = project_root / "scripts" / "digital_meter_detection" / "training" / "train_digital_meter_yolo.py"
    config_path = project_root / "config" / "digital_meter_yolo_config.yaml"
    
    # 检查配置文件是否存在，如果不存在则创建
    if not config_path.exists():
        print("📝 配置文件不存在，正在创建默认配置...")
        create_result = run_command(
            f"python {script_path} --create-config",
            "创建默认配置文件"
        )
        if not create_result:
            return False
        print("✅ 默认配置文件创建成功")
    
    command = f"python {script_path} --config config/digital_meter_yolo_config.yaml"
    return run_command(command, "开始完整模型训练（200轮）")

def train_demo_model():
    """训练演示模型"""
    project_root = get_project_root()
    demo_script = project_root / "scripts" / "digital_meter_detection" / "demo" / "demo_digital_meter_detection.py"
    
    print("🎮 这将运行完整的演示流程（包括快速训练）")
    command = f"python {demo_script}"
    return run_command(command, "运行演示流程")

def run_inference():
    """运行推理"""
    project_root = get_project_root()
    script_path = project_root / "scripts" / "digital_meter_detection" / "inference" / "digital_meter_inference.py"
    
    # 获取用户输入
    print("💡 请提供推理参数：")
    model_path = input("模型路径 (例如: runs/detect/xxx/weights/best.pt): ").strip()
    if not model_path:
        print("❌ 模型路径不能为空")
        return False
    
    input_path = input("输入图像/目录 (例如: data/digital_meters/images/sample.jpg): ").strip()
    if not input_path:
        print("❌ 输入路径不能为空")
        return False
    
    output_path = input("输出目录 (留空使用默认): ").strip()
    
    # 构建命令
    command = f"python {script_path} --model {model_path} --input {input_path}"
    if output_path:
        command += f" --output {output_path}"
    
    return run_command(command, "执行液晶表检测推理")

def run_enhancement_demo():
    """运行液晶屏数字增强演示"""
    project_root = get_project_root()
    script_path = project_root / "scripts" / "digital_meter_detection" / "enhancement" / "demo_enhancement.py"
    
    command = f"python {script_path}"
    return run_command(command, "运行液晶屏数字增强演示")

def run_ocr_demo():
    """运行OCR数字提取演示"""
    project_root = get_project_root()
    script_path = project_root / "scripts" / "digital_meter_detection" / "ocr" / "demo_ocr.py"
    
    command = f"python {script_path}"
    return run_command(command, "运行OCR数字提取演示")

def run_demo():
    """运行完整演示"""
    project_root = get_project_root()
    script_path = project_root / "scripts" / "digital_meter_detection" / "demo" / "demo_digital_meter_detection.py"
    
    command = f"python {script_path}"
    return run_command(command, "运行完整演示流程")

def run_complete_reading():
    """运行完整读数提取"""
    project_root = get_project_root()
    script_path = project_root / "scripts" / "digital_meter_detection" / "digital_meter_reading.py"
    
    # 获取用户输入
    print("🔢 完整液晶数字表读数提取")
    print("💡 这个功能将执行：检测 -> 裁剪 -> 增强 -> OCR")
    print()
    
    input_path = input("输入图像文件或目录路径: ").strip()
    if not input_path:
        print("❌ 输入路径不能为空")
        return False
    
    # 检查模型文件
    model_path = project_root / "models" / "detection" / "digital_detection_model.pt"
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("请确保已训练好的模型文件存在")
        return False
    
    # 构建命令
    command = f"python {script_path} --input \"{input_path}\""
    
    return run_command(command, "执行完整液晶数字表读数提取")

def run_function_test():
    """运行功能测试"""
    project_root = get_project_root()
    script_path = project_root / "scripts" / "digital_meter_detection" / "test_digital_meter_reading.py"
    
    command = f"python {script_path}"
    return run_command(command, "运行功能测试")

def show_help():
    """显示帮助信息"""
    print("📖 使用说明：")
    print()
    print("这个脚本提供了液晶数字表检测系统的所有主要功能：")
    print()
    print("🔍 数据集验证:")
    print("   检查数据集格式和完整性，确保训练前数据无误")
    print()
    print("🚀 模型训练:")
    print("   - 完整训练：200轮，适合生产环境")
    print("   - 快速演示：20轮，用于测试和演示")
    print()
    print("🎯 模型推理:")
    print("   使用训练好的模型检测液晶数字表")
    print("   支持单张图像或批量处理")
    print()
    print("🎨 液晶屏数字增强:")
    print("   专门针对液晶屏显示问题的图像增强")
    print("   - 反光去除：检测并修复高亮反光区域")
    print("   - 对比度增强：使用CLAHE算法提升局部对比度")
    print("   - 图像锐化：增强数字边缘清晰度")
    print("   - 数字提取：多阈值方法分离数字区域")
    print()
    print("🔤 OCR数字提取:")
    print("   从液晶屏图像中提取数字内容")
    print("   - EasyOCR：高精度通用OCR引擎（推荐）")
    print("   - PaddleOCR：百度开源，可选引擎")
    print("   - Tesseract：快速文字识别引擎")
    print("   - 数字验证：自动校验提取结果")
    print("   - 批量处理：支持大量图像处理")
    print()
    print("🎬 完整演示:")
    print("   自动执行：验证→快速训练→推理→结果展示")
    print()
    print("🔢 完整读数提取:")
    print("   集成检测、增强、OCR的端到端数字读取")
    print("   - 自动检测液晶显示屏区域")
    print("   - 智能图像增强去除反光和噪声")
    print("   - 高精度OCR数字识别")
    print("   - 生成完整的处理报告和可视化结果")
    print()
    print("📁 项目结构:")
    print("   - scripts/digital_meter_detection/training/     # 训练脚本")
    print("   - scripts/digital_meter_detection/inference/    # 推理脚本")
    print("   - scripts/digital_meter_detection/validation/   # 验证脚本")
    print("   - scripts/digital_meter_detection/enhancement/  # 图像增强脚本")
    print("   - scripts/digital_meter_detection/ocr/          # OCR提取脚本")
    print("   - scripts/digital_meter_detection/demo/         # 演示脚本")
    print()
    print("📝 配置文件:")
    print("   - config/digital_meter_yolo_config.yaml         # 主要配置")
    print("   - data/digital_meters/dataset.yaml              # 数据集配置")
    print()
    print("💾 输出目录:")
    print("   - runs/detect/                    # 训练结果")
    print("   - outputs/inference/              # 推理结果")
    print("   - outputs/digital_enhancement/    # 图像增强结果")
    print("   - outputs/digital_ocr/            # OCR提取结果")
    print()

def main():
    """主函数"""
    project_root = get_project_root()
    
    # 检查项目结构
    data_dir = project_root / "data" / "digital_meters"
    if not data_dir.exists():
        print("❌ 错误：未找到液晶数字表数据集目录")
        print(f"期望路径: {data_dir}")
        print("请确保数据集已正确放置")
        sys.exit(1)
    
    print_banner()
    
    while True:
        print_menu()
        choice = input("请输入选项 (0-10): ").strip()
        print()
        
        if choice == "0":
            print("👋 退出系统，再见！")
            break
        elif choice == "1":
            validate_dataset()
        elif choice == "2":
            train_full_model()
        elif choice == "3":
            train_demo_model()
        elif choice == "4":
            run_inference()
        elif choice == "5":
            run_enhancement_demo()
        elif choice == "6":
            run_ocr_demo()
        elif choice == "7":
            run_demo()
        elif choice == "8":
            run_complete_reading()
        elif choice == "9":
            run_function_test()
        elif choice == "10":
            show_help()
        else:
            print("❌ 无效选项，请重新选择")
        
        print()
        input("按 Enter 继续...")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main() 