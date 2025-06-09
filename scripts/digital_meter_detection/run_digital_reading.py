#!/usr/bin/env python3
"""
液晶数字表读数提取启动脚本
Simple launcher for digital meter reading extraction

这个脚本解决路径问题，提供用户友好的界面
"""

import os
import sys
from pathlib import Path

def get_project_root():
    """获取项目根目录"""
    current_dir = Path(__file__).parent
    # 从 scripts/digital_meter_detection/ 到项目根目录
    return current_dir.parent.parent

def check_dependencies():
    """检查依赖"""
    missing_deps = []
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import ultralytics
    except ImportError:
        missing_deps.append("ultralytics")
    
    try:
        import easyocr
    except ImportError:
        print("⚠️  EasyOCR未安装，OCR功能可能受限")
    
    if missing_deps:
        print("❌ 缺少必要的依赖库:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n请安装缺少的依赖: pip install " + " ".join(missing_deps))
        return False
    
    return True

def find_model_file():
    """查找模型文件"""
    project_root = get_project_root()
    
    # 可能的模型路径
    model_paths = [
        project_root / "models" / "detection" / "digital_detection_model.pt",
        project_root / "models" / "digital_detection_model.pt",
        project_root / "runs" / "detect" / "train" / "weights" / "best.pt",
        project_root / "runs" / "detect" / "train2" / "weights" / "best.pt",
        project_root / "runs" / "detect" / "train3" / "weights" / "best.pt",
    ]
    
    # 查找最新的训练模型
    train_dirs = list((project_root / "runs" / "detect").glob("train*/weights/best.pt"))
    if train_dirs:
        # 按修改时间排序，选择最新的
        train_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        model_paths.insert(0, train_dirs[0])
    
    for model_path in model_paths:
        if model_path.exists():
            return model_path
    
    return None

def main():
    """主函数"""
    print("🔢 液晶数字表读数提取系统")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 获取项目根目录
    project_root = get_project_root()
    print(f"📁 项目根目录: {project_root}")
    
    # 查找模型文件
    model_path = find_model_file()
    if model_path is None:
        print("❌ 未找到训练好的模型文件")
        print("请先训练模型，或手动指定模型路径")
        print("💡 提示:")
        print("   1. 运行 python run.py 选择训练功能")
        print("   2. 或使用 --model 参数指定模型路径")
        return
    
    print(f"✅ 找到模型文件: {model_path}")
    
    # 获取输入路径
    print("\n📁 请选择输入:")
    print("1. 📄 单张图像")
    print("2. 📂 图像目录")
    print("3. ⚡ 快速测试")
    
    choice = input("请选择 (1-3): ").strip()
    
    if choice == "1":
        input_path = input("图像文件路径: ").strip()
        if not input_path:
            print("❌ 路径不能为空")
            return
    elif choice == "2":
        input_path = input("图像目录路径: ").strip()
        if not input_path:
            print("❌ 路径不能为空")
            return
    elif choice == "3":
        # 使用测试图像
        test_image_dir = project_root / "data" / "test_digital_meters"
        if test_image_dir.exists():
            input_path = str(test_image_dir)
            print(f"使用测试图像: {input_path}")
        else:
            print("❌ 测试图像目录不存在")
            print("请先运行功能测试创建测试数据")
            return
    else:
        print("❌ 无效选择")
        return
    
    # 检查输入路径
    if not Path(input_path).exists():
        print(f"❌ 输入路径不存在: {input_path}")
        return
    
    # 构建命令
    script_path = project_root / "scripts" / "digital_meter_detection" / "digital_meter_reading.py"
    
    import subprocess
    
    cmd = [
        sys.executable,
        str(script_path),
        "--input", input_path,
        "--model", str(model_path)
    ]
    
    print("\n🚀 开始处理...")
    print(f"💻 执行命令: {' '.join(cmd[:4])} ...")
    print("-" * 50)
    
    try:
        # 切换到项目根目录执行
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        os.chdir(original_cwd)
        
        if result.returncode == 0:
            print("\n✅ 处理完成!")
        else:
            print(f"\n❌ 处理失败，退出码: {result.returncode}")
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")

if __name__ == "__main__":
    main() 