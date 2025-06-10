#!/usr/bin/env python3
"""
快速启动液晶数字表检测模型训练脚本

这是一个简化的启动脚本，用于快速开始训练液晶数字表检测模型。
会自动创建配置文件、设置环境并启动训练。

作者: chijiang
日期: 2025-01-15
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    
    print("🚀 液晶数字表检测模型训练启动器")
    print("=" * 50)
    
    # 检查数据是否存在
    data_dir = project_root / "data" / "digits"
    if not data_dir.exists():
        print("❌ 错误：未找到数字数据目录")
        print(f"请确保数据存放在: {data_dir}")
        return
    
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    classes_file = data_dir / "classes.txt"
    
    if not images_dir.exists() or not labels_dir.exists() or not classes_file.exists():
        print("❌ 错误：数据目录结构不完整")
        print("需要以下文件/目录：")
        print(f"  - {images_dir}")
        print(f"  - {labels_dir}")
        print(f"  - {classes_file}")
        return
    
    # 统计数据
    image_count = len(list(images_dir.glob("*.jpg")))
    label_count = len(list(labels_dir.glob("*.txt")))
    
    print(f"📊 数据统计：")
    print(f"  - 图像数量: {image_count}")
    print(f"  - 标注数量: {label_count}")
    
    if image_count == 0:
        print("❌ 错误：未找到图像文件")
        return
    
    if label_count == 0:
        print("❌ 错误：未找到标注文件")
        return
    
    # 配置文件路径
    config_file = project_root / "config" / "digits_config.yaml"
    train_script = project_root / "scripts" / "train_digits.py"
    
    # 检查脚本是否存在
    if not train_script.exists():
        print(f"❌ 错误：训练脚本不存在: {train_script}")
        return
    
    # 创建配置文件（如果不存在）
    if not config_file.exists():
        print("🔧 创建默认配置文件...")
        try:
            subprocess.run([
                sys.executable, str(train_script), 
                "--create-config", 
                "--config", str(config_file)
            ], check=True, cwd=project_root)
            print(f"✅ 配置文件创建成功: {config_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ 创建配置文件失败: {e}")
            return
    
    print(f"📋 使用配置文件: {config_file}")
    
    # 询问用户是否开始训练
    print("\n🎯 准备开始训练...")
    print("配置信息：")
    print("  - 模型: YOLOv10 nano")
    print("  - 训练轮数: 200")
    print("  - 批大小: 16")
    print("  - 图像大小: 640x640")
    print("  - 设备: 自动检测")
    
    response = input("\n是否开始训练？(y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ 训练已取消")
        return
    
    # 开始训练
    print("\n🚀 开始训练...")
    print("=" * 50)
    
    try:
        # 运行训练脚本
        cmd = [
            sys.executable, str(train_script),
            "--config", str(config_file),
            "--visualize"  # 生成可视化结果
        ]
        
        print(f"📝 执行命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=project_root)
        
        print("\n🎉 训练完成！")
        print("📁 查看结果:")
        print(f"  - 模型权重: {project_root}/outputs/checkpoints/digits/")
        print(f"  - 训练日志: {project_root}/outputs/logs/digits/")
        print(f"  - 可视化结果: {project_root}/outputs/results/digits/visualizations/")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        print("请检查错误信息并重试")
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")


if __name__ == "__main__":
    main() 