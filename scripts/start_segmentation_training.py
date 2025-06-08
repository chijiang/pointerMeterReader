#!/usr/bin/env python3
"""
一键启动分割训练脚本

自动执行以下步骤：
1. 统一分割数据格式
2. 创建训练/验证分割
3. 开始分割训练

使用方法:
python scripts/start_segmentation_training.py --config config/segmentation_config.yaml
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """运行命令并处理错误"""
    print(f"\n{'='*50}")
    print(f"执行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print('='*50)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warning:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: {description} 失败")
        print(f"返回码: {e.returncode}")
        print(f"错误输出: {e.stderr}")
        if e.stdout:
            print(f"标准输出: {e.stdout}")
        return False


def main():
    parser = argparse.ArgumentParser(description="一键启动分割训练")
    parser.add_argument(
        "--config",
        type=str,
        default="config/segmentation_config.yaml",
        help="分割训练配置文件路径"
    )
    parser.add_argument(
        "--skip_data_prep",
        action="store_true",
        help="跳过数据准备步骤"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/segmentation",
        help="分割数据目录路径"
    )
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not Path(args.config).exists():
        print(f"错误: 配置文件不存在 {args.config}")
        return
    
    print("🚀 开始分割训练流程...")
    
    # 步骤1: 统一数据格式
    if not args.skip_data_prep:
        print("\n📋 步骤1: 统一分割数据格式")
        
        # 检查是否已经统一过数据
        unified_dir = Path(args.data_dir) / "SegmentationClass_unified"
        if unified_dir.exists():
            print("检测到已统一的数据目录，跳过数据格式统一步骤")
        else:
            cmd_unify = [
                sys.executable, 
                "tools/data_preparation/unify_segmentation_data.py",
                "--data_dir", args.data_dir
            ]
            
            if not run_command(cmd_unify, "统一分割数据格式"):
                print("❌ 数据格式统一失败，退出")
                return
            
            print("✅ 数据格式统一完成")
        
        # 步骤2: 创建训练/验证分割
        print("\n📋 步骤2: 创建训练/验证分割")
        
        # 检查是否已经创建过分割文件
        imageset_dir = Path(args.data_dir) / "ImageSets"
        train_file = imageset_dir / "train.txt"
        val_file = imageset_dir / "val.txt"
        
        if train_file.exists() and val_file.exists():
            print("检测到已存在的分割文件，跳过分割创建步骤")
        else:
            cmd_split = [
                sys.executable,
                "tools/data_preparation/create_segmentation_splits.py",
                "--data_dir", args.data_dir
            ]
            
            if not run_command(cmd_split, "创建训练/验证分割"):
                print("❌ 数据分割失败，退出")
                return
            
            print("✅ 数据分割完成")
    else:
        print("⏭️ 跳过数据准备步骤")
    
    # 步骤3: 开始训练
    print("\n📋 步骤3: 开始分割训练")
    
    cmd_train = [
        sys.executable,
        "scripts/train_segmentation.py",
        "--config", args.config
    ]
    
    if not run_command(cmd_train, "分割模型训练"):
        print("❌ 训练失败")
        return
    
    print("🎉 分割训练完成!")


if __name__ == "__main__":
    main() 