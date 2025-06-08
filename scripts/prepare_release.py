#!/usr/bin/env python3
"""
准备项目发布：打包数据集和模型
"""

import os
import shutil
import zipfile
import tarfile
from pathlib import Path
import json

def create_data_archive():
    """创建数据集压缩包"""
    print("📦 创建数据集压缩包...")
    
    # 检测数据集
    if os.path.exists("data/detection"):
        with zipfile.ZipFile("releases/detection_dataset.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk("data/detection"):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, "data")
                    zipf.write(file_path, arcname)
        print("✅ 检测数据集打包完成: releases/detection_dataset.zip")
    
    # 分割数据集
    if os.path.exists("data/segmentation"):
        with zipfile.ZipFile("releases/segmentation_dataset.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk("data/segmentation"):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, "data")
                    zipf.write(file_path, arcname)
        print("✅ 分割数据集打包完成: releases/segmentation_dataset.zip")

def create_model_archive():
    """创建模型压缩包"""
    print("📦 创建模型压缩包...")
    
    model_files = []
    
    # 检测模型
    detection_paths = [
        "outputs/checkpoints/detection/meter_detection_v1/weights/best.pt",
        "models/detection/detection_model.pt"
    ]
    
    for path in detection_paths:
        if os.path.exists(path):
            model_files.append((path, f"detection/{Path(path).name}"))
            break
    
    # 分割模型
    segmentation_paths = [
        "outputs/segmentation/exported/segmentation_model.onnx",
        "outputs/segmentation/checkpoints/best_model.pth"
    ]
    
    for path in segmentation_paths:
        if os.path.exists(path):
            model_files.append((path, f"segmentation/{Path(path).name}"))
    
    if model_files:
        with zipfile.ZipFile("releases/trained_models.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
            for src_path, arc_name in model_files:
                zipf.write(src_path, arc_name)
        print("✅ 模型打包完成: releases/trained_models.zip")
    else:
        print("⚠️  未找到训练好的模型")

def create_release_info():
    """创建发布信息文件"""
    release_info = {
        "version": "v1.0.0",
        "release_date": "2025-01-XX",
        "description": "工业指针式仪表读数识别系统完整版本",
        "components": {
            "detection_model": {
                "architecture": "YOLOv10",
                "dataset_size": "1836 images",
                "performance": "mAP@0.5 > 0.85"
            },
            "segmentation_model": {
                "architecture": "DeepLabV3+ (ResNet50)",
                "format": "ONNX",
                "classes": ["background", "pointer", "scale"],
                "performance": "mIoU > 0.75"
            },
            "reading_algorithm": {
                "method": "Geometric analysis",
                "accuracy": "< 5% error"
            }
        },
        "files": {
            "detection_dataset.zip": "检测数据集 (COCO格式)",
            "segmentation_dataset.zip": "分割数据集 (Pascal VOC格式)",
            "trained_models.zip": "预训练模型权重",
            "source_code.zip": "完整源代码"
        },
        "usage": {
            "web_app": "python app.py",
            "training": "python scripts/train_detection.py",
            "inference": "参考README.md"
        },
        "requirements": {
            "python": ">=3.8",
            "torch": ">=2.0.0",
            "memory": ">=4GB",
            "storage": ">=2GB"
        },
        "license": "MIT",
        "author": "chijiang",
        "repository": "https://github.com/chijiang/pointerMeterReader"
    }
    
    with open("releases/release_info.json", 'w', encoding='utf-8') as f:
        json.dump(release_info, f, ensure_ascii=False, indent=2)
    
    print("✅ 发布信息创建完成: releases/release_info.json")

def create_source_archive():
    """创建源代码压缩包"""
    print("📦 创建源代码压缩包...")
    
    # 要包含的文件和目录
    include_patterns = [
        "*.py",
        "*.md",
        "*.txt",
        "*.yaml",
        "*.yml",
        "config/",
        "scripts/",
        "tools/",
        "LICENSE"
    ]
    
    # 要排除的文件和目录
    exclude_patterns = [
        "__pycache__/",
        "*.pyc",
        ".git/",
        ".venv/",
        "outputs/",
        "data/",
        "models/",
        "releases/",
        ".DS_Store",
        "*.log"
    ]
    
    with zipfile.ZipFile("releases/source_code.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("."):
            # 过滤目录
            dirs[:] = [d for d in dirs if not any(d.startswith(pattern.rstrip('/')) for pattern in exclude_patterns)]
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, ".")
                
                # 检查是否应该排除
                should_exclude = any(
                    pattern in rel_path or rel_path.endswith(pattern.rstrip('*'))
                    for pattern in exclude_patterns
                )
                
                if not should_exclude:
                    zipf.write(file_path, rel_path)
    
    print("✅ 源代码打包完成: releases/source_code.zip")

def create_download_script():
    """创建下载脚本"""
    download_script = '''#!/bin/bash
# 工业仪表读数识别系统 - 快速下载脚本

echo "🚀 开始下载工业仪表读数识别系统..."

# 创建目录
mkdir -p pointMeterReader
cd pointMeterReader

# 下载源代码
echo "📥 下载源代码..."
curl -L -o source_code.zip "https://github.com/chijiang/pointerMeterReader/releases/latest/download/source_code.zip"
unzip source_code.zip
rm source_code.zip

# 下载预训练模型
echo "📥 下载预训练模型..."
curl -L -o trained_models.zip "https://github.com/chijiang/pointerMeterReader/releases/latest/download/trained_models.zip"
unzip trained_models.zip -d models/
rm trained_models.zip

# 可选：下载数据集
read -p "是否下载训练数据集？(y/N): " download_data
if [[ $download_data =~ ^[Yy]$ ]]; then
    echo "📥 下载检测数据集..."
    curl -L -o detection_dataset.zip "https://github.com/chijiang/pointerMeterReader/releases/latest/download/detection_dataset.zip"
    unzip detection_dataset.zip -d data/
    rm detection_dataset.zip
    
    echo "📥 下载分割数据集..."
    curl -L -o segmentation_dataset.zip "https://github.com/chijiang/pointerMeterReader/releases/latest/download/segmentation_dataset.zip"
    unzip segmentation_dataset.zip -d data/
    rm segmentation_dataset.zip
fi

# 安装依赖
echo "📦 安装Python依赖..."
pip install -r requirements.txt

echo "✅ 下载完成！"
echo "🚀 启动应用: python app.py"
echo "📖 查看文档: cat README.md"
'''
    
    with open("releases/download.sh", 'w') as f:
        f.write(download_script)
    
    # 设置执行权限
    os.chmod("releases/download.sh", 0o755)
    
    print("✅ 下载脚本创建完成: releases/download.sh")

def main():
    """主函数"""
    print("🎯 准备项目发布...")
    
    # 创建发布目录
    os.makedirs("releases", exist_ok=True)
    
    # 创建各种压缩包
    create_data_archive()
    create_model_archive()
    create_source_archive()
    
    # 创建发布信息
    create_release_info()
    create_download_script()
    
    print("\n✅ 发布准备完成！")
    print("\n📁 发布文件:")
    for file in os.listdir("releases"):
        file_path = os.path.join("releases", file)
        size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"   {file} ({size:.1f} MB)")
    
    print("\n📋 下一步:")
    print("1. 检查releases/目录中的文件")
    print("2. 创建GitHub Release并上传文件")
    print("3. 或运行: python scripts/upload_to_huggingface.py")

if __name__ == "__main__":
    main() 