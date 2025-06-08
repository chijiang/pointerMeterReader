#!/usr/bin/env python3
"""
上传数据集和模型到Hugging Face Hub
需要先安装: pip install huggingface_hub
"""

from huggingface_hub import HfApi, create_repo, upload_folder, upload_file
import os
from pathlib import Path

def upload_dataset():
    """上传数据集到Hugging Face"""
    api = HfApi()
    
    # 创建数据集仓库
    repo_id = "chijiang/pointer-meter-detection-dataset"
    
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=False  # 设为True则为私有
        )
        print(f"✅ 创建数据集仓库: {repo_id}")
    except Exception as e:
        print(f"仓库可能已存在: {e}")
    
    # 上传数据集文件夹
    if os.path.exists("data/detection"):
        upload_folder(
            folder_path="data/detection",
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="detection"
        )
        print("✅ 上传检测数据集完成")
    
    if os.path.exists("data/segmentation"):
        upload_folder(
            folder_path="data/segmentation",
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="segmentation"
        )
        print("✅ 上传分割数据集完成")

def upload_models():
    """上传训练好的模型到Hugging Face"""
    api = HfApi()
    
    # 创建模型仓库
    repo_id = "chijiang/pointer-meter-reader"
    
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"✅ 创建模型仓库: {repo_id}")
    except Exception as e:
        print(f"仓库可能已存在: {e}")
    
    # 上传检测模型
    detection_model_paths = [
        "outputs/checkpoints/detection/meter_detection_v1/weights/best.pt",
        "models/detection/detection_model.pt"
    ]
    
    for model_path in detection_model_paths:
        if os.path.exists(model_path):
            upload_file(
                path_or_fileobj=model_path,
                path_in_repo=f"detection/{Path(model_path).name}",
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"✅ 上传检测模型: {model_path}")
            break
    
    # 上传分割模型
    segmentation_model_paths = [
        "outputs/segmentation/exported/segmentation_model.onnx",
        "outputs/segmentation/checkpoints/best_model.pth"
    ]
    
    for model_path in segmentation_model_paths:
        if os.path.exists(model_path):
            upload_file(
                path_or_fileobj=model_path,
                path_in_repo=f"segmentation/{Path(model_path).name}",
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"✅ 上传分割模型: {model_path}")

def create_model_card():
    """创建模型卡片"""
    model_card_content = """
---
license: mit
language:
- zh
- en
tags:
- computer-vision
- object-detection
- semantic-segmentation
- industrial-meters
- pointer-reading
datasets:
- chijiang/pointer-meter-detection-dataset
metrics:
- accuracy
- iou
pipeline_tag: object-detection
---

# 工业指针式仪表读数识别模型

## 模型描述

这是一个完整的工业指针式仪表读数自动识别系统，包含：

1. **检测模型**: 基于YOLOv10的仪表检测
2. **分割模型**: 基于DeepLabV3+的指针和刻度分割
3. **读数算法**: 基于几何分析的读数提取

## 使用方法

```python
from ultralytics import YOLO
import onnxruntime as ort

# 加载检测模型
detector = YOLO("detection/best.pt")

# 加载分割模型
session = ort.InferenceSession("segmentation/segmentation_model.onnx")

# 使用完整应用
# 参考: https://github.com/chijiang/pointerMeterReader
```

## 性能指标

- 检测精度: mAP@0.5 > 0.85
- 分割精度: mIoU > 0.75
- 读数误差: < 5%

## 训练数据

- 检测数据集: 1836张工业仪表图像
- 分割数据集: 手工标注的指针和刻度掩码
- 数据来源: 工业现场采集

## 引用

```bibtex
@misc{pointer-meter-reader-2025,
  title={Industrial Pointer Meter Reading System},
  author={chijiang},
  year={2025},
  url={https://github.com/chijiang/pointerMeterReader}
}
```
"""
    
    with open("MODEL_CARD.md", "w", encoding="utf-8") as f:
        f.write(model_card_content)
    
    print("✅ 创建模型卡片完成")

def create_dataset_card():
    """创建数据集卡片"""
    dataset_card_content = """
---
license: mit
task_categories:
- object-detection
- image-segmentation
language:
- zh
- en
tags:
- industrial-meters
- pointer-detection
- computer-vision
size_categories:
- 1K<n<10K
---

# 工业指针式仪表检测数据集

## 数据集描述

这是一个用于工业指针式仪表检测和分割的数据集，包含：

### 检测数据集
- **图像数量**: 1836张
- **标注格式**: COCO格式
- **类别**: 仪表 (meter)
- **分辨率**: 1000-1920 x 584-1080

### 分割数据集
- **图像数量**: 待补充
- **标注格式**: 像素级掩码
- **类别**: 背景、指针、刻度

## 数据结构

```
detection/
├── train2017/          # 训练图像
├── val2017/            # 验证图像
└── annotations/        # COCO格式标注
    ├── instances_train2017.json
    └── instances_val2017.json

segmentation/
├── images/             # 原始图像
├── masks/              # 分割掩码
└── splits/             # 训练/验证分割
```

## 使用许可

MIT License - 可用于商业和学术用途

## 引用

如果使用此数据集，请引用：

```bibtex
@dataset{pointer-meter-dataset-2025,
  title={Industrial Pointer Meter Detection Dataset},
  author={chijiang},
  year={2025},
  url={https://huggingface.co/datasets/chijiang/pointer-meter-detection-dataset}
}
```
"""
    
    with open("DATASET_CARD.md", "w", encoding="utf-8") as f:
        f.write(dataset_card_content)
    
    print("✅ 创建数据集卡片完成")

def main():
    """主函数"""
    print("开始上传数据和模型到Hugging Face...")
    
    # 检查是否已登录
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✅ 已登录用户: {user['name']}")
    except Exception as e:
        print("❌ 请先登录Hugging Face:")
        print("   huggingface-cli login")
        return
    
    # 创建卡片
    create_model_card()
    create_dataset_card()
    
    # 上传数据集
    print("\n📤 上传数据集...")
    upload_dataset()
    
    # 上传模型
    print("\n📤 上传模型...")
    upload_models()
    
    print("\n✅ 上传完成！")
    print("🔗 数据集: https://huggingface.co/datasets/chijiang/pointer-meter-detection-dataset")
    print("🔗 模型: https://huggingface.co/chijiang/pointer-meter-reader")

if __name__ == "__main__":
    main() 