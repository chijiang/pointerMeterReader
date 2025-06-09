# 液晶数字表检测 YOLO v10 系统

这是一个专门用于液晶数字表检测的YOLO v10训练和推理系统，是完整的液晶表示数识别pipeline的第一步。

## 📋 系统概述

本系统实现了液晶表识别的三步流程中的第一步：

1. **YOLO检测液晶表** ← 本系统
2. 液晶表内数字增强（过滤反光，提高对比度等）
3. 使用OCR模型进行数字识别

## 🎯 功能特点

- **专门优化**：针对液晶数字表的特点进行了专门的参数调优
- **高精度检测**：使用YOLO v10架构，检测精度高
- **智能过滤**：内置多种过滤器，确保检测结果质量
- **ROI提取**：自动提取液晶表区域，为后续OCR步骤做准备
- **完整工具链**：包含训练、验证、推理等完整工具

## 📂 项目结构

```
pointMeterDetection/
├── config/
│   ├── digital_meter_yolo_config.yaml    # YOLO训练配置
│   └── detection_config.yaml             # 通用检测配置
├── data/
│   └── digital_meters/                    # 液晶表数据集
│       ├── images/                        # 图像文件
│       ├── labels/                        # YOLO格式标签
│       ├── classes.txt                    # 类别定义
│       ├── dataset.yaml                   # 数据集配置
│       └── notes.json                     # 数据集描述
├── scripts/
│   └── digital_meter_detection/           # 液晶表检测脚本目录
│       ├── training/                      # 训练相关脚本
│       │   └── train_digital_meter_yolo.py
│       ├── inference/                     # 推理相关脚本
│       │   └── digital_meter_inference.py
│       ├── validation/                    # 验证相关脚本
│       │   └── validate_digital_meter_dataset.py
│       ├── demo/                          # 演示脚本
│       │   └── demo_digital_meter_detection.py
│       ├── run.py                         # 便捷启动脚本 ⭐
│       └── README.md                      # 脚本说明文档
├── runs/                                  # 训练输出目录
├── outputs/                               # 推理输出目录
└── DIGITAL_METER_DETECTION_README.md     # 本文档
```

> 📢 **重要更新**：所有液晶数字表检测脚本现已重新组织到 `scripts/digital_meter_detection/` 目录下，按功能分类。新增了便捷启动脚本 `scripts/digital_meter_detection/run.py`！

## 🚀 快速开始

### 方式一：使用便捷启动脚本（推荐）⭐

```bash
# 运行便捷启动界面
python scripts/digital_meter_detection/run.py
```

这将启动一个交互式菜单，包含所有功能：
- 📊 数据集验证和可视化
- 🚀 完整模型训练（200轮，自动创建配置）
- ⚡ 快速演示训练（20轮）
- 🎯 模型推理和分析（丰富的可视化功能）
- 🎬 完整演示流程
- ❓ 帮助信息

✨ **最新升级功能**：
- 智能设备检测（GPU/MPS/CPU自动选择）
- 丰富的训练和推理可视化
- 详细的性能分析报告
- 自动ROI提取和分析

### 方式二：直接运行各个脚本

#### 1. 环境准备

确保已安装必要的依赖：

```bash
pip install ultralytics opencv-python matplotlib numpy pyyaml
```

#### 2. 数据集验证

在开始训练之前，验证数据集的完整性：

```bash
python scripts/digital_meter_detection/validation/validate_digital_meter_dataset.py --dataset data/digital_meters
```

#### 3. 模型训练

使用配置好的参数开始训练：

```bash
python scripts/digital_meter_detection/training/train_digital_meter_yolo.py --config config/digital_meter_yolo_config.yaml
```

训练过程中可以使用 `--validate-only` 参数仅验证数据集而不训练：

```bash
python scripts/digital_meter_detection/training/train_digital_meter_yolo.py --config config/digital_meter_yolo_config.yaml --validate-only
```

#### 4. 模型推理

使用训练好的模型进行液晶表检测：

```bash
# 单张图像检测
python scripts/digital_meter_detection/inference/digital_meter_inference.py --model runs/detect/digital_meter_detection_*/weights/best.pt --input path/to/image.jpg

# 批量检测
python scripts/digital_meter_detection/inference/digital_meter_inference.py --model runs/detect/digital_meter_detection_*/weights/best.pt --input path/to/images/
```

#### 5. 完整演示

运行端到端演示流程（验证→快速训练→推理→展示）：

```bash
python scripts/digital_meter_detection/demo/demo_digital_meter_detection.py
```

## ⚙️ 配置说明

### 训练配置 (config/digital_meter_yolo_config.yaml)

主要配置项：

- **模型选择**：`yolov10n.pt` (轻量级，适合实时应用)
- **训练轮数**：200轮（液晶表特征相对简单）
- **数据增强**：针对液晶表特点优化的增强策略
- **学习率调度**：余弦学习率调度，有助于更好收敛

### 数据增强策略

针对液晶表的特点，采用了以下增强策略：

- **光照变化**：`hsv_v: 0.5` - 重点处理光照变化
- **几何变换**：适度的旋转和缩放，考虑实际拍摄角度
- **颜色变换**：保守的颜色增强，维持液晶表视觉特征
- **翻转策略**：避免上下翻转，保持液晶表方向性

## 📊 数据集要求

### 目录结构

```
data/digital_meters/
├── images/           # 图像文件 (.jpg, .png, .jpeg)
├── labels/           # YOLO格式标签文件 (.txt)
├── classes.txt       # 类别定义：digital_meter
└── dataset.yaml      # 数据集配置文件
```

### 标签格式

YOLO格式，每行包含：
```
class_id x_center y_center width height
```

其中：
- `class_id`: 类别ID（液晶表为0）
- `x_center, y_center`: 边界框中心坐标（归一化到0-1）
- `width, height`: 边界框宽高（归一化到0-1）

### 数据质量要求

- **最小面积**：1600像素（40x40）
- **宽高比范围**：1.2 - 6.0（液晶表通常是横向矩形）
- **图像格式**：JPG, PNG, JPEG
- **标注完整性**：每张图像需要对应的标签文件

## 🎯 训练流程

### 1. 数据验证
- 检查目录结构
- 验证图像和标签对应关系
- 检查标签格式正确性
- 统计数据集信息

### 2. 模型初始化
- 加载预训练的YOLO v10模型
- 设置训练参数
- 配置数据增强策略

### 3. 训练过程
- 自动分割训练集和验证集（8:2）
- 使用余弦学习率调度
- 早停机制防止过拟合
- 定期保存模型权重

### 4. 模型评估
- 计算mAP@0.5和mAP@0.5:0.95
- 生成混淆矩阵和训练曲线
- 保存验证集预测结果

### 5. 模型导出
- 导出ONNX格式（用于部署）
- 导出TorchScript格式
- 优化模型以提高推理速度

## 🔍 推理流程

### 1. 模型加载
- 加载训练好的YOLO模型
- 设置推理参数（置信度阈值、NMS阈值等）

### 2. 图像检测
- 对输入图像进行前向推理
- 获取检测结果（边界框、置信度、类别）

### 3. 结果过滤
- 按面积过滤（去除过小或过大的检测框）
- 按宽高比过滤（确保符合液晶表特征）
- 按置信度过滤

### 4. ROI提取
- 从原图像中提取液晶表区域
- 添加适当的边界填充
- 保存ROI图像供后续OCR使用

### 5. 结果可视化
- 在原图上绘制检测框
- 添加置信度标签
- 保存可视化结果

## 📈 性能优化

### 训练优化
- **模型选择**：使用YOLOv10n平衡精度和速度
- **批大小**：根据GPU内存调整（默认16）
- **混合精度**：启用AMP加速训练
- **数据加载**：多进程数据加载

### 推理优化
- **模型量化**：支持INT8量化
- **批处理推理**：支持批量图像处理
- **设备选择**：支持CPU、MPS、CUDA
- **内存管理**：优化内存使用

## 🛠️ 常见问题

### Q: 训练时出现"CUDA out of memory"错误
A: 减少batch_size，比如从16改为8或4

### Q: 验证时发现标签格式错误
A: 检查标签文件中的坐标值是否在0-1范围内，类别ID是否为0

### Q: 检测结果中有很多误检
A: 调高置信度阈值（conf）或调整过滤参数（min_area, aspect_ratio）

### Q: 训练收敛慢或不收敛
A: 检查学习率设置，或增加训练轮数

### Q: 模型在新图像上表现不好
A: 可能需要更多样化的训练数据，或调整数据增强策略

## 📦 输出文件说明

### 训练输出
```
runs/detect/digital_meter_detection_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt           # 最佳模型权重
│   └── last.pt           # 最后一轮权重
├── results.png           # 训练曲线
├── confusion_matrix.png  # 混淆矩阵
├── val_batch0_*.jpg      # 验证集预测示例
├── config.yaml           # 配置文件备份
└── training_summary.md   # 训练总结报告
```

### 推理输出
```
outputs/digital_meter_detection/
├── rois/                 # 提取的ROI图像
├── *_detection_*.jpg     # 检测结果可视化
├── *_results_*.json      # 检测结果JSON
└── logs/                 # 推理日志
```

## 🔗 集成到完整Pipeline

本系统的输出（ROI图像）可以直接用于后续的数字增强和OCR识别步骤：

1. **使用本系统检测液晶表**
2. **数字增强模块**：处理ROI图像，减少反光、提高对比度
3. **OCR识别模块**：识别增强后图像中的数字

示例集成代码：

```python
from ultralytics import YOLO
from digital_meter_inference import DigitalMeterDetector

# 1. 检测液晶表
detector = DigitalMeterDetector("path/to/best.pt")
results = detector.detect_and_extract("input_image.jpg")

# 2. 处理每个检测到的液晶表
for detection in results['roi_detections']:
    roi_path = detection['roi']['saved_path']
    
    # 3. 调用数字增强模块
    enhanced_roi = enhance_digital_display(roi_path)
    
    # 4. 调用OCR模块
    reading = ocr_recognize_digits(enhanced_roi)
    
    print(f"液晶表读数: {reading}")
```

## 📞 技术支持

如有问题或建议，请通过以下方式联系：

- 检查训练日志和错误信息
- 验证数据集格式和完整性
- 调整配置参数
- 查看生成的训练报告

## 📄 许可证

本项目遵循项目根目录下的LICENSE文件规定。 