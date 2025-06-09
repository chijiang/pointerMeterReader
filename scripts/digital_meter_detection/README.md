# 液晶数字表检测脚本说明

这个目录包含了液晶数字表检测系统的所有脚本，按功能分类组织。已升级到最新版本，参考指针表训练脚本架构，增加了丰富的可视化和分析功能。

## 📂 目录结构

```
scripts/digital_meter_detection/
├── training/                    # 训练相关脚本
│   └── train_digital_meter_yolo.py    # YOLO v10模型训练脚本 (已升级)
├── inference/                   # 推理相关脚本
│   └── digital_meter_inference.py     # 液晶表检测推理脚本 (已升级)
├── validation/                  # 验证相关脚本
│   └── validate_digital_meter_dataset.py  # 数据集验证脚本
├── demo/                        # 演示脚本
│   └── demo_digital_meter_detection.py    # 完整流程演示
├── run.py                       # 🌟 便捷启动脚本 (新增)
└── README.md                    # 本文档
```

## 🚀 快速使用

### 方式一：使用便捷启动脚本（推荐）⭐

```bash
# 运行便捷启动界面
python scripts/digital_meter_detection/run.py
```

便捷启动脚本提供交互式菜单，包含以下功能：
1. 📊 验证数据集 - 检查数据完整性和格式
2. 🚀 训练模型（完整训练 - 200轮）- 自动创建配置文件
3. ⚡ 训练模型（快速演示 - 20轮）- 快速体验功能
4. 🎯 模型推理 - 丰富的可视化分析
5. 🎬 完整演示流程 - 一键体验全流程
6. ❓ 查看帮助 - 详细使用说明

✨ **新增功能**：
- 自动创建配置文件
- 智能设备检测（GPU/MPS/CPU）
- 丰富的训练和推理可视化
- 详细的性能分析报告
- ROI区域自动提取

### 方式二：从项目根目录运行

所有脚本都设计为从项目根目录运行，路径会自动处理：

```bash
# 回到项目根目录
cd /path/to/pointMeterDetection

# 数据集验证
python scripts/digital_meter_detection/validation/validate_digital_meter_dataset.py --dataset data/digital_meters

# 模型训练
python scripts/digital_meter_detection/training/train_digital_meter_yolo.py --config config/digital_meter_yolo_config.yaml

# 模型推理
python scripts/digital_meter_detection/inference/digital_meter_inference.py --model runs/detect/digital_meter_detection_20250609_172230/weights/best.pt --input path/to/image.jpg

# 完整演示
python scripts/digital_meter_detection/demo/demo_digital_meter_detection.py
```

### 方式三：从脚本目录运行

也可以直接在各个脚本目录中运行，路径会自动调整到项目根目录：

```bash
# 进入训练脚本目录
cd scripts/digital_meter_detection/training
python train_digital_meter_yolo.py --config config/digital_meter_yolo_config.yaml

# 进入推理脚本目录
cd scripts/digital_meter_detection/inference
python digital_meter_inference.py --model runs/detect/xxx/weights/best.pt --input data/digital_meters/images/sample.jpg
```

## 📋 脚本详细说明

### 1. 训练脚本 (training/train_digital_meter_yolo.py) ⭐已升级

**功能**：训练YOLO v10液晶数字表检测模型，参考指针表训练脚本架构完全重写

**主要特性**：
- 智能数据集验证和预处理
- 针对液晶表优化的配置
- 完整的训练流程（训练、验证、导出）
- 丰富的可视化功能（数据集样本、训练曲线、预测结果）
- 详细的性能分析报告
- 智能设备检测（GPU/MPS/CPU）
- 自动模型导出（ONNX、TorchScript）
- Markdown和JSON格式的训练报告

**使用示例**：
```bash
# 创建默认配置
python scripts/digital_meter_detection/training/train_digital_meter_yolo.py --create-config

# 训练模型（完整流程）
python scripts/digital_meter_detection/training/train_digital_meter_yolo.py --config config/digital_meter_yolo_config.yaml

# 仅验证数据集
python scripts/digital_meter_detection/training/train_digital_meter_yolo.py --validate-only --model-path outputs/digital_meter_detection/checkpoints/digital_meter_detection_20250609_184915/weights/best.pt

# 仅评估模型
python scripts/digital_meter_detection/training/train_digital_meter_yolo.py --eval-only --model-path path/to/model.pt
```

**输出**：
- 训练好的模型权重和导出模型
- 丰富的可视化图表（数据集样本、训练曲线、预测结果）
- 详细的训练报告（JSON和Markdown格式）
- 性能评估指标和分析

### 2. 推理脚本 (inference/digital_meter_inference.py) ⭐已升级

**功能**：使用训练好的模型进行液晶表检测，完全重写增强版本

**主要特性**：
- 支持单张图像和批量处理
- 智能结果过滤和后处理
- 自动ROI提取和保存
- 丰富的可视化功能（检测结果、分析图表、结果画廊）
- 详细的统计分析（置信度、面积、宽高比分布）
- 智能设备检测（GPU/MPS/CPU）
- JSON和Markdown格式的分析报告

**使用示例**：
```bash
# 单张图像处理
python scripts/digital_meter_detection/inference/digital_meter_inference.py \
  --model runs/detect/digital_meter_detection_xxx/weights/best.pt \
  --input data/digital_meters/images/sample.jpg

# 批量处理目录
python scripts/digital_meter_detection/inference/digital_meter_inference.py \
  --model runs/detect/digital_meter_detection_xxx/weights/best.pt \
  --input data/digital_meters/images/

# 自定义输出目录
python scripts/digital_meter_detection/inference/digital_meter_inference.py \
  --model runs/detect/digital_meter_detection_xxx/weights/best.pt \
  --input data/digital_meters/images/ \
  --output custom_output_dir \
  --conf 0.3 \
  --no-rois
```

**输出**：
- 检测结果可视化图像和画廊
- 提取的ROI区域图像
- 详细的统计分析图表
- JSON和Markdown格式的分析报告
- 置信度、面积、宽高比分布分析

### 3. 验证脚本 (validation/validate_digital_meter_dataset.py)

**功能**：验证数据集格式和完整性

**主要特性**：
- 检查目录结构
- 验证图像和标签对应关系
- 检查标签格式正确性
- 统计数据集信息

**使用示例**：
```bash
python scripts/digital_meter_detection/validation/validate_digital_meter_dataset.py --dataset data/digital_meters
```

### 4. 演示脚本 (demo/demo_digital_meter_detection.py)

**功能**：完整流程演示（验证→训练→推理）

**主要特性**：
- 环境依赖检查
- 小规模快速训练（20轮演示）
- 自动推理测试
- 结果展示

**使用示例**：
```bash
python scripts/digital_meter_detection/demo/demo_digital_meter_detection.py
```

## ⚙️ 配置文件

所有脚本使用的配置文件位于项目根目录：

- `config/digital_meter_yolo_config.yaml` - 主要训练配置
- `data/digital_meters/dataset.yaml` - 数据集配置

## 📊 输出目录

脚本输出统一组织在项目根目录下：

- `runs/detect/` - 训练结果
- `outputs/` - 推理结果
- `outputs/validation/` - 验证结果

## 🔧 路径处理

所有脚本都包含智能路径处理：

1. **自动检测运行位置**：无论从哪个目录运行，都会自动找到项目根目录
2. **相对路径支持**：所有配置中的路径都相对于项目根目录
3. **跨平台兼容**：使用`pathlib`确保路径在不同操作系统上正确工作

## 🎯 最佳实践

1. **推荐从项目根目录运行**，路径最清晰明确
2. **使用相对路径**指定模型和数据文件
3. **查看脚本帮助**：使用`--help`参数查看详细用法
4. **检查输出目录**：运行前确保有足够的磁盘空间

## 🔗 与主项目集成

这些脚本与主项目的集成点：

- **数据集**：使用`data/digital_meters/`中的标注数据
- **配置**：复用`config/`目录中的配置文件
- **输出**：结果可用于后续的数字增强和OCR步骤
- **模型**：训练的模型可集成到完整的读数识别pipeline中 