# 工业仪表读数自动识别系统

基于深度学习的工业仪表读数自动识别系统，能够从监控画面中自动提取压力表、温度计等圆形指针式仪表的读数。

## 🚀 主要特性

- **完整AI流水线**：检测 → 分割 → 读数提取
- **ONNX推理加速**：支持ONNX Runtime，避免PyTorch依赖
- **智能后处理**：自动去除分割噪声，优化指针和刻度边界
- **Web界面**：基于Gradio的友好用户界面
- **批量处理**：支持批量检测和裁剪仪表区域

## 技术架构

### 处理流程
```
原始图像 → 仪表检测(YOLO) → 区域裁剪 → 语义分割(ONNX) → 后处理优化 → 读数计算 → 最终结果
```

### 核心组件

1. **仪表检测**（YOLOv10）：从完整画面中定位仪表区域
2. **语义分割**（DeepLabV3+）：像素级分割指针和刻度
3. **读数提取**：基于几何算法计算指针角度并转换为读数

## 环境要求

- Python 3.11+
- 推荐：Apple Silicon (M1/M2) 或 CUDA兼容GPU
- 最低：4GB内存，现代CPU

### 主要依赖
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
onnxruntime>=1.22.0
opencv-python>=4.6.0
gradio>=4.0.0
numpy>=1.21.0
PyYAML>=6.0
```

## 快速开始

### 1. 安装依赖

```bash
# 使用uv（推荐）
uv sync

# 或使用pip
pip install -r requirements.txt
```

### 2. 启动Web应用

```bash
python main.py
```

访问 `http://localhost:7860` 使用Web界面。

## 📖 脚本使用指南

### 0. 检测模型训练

```bash
python scripts/train_detection.py --config config/detection_config.yaml
```

### 1. 分割模型训练流水线 (`segmentation_pipeline.py`)

端到端训练流水线，从COCO数据到ONNX模型的完整流程。

#### 功能说明

该脚本执行以下步骤：
1. **数据转换**：将COCO格式数据转换为Pascal VOC分割格式
2. **配置更新**：根据参数更新训练配置
3. **模型训练**：训练DeepLabV3+分割模型
4. **模型导出**：导出ONNX模型用于推理

#### 使用方法

**完整流水线（从COCO数据开始）**：
```bash
python scripts/segmentation_pipeline.py \
    --coco-json data/coco_data/result_coco.json \
    --coco-images data/coco_data/images \
    --config config/segmentation_config.yaml \
    --output-onnx models/segmentation/segmentation_model.onnx \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 0.001
```

**跳过数据转换（使用已有分割数据）**：
```bash
python scripts/segmentation_pipeline.py \
    --skip-conversion \
    --segmentation-dir data/segmentation_new \
    --config config/segmentation_config.yaml \
    --output-onnx models/segmentation/segmentation_model.onnx
```

**只导出ONNX模型（跳过训练）**：
```bash
python scripts/segmentation_pipeline.py \
    --skip-conversion \
    --skip-training \
    --model-checkpoint outputs/segmentation/best_model.pth \
    --config config/segmentation_config.yaml \
    --output-onnx models/segmentation/segmentation_model.onnx
```

**使用默认参数**：
```bash
python scripts/segmentation_pipeline.py
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--coco-json` | COCO格式标注文件路径 | `data/coco_data/result_coco.json` |
| `--coco-images` | COCO图像目录路径 | `data/coco_data/images` |
| `--config` | 训练配置文件路径 | `config/segmentation_config.yaml` |
| `--output-onnx` | 输出ONNX模型路径 | `models/segmentation/segmentation_model.onnx` |
| `--train-ratio` | 训练集比例 | `0.8` |
| `--epochs` | 训练轮数（覆盖配置文件） | 配置文件中的值 |
| `--batch-size` | 批次大小（覆盖配置文件） | 配置文件中的值 |
| `--learning-rate` | 学习率（覆盖配置文件） | 配置文件中的值 |
| `--skip-conversion` | 跳过数据转换步骤 | `False` |
| `--skip-training` | 跳过训练步骤 | `False` |
| `--model-checkpoint` | 已训练模型路径（跳过训练时使用） | `None` |
| `--segmentation-dir` | 分割数据目录（跳过转换时使用） | `None` |
| `--temp-dir` | 临时文件目录 | `temp/pipeline` |
| `--keep-temp` | 保留临时文件 | `False` |
| `--verify-onnx` | 验证导出的ONNX模型 | `True` |
| `--debug` | 启用调试模式 | `False` |

#### 示例

完整示例脚本见 `examples/segmentation_pipeline_example.sh`。

### 2. 指针仪表批量检测 (`pointer_meter_detection.py`)

批量检测图像中的仪表并裁剪仪表区域。

#### 功能说明

- 加载YOLO检测模型
- 批量处理指定目录下的图像
- 检测图像中的指针仪表
- 裁剪仪表区域并保存
- 生成可视化结果和统计报告

#### 使用方法

**基本使用**：
```bash
python scripts/pointer_meter_detection.py \
    --data-dir data/test_images/pointer \
    --target-dir outputs/detection_results
```

**指定模型和参数**：
```bash
python scripts/pointer_meter_detection.py \
    --data-dir data/test_images/pointer \
    --target-dir outputs/detection_results \
    --model models/detection/detection_model.pt \
    --conf 0.6 \
    --padding 30
```

**不保存可视化结果**：
```bash
python scripts/pointer_meter_detection.py \
    --data-dir data/test_images/pointer \
    --target-dir outputs/detection_results \
    --no-visualization
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data-dir` | 输入图像目录（必需） | - |
| `--target-dir` | 输出目录（必需） | - |
| `--model` | YOLO检测模型路径 | `models/detection/detection_model.pt` |
| `--conf` | 检测置信度阈值 | `0.5` |
| `--padding` | 裁剪时的边界扩展像素 | `20` |
| `--no-visualization` | 不保存可视化结果 | `False` |

#### 输出结构

```
target-dir/
├── cropped/              # 裁剪的仪表图像
│   ├── image1.jpg
│   └── image2_meter_01.jpg
├── visualizations/       # 检测可视化结果
│   ├── image1_detection.jpg
│   └── image2_detection.jpg
└── detection_results.json  # 检测结果统计
```

#### 示例

完整示例脚本见 `examples/pointer_meter_detection_example.sh`。


## 项目结构

```
pointMeterDetection/
├── README.md
├── app.py                    # Gradio Web应用
├── main.py                   # 应用入口
├── config/                   # 配置文件
│   ├── detection_config.yaml
│   └── segmentation_config.yaml
├── scripts/                  # 脚本目录
│   ├── segmentation_pipeline.py    # 分割训练流水线
│   ├── pointer_meter_detection.py    # 批量检测脚本
│   ├── train_detection.py            # 检测模型训练
│   ├── train_segmentation.py         # 分割模型训练
│   └── extract_meter_reading.py      # 读数提取算法
├── examples/                 # 示例脚本
│   ├── segmentation_pipeline_example.sh
│   └── pointer_meter_detection_example.sh
├── models/                   # 模型文件
│   ├── detection/
│   └── segmentation/
├── data/                     # 数据集目录
└── outputs/                  # 输出结果
```

## 📄 许可证

MIT License
