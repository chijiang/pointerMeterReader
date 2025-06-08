# 工业仪表读数自动识别系统

## 项目概述

本项目是一个基于深度学习的工业仪表读数自动识别系统，能够从监控摄像头画面中自动提取压力表、温度计等圆形指针式仪表的读数。系统采用三阶段处理流程，结合了YOLO目标检测、DeepLabV3+语义分割和几何算法，实现从图像到数值的完整转换。

## 🚀 主要特性

- **完整的AI流水线**：检测 → 分割 → 读数提取
- **ONNX推理加速**：支持ONNX Runtime，避免PyTorch依赖问题
- **智能后处理**：自动去除分割噪声，优化指针和刻度边界
- **Web界面**：基于Gradio的友好用户界面
- **Apple Silicon优化**：支持MPS加速（M1/M2芯片）
- **灵活配置**：支持自定义刻度范围和后处理参数

## 技术架构

### 处理流程
```
原始图像 → 仪表检测(YOLO) → 区域裁剪 → 语义分割(ONNX) → 后处理优化 → 读数计算 → 最终结果
```

### 三个核心阶段

#### 阶段1：仪表位置检测（YOLOv10）
- **功能**：从完整的监控画面中定位并框选出仪表区域
- **技术**：YOLOv10 目标检测模型
- **输入**：原始监控图像
- **输出**：仪表的边界框坐标和置信度
- **特点**：自动检测多个仪表，支持批量处理

#### 阶段2：指针与刻度分割（DeepLabV3+ + ONNX）
- **功能**：对检测到的仪表区域进行像素级语义分割
- **技术**：DeepLabV3+ (ResNet50) + ONNX Runtime推理
- **分割类别**：
  - 背景（Background）
  - 指针（Pointer）
  - 刻度线（Scale）
- **后处理功能**：
  - 噪声点去除（开运算）
  - 连通域分析（保留主要区域）
  - 指针细化（轻度腐蚀）
  - 刻度收缩（防止边界外移）
  - 孔洞填充

#### 阶段3：读数计算与转换
- **功能**：基于分割结果计算指针角度并转换为实际读数
- **算法**：
  - 指针中心点定位
  - 指针方向角度计算
  - 刻度范围映射
  - 角度插值与读数输出

## 项目结构

```
pointMeterDetection/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖包
├── pyproject.toml           # uv项目配置
├── app.py                   # Gradio Web应用主程序
├── config/                  # 配置文件目录
│   ├── detection_config.yaml # YOLO检测配置
│   └── segmentation_config.yaml # 分割模型配置
├── data/                    # 数据集目录
│   ├── detection/           # 检测数据集（COCO格式）
│   │   ├── train2017/       # 训练图像
│   │   ├── val2017/         # 验证图像
│   │   └── annotations/     # 标注文件
│   └── segmentation/        # 分割数据集（Pascal VOC格式）
│       ├── JPEGImages/      # 原始图像
│       ├── SegmentationClass_unified/ # 统一后的分割标注
│       └── ImageSets/       # 数据集划分
├── scripts/                 # 训练和工具脚本
│   ├── train_detection.py   # 检测模型训练
│   ├── train_segmentation.py # 分割模型训练（支持--export-only）
│   └── extract_meter_reading.py # 读数提取算法
├── tools/                   # 数据处理工具
│   └── data_preparation/    # 数据预处理工具
└── outputs/                 # 输出结果
    ├── checkpoints/         # 模型检查点
    ├── segmentation/        # 分割模型输出
    │   ├── best_model.pth   # 最佳PyTorch模型
    │   └── exported/        # 导出的模型
    │       └── segmentation_model.onnx # ONNX模型
    └── results/             # 推理结果
```

## 环境要求

### 硬件要求
- **推荐**：Apple Silicon (M1/M2) 或 CUDA兼容GPU
- **最低**：4GB内存，现代CPU
- **存储**：2GB可用空间

### 软件要求
- Python 3.11+
- 支持的操作系统：macOS, Linux, Windows

### 主要依赖
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
onnxruntime>=1.22.0
opencv-python>=4.6.0
gradio>=4.0.0
matplotlib>=3.5.0
numpy>=1.21.0
PyYAML>=6.0
```

## 🚀 快速开始

### 1. 环境安装

#### 使用uv（推荐）
```bash
# 克隆项目
git clone https://github.com/chijiang/pointerMeterReader.git
cd pointMeterDetection

# 使用uv创建环境并安装依赖
uv sync
```

#### 使用pip
```bash
# 克隆项目
git clone https://github.com/chijiang/pointerMeterReader.git
cd pointMeterDetection

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动Web应用

```bash
# 启动Gradio Web界面
python app.py
```

然后在浏览器中访问 `http://localhost:7860` 使用Web界面进行仪表读数识别。

### 3. 模型训练（可选）

#### 检测模型训练
```bash
# 验证检测数据集
python tools/data_preparation/check_data.py

# 开始训练YOLOv10检测模型
python scripts/train_detection.py --config config/detection_config.yaml

# 评估现有模型
python scripts/train_detection.py --eval-only --model-path outputs/checkpoints/detection/meter_detection_v1/weights/best.pt --visualize
```

#### 分割模型训练
```bash
# 检查分割数据集
python tools/data_preparation/check_segmentation_data.py --data_dir data/segmentation --mask_dir SegmentationClass_unified --visualize

# 开始训练DeepLabV3+分割模型
python scripts/train_segmentation.py --config config/segmentation_config.yaml

# 导出ONNX模型（用于加速推理）
python scripts/train_segmentation.py --export-only --export-formats onnx torchscript
```

## 📊 功能详解

### Web界面功能

1. **图像上传**：支持JPG、PNG等常见格式
2. **参数调节**：
   - 检测置信度阈值
   - 刻度最小/最大值设置
3. **可视化流程**：
   - 检测结果（边界框）
   - 裁剪的仪表区域
   - 分割掩码（带统计信息）
   - 最终读数结果
4. **结果输出**：数值读数和置信度

### 后处理配置

系统支持灵活的后处理配置，可在 `app.py` 中调整：

```python
post_process_config = {
    'remove_noise': True,           # 去除噪声点
    'keep_largest_component': False, # 保留最大连通域
    'pointer_erosion': 1,           # 指针腐蚀强度
    'scale_erosion': 3,             # 刻度腐蚀强度（防外移）
    'fill_holes': False,            # 填充小孔洞
    'connect_scale_lines': False    # 连接断裂刻度线
}
```

### 数据集格式

#### 检测数据集（COCO格式）
```json
{
    "images": [{"id": 1, "file_name": "image1.jpg", "width": 1920, "height": 1080}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h]}],
    "categories": [{"id": 1, "name": "meter", "supercategory": "instrument"}]
}
```

#### 分割数据集（Pascal VOC格式）
- 图像：`JPEGImages/xxx.jpg`
- 标注：`SegmentationClass_unified/xxx.png`
- 类别映射：`0=背景`, `1=指针`, `2=刻度`

## 🔧 高级使用

### 测试后处理效果
```bash
# 运行后处理效果测试
python test_post_processing.py
```

### ONNX模型测试
```bash
# 测试ONNX模型加载和推理
python test_onnx_simple.py
```

### 单独使用读数提取算法
```python
from scripts.extract_meter_reading import MeterReader

reader = MeterReader(scale_range=(0.0, 1.6))
reading = reader.process_single_meter(image, segmentation_mask)
```

## 🎯 性能指标

### 检测模型
- **数据集**：1836张图像，2082个标注
- **训练集**：1670张图像，1898个标注
- **验证集**：166张图像，184个标注

### 分割模型
- **架构**：DeepLabV3+ (ResNet50)
- **输入尺寸**：224×224
- **输出格式**：ONNX (160MB)
- **推理设备**：CPU/MPS/CUDA

### 系统性能
- **检测速度**：~60ms/图（YOLOv10）
- **分割速度**：~30ms/图（ONNX CPU）
- **端到端延迟**：<200ms/图
- **内存占用**：<2GB

## 🐛 故障排除

### 常见问题

1. **ONNX模型加载失败**
   ```bash
   pip install onnxruntime
   ```

2. **分割结果有噪声**
   - 调整后处理配置中的腐蚀参数
   - 启用噪声去除：`'remove_noise': True`

3. **检测不到仪表**
   - 降低置信度阈值
   - 检查图像质量和光照条件

4. **读数不准确**
   - 验证刻度范围设置
   - 检查指针和刻度分割质量

## 📝 更新日志

### v1.0.0 (当前版本)
- ✅ 完整的三阶段识别流水线
- ✅ ONNX推理支持，避免PyTorch依赖问题
- ✅ 智能后处理，去除噪声和优化边界
- ✅ Gradio Web界面
- ✅ Apple Silicon (MPS) 支持
- ✅ 灵活的配置系统

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进本项目！

---

*本项目专注于工业仪表读数识别，适用于压力表、温度计、流量计等圆形指针式仪表的自动化读数场景。* 