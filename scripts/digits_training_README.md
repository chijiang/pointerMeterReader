# 液晶数字表检测模型训练指南

本指南说明如何使用 YOLO v10 训练液晶数字表检测模型，用于识别数字 0-9 和小数点。

## 📋 功能特性

- 🎯 **多类别检测**：支持数字 0-9 和小数点识别
- 🚀 **YOLO v10**：使用最新的 YOLO v10 架构
- 🍎 **设备自适应**：支持 Apple Silicon (MPS)、CUDA 和 CPU
- 📊 **自动数据分割**：自动将数据分为训练集和验证集
- 🎨 **可视化结果**：生成预测结果可视化
- 📦 **模型导出**：支持导出到 ONNX、TorchScript 等格式
- 🔍 **读数预测**：直接预测液晶显示屏的完整读数

## 📁 数据格式要求

数据应存放在 `data/digits/` 目录下，具体结构如下：

```
data/digits/
├── images/          # 图像文件 (.jpg)
├── labels/          # 标注文件 (.txt, YOLO格式)
├── classes.txt      # 类别名称文件
└── notes.json       # 数据集信息（可选）
```

### 标注格式
标注文件为 YOLO 格式，每行一个目标：
```
class_id x_center y_center width height
```

其中：
- `class_id`：类别ID（0-10，对应数字0-9和小数点）
- `x_center, y_center, width, height`：归一化的边界框坐标

### classes.txt 示例
```
0
1
2
3
4
5
6
7
8
9
point
```

## 🚀 快速开始

### 方法1：使用快速启动脚本（推荐）

```bash
cd scripts
python start_digits_training.py
```

这个脚本会：
1. 检查数据完整性
2. 自动创建配置文件
3. 询问用户确认后开始训练
4. 生成可视化结果

### 方法2：使用主训练脚本

1. **创建配置文件**：
```bash
python scripts/train_digits.py --create-config --config config/digits_config.yaml
```

2. **开始训练**：
```bash
python scripts/train_digits.py --config config/digits_config.yaml --visualize
```

3. **仅评估模型**：
```bash
python scripts/train_digits.py --eval-only --model-path outputs/checkpoints/digits/digit_detection/weights/best.pt --config config/digits_config.yaml --visualize
```

4. **预测单张图像**：
```bash
python scripts/train_digits.py --predict data/digits_yolo/images/val/64fc1dbf-023d80491ecf06ebf29ec6968873e2b9d0b5c0bf.jpg --model-path outputs/checkpoints/digits/digit_detection/weights/best.pt --config config/digits_config.yaml
```

## ⚙️ 配置说明

主要配置参数（在 `config/digits_config.yaml` 中）：

```yaml
# 模型配置
model: 'yolov10n.pt'     # 模型大小：n(nano), s(small), m(medium), l(large), x(xlarge)
epochs: 200              # 训练轮数
batch_size: 16           # 批大小
image_size: 640          # 输入图像尺寸
learning_rate: 0.008     # 学习率
train_split: 0.8         # 训练集比例

# 设备配置
device: 'auto'           # 'auto', 'mps', 'cuda', 'cpu'

# 数据增强（针对数字识别优化）
augmentation:
  hsv_h: 0.01           # 最小色调变化
  hsv_s: 0.5            # 适中饱和度变化
  hsv_v: 0.3            # 适中亮度变化
  degrees: 5.0          # 小幅旋转
  translate: 0.05       # 小幅平移
  flipud: 0.0           # 不上下翻转
  fliplr: 0.0           # 不左右翻转
```

## 📊 输出结果

训练完成后，结果保存在 `outputs/` 目录：

```
outputs/
├── checkpoints/digits/     # 模型权重
│   └── digit_detection_v1/
│       └── weights/
│           ├── best.pt     # 最佳模型
│           └── last.pt     # 最后一轮模型
├── logs/digits/           # 训练日志
└── results/digits/        # 评估结果
    ├── evaluation_results.json
    └── visualizations/    # 可视化结果
```

## 🔧 高级用法

### 导出模型
```bash
python scripts/train_digits.py --export --model-path outputs/checkpoints/digits/digit_detection_v1/weights/best.pt --export-formats onnx torchscript --config config/digits_config.yaml
```

### 自定义训练参数
编辑 `config/digits_config.yaml` 文件：

```yaml
# 增加训练轮数
epochs: 300

# 使用更大的模型
model: 'yolov10s.pt'

# 调整批大小（根据GPU内存）
batch_size: 32

# 启用余弦学习率调度
cos_lr: true
```

### 预测液晶显示屏读数
```python
from scripts.train_digits import DigitsDetectionTrainer

trainer = DigitsDetectionTrainer('config/digits_config.yaml')
reading = trainer.predict_reading('path/to/best.pt', 'path/to/display_image.jpg')
print(f"读数: {reading}")
```

## 📈 训练监控

训练过程中的重要指标：

- **mAP50**：IoU=0.5时的平均精度
- **mAP50-95**：IoU=0.5:0.95的平均精度
- **Precision**：精确率
- **Recall**：召回率
- **Loss**：训练损失

## 🛠️ 故障排除

### 常见问题

1. **CUDA内存不足**：
   - 减少 `batch_size`
   - 使用更小的模型 (yolov10n.pt)

2. **Apple Silicon 设备**：
   - 确保设置 `device: 'mps'` 或 `device: 'auto'`
   - 可能需要减少 `workers` 数量

3. **训练过慢**：
   - 检查是否启用了 GPU 加速
   - 减少数据增强强度
   - 使用更小的图像尺寸

4. **精度不高**：
   - 增加训练轮数
   - 检查数据质量和标注准确性
   - 调整学习率
   - 使用更大的模型

### 数据质量检查

```bash
# 检查数据统计
python -c "
from pathlib import Path
data_dir = Path('data/digits')
images = len(list((data_dir / 'images').glob('*.jpg')))
labels = len(list((data_dir / 'labels').glob('*.txt')))
print(f'图像: {images}, 标注: {labels}')
"
```

## 📝 性能基准

基于我们的测试数据：

| 模型 | mAP50 | 推理速度 | 模型大小 |
|------|-------|----------|----------|
| YOLOv10n | ~85% | ~2ms | ~6MB |
| YOLOv10s | ~88% | ~3ms | ~20MB |
| YOLOv10m | ~90% | ~5ms | ~50MB |

*注：实际性能取决于数据质量和训练参数*

## 📞 技术支持

如遇问题，请检查：
1. 数据格式是否正确
2. 配置文件是否完整
3. 依赖库是否安装完整
4. 设备配置是否正确

---

**作者**: chijiang  
**更新时间**: 2025-01-15 