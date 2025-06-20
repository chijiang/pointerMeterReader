# LED颜色识别模型训练

此项目用于训练识别LED灯颜色（绿色、红色、黄色）的YOLO模型。

## 📁 数据结构

```
data/led_color/
├── images/          # 图像文件
├── labels/          # YOLO格式标签文件
├── classes.txt      # 类别定义
└── notes.json       # 数据集信息
```

## 🚀 快速开始

### 1. 直接训练（推荐）

```bash
# 使用默认配置开始训练
python run_led_color_training.py

# 自定义参数训练
python run_led_color_training.py --epochs 150 --batch 32 --imgsz 640
```

### 2. 使用详细脚本

```bash
# 训练模型
python scripts/train_led_color.py --mode train

# 评估模型
python scripts/train_led_color.py --mode eval --model path/to/best.pt

# 可视化预测结果
python scripts/train_led_color.py --mode visualize --model path/to/best.pt

# 导出模型
python scripts/train_led_color.py --mode export --model path/to/best.pt
```

## 📊 模型评估

训练完成后，可以评估模型性能：

```bash
# 评估最新训练的模型
python run_led_color_training.py --eval

# 评估指定模型
python run_led_color_training.py --eval --model path/to/model.pt
```

## 🎨 结果可视化

生成预测结果的可视化图像：

```bash
# 可视化最新训练的模型结果
python run_led_color_training.py --viz

# 可视化指定模型结果
python run_led_color_training.py --viz --model path/to/model.pt
```

## 📤 模型导出

将训练好的模型导出为不同格式：

```bash
# 导出最新训练的模型
python run_led_color_training.py --export

# 导出指定模型
python run_led_color_training.py --export --model path/to/model.pt
```

## ⚙️ 配置说明

训练配置位于 `configs/led_color_config.yaml`：

```yaml
# 模型配置
model:
  name: "yolo11n.pt"  # 预训练模型
  pretrained: true

# 训练配置
training:
  epochs: 100         # 训练轮数
  imgsz: 640         # 输入图像尺寸
  batch: 16          # 批次大小
  lr0: 0.01          # 初始学习率
  patience: 50       # 早停耐心值
  
# 设备配置
device: "auto"       # 自动选择最佳设备
```

## 📈 类别信息

模型识别3种LED颜色：

- **0**: green（绿色）
- **1**: red（红色）  
- **2**: yellow（黄色）

## 📂 输出目录

训练结果保存在以下目录：

```
outputs/
├── checkpoints/led_color/     # 模型权重文件
├── logs/led_color/           # 训练日志
└── results/led_color/        # 评估结果和可视化
    ├── evaluation_results.json
    ├── training_summary.json
    └── visualizations/       # 预测结果图像
```

## 🔧 环境要求

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- OpenCV
- 其他依赖见 requirements.txt

## 💡 使用技巧

1. **数据量较小**：使用较小的模型（yolo11n.pt）和适当的学习率
2. **提高精度**：增加训练轮数或使用更大的模型（yolo11s.pt, yolo11m.pt）
3. **加速训练**：在GPU/MPS上训练，启用混合精度（amp=True）
4. **过拟合问题**：增加数据增强或降低学习率

## 📞 问题反馈

如有问题，请检查：
1. 数据路径是否正确
2. 标签格式是否为YOLO格式
3. 环境依赖是否安装完整
4. 设备配置是否适当 