# 🔢 液晶数字表读数提取系统 - 使用说明

## 📋 概述

这是一个完整的液晶数字表读数提取系统，集成了目标检测、图像增强和OCR识别功能。

## 🛠️ 安装要求

### 必需依赖
```bash
# 基础依赖
pip install opencv-python torch ultralytics

# OCR引擎（选择一个或多个）
pip install easyocr              # 推荐，效果好
pip install paddleocr            # 可选
pip install pytesseract          # 可选，需要单独安装tesseract
```

### 可选依赖
```bash
pip install matplotlib seaborn  # 可视化
pip install pandas              # 数据分析
```

## 🚀 使用方法

### 方法1：简单启动脚本（推荐）

从项目根目录运行：
```bash
python scripts/digital_meter_detection/run_digital_reading.py
```

这个脚本会：
- 自动检查依赖
- 查找模型文件
- 提供友好的交互界面
- 自动处理路径问题

### 方法2：直接调用主脚本

从项目根目录运行：
```bash
# 单张图像
python scripts/digital_meter_detection/digital_meter_reading.py \
  --input your_image.jpg \
  --model models/detection/digital_detection_model.pt

# 图像目录
python scripts/digital_meter_detection/digital_meter_reading.py \
  --input image_folder/ \
  --model models/detection/digital_detection_model.pt

# 自定义参数
python scripts/digital_meter_detection/digital_meter_reading.py \
  --input your_image.jpg \
  --model models/detection/digital_detection_model.pt \
  --ocr-engine paddleocr \
  --device cuda \
  --confidence 0.7 \
  --output custom_output_dir
```

### 方法3：通过主菜单

```bash
python scripts/digital_meter_detection/run.py
# 选择 "8. 🔢 完整读数提取（检测+增强+OCR）"
```

### 方法4：Python API

```python
from scripts.digital_meter_detection.digital_meter_reading import DigitalMeterReader

# 创建读数提取器
reader = DigitalMeterReader(
    model_path="models/detection/digital_detection_model.pt",
    ocr_engine="easyocr",
    device="auto"
)

# 处理单张图像
result = reader.process_single_image("your_image.jpg")
print(f"检测到: {result['detections_count']} 个显示屏")
print(f"成功读取: {result['successful_readings']} 个数值")

# 批量处理
batch_result = reader.process_batch("image_folder/")
```

## ⚙️ 参数说明

| 参数 | 说明 | 默认值 | 选项 |
|------|------|--------|------|
| `--input` | 输入图像文件或目录 | 必需 | 任何图像文件/目录 |
| `--model` | YOLO检测模型路径 | `models/detection/digital_detection_model.pt` | 任何.pt文件 |
| `--output` | 输出目录 | 自动生成时间戳目录 | 任何目录路径 |
| `--ocr-engine` | OCR引擎 | `easyocr` | `easyocr`, `paddleocr`, `tesseract` |
| `--device` | 计算设备 | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `--confidence` | 检测置信度阈值 | `0.5` | 0.0-1.0 |
| `--no-enhancement` | 禁用图像增强 | False | 标志参数 |
| `--debug` | 启用调试模式 | False | 标志参数 |

## 📁 输出结构

处理完成后，会在输出目录生成以下结构：
```
outputs/digital_meter_reading/reading_YYYYMMDD_HHMMSS/
├── 1_detection/              # 检测可视化结果
│   └── image_detection.jpg   # 标注了检测框的图像
├── 2_cropped/                # 裁剪的显示屏区域
│   ├── image_crop_00.jpg     # 第一个检测区域
│   └── image_crop_01.jpg     # 第二个检测区域
├── 3_enhanced/               # 增强后的图像
│   ├── image_crop_00_enhanced.jpg
│   └── enhancer_workspace/   # 增强器工作空间
├── 4_ocr_results/            # OCR详细结果
│   ├── image_crop_00_ocr.json
│   └── image_crop_00_ocr_visualization.jpg
├── 5_visualization/          # 最终可视化结果
│   └── image_final_result.jpg  # 标注了所有结果的图像
├── batch_results.json        # 结构化结果数据
├── batch_report.md          # 处理报告
└── digital_meter_reading.log # 详细处理日志
```

## 🔧 常见问题排除

### 1. 模型文件不存在
```
❌ 模型文件不存在: models/detection/digital_detection_model.pt
```
**解决方案：**
- 使用训练脚本生成模型：`python scripts/digital_meter_detection/run.py` → 选择训练功能
- 或者指定其他模型路径：`--model your_model.pt`

### 2. 路径错误
```
python: can't open file 'digital_meter_detection/...'
```
**解决方案：**
- 确保在项目根目录执行命令
- 使用完整路径：`python scripts/digital_meter_detection/...`

### 3. 依赖缺失
```
❌ 缺少必要的依赖库: ultralytics
```
**解决方案：**
```bash
pip install ultralytics opencv-python torch easyocr
```

### 4. OCR识别失败
```
⚠️ OCR失败: image_crop_00_enhanced.jpg
```
**可能原因：**
- 显示屏区域太小或不清晰
- 文字颜色对比度不够
- 检测框不准确

**解决方案：**
- 调整检测置信度：`--confidence 0.3`
- 尝试不同OCR引擎：`--ocr-engine paddleocr`
- 手动检查增强后的图像质量

### 5. GPU内存不足
```
CUDA out of memory
```
**解决方案：**
- 使用CPU：`--device cpu`
- 或使用MPS（Mac）：`--device mps`

## 📊 性能优化建议

### 1. 设备选择
- **GPU可用**：自动选择最佳设备
- **仅CPU**：指定 `--device cpu`
- **Mac用户**：可尝试 `--device mps`

### 2. OCR引擎选择
- **EasyOCR**：准确率高，速度中等（推荐）
- **PaddleOCR**：速度快，准确率良好
- **Tesseract**：速度快，但需要额外配置

### 3. 批量处理
- 使用目录输入自动批量处理
- 大量图像时考虑分批处理

## 🔬 测试功能

运行功能测试：
```bash
python scripts/digital_meter_detection/test_digital_meter_reading.py
```

这会测试：
- 模块导入
- 图像增强
- OCR提取
- 完整流水线

## 🤝 集成到其他项目

```python
# 最小化集成示例
import sys
from pathlib import Path

# 添加项目路径
project_root = Path("path/to/pointMeterDetection")
sys.path.append(str(project_root))

from scripts.digital_meter_detection.digital_meter_reading import DigitalMeterReader

# 创建读数器
reader = DigitalMeterReader(
    model_path=str(project_root / "models/detection/digital_detection_model.pt"),
    ocr_engine="easyocr"
)

# 处理图像
result = reader.process_single_image("image.jpg")
print(f"读取结果: {result}")
```

## 📞 技术支持

如果遇到问题：
1. 查看生成的日志文件：`digital_meter_reading.log`
2. 使用调试模式：`--debug`
3. 检查各阶段的中间结果图像
4. 运行功能测试验证环境配置

---

*最后更新: 2025-06-09*  
*系统版本: v1.0* 