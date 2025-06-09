# 液晶数字OCR提取

这个目录包含专门用于从液晶屏显示中提取数字的OCR（光学字符识别）脚本。

## 功能特性

### 🔤 高精度数字识别
- **EasyOCR引擎**: 基于深度学习的高精度OCR，特别适合液晶屏数字（推荐）
- **PaddleOCR引擎**: 百度开源，适合液晶屏数字（可选）
- **Tesseract引擎**: 传统OCR引擎，速度快，适合清晰文本
- **智能预处理**: 自动调整图像以优化OCR效果

### 📊 智能结果验证
- **数字格式检测**: 自动识别整数、小数等格式
- **合理性校验**: 检查数值范围和格式合理性
- **置信度评估**: 为每个识别结果提供置信度分数

### 🎯 专门优化
- **液晶屏适配**: 针对液晶显示器的特点进行优化
- **噪声过滤**: 自动过滤非数字内容
- **多尺度处理**: 支持不同大小的数字显示

## 文件说明

- `digital_ocr_extractor.py` - 主要的OCR提取脚本
- `demo_ocr.py` - OCR功能演示脚本
- `README.md` - 本说明文档

## 安装依赖

### 必需依赖
```bash
pip install opencv-python numpy matplotlib pillow tqdm
```

### OCR引擎

#### EasyOCR (推荐)
```bash
pip install easyocr
```
- 优点：高精度，支持多种语言，易于使用，稳定可靠
- 缺点：首次使用需要下载模型，文件较大

#### PaddleOCR (可选)
```bash
pip install paddleocr
```
- 优点：下载速度快，准确率高，对中文数字识别优秀，模型轻量
- 缺点：可能存在兼容性问题，功能相对简单

#### Tesseract
```bash
pip install pytesseract
```
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- Windows: 下载并安装Tesseract-OCR

## 使用方法

### 1. 直接使用OCR提取器

```bash
# 单张图像
python digital_ocr_extractor.py --input path/to/image.jpg

# 批量处理
python digital_ocr_extractor.py --input path/to/images/

# 指定OCR引擎
python digital_ocr_extractor.py --input path/to/images/ --engine paddleocr
python digital_ocr_extractor.py --input path/to/images/ --engine tesseract

# 自定义输出目录
python digital_ocr_extractor.py --input path/to/images/ --output custom_output/
```

### 2. 交互式演示

```bash
python demo_ocr.py
```

演示菜单包含：
- 单张图像OCR演示
- 批量OCR演示
- 不同OCR引擎对比
- 对增强图像的OCR
- 依赖检查

### 3. 通过主启动脚本

从项目根目录运行：
```bash
cd scripts/digital_meter_detection/
python run.py
```

选择选项 "6. 🔤 OCR数字提取演示"

## 输出结果

### 目录结构
```
outputs/digital_ocr/ocr_results_YYYYMMDD_HHMMSS/
├── extracted_text/          # 详细OCR结果 (JSON格式)
│   ├── image1_ocr_results.json
│   └── image2_ocr_results.json
├── visualization/           # OCR检测可视化
│   ├── image1_ocr_visualization.png
│   └── image2_ocr_visualization.png
├── analysis/               # 分析报告
├── ocr_extraction_results.json    # 批量处理总结
├── ocr_summary.json               # 统计摘要
└── ocr_report.md                  # Markdown报告
```

### 结果格式

#### JSON结果示例
```json
{
  "image_path": "path/to/image.jpg",
  "image_name": "sample_image",
  "extracted_value": 123.45,
  "confidence": 0.987,
  "total_detections": 1,
  "timestamp": "2025-06-09T10:30:45"
}
```

#### 详细OCR结果
```json
{
  "raw_results": [...],
  "validated_results": [
    {
      "text": "123.45",
      "clean_text": "123.45",
      "confidence": 0.987,
      "bbox": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "is_valid": true,
      "value": 123.45,
      "format_type": "decimal"
    }
  ],
  "best_result": {...},
  "extraction_summary": {...}
}
```

## 技术细节

### 图像预处理
1. **灰度转换**: 将彩色图像转为灰度以提高OCR效果
2. **背景反转**: 确保文字为黑色，背景为白色
3. **尺寸调整**: 放大小图像以达到最佳OCR分辨率
4. **噪声去除**: 使用中值滤波去除噪声

### 结果验证
1. **格式检查**: 验证是否为有效的数字格式
2. **范围检查**: 检查数值是否在合理范围内
3. **精度检查**: 验证小数位数是否合理
4. **置信度筛选**: 根据OCR置信度过滤结果

### OCR引擎对比

| 特性 | EasyOCR | PaddleOCR | Tesseract |
|------|---------|-----------|-----------|
| 精度 | 高 | 高 | 中等 |
| 速度 | 中等 | 快 | 快 |
| 稳定性 | 优秀 | 一般 | 优秀 |
| 安装 | 简单 | 简单 | 需要系统安装 |
| 兼容性 | 优秀 | 可能有问题 | 优秀 |
| 模型大小 | 大 | 中等 | 小 |
| GPU支持 | 是 | 是 | 否 |

## 最佳实践

### 图像质量要求
- **分辨率**: 建议至少100x300像素
- **对比度**: 数字与背景有明显对比
- **清晰度**: 避免模糊和失焦
- **噪声**: 尽量减少背景噪声

### 使用建议
1. **图像增强**: 先使用图像增强功能改善图像质量
2. **引擎选择**: 
   - **EasyOCR**: 推荐首选，稳定可靠，对复杂场景适应性好
   - **PaddleOCR**: 可选，但可能存在兼容性问题
   - **Tesseract**: 适合清晰图像，无需下载模型
3. **批量处理**: 大量图像建议使用批量模式
4. **结果验证**: 始终检查提取结果的合理性

## 故障排除

### 常见问题

#### 1. EasyOCR安装失败
```bash
# 升级pip
pip install --upgrade pip

# 安装torch (如果需要)
pip install torch torchvision

# 重新安装EasyOCR
pip install easyocr
```

#### 2. PaddleOCR兼容性问题
如果遇到PaddleOCR兼容性问题，建议：
```bash
# 使用EasyOCR作为替代
pip install easyocr

# 或者尝试不同版本的PaddleOCR
pip install paddlepaddle==2.5.2
pip install paddleocr
```

#### 3. Tesseract不可用
- 确保系统已安装Tesseract
- 检查PATH环境变量
- 在某些系统上可能需要指定tesseract路径

#### 3. 识别精度低
- 检查图像质量（分辨率、对比度）
- 尝试使用图像增强功能预处理
- 尝试不同的OCR引擎
- 检查图像中是否有非数字干扰

#### 4. 内存不足
- 减少批量处理的图像数量
- 关闭GPU模式（EasyOCR）
- 降低图像分辨率

### 性能优化

1. **GPU加速**: 使用支持GPU的EasyOCR可显著提升速度
2. **批量大小**: 适当的批量大小可以平衡速度和内存使用
3. **图像尺寸**: 避免过度放大图像，适中即可
4. **预处理**: 合适的预处理可以提高识别速度和精度

## 与其他模块的集成

### 与图像增强的配合
```bash
# 1. 先进行图像增强
python enhancement/digital_display_enhancer.py --input data/digital_meters/

# 2. 对增强结果进行OCR
python ocr/digital_ocr_extractor.py --input outputs/digital_enhancement/latest/2_enhanced/
```

### 完整工作流程
1. **数据准备**: 收集液晶屏图像
2. **YOLO检测**: 定位液晶显示区域
3. **图像增强**: 改善显示质量
4. **OCR提取**: 提取数字内容
5. **结果验证**: 验证和后处理

## 技术支持

如有问题或建议，请查看：
- 项目文档
- 错误日志输出
- 依赖检查结果

---
*更新时间: 2025-06-09* 