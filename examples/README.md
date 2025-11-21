# 示例脚本说明

本目录包含两个主要脚本的使用示例。

## segmentation_pipeline_example.sh

分割模型训练流水线示例，展示如何使用 `segmentation_pipeline.py` 进行端到端训练。

### 运行示例

```bash
# 给脚本添加执行权限
chmod +x examples/segmentation_pipeline_example.sh

# 运行示例（根据需要修改路径和参数）
./examples/segmentation_pipeline_example.sh
```

### 示例说明

1. **完整流水线**：从COCO数据转换到ONNX模型导出的完整流程
2. **跳过数据转换**：使用已有的分割数据集直接训练
3. **只导出ONNX**：使用已训练模型直接导出ONNX格式
4. **默认参数**：使用脚本默认参数运行

## pointer_meter_detection_example.sh

指针仪表批量检测示例，展示如何使用 `pointer_meter_detection.py` 进行批量检测和裁剪。

### 运行示例

```bash
# 给脚本添加执行权限
chmod +x examples/pointer_meter_detection_example.sh

# 运行示例（根据需要修改路径和参数）
./examples/pointer_meter_detection_example.sh
```

### 示例说明

1. **基本使用**：使用默认模型和参数进行检测
2. **指定参数**：自定义模型路径、置信度阈值和边界扩展
3. **仅裁剪**：不保存可视化结果，只保存裁剪的仪表图像
4. **高置信度**：使用更高的置信度阈值进行更严格的检测

## 注意事项

- 运行前请确保已安装所有依赖
- 根据实际情况修改数据路径和模型路径
- 确保输出目录有写入权限

