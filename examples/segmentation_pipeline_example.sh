#!/bin/bash
# 分割模型训练流水线示例
# Segmentation Pipeline Example

# 示例1: 完整流水线（从COCO数据到ONNX模型）
# Example 1: Full pipeline (from COCO data to ONNX model)
python scripts/segmentation_pipeline.py \
    --coco-json examples/example_data/coco_data/result_coco.json \
    --coco-images examples/example_data/coco_data/images \
    --config config/segmentation_config.yaml \
    --output-onnx models/test/segmentation_model.onnx \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 0.001

# 示例2: 跳过数据转换（使用已有分割数据）
# Example 2: Skip data conversion (use existing segmentation data)
python scripts/segmentation_pipeline.py \
    --skip-conversion \
    --segmentation-dir data/segmentation \
    --config config/segmentation_config.yaml \
    --output-onnx models/segmentation/segmentation_model.onnx

# 示例3: 只导出ONNX模型（跳过训练）
# Example 3: Export ONNX only (skip training)
python scripts/segmentation_pipeline.py \
    --skip-conversion \
    --skip-training \
    --model-checkpoint outputs/segmentation/best_model.pth \
    --config config/segmentation_config.yaml \
    --output-onnx models/segmentation/segmentation_model.onnx

# 示例4: 使用默认参数
# Example 4: Use default parameters
python scripts/segmentation_pipeline.py

