#!/bin/bash
# 指针仪表批量检测和裁剪示例
# Pointer Meter Detection and Cropping Example

# 示例1: 基本使用（使用默认模型和参数）
# Example 1: Basic usage (with default model and parameters)
python scripts/pointer_meter_detection.py \
    --data-dir examples/example_data/original_img \
    --target-dir examples/output/cropped_img

# 示例2: 指定模型和置信度阈值
# Example 2: Specify model and confidence threshold
python scripts/pointer_meter_detection.py \
    --data-dir examples/example_data/original_img \
    --target-dir examples/output/cropped_img \
    --model models/detection/detection_model.pt \
    --conf 0.3 \
    --padding 30

# 示例3: 不保存可视化结果（仅裁剪）
# Example 3: Skip visualization (crop only)
python scripts/pointer_meter_detection.py \
    --data-dir examples/example_data/original_img \
    --target-dir examples/output/cropped_img \
    --no-visualization

# 示例4: 高置信度检测（更严格的检测）
# Example 4: High confidence detection (stricter detection)
python scripts/pointer_meter_detection.py \
    --data-dir examples/example_data/original_img \
    --target-dir examples/output/cropped_img \
    --conf 0.3 \
    --padding 20

