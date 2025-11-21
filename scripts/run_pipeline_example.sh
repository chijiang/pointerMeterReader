#!/bin/bash

# 分割模型端到端训练流水线示例脚本
# 从COCO数据集到ONNX模型的完整流程

echo "=========================================="
echo "分割模型端到端训练流水线"
echo "=========================================="

# 设置变量
COCO_JSON="data/coco_data/result_coco.json"
COCO_IMAGES="data/coco_data/images"
CONFIG="config/segmentation_config.yaml"
OUTPUT_ONNX="models/segmentation/segmentation_model.onnx"

# 示例1: 完整流水线（推荐用于生产环境）
echo -e "\n示例1: 完整流水线"
echo "python3 scripts/segmentation_pipeline.py \\"
echo "    --coco-json $COCO_JSON \\"
echo "    --coco-images $COCO_IMAGES \\"
echo "    --config $CONFIG \\"
echo "    --output-onnx $OUTPUT_ONNX"

# 示例2: 快速测试（少量epoch）
echo -e "\n示例2: 快速测试（5个epoch）"
echo "python3 scripts/segmentation_pipeline.py \\"
echo "    --coco-json $COCO_JSON \\"
echo "    --coco-images $COCO_IMAGES \\"
echo "    --config $CONFIG \\"
echo "    --output-onnx $OUTPUT_ONNX \\"
echo "    --epochs 5 \\"
echo "    --batch-size 4"

# 示例3: 使用已有分割数据（跳过转换）
echo -e "\n示例3: 使用已有分割数据"
echo "python3 scripts/segmentation_pipeline.py \\"
echo "    --skip-conversion \\"
echo "    --segmentation-dir data/segmentation_new \\"
echo "    --config $CONFIG \\"
echo "    --output-onnx $OUTPUT_ONNX"

# 示例4: 只导出ONNX（已有训练好的模型）
echo -e "\n示例4: 只导出ONNX模型"
echo "python3 scripts/segmentation_pipeline.py \\"
echo "    --skip-conversion \\"
echo "    --skip-training \\"
echo "    --model-checkpoint outputs/segmentation/best_model.pth \\"
echo "    --config $CONFIG \\"
echo "    --output-onnx $OUTPUT_ONNX"

# 示例5: 保存中间文件用于调试
echo -e "\n示例5: 调试模式（保留临时文件）"
echo "python3 scripts/segmentation_pipeline.py \\"
echo "    --coco-json $COCO_JSON \\"
echo "    --coco-images $COCO_IMAGES \\"
echo "    --config $CONFIG \\"
echo "    --output-onnx $OUTPUT_ONNX \\"
echo "    --keep-temp \\"
echo "    --debug"

echo -e "\n=========================================="
echo "选择要运行的示例："
echo "1) 完整流水线"
echo "2) 快速测试（5个epoch）"
echo "3) 使用已有分割数据"
echo "4) 只导出ONNX"
echo "5) 调试模式"
echo "0) 退出"
echo "=========================================="

read -p "请输入选项 [0-5]: " choice

case $choice in
    1)
        echo "运行完整流水线..."
        python3 scripts/segmentation_pipeline.py \
            --coco-json "$COCO_JSON" \
            --coco-images "$COCO_IMAGES" \
            --config "$CONFIG" \
            --output-onnx "$OUTPUT_ONNX"
        ;;
    2)
        echo "运行快速测试..."
        python3 scripts/segmentation_pipeline.py \
            --coco-json "$COCO_JSON" \
            --coco-images "$COCO_IMAGES" \
            --config "$CONFIG" \
            --output-onnx "$OUTPUT_ONNX" \
            --epochs 5 \
            --batch-size 4
        ;;
    3)
        echo "使用已有分割数据..."
        python3 scripts/segmentation_pipeline.py \
            --skip-conversion \
            --segmentation-dir data/segmentation_new \
            --config "$CONFIG" \
            --output-onnx "$OUTPUT_ONNX"
        ;;
    4)
        echo "只导出ONNX模型..."
        python3 scripts/segmentation_pipeline.py \
            --skip-conversion \
            --skip-training \
            --model-checkpoint outputs/segmentation/best_model.pth \
            --config "$CONFIG" \
            --output-onnx "$OUTPUT_ONNX"
        ;;
    5)
        echo "运行调试模式..."
        python3 scripts/segmentation_pipeline.py \
            --coco-json "$COCO_JSON" \
            --coco-images "$COCO_IMAGES" \
            --config "$CONFIG" \
            --output-onnx "$OUTPUT_ONNX" \
            --keep-temp \
            --debug
        ;;
    0)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

