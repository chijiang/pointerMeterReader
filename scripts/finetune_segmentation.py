#!/usr/bin/env python3
"""
分割模型微调脚本
"""

import argparse
import yaml
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_segmentation import SegmentationTrainer

def main():
    parser = argparse.ArgumentParser(description="分割模型微调")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--pretrained", required=True, help="预训练权重路径")
    parser.add_argument("--data-dir", required=True, help="新数据集路径")
    parser.add_argument("--freeze-backbone", action="store_true", help="冻结骨干网络")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    
    args = parser.parse_args()
    
    # 检查预训练权重是否存在
    if not os.path.exists(args.pretrained):
        print(f"错误: 预训练权重文件不存在 {args.pretrained}")
        return
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改配置用于微调
    config['model']['resume_from'] = args.pretrained
    config['model']['freeze_backbone'] = args.freeze_backbone
    config['data']['root_dir'] = args.data_dir
    config['training']['learning_rate'] = args.lr
    config['training']['epochs'] = args.epochs
    
    # 修改输出路径
    output_base = f"outputs/segmentation_finetune_{Path(args.data_dir).name}"
    config['save']['checkpoint_dir'] = f"{output_base}/checkpoints"
    config['save']['log_dir'] = f"{output_base}/logs"
    config['save']['best_model_path'] = f"{output_base}/best_model.pth"
    config['visualization']['prediction_dir'] = f"{output_base}/predictions"
    
    print("🚀 开始微调训练...")
    print(f"预训练权重: {args.pretrained}")
    print(f"数据集: {args.data_dir}")
    print(f"冻结骨干网络: {args.freeze_backbone}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    
    # 创建训练器并开始训练
    trainer = SegmentationTrainer(config)
    trainer.train()
    
    print("✅ 微调完成!")

if __name__ == "__main__":
    main() 