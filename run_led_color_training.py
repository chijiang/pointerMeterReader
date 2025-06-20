#!/usr/bin/env python3
"""
LED颜色识别训练启动脚本

快速启动LED颜色识别模型训练的便捷脚本。

使用方法:
    python run_led_color_training.py           # 开始训练
    python run_led_color_training.py --eval    # 评估模型
    python run_led_color_training.py --viz     # 可视化预测结果

作者: chijiang
日期: 2025-06-06
"""

import os
import sys
import argparse
from pathlib import Path

# 添加scripts目录到路径
sys.path.append(str(Path(__file__).parent / "scripts"))

from train_led_color import LEDColorTrainer, create_default_config
import yaml

def main():
    parser = argparse.ArgumentParser(description='LED颜色识别训练启动器')
    parser.add_argument('--eval', action='store_true', help='评估已训练的模型')
    parser.add_argument('--viz', action='store_true', help='可视化预测结果')
    parser.add_argument('--export', action='store_true', help='导出模型')
    parser.add_argument('--model', type=str, help='模型路径（用于eval/viz/export）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    
    args = parser.parse_args()
    
    # 配置文件路径
    config_path = Path(__file__).parent / "config" / "led_color_config.yaml"
    
    # 确保配置文件存在
    if not config_path.exists():
        print("📝 Creating default configuration...")
        config_path.parent.mkdir(exist_ok=True)
        default_config = create_default_config()
        
        # 应用命令行参数
        default_config['training']['epochs'] = args.epochs
        default_config['training']['batch'] = args.batch
        default_config['training']['imgsz'] = args.imgsz
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ Configuration saved to: {config_path}")
    
    # 初始化训练器
    trainer = LEDColorTrainer(str(config_path))
    
    try:
        if args.eval:
            if args.model is None:
                # 查找最新的训练模型
                checkpoint_dir = Path(__file__).parent / "outputs" / "checkpoints" / "led_color"
                model_dirs = list(checkpoint_dir.glob("led_color_model*"))
                if model_dirs:
                    latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
                    best_model = latest_model_dir / "weights" / "best.pt"
                    if best_model.exists():
                        args.model = str(best_model)
                        print(f"🔍 Found model: {args.model}")
                    else:
                        print("❌ No trained model found. Please train first or specify --model path")
                        return
                else:
                    print("❌ No trained model found. Please train first or specify --model path")
                    return
            
            print("📊 Starting model evaluation...")
            trainer.evaluate(args.model)
            
        elif args.viz:
            if args.model is None:
                # 查找最新的训练模型
                checkpoint_dir = Path(__file__).parent / "outputs" / "checkpoints" / "led_color"
                model_dirs = list(checkpoint_dir.glob("led_color_model*"))
                if model_dirs:
                    latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
                    best_model = latest_model_dir / "weights" / "best.pt"
                    if best_model.exists():
                        args.model = str(best_model)
                        print(f"🔍 Found model: {args.model}")
                    else:
                        print("❌ No trained model found. Please train first or specify --model path")
                        return
                else:
                    print("❌ No trained model found. Please train first or specify --model path")
                    return
            
            print("🎨 Starting visualization...")
            trainer.visualize_predictions(args.model, num_samples=10)
            
        elif args.export:
            if args.model is None:
                # 查找最新的训练模型
                checkpoint_dir = Path(__file__).parent / "outputs" / "checkpoints" / "led_color"
                model_dirs = list(checkpoint_dir.glob("led_color_model*"))
                if model_dirs:
                    latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
                    best_model = latest_model_dir / "weights" / "best.pt"
                    if best_model.exists():
                        args.model = str(best_model)
                        print(f"🔍 Found model: {args.model}")
                    else:
                        print("❌ No trained model found. Please train first or specify --model path")
                        return
                else:
                    print("❌ No trained model found. Please train first or specify --model path")
                    return
            
            print("📤 Starting model export...")
            trainer.export_model(args.model, ['onnx', 'torchscript'])
            
        else:
            # 默认进行训练
            print("🚀 Starting LED color detection training...")
            print(f"📊 Dataset: data/led_color")
            print(f"🎯 Classes: green, red, yellow")
            print(f"⚙️  Epochs: {args.epochs}")
            print(f"📦 Batch size: {args.batch}")
            print(f"🖼️  Image size: {args.imgsz}")
            print()
            
            # 开始训练
            model_path = trainer.train()
            
            print("\n🎉 Training completed successfully!")
            print(f"📍 Model saved to: {model_path}")
            
            # 自动进行评估
            print("\n📊 Starting automatic evaluation...")
            trainer.evaluate(model_path)
            
            # 自动生成可视化
            print("\n🎨 Generating sample visualizations...")
            trainer.visualize_predictions(model_path, num_samples=5)
            
            print("\n✅ All tasks completed!")
            print("\n📋 Next steps:")
            print(f"   - Check results in: outputs/results/led_color/")
            print(f"   - Run evaluation: python {sys.argv[0]} --eval")
            print(f"   - Generate visualizations: python {sys.argv[0]} --viz")
            print(f"   - Export model: python {sys.argv[0]} --export")
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 