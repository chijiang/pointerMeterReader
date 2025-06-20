#!/usr/bin/env python3
"""
LED颜色识别训练示例

这个示例展示了如何使用LED颜色训练脚本进行模型训练、评估和可视化。

作者: chijiang
日期: 2025-06-06
"""

import sys
from pathlib import Path

# 添加scripts目录到路径
sys.path.append(str(Path(__file__).parent / "scripts"))

from train_led_color import LEDColorTrainer, create_default_config
import yaml

def example_training():
    """训练示例"""
    print("🚀 LED颜色识别训练示例")
    print("=" * 50)
    
    # 1. 创建配置
    print("📝 Step 1: Creating configuration...")
    config = create_default_config()
    
    # 可以自定义配置
    config['training']['epochs'] = 50  # 减少轮数用于快速测试
    config['training']['batch'] = 8   # 减少批次大小
    
    config_path = Path("example_config.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ Configuration saved to: {config_path}")
    
    # 2. 初始化训练器
    print("\n🔧 Step 2: Initializing trainer...")
    trainer = LEDColorTrainer(str(config_path))
    print("✅ Trainer initialized")
    
    # 3. 准备数据集
    print("\n📊 Step 3: Preparing dataset...")
    dataset_config = trainer.prepare_yolo_dataset()
    print(f"✅ Dataset prepared: {dataset_config}")
    
    # 4. 开始训练
    print("\n🏃‍♂️ Step 4: Starting training...")
    print("ℹ️  This will start the actual training process.")
    print("ℹ️  Training can take some time depending on your hardware.")
    
    # 取消注释下面的行来实际开始训练
    # model_path = trainer.train()
    # print(f"✅ Training completed! Model saved to: {model_path}")
    
    # 模拟训练完成，使用示例路径
    model_path = "outputs/checkpoints/led_color/led_color_model/weights/best.pt"
    print(f"🎯 Simulated training completed. Model would be saved to: {model_path}")
    
    # 5. 评估模型（如果模型存在）
    print("\n📈 Step 5: Model evaluation...")
    print("ℹ️  After training, you can evaluate the model:")
    print(f"   trainer.evaluate('{model_path}')")
    
    # 6. 可视化预测结果
    print("\n🎨 Step 6: Visualization...")
    print("ℹ️  Generate prediction visualizations:")
    print(f"   trainer.visualize_predictions('{model_path}', num_samples=10)")
    
    # 7. 导出模型
    print("\n📤 Step 7: Model export...")
    print("ℹ️  Export trained model to different formats:")
    print(f"   trainer.export_model('{model_path}', ['onnx', 'torchscript'])")
    
    # 清理
    config_path.unlink()
    
    print("\n" + "=" * 50)
    print("🎉 Example completed!")
    print("\n📋 To actually start training:")
    print("   1. Uncomment the training line in this script, or")
    print("   2. Run: python run_led_color_training.py")
    print("   3. Or: python scripts/train_led_color.py --mode train")

def example_quick_commands():
    """展示快速命令"""
    print("\n🚀 Quick Command Examples")
    print("=" * 50)
    
    commands = [
        ("开始训练", "python run_led_color_training.py"),
        ("自定义参数训练", "python run_led_color_training.py --epochs 150 --batch 32"),
        ("评估模型", "python run_led_color_training.py --eval"),
        ("可视化结果", "python run_led_color_training.py --viz"),
        ("导出模型", "python run_led_color_training.py --export"),
        ("详细模式训练", "python scripts/train_led_color.py --mode train"),
        ("详细模式评估", "python scripts/train_led_color.py --mode eval --model path/to/best.pt"),
    ]
    
    for desc, cmd in commands:
        print(f"📌 {desc}:")
        print(f"   {cmd}")
        print()

def example_data_analysis():
    """数据集分析示例"""
    print("\n📊 Dataset Analysis Example")
    print("=" * 50)
    
    data_root = Path("data/led_color")
    
    # 分析图像
    image_dir = data_root / "images"
    images = list(image_dir.glob("*"))
    images = [f for f in images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    
    print(f"📁 Total images: {len(images)}")
    
    # 分析标签
    label_dir = data_root / "labels"
    labels = list(label_dir.glob("*.txt"))
    
    print(f"🏷️  Total labels: {len(labels)}")
    
    # 统计类别分布
    class_counts = {'green': 0, 'red': 0, 'yellow': 0}
    total_objects = 0
    
    for label_file in labels:
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.strip().split()[0])
                    class_names = ['green', 'red', 'yellow']
                    if 0 <= class_id < len(class_names):
                        class_counts[class_names[class_id]] += 1
                        total_objects += 1
    
    print(f"🎯 Object distribution:")
    for class_name, count in class_counts.items():
        percentage = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    print(f"📦 Total objects: {total_objects}")

def main():
    """主函数"""
    try:
        example_training()
        example_quick_commands()
        example_data_analysis()
        
        print("\n🎓 Training Tips:")
        print("   • 数据量较小时，使用较小的模型（yolo11n.pt）")
        print("   • 增加epochs可以提高精度，但要注意过拟合")
        print("   • 在Apple Silicon Mac上会自动使用MPS加速")
        print("   • 训练结果保存在outputs/目录下")
        
    except Exception as e:
        print(f"❌ Error running example: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 