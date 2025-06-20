#!/usr/bin/env python3
"""
LED颜色识别训练设置测试脚本

验证所有组件是否正确配置和可用。
"""

import sys
from pathlib import Path
import yaml

# 添加scripts路径
sys.path.append(str(Path(__file__).parent / "scripts"))

def test_imports():
    """测试必要的模块导入"""
    print("🔍 Testing imports...")
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        
        from ultralytics import YOLO
        print(f"   ✅ Ultralytics YOLO imported successfully")
        
        import cv2
        print(f"   ✅ OpenCV: {cv2.__version__}")
        
        import numpy as np
        print(f"   ✅ NumPy: {np.__version__}")
        
        from train_led_color import LEDColorTrainer, create_default_config
        print(f"   ✅ LED Color Trainer imported successfully")
        
        return True
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

def test_data_structure():
    """测试数据结构"""
    print("\n📁 Testing data structure...")
    
    data_root = Path("data/led_color")
    
    # 检查基本目录
    if not data_root.exists():
        print(f"   ❌ Data directory not found: {data_root}")
        return False
    print(f"   ✅ Data directory exists: {data_root}")
    
    # 检查图像目录
    image_dir = data_root / "images"
    if not image_dir.exists():
        print(f"   ❌ Images directory not found: {image_dir}")
        return False
    
    images = list(image_dir.glob("*"))
    images = [f for f in images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    print(f"   ✅ Found {len(images)} images")
    
    # 检查标签目录
    label_dir = data_root / "labels"
    if not label_dir.exists():
        print(f"   ❌ Labels directory not found: {label_dir}")
        return False
    
    labels = list(label_dir.glob("*.txt"))
    print(f"   ✅ Found {len(labels)} label files")
    
    # 检查类别文件
    classes_file = data_root / "classes.txt"
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        print(f"   ✅ Classes: {classes}")
    else:
        print(f"   ⚠️  Classes file not found: {classes_file}")
    
    return True

def test_device_detection():
    """测试设备检测"""
    print("\n🎯 Testing device detection...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"   ✅ MPS available (Apple Silicon)")
        else:
            print(f"   ✅ CPU available")
        
        return True
    except Exception as e:
        print(f"   ❌ Device detection error: {e}")
        return False

def test_config_creation():
    """测试配置文件创建"""
    print("\n⚙️  Testing configuration creation...")
    
    try:
        from train_led_color import create_default_config
        
        config = create_default_config()
        
        # 验证配置结构
        required_keys = ['model', 'training', 'device', 'data']
        for key in required_keys:
            if key not in config:
                print(f"   ❌ Missing config key: {key}")
                return False
        
        print(f"   ✅ Default configuration created successfully")
        print(f"   ✅ Model: {config['model']['name']}")
        print(f"   ✅ Epochs: {config['training']['epochs']}")
        print(f"   ✅ Batch size: {config['training']['batch']}")
        print(f"   ✅ Image size: {config['training']['imgsz']}")
        
        return True
    except Exception as e:
        print(f"   ❌ Configuration creation error: {e}")
        return False

def test_dataset_preparation():
    """测试数据集准备"""
    print("\n📊 Testing dataset preparation...")
    
    try:
        from train_led_color import LEDColorTrainer, create_default_config
        
        # 创建临时配置
        config = create_default_config()
        config_path = Path("temp_test_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # 测试训练器初始化
        trainer = LEDColorTrainer(str(config_path))
        print(f"   ✅ Trainer initialized successfully")
        
        # 测试数据集准备
        dataset_config = trainer.prepare_yolo_dataset()
        print(f"   ✅ Dataset prepared: {dataset_config}")
        
        # 验证数据集配置文件
        if Path(dataset_config).exists():
            with open(dataset_config, 'r') as f:
                dataset_info = yaml.safe_load(f)
            print(f"   ✅ Dataset classes: {dataset_info['names']}")
            print(f"   ✅ Number of classes: {dataset_info['nc']}")
        
        # 清理
        config_path.unlink()
        
        return True
    except Exception as e:
        print(f"   ❌ Dataset preparation error: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 LED Color Detection Training Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Structure Test", test_data_structure),
        ("Device Detection Test", test_device_detection),
        ("Configuration Test", test_config_creation),
        ("Dataset Preparation Test", test_dataset_preparation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready to start training.")
        print("\n📋 Next steps:")
        print("   1. Run: python run_led_color_training.py")
        print("   2. Or: python scripts/train_led_color.py --mode train")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 