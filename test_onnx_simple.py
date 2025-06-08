#!/usr/bin/env python3
"""
简单的ONNX模型测试脚本
"""

import numpy as np
import cv2

def test_onnx_availability():
    """测试ONNX Runtime是否可用"""
    try:
        import onnxruntime as ort
        print("✅ ONNX Runtime 已安装")
        print(f"📦 版本: {ort.__version__}")
        print(f"🔧 可用提供者: {ort.get_available_providers()}")
        return True
    except ImportError:
        print("❌ ONNX Runtime 未安装")
        print("💡 请运行: pip install onnxruntime")
        return False

def test_onnx_model():
    """测试ONNX模型加载和推理"""
    try:
        import onnxruntime as ort
        
        model_path = "outputs/segmentation/exported/segmentation_model.onnx"
        
        # 创建推理会话
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        print(f"✅ 模型加载成功: {model_path}")
        print(f"📊 输入信息:")
        for input_info in session.get_inputs():
            print(f"  - 名称: {input_info.name}")
            print(f"  - 形状: {input_info.shape}")
            print(f"  - 类型: {input_info.type}")
        
        print(f"📊 输出信息:")
        for output_info in session.get_outputs():
            print(f"  - 名称: {output_info.name}")
            print(f"  - 形状: {output_info.shape}")
            print(f"  - 类型: {output_info.type}")
        
        # 创建测试输入
        input_shape = session.get_inputs()[0].shape
        if input_shape[0] == 'batch_size':
            input_shape[0] = 1
        if input_shape[2] == 'height':
            input_shape[2] = 224
        if input_shape[3] == 'width':
            input_shape[3] = 224
            
        test_input = np.random.randn(*input_shape).astype(np.float32)
        input_name = session.get_inputs()[0].name
        
        print(f"🧪 测试输入形状: {test_input.shape}")
        
        # 运行推理
        outputs = session.run(None, {input_name: test_input})
        
        print(f"✅ 推理成功!")
        print(f"📊 输出形状: {outputs[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

def main():
    print("🔍 ONNX 模型测试")
    print("=" * 50)
    
    # 测试ONNX Runtime可用性
    if not test_onnx_availability():
        return
    
    # 测试模型
    test_onnx_model()

if __name__ == "__main__":
    main() 