#!/usr/bin/env python3
"""
模型导出脚本
用于将训练好的分割模型导出为ONNX、TorchScript等格式
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation

def create_model(model_config):
    """创建模型"""
    num_classes = model_config['num_classes']
    
    # 创建DeepLabV3+模型
    if model_config['architecture'] == 'deeplabv3_resnet50':
        model = segmentation.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
    elif model_config['architecture'] == 'deeplabv3_resnet101':
        model = segmentation.deeplabv3_resnet101(pretrained=False, num_classes=num_classes)
    elif model_config['architecture'] == 'deeplabv3_mobilenet_v3_large':
        model = segmentation.deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型架构: {model_config['architecture']}")
    
    return model

def export_models(config):
    """导出模型"""
    # 获取设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_model(config['model'])
    model = model.to(device)
    
    # 加载训练好的权重
    best_model_path = config['save']['best_model_path']
    if os.path.exists(best_model_path):
        print(f"加载模型权重: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"警告: 找不到训练好的模型 {best_model_path}")
        return
    
    model.eval()
    
    # 创建输出目录
    export_config = config.get('export', {})
    output_dir = Path(export_config.get('output_dir', 'outputs/exported_models'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建示例输入
    input_size = export_config.get('input_size', [224, 224])
    example_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    formats = export_config.get('formats', ['onnx', 'torchscript'])
    
    # 导出ONNX
    if 'onnx' in formats:
        onnx_path = output_dir / 'segmentation_model.onnx'
        onnx_config = export_config.get('onnx', {})
        
        try:
            print(f"导出ONNX模型到: {onnx_path}")
            torch.onnx.export(
                model,
                example_input,
                str(onnx_path),
                export_params=True,
                opset_version=onnx_config.get('opset_version', 11),
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=onnx_config.get('dynamic_axes', {
                    'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'output': {0: 'batch_size', 2: 'height', 3: 'width'}
                })
            )
            print(f"ONNX模型导出成功: {onnx_path}")
        except Exception as e:
            print(f"ONNX导出失败: {e}")
    
    # 导出TorchScript
    if 'torchscript' in formats:
        script_path = output_dir / 'segmentation_model.pt'
        try:
            print(f"导出TorchScript模型到: {script_path}")
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save(str(script_path))
            print(f"TorchScript模型导出成功: {script_path}")
        except Exception as e:
            print(f"TorchScript导出失败: {e}")
    
    # 导出纯PyTorch模型（只保存模型结构和权重）
    pytorch_path = output_dir / 'segmentation_model_pytorch.pth'
    try:
        print(f"导出PyTorch模型到: {pytorch_path}")
        # 保存完整的模型信息
        model_info = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'architecture': config['model']['architecture'],
                'num_classes': config['model']['num_classes'],
                'pretrained': config['model']['pretrained']
            },
            'input_size': input_size,
            'class_names': config['classes']['names']
        }
        torch.save(model_info, pytorch_path)
        print(f"PyTorch模型导出成功: {pytorch_path}")
    except Exception as e:
        print(f"PyTorch模型导出失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="模型导出")
    parser.add_argument(
        "--config",
        type=str,
        default="config/segmentation_config.yaml",
        help="配置文件路径"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 导出模型
    export_models(config)
    print("模型导出完成!")

if __name__ == "__main__":
    main() 