#!/usr/bin/env python3
"""
测试分割后处理效果的脚本
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from app import MeterSegmentor

def create_test_mask():
    """创建一个测试用的分割掩码，包含噪声和边界问题"""
    mask = np.zeros((200, 200), dtype=np.uint8)
    
    # 创建指针区域（带噪声）
    cv2.line(mask, (100, 100), (150, 80), 1, 3)  # 主指针
    cv2.circle(mask, (120, 90), 2, 1, -1)        # 噪声点1
    cv2.circle(mask, (180, 180), 1, 1, -1)       # 噪声点2
    
    # 创建刻度区域（边界外移）
    for angle in range(0, 180, 20):
        x1 = int(100 + 70 * np.cos(np.radians(angle)))
        y1 = int(100 + 70 * np.sin(np.radians(angle)))
        x2 = int(100 + 85 * np.cos(np.radians(angle)))
        y2 = int(100 + 85 * np.sin(np.radians(angle)))
        cv2.line(mask, (x1, y1), (x2, y2), 2, 4)  # 粗刻度线
    
    # 添加一些随机噪声
    noise_points = np.random.randint(0, 200, (10, 2))
    for point in noise_points:
        cv2.circle(mask, tuple(point), 1, 2, -1)
    
    return mask

def test_post_processing_effects():
    """测试不同后处理配置的效果"""
    
    # 创建测试掩码
    test_mask = create_test_mask()
    
    # 不同的后处理配置
    configs = {
        'Original': None,  # 不进行后处理
        'Light Processing': {
            'remove_noise': True,
            'keep_largest_component': True,
            'pointer_erosion': 1,
            'scale_erosion': 1,
            'fill_holes': True,
            'connect_scale_lines': False
        },
        'Standard Processing': {
            'remove_noise': True,
            'keep_largest_component': True,
            'pointer_erosion': 1,
            'scale_erosion': 2,
            'fill_holes': True,
            'connect_scale_lines': True
        },
        'Heavy Processing': {
            'remove_noise': True,
            'keep_largest_component': True,
            'pointer_erosion': 2,
            'scale_erosion': 3,
            'fill_holes': True,
            'connect_scale_lines': True
        }
    }
    
    # 创建分割器实例（不需要真实模型）
    segmentor = MeterSegmentor("dummy_path", post_process_config=configs['Standard Processing'])
    
    # 测试每种配置
    results = {}
    for name, config in configs.items():
        if config is None:
            # 原始掩码
            results[name] = test_mask
        else:
            # 临时更改配置
            segmentor.post_process_config = config
            results[name] = segmentor.post_process_mask(test_mask)
    
    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0]}  # 背景、指针、刻度
    
    for i, (name, mask) in enumerate(results.items()):
        # 创建彩色掩码
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for class_id, color in colors.items():
            colored_mask[mask == class_id] = color
        
        axes[i].imshow(colored_mask)
        axes[i].set_title(f'{name}\nPointer: {np.sum(mask==1)} px, Scale: {np.sum(mask==2)} px')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('post_processing_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("后处理效果对比:")
    print("-" * 50)
    for name, mask in results.items():
        pointer_pixels = np.sum(mask == 1)
        scale_pixels = np.sum(mask == 2)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        print(f"{name:20s}: 指针 {pointer_pixels:4d} px ({pointer_pixels/total_pixels*100:.1f}%), "
              f"刻度 {scale_pixels:4d} px ({scale_pixels/total_pixels*100:.1f}%)")

def main():
    print("🧪 测试分割后处理效果")
    print("=" * 50)
    
    test_post_processing_effects()
    
    print("\n✅ 测试完成！查看生成的图片 'post_processing_comparison.png'")

if __name__ == "__main__":
    main() 