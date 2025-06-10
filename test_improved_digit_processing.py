#!/usr/bin/env python3
"""
测试改进的数字处理逻辑
Test improved digit processing logic with duplicate filtering and intelligent grouping
"""

import numpy as np
import cv2
from typing import List, Dict
from app import DigitDetector

def create_mock_detections() -> List[Dict]:
    """创建模拟的检测结果来测试处理逻辑"""
    # 模拟一个LCD显示 "123.45" 的检测结果，包含一些重复检测
    mock_detections = [
        # 正确的检测
        {'class': '1', 'confidence': 0.95, 'center_x': 100, 'center_y': 50, 'bbox': [90, 40, 110, 60]},
        {'class': '2', 'confidence': 0.92, 'center_x': 130, 'center_y': 50, 'bbox': [120, 40, 140, 60]},
        {'class': '3', 'confidence': 0.88, 'center_x': 160, 'center_y': 50, 'bbox': [150, 40, 170, 60]},
        {'class': 'point', 'confidence': 0.85, 'center_x': 180, 'center_y': 55, 'bbox': [175, 50, 185, 60]},
        {'class': '4', 'confidence': 0.90, 'center_x': 200, 'center_y': 50, 'bbox': [190, 40, 210, 60]},
        {'class': '5', 'confidence': 0.87, 'center_x': 230, 'center_y': 50, 'bbox': [220, 40, 240, 60]},
        
        # 重复检测（应该被过滤掉）
        {'class': '1', 'confidence': 0.75, 'center_x': 102, 'center_y': 48, 'bbox': [92, 38, 112, 58]},  # 与第一个1重复
        {'class': '2', 'confidence': 0.70, 'center_x': 132, 'center_y': 52, 'bbox': [122, 42, 142, 62]},  # 与2重复
        {'class': 'point', 'confidence': 0.60, 'center_x': 182, 'center_y': 54, 'bbox': [177, 49, 187, 59]},  # 与小数点重复
        
        # 另一个数字组，距离较远
        {'class': '9', 'confidence': 0.93, 'center_x': 350, 'center_y': 50, 'bbox': [340, 40, 360, 60]},
        {'class': '8', 'confidence': 0.89, 'center_x': 380, 'center_y': 50, 'bbox': [370, 40, 390, 60]},
    ]
    
    return mock_detections

def test_digit_processing():
    """测试数字处理逻辑"""
    print("🧪 测试改进的数字处理逻辑")
    print("=" * 50)
    
    # 创建模拟的数字检测器（不需要实际模型）
    class MockDigitDetector(DigitDetector):
        def __init__(self):
            # 跳过模型加载
            self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'point']
    
    detector = MockDigitDetector()
    
    # 获取模拟检测结果
    mock_detections = create_mock_detections()
    
    print(f"📊 原始检测结果: {len(mock_detections)} 个")
    for i, det in enumerate(mock_detections):
        print(f"  {i+1}. '{det['class']}' (置信度: {det['confidence']:.3f}, 位置: {det['center_x']},{det['center_y']})")
    
    print("\n" + "─" * 50)
    
    # 测试重复检测过滤
    print("🔧 测试重复检测过滤...")
    filtered_detections = detector.filter_duplicate_detections(
        mock_detections, 
        overlap_threshold=0.7, 
        distance_threshold=30
    )
    
    print(f"📊 过滤后检测结果: {len(filtered_detections)} 个")
    for i, det in enumerate(filtered_detections):
        print(f"  {i+1}. '{det['class']}' (置信度: {det['confidence']:.3f}, 位置: {det['center_x']},{det['center_y']})")
    
    print("\n" + "─" * 50)
    
    # 测试读数提取
    print("🔍 测试读数提取...")
    reading = detector.extract_reading(filtered_detections)
    
    print(f"✅ 最终读数: **{reading}**")
    
    # 测试数字分组逻辑
    print("\n" + "─" * 50)
    print("📋 测试数字分组逻辑...")
    
    sorted_detections = sorted(filtered_detections, key=lambda x: x['center_x'])
    digit_groups = detector._group_digits_by_position(sorted_detections)
    
    print(f"📊 识别到 {len(digit_groups)} 个数字组:")
    for i, group in enumerate(digit_groups):
        group_reading = detector._construct_group_reading(group)
        print(f"  组 {i+1}: {len(group)} 个数字 → '{group_reading}'")
        for det in group:
            print(f"    - '{det['class']}' (位置: {det['center_x']})")
    
    print("\n" + "─" * 50)
    
    # 测试边界情况
    print("🧪 测试边界情况...")
    
    # 测试空检测
    empty_reading = detector.extract_reading([])
    print(f"空检测结果: '{empty_reading}'")
    
    # 测试只有小数点
    point_only = [{'class': 'point', 'confidence': 0.8, 'center_x': 100, 'center_y': 50, 'bbox': [95, 45, 105, 55]}]
    point_reading = detector.extract_reading(point_only)
    print(f"只有小数点: '{point_reading}'")
    
    # 测试多个连续小数点
    multi_points = [
        {'class': '1', 'confidence': 0.9, 'center_x': 100, 'center_y': 50, 'bbox': [90, 40, 110, 60]},
        {'class': 'point', 'confidence': 0.8, 'center_x': 120, 'center_y': 50, 'bbox': [115, 45, 125, 55]},
        {'class': 'point', 'confidence': 0.7, 'center_x': 130, 'center_y': 50, 'bbox': [125, 45, 135, 55]},
        {'class': '5', 'confidence': 0.9, 'center_x': 150, 'center_y': 50, 'bbox': [140, 40, 160, 60]},
    ]
    multi_point_reading = detector.extract_reading(multi_points)
    print(f"多个小数点: '{multi_point_reading}'")
    
    print("\n" + "=" * 50)
    print("✅ 测试完成！")

def test_specific_scenarios():
    """测试特定场景"""
    print("\n🎯 测试特定场景")
    print("=" * 50)
    
    class MockDigitDetector(DigitDetector):
        def __init__(self):
            self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'point']
    
    detector = MockDigitDetector()
    
    # 场景1: 非常接近的重复数字
    print("📍 场景1: 非常接近的重复数字")
    close_duplicates = [
        {'class': '7', 'confidence': 0.95, 'center_x': 100, 'center_y': 50, 'bbox': [90, 40, 110, 60]},
        {'class': '7', 'confidence': 0.85, 'center_x': 105, 'center_y': 52, 'bbox': [95, 42, 115, 62]},  # 很接近的重复
        {'class': '7', 'confidence': 0.75, 'center_x': 98, 'center_y': 48, 'bbox': [88, 38, 108, 58]},   # 另一个重复
    ]
    
    filtered = detector.filter_duplicate_detections(close_duplicates, distance_threshold=20)
    reading = detector.extract_reading(filtered)
    print(f"  原始: {len(close_duplicates)} 个 → 过滤后: {len(filtered)} 个 → 读数: '{reading}'")
    
    # 场景2: 不同数字但位置重叠
    print("\n📍 场景2: 不同数字但位置重叠（可能的误识别）")
    overlapping_different = [
        {'class': '8', 'confidence': 0.90, 'center_x': 100, 'center_y': 50, 'bbox': [90, 40, 110, 60]},
        {'class': '3', 'confidence': 0.85, 'center_x': 102, 'center_y': 51, 'bbox': [92, 41, 112, 61]},  # 高重叠但不同数字
        {'class': '0', 'confidence': 0.80, 'center_x': 99, 'center_y': 49, 'bbox': [89, 39, 109, 59]},   # 另一个高重叠
    ]
    
    filtered = detector.filter_duplicate_detections(overlapping_different, overlap_threshold=0.8)
    reading = detector.extract_reading(filtered)
    print(f"  原始: {len(overlapping_different)} 个 → 过滤后: {len(filtered)} 个 → 读数: '{reading}'")
    
    # 场景3: 复杂的多组数字
    print("\n📍 场景3: 复杂的多组数字（两个分离的显示器）")
    multi_group = [
        # 第一组: "12.3"
        {'class': '1', 'confidence': 0.95, 'center_x': 100, 'center_y': 50, 'bbox': [90, 40, 110, 60]},
        {'class': '2', 'confidence': 0.92, 'center_x': 130, 'center_y': 50, 'bbox': [120, 40, 140, 60]},
        {'class': 'point', 'confidence': 0.88, 'center_x': 150, 'center_y': 55, 'bbox': [145, 50, 155, 60]},
        {'class': '3', 'confidence': 0.90, 'center_x': 170, 'center_y': 50, 'bbox': [160, 40, 180, 60]},
        
        # 间隔较大
        
        # 第二组: "45.67"
        {'class': '4', 'confidence': 0.93, 'center_x': 300, 'center_y': 50, 'bbox': [290, 40, 310, 60]},
        {'class': '5', 'confidence': 0.91, 'center_x': 330, 'center_y': 50, 'bbox': [320, 40, 340, 60]},
        {'class': 'point', 'confidence': 0.87, 'center_x': 350, 'center_y': 55, 'bbox': [345, 50, 355, 60]},
        {'class': '6', 'confidence': 0.89, 'center_x': 370, 'center_y': 50, 'bbox': [360, 40, 380, 60]},
        {'class': '7', 'confidence': 0.85, 'center_x': 400, 'center_y': 50, 'bbox': [390, 40, 410, 60]},
    ]
    
    filtered = detector.filter_duplicate_detections(multi_group)
    reading = detector.extract_reading(filtered)
    print(f"  原始: {len(multi_group)} 个 → 过滤后: {len(filtered)} 个 → 读数: '{reading}'")
    
    # 显示分组详情
    sorted_detections = sorted(filtered, key=lambda x: x['center_x'])
    groups = detector._group_digits_by_position(sorted_detections)
    print(f"  分组详情: {len(groups)} 个组")
    for i, group in enumerate(groups):
        group_reading = detector._construct_group_reading(group)
        print(f"    组 {i+1}: '{group_reading}' ({len(group)} 个数字)")

if __name__ == "__main__":
    try:
        test_digit_processing()
        test_specific_scenarios()
        
        print("\n🎉 所有测试完成！")
        print("\n💡 主要改进点:")
        print("  ✅ 智能重复检测过滤")
        print("  ✅ 按位置自动分组")
        print("  ✅ 置信度优先保留")
        print("  ✅ 小数点特殊处理")
        print("  ✅ 多显示器支持")
        print("  ✅ 格式验证和清理")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 