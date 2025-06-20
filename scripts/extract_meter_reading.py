#!/usr/bin/env python3
"""
Meter Reading Extraction Script
Based on the C++ implementation in meter_reader.cpp

This script extracts numerical readings from segmented meter images using:
1. Pointer and scale segmentation masks
2. Geometric analysis of pointer position
3. Angle-based calculation for final reading

Classes in segmentation mask:
- 0: Background
- 1: Pointer
- 2: Scale/Dial
"""

import cv2
import numpy as np
import math
import argparse
import os
from pathlib import Path
import json
from typing import Tuple, List, Optional, Dict, Any
import matplotlib.pyplot as plt


class MeterReader:
    """Python implementation of meter reading extraction algorithm"""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.0, 1.6), debug: bool = False, use_improved_algorithm: bool = True):
        """
        Initialize meter reader
        
        Args:
            scale_range: (min_value, max_value) of the meter scale
            debug: Whether to show debug visualizations
            use_improved_algorithm: Whether to use the improved line-circle intersection algorithm
        """
        self.scale_beginning, self.scale_end = scale_range
        self.debug = debug
        self.use_improved_algorithm = use_improved_algorithm
        
    def get_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def threshold_by_category(self, src: np.ndarray, category: int) -> np.ndarray:
        """
        Create binary mask for specific category
        
        Args:
            src: Input segmentation mask
            category: Category ID to extract
            
        Returns:
            Binary mask (255 for category pixels, 0 for others)
        """
        dst = np.zeros_like(src, dtype=np.uint8)
        dst[src == category] = 255
        return dst
    
    def get_scale_locations(self, scale_mask: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Find scale start and end points
        
        Args:
            scale_mask: Binary mask of scale pixels
            
        Returns:
            ((start_x, start_y), (end_x, end_y)) or None if not found
        """
        height, width = scale_mask.shape
        beginning = None
        end = None
        
        # Scan from bottom to top
        for row in range(height - 1, -1, -1):
            for col in range(width):
                if scale_mask[row, col] == 255:
                    # Left half for beginning point
                    if col < width // 2 and beginning is None:
                        beginning = (col, row)
                    # Right half for end point
                    if col >= width // 2 and end is None:
                        end = (col, row)
            
            # Found both points
            if beginning is not None and end is not None:
                return beginning, end
        
        return None
    
    def get_center_location(self, image: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Calculate meter center based on image analysis
        
        Args:
            image: Original meter image
            
        Returns:
            (center_x, center_y) or None if not found
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        height, width = gray.shape
        diameter = 0
        diameter_indices = [-1, -1, -1, -1]  # [top_row, bottom_row, left_col, right_col]
        
        # Find the row with maximum width (diameter)
        for row in range(height - 1, -1, -1):
            left = 0
            right = width - 1
            
            # Find leftmost and rightmost non-zero pixels
            while left < right:
                if gray[row, left] == 0:
                    left += 1
                if gray[row, right] == 0:
                    right -= 1
                if gray[row, left] != 0 and gray[row, right] != 0:
                    break
            
            current_diameter = right - left
            if current_diameter >= diameter:
                if current_diameter > diameter:
                    diameter_indices[1] = row  # bottom row
                diameter_indices[0] = row  # top row
                diameter = current_diameter
                diameter_indices[2] = left  # left col
                diameter_indices[3] = right  # right col
        
        if diameter > 0:
            center_x = (diameter_indices[2] + diameter_indices[3]) / 2
            center_y = (diameter_indices[0] + diameter_indices[1]) / 2
            return (center_x, center_y)
        
        return None
    
    def get_min_area_rect_points(self, pointer_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Get minimum area rectangle points for the largest contour
        
        Args:
            pointer_mask: Binary mask of pointer pixels
            
        Returns:
            Array of 4 corner points or None if not found
        """
        contours, _ = cv2.findContours(pointer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Find largest contour by area
        max_area = 0
        max_contour = None
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            area = rect[1][0] * rect[1][1]  # width * height
            if area > max_area:
                max_area = area
                max_contour = contour
        
        if max_contour is None:
            return None
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(max_contour)
        points = cv2.boxPoints(rect)
        return points.astype(np.float32)
    
    def get_pointer_vertex_index(self, center: Tuple[float, float], points: np.ndarray) -> int:
        """
        Find the index of pointer head vertex
        
        Args:
            center: Center point of the meter
            points: 4 corner points of the minimum area rectangle
            
        Returns:
            Index of the head vertex
        """
        max_distance = -1
        vertex_index = 0
        
        for i in range(4):
            # Check if clockwise direction is short edge
            edge1_length = self.get_distance(points[i], points[(i + 1) % 4])
            edge2_length = self.get_distance(points[i], points[(i + 3) % 4])
            
            if edge1_length > edge2_length:
                # This is a head vertex candidate
                distance = self.get_distance(points[i], center)
                if distance > max_distance:
                    max_distance = distance
                    vertex_index = i
        
        return vertex_index
    
    def threshold_by_contour(self, src: np.ndarray, contour_points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Set pixels inside contour to 0
        
        Args:
            src: Source binary image
            contour_points: List of contour points
            
        Returns:
            Modified binary image
        """
        dst = src.copy()
        contour = np.array(contour_points, dtype=np.int32)
        
        # Create mask for the contour
        mask = np.zeros_like(src)
        cv2.fillPoly(mask, [contour], 255)
        
        # Set pixels inside contour to 0
        dst[mask == 255] = 0
        
        return dst
    
    def get_pointer_locations(self, pointer_mask: np.ndarray, center: Tuple[float, float]) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Get pointer head and tail locations
        
        Args:
            pointer_mask: Binary mask of pointer pixels
            center: Center point of the meter
            
        Returns:
            ((head_x, head_y), (tail_x, tail_y)) or None if not found
        """
        # Get minimum area rectangle points
        points = self.get_min_area_rect_points(pointer_mask)
        if points is None:
            return None
        
        # Get pointer head vertex index
        vertex_index = self.get_pointer_vertex_index(center, points)
        
        # Calculate mid points for contour masking
        left_mid = (
            (points[vertex_index][0] + points[(vertex_index + 1) % 4][0]) / 2,
            (points[vertex_index][1] + points[(vertex_index + 1) % 4][1]) / 2
        )
        right_mid = (
            (points[(vertex_index + 2) % 4][0] + points[(vertex_index + 3) % 4][0]) / 2,
            (points[(vertex_index + 2) % 4][1] + points[(vertex_index + 3) % 4][1]) / 2
        )
        
        # Create contour to mask out part of pointer near center
        contour_points = [
            left_mid,
            tuple(points[(vertex_index + 1) % 4]),
            tuple(points[(vertex_index + 2) % 4]),
            right_mid
        ]
        
        # Apply contour masking
        masked_pointer = self.threshold_by_contour(pointer_mask, contour_points)
        
        # Get new minimum area rectangle after masking
        new_points = self.get_min_area_rect_points(masked_pointer)
        if new_points is None:
            return None
        
        new_vertex_index = self.get_pointer_vertex_index(center, new_points)
        
        # Calculate head and tail mid points
        head_mid = (
            (new_points[new_vertex_index][0] + new_points[(new_vertex_index + 3) % 4][0]) / 2,
            (new_points[new_vertex_index][1] + new_points[(new_vertex_index + 3) % 4][1]) / 2
        )
        tail_mid = (
            (new_points[(new_vertex_index + 1) % 4][0] + new_points[(new_vertex_index + 2) % 4][0]) / 2,
            (new_points[(new_vertex_index + 1) % 4][1] + new_points[(new_vertex_index + 2) % 4][1]) / 2
        )
        
        return head_mid, tail_mid
    
    def get_angle_ratio(self, scale_locations: Tuple[Tuple[int, int], Tuple[int, int]], 
                       pointer_head: Tuple[float, float], center: Tuple[float, float]) -> float:
        """
        Calculate angle ratio of pointer position relative to scale range
        
        Args:
            scale_locations: ((start_x, start_y), (end_x, end_y))
            pointer_head: (head_x, head_y)
            center: (center_x, center_y)
            
        Returns:
            Angle ratio (0.0 to 1.0)
        """
        # Calculate angles relative to positive x-axis
        beginning_angle = math.atan2(center[1] - scale_locations[0][1], 
                                   scale_locations[0][0] - center[0])
        end_angle = math.atan2(center[1] - scale_locations[1][1], 
                             scale_locations[1][0] - center[0])
        
        # Total angle span
        total_angle = 2 * math.pi - (end_angle - beginning_angle)
        
        # Pointer angle
        pointer_angle = math.atan2(center[1] - pointer_head[1], 
                                 pointer_head[0] - center[0])
        
        # Calculate pointer position relative to beginning
        if pointer_head[1] > center[1] and pointer_head[0] < center[0]:
            pointer_relative_angle = pointer_angle - beginning_angle
        else:
            pointer_relative_angle = 2 * math.pi - (pointer_angle - beginning_angle)
        
        # Ensure positive angle
        if pointer_relative_angle < 0:
            pointer_relative_angle += 2 * math.pi
        
        angle_ratio = pointer_relative_angle / total_angle
        return max(0.0, min(1.0, angle_ratio))  # Clamp to [0, 1]
    
    def get_scale_value(self, angle_ratio: float) -> float:
        """
        Convert angle ratio to scale value
        
        Args:
            angle_ratio: Ratio from 0.0 to 1.0
            
        Returns:
            Scale value within the configured range
        """
        scale_range = self.scale_end - self.scale_beginning
        return scale_range * angle_ratio + self.scale_beginning
    
    def visualize_result(self, image: np.ndarray, scale_locations: Tuple[Tuple[int, int], Tuple[int, int]], 
                        pointer_locations: Tuple[Tuple[float, float], Tuple[float, float]], 
                        center: Tuple[float, float], scale_value: float) -> np.ndarray:
        """
        Create visualization of the reading extraction process
        
        Args:
            image: Original image
            scale_locations: Scale start and end points
            pointer_locations: Pointer head and tail points
            center: Center point
            scale_value: Calculated scale value
            
        Returns:
            Visualization image
        """
        vis_img = image.copy()
        
        # Draw scale points (red)
        cv2.circle(vis_img, scale_locations[0], 3, (0, 0, 255), -1)
        cv2.circle(vis_img, scale_locations[1], 3, (0, 0, 255), -1)
        
        # Draw center (green)
        cv2.circle(vis_img, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
        
        # Draw pointer line (blue)
        cv2.line(vis_img, 
                (int(pointer_locations[0][0]), int(pointer_locations[0][1])),
                (int(pointer_locations[1][0]), int(pointer_locations[1][1])),
                (255, 0, 0), 3)
        
        # Add text with scale value
        text = f"Reading: {scale_value:.2f}"
        cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return vis_img
    
    def process_single_meter(self, image: np.ndarray, mask: np.ndarray) -> Optional[float]:
        """
        Process a single meter image and extract reading
        
        Args:
            image: Original meter image (BGR)
            mask: Segmentation mask with classes 0=background, 1=pointer, 2=scale
            
        Returns:
            Scale reading value or None if extraction failed
        """
        try:
            # Extract pointer and scale masks
            pointer_mask = self.threshold_by_category(mask, 1)  # Pointer
            scale_mask = self.threshold_by_category(mask, 2)    # Scale
            
            if self.debug:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(pointer_mask, cmap='gray')
                axes[0].set_title("Pointer Mask")
                axes[0].axis('off')
                axes[1].imshow(scale_mask, cmap='gray')
                axes[1].set_title("Scale Mask")
                axes[1].axis('off')
                plt.tight_layout()
                plt.show()
            
            # Apply erosion to scale mask to remove noise
            kernel = np.ones((5, 5), np.uint8)
            scale_mask = cv2.erode(scale_mask, kernel, iterations=1)
            
            if self.debug:
                plt.figure(figsize=(8, 6))
                plt.imshow(scale_mask, cmap='gray')
                plt.title("Scale Mask After Erosion")
                plt.axis('off')
                plt.show()
            
            # Find scale start and end points
            scale_locations = self.get_scale_locations(scale_mask)
            if scale_locations is None:
                print("Failed to find scale locations")
                return None
            
            # 使用改进的圆心计算方法
            center = self.get_improved_center_location(image, pointer_mask, scale_mask)
            if center is None:
                print("Failed to find meter center using improved method")
                return None
            
            # Find pointer locations
            pointer_locations = self.get_pointer_locations(pointer_mask, center)
            if pointer_locations is None:
                print("Failed to find pointer locations")
                return None
            
            # Calculate angle ratio
            angle_ratio = self.get_angle_ratio(scale_locations, pointer_locations[0], center)
            
            # Convert to scale value
            scale_value = self.get_scale_value(angle_ratio)
            
            # Visualization for debugging
            if self.debug:
                vis_img = self.visualize_result(image, scale_locations, pointer_locations, center, scale_value)
                plt.figure(figsize=(10, 8))
                # Convert BGR to RGB for matplotlib
                vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                plt.imshow(vis_img_rgb)
                plt.title(f"Meter Reading Result: {scale_value:.3f}")
                plt.axis('off')
                plt.show()
            
            return scale_value
            
        except Exception as e:
            print(f"Error processing meter: {e}")
            return None

    def get_improved_center_location(self, image: np.ndarray, pointer_mask: np.ndarray, 
                                    scale_mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        改进的圆心计算方法，基于指针和刻度掩码信息
        
        Args:
            image: 原始图像
            pointer_mask: 指针二值掩码
            scale_mask: 刻度二值掩码
            
        Returns:
            (center_x, center_y) 或 None
        """
        # 方法1: 基于指针质心和延长线计算圆心
        pointer_center = self.get_pointer_based_center(pointer_mask)
        
        # 方法2: 基于刻度几何特征计算圆心
        scale_center = self.get_scale_based_center(scale_mask)
        
        # 方法3: 传统的图像边界方法作为备选
        fallback_center = self.get_center_location(image)
        
        # 综合判断最佳圆心
        final_center = self.select_best_center(pointer_center, scale_center, fallback_center,
                                             pointer_mask, scale_mask)
        
        return final_center

    def get_pointer_based_center(self, pointer_mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        基于指针几何特征估算圆心
        
        Args:
            pointer_mask: 指针二值掩码
            
        Returns:
            估算的圆心坐标
        """
        contours, _ = cv2.findContours(pointer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # 找到最大的轮廓（主指针）
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 50:  # 太小的轮廓
            return None
        
        # 获取指针的最小外接矩形
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        
        # 计算指针的主轴方向
        center, (width, height), angle = rect
        
        # 指针通常是细长的，选择长边作为主轴
        if width > height:
            # 计算沿长边的延长线
            angle_rad = np.deg2rad(angle)
            direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        else:
            # 旋转90度
            angle_rad = np.deg2rad(angle + 90)
            direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        # 沿指针方向延长，估算圆心
        # 假设指针长度的0.3-0.8倍为到圆心的距离
        pointer_length = max(width, height)
        
        potential_centers = []
        for ratio in [0.3, 0.5, 0.8]:
            extension_length = pointer_length * ratio
            estimated_center = center - direction * extension_length
            potential_centers.append(estimated_center)
        
        # 返回中间值作为最佳估计
        return tuple(potential_centers[1])

    def get_scale_based_center(self, scale_mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        基于刻度几何特征估算圆心
        
        Args:
            scale_mask: 刻度二值掩码
            
        Returns:
            估算的圆心坐标
        """
        # 查找刻度轮廓
        contours, _ = cv2.findContours(scale_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # 合并所有刻度点
        all_points = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # 过滤小噪声
                all_points.extend(contour.reshape(-1, 2))
        
        if len(all_points) < 6:  # 需要足够的点来拟合圆
            return None
        
        all_points = np.array(all_points)
        
        # 使用最小二乘法拟合圆
        center_estimate = self.fit_circle_least_squares(all_points)
        
        return center_estimate

    def fit_circle_least_squares(self, points: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        使用最小二乘法拟合圆，估算圆心
        
        Args:
            points: 点集 [[x1, y1], [x2, y2], ...]
            
        Returns:
            (center_x, center_y) 或 None
        """
        if len(points) < 3:
            return None
        
        try:
            # 设置方程 (x-a)² + (y-b)² = r²
            # 展开为 x² + y² - 2ax - 2by + (a² + b² - r²) = 0
            # 转换为线性方程组 Ac = b
            
            x = points[:, 0]
            y = points[:, 1]
            
            # 构建系数矩阵 A 和常数向量 b
            A = np.column_stack([2*x, 2*y, np.ones(len(points))])
            b = x**2 + y**2
            
            # 求解线性方程组
            # c = [a, b, (a² + b² - r²)]
            c, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            center_x, center_y = c[0], c[1]
            
            # 验证结果的合理性
            if np.isnan(center_x) or np.isnan(center_y):
                return None
            
            # 检查拟合质量（计算平均距离误差）
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            avg_radius = np.mean(distances)
            radius_std = np.std(distances)
            
            # 如果标准差太大，说明拟合质量不好
            if radius_std > avg_radius * 0.3:  # 30%的容差
                return None
            
            return (center_x, center_y)
            
        except Exception as e:
            print(f"Circle fitting error: {e}")
            return None

    def select_best_center(self, pointer_center: Optional[Tuple[float, float]], 
                          scale_center: Optional[Tuple[float, float]], 
                          fallback_center: Optional[Tuple[float, float]],
                          pointer_mask: np.ndarray, scale_mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        从多个圆心候选中选择最佳的
        
        Args:
            pointer_center: 基于指针的圆心估计
            scale_center: 基于刻度的圆心估计
            fallback_center: 传统方法的圆心估计
            pointer_mask: 指针掩码
            scale_mask: 刻度掩码
            
        Returns:
            最佳圆心坐标
        """
        candidates = []
        
        # 收集所有有效的候选圆心
        if pointer_center is not None:
            candidates.append(('pointer', pointer_center))
        if scale_center is not None:
            candidates.append(('scale', scale_center))
        if fallback_center is not None:
            candidates.append(('fallback', fallback_center))
        
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0][1]
        
        # 评估每个候选圆心的质量
        best_center = None
        best_score = -1
        
        for method, center in candidates:
            score = self.evaluate_center_quality(center, pointer_mask, scale_mask)
            
            # 对不同方法给予不同的权重
            if method == 'scale':
                score *= 1.2  # 优先考虑基于刻度的方法
            elif method == 'pointer':
                score *= 1.1  # 指针方法次之
            # fallback方法不加权重
            
            if score > best_score:
                best_score = score
                best_center = center
        
        return best_center

    def evaluate_center_quality(self, center: Tuple[float, float], 
                               pointer_mask: np.ndarray, scale_mask: np.ndarray) -> float:
        """
        评估圆心质量的评分函数
        
        Args:
            center: 待评估的圆心
            pointer_mask: 指针掩码
            scale_mask: 刻度掩码
            
        Returns:
            质量评分 (0-100，越高越好)
        """
        cx, cy = center
        h, w = pointer_mask.shape
        
        # 检查圆心是否在图像范围内
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return 0
        
        score = 0
        
        # 1. 检查圆心附近是否有指针像素（应该很少）
        center_region_size = 10
        x1 = max(0, int(cx - center_region_size))
        x2 = min(w, int(cx + center_region_size))
        y1 = max(0, int(cy - center_region_size))
        y2 = min(h, int(cy + center_region_size))
        
        center_region_pointer = pointer_mask[y1:y2, x1:x2]
        center_pointer_density = np.sum(center_region_pointer) / (center_region_size * center_region_size * 255)
        
        # 圆心附近指针密度应该较低
        if center_pointer_density < 0.2:
            score += 30
        elif center_pointer_density < 0.5:
            score += 15
        
        # 2. 检查刻度点到圆心的距离分布
        scale_points = np.where(scale_mask == 255)
        if len(scale_points[0]) > 0:
            scale_coords = np.column_stack([scale_points[1], scale_points[0]])  # x, y
            distances = np.sqrt((scale_coords[:, 0] - cx)**2 + (scale_coords[:, 1] - cy)**2)
            
            if len(distances) > 0:
                # 刻度点到圆心的距离应该相对一致
                distance_std = np.std(distances)
                avg_distance = np.mean(distances)
                
                if avg_distance > 0:
                    cv = distance_std / avg_distance  # 变异系数
                    if cv < 0.2:  # 距离很一致
                        score += 40
                    elif cv < 0.4:  # 距离较一致
                        score += 25
                    elif cv < 0.6:  # 距离一般
                        score += 10
        
        # 3. 检查指针是否指向圆心方向
        pointer_points = np.where(pointer_mask == 255)
        if len(pointer_points[0]) > 0:
            pointer_coords = np.column_stack([pointer_points[1], pointer_points[0]])
            
            # 计算指针重心
            pointer_centroid = np.mean(pointer_coords, axis=0)
            
            # 检查重心到圆心的方向是否与指针主轴一致
            direction_to_center = np.array([cx - pointer_centroid[0], cy - pointer_centroid[1]])
            
            if np.linalg.norm(direction_to_center) > 0:
                score += 20
        
        # 4. 检查圆心位置的合理性（通常在图像中下部分）
        relative_y = cy / h
        if 0.3 <= relative_y <= 0.8:  # 圆心在图像中下部分
            score += 10
        
        return score

    def get_scale_circle_parameters(self, scale_mask: np.ndarray) -> Optional[Tuple[Tuple[float, float], float, np.ndarray]]:
        """
        基于刻度掩码拟合圆，获取圆心和半径，并返回用于拟合的点集
        """
        contours, _ = cv2.findContours(scale_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        all_points = []
        for contour in contours:
            if cv2.contourArea(contour) > 5:
                all_points.extend(contour.reshape(-1, 2))
        if len(all_points) < 10:
            return None
        all_points = np.array(all_points)
        h, w = scale_mask.shape
        img_center = np.array([w/2, h/2])
        dists = np.linalg.norm(all_points - img_center, axis=1)
        dist_thresh = np.percentile(dists, 80)
        outer_points = all_points[dists >= dist_thresh]
        if len(outer_points) < 10:
            outer_points = all_points
        circle_params = self.fit_circle_with_radius(outer_points)
        if circle_params is None:
            return None
        return (*circle_params, outer_points)

    def fit_circle_with_radius(self, points: np.ndarray) -> Optional[Tuple[Tuple[float, float], float]]:
        """
        使用最小二乘法拟合圆，返回圆心和半径
        
        Args:
            points: 点集 [[x1, y1], [x2, y2], ...]
            
        Returns:
            ((center_x, center_y), radius) 或 None
        """
        if len(points) < 3:
            return None
        
        try:
            # 使用Kasa法拟合圆，避免半径偏大问题
            # 方程: (x-a)^2 + (y-b)^2 = r^2
            x = points[:, 0]
            y = points[:, 1]
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            u = x - x_mean
            v = y - y_mean

            Suu = np.sum(u * u)
            Suv = np.sum(u * v)
            Svv = np.sum(v * v)
            Suuu = np.sum(u * u * u)
            Svvv = np.sum(v * v * v)
            Suvv = np.sum(u * v * v)
            Svuu = np.sum(v * u * u)

            # 解线性方程组
            A = np.array([[Suu, Suv], [Suv, Svv]])
            B = np.array([
                0.5 * (Suuu + Suvv),
                0.5 * (Svvv + Svuu)
            ])
            try:
                uc, vc = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                return None

            center_x = x_mean + uc
            center_y = y_mean + vc
            radius = np.sqrt(np.mean((x - center_x) ** 2 + (y - center_y) ** 2))
            
            # 验证结果的合理性
            if np.isnan(center_x) or np.isnan(center_y) or np.isnan(radius):
                return None
            
            # 检查拟合质量
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            radius_std = np.std(distances)
            
            # 如果标准差太大，说明拟合质量不好
            if radius_std > radius * 0.15:  # 15%的容差
                return None
            
            return ((center_x, center_y), radius)
            
        except Exception as e:
            print(f"Circle fitting error: {e}")
            return None

    def get_pointer_line_parameters(self, pointer_mask: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        将指针拟合为直线，返回直线的两个端点
        
        Args:
            pointer_mask: 指针二值掩码
            
        Returns:
            ((x1, y1), (x2, y2)) 直线的两个端点，或 None
        """
        # 查找指针轮廓
        contours, _ = cv2.findContours(pointer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # 找到最大的轮廓（主指针）
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 50:  # 太小的轮廓
            return None
        
        # 获取指针的所有点
        points = largest_contour.reshape(-1, 2)
        
        if len(points) < 4:
            return None
        
        # 使用主成分分析(PCA)来找到指针的主轴方向
        center = np.mean(points, axis=0)
        centered_points = points - center
        
        # 计算协方差矩阵
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 主成分方向（最大特征值对应的特征向量）
        main_direction = eigenvectors[:, np.argmax(eigenvalues)]
        
        # 沿主轴方向投影所有点，找到两端
        projections = np.dot(centered_points, main_direction)
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        
        # 计算直线的两个端点
        point1 = center + min_proj * main_direction
        point2 = center + max_proj * main_direction
        
        return (tuple(point1), tuple(point2))

    def get_line_circle_intersection(self, line_p1: Tuple[float, float], line_p2: Tuple[float, float],
                                    circle_center: Tuple[float, float], radius: float) -> List[Tuple[float, float]]:
        """
        计算直线与圆的交点
        
        Args:
            line_p1, line_p2: 直线上的两点
            circle_center: 圆心
            radius: 圆的半径
            
        Returns:
            交点列表 [(x1, y1), (x2, y2), ...]
        """
        x1, y1 = line_p1
        x2, y2 = line_p2
        cx, cy = circle_center
        
        # 直线参数化: P = P1 + t * (P2 - P1)
        dx = x2 - x1
        dy = y2 - y1
        
        # 直线到圆心的向量
        fx = x1 - cx
        fy = y1 - cy
        
        # 二次方程: |P1 + t*(P2-P1) - C|² = r²
        # 展开: a*t² + b*t + c = 0
        a = dx*dx + dy*dy
        b = 2*(fx*dx + fy*dy)
        c = fx*fx + fy*fy - radius*radius
        
        if a == 0:  # 直线退化为点
            return []
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:  # 无交点
            return []
        elif discriminant == 0:  # 一个交点（相切）
            t = -b / (2*a)
            x = x1 + t * dx
            y = y1 + t * dy
            return [(x, y)]
        else:  # 两个交点
            sqrt_discriminant = math.sqrt(discriminant)
            t1 = (-b - sqrt_discriminant) / (2*a)
            t2 = (-b + sqrt_discriminant) / (2*a)
            
            x_1 = x1 + t1 * dx
            y_1 = y1 + t1 * dy
            x_2 = x1 + t2 * dx
            y_2 = y1 + t2 * dy
            
            return [(x_1, y_1), (x_2, y_2)]

    def calculate_reading_from_intersection(self, intersection_point: Tuple[float, float],
                                          scale_start: Tuple[int, int], scale_end: Tuple[int, int],
                                          circle_center: Tuple[float, float]) -> float:
        """
        根据交点在刻度圆弧上的位置计算读数
        
        Args:
            intersection_point: 指针与刻度圆的交点
            scale_start: 刻度起始点
            scale_end: 刻度结束点
            circle_center: 刻度圆心
            
        Returns:
            角度比例 (0.0 到 1.0)
        """
        cx, cy = circle_center
        
        # 计算各点相对于圆心的角度
        start_angle = math.atan2(scale_start[1] - cy, scale_start[0] - cx)
        end_angle = math.atan2(scale_end[1] - cy, scale_end[0] - cx)
        intersection_angle = math.atan2(intersection_point[1] - cy, intersection_point[0] - cx)
        
        # 将角度标准化到 [0, 2π] 范围
        start_angle = start_angle % (2 * math.pi)
        end_angle = end_angle % (2 * math.pi)
        intersection_angle = intersection_angle % (2 * math.pi)
        
        # 计算刻度的总角度跨度
        if end_angle > start_angle:
            total_angle = end_angle - start_angle
            if intersection_angle >= start_angle and intersection_angle <= end_angle:
                relative_angle = intersection_angle - start_angle
            else:
                # 处理跨越0度的情况
                if intersection_angle < start_angle:
                    relative_angle = intersection_angle + 2*math.pi - start_angle
                else:
                    relative_angle = intersection_angle - start_angle
        else:
            # 刻度跨越0度
            total_angle = (2 * math.pi) - (start_angle - end_angle)
            if intersection_angle >= start_angle or intersection_angle <= end_angle:
                if intersection_angle >= start_angle:
                    relative_angle = intersection_angle - start_angle
                else:
                    relative_angle = intersection_angle + 2*math.pi - start_angle
            else:
                relative_angle = intersection_angle - start_angle
        
        # 确保角度为正
        if relative_angle < 0:
            relative_angle += 2 * math.pi
        
        # 计算比例
        if total_angle > 0:
            ratio = relative_angle / total_angle
            return max(0.0, min(1.0, ratio))  # 限制在 [0, 1] 范围内
        else:
            return 0.0

    def process_single_meter_improved(self, image: np.ndarray, mask: np.ndarray) -> Tuple[Optional[float], Optional[Dict]]:
        """
        使用改进的直线-圆交点算法处理单个仪表
        
        Args:
            image: 原始仪表图像 (BGR)
            mask: 分割掩码，类别 0=背景, 1=指针, 2=刻度
            
        Returns:
            (刻度读数值, 几何信息字典)，失败时返回 (None, None)
        """
        try:
            # 提取指针和刻度掩码
            pointer_mask = self.threshold_by_category(mask, 1)  # 指针
            scale_mask = self.threshold_by_category(mask, 2)    # 刻度
            
            # 对刻度掩码进行预处理
            kernel = np.ones((3, 3), np.uint8)
            scale_mask = cv2.erode(scale_mask, kernel, iterations=1)
            
            # 步骤1: 拟合刻度圆
            circle_params = self.get_scale_circle_parameters(scale_mask)
            if circle_params is None:
                print("Failed to fit scale circle")
                return None, None
            
            circle_center, radius, fit_points = circle_params
            
            # 步骤2: 找到刻度的起始和结束点
            scale_locations = self.get_scale_locations(scale_mask)
            if scale_locations is None:
                print("Failed to find scale locations")
                return None, None
            
            # 步骤3: 拟合指针直线
            pointer_line = self.get_pointer_line_parameters(pointer_mask)
            if pointer_line is None:
                print("Failed to fit pointer line")
                return None, None
            
            line_p1, line_p2 = pointer_line
            
            # 步骤4: 计算直线与圆的交点
            intersections = self.get_line_circle_intersection(line_p1, line_p2, circle_center, radius)
            if not intersections:
                print("No intersection between pointer line and scale circle")
                return None, None
            
            # 步骤5: 选择合适的交点
            if len(intersections) == 1:
                intersection_point = intersections[0]
            else:
                # 选择距离指针质心更近的交点
                pointer_points = np.where(pointer_mask == 255)
                if len(pointer_points[0]) > 0:
                    pointer_coords = np.column_stack([pointer_points[1], pointer_points[0]])
                    pointer_centroid = np.mean(pointer_coords, axis=0)
                    
                    dist1 = math.sqrt((intersections[0][0] - pointer_centroid[0])**2 + 
                                    (intersections[0][1] - pointer_centroid[1])**2)
                    dist2 = math.sqrt((intersections[1][0] - pointer_centroid[0])**2 + 
                                    (intersections[1][1] - pointer_centroid[1])**2)
                    
                    intersection_point = intersections[0] if dist1 < dist2 else intersections[1]
                else:
                    intersection_point = intersections[0]
            
            # 步骤6: 根据交点位置计算读数
            angle_ratio = self.calculate_reading_from_intersection(
                intersection_point, scale_locations[0], scale_locations[1], circle_center)
            
            # 转换为刻度值
            scale_value = self.get_scale_value(angle_ratio)
            
            # 准备几何信息用于可视化
            geometry_info = {
                'circle_center': circle_center,
                'radius': radius,
                'scale_locations': scale_locations,
                'pointer_line': pointer_line,
                'intersection_point': intersection_point,
                'all_intersections': intersections,
                'fit_points': fit_points,
            }
            
            return scale_value, geometry_info
            
        except Exception as e:
            print(f"Error in improved processing: {e}")
            return None, None

    def visualize_improved_result(self, image: np.ndarray, circle_center: Tuple[float, float], 
                                 radius: float, scale_locations: Tuple[Tuple[int, int], Tuple[int, int]],
                                 pointer_line: Tuple[Tuple[float, float], Tuple[float, float]],
                                 intersection_point: Tuple[float, float], scale_value: float,
                                 fit_points: np.ndarray = None) -> np.ndarray:
        """
        创建改进算法的可视化结果
        
        Args:
            image: 原始图像
            circle_center: 刻度圆心
            radius: 刻度半径
            scale_locations: 刻度起始和结束点
            pointer_line: 指针直线的两个端点
            intersection_point: 交点
            scale_value: 计算出的刻度值
            fit_points: 用于拟合圆的点集
            
        Returns:
            可视化图像
        """
        vis_img = image.copy()
        
        # 绘制刻度圆 (黄色)
        cv2.circle(vis_img, (int(circle_center[0]), int(circle_center[1])), 
                   int(radius), (0, 255, 255), 2)
        
        # 绘制刻度圆心 (绿色)
        cv2.circle(vis_img, (int(circle_center[0]), int(circle_center[1])), 5, (0, 255, 0), -1)
        
        # 绘制刻度起始和结束点 (红色)
        cv2.circle(vis_img, scale_locations[0], 4, (0, 0, 255), -1)
        cv2.circle(vis_img, scale_locations[1], 4, (0, 0, 255), -1)
        
        # 绘制指针直线 (蓝色)
        cv2.line(vis_img, 
                 (int(pointer_line[0][0]), int(pointer_line[0][1])),
                 (int(pointer_line[1][0]), int(pointer_line[1][1])),
                 (255, 0, 0), 3)
        
        # 绘制交点 (洋红色)
        cv2.circle(vis_img, (int(intersection_point[0]), int(intersection_point[1])), 
                   6, (255, 0, 255), -1)
        
        # 添加文字说明
        text = f"Reading: {scale_value:.2f}"
        cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        text2 = f"Circle: ({circle_center[0]:.0f}, {circle_center[1]:.0f}), R={radius:.0f}"
        cv2.putText(vis_img, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw fit points if provided
        if fit_points is not None:
            for pt in fit_points:
                cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
        
        return vis_img


def load_test_data(data_dir: str) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Load test images and masks for evaluation
    
    Args:
        data_dir: Directory containing test data
        
    Returns:
        List of (image, mask, filename) tuples
    """
    data_dir = Path(data_dir)
    test_data = []
    
    # Look for image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for img_path in data_dir.rglob('*'):
        if img_path.suffix.lower() in image_extensions and 'mask' not in img_path.name.lower():
            # Try to find corresponding mask
            mask_candidates = [
                img_path.with_name(img_path.stem + '_mask' + img_path.suffix),
                img_path.with_name(img_path.stem + '_seg' + img_path.suffix),
                img_path.with_name('mask_' + img_path.name),
                img_path.with_name('seg_' + img_path.name),
            ]
            
            mask_path = None
            for candidate in mask_candidates:
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if mask_path:
                try:
                    image = cv2.imread(str(img_path))
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    
                    if image is not None and mask is not None:
                        test_data.append((image, mask, img_path.name))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    return test_data


def main():
    parser = argparse.ArgumentParser(description="Extract meter readings from segmentation masks")
    parser.add_argument("--input", "-i", required=True, help="Input image file or directory")
    parser.add_argument("--mask", "-m", help="Segmentation mask file (if single image)")
    parser.add_argument("--output", "-o", help="Output directory for results")
    parser.add_argument("--scale-range", nargs=2, type=float, default=[0.0, 1.6], 
                       help="Scale range (min max) default: 0.0 1.6")
    parser.add_argument("--debug", action="store_true", help="Show debug visualizations")
    parser.add_argument("--save-vis", action="store_true", help="Save visualization images")
    
    args = parser.parse_args()
    
    # Initialize meter reader
    meter_reader = MeterReader(
        scale_range=(args.scale_range[0], args.scale_range[1]),
        debug=args.debug
    )
    
    # Prepare output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Process single image or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image processing
        if not args.mask:
            print("Error: Mask file required for single image processing")
            return
        
        image = cv2.imread(str(input_path))
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            print("Error: Could not load image or mask")
            return
        
        print(f"Processing {input_path.name}...")
        scale_value, geometry_info = meter_reader.process_single_meter_improved(image, mask)

        
        if scale_value is not None:
            print(f"Reading: {scale_value:.3f}")
            results.append({
                "filename": input_path.name,
                "reading": scale_value,
                "geometry_info": geometry_info
            })
            
            # Save visualization if requested
            if args.save_vis and args.output:
                vis_img = meter_reader.visualize_improved_result(
                    image, 
                    geometry_info['circle_center'],
                    geometry_info['radius'],
                    geometry_info['scale_locations'],
                    geometry_info['pointer_line'],
                    geometry_info['intersection_point'],
                    scale_value,
                    geometry_info['fit_points']
                )
                vis_path = output_dir / f"{input_path.stem}_result.jpg"
                cv2.imwrite(str(vis_path), vis_img)
                print(f"Visualization saved to {vis_path}")
        else:
            print("Failed to extract reading")
    
    else:
        # Directory processing
        test_data = load_test_data(str(input_path))
        
        if not test_data:
            print(f"No valid image-mask pairs found in {input_path}")
            return
        
        print(f"Found {len(test_data)} image-mask pairs")
        
        for image, mask, filename in test_data:
            print(f"Processing {filename}...")
            scale_value, geometry_info = meter_reader.process_single_meter_improved(image, mask)

            
            if scale_value is not None:
                print(f"  Reading: {scale_value:.3f}")
                results.append({
                    "filename": filename,
                    "reading": scale_value,
                    "geometry_info": geometry_info
                })
                
                # Save visualization if requested
                if args.save_vis and args.output:
                    vis_img = meter_reader.visualize_improved_result(
                        image,
                        geometry_info['circle_center'],
                        geometry_info['radius'],
                        geometry_info['scale_locations'],
                        geometry_info['pointer_line'],
                        geometry_info['intersection_point'],
                        scale_value,
                        geometry_info['fit_points']
                    )
                    vis_path = output_dir / f"{Path(filename).stem}_result.jpg"
                    cv2.imwrite(str(vis_path), vis_img)
            else:
                print(f"  Failed to extract reading")
    
    # Save results to JSON
    if results and args.output:
        results_path = output_dir / "readings.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")
    
    # Print summary
    if results:
        readings = [r["reading"] for r in results]
        print(f"\nSummary:")
        print(f"  Processed: {len(results)} meters")
        print(f"  Average reading: {np.mean(readings):.3f}")
        print(f"  Min reading: {np.min(readings):.3f}")
        print(f"  Max reading: {np.max(readings):.3f}")


if __name__ == "__main__":
    main()