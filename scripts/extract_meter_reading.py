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
    
    def __init__(self, scale_range: Tuple[float, float] = (0.0, 1.6), debug: bool = False):
        """
        Initialize meter reader
        
        Args:
            scale_range: (min_value, max_value) of the meter scale
            debug: Whether to show debug visualizations
        """
        self.scale_beginning, self.scale_end = scale_range
        self.debug = debug
        
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
            
            # Find meter center
            center = self.get_center_location(image)
            if center is None:
                print("Failed to find meter center")
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
        scale_value = meter_reader.process_single_meter(image, mask)
        
        if scale_value is not None:
            print(f"Reading: {scale_value:.3f}")
            results.append({
                "filename": input_path.name,
                "reading": scale_value
            })
            
            # Save visualization if requested
            if args.save_vis and args.output:
                vis_img = meter_reader.visualize_result(
                    image, 
                    meter_reader.get_scale_locations(meter_reader.threshold_by_category(mask, 2)),
                    meter_reader.get_pointer_locations(meter_reader.threshold_by_category(mask, 1), 
                                                     meter_reader.get_center_location(image)),
                    meter_reader.get_center_location(image),
                    scale_value
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
            scale_value = meter_reader.process_single_meter(image, mask)
            
            if scale_value is not None:
                print(f"  Reading: {scale_value:.3f}")
                results.append({
                    "filename": filename,
                    "reading": scale_value
                })
                
                # Save visualization if requested
                if args.save_vis and args.output:
                    vis_img = meter_reader.visualize_result(
                        image,
                        meter_reader.get_scale_locations(meter_reader.threshold_by_category(mask, 2)),
                        meter_reader.get_pointer_locations(meter_reader.threshold_by_category(mask, 1), 
                                                         meter_reader.get_center_location(image)),
                        meter_reader.get_center_location(image),
                        scale_value
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