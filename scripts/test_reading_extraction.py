#!/usr/bin/env python3
"""
Simple test script for meter reading extraction
Tests the reading extraction algorithm on sample data
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add the scripts directory to path to import extract_meter_reading
sys.path.append(str(Path(__file__).parent))
from extract_meter_reading import MeterReader


def create_synthetic_test_data():
    """Create synthetic test data for algorithm validation"""
    
    # Create a simple synthetic meter image (224x224)
    img_size = 224
    image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Draw a circular meter background (gray)
    center = (img_size // 2, img_size // 2)
    radius = img_size // 3
    cv2.circle(image, center, radius, (100, 100, 100), -1)
    
    # Create segmentation mask
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Add scale markings (class 2) - arc from left to right
    scale_start_angle = 210  # degrees
    scale_end_angle = 330
    for angle in range(scale_start_angle, scale_end_angle + 1, 5):
        rad = np.radians(angle)
        x = int(center[0] + (radius - 10) * np.cos(rad))
        y = int(center[1] + (radius - 10) * np.sin(rad))
        cv2.circle(mask, (x, y), 3, 2, -1)  # class 2 for scale
    
    # Add pointer (class 1) - pointing to middle position
    pointer_angle = 270  # pointing down (middle of scale)
    pointer_length = radius - 20
    pointer_end_x = int(center[0] + pointer_length * np.cos(np.radians(pointer_angle)))
    pointer_end_y = int(center[1] + pointer_length * np.sin(np.radians(pointer_angle)))
    
    # Draw pointer as a thick line
    cv2.line(mask, center, (pointer_end_x, pointer_end_y), 1, 5)  # class 1 for pointer
    
    return image, mask


def test_with_synthetic_data():
    """Test the algorithm with synthetic data"""
    print("Testing with synthetic data...")
    
    # Create synthetic test data
    image, mask = create_synthetic_test_data()
    
    # Initialize meter reader
    meter_reader = MeterReader(scale_range=(0.0, 1.6), debug=False)
    
    # Process the synthetic meter
    reading = meter_reader.process_single_meter(image, mask)
    
    if reading is not None:
        print(f"Synthetic test reading: {reading:.3f}")
        print("Expected reading should be around 0.8 (middle of 0.0-1.6 range)")
        
        # Save test images
        cv2.imwrite("test_synthetic_image.jpg", image)
        # Save mask with proper class values (multiply by 127 for visibility but keep original values)
        mask_vis = mask.copy()
        mask_vis[mask == 1] = 127  # Pointer -> gray
        mask_vis[mask == 2] = 255  # Scale -> white
        cv2.imwrite("test_synthetic_mask.jpg", mask_vis)
        # Also save the original mask for processing
        cv2.imwrite("test_synthetic_mask_original.png", mask)
        print("Saved test_synthetic_image.jpg and test_synthetic_mask.jpg")
        
        return True
    else:
        print("Failed to extract reading from synthetic data")
        return False


def test_with_real_data():
    """Test with real segmentation data if available"""
    print("\nTesting with real data...")
    
    # Look for real test data in the data directory
    data_dir = Path("data")
    if not data_dir.exists():
        print("No data directory found, skipping real data test")
        return True
    
    # Look for segmentation validation data
    val_dir = data_dir / "segmentation" / "val"
    if not val_dir.exists():
        print("No validation data found, skipping real data test")
        return True
    
    # Find first image-mask pair
    image_files = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
    if not image_files:
        print("No image files found in validation directory")
        return True
    
    # Try to find corresponding mask
    for img_path in image_files[:3]:  # Test first 3 images
        # Look for corresponding mask file
        mask_path = None
        possible_mask_names = [
            img_path.with_suffix('.png'),  # Same name, different extension
            img_path.parent / f"mask_{img_path.name}",
            img_path.parent / f"{img_path.stem}_mask.png",
            img_path.parent / f"{img_path.stem}_seg.png"
        ]
        
        for candidate in possible_mask_names:
            if candidate.exists():
                mask_path = candidate
                break
        
        if mask_path:
            print(f"Testing with {img_path.name} and {mask_path.name}")
            
            # Load image and mask
            image = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if image is not None and mask is not None:
                # Initialize meter reader
                meter_reader = MeterReader(scale_range=(0.0, 1.6), debug=True)
                
                # Process the real meter
                reading = meter_reader.process_single_meter(image, mask)
                
                if reading is not None:
                    print(f"  Reading: {reading:.3f}")
                else:
                    print(f"  Failed to extract reading")
            else:
                print(f"  Failed to load image or mask")
        else:
            print(f"No mask found for {img_path.name}")
    
    return True


def main():
    """Main test function"""
    print("Meter Reading Extraction Test")
    print("=" * 40)
    
    # Test 1: Synthetic data
    success1 = test_with_synthetic_data()
    
    # Test 2: Real data (if available)
    success2 = test_with_real_data()
    
    print("\n" + "=" * 40)
    if success1 and success2:
        print("All tests completed successfully!")
        print("\nUsage examples:")
        print("1. Single image:")
        print("   python scripts/extract_meter_reading.py -i image.jpg -m mask.png --debug")
        print("\n2. Batch processing:")
        print("   python scripts/extract_meter_reading.py -i data/test_images/ -o results/ --save-vis")
        print("\n3. Custom scale range:")
        print("   python scripts/extract_meter_reading.py -i image.jpg -m mask.png --scale-range 0 2.5")
    else:
        print("Some tests failed. Please check the implementation.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 