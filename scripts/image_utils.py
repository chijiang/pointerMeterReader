import cv2
import numpy as np
from pathlib import Path

def unwarp_meter_using_ellipse(image_path, output_path, debug_path="debug_detection.jpg"):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("Error: 无法读取图像")
        return
    
    debug_img = img.copy()
    
    # 2. contour detection
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # edge detection on the gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # gaussian blur to denoise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # 3. contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    target_ellipse = None
    max_area = 0
    
    # 4. fit ellipse to the best contour
    for cnt in contours:
        # filter out small noise
        if cv2.contourArea(cnt) < 300:
            continue
            
        # fit ellipse (requires at least 5 points)
        if len(cnt) < 5:
            continue
            
        ellipse = cv2.fitEllipse(cnt)
        (xc, yc), (MA, ma), angle = ellipse
        
        # filter logic:
        # 1. the gauge should be close to a circle, the aspect ratio should not be too exaggerated (unless the angle is extremely skewed)
        # 2. the area should be the largest (or the largest closed circle)
        
        # ellipse with the largest area as the target
        # MA/ma < 2.0 (to prevent misrecognition of long objects)
        area = np.pi * (MA/2) * (ma/2)
        if area > max_area:
            max_area = area
            target_ellipse = ellipse

    if target_ellipse is None:
        print("未找到合适的仪表盘轮廓")
        return

    # draw ellipse
    cv2.ellipse(debug_img, target_ellipse, (0, 0, 255), 2)
    
    # 5. conv matrix
    box = cv2.boxPoints(target_ellipse)
    box = np.int32(box)
    
    # 绘制检测到的矩形框（绿色）
    cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 2)
    
    # 保存调试图看看抓得对不对
    cv2.imwrite(debug_path, debug_img)
    
    # --- 核心矫正逻辑 ---
    
    # reorder the four points
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # top left (sum最小)
        rect[2] = pts[np.argmax(s)] # bottom right (sum最大)
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # top right (y-x 最小)
        rect[3] = pts[np.argmax(diff)] # bottom left (y-x 最大)
        return rect

    src_pts = order_points(box)
    
    # 目标尺寸：
    # 理论上，正圆的外接矩形是正方形。
    (xc, yc), (MA, ma), angle = target_ellipse
    side_len = int(MA)
    
    # standard square
    dst_pts = np.array([
        [0, 0],
        [side_len - 1, 0],
        [side_len - 1, side_len - 1],
        [0, side_len - 1]
    ], dtype="float32")
    
    # perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # perform transformation
    warped = cv2.warpPerspective(img, M, (side_len, side_len))
    
    # 6. 旋转修正 (可选)
    # 因为 fitEllipse 的角度可能会导致矫正后的图片是旋转的（比如刻度盘歪了）
    # 如果需要正向（比如字是正的），可能需要后处理，或者根据 fitEllipse 的 angle 再次旋转
    # 这里我们做简单保存
    
    cv2.imwrite(output_path, warped)
    print(f"处理完成: {output_path} (调试图: {debug_path})")

def batch_unwarp_meters_in_folder(input_dir, output_dir):
    """Read all jpg/jpeg images in a folder and unwarp them to the output folder."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")) + list(input_path.glob("*.JPG")))
    if not images:
        print(f"未在 {input_path} 找到 jpg/jpeg 图片")
        return

    for img_path in images:
        cropped_path = output_path / f"{img_path.stem}_cropped.jpg"
        debug_img_path = output_path / f"{img_path.stem}_debug.jpg"
        print(f"处理文件: {img_path} -> {cropped_path}")
        unwarp_meter_using_ellipse(str(img_path), str(cropped_path), str(debug_img_path))

if __name__ == "__main__":
    # 运行
    # unwarp_meter_using_ellipse('9ef36abc35b0ab0291c95745af34b433.jpg', 'result_circle.jpg')

    # 使用示例
    # unwarp_meter_using_ellipse('/Users/joe.lu/Desktop/Screenshot 2025-11-24 at 21.31.17.png', '/Users/joe.lu/Desktop/corrected3.jpg')
    batch_unwarp_meters_in_folder(
        '/Users/joe.lu/Desktop/诺华工厂/panel_image',
        '/Users/joe.lu/Desktop/诺华工厂/panel_image_output'
    )
