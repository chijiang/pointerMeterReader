import cv2
import numpy as np

# 读取图片（灰度模式）
scale = cv2.imread('task-173-annotation-243-by-1-tag-scale-0.png', cv2.IMREAD_GRAYSCALE)
pointer = cv2.imread('task-173-annotation-243-by-1-tag-pointer-0.png', cv2.IMREAD_GRAYSCALE)

# 创建三通道黑色背景
h, w = scale.shape
result = np.zeros((h, w, 3), dtype=np.uint8)

# 刻度为绿色
scale_mask = scale > 128  # 白色部分
result[scale_mask] = [0, 255, 0]  # BGR: 绿色

# 指针为红色（覆盖刻度）
pointer_mask = pointer > 128
result[pointer_mask] = [0, 0, 255]  # BGR: 红色

# 保存结果
cv2.imwrite('00_pic2.png', result)