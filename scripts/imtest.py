import cv2
import numpy as np

image = cv2.imread('/Users/chijiang_1/dev_env/pointMeterDetection/data/segmentation/SegmentationClass/new_001.png')
print("Min pixel value:", np.min(image))
print("Max pixel value:", np.max(image))

image_scaled = (image * 127.5).astype(np.uint8)

# 显示图像
cv2.imshow('Image', image_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()

