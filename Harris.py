import cv2
import os
import numpy as np

block_size = 5 # 用于角点检测的邻域大小
sobel_size = 3 # 用于计算梯度图的sobel算子的尺寸
k = 0.04       # 用于计算角点响应函数的参数k, 取值范围常在0.04~0.06之间

image = cv2.imread('./img/harris.jpg')
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_img = np.float32(gray_img)

corners_img = cv2.cornerHarris(gray_img, block_size, sobel_size, k)

# result is dilated for marking the corners, not necessary
dst = cv2.dilate(corners_img, None)

# Threshold for an optimal value, marking the corners in Green
image[corners_img>0.01*corners_img.max()] = [0,0,255]

cv2.imwrite('new_harris.jpg', image)
