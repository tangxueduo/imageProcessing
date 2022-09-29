import random

import cv2
import numpy as np
import SimpleITK as sitk

# 定义一个512*512三通道彩色图像，设像素值[255,255,255]为正常点
grey_img = np.ones([512, 512, 3], np.uint8) * 255

# 随机选择20000个点像素值设为[0,0,0]，设[0,0,0]为异常点
for i in range(20000):
    x = random.randint(0, 511)
    y = random.randint(0, 511)
    # print(x, y)
    grey_img[x, y, :] = 0

cv2.imwrite("./input.jpg", grey_img)

point1 = [10, 10]  # 中线端点1
point2 = [80, 80]  # 中线端点2

# 注意采用x=ky+b模型，原因是如果中线没有偏转，例x=10这种情况，x=ky+b不用考虑这种特殊情况，不用额外写个if条件判断
k = (point2[0] - point1[0]) / (point2[1] - point1[1])  # 计算中线斜率k
b = point1[0] - k * point1[1]  # 计算b

error_arr = np.argwhere(grey_img == [0, 0, 0])  # 查找异常点，即查找像素值为[0,0,0]
print(error_arr)

idx_bottom = np.argwhere(
    (error_arr[:, 0] - k * error_arr[:, 1] - b) > 0
)  # 在异常点位置数组中判断哪些点的索引在中线下方
idx_top = np.argwhere(
    (error_arr[:, 0] - k * error_arr[:, 1] - b) <= 0
)  # 在异常点位置数组中判断哪些点的索引在中线上方
etl = idx_top[:, 0]  # 因argwhere输出二维数组，第二列全是0,只提取第一列索引
ebr = idx_bottom[:, 0]  # 同上
print(error_arr[etl])

# 将在中线上方的异常点像素值置为[0,255,0]
grey_img[error_arr[etl][:, 0], error_arr[etl][:, 1], 0] = 0
grey_img[error_arr[etl][:, 0], error_arr[etl][:, 1], 1] = 255
grey_img[error_arr[etl][:, 0], error_arr[etl][:, 1], 2] = 0

cv2.imwrite("./result.jpg", grey_img)
