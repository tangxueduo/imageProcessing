import numpy as np
import cv2

jpg_path = "/home/tx-deepocean/Downloads/tree_subset.jpg"
img = cv2.imread(jpg_path)
img = cv2.resize(img, (512, 512))
print(img.shape)
# 逆时针旋转45度
center = (img.shape[0]//2, img.shape[1]//2)
angle = 20
# 注意 scale 调整为小于1 才能保证旋转后的图片和原来相似，但是需要确认这一做法是否对图片信息丢失
M1 = cv2.getRotationMatrix2D(center, angle, 0.75)

# 矩阵逆变换
M2 = np.vstack((M1, (0,0,1)))
M2 = np.linalg.inv(M2)
M2 = np.delete(M2, 2, axis = 0)
print(M2)

img1 = cv2.warpAffine(img, M1, (img.shape[0], img.shape[1]))
# M2 = cv2.getRotationMatrix2D((256,256), -angle, 0.7)
img2 = cv2.warpAffine(img1, M2, (img.shape[0], img.shape[1]))
print(img2.shape)
cv2.imshow("affine", img2)
cv2.waitKey(0)
i = 1