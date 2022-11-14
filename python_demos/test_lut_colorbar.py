import cv2
import numpy as np

grey_img = np.zeros([15, 256], np.uint8)
for i in range(15):
    for j in range(256):
        grey_img[i, j] = j
color_img = cv2.applyColorMap(grey_img, cv2.COLORMAP_JET)  # 自己测试turbo
for j in range(256):
    print(f"{[color_img[0, j, 2], color_img[0, j, 1], color_img[0, j, 0]]}")
color_img = np.rot90(color_img)
cv2.imwrite("./colormap.png", color_img)

# key = cv2.waitKey(0)
