import cv2 as cv
import numpy as np

contour = []
jpg_path = "/home/tx-deepocean/Downloads/black_cat.jpg"
img = cv.imread(jpg_path)
img = cv.resize(img, (512, 512))
print(img.shape)


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# 获取模型的 contour 点， 转像素坐标
print(f"contour 个数: len(cnts)")
# print(len(cnts[0]))
rect = cv.minAreaRect(cnts[0])
# 获取四个顶点
box = cv.boxPoints(rect)
box = np.int0(box)
print(f"*****bbox: {box}")
img = cv.drawContours(img, [box], 0, (0, 0, 255), 2)


cv.imshow("bbox", img)
while 1:
    if cv.waitKey(100) == 27:  # 100ms or esc(ascii 等于27, 跳出循环)
        break
cv.destroyAllWindows()
