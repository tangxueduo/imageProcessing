from collections import deque

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# 计算种子点和其领域的像素值之差
def getGrayDiff(gray, current_seed, tmp_seed):
    return abs(
        int(gray[current_seed[0], current_seed[1]])
        - int(gray[tmp_seed[0], tmp_seed[1]])
    )


# 区域生长算法
def regional_growth(gray):
    # 八领域
    connects = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    seedMark = np.zeros((gray.shape))
    height, width = gray.shape
    threshold = 6
    seedque = deque()
    label = 255
    seedque.extend(seeds)

    while seedque:
        # 队列具有先进先出的性质。所以要左删
        current_seed = seedque.popleft()
        seedMark[current_seed[0], current_seed[1]] = label
        for i in range(8):
            tmpX = current_seed[0] + connects[i][0]
            tmpY = current_seed[1] + connects[i][1]
            # 处理边界情况
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width:
                continue

            grayDiff = getGrayDiff(gray, current_seed, (tmpX, tmpY))
            if grayDiff < threshold and seedMark[tmpX, tmpY] != label:
                seedque.append((tmpX, tmpY))
                seedMark[tmpX, tmpY] = label
    return seedMark


# 交互函数
def Event_Mouse(event, x, y, flags, param):
    global seeds
    # 左击鼠标
    if event == cv.EVENT_LBUTTONDOWN:
        # 添加种子
        seeds.append((y, x))
        print(y, x)
        # 画实心点
        cv.circle(param, center=(x, y), radius=2, color=(0, 0, 255), thickness=-1)
        # return seeds
        print(seeds)


def Region_Grow(img):
    cv.namedWindow("img")
    cv.setMouseCallback("img", Event_Mouse, img)
    cv.imshow("img", img)

    while True:
        cv.imshow("img", img)
        key = cv.waitKey(10)
        if key == 27 or key == "q":
            break
    cv.destroyAllWindows()

    CT = cv.imread("images/png/CT.png", 1)
    seedMark = np.uint8(regional_growth(cv.cvtColor(CT, cv.COLOR_BGR2GRAY)))

    cv.imshow("seedMark", seedMark)
    # 若需要查看seedMask，取消下面注释即可
    # cv.waitKey(0)

    plt.figure(figsize=(100, 100))
    plt.subplot(131), plt.imshow(cv.cvtColor(CT, cv.COLOR_BGR2RGB))
    plt.axis("off"), plt.title(f"$input\_image$")
    plt.subplot(132), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis("off"), plt.title(f"$seeds\_image$")
    plt.subplot(133), plt.imshow(seedMark, cmap="gray", vmin=0, vmax=255)
    plt.axis("off"), plt.title(f"$segmented\_image$")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sitk = __import__("SimpleITK")
    sitk_img = sitk.ReadImage(
        "/media/tx-deepocean/Data/DICOMS/demos/ct_heart/1.2.392.200036.9116.2.6.1.37.2420991567.1641645488.948725/1.2.392.200036.9116.2.6.1.37.2420991567.1641645571.623527.dcm"
    )
    img = sitk.GetArrayFromImage(sitk_img)
    img = np.transpose(img, (1, 2, 0))
    print(img.shape)
    cv.imwrite("images/png/CT.png", img)
    img = cv.imread("images/png/CT.png")
    seeds = []
    Region_Grow(img)
