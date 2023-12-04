import cv2
import numpy as np
import pydicom


def dicom2jpg():
    """Adjust according to VOI LUT, window center(level) and width values not normalized"""
    path = "/home/tx-deepocean/Downloads/1.2.840.113619.2.416.7508114979325500849480401061818955984.222"
    ds = pydicom.read_file(path)

    print(ds.RescaleSlope, ds.RescaleIntercept)

    arr = ds.pixel_array
    arr = (arr) * ds.RescaleSlope + ds.RescaleIntercept

    ww, wl = 800, 150
    ww_max = wl + (ww / 2)
    ww_min = wl - (ww / 2)

    # 归一化到指定区间
    arr = np.where(arr < ww_min, arr.min(), arr)
    arr = np.where(arr > ww_max, arr.max(), arr)

    # 线性插值（x0,y0）, (x1, y1), https://zhuanlan.zhihu.com/p/34492145
    # 已知想求y: y = y0 + (x-x0)(y1-y0)/(x1-x0)
    gray_array = np.where(
        (arr >= ww_min) & (arr < ww_max),
        arr.min() + (arr - wl + ww / 2) / ww * (arr.max() - arr.min()),
        arr,
    )

    # 归一化0， 255 至灰度
    gray_array = (
        (gray_array - gray_array.min()) / (gray_array.max() - gray_array.min())
    ) * 255.0
    cv2.imwrite("./test.png", gray_array)
