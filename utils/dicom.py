import cv2
import numpy as np
import pydicom
import SimpleITK as sitk


def dicom2jpg(ww: float = 800, wl: float = 150):
    """Adjust according to VOI LUT, window center(level) and width values not normalized"""
    path = "/home/tx-deepocean/Downloads/1.2.840.113619.2.416.7508114979325500849480401061818955984.222"
    ds = pydicom.read_file(path)

    print(ds.RescaleSlope, ds.RescaleIntercept)

    arr = ds.pixel_array
    arr = (arr) * ds.RescaleSlope + ds.RescaleIntercept

    ww_max = wl + (ww / 2)
    ww_min = wl - (ww / 2)

    # 归一化到指定区间
    print(arr.min(), arr.max())
    gray_array = np.clip(arr, ww_min, ww_max)
    min_pt = np.ones_like(gray_array) * ww_min
    gray_array = (gray_array - min_pt) * 255 / (ww_max - ww_min)
    # arr = np.where(arr < ww_min, arr.min(), arr)
    # arr = np.where(arr > ww_max, arr.max(), arr)

    # # 线性插值（x0,y0）, (x1, y1), https://zhuanlan.zhihu.com/p/34492145
    # # 已知想求y: y = y0 + (x-x0)(y1-y0)/(x1-x0)
    # gray_array = np.where(
    #     (arr >= ww_min) & (arr < ww_max),
    #     arr.min() + (arr - wl + ww / 2) / ww * (arr.max() - arr.min()),
    #     arr,
    # )
    # # TODO　下面这个计算不对
    # # gray_array = np.piecewise(arr, [arr<ww_min, arr>=ww_max], [arr.min(), arr.max(), lambda arr: arr.min() + (arr - wl + ww / 2) / ww * (arr.max() - arr.min())])
    # # 归一化0， 255 至灰度
    # gray_array = (
    #     (gray_array - gray_array.min()) / (gray_array.max() - gray_array.min())
    # ) * 255.0
    cv2.imwrite("./test.png", gray_array)


dicom2jpg()


def resample_img(src_img, interpolator=sitk.sitkLinear, dest_spacing=[1.0, 1.0, 1.0]):
    """灰度图重采样"""
    src_size = src_img.GetSize()
    src_spacing = src_img.GetSpacing()
    #
    dest_size = np.array(src_size) * np.array(src_spacing) / np.array(dest_spacing)
    dest_size = np.round(dest_size).astype(np.int64).tolist()

    dst_img = sitk.Resample(
        src_img,
        dest_size,
        sitk.Transform(),
        interpolator,
        src_img.GetOrigin(),
        dest_spacing,
        src_img.GetDirection(),
        0,
        src_img.GetPixelID(),
    )
    return dst_img
