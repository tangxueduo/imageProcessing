import math
import time
from functools import wraps

import cv2
import numpy as np
from skimage import measure


def timer(func):
    """
    通过decorator 显示函数运行的性能
    例如，可快速对比map和list comprehension性能差。
    要求： 除了显示CPU用时信息外，尽可能详细地显示所有性能相关信息。可以google
    """

    @wraps(func)  # <- 用于保留原函数信息
    def inner(*args, **kwargs):
        before = time.time()
        result = func(*args, **kwargs)
        after = time.time()
        print(f"*******{func.__name__} elapsed: ", after - before)
        return result

    return inner


# @timer
def get_cc(mask1, kernel):
    """
    :param mask1 (2d)
    return 主要连通域
    """
    # https://www.cnblogs.com/er-gou-zi/p/11985633.html
    thresh_A_copy = mask1.copy()
    thresh_B = np.zeros(mask1.shape, np.uint8)  # thresh_B大小与A相同，像素值为0
    thresh_B_old = np.zeros(mask1.shape, np.uint8)  # thresh_B大小与A相同，像素值为0
    sum_B = np.zeros(mask1.shape, np.uint8)
    while thresh_A_copy.any():
        Xa_copy, Ya_copy = np.where(thresh_A_copy > 0)  # thresh_A_copy中值为255的像素的坐标
        thresh_B[Xa_copy[0]][Ya_copy[0]] = 255  # 选取第一个点，并将thresh_B中对应像素值改为255
        thresh_B_old = thresh_B
        # 连通分量算法，先对thresh_B进行膨胀，再和 thresh_A 执行and操作（取交集）
        for i in range(50):
            dilation_B = cv2.dilate(thresh_B, kernel, iterations=1)
            thresh_B = cv2.bitwise_and(mask1, dilation_B)
            if (thresh_B_old == thresh_B).all():
                break
            thresh_B_old = thresh_B

        # 取 thresh_B 值为 255 的像素坐标，并将 thresh_A_copy 中对应坐标像素值变为0
        Xb = np.where(thresh_B > 0)
        thresh_A_copy[Xb] = 0
        # 过滤小于300的
        if len(Xb[0]) > 200:
            sum_B[Xb] = 255
        thresh_B = np.zeros(mask1.shape, np.uint8)  # thresh_B 大小与A相同，像素值为0
    return sum_B


def get_rotate_direction(k, p1_x, p1_y, p2_x, p2_y):
    """根据斜率和两点确定旋转方向"""
    # 计算倾斜角，中线中点
    if k == 0:
        angle, center = 0, (
            (p1_x + p2_x) / 2,
            (p1_y + p2_y) / 2,
        )
    else:
        angle, center = (math.atan(k)) / math.pi * 180, (
            (p1_x + p2_x) / 2,
            (p1_y + p2_y) / 2,
        )
    return angle, center


def trans_list_to_array(mask_list, final_shape):
    mask = np.array(mask_list).reshape(final_shape)
    return mask


def remove_small_volume(mask, threshold=1000):
    """
    :param mask:  二值图像（3D）
    :param thread: 保留体积大于指定阈值的连通域
    :return: 主要连通域 和 label(含背景)
    """
    cc, cc_num = measure.label(mask, return_num=True, connectivity=1)
    res = np.zeros(mask.shape, np.uint8)
    properties = measure.regionprops(cc)
    slice_ids = {}
    for i, prop in enumerate(properties):
        if prop.area > threshold:
            # 记录区域像素个数
            res[cc == prop.label] = prop.label
            # 获得连通域所在层
            points = np.argwhere(cc == prop.label)
            zmin = np.min(points[:, 0])
            zmax = np.max(points[:, 0])
            slice_ids[prop.label] = [int(zmin), int(zmax)]
    res_unique = np.unique(res).tolist()
    res_unique.remove(0)
    print(f"*****res_unique: {res_unique}")
    # TODO2: 知道病灶在哪几层，(连通域在哪几层)
    # print(np.count_nonzero(res==4), np.count_nonzero(res==7))
    print(f"连通域个数: {len(res_unique)}")
    return res, res_unique, slice_ids


def get_line(x1, x2, y1, y2):
    """由两点坐标求k,b"""
    k = (x2 - x1) / (y2 - y1)  # 计算中线斜率k
    b = x1 - k * y1  # 计算b
    # 不知道为什么要关于 y=x 做个对称
    b = -b / k
    k = 1 / k
    return k, b


def convert_noraml_to_standard_plane(point, vector):
    """点法式平面方程转为标准式方程"""
    A, B, C = vector[0], vector[1], vector[2]
    D = -(np.dot(vector, point))
    return A, B, C, D


def get_point_on_plane_by_yz(a, b, c, d, height, z, spacing_y, origin_y):
    """由平面方程获取直线方程
    产品约定: 平面 size 取为距原图上下各 10% 大小
    """
    # 由 Ax + By + Cz + D = 0 推出， x = - (D+By+Cz)/A
    y_top, y_bottom = (
        height * 0.1 * spacing_y + origin_y,
        height * 0.9 * spacing_y + origin_y,
    )

    # 大脑中线上下端点
    x_top, x_bottom = (
        -(d + b * y_top + c * z) / a,
        -(d + b * y_bottom + c * z) / a,
    )
    p1 = (x_top, y_top)
    p2 = (x_bottom, y_bottom)
    return p1, p2


def convert_ijk_to_xyz(point, patient_position, spacing):
    """物理坐标，转像素坐标"""
    sub_abs = np.abs(np.subtract(point, patient_position))
    xyz = np.divide(sub_abs, spacing)
    xyz = [round(i) for i in xyz]
    return xyz


def gray2rgb_array(gray_array, ww, wl, is_colormap=False):
    temp_array = gray_array
    window_width = ww
    window_level = wl
    true_max_pt = window_level + (window_width / 2)
    true_min_pt = window_level - (window_width / 2)

    scale = 255 / (true_max_pt - true_min_pt)
    temp_array = np.clip(temp_array, true_min_pt, true_max_pt)
    min_pt_array = np.ones((temp_array.shape[0], temp_array.shape[1])) * true_min_pt
    temp_array = (temp_array - min_pt_array) * scale

    if not is_colormap:
        rgb_array = np.zeros((temp_array.shape[0], temp_array.shape[1], 3))
        rgb_array[:, :, 0] = temp_array
        rgb_array[:, :, 1] = temp_array
        rgb_array[:, :, 2] = temp_array
    else:
        rgb_array = np.zeros((temp_array.shape[0], temp_array.shape[1]))
    return rgb_array


def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)


def bgr_to_rgb(rgb_array):
    rgb_array = rgb_array.astype(np.uint8)
    rgb_array = rgb_array[:, :, ::-1]
    return rgb_array


def np_array_to_dcm(
    ds,
    np_array: np.ndarray,
    save_path: str,
    ww,
    wl,
    is_rgb=False,
):
    """save numpy array to dicom"""
    # TODO: 是否需要补充tag
    if is_rgb:
        ds.WindowWidth = ww
        ds.WindowCenter = wl
        ds.BitsStored = 8
        ds.BitsAllocated = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PhotometricInterpretation = "RGB"
        ds.SamplesPerPixel = 3
        ds.PlanarConfiguration = 0
    else:
        print(ww, wl)
        ds.WindowWidth = ww
        ds.WindowCenter = wl
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    ds.PixelData = np_array.tobytes()
    ds.save_as(save_path)
