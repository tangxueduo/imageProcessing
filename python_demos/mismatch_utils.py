import cv2
import numpy as np
from skimage import measure


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
        # 连通分量算法，先对thresh_B进行膨胀，再和thresh_A执行and操作（取交集）
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
        if len(Xb[0]) > 300:
            sum_B[Xb] = 255
        thresh_B = np.zeros(mask1.shape, np.uint8)  # thresh_B大小与A相同，像素值为0
    return sum_B


def remove_small_volume(mask, threshold=5000):
    """
    :param mask:  二值图像（3D）
    :param thread: 保留体积大于指定阈值的连通域
    :return: 主要连通域 和 label(含背景)
    """
    cc, cc_num = measure.label(mask, return_num=True, connectivity=1)
    res = np.zeros(mask.shape, np.uint8)
    properties = measure.regionprops(cc)
    for i, prop in enumerate(properties):
        if prop.area > threshold:
            # 记录区域像素个数
            res[cc == prop.label] = prop.label
    res_unique = np.unique(res).tolist()
    res_unique.remove(0)
    print(f"*****res_unique: {res_unique}")
    # TODO2: 知道病灶在哪几层，(连通域在哪几层)
    # print(np.count_nonzero(res==4), np.count_nonzero(res==7))
    print(f"连通域个数: {len(res_unique) - 1}")
    return res, res_unique


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


def get_point_on_plane_by_yz(A, B, C, D, height, start_index, end_index):
    """由平面方程获取直线方程
    产品约定: 平面 size 取为距原图上下各 10% 大小
    """
    # 由 Ax + By + Cz + D = 0 推出， x = - (D+By+Cz)/A
    y_bottom, y_top = height * 0.1, height * 0.9

    z_bottom, z_top = start_index, end_index
    # 大脑中线上下端点
    x_bottom, x_top = (
        -(D + B * y_bottom + C * z_bottom) / A,
        -(D + B * y_top + C * z_top) / A,
    )
    return (x_bottom, y_bottom), (x_top, y_top)


def gray2rgb_array(gray_array):
    temp_array = gray_array
    max_pt = np.max(temp_array)
    min_pt = np.min(temp_array)
    window_width = 100
    window_level = 50
    true_max_pt = window_level + (window_width / 2)
    true_min_pt = window_level - (window_width / 2)

    scale = 255 / (true_max_pt - true_min_pt)
    temp_array = np.clip(temp_array, true_min_pt, true_max_pt)
    min_pt_array = np.ones((temp_array.shape[0], temp_array.shape[1])) * true_min_pt
    temp_array = (temp_array - min_pt_array) * scale

    rgb_array = np.zeros((temp_array.shape[0], temp_array.shape[1], 3))
    rgb_array[:, :, 0] = temp_array
    rgb_array[:, :, 1] = temp_array
    rgb_array[:, :, 2] = temp_array

    return rgb_array
