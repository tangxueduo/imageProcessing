import math

import cv2
import numpy as np
import SimpleITK as sitk


def gray2rgb_array(gray_array):
    temp_array = gray_array
    # max_pt = np.max(temp_array)
    # min_pt = np.min(temp_array)
    window_width = 1700
    window_level = -600
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


def get_line(x1, x2, y1, y2):
    """由两点坐标求k,b"""
    k = (x2 - x1) / (y2 - y1)  # 计算中线斜率k
    b = x1 - k * y1  # 计算b
    # 不知道为什么要关于 y=x 做个对称
    b = -b / k
    k = 1 / k
    return k, b


if __name__ == "__main__":
    cbf_img = sitk.ReadImage("CBF.nii.gz")
    cbf_img = sitk.GetArrayFromImage(cbf_img)
    cbf_img = cbf_img[8, :, :]
    tmip_img = sitk.ReadImage("TMIP_NO_SKULL.nii.gz")
    tmip_img = sitk.GetArrayFromImage(tmip_img)
    tmip_img = tmip_img[8, :, :]
    tmip_img = gray2rgb_array(tmip_img)
    # print(tmip_img.shape, np.max(tmip_img), np.min(tmip_img))
    start = cv2.getTickCount()
    (x1, y1), (x2, y2) = (282, 106), (240, 412)
    k, b = get_line(x1, x2, y1, y2)
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    angle = (math.atan(k)) / math.pi * 180
    # tmip_img = cv2.line(tmip_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle + 90, scale=1)
    cbf_rotated_image = cv2.warpAffine(
        src=cbf_img, M=rotate_matrix, dsize=(cbf_img.shape[1], cbf_img.shape[0])
    )
    tmip_rotated_image = cv2.warpAffine(
        src=tmip_img, M=rotate_matrix, dsize=(tmip_img.shape[1], tmip_img.shape[0])
    )
    print(cbf_rotated_image.shape, np.max(cbf_rotated_image), np.min(cbf_rotated_image))

    left_cbf = cbf_rotated_image
    mirror_cbf = np.fliplr(cbf_rotated_image)
    # https://blog.csdn.net/qq_41873079/article/details/115484857
    temp_cbf = np.divide(
        left_cbf, mirror_cbf, out=np.zeros_like(left_cbf), where=mirror_cbf > 0
    )

    setting_value = 40
    errorIdx1 = np.where((0 < temp_cbf) & (temp_cbf < setting_value / 100))
    setting_value = 30
    errorIdx2 = np.where((0 < temp_cbf) & (temp_cbf < setting_value / 100))
    setting_value = 20
    errorIdx3 = np.where((0 < temp_cbf) & (temp_cbf < setting_value / 100))

    mask1 = np.zeros((cbf_img.shape[1], cbf_img.shape[0]), np.uint8)
    mask1[errorIdx1] = 255
    mask2 = np.zeros((cbf_img.shape[1], cbf_img.shape[0]), np.uint8)
    mask2[errorIdx2] = 255
    mask3 = np.zeros((cbf_img.shape[1], cbf_img.shape[0]), np.uint8)
    mask3[errorIdx3] = 255
    # mask2 = cv2.dilate(mask, np.ones(shape=[3,3],dtype=np.uint8), iterations=2)
    # mask2 = cv2.erode(mask2, np.ones(shape=[3,3],dtype=np.uint8), iterations=2)

    # https://www.cnblogs.com/er-gou-zi/p/11985633.html
    thresh_A_copy = mask1.copy()
    thresh_B = np.zeros(mask1.shape, np.uint8)  # thresh_B大小与A相同，像素值为0
    thresh_B_old = np.zeros(mask1.shape, np.uint8)  # thresh_B大小与A相同，像素值为0
    sum_B = np.zeros(mask1.shape, np.uint8)  # thresh_B大小与A相同，像素值为0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 3×3结构元
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

        # 取thresh_B值为255的像素坐标，并将thresh_A_copy中对应坐标像素值变为0
        Xb = np.where(thresh_B > 0)
        thresh_A_copy[Xb] = 0
        if len(Xb[0]) > 300:
            tmip_rotated_image[Xb] = [0, 255, 0]
            sum_B[Xb] = 255
        thresh_B = np.zeros(mask1.shape, np.uint8)  # thresh_B大小与A相同，像素值为0

    thresh2 = cv2.bitwise_and(mask2, sum_B)
    tmip_rotated_image[np.where(thresh2 > 0)] = [255, 0, 0]
    thresh3 = cv2.bitwise_and(mask3, sum_B)
    tmip_rotated_image[np.where(thresh3 > 0)] = [0, 0, 255]
    end = cv2.getTickCount()
    use_time = (end - start) / cv2.getTickFrequency()
    print("use-time: %.4fs" % use_time)
    cv2.imwrite("result.jpg", tmip_rotated_image)
