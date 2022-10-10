import math
import multiprocessing
import threading
import time
from time import time
from unittest import result

import cv2
import numpy as np
import SimpleITK as sitk

from python_demos.constants import (COLOR_MAP, LEFT_BRAIN_LABEL,
                                    RIGHT_BRAIN_LABEL)
from python_demos.mismatch_utils import (bgr_to_rgb, get_cc, get_line,
                                         gray2rgb_array)

MISMATCH_THRESHOLD = [
    {"type": "rCBF", "compare": "lt", "threshold": 30, "color": "#FF5959"},
    {"type": "Tmax", "compare": "gt", "threshold": 6, "color": "#FFFF00"},
]
RCBF_THRESHOLD = [
    {"type": "rCBF", "compare": "lt", "threshold": 40, "color": "#26CF70"},
    {"type": "rCBF", "compare": "lt", "threshold": 30, "color": "#FFFF00"},
    {"type": "rCBF", "compare": "lt", "threshold": 20, "color": "#FF5959"},
]
RCBV_THRESHOLD = [
    {"type": "rCBV", "compare": "lt", "threshold": 45, "color": "#26CF70"},
    {"type": "rCBV", "compare": "lt", "threshold": 40, "color": "#FFFF00"},
    {"type": "rCBV", "compare": "lt", "threshold": 35, "color": "#FF5959"},
]
TMAX_THRESHOLD = [
    {"type": "Tmax", "compare": "gt", "threshold": 6, "color": "#26CF70"},
    {"type": "Tmax", "compare": "gt", "threshold": 8, "color": "#FFFF00"},
    {"type": "Tmax", "compare": "gt", "threshold": 10, "color": "#FF5959"},
]
global ROWS, COLS
ROWS, COLS = 512, 512


class A:
    def __init__(
        self,
        tmip_arr,
        cbf_array,
        cbv_array,
        mtt_array,
        tmax_array,
        ttp_array,
        brain_area_array,
        physical_info,
    ):
        self.tmip_arr = tmip_arr
        self.cbf_array = cbf_array
        self.cbv_array = cbv_array
        self.mtt_array = mtt_array
        self.tmax_array = tmax_array
        self.ttp_array = ttp_array
        self.brain_area_array = brain_area_array
        self.spacing = physical_info["spacing"]
        self.result = {}
        self.voxel = self.spacing[0] * self.spacing[1] * self.spacing[2] / 1000

    def get_rcbf_result(self, rgb_array, dcm_slice):
        (x1, y1), (x2, y2) = (282, 106), (240, 412)
        k, _ = get_line(x1, x2, y1, y2)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        angle = (math.atan(k)) / math.pi * 180

        cbf_gray_array = self.cbf_array[dcm_slice, :, :]
        # rgb_array = cv2.line(rgb_array, (x1, y1), (x2, y2), (255, 0, 0), 1)
        rotate_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle + 90, scale=1
        )
        cbf_rotated_image = cv2.warpAffine(
            src=cbf_gray_array, M=rotate_matrix, dsize=(ROWS, COLS)
        )
        tmip_rotated_image = cv2.warpAffine(
            src=rgb_array, M=rotate_matrix, dsize=(ROWS, COLS)
        )

        mirror_cbf = np.fliplr(cbf_rotated_image)
        mask1 = np.zeros((cbf_gray_array.shape), np.uint8)
        mask2 = np.zeros((cbf_gray_array.shape), np.uint8)
        mask3 = np.zeros((cbf_gray_array.shape), np.uint8)
        lesion_mask = np.zeros((cbf_gray_array.shape), np.uint8)
        for rcbf_threshold in RCBF_THRESHOLD:
            cbf_setting = rcbf_threshold["threshold"]
            # 小侧值 / 大侧值,TODO: 没有用到正常值这一参数

            temp_cbf = np.divide(
                cbf_rotated_image,
                mirror_cbf,
                out=np.zeros_like(cbf_rotated_image),
                where=mirror_cbf > 0,
            )
            error_arr = np.where((0 < temp_cbf) & (temp_cbf < cbf_setting / 100))
            if cbf_setting == 40:
                cbf_high = list(error_arr)
                mask1[error_arr] = 255
            if cbf_setting == 30:
                core_infarct_errors = error_arr
                mask2[error_arr] = 255
            if cbf_setting == 20:
                cbf_low = list(error_arr)
                mask3[error_arr] = 255
        setting_value = 40
        errorIdx1 = np.where((0 < temp_cbf) & (temp_cbf < setting_value / 100))
        setting_value = 30
        errorIdx2 = np.where((0 < temp_cbf) & (temp_cbf < setting_value / 100))
        setting_value = 20
        errorIdx3 = np.where((0 < temp_cbf) & (temp_cbf < setting_value / 100))

        mask1 = np.zeros((cbf_gray_array.shape), np.uint8)
        mask1[errorIdx1] = 255
        mask2 = np.zeros((cbf_gray_array.shape), np.uint8)
        mask2[errorIdx2] = 255
        mask3 = np.zeros((cbf_gray_array.shape), np.uint8)
        mask3[errorIdx3] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sum_B = get_cc(mask1, kernel)
        sum_B = cv2.morphologyEx(sum_B, cv2.MORPH_CLOSE, kernel, iterations=3)
        # 着色
        tmip_rotated_image[np.where(sum_B > 0)] = COLOR_MAP[RCBF_THRESHOLD[0]["color"]]
        thresh2 = cv2.bitwise_and(mask2, sum_B)
        thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel, iterations=3)
        tmip_rotated_image[np.where(thresh2 > 0)] = COLOR_MAP[
            RCBF_THRESHOLD[1]["color"]
        ]
        thresh3 = cv2.bitwise_and(mask3, sum_B)
        thresh3 = cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, kernel, iterations=3)
        tmip_rotated_image[np.where(thresh3 > 0)] = COLOR_MAP[
            RCBF_THRESHOLD[2]["color"]
        ]
        rotate_matrix = cv2.getRotationMatrix2D(
            center=center, angle=-angle - 90, scale=1
        )
        origin_image = cv2.warpAffine(
            src=tmip_rotated_image, M=rotate_matrix, dsize=(ROWS, COLS)
        )
        origin_image = bgr_to_rgb(origin_image)
        self.result["rCBF_Mismatch_" + str(dcm_slice)] = origin_image
        # cv2.imwrite(f"result{str(dcm_slice)}.jpg", origin_image)

    def get_rcbv_result(
        self,
        rgb_array,
        dcm_slice,
    ):
        (x1, y1), (x2, y2) = (282, 106), (240, 412)
        k, _ = get_line(x1, x2, y1, y2)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        angle = (math.atan(k)) / math.pi * 180
        cbv_gray_array = self.cbv_array[dcm_slice, :, :]
        # tmip 沿旋转矩阵旋转angle
        rotate_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle + 90, scale=1
        )
        tmip_rotated_image = cv2.warpAffine(
            src=rgb_array, M=rotate_matrix, dsize=(ROWS, COLS)
        )

        # cbv 沿旋转矩阵旋转angle
        rotate_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle + 90, scale=1
        )
        rotated_image = cv2.warpAffine(
            src=cbv_gray_array, M=rotate_matrix, dsize=(ROWS, COLS)
        )
        # 镜像翻转
        mirror = np.fliplr(rotated_image)
        # 每个阈值一个mask, 取连通分量
        mask1 = np.zeros((cbv_gray_array.shape), np.uint8)
        mask2 = np.zeros((cbv_gray_array.shape), np.uint8)
        mask3 = np.zeros((cbv_gray_array.shape), np.uint8)
        lesion_mask = np.zeros((cbv_gray_array.shape), np.uint8)
        for rcbv_threshold in RCBV_THRESHOLD:
            cbv_setting = rcbv_threshold["threshold"]
            # 小侧 / 大侧, TODO: 没有用到正常值这一参数
            temp_cbv = np.divide(
                rotated_image,
                mirror,
                out=np.zeros_like(rotated_image),
                where=mirror > 0,
            )
            error_arr = np.where((0 < temp_cbv) & (temp_cbv < cbv_setting / 100))
            if cbv_setting == 45:
                mask1[error_arr] = 255
                cbv_high = list(error_arr)
            if cbv_setting == 40:
                cbv_mid = list(error_arr)
                mask2[error_arr] = 255
            if cbv_setting == 35:
                cbv_low = list(error_arr)
                mask3[error_arr] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        sum_B = get_cc(mask1, kernel)
        sum_B = cv2.morphologyEx(sum_B, cv2.MORPH_CLOSE, kernel, iterations=8)
        tmip_rotated_image[np.where(sum_B > 0)] = COLOR_MAP[RCBV_THRESHOLD[0]["color"]]

        thresh2 = cv2.bitwise_and(mask2, sum_B)
        thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel, iterations=8)
        tmip_rotated_image[np.where(thresh2 > 0)] = COLOR_MAP[
            RCBV_THRESHOLD[1]["color"]
        ]

        thresh3 = cv2.bitwise_and(mask3, sum_B)
        thresh3 = cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, kernel, iterations=8)
        tmip_rotated_image[np.where(thresh3 > 0)] = COLOR_MAP[
            RCBV_THRESHOLD[2]["color"]
        ]

        # 沿旋转矩阵反向旋转 angle
        rotate_matrix = cv2.getRotationMatrix2D(
            center=center, angle=-angle - 90, scale=1
        )
        origin_image = cv2.warpAffine(
            src=tmip_rotated_image, M=rotate_matrix, dsize=(ROWS, COLS)
        )
        origin_image = bgr_to_rgb(origin_image)
        self.result["rCBV_Mismatch_" + str(dcm_slice)] = origin_image
        return self.result, lesion_mask.tolist(), cbv_high, cbv_mid, cbv_low

    def get_tmax_result(self, rgb_array, dcm_slice):
        tmax_gray_array = self.tmax_array[dcm_slice, :, :]
        low_perfusion_errors = []
        mask1 = np.zeros((tmax_gray_array.shape), np.uint8)
        mask2 = np.zeros((tmax_gray_array.shape), np.uint8)
        mask3 = np.zeros((tmax_gray_array.shape), np.uint8)
        lesion_mask = np.zeros((tmax_gray_array.shape), np.uint8)

        for tmax_threshold in TMAX_THRESHOLD:
            tmax_setting = tmax_threshold["threshold"]
            error_arr = np.where((tmax_gray_array > tmax_setting))

            # Tmax 大于6s为低灌注区域
            if tmax_setting == 6:
                mask1[error_arr] = 255
                low_perfusion_errors = np.where((tmax_gray_array > 6))
            if tmax_setting == 8:
                tmax_low = list(error_arr)
                mask2[error_arr] = 255
            if tmax_setting == 10:
                tmax_high = list(error_arr)
                mask3[error_arr] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sum_B = get_cc(mask1, kernel)
        sum_B = cv2.morphologyEx(sum_B, cv2.MORPH_CLOSE, kernel, iterations=8)
        lesion_mask[np.where(sum_B > 0)] = 1
        rgb_array[np.where(sum_B > 0)] = COLOR_MAP[TMAX_THRESHOLD[0]["color"]]

        thresh2 = cv2.bitwise_and(mask2, sum_B)
        thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel, iterations=8)
        lesion_mask[np.where(thresh2 > 0)] = 1
        rgb_array[np.where(thresh2 > 0)] = COLOR_MAP[TMAX_THRESHOLD[1]["color"]]

        thresh3 = cv2.bitwise_and(mask3, sum_B)
        thresh3 = cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, kernel, iterations=8)
        lesion_mask[np.where(thresh3 > 0)] = 1
        rgb_array[np.where(thresh3 > 0)] = COLOR_MAP[TMAX_THRESHOLD[2]["color"]]
        # bgr_to_rgb
        rgb_array = bgr_to_rgb(rgb_array)
        self.result["TmaxMismatch_" + str(dcm_slice)] = rgb_array
        return (
            low_perfusion_errors,
            lesion_mask.tolist(),
            tmax_high,
            tmax_low,
        )

    def get_mismatch_result(
        self,
        rgb_array,
        dcm_slice,
    ):
        """mismatch 视图: rCBF<0.3 红色，tmax>6 黄色"""
        cbf_gray_array = self.cbf_array[dcm_slice, :, :]
        tmax_gray_array = self.tmax_array[dcm_slice, :, :]

        (x1, y1), (x2, y2) = (282, 106), (240, 412)
        k, _ = get_line(x1, x2, y1, y2)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        angle = (math.atan(k)) / math.pi * 180
        # tmip 沿旋转矩阵旋转angle
        rotate_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle + 90, scale=1
        )
        cbf_rotated_image = cv2.warpAffine(
            src=cbf_gray_array, M=rotate_matrix, dsize=(ROWS, COLS)
        )
        tmip_rotated_image = cv2.warpAffine(
            src=rgb_array, M=rotate_matrix, dsize=(ROWS, COLS)
        )
        # 二值化mask
        mask1 = np.zeros((cbf_gray_array.shape), np.uint8)
        mask2 = np.zeros((cbf_gray_array.shape), np.uint8)
        # 低灌注和梗死区 0,1 mask
        single_low_mask = np.zeros((cbf_gray_array.shape), np.uint8)
        single_core_mask = np.zeros((cbf_gray_array.shape), np.uint8)
        # 病灶 0，1 mask
        lesion_mask = np.zeros((cbf_gray_array.shape), np.uint8)
        mirror_cbf = np.fliplr(cbf_rotated_image)
        temp_cbf = np.divide(
            cbf_rotated_image,
            mirror_cbf,
            out=np.zeros_like(cbf_rotated_image),
            where=mirror_cbf > 0,
        )
        # TODO 这里不太对，不能直接使用未翻转的索引
        low_perfusion_errors = np.where(tmax_gray_array > 6)
        core_infarct_errors = np.where((0 < temp_cbf) & (temp_cbf < 0.3))
        mask1[low_perfusion_errors] = 255
        mask2[core_infarct_errors] = 255

        single_low_mask[low_perfusion_errors] = 1
        single_core_mask[core_infarct_errors] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sum_B = get_cc(mask1, kernel)

        sum_B = cv2.morphologyEx(sum_B, cv2.MORPH_CLOSE, kernel, iterations=8)
        lesion_mask[np.where(sum_B > 0)] = 1
        # mask2 腐蚀膨胀
        mask2 = cv2.erode(cv2.dilate(mask2, kernel, iterations=1), kernel, iterations=5)
        mask2 = cv2.dilate(mask2, kernel, iterations=3)
        # 取交集
        thresh2 = cv2.bitwise_and(mask2, sum_B)
        thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel, iterations=8)
        lesion_mask[np.where(thresh2 > 0)] = 1
        tmip_rotated_image[np.where(thresh2 > 0)] = COLOR_MAP[
            MISMATCH_THRESHOLD[0]["color"]
        ]
        # 沿旋转矩阵反向旋转 angle
        rotate_matrix = cv2.getRotationMatrix2D(
            center=center, angle=-angle - 90, scale=1
        )
        origin_image = cv2.warpAffine(
            src=tmip_rotated_image, M=rotate_matrix, dsize=(ROWS, COLS)
        )
        origin_image[np.where(sum_B > 0)] = COLOR_MAP[MISMATCH_THRESHOLD[1]["color"]]
        origin_image = bgr_to_rgb(origin_image)
        self.result["Mismatch_" + str(dcm_slice)] = origin_image
        return lesion_mask, single_low_mask, single_core_mask

    def run(self):
        start_time = time.time()
        l = []
        for i in range(self.cbf_array.shape[0]):
            tmip_img = self.tmip_arr[i, :, :]
            rgb_tmip_arr = gray2rgb_array(tmip_img, 100, 50)
            rgb_array = rgb_tmip_arr.copy()
            p1 = threading.Thread(target=self.get_rcbf_result, args=(rgb_array, i))
            rgb_array = rgb_tmip_arr.copy()
            p2 = threading.Thread(target=self.get_rcbv_result, args=(rgb_array, i))
            rgb_array = rgb_tmip_arr.copy()
            p3 = threading.Thread(target=self.get_tmax_result, args=(rgb_array, i))
            rgb_array = rgb_tmip_arr.copy()
            p4 = threading.Thread(target=self.get_mismatch_result, args=(rgb_array, i))
            l.append(p1)
            l.append(p2)
            l.append(p3)
            l.append(p4)
            # l.append(p4)
            p1.start()
            p2.start()
            p3.start()
            p4.start()
        for p in l:
            # 等待所有代码结束
            p.join()
        end = time.time()
        print("use-time: %.4fs" % (end - start_time))
        return self.result

    def get_lesions(self):
        pass


if __name__ == "__main__":
    cbf_img = sitk.ReadImage("/media/tx-deepocean/Data/DICOMS/demos/28/CBF.nii.gz")
    cbf_arr = sitk.GetArrayFromImage(cbf_img)

    tmip_img = sitk.ReadImage(
        "/media/tx-deepocean/Data/DICOMS/demos/28/TMIP_NO_SKULL.nii.gz"
    )
    tmip_arr = sitk.GetArrayFromImage(tmip_img)

    cbv_img = sitk.ReadImage("/media/tx-deepocean/Data/DICOMS/demos/28/CBV.nii.gz")
    cbv_arr = sitk.GetArrayFromImage(cbv_img)

    mtt_img = sitk.ReadImage("/media/tx-deepocean/Data/DICOMS/demos/28/MTT.nii.gz")
    mtt_arr = sitk.GetArrayFromImage(mtt_img)

    tmax_img = sitk.ReadImage("/media/tx-deepocean/Data/DICOMS/demos/28/TMAX.nii.gz")
    tmax_arr = sitk.GetArrayFromImage(tmax_img)

    ttp_img = sitk.ReadImage("/media/tx-deepocean/Data/DICOMS/demos/28/TTP.nii.gz")
    ttp_arr = sitk.GetArrayFromImage(ttp_img)
    physical_info = {"spacing": [0.5, 0.5, 5], "orgin": tmip_img.GetOrigin()}
    brain_area_img = sitk.ReadImage(
        "/media/tx-deepocean/Data/DICOMS/demos/28/brain_area_mask.nii.gz"
    )
    brain_area_array = sitk.GetArrayFromImage(brain_area_img)
    mis = A(
        tmip_arr,
        cbf_arr,
        cbv_arr,
        mtt_arr,
        tmax_arr,
        ttp_arr,
        brain_area_array,
        physical_info,
    )
    img_result = mis.run()
    print(img_result.keys())
    mis.get_lesions()
