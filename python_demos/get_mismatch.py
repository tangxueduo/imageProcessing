import io
import json
import math
import os
import time
from collections import defaultdict
from threading import Thread

import cc3d
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import SimpleITK as sitk
from constants import BRAIN_AREA_MAP, LEFT_BRAIN_LABEL, RIGHT_BRAIN_LABEL
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from skimage import measure

from python_demos.mismatch_utils import (
    convert_ijk_to_xyz,
    convert_noraml_to_standard_plane,
    ddict2dict,
    get_cc,
    get_line,
    get_point_on_plane_by_yz,
    gray2rgb_array,
    np_array_to_dcm,
    remove_small_volume,
    trans_list_to_array,
)

mask_path = "/media/tx-deepocean/Data/DICOMS/demos/37"
dcm_file = "/media/tx-deepocean/Data/DICOMS/demos/TMAX/TMAX023.dcm"
ds = pydicom.read_file(dcm_file, force=True)
rows, cols = 512, 512

mask_info_map = {
    "CBV": {
        "min_pixel": 0,
        "max_pixel": 20,
        "ww": 20,
        "wl": 10,
        "rgb_list": [
            [0, 0, 0],
            [0, 85, 255],
            [46, 232, 220],
            [41, 250, 90],
            [163, 255, 25],
            [0, 0, 0],
        ],
        "thresholds": [0, 31.2, 65.6, 83.4, 91.0, 100],
    },
    "CBF": {
        "min_pixel": 0,
        "max_pixel": 200,
        "ww": 200,
        "wl": 100,
        "rgb_list": [
            [0, 0, 0],
            [0, 85, 255],
            [126, 255, 255],
            [
                124,
                255,
                88,
            ],
            [234, 236, 103],
            [255, 206, 28],
            [0, 0, 0],
        ],
        "thresholds": [0, 15, 32.3, 57.1, 77.5, 92, 100],
    },
    "TTP": {
        "min_pixel": 0,
        "max_pixel": 25,
        "ww": 25,
        "wl": 12.5,
        "rgb_list": [
            [0, 0, 0],
            [217, 74, 8],
            [255, 197, 21],
            [255, 255, 0],
            [95, 250, 78],
            [28, 230, 99],
            [0, 0, 255],
            [0, 0, 0],
        ],
        "thresholds": [0, 7.5, 17.9, 31.6, 48.7, 62.3, 83.9, 100],
    },
    "MTT": {
        "min_pixel": 0,
        "max_pixel": 20,
        "ww": 20,
        "wl": 10,
        "rgb_list": [
            [0, 0, 0],
            [225, 87, 7],
            [255, 255, 0],
            [161, 251, 86],
            [60, 248, 13],
            [32, 43, 250],
            [0, 0, 0],
        ],
        "thresholds": [0, 12.8, 22.2, 42.9, 68.3, 90.2, 100],
    },
    "TMAX": {
        "min_pixel": 0.15,
        "max_pixel": 57.54,
        "ww": 28.85,
        "wl": 57.39,
        "rgb_list": [
            [0, 0, 0],
            [0, 85, 255],
            [46, 232, 220],
            [41, 250, 90],
            [163, 255, 25],
            [0, 0, 0],
        ],
        "thresholds": [0, 23.2, 47.1, 75.5, 93.2, 100],
    },
    "tAve_NO_SKULL": {"ww": 100, "wl": 40},
    "tAve_WITH_SKULL": {"ww": 100, "wl": 40},
    "TMIP_NO_SKULL": {"ww": 100, "wl": 50},
    "TMIP_WITH_SKULL": {"ww": 100, "wl": 50},
}

# bgr 值
Mismatch_conf = [
    {"type": "rCBF", "compare": "lt", "threshold": 30, "color": "#FF5959"},
    {"type": "Tmax", "compare": "gt", "threshold": 6, "color": "#FFFF00"},
]
rCBF = [
    {"type": "rCBF", "compare": "lt", "threshold": 40, "color": "#26CF70"},
    {"type": "rCBF", "compare": "lt", "threshold": 30, "color": "#FFFF00"},
    {"type": "rCBF", "compare": "lt", "threshold": 20, "color": "#FF5959"},
]
rCBV = [
    {"type": "rCBV", "compare": "lt", "threshold": 45, "color": "#26CF70"},
    {"type": "rCBV", "compare": "lt", "threshold": 40, "color": "#FFFF00"},
    {"type": "rCBV", "compare": "lt", "threshold": 35, "color": "#FF5959"},
]
Tmax = [
    {"type": "Tmax", "compare": "gt", "threshold": 6, "color": "#26CF70"},
    {"type": "Tmax", "compare": "gt", "threshold": 8, "color": "#FFFF00"},
    {"type": "Tmax", "compare": "gt", "threshold": 10, "color": "#FF5959"},
]
color_map = {
    "#FFFF00": [0, 255, 255],
    "#FF5959": [89, 89, 255],
    "#26CF70": [112, 207, 38],
    "#3F87F5": [245, 135, 63],
}


class Mismatch:
    def __init__(
        self,
        cbf_array,
        cbv_array,
        mtt_array,
        tmax_array,
        ttp_array,
        brain_area_array,
        spacing,
        depth,
    ):
        self.cbf_array = cbf_array
        self.cbv_array = cbv_array
        self.mtt_array = mtt_array
        self.tmax_array = tmax_array
        self.ttp_array = ttp_array
        self.brain_area_array = brain_area_array
        self.spacing = spacing
        self.voxel = self.spacing[0] * self.spacing[1] * self.spacing[2] / 1000
        self.depth = depth

    def get_mismatch(
        self,
        tmip_array: np.ndarray,
        config: list,
        centerline: dict,
        origin,
        direction,
    ):
        """
        rCBF计算公式:
            先和大脑对侧比,保证两侧 小值/大值<0.3 作为异常点, 进而判定异常点所在侧, 着色; 若镜像侧值为背景, 与正常值50比;
            若两侧均有异常， 且相差小于10%, 认定两点均不是异常， 否则取较大值为异常值， 着色;
        """
        slice_view_info = {}

        result = {}
        dd = lambda: defaultdict(dd)
        ctp_lesion = dd()

        (
            rcbv_lesion_mask,
            rcbf_lesion_mask,
            low_perfusion_mask,
            core_infarct_mask,
            mismatch_lesion_mask,
            tmax_lesion_mask,
            left_stem_mask,
            right_stem_mask,
        ) = ([], [], [], [], [], [], [], [])

        cbf_normal = 50  # cbf 正常值
        cbv_normal = 4  # 4ml 正常值

        # 1、区分左右侧大脑:点在直线左、右侧
        # 点法式 转 标准式
        (
            coefficient_a,
            coefficient_b,
            coefficient_c,
            coefficient_d,
        ) = convert_noraml_to_standard_plane(centerline["point"], centerline["vector"])
        rcbf_result, tmax_result, rcbv_result, mismatch_result = {}, {}, {}, {}
        index, low_perfusion_area, core_infarct_area = 0, 0, 1
        (
            cbf_sum_high,
            cbf_sum_low,
            cbv_sum_high,
            cbv_sum_mid,
            cbv_sum_low,
            tmax_high,
            tmax_low,
        ) = (0, 0, 0, 0, 0, 0, 0)
        for dcm_slice in range(self.cbf_array.shape[0]):
            dcm_slice = 15
            rows, cols = self.cbf_array.shape[1:]
            # 视图选取去骨 tmip
            tmip_gray_array = tmip_array[dcm_slice, :, :]
            # gray to rgb
            tmip_rgb_array = gray2rgb_array(tmip_gray_array, 100, 50)
            [i, j, loc] = origin

            loc = loc + (self.depth - 1 - dcm_slice) * self.spacing[2]
            # # 由两点式确定直线方程 ，由于原点为左上，取Ax+By+C=0关于x轴对称的直线方程:Ax-By+C=0
            p1, p2 = get_point_on_plane_by_yz(
                coefficient_a,
                coefficient_b,
                coefficient_c,
                coefficient_d,
                rows,
                loc,
                self.spacing[1],
                origin[1],
            )
            # 转像素坐标
            new_p1, new_p2 = [p1[0], p1[1], loc], [p2[0], p2[1], loc]
            xyzp1 = convert_ijk_to_xyz(new_p1, origin, self.spacing)
            xyzp2 = convert_ijk_to_xyz(new_p2, origin, self.spacing)
            (x1, y1), (x2, y2) = (xyzp1[0], xyzp1[1]), (xyzp2[0], xyzp2[1])
            # (x1, y1), (x2, y2) = (563.3137852888025, 389.5496597290039), (264.29764387483607, 460.7999979853627)
            # 注意采用x=ky+b模型，原因是如果中线没有偏转，例x=10这种情况，x=ky+b不用考虑这种特殊情况，不用额外写个if条件判断
            k, b = get_line(x1, x2, y1, y2)
            # 获取某层
            cbf_gray_array = self.cbf_array[dcm_slice, :, :]
            cbv_gray_array = self.cbv_array[dcm_slice, :, :]
            tmax_gray_array = self.tmax_array[dcm_slice, :, :]

            # # 计算倾斜角
            angle = (math.atan2(y1 - y2, x1 - x2)) / math.pi * 180
            # # 计算中线中点
            center = (512 / 2, 512 / 2)
            print(f"****angle: {angle}， center： {center}")

            # 所有层的灌注和梗死像素
            (
                rcbf_result,
                core_infarct_errors,
                single_core_mask,
                cbf_lesion_mask,
                cbf_high,
                cbf_low,
            ) = self.get_rcbf_result(
                tmip_rgb_array,
                cbf_gray_array,
                center,
                angle,
                cbf_normal,
                index,
                rcbf_result,
            )
            slice_view_info[dcm_slice] = {}
            slice_view_info[dcm_slice].update(
                {"core_infarct": len(core_infarct_errors[0])}
            )
            rcbf_lesion_mask.append(cbf_lesion_mask)
            cbf_high = len(cbf_high[0]) if cbf_high else 0
            cbf_low = len(cbf_low[0]) if cbf_low else 0
            slice_view_info[dcm_slice].update(
                {"rcbf<0.4": cbf_high, "rcbf<0.2": cbf_low}
            )
            # 计算 rcbf 视图的 summary

            cbf_sum_high += cbf_high
            cbf_sum_low += cbf_low
            # 获取 rcbv 视图, 小于0.45绿色， 小于0.4黄色， 小于0.35红色
            (
                rcbv_result,
                cbv_lesion_mask,
                cbv_high,
                cbv_mid,
                cbv_low,
            ) = self.get_rcbv_result(
                tmip_rgb_array,
                cbv_gray_array,
                center,
                angle,
                cbv_normal,
                index,
                rcbv_result,
                rows,
                cols,
            )
            rcbv_lesion_mask.append(cbv_lesion_mask)
            # 计算 rcbv 视图的 summary
            cbv_high = len(cbv_high[0]) if cbv_high else 0
            cbv_mid = len(cbv_mid[0]) if cbv_mid else 0
            cbv_low = len(cbv_low[0]) if cbv_low else 0
            slice_view_info[dcm_slice].update(
                {"rcbv<0.45": cbv_high, "rcbv<0.4": cbv_mid, "rcbv<0.35": cbv_low}
            )
            cbv_sum_high += cbv_high
            cbv_sum_mid += cbv_mid
            cbv_sum_low += cbv_low
            # 获取 tmax 视图
            (
                tmax_result,
                low_perfusion_errors,
                tmax_lesion_mask,
                tmax_high,
                tmax_low,
            ) = self.get_tmax_result(
                tmip_rgb_array, tmax_gray_array, tmax_result, index
            )
            slice_view_info[dcm_slice].update(
                {"low_perfusion": len(low_perfusion_errors[0])}
            )
            tmax_lesion_mask.append(tmax_lesion_mask)
            # 计算 tmax 视图的 summary
            tmax_high = len(tmax_high[0] if tmax_high else 0)
            tmax_low = len(tmax_low[0] if tmax_low else 0)

            slice_view_info[dcm_slice].update(
                {"tmax>10": tmax_high, "tmax>8": tmax_low}
            )
            brain_gray_array = self.brain_area_array[dcm_slice, :, :]
            # 获取 脑干 mask
            left_stem_mask, right_stem_mask = self._get_stem_slice(
                brain_gray_array, k, b, left_stem_mask, right_stem_mask
            )

            # 获取 mismatch 视图 ,这里取巧， 利用rcbv和tmax计算视图计算好的像素 list
            (
                mismatch_result,
                mis_lesion_mask,
                single_low_mask,
            ) = self.get_mismatch_result(
                tmip_rgb_array,
                cbf_gray_array,
                core_infarct_errors,
                single_core_mask,
                low_perfusion_errors,
                center,
                angle,
                index,
                mismatch_result,
                rows,
                cols,
            )
            # 分别获取低灌注 和 梗死区 3Dmask
            low_perfusion_mask.append(single_low_mask)
            core_infarct_mask.append(single_core_mask)
            mismatch_lesion_mask.append(mis_lesion_mask)
            #
            index += 1
            # low_perfusion_area += len(low_perfusion_errors)
            # core_infarct_area += len(core_infarct_errors)

        print(json.dumps(slice_view_info))
        ctp_lesion = self.get_lesions(
            mismatch_lesion_mask,
            low_perfusion_mask,
            core_infarct_mask,
            slice_view_info,
            ctp_lesion,
            left_stem_mask,
            right_stem_mask,
        )
        ctp_lesion = ddict2dict(ctp_lesion)
        return result, ctp_lesion

    def _get_stem_slice(
        self,
        brain_gray_array,
        k: float,
        b: float,
        left_stem_mask: list,
        right_stem_mask: list,
    ):
        """获取脑干 0 ,1 mask"""
        l_stem_single_mask = np.zeros((brain_gray_array.shape), np.uint8)
        r_stem_single_mask = np.zeros((brain_gray_array.shape), np.uint8)

        # 分别获取脑中线 上方 下方脑干区域坐标
        stem_arr = np.argwhere(brain_gray_array == 15)
        # 在异常点位置数组中判断哪些点的索引在中线下方
        idx_bottom = np.argwhere((stem_arr[:, 0] - k * stem_arr[:, 1] - b) > 0)
        # 在异常点位置数组中判断哪些点的索引在中线上方
        idx_top = np.argwhere((stem_arr[:, 0] - k * stem_arr[:, 1] - b) <= 0)
        # 因 argwhere输出二维数组，第二列全是0,只提取第一列索引
        er_idx = idx_top[:, 0]  # 右侧
        el_idx = idx_bottom[:, 0]  # 左侧
        l_stem_single_mask[stem_arr[el_idx][:, 0], stem_arr[el_idx][:, 1]] = 1
        r_stem_single_mask[stem_arr[er_idx][:, 0], stem_arr[er_idx][:, 1]] = 1
        left_stem_mask.append(l_stem_single_mask)
        right_stem_mask.append(r_stem_single_mask)
        return left_stem_mask, right_stem_mask

    def get_lesions(
        self,
        mismatch_lesion_mask,
        low_perfusion_mask,
        core_infarct_mask,
        slice_view_info,
        ctp_lesion,
        left_stem_mask,
        right_stem_mask,
    ):
        mismatch_lesion_mask = trans_list_to_array(
            mismatch_lesion_mask, self.cbf_array.shape
        )
        low_perfusion_mask = trans_list_to_array(
            low_perfusion_mask, self.cbf_array.shape
        )
        core_infarct_mask = trans_list_to_array(core_infarct_mask, self.cbf_array.shape)
        left_stem_mask = trans_list_to_array(left_stem_mask, self.cbf_array.shape)
        right_stem_mask = trans_list_to_array(right_stem_mask, self.cbf_array.shape)

        lesions, lesion_brains = self.get_data_result(
            mismatch_lesion_mask,
            slice_view_info,
        )
        report = self.get_brain_area_paras(
            list(lesion_brains),
            low_perfusion_mask,
            core_infarct_mask,
            left_stem_mask,
            right_stem_mask,
        )
        ctp_lesion["lesions"].update(lesions)
        ctp_lesion["lesions"]["CT_perfusion"]["report"] = report
        return ctp_lesion

    def get_data_result(
        self,
        mismatch_lesion_mask,
        slice_view_info,
    ):
        """获取3d连通域, 计算病灶"""
        # 获取病灶数量， 病灶标签， 病灶所在层
        cc, cc_unique, slice_ids = remove_small_volume(mismatch_lesion_mask)

        t1 = time.time()
        tmp = lambda: defaultdict(tmp)
        lesions = tmp()
        report, lesion_brains = {}, []
        (
            rcbf_sum_high,
            rcbf_sum_mid,
            rcbf_sum_low,
            rcbv_sum_high,
            rcbv_sum_mid,
            rcbv_sum_low,
            tmax_sum_high,
            tmax_sum_mid,
            tmax_sum_low,
            mis_sum_core,
            mis_sum_low,
        ) = [0 for _ in range(11)]

        # 若无符合条件的连通域, return 空
        if not cc_unique:
            return {}, report
        rcbf_lesions, rcbv_lesions, mis_lesions, tmax_lesions = [], [], [], []
        for lesion in cc_unique:
            brain_areas = []
            (
                mis_single_lesion,
                rcbf_single_lesion,
                rcbv_single_lesion,
                tmax_single_lesion,
            ) = ({}, {}, {}, {})
            brain_list = list(np.unique(self.brain_area_array[cc == lesion]))
            try:
                brain_list.remove(0)
            except Exception as e:
                print(f"0 not in list: {e}")
            # 病灶所经过的脑区
            lesion_brains += brain_list
            for area_label in brain_list:
                brain_areas.append(BRAIN_AREA_MAP[area_label]["origin"])
                # TODO: 遍历层求 mismatch 低灌注体积、梗死区体积
                mis_single_lesion["section"] = brain_areas
                mis_single_lesion["slice_id"] = slice_ids[lesion]
                mis_single_lesion["info"] = {}
                res1 = (
                    res2
                ) = (
                    res_cbf_high
                ) = (
                    res_cbf_low
                ) = (
                    res_cbv_high
                ) = res_cbv_low = res_cbv_mid = res_tmax_high = res_tmax_mid = 0
                for dcm_slice in range(slice_ids[lesion][0], slice_ids[lesion][1] + 1):
                    res1 += slice_view_info[dcm_slice]["low_perfusion"]
                    res2 += slice_view_info[dcm_slice]["core_infarct"]
                    res_cbf_high += slice_view_info[dcm_slice]["rcbf<0.4"]
                    res_cbf_low += slice_view_info[dcm_slice]["rcbf<0.2"]

                    res_cbv_high += slice_view_info[dcm_slice]["rcbv<0.45"]
                    res_cbv_mid += slice_view_info[dcm_slice]["rcbv<0.4"]
                    res_cbv_low += slice_view_info[dcm_slice]["rcbv<0.35"]

                    res_tmax_high += slice_view_info[dcm_slice]["tmax>10"]
                    res_tmax_mid += slice_view_info[dcm_slice]["tmax>8"]
            # Mismatch
            mis_single_lesion["info"]["Tmax > 6s"] = round(res1 * self.voxel, 1)
            mis_sum_low += round(res1 * self.voxel, 1)
            mis_single_lesion["info"]["rCBF < 30%"] = round(res2 * self.voxel, 1)
            mis_sum_core += round(res2 * self.voxel, 1)
            mis_single_lesion["info"]["MismatchVolume"] = round(
                res1 * self.voxel - res2 * self.voxel, 1
            )
            res2 = 1 if res2 == 0 else res2
            mis_single_lesion["info"]["rate"] = round(res1 / res2, 1)
            # rCBF
            rcbf_single_lesion["section"] = brain_areas
            rcbf_single_lesion["slice_id"] = slice_ids[lesion]
            rcbf_single_lesion["info"] = {}
            rcbf_single_lesion["info"]["rCBF < 40%"] = round(
                res_cbf_high * self.voxel, 1
            )
            rcbf_sum_high += round(res_cbf_high * self.voxel, 1)
            rcbf_single_lesion["info"]["rCBF < 30%"] = round(res2 * self.voxel, 1)
            rcbf_sum_mid += round(res2 * self.voxel, 1)
            rcbf_single_lesion["info"]["rCBF < 20%"] = round(
                res_cbf_low * self.voxel, 1
            )
            rcbf_sum_low += round(res_cbf_low * self.voxel, 1)
            # rCBV
            rcbv_single_lesion["section"] = brain_areas
            rcbv_single_lesion["slice_id"] = slice_ids[lesion]
            rcbv_single_lesion["info"] = {}
            rcbv_single_lesion["info"]["rCBV < 45%"] = round(
                res_cbv_high * self.voxel, 1
            )
            rcbv_sum_high += round(res_cbv_high * self.voxel, 1)
            rcbv_single_lesion["info"]["rCBV < 40%"] = round(
                res_cbv_mid * self.voxel, 1
            )
            rcbv_sum_mid += round(res_cbv_mid * self.voxel, 1)
            rcbv_single_lesion["info"]["rCBV < 35%"] = round(
                res_cbv_low * self.voxel, 1
            )
            rcbv_sum_low += round(res_cbv_low * self.voxel, 1)
            # Tmax
            tmax_single_lesion["section"] = brain_areas
            tmax_single_lesion["slice_id"] = slice_ids[lesion]
            tmax_single_lesion["info"] = {}
            tmax_single_lesion["info"]["Tmax > 10s"] = round(
                res_tmax_high * self.voxel, 1
            )
            tmax_sum_high += round(res_tmax_high * self.voxel, 1)
            tmax_single_lesion["info"]["Tmax > 8s"] = round(
                res_tmax_mid * self.voxel, 1
            )
            tmax_sum_mid += round(res_tmax_mid * self.voxel, 1)
            tmax_single_lesion["info"]["Tmax > 6s"] = round(res1 * self.voxel, 1)
            tmax_sum_low += round(res1 * self.voxel, 1)
            rcbf_lesions.append(rcbf_single_lesion)
            rcbv_lesions.append(rcbv_single_lesion)
            tmax_lesions.append(tmax_single_lesion)
            mis_lesions.append(mis_single_lesion)

        lesions["Mismatch"]["rCBF_view"]["lesions"] = rcbf_lesions
        # rcbf summary
        lesions["Mismatch"]["rCBF_view"]["Mismatch_abnormality"][
            "rCBF < 40%"
        ] = rcbf_sum_high
        lesions["Mismatch"]["rCBF_view"]["Mismatch_abnormality"][
            "rCBF < 30%"
        ] = rcbf_sum_mid
        lesions["Mismatch"]["rCBF_view"]["Mismatch_abnormality"][
            "rCBF < 20%"
        ] = rcbf_sum_low

        lesions["Mismatch"]["rCBV_view"]["lesions"] = rcbv_lesions
        # rcbv summary
        lesions["Mismatch"]["rCBV_view"]["Mismatch_abnormality"][
            "rCBV < 45%"
        ] = rcbv_sum_high
        lesions["Mismatch"]["rCBV_view"]["Mismatch_abnormality"][
            "rCBV < 40%"
        ] = rcbv_sum_mid
        lesions["Mismatch"]["rCBV_view"]["Mismatch_abnormality"][
            "rCBV < 35%"
        ] = rcbv_sum_low

        lesions["Mismatch"]["Tmax_view"]["lesions"] = tmax_lesions
        # tmax summary
        lesions["Mismatch"]["Tmax_view"]["Mismatch_abnormality"][
            "Tmax > 10s%"
        ] = tmax_sum_high
        lesions["Mismatch"]["Tmax_view"]["Mismatch_abnormality"][
            "Tmax > 8s%"
        ] = tmax_sum_mid
        lesions["Mismatch"]["Tmax_view"]["Mismatch_abnormality"][
            "Tmax > 6s%"
        ] = tmax_sum_low

        lesions["Mismatch"]["Mismatch_view"]["lesions"] = mis_lesions
        # mismatch summary
        lesions["Mismatch"]["Mismatch_view"]["Mismatch_abnormality"][
            "rCBF < 30%"
        ] = mis_sum_core
        lesions["Mismatch"]["Mismatch_view"]["Mismatch_abnormality"][
            "Tmax > 6s"
        ] = mis_sum_low
        lesions["Mismatch"]["Mismatch_view"]["Mismatch_abnormality"][
            "MismatchVolume"
        ] = (mis_sum_low - mis_sum_core)
        mis_sum_core = 1 if mis_sum_core == 0 else mis_sum_core
        lesions["Mismatch"]["Mismatch_view"]["Mismatch_abnormality"]["rate"] = round(
            mis_sum_low / mis_sum_core, 1
        )
        lesion_brains = set(lesion_brains)
        return lesions, lesion_brains

    def get_brain_area_paras(
        self,
        lesion_brains: list,
        low_perfusion_mask: np.ndarray,
        core_infarct_mask: np.ndarray,
        left_stem_mask: np.ndarray,
        right_stem_mask: np.ndarray,
    ):
        report = {}
        for brain_label in lesion_brains:
            # 左右脑区都有灌注，计算左脑区，对侧取反
            if brain_label == 15 or brain_label == 16:
                label_origin = brain_label
                origin_area = BRAIN_AREA_MAP[label_origin]["origin"]
                relative_area = 9999
            else:
                # label_origin: 1, 12,...
                label_origin = brain_label
                label_relative = BRAIN_AREA_MAP[label_origin]["relative"]
                # origin_area: L-Parietal lobe...
                origin_area = BRAIN_AREA_MAP[label_origin]["origin"]  # 脑区
                relative_area = BRAIN_AREA_MAP[label_relative]["origin"]  # 对侧脑区

            if origin_area in report or relative_area in report:
                continue
            if label_origin == 15 or label_origin == 16:
                report = self._get_alone_brain_tendency(
                    report,
                    label_origin,
                    origin_area,
                    low_perfusion_mask,
                    core_infarct_mask,
                    left_stem_mask,
                    right_stem_mask,
                )
            else:
                report = self._get_double_sides_tendency(
                    report,
                    label_origin,
                    label_relative,
                    origin_area,
                    relative_area,
                    low_perfusion_mask,
                    core_infarct_mask,
                )
        return report

    def _get_double_sides_tendency(
        self,
        report,
        label_origin,
        label_relative,
        origin_area,
        relative_area,
        low_perfusion_mask,
        core_infarct_mask,
    ):
        left_area_cbv, right_area_cbv = (
            self.cbv_array[self.brain_area_array == label_origin].mean(),
            self.cbv_array[self.brain_area_array == label_relative].mean(),
        )
        left_area_cbf, right_area_cbf = (
            self.cbf_array[self.brain_area_array == label_origin].mean(),
            self.cbf_array[self.brain_area_array == label_relative].mean(),
        )
        left_area_mtt, right_area_mtt = (
            self.mtt_array[self.brain_area_array == label_origin].mean(),
            self.mtt_array[self.brain_area_array == label_relative].mean(),
        )
        left_area_ttp, right_area_ttp = (
            self.ttp_array[self.brain_area_array == label_origin].mean(),
            self.ttp_array[self.brain_area_array == label_relative].mean(),
        )
        left_area_tmax, right_area_tmax = (
            self.tmax_array[self.brain_area_array == label_origin].mean(),
            self.tmax_array[self.brain_area_array == label_relative].mean(),
        )
        # 左右脑区均有病灶
        if label_origin in LEFT_BRAIN_LABEL and label_relative in RIGHT_BRAIN_LABEL:
            report[origin_area] = {}
            report[relative_area] = {}
            # 左脑区低灌注
            origin_low_perfusion = round(
                np.count_nonzero(
                    low_perfusion_mask[self.brain_area_array == label_origin]
                )
                * self.voxel,
                1,
            )
            # 右脑区低灌注
            relative_low_perfusion = round(
                np.count_nonzero(
                    low_perfusion_mask[self.brain_area_array == label_relative]
                )
                * self.voxel,
                1,
            )
            # 左脑区梗死区
            origin_core_penumbra = round(
                np.count_nonzero(
                    core_infarct_mask[self.brain_area_array == label_origin]
                )
                * self.voxel,
                1,
            )

            # 右脑区梗死区
            relative_core_penumbra = round(
                np.count_nonzero(
                    core_infarct_mask[self.brain_area_array == label_relative]
                )
                * self.voxel,
                1,
            )

            # cbf 趋势
            origin_cbf, relative_cbf = self.get_brain_area_tendency(
                left_area_cbf, right_area_cbf, both_area=True
            )
            # cbv 趋势
            origin_cbv, relative_cbv = self.get_brain_area_tendency(
                left_area_cbv, right_area_cbv, both_area=True
            )
            # mtt 趋势
            origin_mtt, relative_mtt = self.get_brain_area_tendency(
                left_area_mtt, right_area_mtt, both_area=True, is_time=True
            )
            # ttp 趋势
            origin_ttp, relative_ttp = self.get_brain_area_tendency(
                left_area_ttp, right_area_ttp, both_area=True, is_time=True
            )
            # tmax 趋势
            origin_tmax, relative_tmax = self.get_brain_area_tendency(
                left_area_tmax, right_area_tmax, both_area=True, is_time=True
            )

            report[origin_area]["CBF"] = origin_cbf
            report[relative_area]["CBF"] = relative_cbf

            report[origin_area]["CBV"] = origin_cbv
            report[relative_area]["CBV"] = relative_cbv

            report[origin_area]["MTT"] = origin_mtt
            report[relative_area]["MTT"] = relative_mtt

            report[origin_area]["TTP"] = origin_ttp
            report[relative_area]["TTP"] = relative_ttp

            report[origin_area]["Tmax"] = origin_tmax
            report[relative_area]["Tmax"] = relative_tmax

            report[origin_area]["corePenumbra"] = origin_core_penumbra
            report[origin_area]["lowPerfusion"] = origin_low_perfusion

            report[relative_area]["corePenumbra"] = relative_core_penumbra
            report[relative_area]["lowPerfusion"] = relative_low_perfusion
        # 病灶只在左脑区
        elif label_origin in LEFT_BRAIN_LABEL:
            report[origin_area] = {}
            # 计算 低灌注和梗死区
            origin_core_penumbra = round(
                np.count_nonzero(
                    core_infarct_mask[self.brain_area_array == label_origin]
                )
                * self.voxel,
                1,
            )

            origin_low_perfusion = round(
                np.count_nonzero(
                    low_perfusion_mask[self.brain_area_array == label_origin]
                )
                * self.voxel,
                1,
            )
            origin_cbf, _ = self.get_brain_area_tendency(left_area_cbf, right_area_cbf)
            report[origin_area]["CBF"] = origin_cbf
            #  cbv
            origin_cbv, _ = self.get_brain_area_tendency(left_area_cbv, right_area_cbv)
            report[origin_area]["CBV"] = origin_cbv
            # mtt
            origin_mtt, _ = self.get_brain_area_tendency(
                left_area_mtt, right_area_mtt, is_time=True
            )
            report[origin_area]["MTT"] = origin_mtt
            # ttp
            origin_ttp, _ = self.get_brain_area_tendency(
                left_area_ttp, right_area_ttp, is_time=True
            )
            report[origin_area]["TTP"] = origin_ttp
            # tmax
            origin_tmax, _ = self.get_brain_area_tendency(
                left_area_tmax, right_area_tmax, is_time=True
            )
            report[origin_area]["TMAX"] = origin_tmax
            report[origin_area]["corePenumbra"] = origin_core_penumbra
            report[origin_area]["lowPerfusion"] = origin_low_perfusion
        elif label_origin in RIGHT_BRAIN_LABEL:
            report[relative_area] = {}
            # 计算 低灌注体积和梗死区体积
            relative_core_penumbra = round(
                np.count_nonzero(
                    core_infarct_mask[self.brain_area_array == label_relative]
                )
                * self.voxel,
                1,
            )

            relative_low_perfusion = round(
                np.count_nonzero(
                    low_perfusion_mask[self.brain_area_array == label_relative]
                )
                * self.voxel,
                1,
            )
            relative_cbf, _ = self.get_brain_area_tendency(
                right_area_cbf, left_area_cbf
            )
            report[relative_area]["CBF"] = relative_cbf

            #  cbv
            relative_cbv, _ = self.get_brain_area_tendency(
                right_area_cbv, left_area_cbv
            )
            report[relative_area]["CBV"] = relative_cbv
            # mtt
            relative_mtt, _ = self.get_brain_area_tendency(
                right_area_mtt, left_area_mtt, is_time=True
            )
            report[relative_area]["MTT"] = relative_mtt
            # ttp
            relative_ttp, _ = self.get_brain_area_tendency(
                right_area_ttp, left_area_ttp, is_time=True
            )
            report[relative_area]["TTP"] = relative_ttp
            # tmax
            relative_tmax, _ = self.get_brain_area_tendency(
                right_area_tmax, left_area_tmax, is_time=True
            )
            report[relative_area]["Tmax"] = relative_tmax
            report[relative_area]["corePenumbra"] = relative_core_penumbra
            report[relative_area]["lowPerfusion"] = relative_low_perfusion
        return report

    def get_brain_area_tendency(
        self, label1, label2, both_area=False, is_time=False, threshold=0.5
    ):
        """获取脑区异常趋势
        Args:
            both_area: 是否涉及双侧脑区
            is_time: 是否为时间维度参数
            threshold: 左右持平的阈值
        Return: tendency
        """
        if not both_area:
            if label1 / label2 < threshold:
                tendency = "average"
            elif label1 > label2:
                tendency = "above"
            else:
                tendency = "average" if is_time else "below"
            return tendency, ""
        else:
            if label1 / label2 < threshold or label2 / label1 < threshold:
                tendency1, tendency2 = "average", "average"
            elif label1 > label2:
                tendency1, tendency2 = (
                    ("above", "average")
                    if is_time
                    else (
                        "above",
                        "below",
                    )
                )
            else:
                tendency1, tendency2 = (
                    ("average", "above")
                    if is_time
                    else (
                        "below",
                        "above",
                    )
                )
            return tendency1, tendency2

    def _get_alone_brain_tendency(
        self,
        report,
        label_origin,
        origin_area,
        low_perfusion_mask,
        core_infarct_mask,
        left_stem_mask,
        right_stem_mask,
    ):
        """根据大脑中线位置，获取脑干、胼胝体 CTP 参数信息"""
        report[origin_area] = {}

        l_cbf_value, r_cbf_value = (
            self.cbf_array[left_stem_mask == 1].mean(),
            self.cbf_array[right_stem_mask == 1].mean(),
        )
        l_cbv_value, r_cbv_value = (
            self.cbv_array[left_stem_mask == 1].mean(),
            self.cbf_array[right_stem_mask == 1].mean(),
        )
        l_mtt_value, r_mtt_value = (
            self.mtt_array[left_stem_mask == 1].mean(),
            self.cbf_array[right_stem_mask == 1].mean(),
        )
        l_ttp_value, r_ttp_value = (
            self.ttp_array[left_stem_mask == 1].mean(),
            self.cbf_array[right_stem_mask == 1].mean(),
        )
        l_tmax_value, r_tmax_value = (
            self.tmax_array[left_stem_mask == 1].mean(),
            self.cbf_array[right_stem_mask == 1].mean(),
        )

        cbf_tendency, _ = self.get_brain_area_tendency(l_cbf_value, r_cbf_value)
        cbv_tendency, _ = self.get_brain_area_tendency(l_cbv_value, r_cbv_value)
        mtt_tendency, _ = self.get_brain_area_tendency(l_mtt_value, r_mtt_value)
        ttp_tendency, _ = self.get_brain_area_tendency(l_ttp_value, r_ttp_value)
        tmax_tendency, _ = self.get_brain_area_tendency(l_tmax_value, r_tmax_value)
        low_perfusion = round(
            np.count_nonzero(low_perfusion_mask[self.brain_area_array == label_origin])
            * self.voxel,
            1,
        )
        core_penumbra = round(
            np.count_nonzero(core_infarct_mask[self.brain_area_array == label_origin])
            * self.voxel,
            1,
        )
        report[origin_area]["CBF"] = cbf_tendency
        report[origin_area]["CBV"] = cbv_tendency
        report[origin_area]["MTT"] = mtt_tendency
        report[origin_area]["TTP"] = ttp_tendency
        report[origin_area]["Tmax"] = tmax_tendency
        report[origin_area]["corePenumbra"] = core_penumbra
        report[origin_area]["lowPerfusion"] = low_perfusion
        return report

    def get_brain_area_tendency(self, label1, label2, both_area=False):
        """获取脑区异常趋势
        Return: tendency
        """
        if not both_area:
            if label1 / label2 < 0.5:
                tendency = "average"
            elif label1 > label2:
                tendency = "above"
            else:
                tendency = "below"
            return tendency, ""
        else:
            if label1 / label2 < 0.5 or label2 / label1 < 0.5:
                tendency1, tendency2 = "average", "average"
            elif label1 > label2:
                tendency1, tendency2 = "above", "below"
            else:
                tendency1, tendency2 = "below", "above"
            return tendency1, tendency2

    def get_mismatch_result(
        self,
        tmip_rgb_array,
        cbf_gray_array,
        core_infarct_errors,
        single_core_mask,
        low_perfusion_errors,
        center,
        angle,
        index,
        mismatch_result,
        rows,
        cols,
    ):
        """mismatch 视图: rCBF<0.3 红色，tmax>6 黄色"""
        rgb_array = tmip_rgb_array.copy()
        # tmip 沿旋转矩阵旋转angle
        rotate_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle + 90, scale=1
        )

        # 二值化mask
        mask1 = np.zeros((cbf_gray_array.shape), np.uint8)
        mask2 = np.zeros((cbf_gray_array.shape), np.uint8)
        # 低灌注和梗死区 0,1 mask
        single_low_mask = np.zeros((cbf_gray_array.shape), np.uint8)
        # 病灶 0，1 mask
        lesion_mask = np.zeros((cbf_gray_array.shape), np.uint8)

        mask1[low_perfusion_errors] = 255
        mask2[single_core_mask == 1] = 255
        single_low_mask[low_perfusion_errors] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sum_B = get_cc(mask1, kernel)

        sum_B = cv2.dilate(sum_B, kernel, iterations=8)
        sum_B = cv2.erode(sum_B, kernel, iterations=8)
        lesion_mask[np.where(sum_B > 0)] = 1
        rgb_array[np.where(sum_B > 0)] = color_map[Mismatch_conf[1]["color"]]

        # mask2 腐蚀膨胀
        # mask2 = cv2.erode(cv2.dilate(mask2, kernel, iterations=1), kernel, iterations=5)
        # mask2 = cv2.dilate(mask2, kernel, iterations=3)
        # 取交集
        tmip_rotated_image = cv2.warpAffine(
            src=rgb_array, M=rotate_matrix, dsize=(rows, cols)
        )
        thresh2 = cv2.bitwise_and(mask2, sum_B)
        thresh2 = cv2.dilate(thresh2, kernel, iterations=8)
        thresh2 = cv2.erode(thresh2, kernel, iterations=8)
        lesion_mask[np.where(thresh2 > 0)] = 1
        rgb_array[np.where(thresh2 > 0)] = color_map[Mismatch_conf[0]["color"]]
        # tmip_rotated_image[np.where(thresh2 > 0)] = color_map[Mismatch_conf[0]["color"]]

        # 沿旋转矩阵反向旋转 angle
        # rotate_matrix = cv2.getRotationMatrix2D(
        #     center=center, angle=-angle - 90, scale=1
        # )
        # origin_image = cv2.warpAffine(
        #     src=tmip_rotated_image, M=rotate_matrix, dsize=(rows, cols)
        # )
        origin_image = rgb_array
        cv2.imwrite(f"./result_mismatch{str(index)}.jpg", origin_image)
        mismatch_result["Mismatch" + str(index)] = origin_image
        # np_array_to_dcm(ds, origin_image, f"./test{str(index)}.dcm", 127, 255, True)
        return mismatch_result, lesion_mask, single_low_mask

    def get_rcbf_result(
        self,
        tmip_rgb_array,
        cbf_gray_array,
        center,
        angle,
        cbf_normal,
        index,
        rcbf_result,
    ):
        core_infarct_errors = []
        rgb_array = tmip_rgb_array.copy()
        single_core_mask = np.zeros((cbf_gray_array.shape), np.uint8)

        # tmip 沿旋转矩阵旋转angle
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
        tmip_rotated_image = cv2.warpAffine(
            src=rgb_array, M=rotate_matrix, dsize=(rows, cols)
        )
        cv2.imwrite("./tmip_rotated_image.jpg", tmip_rotated_image)

        # cbf 沿旋转矩阵旋转angle
        rotated_image = cv2.warpAffine(
            src=cbf_gray_array, M=rotate_matrix, dsize=(rows, cols)
        )
        rotated_core_mask = cv2.warpAffine(
            src=single_core_mask, M=rotate_matrix, dsize=(rows, cols)
        )
        # 镜像翻转
        mirror = np.fliplr(rotated_image)
        mask1 = np.zeros((cbf_gray_array.shape[1], cbf_gray_array.shape[0]), np.uint8)
        mask2 = np.zeros((cbf_gray_array.shape[1], cbf_gray_array.shape[0]), np.uint8)
        mask3 = np.zeros((cbf_gray_array.shape[1], cbf_gray_array.shape[0]), np.uint8)
        lesion_mask = np.zeros((cbf_gray_array.shape), np.uint8)
        for rcbf_threshold in rCBF:
            cbf_setting = rcbf_threshold["threshold"]
            color_bgr = color_map[rcbf_threshold["color"]]
            l_mean = rotated_image[:, :256].mean()
            r_mean = mirror[:, :256].mean()
            # 原侧值 / 镜像值,TODO: 没有用到正常值这一参数
            temp_cbf = np.divide(
                rotated_image,
                mirror,
                out=np.zeros_like(rotated_image),
                where=mirror > 0,
            )
            condition = (
                (temp_cbf < (cbf_setting / 100))
                if l_mean < r_mean
                else (temp_cbf > 1 / (cbf_setting / 100))
            )

            # if l_mean < r_mean
            error_arr = np.where((0 < temp_cbf) & condition)

            if cbf_setting == 40:
                cbf_high = list(error_arr)
                mask1[error_arr] = 255
            if cbf_setting == 30:
                core_infarct_errors = error_arr
                rotated_core_mask[error_arr] = 1
                mask2[error_arr] = 255
            if cbf_setting == 20:
                cbf_low = list(error_arr)
                mask3[error_arr] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sum_B = get_cc(mask1, kernel)

        sum_B = cv2.morphologyEx(sum_B, cv2.MORPH_CLOSE, kernel, iterations=8)
        lesion_mask[np.where(sum_B > 0)] = 1
        tmip_rotated_image[np.where(sum_B > 0)] = color_map[rCBF[0]["color"]]

        thresh2 = cv2.bitwise_and(mask2, sum_B)
        thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel, iterations=8)
        lesion_mask[np.where(thresh2 > 0)] = 1
        tmip_rotated_image[np.where(thresh2 > 0)] = color_map[rCBF[1]["color"]]

        thresh3 = cv2.bitwise_and(mask3, sum_B)
        thresh3 = cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, kernel, iterations=8)
        lesion_mask[np.where(thresh3 > 0)] = 1
        tmip_rotated_image[np.where(thresh3 > 0)] = color_map[rCBF[2]["color"]]

        # 沿旋转矩阵反向旋转 angle
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-angle, scale=1)
        origin_image = cv2.warpAffine(
            src=tmip_rotated_image, M=rotate_matrix, dsize=(rows, cols)
        )
        rcbf_result["rCBF" + str(index)] = origin_image
        cv2.imwrite(f"./result_rcbf{str(index)}.jpg", origin_image)
        single_core_mask = cv2.warpAffine(
            src=rotated_core_mask, M=rotate_matrix, dsize=(rows, cols)
        )
        core_infarct_errors = np.where(thresh2 > 0)
        return (
            rcbf_result,
            core_infarct_errors,
            single_core_mask,
            lesion_mask.tolist(),
            cbf_high,
            cbf_low,
        )

    def get_rcbv_result(
        self,
        tmip_rgb_array,
        cbv_gray_array,
        center,
        angle,
        cbv_normal,
        index,
        rcbv_result,
        rows,
        cols,
    ):
        rgb_array = tmip_rgb_array.copy()
        # tmip 沿旋转矩阵旋转angle
        rotate_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle + 90, scale=1
        )
        tmip_rotated_image = cv2.warpAffine(
            src=rgb_array, M=rotate_matrix, dsize=(rows, cols)
        )

        # cbv 沿旋转矩阵旋转angle
        rotate_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle + 90, scale=1
        )
        rotated_image = cv2.warpAffine(
            src=cbv_gray_array, M=rotate_matrix, dsize=(rows, cols)
        )
        # 镜像翻转
        mirror = np.fliplr(rotated_image)
        # 每个阈值一个mask, 取连通分量
        mask1 = np.zeros((cbv_gray_array.shape), np.uint8)
        mask2 = np.zeros((cbv_gray_array.shape), np.uint8)
        mask3 = np.zeros((cbv_gray_array.shape), np.uint8)
        lesion_mask = np.zeros((cbv_gray_array.shape), np.uint8)
        for rcbv_threshold in rCBV:
            cbv_setting = rcbv_threshold["threshold"]
            # 原侧值 / 镜像值, TODO: 没有用到正常值这一参数
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
        tmip_rotated_image[np.where(sum_B > 0)] = color_map[rCBV[0]["color"]]

        thresh2 = cv2.bitwise_and(mask2, sum_B)
        thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel, iterations=8)
        tmip_rotated_image[np.where(thresh2 > 0)] = color_map[rCBV[1]["color"]]

        thresh3 = cv2.bitwise_and(mask3, sum_B)
        thresh3 = cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, kernel, iterations=8)
        tmip_rotated_image[np.where(thresh3 > 0)] = color_map[rCBV[2]["color"]]

        # 沿旋转矩阵反向旋转 angle
        rotate_matrix = cv2.getRotationMatrix2D(
            center=center, angle=-angle - 90, scale=1
        )
        origin_image = cv2.warpAffine(
            src=tmip_rotated_image, M=rotate_matrix, dsize=(rows, cols)
        )
        rcbv_result["rCBV_" + str(index)] = origin_image
        # cv2.imwrite("./result_rcbv.jpg", origin_image)
        return rcbv_result, lesion_mask.tolist(), cbv_high, cbv_mid, cbv_low

    def get_tmax_result(self, tmip_rgb_array, tmax_gray_array, tmax_result, index):
        low_perfusion_errors = []
        rgb_array = tmip_rgb_array.copy()
        mask1 = np.zeros((tmax_gray_array.shape), np.uint8)
        mask2 = np.zeros((tmax_gray_array.shape), np.uint8)
        mask3 = np.zeros((tmax_gray_array.shape), np.uint8)
        lesion_mask = np.zeros((tmax_gray_array.shape), np.uint8)

        for tmax_threshold in Tmax:
            tmax_setting = tmax_threshold["threshold"]
            error_arr = np.where((tmax_gray_array > tmax_setting))

            # Tmax 大于6s为低灌注区域
            if tmax_setting == 6:
                mask1[error_arr] = 255
                low_perfusion_errors = np.where((tmax_gray_array > tmax_setting))
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
        rgb_array[np.where(sum_B > 0)] = color_map[Tmax[0]["color"]]

        thresh2 = cv2.bitwise_and(mask2, sum_B)
        thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel, iterations=8)
        lesion_mask[np.where(thresh2 > 0)] = 1
        rgb_array[np.where(thresh2 > 0)] = color_map[Tmax[1]["color"]]

        thresh3 = cv2.bitwise_and(mask3, sum_B)
        thresh3 = cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, kernel, iterations=8)
        lesion_mask[np.where(thresh3 > 0)] = 1
        rgb_array[np.where(thresh3 > 0)] = color_map[Tmax[2]["color"]]
        tmax_result["tmax_" + str(index)] = rgb_array
        # cv2.imwrite(f"./result_tmax{str(index)}.jpg", rgb_array)
        return (
            tmax_result,
            low_perfusion_errors,
            lesion_mask.tolist(),
            tmax_high,
            tmax_low,
        )


def main():
    for mask_file in os.listdir(mask_path):
        if mask_file == "CBF.nii.gz":
            sitk_img = sitk.ReadImage(os.path.join(mask_path, mask_file))
            direction = sitk_img.GetDirection()

            mask_array = sitk.GetArrayFromImage(sitk_img)
        if mask_file == "CBV.nii.gz":
            sitk_img = sitk.ReadImage(os.path.join(mask_path, mask_file))
            cbv_array = sitk.GetArrayFromImage(sitk_img)

    brain_array = sitk.GetArrayFromImage(
        sitk.ReadImage(
            "/media/tx-deepocean/Data/DICOMS/demos/37/brain_area_mask.nii.gz"
        )
    )

    tmax_array = sitk.GetArrayFromImage(
        sitk.ReadImage("/media/tx-deepocean/Data/DICOMS/demos/37/TMAX.nii.gz")
    )
    tmip_img = sitk.ReadImage(
        "/media/tx-deepocean/Data/DICOMS/demos/37/TMIP_NO_SKULL.nii.gz"
    )
    spacing = tmip_img.GetSpacing()
    origin = tmip_img.GetOrigin()
    depth = tmip_img.GetDepth()
    print(f"******depth: {depth}")
    tmip_array = sitk.GetArrayFromImage(tmip_img)
    mtt_array = sitk.GetArrayFromImage(
        sitk.ReadImage("/media/tx-deepocean/Data/DICOMS/demos/37/MTT.nii.gz")
    )
    ttp_array = sitk.GetArrayFromImage(
        sitk.ReadImage("/media/tx-deepocean/Data/DICOMS/demos/37/TTP.nii.gz")
    )
    centerline = {
        "point": [13.614892619395746, 70.6815248380492, 356.8286597371511],
        "vector": [-0.9994793046675267, -0.003673192254234554, 0.03205662490002515],
    }
    t_3 = time.time()
    depth = tmip_img.GetDepth()
    mis = Mismatch(
        mask_array,
        cbv_array,
        mtt_array,
        tmax_array,
        ttp_array,
        brain_array,
        spacing,
        depth,
    )
    image_result, ctp_lesion = mis.get_mismatch(
        tmip_array,
        Mismatch_conf,
        centerline,
        origin,
        direction,
    )
    print(f"重新生成时间: {time.time() - t_3}")
    print(image_result.keys())
    print(json.dumps(ctp_lesion))


if __name__ == "__main__":
    main()
