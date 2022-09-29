import io
import math
import os
import time

import cc3d
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pydicom
import SimpleITK as sitk
from constans import BRAIN_AREA_MAP, left_brain_label, right_brain_label
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from skimage import measure

from .mismatch_utils import (convert_noraml_to_standard_plane, get_cc,
                             get_line, get_point_on_plane_by_yz,
                             gray2rgb_array, remove_small_volume)

mask_path = "/media/tx-deepocean/Data/DICOMS/demos/CN010023-000070191600-939-301"
# dcm_file = "/media/tx-deepocean/Data/DICOMS/demos/TMAX/TMAX023.dcm"
# ds = pydicom.read_file(dcm_file, force=True)


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
Mismatch = [
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
    {"type": "Tmax", "compare": "gt", "threshold": 4, "color": "#3F87F5"},
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

# class Mismatch:
#     def __init__(self, cbf_array, cbv_array, mtt_array, tmax_array, ):
#         self.cbf_array = cbf_array
#         self.cbv_array = cbv_array


def get_mismatch(
    cbf_array: np.ndarray,
    tmip_array: np.ndarray,
    tmax_array: np.ndarray,
    mtt_array: np.ndarray,
    ttp_array: np.ndarray,
    cbv_array: np.ndarray,
    brain_area_array: np.ndarray,
    config: list,
    centerline: dict,
    origin,
    spacing,
    direction,
):
    """
    rCBF计算公式:
        先和大脑对侧比,保证两侧 小值/大值<0.3 作为异常点, 进而判定异常点所在侧, 着色; 若镜像侧值为背景, 与正常值50比;
        若两侧均有异常， 且相差小于10%, 认定两点均不是异常， 否则取较大值为异常值， 着色;
    """
    print(f"cbf_array min: {cbf_array.min()}, max: {cbf_array.max()}")
    perfusion_infarct_pixel = {}

    result, ctp_lesion = {}, {}
    (
        rcbv_lesion_mask,
        rcbf_lesion_mask,
        mismatch_lesion_mask,
        tmax_lesion_mask,
        mtt_lesion_mask,
        ttp_lesion_mask,
    ) = ([], [], [], [], [], [])

    cbf_normal = 50  # cbf 正常值
    cbv_normal = 4  # 4ml 正常值

    # 1、区分左右侧大脑:点在直线左、右侧
    # 点法式 转 标准式
    A, B, C, D = convert_noraml_to_standard_plane(
        centerline["point"], centerline["vector"]
    )
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
    for dcm_slice in range(cbf_array.shape[0]):
        # if dcm_slice == 10:
        rows, cols = cbf_array.shape[1:]
        # 视图选取去骨 tmip
        tmip_gray_array = tmip_array[dcm_slice, :, :]
        # gray to rgb
        tmip_rgb_array = gray2rgb_array(tmip_gray_array)
        # 由两点式确定直线方程 ，由于原点为左上，取Ax+By+C=0关于x轴对称的直线方程:Ax-By+C=0
        # (x1, y1), (x2, y2) = get_point_on_plane_by_yz(
        #     A, B, C, D, rows, 0, 15
        # )
        (x1, y1), (x2, y2) = (282, 106), (240, 412)  #
        # 注意采用x=ky+b模型，原因是如果中线没有偏转，例x=10这种情况，x=ky+b不用考虑这种特殊情况，不用额外写个if条件判断
        k, b = get_line(x1, x2, y1, y2)
        # 获取某层
        cbf_gray_array = cbf_array[dcm_slice, :, :]
        cv2.imwrite("./result.jpg", cbf_gray_array)
        cbv_gray_array = cbv_array[dcm_slice, :, :]
        tmax_gray_array = tmax_array[dcm_slice, :, :]

        # 计算倾斜角
        angle = (math.atan(k)) / math.pi * 180
        # 计算中线中点
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        # 所有层的灌注和梗死像素
        # 获取 rcbf 视图， 小于0.2红色，0.2>=x<0.3 黄色， 0.3>=x<0.4 绿色
        t0 = time.time()
        (
            rcbf_result,
            core_infarct_errors,
            cbf_lesion_mask,
            cbf_high,
            cbf_low,
        ) = get_rcbf_result(
            tmip_rgb_array,
            cbf_gray_array,
            brain_area_array,
            center,
            angle,
            cbf_normal,
            rCBF,
            index,
            rcbf_result,
            rows,
            cols,
        )
        perfusion_infarct_pixel[dcm_slice] = {}
        perfusion_infarct_pixel[dcm_slice].update(
            {"core_infarct": len(core_infarct_errors[0])}
        )
        rcbf_lesion_mask.append(cbf_lesion_mask)
        # 计算 rcbf 视图的 summary
        cbf_high = len(cbf_high[0]) if cbf_high else 0
        cbf_low = len(cbf_low[0]) if cbf_low else 0
        cbf_sum_high += cbf_high
        cbf_sum_low += cbf_low
        print(f"一张rcbf: {time.time() - t0}")
        # 获取 rcbv 视图, 小于0.45绿色， 小于0.4黄色， 小于0.35红色
        rcbv_result, cbv_lesion_mask, cbv_high, cbv_mid, cbv_low = get_rcbv_result(
            tmip_rgb_array,
            cbv_gray_array,
            center,
            angle,
            cbv_normal,
            rCBV,
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
        ) = get_tmax_result(
            tmip_rgb_array, tmax_gray_array, Tmax, k, b, tmax_result, index
        )
        perfusion_infarct_pixel[dcm_slice].update(
            {"low_perfusion": len(low_perfusion_errors[0])}
        )
        tmax_lesion_mask.append(tmax_lesion_mask)
        # 计算 tmax 视图的 summary
        tmax_high = len(tmax_high[0] if tmax_high else 0)
        tmax_low = len(tmax_low[0] if tmax_low else 0)

        # 获取 mismatch 视图 ,这里取巧， 利用rcbv和tmax计算视图计算好的像素 list
        mismatch_result, mis_lesion_mask = get_mismatch_result(
            tmip_rgb_array,
            cbf_gray_array,
            core_infarct_errors,
            low_perfusion_errors,
            center,
            angle,
            index,
            mismatch_result,
            rows,
            cols,
        )
        mismatch_lesion_mask.append(mis_lesion_mask)
        #
        index += 1
        low_perfusion_area += len(low_perfusion_errors)
        core_infarct_area += len(core_infarct_errors)

    print(perfusion_infarct_pixel)
    # 返回病灶信息
    # 灌注区、梗死区分析报告(脑区、cbf,cbv,mtt,ttp,tmax,核心梗死区、低灌注区)
    if not mismatch_lesion_mask:
        lesions, brain_area_paras_dict = [], {}
    else:
        mismatch_lesion_mask = np.array(mismatch_lesion_mask).reshape(
            (len(mismatch_lesion_mask), 512, 512)
        )
        lesions, brain_area_paras_dict = get_data_result(
            mismatch_lesion_mask,
            perfusion_infarct_pixel,
            cbf_array,
            cbv_array,
            mtt_array,
            ttp_array,
            tmax_array,
            brain_area_array,
        )
    ctp_lesion["lesions"] = lesions
    ctp_lesion["brain_area_paras"] = brain_area_paras_dict
    # 1111111111111111111111111111111111111111111 Mismatch异常内容
    if mismatch_lesion_mask.size == 0:
        mismatch_info = {}
    else:
        voxel = spacing[0] * spacing[1] * spacing[2]
        # 低灌注区域 将6s＜Tmax的区域 或 rMTT＞145%
        low_perfusion_volume = low_perfusion_area * voxel / 1000
        # 核心梗死区 将Tmax＞10s 或 rCBF＜30%对侧正常脑组织 或 绝对值：CBV＜2ml/100g
        core_infarct_volume = core_infarct_area * voxel / 1000

        # mismatch 比率: 低灌注体积/核心梗死体积
        mismatch_ratio = round(low_perfusion_volume / core_infarct_volume, 1)
        # 半暗带体积(mismatch 体积): 低灌注体积-核心梗死体积
        mismatch_volume = low_perfusion_volume - core_infarct_volume
        mismatch_info.update(
            {
                "ratio": mismatch_ratio,
                "mismatch_volume": mismatch_volume,
                "low_perfusion_volume": low_perfusion_volume,
                "core_infarct_volume": core_infarct_volume,
            }
        )
    ctp_lesion["mismatch_info"] = mismatch_info

    # 更新 summary
    rcbf_result["summary"] = {}
    rcbf_result["summary"]["high"] = cbf_sum_high * voxel / 1000
    rcbf_result["summary"]["mid"] = core_infarct_volume
    rcbf_result["summary"]["low"] = cbf_sum_low * voxel / 1000

    rcbv_result["summary"] = {}
    rcbv_result["summary"]["high"] = cbv_sum_high * voxel / 1000
    rcbv_result["summary"]["mid"] = cbv_sum_mid * voxel / 1000
    rcbv_result["summary"]["low"] = cbv_sum_low * voxel / 1000

    tmax_result["summary"] = {}
    tmax_result["summary"]["low"] = low_perfusion_volume
    tmax_result["summary"]["mid"] = tmax_low * voxel / 1000
    tmax_result["summary"]["high"] = tmax_high * voxel / 1000

    mismatch_result["summary"] = {}
    mismatch_result["summary"]["rcbf"] = core_infarct_volume
    mismatch_result["summary"]["tmax"] = low_perfusion_volume
    mismatch_result["summary"]["mismatch"] = mismatch_volume
    # result.update[]
    return result, ctp_lesion


def get_data_result(
    mismatch_lesion_mask,
    perfusion_infarct_pixel,
    cbf_array,
    cbv_array,
    mtt_array,
    ttp_array,
    tmax_array,
    brain_area_array,
):
    """获取3d连通域"""
    brain_area_paras_dict, single_brain_paras, brain_areas = {}, {}, []
    # 获取病灶数量， 病灶标签
    cc, cc_unique = remove_small_volume(mismatch_lesion_mask)
    t1 = time.time()
    lesions, brain_area_paras_dict = get_view_lesion(
        cc,
        cc_unique,
        cbv_array,
        cbf_array,
        mtt_array,
        tmax_array,
        ttp_array,
        brain_area_array,
    )
    print(f"*******病灶和参数计算时间: {time.time() - t1}")

    i = 1

    # TODO3: 计算 summary 数值、脑区参数分析、(低灌注体积，mismatch 体积，梗死区体积)

    # 计算脑区低灌注体积、梗死体积

    brain_area_paras_dict.update({BRAIN_AREA_MAP[label_left]["origin"]: brain_paras})

    print(f"lesions: {lesions}")
    print(f"brain_area_paras_dict: {brain_area_paras_dict}")
    return lesions, brain_area_paras_dict


def get_view_lesion(
    cc: np.ndarray,
    cc_unique: list,
    cbv_array,
    cbf_array,
    mtt_array,
    tmax_array,
    ttp_array,
    brain_area_array,
):
    """计算病灶
    Args:
        cc: 病灶mask (3D)
        cc_unique: 病灶 label
    Returns:
        lesions: dict
    """
    lesions, brain_area_paras_dict = {}, {}
    for lesion in cc_unique:
        brain_areas = []
        # 病灶所经过的脑区
        rcbf_lesions, rcbv_lesions, mis_lesions, tmax_lesions = [], [], [], []
        (
            mis_single_lesion,
            rcbf_single_lesion,
            rcbv_single_lesion,
            tmax_single_lesion,
        ) = ({}, {}, {}, {})
        brain_list = list(np.unique(brain_area_array[cc == lesion])).remove(0)
        print(f"*****brain_list: {brain_list}, lesion: {lesion}")
        for area_label in brain_list:
            if area_label == 15 or area_label == 16:
                continue
            brain_areas.append(BRAIN_AREA_MAP[area_label]["origin"])
            # TODO: 在第几层
            lesion_slice = [3, 4]
            # TODO: 遍历层求 mismatch 低灌注体积、梗死区体积
            mis_single_lesion["brain_areas"] = brain_areas
            mis_single_lesion["slcie"] = lesion_slice
            mis_single_lesion["low_per"] = 11
            mis_single_lesion["core_infarct"] = 33

            rcbf_single_lesion["brain_areas"] = brain_areas
            rcbf_single_lesion["slice"] = lesion_slice
            rcbf_single_lesion["high"] = 15
            rcbf_single_lesion["mid"] = 16
            rcbf_single_lesion["low"] = 17

            rcbv_single_lesion["brain_areas"] = brain_areas
            rcbv_single_lesion["slice"] = lesion_slice
            rcbv_single_lesion["high"] = 15
            rcbv_single_lesion["mid"] = 16
            rcbv_single_lesion["low"] = 17

            tmax_single_lesion["brain_areas"] = 22
            tmax_single_lesion["slcie"] = lesion_slice
            tmax_single_lesion["high"] = 15
            tmax_single_lesion["mid"] = 16
            tmax_single_lesion["low"] = 17

            # 计算脑区异常参数
            # 左右脑区都有灌注，计算左脑区，对侧取反
            brain_area_paras_dict[label_origin] = {}
            if (
                label_origin in brain_area_paras_dict
                or label_relative in brain_area_paras_dict
            ):
                continue
            brain_area_paras_dict = get_brain_area_paras(
                label_origin,
                cbv_array,
                cbf_array,
                mtt_array,
                ttp_array,
                tmax_array,
                brain_area_array,
                brain_area_paras_dict,
            )

        rcbf_lesions.append(rcbf_single_lesion)
        rcbv_lesions.append(rcbv_single_lesion)
        tmax_lesions.append(tmax_single_lesion)
        mis_lesions.append(mis_single_lesion)

    lesions["rcbf"] = rcbf_single_lesion
    lesions["rcbv"] = rcbv_single_lesion
    lesions["tmax"] = tmax_single_lesion
    lesions["mismatch"] = mis_single_lesion
    return lesions, brain_area_paras_dict


def get_brain_area_paras(
    area_label,
    cbv_array,
    cbf_array,
    mtt_array,
    ttp_array,
    tmax_array,
    brain_area_array,
    brain_area_paras_dict,
):
    if (
        area_label in left_brain_label
        and BRAIN_AREA_MAP[area_label]["relative"] in right_brain_label
    ):
        label_origin = area_label
        label_relative = BRAIN_AREA_MAP[area_label]["relative"]
        # 计算灌注脑区的异常参数
        left_area_cbv, right_area_cbv = (
            cbv_array[brain_area_array == label_origin].mean(),
            cbv_array[brain_area_array == label_relative].mean(),
        )
        left_area_cbf, right_area_cbf = (
            cbf_array[brain_area_array == label_origin].mean(),
            cbf_array[brain_area_array == label_relative].mean(),
        )
        left_area_mtt, right_area_mtt = (
            mtt_array[brain_area_array == label_origin].mean(),
            mtt_array[brain_area_array == label_relative].mean(),
        )
        left_area_ttp, right_area_ttp = (
            ttp_array[brain_area_array == label_origin].mean(),
            ttp_array[brain_area_array == label_relative].mean(),
        )
        left_area_tmax, right_area_tmax = (
            tmax_array[brain_area_array == label_origin].mean(),
            tmax_array[brain_area_array == label_relative].mean(),
        )

        # cbf 趋势
        if left_area_cbf == right_area_cbf:
            origin_cbf, relative_cbf = "equal", "equal"
        elif left_area_cbf > right_area_cbf:
            origin_cbf, relative_cbf = "up", "down"
        else:
            origin_cbf, relative_cbf = "down", "up"

        # cbv 趋势
        if left_area_cbv == right_area_cbv:
            origin_cbv, relative_cbv = "equal", "equal"
        elif left_area_cbv > right_area_cbv:
            origin_cbv, relative_cbv = "up", "down"
        else:
            origin_cbv, relative_cbv = "down", "up"

        # mtt 趋势
        if left_area_mtt == right_area_mtt:
            origin_mtt, relative_mtt = "equal", "equal"
        elif left_area_cbv > right_area_cbv:
            origin_mtt, relative_mtt = "up", "down"
        else:
            origin_mtt, relative_mtt = "down", "up"

        # ttp 趋势
        if left_area_ttp == right_area_ttp:
            origin_ttp, relative_ttp = "equal", "equal"
        elif left_area_ttp > right_area_ttp:
            origin_ttp, relative_ttp = "up", "down"
        else:
            origin_ttp, relative_ttp = "down", "up"

        # tmax 趋势
        if left_area_tmax == right_area_tmax:
            origin_tmax, relative_tmax = "equal", "equal"
        elif left_area_tmax > right_area_tmax:
            origin_tmax, relative_tmax = "up", "down"
        else:
            origin_tmax, relative_tmax = "down", "up"

    elif label_origin in left_brain_label:
        relative_cbf, relative_cbv, relative_mtt, relative_tmax, relative_ttp = (
            "",
            "",
            "",
            "",
            "",
        )
        if left_area_cbf == right_area_cbf:
            origin_cbf = "equal"
        elif left_area_cbf > right_area_cbf:
            origin_cbf = "up"
        else:
            origin_cbf = "down"
        #  cbv
        if left_area_cbv == right_area_cbv:
            origin_cbv = "equal"
        elif left_area_cbv > right_area_cbv:
            origin_cbv = "up"
        else:
            origin_cbv = "down"
        # mtt
        if left_area_mtt == right_area_mtt:
            origin_mtt = "equal"
        elif left_area_mtt > right_area_mtt:
            origin_mtt = "up"
        else:
            origin_mtt = "down"
        # ttp
        if left_area_ttp == right_area_ttp:
            origin_ttp = "equal"
        elif left_area_ttp > right_area_ttp:
            origin_ttp = "up"
        else:
            origin_ttp = "down"
        # tmax
        if left_area_tmax == right_area_tmax:
            origin_tmax = "equal"
        elif left_area_tmax > right_area_tmax:
            origin_tmax = "up"
        else:
            origin_tmax = "down"
    elif label_origin in right_brain_label:
        origin_cbf, origin_cbv, origin_mtt, origin_tmax, origin_ttp = "", "", "", "", ""
        if left_area_cbf == right_area_cbf:
            relative_cbf = "equal"
        elif left_area_cbf > right_area_cbf:
            relative_cbf = "up"
        else:
            relative_cbf = "down"
        #  cbv
        if left_area_cbv == right_area_cbv:
            relative_cbv = "equal"
        elif left_area_cbv > right_area_cbv:
            relative_cbv = "up"
        else:
            relative_cbv = "down"
        # mtt
        if left_area_mtt == right_area_mtt:
            relative_mtt = "equal"
        elif left_area_mtt > right_area_mtt:
            relative_mtt = "up"
        else:
            relative_mtt = "down"
        # ttp
        if left_area_ttp == right_area_ttp:
            relative_ttp = "equal"
        elif left_area_ttp > right_area_ttp:
            relative_ttp = "up"
        else:
            relative_ttp = "down"
        # tmax
        if left_area_tmax == right_area_tmax:
            relative_tmax = "equal"
        elif left_area_tmax > right_area_tmax:
            relative_ttp = "up"
        else:
            relative_ttp = "down"
    brain_area_paras_dict[label_origin]["CBF"] = origin_cbf
    brain_area_paras_dict[relative_origin]["CBF"] = relative_cbf

    brain_area_paras_dict[label_origin]["CBV"] = origin_cbv
    brain_area_paras_dict[relative_origin]["CBV"] = relative_cbv

    brain_area_paras_dict[label_origin]["MTT"] = origin_mtt
    brain_area_paras_dict[relative_origin]["MTT"] = relative_mtt

    brain_area_paras_dict[label_origin]["TTP"] = origin_ttp
    brain_area_paras_dict[relative_origin]["TTP"] = relative_ttp

    brain_area_paras_dict[label_origin]["Tmax"] = origin_tmax
    brain_area_paras_dict[relative_origin]["Tmax"] = relative_tmax
    return brain_area_paras_dict


def get_mismatch_result(
    tmip_rgb_array,
    cbf_gray_array,
    core_infarct_errors,
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
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle + 90, scale=1)
    tmip_rotated_image = cv2.warpAffine(
        src=rgb_array, M=rotate_matrix, dsize=(rows, cols)
    )
    print(core_infarct_errors)
    mask1 = np.zeros((cbf_gray_array.shape), np.uint8)
    mask2 = np.zeros((cbf_gray_array.shape), np.uint8)
    lesion_mask = np.zeros((cbf_gray_array.shape), np.uint8)

    print(type(core_infarct_errors))
    mask1[low_perfusion_errors] = 255
    mask2[core_infarct_errors] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sum_B = get_cc(mask1, kernel)

    sum_B = cv2.dilate(sum_B, kernel, iterations=8)
    sum_B = cv2.erode(sum_B, kernel, iterations=8)
    lesion_mask[np.where(sum_B > 0)] = 1
    tmip_rotated_image[np.where(sum_B > 0)] = color_map[Mismatch[1]["color"]]
    # mask2 腐蚀膨胀
    mask2 = cv2.erode(cv2.dilate(mask2, kernel, iterations=1), kernel, iterations=5)
    mask2 = cv2.dilate(mask2, kernel, iterations=3)
    # 取交集
    thresh2 = cv2.bitwise_and(mask2, sum_B)
    thresh2 = cv2.dilate(thresh2, kernel, iterations=8)
    thresh2 = cv2.erode(thresh2, kernel, iterations=8)
    lesion_mask[np.where(thresh2 > 0)] = 1
    tmip_rotated_image[np.where(thresh2 > 0)] = color_map[Mismatch[0]["color"]]
    # 沿旋转矩阵反向旋转 angle
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-angle - 90, scale=1)
    origin_image = cv2.warpAffine(
        src=tmip_rotated_image, M=rotate_matrix, dsize=(rows, cols)
    )
    mismatch_result["Mismatch" + str(index)] = origin_image
    cv2.imwrite("./result_mismatch.jpg", origin_image)
    return mismatch_result, lesion_mask


def get_rcbf_result(
    tmip_rgb_array,
    cbf_gray_array,
    brain_area_array,
    center,
    angle,
    cbf_normal,
    rCBF,
    index,
    rcbf_result,
    rows,
    cols,
):
    core_infarct_errors = []
    rgb_array = tmip_rgb_array.copy()
    # tmip 沿旋转矩阵旋转angle
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle + 90, scale=1)
    tmip_rotated_image = cv2.warpAffine(
        src=rgb_array, M=rotate_matrix, dsize=(rows, cols)
    )
    # cbf 沿旋转矩阵旋转angle
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle + 90, scale=1)
    rotated_image = cv2.warpAffine(
        src=cbf_gray_array, M=rotate_matrix, dsize=(rows, cols)
    )
    # 镜像翻转
    mirror = np.fliplr(rotated_image)
    mask1 = np.zeros((cbf_gray_array.shape[1], cbf_gray_array.shape[0]), np.uint8)
    mask2 = np.zeros((cbf_gray_array.shape[1], cbf_gray_array.shape[0]), np.uint8)
    mask3 = np.zeros((cbf_gray_array.shape[1], cbf_gray_array.shape[0]), np.uint8)
    lesion_mask = np.zeros((cbf_gray_array.shape[1], cbf_gray_array.shape[0]), np.uint8)
    for rcbf_threshold in rCBF:
        cbf_setting = rcbf_threshold["threshold"]
        color_bgr = color_map[rcbf_threshold["color"]]
        # 原侧值 / 镜像值,TODO: 没有用到正常值这一参数
        temp_cbf = np.divide(
            rotated_image, mirror, out=np.zeros_like(rotated_image), where=mirror > 0
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    start = time.time()
    sum_B = get_cc(mask1, kernel)
    print(f"连通分量加染色: {time.time() - start}")

    sum_B = cv2.dilate(sum_B, kernel, iterations=8)
    sum_B = cv2.erode(sum_B, kernel, iterations=8)
    lesion_mask[np.where(sum_B > 0)] = 1
    tmip_rotated_image[np.where(sum_B > 0)] = color_map[rCBF[0]["color"]]

    thresh2 = cv2.bitwise_and(mask2, sum_B)
    thresh2 = cv2.dilate(thresh2, kernel, iterations=8)
    thresh2 = cv2.erode(thresh2, kernel, iterations=8)
    lesion_mask[np.where(thresh2 > 0)] = 1
    tmip_rotated_image[np.where(thresh2 > 0)] = color_map[rCBF[1]["color"]]

    thresh3 = cv2.bitwise_and(mask3, sum_B)
    thresh3 = cv2.dilate(thresh3, kernel, iterations=8)
    thresh3 = cv2.erode(thresh3, kernel, iterations=8)
    lesion_mask[np.where(thresh3 > 0)] = 1
    tmip_rotated_image[np.where(thresh3 > 0)] = color_map[rCBF[2]["color"]]
    # end = cv2.getTickCount()
    # use_time = (end - start) / cv2.getTickFrequency()
    # 沿旋转矩阵反向旋转 angle
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-angle - 90, scale=1)
    origin_image = cv2.warpAffine(
        src=tmip_rotated_image, M=rotate_matrix, dsize=(rows, cols)
    )
    rcbf_result["rCBF" + str(index)] = origin_image
    cv2.imwrite("./result_rcbf.jpg", origin_image)
    return rcbf_result, core_infarct_errors, lesion_mask.tolist(), cbf_high, cbf_low


def get_rcbv_result(
    tmip_rgb_array,
    cbv_gray_array,
    center,
    angle,
    cbv_normal,
    rCBV,
    index,
    rcbv_result,
    rows,
    cols,
):
    rgb_array = tmip_rgb_array.copy()
    # tmip 沿旋转矩阵旋转angle
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle + 90, scale=1)
    tmip_rotated_image = cv2.warpAffine(
        src=rgb_array, M=rotate_matrix, dsize=(rows, cols)
    )

    # cbv 沿旋转矩阵旋转angle
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle + 90, scale=1)
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
            rotated_image, mirror, out=np.zeros_like(rotated_image), where=mirror > 0
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    sum_B = get_cc(mask1, kernel)
    sum_B = cv2.morphologyEx(sum_B, cv2.MORPH_CLOSE, kernel, iterations=8)
    # sum_B = cv2.dilate(sum_B, kernel, iterations=8)
    # sum_B = cv2.erode(sum_B, kernel, iterations=8)
    tmip_rotated_image[np.where(sum_B > 0)] = color_map[rCBV[0]["color"]]

    thresh2 = cv2.bitwise_and(mask2, sum_B)
    thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel, iterations=8)
    # thresh2 = cv2.dilate(thresh2, kernel, iterations=8)
    # thresh2 = cv2.erode(thresh2, kernel, iterations=8)
    tmip_rotated_image[np.where(thresh2 > 0)] = color_map[rCBV[1]["color"]]

    thresh3 = cv2.bitwise_and(mask3, sum_B)
    thresh3 = cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, kernel, iterations=8)
    # thresh3 = cv2.dilate(thresh3, kernel, iterations=8)
    # thresh3 = cv2.erode(thresh3, kernel, iterations=8)
    tmip_rotated_image[np.where(thresh3 > 0)] = color_map[rCBV[2]["color"]]

    # 沿旋转矩阵反向旋转 angle
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-angle - 90, scale=1)
    origin_image = cv2.warpAffine(
        src=tmip_rotated_image, M=rotate_matrix, dsize=(rows, cols)
    )
    rcbv_result["rCBV_" + str(index)] = origin_image
    # rgb_array=cv2.line(rgb_array,(int(x1), int(y1)), (int(x2), int(y2)),(255,0,0),3)
    cv2.imwrite("./result_rcbv.jpg", origin_image)
    return rcbv_result, lesion_mask.tolist(), cbv_high, cbv_mid, cbv_low


def get_tmax_result(tmip_rgb_array, tmax_gray_array, Tmax, k, b, tmax_result, index):
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
            low_perfusion_errors = np.where((tmax_gray_array > 6))
        if tmax_setting == 8:
            tmax_low = list(error_arr)
            mask2[error_arr] = 255
        if tmax_setting == 10:
            tmax_high = list(error_arr)
            mask3[error_arr] = 255
        if tmax_setting == 4:
            continue
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sum_B = get_cc(mask1, kernel)
    # 每个mask做闭运算TODO: 改成一行 close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    sum_B = cv2.dilate(sum_B, kernel, iterations=8)
    sum_B = cv2.erode(sum_B, kernel, iterations=8)
    lesion_mask[np.where(sum_B > 0)] = 1
    rgb_array[np.where(sum_B > 0)] = color_map[Tmax[1]["color"]]

    thresh2 = cv2.bitwise_and(mask2, sum_B)
    thresh2 = cv2.dilate(thresh2, kernel, iterations=8)
    thresh2 = cv2.erode(thresh2, kernel, iterations=8)
    lesion_mask[np.where(thresh2 > 0)] = 1
    rgb_array[np.where(thresh2 > 0)] = color_map[Tmax[2]["color"]]

    thresh3 = cv2.bitwise_and(mask3, sum_B)
    thresh3 = cv2.dilate(thresh3, kernel, iterations=8)
    thresh3 = cv2.erode(thresh3, kernel, iterations=8)
    lesion_mask[np.where(thresh3 > 0)] = 1
    rgb_array[np.where(thresh3 > 0)] = color_map[Tmax[3]["color"]]

    tmax_result["tmax_" + str(index)] = rgb_array
    cv2.imwrite("./result_tmax.jpg", rgb_array)
    return tmax_result, low_perfusion_errors, lesion_mask.tolist(), tmax_high, tmax_low


def main():
    for mask_file in os.listdir(mask_path):
        if mask_file == "CBF.nii.gz":
            sitk_img = sitk.ReadImage(os.path.join(mask_path, mask_file))
            origin = sitk_img.GetOrigin()
            spacing = sitk_img.GetSpacing()
            direction = sitk_img.GetDirection()

            mask_array = sitk.GetArrayFromImage(sitk_img)
            print(type(mask_array))
        if mask_file == "CBV.nii.gz":
            sitk_img = sitk.ReadImage(os.path.join(mask_path, mask_file))
            cbv_array = sitk.GetArrayFromImage(sitk_img)
        if mask_file == "TMIP.nii.gz":
            pass

    Mismatch = [
        {"type": "Mismatch", "compare": "", "threshold": "", "color": "#FFFF00"},
        {"type": "rCBF", "compare": "lt", "threshold": "30", "color": "#FF5959"},
        {"type": "Tmax", "compare": "gt", "threshold": "6", "color": "#26CF70"},
    ]

    brain_array = sitk.GetArrayFromImage(
        sitk.ReadImage(
            "/media/tx-deepocean/Data/DICOMS/demos/CN010023-000070191600-939-301/brain_area_mask.nii.gz"
        )
    )
    print(
        f"brain_array min: {brain_array.min()}, max: {brain_array.max()}, shape: {brain_array.shape}"
    )
    tmax_array = sitk.GetArrayFromImage(
        sitk.ReadImage(
            "/media/tx-deepocean/Data/DICOMS/demos/CN010023-000070191600-939-301/TMAX.nii.gz"
        )
    )
    print(f"tmax min: {tmax_array.min()}, max: {tmax_array.max()}")
    print(f"cbf_array shape: {mask_array.shape}")
    tmip_array = sitk.GetArrayFromImage(
        sitk.ReadImage(
            "/media/tx-deepocean/Data/DICOMS/demos/CN010023-000070191600-939-301/TMIP_NO_SKULL.nii.gz"
        )
    )
    mtt_array = sitk.GetArrayFromImage(
        sitk.ReadImage(
            "/media/tx-deepocean/Data/DICOMS/demos/CN010023-000070191600-939-301/MTT.nii.gz"
        )
    )
    ttp_array = sitk.GetArrayFromImage(
        sitk.ReadImage(
            "/media/tx-deepocean/Data/DICOMS/demos/CN010023-000070191600-939-301/TTP.nii.gz"
        )
    )
    print(f"tmip_array shape: {tmip_array.shape}")
    centerline = {
        "point": [2.252883680600437, 139.28484847107637, 229.28803231304485],
        "vector": [-0.9963102262847774, -0.0819163770183483, -0.02560547161376709],
    }
    image_result, ctp_lesion = get_mismatch(
        mask_array,
        tmip_array,
        tmax_array,
        cbv_array,
        mtt_array,
        ttp_array,
        brain_array,
        Mismatch,
        centerline,
        origin,
        spacing,
        direction,
    )

    print(image_result.keys())


if __name__ == "__main__":
    main()
