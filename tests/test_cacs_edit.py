import json
import math
import os

# import cc3d
import cv2
import numpy as np
import pytest
import requests
import SimpleITK as sitk
from loguru import logger

from utils.constants import (
    BACKGROUND_HU,
    CONTOUR,
    DEFAULT_ERROR_STATUS,
    DEFAULT_SUCCESS_STATUS,
)

CAC_LABEL_MAP = {"vessel9": 3, "vessel1": 4, "vessel2": 5, "vessel5": 6}


def read_series(series_dir: str):
    series_reader = sitk.ImageSeriesReader()
    dicom_names = series_reader.GetGDCMSeriesFileNames(series_dir)
    series_reader.SetFileNames(dicom_names)
    cac_image = series_reader.Execute()
    return cac_image


@pytest.fixture
def prepare_data():
    # 读取斑块分析mask
    root_dir = "/media/tx-deepocean/Data/DICOMS/demos/ct_heart"
    cac_seg_path = os.path.join(root_dir, "cac_seg.nii.gz")
    cac_seg = sitk.ReadImage(cac_seg_path)

    # 获取钙化积分结果
    url = "http://10.0.70.3:3333/series/1.2.840.113704.7.32.07.5.1.4.76346.30000021051400263961500137608/predict/ct_heart_cac_score"
    # treeline_url = "http://10.0.70.3:3333/series/1.3.12.2.1107.5.1.4.76346.30000021051400263961500137703/feedback/ct_heart_treeline"
    cac_res = requests.get(url).json()
    # 获取钙化积分原图
    cac_series_dir = os.path.join(
        root_dir, "1.2.840.113704.7.32.07.5.1.4.76346.30000021051400263961500137608"
    )
    cac_image = read_series(cac_series_dir)
    return dict(
        cac_image=cac_image,
        cac_res=cac_res,
        cac_seg=cac_seg,
    )


def update_cacs_one_slice(
    cac_hu: np.ndarray,
    cac_seg_mask: np.ndarray,
    slice_id: int,
    vessel_label: int,
    contours: list,
    hu_threshold: list = [130, "/"],
    add: bool = False,
):
    """tool to update cac mask label by contour"""
    hu_slice_mask = cac_hu[slice_id, :, :]
    seg_slice_mask = cac_seg_mask[slice_id, :, :]
    # for idx in (43, 44, 45):
    #     cac_seg_mask[cac_seg_mask==3] = 255
    #     cv2.imwrite(f"./{idx}_mask.png", cac_seg_mask[idx,:,:])
    # 获取上限下限钙化 hu
    min_hu = hu_threshold[0]
    max_hu = float(hu_threshold[1]) if hu_threshold[1] != "/" else np.max(cac_hu)
    logger.warning(f"****寻找轮廓内点slice_id: {slice_id}")
    coords = contours_in(seg_slice_mask, contours)
    # tmp = cac_hu[43, :, :]
    # tmp[coords] = 255
    # cv2.imwrite("./hu.png", tmp)

    # 减小操作，根据contour大小给 mask 置0
    if not add:
        seg_slice_mask[coords] = 0
    else:
        # 过滤不在范围内的 hu
        true_cac_hu = np.where(
            ((hu_slice_mask * coords).astype("bool"))
            & (hu_slice_mask > min_hu)
            & (hu_slice_mask < max_hu),
            hu_slice_mask,
            0,
        ).astype("bool")
        # 无有效hu　不修改
        if np.all(true_cac_hu == False):  # noqa: E712
            logger.warning(f"未找到钙化阈值内的HU true_cac_hu ALL {np.unique(true_cac_hu)}")
            return cac_seg_mask, DEFAULT_ERROR_STATUS, coords
        seg_slice_mask[true_cac_hu] = vessel_label
    # 单层编辑后替换
    cac_seg_mask[slice_id, :, :] = seg_slice_mask
    return cac_seg_mask, DEFAULT_SUCCESS_STATUS, coords


def calculate_calcification_integral(
    cacs_res: dict,
    cacs_hu_np: np.ndarray,
    cacs_seg_np: np.ndarray,
    slice_id: int,
    vessel_label: str,
    min_island_size_mm2=1.0,  # 模型根据钙化积分计算指南设定的变量
):
    """tool to get some data to calculate cacs score"""
    cacs_seg_np = cacs_seg_np.copy()
    spacing = cacs_res["summary"]["spacing"]
    hu_mask = cacs_hu_np[slice_id, :, :]
    seg_mask = cacs_seg_np[slice_id, :, :]

    # 计算钙化积分层面数值
    pixel_area = np.count_nonzero(seg_mask == vessel_label)
    avg_hu = np.sum(hu_mask[seg_mask == vessel_label]) / pixel_area
    agatston_score = 0
    agatston_pixel_area = [0, 0, 0, 0]  # 物理面积

    # 因为最初计算来自模型，所以不清楚这个变量意义，可能来自钙化积分计算指南
    phy_pix_area = spacing[0] * spacing[1]
    min_island_size_pixels = int(round(min_island_size_mm2 / phy_pix_area))

    hu_in_mask = hu_mask * (seg_mask == vessel_label)
    max_hu = np.max(
        hu_in_mask,
        initial=BACKGROUND_HU,
    )
    # 计算当前层面钙化hu 物理面积
    if pixel_area >= min_island_size_pixels:
        aga_hu_weight = math.floor(np.max(hu_in_mask) / 100)
        if aga_hu_weight > 4:
            aga_hu_weight = 4.0
        agatston_score = aga_hu_weight * pixel_area * phy_pix_area * (spacing[2] / 3)
        agatston_pixel_area[int(aga_hu_weight - 1)] = pixel_area * phy_pix_area
    return pixel_area, max_hu, avg_hu, agatston_pixel_area, agatston_score


def find_contours(cac_seg_mask, slice_id, vessel_label):
    seg_mask = cac_seg_mask[slice_id, :, :]
    binary_arr = np.zeros(seg_mask.shape, dtype="uint8")
    binary_arr[seg_mask == vessel_label] = 255
    if slice_id == 43:
        cv2.imwrite("./test.png", binary_arr)
    contours, _ = cv2.findContours(binary_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    res = []
    if not contours:
        logger.info(f"{slice_id} 未找到 contour")
        return res

    res = [contour.reshape(contour.shape[0], 2).tolist() for contour in contours]
    return res


def contours_in(img, contour, label=255):
    """轮廓内的点"""
    contour = np.array(contour)
    contour = contour.reshape([contour.shape[0], 1, contour.shape[1]])
    cv2.drawContours(img, np.array([contour]), -1, label, -1)

    cv2.imwrite("./test111.png", img)
    coords = np.where(img == label, img, 0).astype("bool")

    # bool　值
    return coords


def calculate_contour_cacs(
    slice_id: int,
    vessel_label: str,
    contour: list,
    hu_threshold: list,
    cacs_image: sitk.Image,
    cacs_res: dict,
    cacs_hu_np: np.ndarray,
    cacs_seg_np: np.ndarray,
    operation: str = "add",
    to_up: bool = False,
):
    """
    根据轮廓进行 FilterHU -- 计算钙化积分数值 -- update contour
    Return:
        res: dict, 返回体(层面,contour, 其他计算结果等),
        cacs_seg_np: 更新后的钙化积分mask,
        valid_hu_status: 是否有　有效hu, 默认为　"error",
        contour_status: 层面是否有contour, 增加时为当前层 cur_contour_status, 减小时为下一层next_contour_status
    """
    contour_status = DEFAULT_ERROR_STATUS
    cacs_seg_np, valid_hu_status, coords = update_cacs_one_slice(
        cacs_hu_np,
        cacs_seg_np,
        slice_id,
        vessel_label,
        contour,
        hu_threshold,
        add=operation == "add",
    )
    if valid_hu_status == DEFAULT_ERROR_STATUS and operation == "add":
        return None, cacs_seg_np, valid_hu_status, contour_status
    # 计算求取平面钙化积分所需 data
    (
        pixel_area,
        max_hu,
        avg_hu,
        agatston_pixel_area,
        agatston_score,
    ) = calculate_calcification_integral(
        cacs_res, cacs_hu_np, cacs_seg_np, slice_id, vessel_label
    )

    # 计算contour
    contours = find_contours(cacs_seg_np, slice_id, vessel_label)

    agatston_score = float(agatston_score)
    res = {
        "contour": contours,
        "sliceId": slice_id + 1,
        "z": cacs_image.TransformIndexToPhysicalPoint([0, 0, slice_id])[2],
        "pixelArea": pixel_area,  # 像素面积
        "maxHU": max_hu.tolist(),
        "avgHU": avg_hu.tolist() if not np.isnan(avg_hu) else BACKGROUND_HU,
        "agatstonPixelArea": agatston_pixel_area,  # 物理面积
        "agatston_score": [agatston_score, agatston_score],  # 当前层面的钙化积分
    }
    # 如果没有有效hu　/ 无contour ，　增大返回空，减小需要返回完整返回体，且contour为空
    if not contours or valid_hu_status == DEFAULT_ERROR_STATUS:
        if operation == "remove":
            # TODO: 通过找contour的方式更准确判断
            # 多层减小时，判断下一层是否有 contour
            if slice_id - 1 > 0 and to_up:
                next_cac_seg_mask = cacs_seg_np[slice_id - 1, :, :]
            if slice_id + 1 < cacs_seg_np.shape[0] - 1 and not to_up:
                next_cac_seg_mask = cacs_seg_np[slice_id + 1, :, :]

            if np.count_nonzero(next_cac_seg_mask * coords) == 0:
                logger.warning(f"下一层无label 值，不再检索： {slice_id + 1}")
                contour_status = DEFAULT_ERROR_STATUS
            else:
                contour_status = DEFAULT_SUCCESS_STATUS
            return res, cacs_seg_np, valid_hu_status, contour_status
        else:
            return None, cacs_seg_np, valid_hu_status, contour_status
    contour_status = DEFAULT_SUCCESS_STATUS
    return res, cacs_seg_np, valid_hu_status, contour_status


def more_layer_cacs_edit(
    cacs_res,
    cacs_image,
    cacs_seg_np,
    cacs_hu_np,
    res,
    cur_idx,
    vessel,
    vessel_label,
    contour,
    hu_threshold,
    operation,
):
    """钙化积分多层编辑"""
    length = cacs_seg_np.shape[0]

    def calculate_and_append_cacs(slice_id, cacs_seg_np, to_up):
        (
            single_slice_res,
            cacs_seg_np,
            valid_hu_status,
            next_contour_status,
        ) = calculate_contour_cacs(
            slice_id=slice_id,
            vessel_label=vessel_label,
            contour=contour,
            hu_threshold=hu_threshold,
            cacs_image=cacs_image,
            cacs_res=cacs_res,
            cacs_hu_np=cacs_hu_np,
            cacs_seg_np=cacs_seg_np,
            operation=operation,
            to_up=to_up,
        )
        if single_slice_res is not None:
            res[vessel]["contour"]["data"].append(single_slice_res)

        """
        此处更新比较复杂
        多层增大: 如果没有有效hu, 或者无contour 停止更新
        多层减小: 比较特殊, 存在无contour 但是可能后面有有效hu 的特殊情况
                这时判断下一层是否有contour, 如果没有，停止更新。
        """
        # 无有效hu，　一定停止更新
        logger.info(
            f"{slice_id} valid_hu_status: {valid_hu_status}, next_contour_status: {next_contour_status}"
        )
        if (
            valid_hu_status == DEFAULT_ERROR_STATUS
            or next_contour_status == DEFAULT_ERROR_STATUS
        ):
            return False, cacs_seg_np
        else:
            return True, cacs_seg_np

    # 向上检索
    for slice_id in range(cur_idx, -1, -1):
        logger.warning(f"向上检索: {slice_id}")
        has_contour, cacs_seg_np = calculate_and_append_cacs(
            slice_id, cacs_seg_np, to_up=True
        )
        if not has_contour:
            break

    # 向下检索
    for slice_id in range(cur_idx + 1, length):
        logger.warning(f"向下检索: {slice_id}")
        has_contour, cacs_seg_np = calculate_and_append_cacs(
            slice_id, cacs_seg_np, to_up=False
        )
        if not has_contour:
            break
    return res, cacs_seg_np


def edit_cacs(
    vessel,
    slice_id,
    operation,
    layer_action,
    contour,
    cacs_res,
    cacs_image,
    cacs_hu_np,
    cacs_seg_np,
    hu_threshold=[130, "/"],
):
    res = {}
    res[vessel] = {"contour": {"_type": "contours", "data": []}}
    vessel_label = CAC_LABEL_MAP[vessel]
    slice_id = slice_id - 1
    if layer_action == "single":
        single_slice_res, cacs_seg_np, _, _ = calculate_contour_cacs(
            slice_id=slice_id,
            vessel_label=vessel,
            contour=contour,
            hu_threshold=hu_threshold,
            cacs_image=cacs_image,
            cacs_res=cacs_res,
            cacs_hu_np=cacs_hu_np,
            cacs_seg_np=cacs_seg_np,
            operation=operation,
        )
        if single_slice_res is not None:
            res[vessel]["contour"]["data"].append(single_slice_res)
    # 从当前位置向左右 直到无hu值，即无contour, 单向检索结束　break
    else:
        res, cacs_seg_np = more_layer_cacs_edit(
            cacs_res=cacs_res,
            cacs_image=cacs_image,
            cacs_seg_np=cacs_seg_np,
            cacs_hu_np=cacs_hu_np,
            res=res,
            cur_idx=slice_id,
            vessel=vessel,
            vessel_label=vessel_label,
            contour=contour,
            hu_threshold=hu_threshold,
            operation=operation,
        )
    return res


def test_cacs_edit(prepare_data):
    cac_res, cac_seg = prepare_data["cac_res"], prepare_data["cac_seg"]
    cac_image = prepare_data["cac_image"]
    cacs_hu_np = sitk.GetArrayFromImage(cac_image)[::-1]
    cac_seg_mask = sitk.GetArrayFromImage(cac_seg)[::-1]
    slice_id = 44
    # 真实　contour
    res = edit_cacs(
        vessel="vessel9",
        slice_id=slice_id,
        operation="remove",
        layer_action="more",
        contour=CONTOUR,
        cacs_res=cac_res,
        cacs_image=cac_image,
        cacs_seg_np=cac_seg_mask,
        cacs_hu_np=cacs_hu_np,
        hu_threshold=[130, "/"],
    )
