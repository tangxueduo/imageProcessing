import json
import os

# import cc3d
import cv2
import numpy as np
import pytest
import requests
import SimpleITK as sitk
from loguru import logger

from utils.constants import CONTOUR

CAC_LABEL_MAP = {"RCA": 3, "LM": 4, "LAD": 5, "LCX": 6}


def read_series(series_dir: str):
    series_reader = sitk.ImageSeriesReader()
    dicom_names = series_reader.GetGDCMSeriesFileNames(series_dir)
    series_reader.SetFileNames(dicom_names)
    cac_image = series_reader.Execute()
    return cac_image


def contours_in(img, contour, label=255):
    """轮廓内的点"""
    contour = np.array(contour)
    contour = contour.reshape([contour.shape[0], 1, contour.shape[1]])
    logger.warning(f"contour shape: {contour.shape}")

    cv2.drawContours(img, np.array([contour]), -1, 255, -1)
    # cv2.drawContours(img, np.array([contour[0]]), -1, 255, -1)
    cv2.imwrite("./test.png", img)
    coords = np.where(img == 255, img, 0).astype("bool")
    logger.warning(f"coords shape: {np.unique(coords)}")
    # bool　值
    return coords


def update_cac_one_slice(
    cac_hu: np.ndarray,
    cac_seg_mask: np.ndarray,
    slice_id: int,
    vessel_name: str,
    contours: list,
    hu_threshold: list = [130, "/"],
    add: bool = False,
):
    """tool to update cac mask label by contour"""
    hu_slice_mask = cac_hu[slice_id, :, :]
    seg_slice_mask = cac_seg_mask[slice_id, :, :]
    # 获取上限下限钙化 hu
    min_hu = hu_threshold[0]
    max_hu = float(hu_threshold[1]) if hu_threshold[1] != "/" else np.max(cac_hu)
    logger.warning(f"min_hu: {min_hu}, max_hu: {max_hu}")
    # cac_seg_mask[slice_id,:,:] = hu_slice_mask
    coords = contours_in(seg_slice_mask, contours)
    # 过滤不在范围内的 hu
    true_cac_hu = np.where(
        ((hu_slice_mask * coords).astype("bool"))
        & (hu_slice_mask > min_hu)
        & (hu_slice_mask < max_hu),
        hu_slice_mask,
        0,
    ).astype("bool")

    logger.warning(f"****true_cac_hu: {np.unique(true_cac_hu)}")
    # 如果没有有效hu,不修改
    if np.all(true_cac_hu == False):  # noqa:E712
        return cac_seg_mask, "error"
    if add:
        seg_slice_mask[true_cac_hu] = CAC_LABEL_MAP[vessel_name]
        logger.warning(f"add success")
    else:
        seg_slice_mask[true_cac_hu] = 0
    # 单层编辑后替换
    cac_seg_mask[slice_id, :, :] = seg_slice_mask
    return cac_seg_mask, "success"


@pytest.fixture
def prepare_data():
    # 读取斑块分析mask
    root_dir = "/media/tx-deepocean/Data/DICOMS/demos/ct_heart"
    cac_seg_path = os.path.join(root_dir, "cac_seg.nii.gz")
    cac_seg = sitk.ReadImage(cac_seg_path)

    # 获取钙化积分结果
    url = "http://172.16.4.5:3333/series/1.3.12.2.1107.5.1.4.74241.30000021111501114406300320302/predict/ct_heart_cac_score"
    treeline_url = "http://172.16.4.5:3333/series/1.3.12.2.1107.5.1.4.74241.30000021111501114406300320481/predict/ct_heart_treeline"
    cac_res = requests.get(url).json()
    # 读取treeline
    treeline = requests.get(treeline_url).json()
    # 获取钙化积分原图
    cac_series_dir = os.path.join(
        root_dir, "1.3.12.2.1107.5.1.4.74241.30000021111501114406300320302/"
    )
    cac_image = read_series(cac_series_dir)
    return dict(
        cac_image=cac_image,
        cac_res=cac_res,
        cac_seg=cac_seg,
        treeline=treeline["treeLines"]["data"],
    )


def find_contours(cac_seg_mask, slice_id, vessel_name):
    seg_mask = cac_seg_mask[slice_id, :, :]
    binary_arr = np.zeros_like(seg_mask)
    binary_arr[seg_mask == CAC_LABEL_MAP[vessel_name]] = 1

    contours, _ = cv2.findContours(binary_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    res = []
    if not contours:
        return res
    for contour in contours:
        res.append(contour.reshape(contour.shape[0], contour.shape[2]).tolist())
    return res


def edit_cacs(
    cac_image: sitk.Image,
    cac_seg_mask: np.ndarray,
    cac_res: dict,
    vessel_mapping: dict,
    vessel: str,
    layer_action: str,
    operation: str,
    hu_threshold: list,
    contour: list,
    slice_id: int,
):
    """tool to edit cacs mask and calculate cac score."""
    res = {}
    spacing = cac_res["summary"]["spacing"]
    cac_hu = sitk.GetArrayFromImage(cac_image)
    vessel_name = vessel_mapping[vessel]
    # 单层编辑, 将层面对应label 先置为0，　然后再根据contour　填充对应的label值
    if layer_action == "single":
        # 更新层面contour
        cac_seg_mask, _ = update_cac_one_slice(
            cac_hu,
            cac_seg_mask,
            slice_id,
            vessel_name,
            contour,
            hu_threshold,
            add=operation == "add",
        )
        # 计算contour
        contours = find_contours(cac_seg_mask, slice_id, vessel_name)
    # 多层编辑，找连通域 for　层做
    else:
        # 从当前位置向左右 直到无hu值，即无contour, 单向检索结束　break
        cur_idx = slice_id
        length = cac_seg_mask.shape[0]
        # 左　越界或者无 contour , 结束
        for slice_id in range(cur_idx, 0, -1):
            if slice_id < 0:
                break
            # filterhu
            cac_seg_mask, status = update_cac_one_slice(
                cac_hu,
                cac_seg_mask,
                slice_id,
                vessel_name,
                contour,
                hu_threshold,
                add=operation == "add",
            )
            if status == "error":
                break
            # 计算contour
            contours = find_contours(cac_seg_mask, slice_id, vessel_name)

        # 右　越界或者无contour 结束
        for slice_id in range(cur_idx, length):
            if slice_id == length:
                break
            cac_seg_mask, status = update_cac_one_slice(
                cac_hu,
                cac_seg_mask,
                slice_id,
                vessel_name,
                contour,
                hu_threshold,
                add=operation == "add",
            )
            if status == "error":
                break
            # 计算contour
            contours = find_contours(cac_seg_mask, slice_id, vessel_name)

    # 计算数值
    hu_slice_mask = cac_hu[slice_id, :, :]
    pixel_area = hu_slice_mask[hu_slice_mask == CAC_LABEL_MAP[vessel_name]].sum()
    max_hu = np.max(
        hu_slice_mask[hu_slice_mask == CAC_LABEL_MAP[vessel_name]], initial=-9999
    )
    avg_hu = np.nanmean(hu_slice_mask[hu_slice_mask == CAC_LABEL_MAP[vessel_name]])
    # TODO 根据max_hu 计算
    agatston_pixel_area = [0] * 4  # 物理面积
    agatston_pixel_area[1] = 2.34
    if np.nonzero(agatston_pixel_area)[0].size == 0:
        weight_idx = 0
    else:
        weight_idx = np.nonzero(agatston_pixel_area)[0][0]
    weight = cac_res["summary"]["score"][weight_idx]
    agatston_score = pixel_area * weight * spacing[2] / 3

    res = {
        "contour": contours,
        "sliceId": slice_id,
        "z": cac_image.TransformIndexToPhysicalPoint([0, 0, slice_id])[2],
        "pixelArea": pixel_area.tolist(),  # 像素面积
        "maxHU": max_hu.tolist(),
        "avgHU": avg_hu.tolist() if not np.isnan(avg_hu) else -9999,
        "agatstonPixelArea": agatston_pixel_area,
        "agatston_score": [agatston_score, agatston_score],  # 当前层面的钙化积分
    }
    return res


def get_contour(mask_np, slice_idx, plaque_label):
    rgb_np = np.repeat(mask_np[slice_idx, :, :][..., None], 3, axis=-1)
    cv2.ellipse(rgb_np, (200, 200), (60, 30), 0, 0, 360, (0, 0, 255), -1, 8)

    # slice_arr = mask_np[slice_idx,:,:]
    # rgb_np = np.repeat(slice_arr[..., None], 3, axis=-1)

    # binary_arr = np.zeros(slice_arr.shape).astype("uint8")
    # binary_arr[slice_arr == plaque_label] = 1

    gray = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2GRAY)
    ret, binary_arr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=5)
    contours, hierarchy = cv2.findContours(
        binary_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    logger.info(f"*****len contour: {len(contours)}")
    return contours


def test_cacs_edit(prepare_data):
    cac_res, cac_seg = prepare_data["cac_res"], prepare_data["cac_seg"]
    cac_image = prepare_data["cac_image"]
    treeline = prepare_data["treeline"]
    vessel_mapping = {k: v for k, v in treeline["vesselMapping"].items()}
    contour_np = np.zeros(sitk.GetArrayFromImage(cac_image).shape, dtype="uint8")
    contour_np[2, 200:240, 200:250] = 10
    contour = get_contour(contour_np, 2, 10)

    # 真实　contour
    contour = CONTOUR
    res = edit_cacs(
        cac_image=cac_image,
        cac_seg_mask=sitk.GetArrayFromImage(cac_seg),
        cac_res=cac_res,
        vessel_mapping=vessel_mapping,
        vessel="vessel1",
        layer_action="single",
        operation="add",
        hu_threshold=[130, "/"],
        contour=contour,
        slice_id=2,
    )
    json.dumps(res)
