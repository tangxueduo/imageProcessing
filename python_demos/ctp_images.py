import io
import os
import time

import cv2
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pydicom
import SimpleITK as sitk
from constants import BRAIN_AREA_MAP
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# import pandas as pd


mask_path = "/media/tx-deepocean/Data/DICOMS/demos/29"
dcm_file = "/media/tx-deepocean/Data/DICOMS/demos/TMAX/TMAX023.dcm"
ds = pydicom.read_file(dcm_file, force=True)

mask_info_map = {
    "CBV": {
        "min_pixel": 0,
        "max_pixel": 8,
        "ww": 8,
        "wl": 4,
        "rgb_list": [
            [0, 0, 0],
            [0, 0, 128],
            [0, 0, 224],
            [0, 60, 255],
            [0, 152, 255],
            [0, 248, 255],
            [86, 255, 170],
            [178, 255, 78],
            [255, 240, 0],
            [255, 144, 0],
            [255, 52, 0],
            [216, 0, 0],
        ],
        "thresholds": [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    },
    "CBF": {
        "min_pixel": 0,
        "max_pixel": 60,
        "ww": 60,
        "wl": 30,
        "rgb_list": [
            [0, 0, 0],
            [0, 0, 128],
            [0, 0, 224],
            [0, 60, 255],
            [0, 152, 255],
            [0, 248, 255],
            [86, 255, 170],
            [178, 255, 78],
            [255, 240, 0],
            [255, 144, 0],
            [132, 10, 0],
            [132, 0, 0],
        ],
        "thresholds": [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    },
    "TTP": {
        "min_pixel": 0,
        "max_pixel": 16,
        "ww": 16,
        "wl": 8,
        "rgb_list": [
            [0, 0, 0],
            [0, 0, 128],
            [0, 0, 224],
            [0, 60, 255],
            [0, 152, 255],
            [0, 248, 255],
            [86, 255, 170],
            [178, 255, 78],
            [255, 240, 0],
            [255, 144, 0],
            [255, 52, 0],
            [216, 0, 0],
        ],
        "thresholds": [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    },
    "MTT": {
        "min_pixel": 0,
        "max_pixel": 20,
        "ww": 20,
        "wl": 10,
        "rgb_list": [
            [0, 0, 0],
            [0, 0, 128],
            [0, 0, 224],
            [0, 60, 255],
            [0, 152, 255],
            [0, 248, 255],
            [86, 255, 170],
            [178, 255, 78],
            [255, 240, 0],
            [255, 144, 0],
            [255, 52, 0],
            [216, 0, 0],
        ],
        "thresholds": [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    },
    "TMAX": {
        "min_pixel": 0,
        "max_pixel": 12,
        "ww": 12,
        "wl": 6,
        "rgb_list": [
            [0, 0, 0],
            [0, 0, 128],
            [0, 0, 224],
            [0, 60, 255],
            [0, 152, 255],
            [0, 248, 255],
            [86, 255, 170],
            [178, 255, 78],
            [255, 240, 0],
            [255, 144, 0],
            [255, 52, 0],
            [216, 0, 0],
        ],
        "thresholds": [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    },
    "tAve_NO_SKULL": {"ww": 100, "wl": 40},
    "tAve_WITH_SKULL": {"ww": 100, "wl": 40},
    "TMIP_NO_SKULL": {"ww": 100, "wl": 50},
    "TMIP_WITH_SKULL": {"ww": 100, "wl": 50},
}

# bgr 值
color_map = {
    "#FFFF00": [0, 255, 255],
    "#FF5959": [89, 89, 255],
    "#26CF70": [112, 207, 38],
    "#3F87F5": [245, 135, 63],
}


def get_images(image_type: str, mask_array: np.ndarray):
    """生成伪彩图、tMIP、tAverage、mismatch 视图、rCBF、rCBV、tmax
    Args:
    Return:
        dict(str, np.ndarray)
    """
    # t0 = time.time()
    result = {}
    if image_type in ["CBV", "CBF", "MTT", "TTP", "TMAX"]:
        # pass
        # 确定 pixel value 上限 下限
        # mask_array = mask_array.astype(np.int16)

        mask_array = np.clip(
            mask_array,
            mask_info_map[image_type]["min_pixel"],
            mask_info_map[image_type]["max_pixel"],
        )
        # ------归一化 y=(x-MinValue)/(MaxValue-MinValue)
        mask_array = (mask_array - mask_array.min()) / (
            mask_array.max() - mask_array.min()
        )
        # 这个 thresholds 是限制你某个区间段的
        thresholds = np.percentile(
            (np.unique(mask_array)), mask_info_map[image_type]["thresholds"]
        ).tolist()

        # ------ 这个 rgb_list 就是你的0到1 的rgb 值
        cbv_rgb_list = mask_info_map[image_type]["rgb_list"]
        # rgb 值归一
        for level, rgb_value_list in enumerate(cbv_rgb_list):
            cbv_rgb_list[level] = [i / 255 for i in rgb_value_list]

        my_cmap = LinearSegmentedColormap.from_list(
            "my_cmap", list(zip(thresholds, cbv_rgb_list))
        )
        index = 0
        print(len(mask_array))
        # ------ mask_array 是模型给你的mask， 不过你这次我感觉不需要染色，貌似就是crop paste.....
        for dcm_slice in range(len(mask_array)):
            if dcm_slice == 10:
                dcm_2d_array = mask_array[dcm_slice, :, :]
                plt.figure(figsize=(5.12, 5.12), facecolor="#000000")
                plt.axis("off")
                im = plt.imshow(np.where(dcm_2d_array==0, -1, dcm_2d_array), cmap=plt.get_cmap(my_cmap))
                # 显示色读条
                plt.colorbar(im)
                plt.savefig("./model_map.jpg")
                with io.BytesIO() as buffer:
                    plt.savefig(buffer, format="jpg")
                    buffer.seek(0)
                    image = Image.open(buffer)
                    ar = np.array(image)
    # np_array_to_dcm(
    #     ds,
    #     ar.astype(np.uint8),
    #     f"./model_map{str(index)}.dcm",
    #     ww=mask_info_map[image_type]["ww"],
    #     wl=mask_info_map[image_type]["wl"],
    #     is_rgb=True,
    # )
    # index += 1
    # print(time.time() - t0)
    # ----------以下不用看了
    # tMIP 和 tAverage 需要更改 array dtype
    index = 1111
    # if (
    #     image_type == "tAve_NO_SKULL"
    #     or image_type == "tAve_WITH_SKULL"
    #     or image_type == "TMIP_NO_SKULL"
    #     or image_type == "MTT"
    # ):
    #     # TODO: cbv 时注释这行
    #     mask_array = mask_array.astype(np.int16)
    #     for dcm_slice in range(mask_array.shape[0]):
    #         # 发布时删除下面 if
    #         # if dcm_slice == 12:
    #         dcm_2d_array = mask_array[dcm_slice, :, :]
    #         ww = mask_info_map[image_type]["ww"]
    #         wl = mask_info_map[image_type]["wl"]
    #         np_array_to_dcm(
    #             ds,
    #             dcm_2d_array,
    #             f"./img_cp/test{str(index)}.dcm",
    #             ww=ww,
    #             wl=wl,
    #         )
    #         index += 1
    return result


def np_array_to_dcm(
    ds: pydicom.dataset.FileDataset,
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


def main():
    sitk_img = sitk.ReadImage("/media/tx-deepocean/Data/DICOMS/demos/29/MTT.nii.gz")
    mask_array = sitk.GetArrayFromImage(sitk_img)
    image_result = get_images("MTT", mask_array)

    print(image_result.keys())


if __name__ == "__main__":
    main()
