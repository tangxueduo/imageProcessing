import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import SimpleITK as sitk
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from .constans import BRAIN_AREA_MAP

mask_path = "/media/tx-deepocean/Data/DICOMS/demos/output_new_data_2d/1.3.46.670589.33.1.63790032426654424200002.5278662375047719733"
dcm_file = "/media/tx-deepocean/Data/DICOMS/demos/TMAX/TMAX023.dcm"
ds = pydicom.read_file(dcm_file, force=True)
mask_info_map = {
    "CBV": {
        "min_pixel": 0,
        "max_pixel": 30,
        "ww": 30,
        "wl": 15,
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
        "min_pixel": 0.01,
        "max_pixel": 8.65,
        "ww": 8.64,
        "wl": 4.33,
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
        "min_pixel": -1,
        "max_pixel": 16.79,
        "ww": 17.79,
        "wl": 7.9,
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
        "min_pixel": 3.33,
        "max_pixel": 28.72,
        "ww": 16.03,
        "wl": 25.39,
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


def get_images(image_type: str, mask_array: np.ndarray):
    """生成伪彩图、tMIP、tAverage、mismatch 视图、rCBF、rCBV、tmax
    Args:
    Return:
        dict(str, np.ndarray)
    """
    result = {}
    # if image_type in ["CBF", "CBV", "TMAX", "TTP", "MTT"]:
    if image_type in ["CBF", "MTT", "TTP", "TMAX"]:
        # pass
        # 确定 pixel value 上限 下限
        mask_array = np.clip(
            mask_array,
            mask_info_map[image_type]["min_pixel"],
            mask_info_map[image_type]["max_pixel"],
        )
        # 归一化 y=(x-MinValue)/(MaxValue-MinValue)
        mask_array = (mask_array - mask_array.min()) / (
            mask_array.max() - mask_array.min()
        )
        mask_array_unique = np.unique(mask_array)

        thresholds = np.percentile(
            mask_array_unique, mask_info_map[image_type]["thresholds"]
        ).tolist()

        cbv_rgb_list = mask_info_map[image_type]["rgb_list"]

        for level in range(len(cbv_rgb_list)):
            for rgb_value in range(len(cbv_rgb_list[level])):
                cbv_rgb_list[level][rgb_value] /= 255
        # print(cbv_rgb_list)

        my_cmap = LinearSegmentedColormap.from_list(
            "my_cmap", list(zip(thresholds, cbv_rgb_list))
        )
        index = 0
        for dcm_slice in range(len(mask_array)):
            # if dcm_slice == 1:
            plt.figure(figsize=(5.12, 5.12), facecolor="#000000")
            plt.axis("off")
            im = plt.imshow(mask_array[dcm_slice], cmap=plt.get_cmap(my_cmap))
            print(type(im))
            # 显示色读条
            # plt.colorbar(im)
            plt.savefig("./model_map.jpg")
            with io.BytesIO() as buffer:
                plt.savefig(buffer, format="jpg")
                buffer.seek(0)
                image = Image.open(buffer)
                ar = np.array(image)
            # print(ar.shape)
            np_array_to_dcm(
                ds,
                ar.astype(np.uint8),
                f"./model_map{str(index)}.dcm",
                ww=mask_info_map[image_type]["ww"],
                wl=mask_info_map[image_type]["wl"],
                is_rgb=True,
            )
            index += 1

    # tMIP 和 tAverage 需要更改 array dtype
    index = 0
    if (
        image_type == "tAve_NO_SKULL"
        or image_type == "tAve_WITH_SKULL"
        or image_type == "TMIP_NO_SKULL"
        or image_type == "CBV"
    ):
        # TODO: cbv 时注释这行
        mask_array = mask_array.astype(np.int16)
        for dcm_slice in range(mask_array.shape[0]):
            # 发布时删除下面 if
            if dcm_slice == 12:
                dcm_2d_array = mask_array[dcm_slice, :, :]
                ww = mask_info_map[image_type]["ww"]
                wl = mask_info_map[image_type]["wl"]
                np_array_to_dcm(
                    ds, dcm_2d_array, f"./test{str(index)}.dcm", ww=ww, wl=wl
                )
            index += 1
    return result


def get_mismatch(cbf_arrray: np.ndarray, config: list):
    """
    rCBF计算公式:首先计算出每个点的CBF, 该点的CBF与正常值对比,
    如果小于后台设置的值那么显示该点异常，如果该点与正常值对比为非异常，那么与大脑对侧做对比。
    如果大脑对侧没有对应的值，那么只是显示与正常值对比结果。
    """
    print(config)

    # 低灌注区域
    low_perfusion_area = 1000
    core_infarct_area = 1000
    mismatch_ratio = 1.00000  # TODO: 保留几位
    # 半暗带体积
    mismatch_volume = 17777

    print(1111)


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
        ds.WindowWidth = 255
        ds.WindowCenter = 127
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
    for mask_file in os.listdir(mask_path):
        if mask_file == "CBF.nii.gz":
            mask_array = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(mask_path, mask_file))
            )
            print(type(mask_array))
        if mask_file == "TMIP.nii.gz":
            pass
            # mask_array = sitk.GetArrayFromImage(sitk.ReadImage("/media/tx-deepocean/Data/DICOMS/demos/data/TMIP_NO_SKULL.nii.gz"))
    # image_result = get_images("CBV", mask_array)
    cbf_normal = 50
    Mismatch = [
        {"type": "Mismatch", "compare": "", "threshold": "", "color": "#FFFF00"},
        {"type": "rCBF", "compare": "lt", "threshold": "30", "color": "#FF5959"},
        {"type": "Tmax", "compare": "gt", "threshold": "6", "color": "#26CF70"},
    ]

    brain_array = sitk.GetArrayFromImage(
        sitk.ReadImage("/media/tx-deepocean/Data/DICOMS/demos/segments_label.nii.gz")
    )
    image_result = get_mismatch(mask_array, Mismatch)

    print(image_result.keys())


if __name__ == "__main__":
    main()
