import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pydicom
import SimpleITK as sitk
from constans import BRAIN_AREA_MAP
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

mask_path = "/media/tx-deepocean/Data/DICOMS/demos/CN010023-000070191600-939-301"
dcm_file = "/media/tx-deepocean/Data/DICOMS/demos/TMAX/TMAX023.dcm"
ds = pydicom.read_file(dcm_file, force=True)

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
    result = {}
    # if image_type in ["CBF", "CBV", "TMAX", "TTP", "MTT"]:
    if image_type in ["CBV", "CBF", "MTT", "TTP", "TMAX"]:
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

        for level, rgb_value_list in enumerate(cbv_rgb_list):
            print(level)
            cbv_rgb_list[level] = [i / 255 for i in rgb_value_list]
        print(cbv_rgb_list)

        my_cmap = LinearSegmentedColormap.from_list(
            "my_cmap", list(zip(thresholds, cbv_rgb_list))
        )
        index = 0
        for dcm_slice in range(len(mask_array)):
            if dcm_slice == 10:
                plt.figure(figsize=(5.12, 5.12), facecolor="#000000")
                plt.axis("off")
                im = plt.imshow(mask_array[dcm_slice], cmap=plt.get_cmap(my_cmap))
                print(type(im))
                # 显示色读条
                plt.colorbar(im)
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


def get_mismatch(
    cbf_array: np.ndarray,
    tmip_array: np.ndarray,
    tmax_array: np.ndarray,
    config: list,
    centerline: dict,
    origin,
    spacing,
    direction,
):
    """
    rCBF计算公式:首先计算出每个点的CBF, 该点的CBF与正常值对比,
    如果小于后台设置的值那么显示该点异常，如果该点与正常值对比为非异常，那么与大脑对侧做对比。
    如果大脑对侧没有对应的值，那么只是显示与正常值对比结果。
    """
    print(f"cbf_array min: {cbf_array.min()}, max: {cbf_array.max()}")

    result = {}
    cbf_setting = 30  # cbf 配置值
    cbf_normal = 50  # cbf 正常值

    # 1、区分左右侧大脑:点在直线左、右侧
    # 点法式 转 标准式
    A, B, C, D = convert_noraml_to_standard_plane(
        centerline["point"], centerline["vector"]
    )

    for dcm_slice in range(cbf_array.shape[0]):
        if dcm_slice == 10:
            rows, cols = cbf_array.shape[1:]
            # 由两点式确定直线方程 ，由于原点为左上，取Ax+By+C=0关于x轴对称的直线方程:Ax-By+C=0
            # (x1, y1), (x2, y2) = get_point_on_plane_by_yz(
            #     A, B, C, D, rows, 0, 15, spacing
            # )
            (x1, y1), (x2, y2) = (282, 81), (226, 426)
            # Ax+By+C=0, A=(y2-y1), B=(x1-x2), C=x2*y1 - x1*y2, D= A*x + By +C, 大于0在右侧，小于0 在左侧
            # 注意采用x=ky+b模型，原因是如果中线没有偏转，例x=10这种情况，x=ky+b不用考虑这种特殊情况，不用额外写个if条件判断
            k = (x2 - x1) / (y2 - y1)  # 计算中线斜率k
            b = x1 - k * y1  # 计算b
            # 不知道为什么要关于 y=x 做个对称

            b = -b / k
            k = 1 / k

            cbf_gray_array = cbf_array[dcm_slice, :, :]
            tmip_gray_array = tmip_array[dcm_slice, :, :]
            # gray to rgb
            tmip_rgb_array = gray2rgb_array(tmip_gray_array)

            # 获取 rcbf 视图， 小于0.2红色，0.2>=x<0.3黄色， 0.3>=x<0.4绿色

            error_arr = np.argwhere(
                (cbf_gray_array > 0)
                & ((cbf_gray_array / cbf_normal) < cbf_setting / 100)
            )

            print(error_arr)
            if error_arr.size != 0:
                idx_bottom = np.argwhere(
                    (error_arr[:, 0] - k * error_arr[:, 1] - b) > 0
                )  # 在异常点位置数组中判断哪些点的索引在中线下方
                idx_top = np.argwhere(
                    (error_arr[:, 0] - k * error_arr[:, 1] - b) <= 0
                )  # 在异常点位置数组中判断哪些点的索引在中线上方
                etl = idx_top[:, 0]  # 因 argwhere输出二维数组，第二列全是0,只提取第一列索引
                ebr = idx_bottom[:, 0]  # 同上
                print(error_arr[etl])

                # 将在中线上方的异常点像素值置为[0,255,0]
                tmip_rgb_array[error_arr[etl][:, 0], error_arr[etl][:, 1], 0] = 89
                tmip_rgb_array[error_arr[etl][:, 0], error_arr[etl][:, 1], 1] = 89
                tmip_rgb_array[error_arr[etl][:, 0], error_arr[etl][:, 1], 2] = 255

                # 将在中线下方的异常点像素值改为
                tmip_rgb_array[error_arr[ebr][:, 0], error_arr[ebr][:, 1], 0] = 0
                tmip_rgb_array[error_arr[ebr][:, 0], error_arr[ebr][:, 1], 1] = 255
                tmip_rgb_array[error_arr[ebr][:, 0], error_arr[ebr][:, 1], 2] = 0

            rgb_array = cv2.line(
                tmip_rgb_array, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3
            )
            cv2.imwrite("./result.jpg", tmip_rgb_array)

            # 获取 rcbv 视图, 小于0.45绿色， 小于0.4黄色， 小于0.35红色
            rcbv_result = get_mismatch_rcbv()
            # 获取 tmax 视图
            tmax_result = get_mismatch_tmax()

    print(11111111111111111)
    # rcbv = rcbf * MTT

    # Tmax

    # 111111111111111111111111111111111111111111111111
    # 低灌注区域 将6s＜Tmax的区域， rMTT＞145%
    low_perfusion_area = 1000
    # 核心梗死区 将Tmax＞10s、rCBF＜30%对侧正常脑组织、绝对值：CBV＜2ml/100g
    core_infarct_area = 1000

    # mismatch 比率: 低灌注体积/核心梗死体积
    mismatch_ratio = round(low_perfusion_area / core_infarct_area, 1)
    # 半暗带体积: 低灌注体积-核心梗死体积
    mismatch_volume = low_perfusion_area - core_infarct_area

    print(1111)
    return result


def split(arr, cond):
    return [arr[cond], arr[~cond]]


def convert_noraml_to_standard_plane(point, vector):
    """点法式平面方程转为标准式方程"""
    A = vector[0]
    B = vector[1]
    C = vector[2]
    D = -(np.dot(vector, point))
    return A, B, C, D


def get_point_on_plane_by_yz(A, B, C, D, height, start_index, end_index, spacing):
    """目前这一实现为前端方案：
    由平面方程获取直线方程
    产品约定: 平面 size 取为距原图上下各 10% 大小
    """
    # 由 Ax + By + Cz + D = 0 推出， x = - (D+By+Cz)/A
    y_bottom, y_top = height * 0.1, height * 0.9

    z_bottom, z_top = start_index * spacing[2], end_index * spacing[2]
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
    # 矩阵计算请尽量用numpy自己的库函数
    temp_array = np.clip(temp_array, true_min_pt, true_max_pt)
    min_pt_array = np.ones((temp_array.shape[0], temp_array.shape[1])) * true_min_pt
    temp_array = (temp_array - min_pt_array) * scale

    rgb_array = np.zeros((temp_array.shape[0], temp_array.shape[1], 3))
    rgb_array[:, :, 0] = temp_array
    rgb_array[:, :, 1] = temp_array
    rgb_array[:, :, 2] = temp_array

    return rgb_array


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
    for mask_file in os.listdir(mask_path):
        if mask_file == "CBV.nii.gz":
            sitk_img = sitk.ReadImage(os.path.join(mask_path, mask_file))
            origin = sitk_img.GetOrigin()
            spacing = sitk_img.GetSpacing()
            direction = sitk_img.GetDirection()

            mask_array = sitk.GetArrayFromImage(sitk_img)
            print(type(mask_array))
        if mask_file == "TMIP.nii.gz":
            pass
            # mask_array = sitk.GetArrayFromImage(sitk.ReadImage("/media/tx-deepocean/Data/DICOMS/demos/data/TMIP_NO_SKULL.nii.gz"))
    image_result = get_images("CBV", mask_array)
    # Mismatch = [
    #     {"type": "Mismatch", "compare": "", "threshold": "", "color": "#FFFF00"},
    #     {"type": "rCBF", "compare": "lt", "threshold": "30", "color": "#FF5959"},
    #     {"type": "Tmax", "compare": "gt", "threshold": "6", "color": "#26CF70"},
    # ]

    # brain_array = sitk.GetArrayFromImage(
    #     sitk.ReadImage(
    #         "//media/tx-deepocean/Data/DICOMS/demos/CN010023-000070191600-939-301/brain_mask.nii.gz"
    #     )
    # )
    # tmax_array = sitk.GetArrayFromImage(
    #     sitk.ReadImage(
    #         "/media/tx-deepocean/Data/DICOMS/demos/output_new_data_2d/1.3.46.670589.33.1.63790032426654424200002.5278662375047719733/TMAX.nii.gz"
    #     )
    # )
    # print(f"tmax min: {tmax_array.min()}, max: {tmax_array.max()}")
    # print(f"cbf_array shape: {mask_array.shape}")
    # tmip_array = sitk.GetArrayFromImage(
    #     sitk.ReadImage(
    #         "//media/tx-deepocean/Data/DICOMS/demos/CN010023-000070191600-939-301/TMIP_NO_SKULL.nii.gz"
    #     )
    # )
    # print(f"tmip_array shape: {tmip_array.shape}")
    # centerline = {
    #     "point": [
    #         2.252883680600437,
    #         139.28484847107637,
    #         229.28803231304485
    #     ],
    #     "vector": [
    #         -0.9963102262847774,
    #         -0.0819163770183483,
    #         -0.02560547161376709
    #     ],
    # }
    # image_result = get_mismatch(
    #     mask_array, tmip_array, tmax_array, Mismatch, centerline, origin, spacing, direction
    # )

    print(image_result.keys())


if __name__ == "__main__":
    main()
