"""
头颈颅内mip demo
"""
# import ctypes
# import json
import math
import os

# import sys
import time
from copy import deepcopy

# import cv2
import numpy as np
import pytest
import SimpleITK as sitk
import vtk
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

# from vtk.util.numpy_support import numpy_to_vtk
from vtkmodules.util.vtkImageExportToArray import vtkImageExportToArray
from vtkmodules.util.vtkImageImportFromArray import vtkImageImportFromArray

from tests.constants import IntracranialVR
from utils.boundbox import boundbox_3d

FONT_SIZE = 18
FONT = ImageFont.truetype("statics/SourceHanSansCN-Normal.ttf", FONT_SIZE)


class MIPPostion:
    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"


MIP_POSITION_MAP = {"axial": "轴状位", "coronal": "冠状位", "sagittal": "矢状位"}
MIP_LOCALS_MAP = {
    "axial": ["A", "P", "R", "L"],
    "coronal": ["S", "I", "R", "L"],
    "sagittal": ["S", "I", "A", "P"],
}
MATRIX_MAP = {
    "axial": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    "sagittal": [0, 0, 1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1],
    "coronal": [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1],
}
output_dir = "./data/output/mip/"


def gen_Intracranial_data(seg_lps_array, hu_volume_lps, mask, with_bone=False):
    res_np = hu_volume_lps.copy()
    res_np[seg_lps_array == 101] = -120  # 去背板
    if not with_bone:
        res_np[seg_lps_array == 80] = -120  # 去骨
    # nobone_array = connect_area_abstraction(nobone_array, mylib)　＃慢，注释
    # 9是什么意思
    seg_label = 9
    intracranial_label_mask = np.zeros(shape=seg_lps_array.shape, dtype=np.bool_)
    intracranial_label_mask[seg_lps_array == seg_label] = True
    [zmin, _, _, _, _, _] = boundbox_3d(intracranial_label_mask)
    mask[:zmin, :, :] = False
    return mask, res_np


def _get_width(tag: str) -> int:
    draw = ImageDraw.Draw(Image.new(mode="L", size=(10, 10)))
    text_width, _ = draw.textsize(tag, font=FONT)
    return text_width


# 已有函数，需要新增字段
def draw_text(np_array: np.ndarray, text: str, p_names=None, locals=None) -> np.ndarray:
    final_height, final_width = np_array.shape
    im_frame = Image.new("L", (final_width, final_height))
    draw = ImageDraw.Draw(im_frame)
    draw.text(
        ((final_width - _get_width(text)) // 2, final_height - 18),
        text,
        fill="#ffffff",
        font=FONT,
    )
    # 自上而下，自左而右添加
    if p_names and locals:
        for p, local in zip(p_names, locals):
            draw.text(
                local,
                p,
                fill="#ffffff",
                font=FONT,
            )

    im_array = np.array(im_frame)
    ww, wc = 800, 300
    slope, inter = 1, 0
    plus_array = deepcopy(im_array)
    plus_array[plus_array > 0] = 2
    plus_array[plus_array == 0] = 1
    plus_array[plus_array > 1] = 0
    new_array = np_array * plus_array
    temp = wc - ww / 2
    im_array = im_array / 255.0 * ww + temp
    im_array[im_array == temp] = inter
    im_array = (im_array - inter) / slope
    add_array = np.array(im_array, np.int16)
    new_array += add_array
    return new_array


def _calculate_width(width: int, height: int):
    aspect = width / height
    x, y = 512, 477

    def round_aspect(number, key):
        return max(min(math.floor(number), math.ceil(number), key=key), 1)

    if x / y >= aspect:
        x = round_aspect(y * aspect, key=lambda n: abs(aspect - n / y))
    return x


def resize_hu(np_array: np.ndarray, final_width=512, final_height=512):
    """Resize np array to (final_width, final_height)"""
    lh, lw = np_array.shape
    if lw != 512 or lh != 512:
        new_width = _calculate_width(lw, lh)
        image = sitk.GetImageFromArray(np_array)
        original_spacing = image.GetSpacing()
        new_spacing = [(lw - 1) * original_spacing[0] / (new_width - 1)] * 2
        new_height = int((lh - 1) * original_spacing[1] / new_spacing[1])
        new_size = [new_width, new_height]

        image = sitk.Resample(
            image,
            new_size,
            sitk.Transform(),
            sitk.sitkLinear,
            image.GetOrigin(),
            new_spacing,
            image.GetDirection(),
            0,
            sitk.sitkInt16,
        )
        new_image = sitk.Image([final_width, final_height], sitk.sitkInt16)
        new_image = sitk.RescaleIntensity(new_image, -2000, -2000)
        x, y = (final_width - new_width) // 2, (final_height - new_height) // 2
        new_array = sitk.GetArrayFromImage(new_image)
        image_array = sitk.GetArrayFromImage(image)
        new_array[y : y + new_height, x : x + new_width] = image_array
        return new_array
    return np_array


def convert_vtk_to_dcm(
    vtk_image,
    slab_thickness,
    matrix,
    center,
    file_name,
    spacing,
    azimuth_type: str = "axial",
    _type: str = "MIP",
):
    vtk_matrix = vtk.vtkMatrix4x4()
    vtk_matrix.DeepCopy(matrix)
    if center:
        vtk_matrix.SetElement(0, 3, center[0])
        vtk_matrix.SetElement(1, 3, center[1])
        vtk_matrix.SetElement(2, 3, center[2])
    # 切面提取
    reslice = vtk.vtkImageSlabReslice()
    reslice.SetInputConnection(vtk_image.GetOutputPort())
    reslice.SetOutputDimensionality(2)  # 设置输出为2维图片
    # 设置变换矩阵
    reslice.SetResliceAxes(vtk_matrix)
    # 线性插值
    reslice.SetInterpolationModeToLinear()
    # 判断是否是最大密度投影
    if _type == "MIP":
        reslice.SetBlendModeToMax()
    elif _type == "MinIP":
        reslice.SetBlendModeToMin()
    elif _type == "AIP":
        reslice.SetBlendModeToMean()

    reslice.SetSlabThickness(slab_thickness)
    reslice.Update()
    width, height = reslice.GetOutput().GetDimensions()[:2]
    # 冠,矢需要重新采样
    if azimuth_type != MIPPostion.AXIAL:
        resampler = vtk.vtkResampleToImage()
        resampler.SetInputConnection(reslice.GetOutputPort())
        new_width, new_height = width, int((height - 1) / spacing[0] * spacing[2])
        resampler.SetSamplingDimensions(new_width, new_height, 1)
        resampler.Update()
        # vtk img to sitkImage
        vtk_img_export = vtkImageExportToArray()
        vtk_img_export.SetInputConnection(resampler.GetOutputPort())
    else:
        # vtk img to sitkImage
        vtk_img_export = vtkImageExportToArray()
        vtk_img_export.SetInputConnection(reslice.GetOutputPort())

    sitk_array = vtk_img_export.GetArray()
    sitk_array = sitk_array.astype(np.int16).reshape(sitk_array.shape[1:])
    logger.info(f"sitk_array.shape: {sitk_array.shape}")
    # draw_2d_position
    # TODO: 优化下位置计算, 数字改为常量
    film_w, film_h = 512, 512
    locals = [
        (film_w // 2, 0),
        (film_w // 2, film_h - 2 * FONT_SIZE),
        (0, film_h // 2),
        (film_w - 10, film_h // 2),
    ]
    res_np = draw_text(
        resize_hu(sitk_array),
        f"{MIP_POSITION_MAP[azimuth_type]} MIP {slab_thickness}",
        MIP_LOCALS_MAP[azimuth_type],
        locals,
    )
    sitk.WriteImage(sitk.GetImageFromArray(res_np), file_name.replace("png", "dcm"))


def get_dicom_poetry(img_arr, slab_thickness, azimuth_type, spacing, origin):
    extent = list(img_arr.shape)
    # 根据azimuth_type更改切面厚度
    if azimuth_type == "axial":
        extent[2] = slab_thickness
    elif azimuth_type == "sagittal":
        extent[0] = slab_thickness
    elif azimuth_type == "coronal":
        extent[1] = slab_thickness

    center = [
        origin[0] + spacing[0] * 0.5 * (extent[0]),
        origin[1] + spacing[1] * 0.5 * (extent[1]),
        origin[2] + spacing[2] * 0.5 * (extent[2]),
    ]
    return center


def vtk_image_from_array(slice_data, spacing, origin):
    vtk_image = vtkImageImportFromArray()
    vtk_image.SetArray(slice_data)
    vtk_image.SetDataSpacing(spacing)
    vtk_image.SetDataOrigin(origin)
    # vtk_image.SetDataExtent(extent)
    vtk_image.Update()
    return vtk_image


def get_st_ed(
    img_arr,
    spacing,
    origin,
    sitk_image,
    image_count: int,
    thickness: float,
    thick_interval: float,
    azimuth_type: str = "axial",
):
    """ "
    获取重建MIP的起止 index
    Args:
        img_arr: 颅内数据， np_array
        sitk_image: 头颈完体数据
        image_count: MIP 重建张数
        thickness: 层厚
        thick_interval: 层间距
    Returns:

    """
    intracranial_img = sitk.GetImageFromArray(img_arr)
    intracranial_img.SetSpacing(spacing)
    intracranial_img.SetOrigin(origin)
    intracranial_img.SetDirection(sitk_image.GetDirection())
    if azimuth_type == MIPPostion.AXIAL:
        mid = img_arr.shape[0] // 2
        mid_phyc = list(intracranial_img.TransformIndexToPhysicalPoint([mid, 0, 0]))[0]
        left_num = image_count // 2
        right_num = image_count - left_num
        st = intracranial_img.TransformPhysicalPointToIndex(
            [mid_phyc - left_num * (thick_interval + thickness), origin[1], origin[2]]
        )[0]
        slab_interval = int((thickness) // spacing[2])
        pixel_interval = int((thick_interval) // spacing[2])
        if st < 0:
            st = 0
            end_index = intracranial_img.TransformPhysicalPointToIndex(
                [
                    origin[0] + image_count * (thick_interval + thickness),
                    origin[1],
                    origin[2],
                ]
            )[0]
        else:
            end_index = intracranial_img.TransformPhysicalPointToIndex(
                [
                    mid_phyc + right_num * (thick_interval + thickness),
                    origin[1],
                    origin[2],
                ]
            )[0]
        if end_index > img_arr.shape[0]:
            end_index = img_arr.shape[0]
    elif azimuth_type == MIPPostion.SAGITTAL:
        mid = img_arr.shape[2] // 2
        mid_phyc = list(intracranial_img.TransformIndexToPhysicalPoint([0, 0, mid]))[2]
        left_num = image_count // 2
        right_num = image_count - left_num
        st = intracranial_img.TransformPhysicalPointToIndex(
            [origin[0], origin[1], mid_phyc - left_num * (thick_interval + thickness)]
        )[2]
        slab_interval = int((thickness) // spacing[1])
        pixel_interval = int((thick_interval) // spacing[1])
        if st < 0:
            st = 0
            end_index = intracranial_img.TransformPhysicalPointToIndex(
                [
                    origin[0],
                    origin[1],
                    origin[2] + image_count * (thick_interval + thickness),
                ]
            )[2]
        else:
            end_index = intracranial_img.TransformPhysicalPointToIndex(
                [
                    origin[0],
                    origin[1],
                    mid_phyc + right_num * (thick_interval + thickness),
                ]
            )[2]
        if end_index > img_arr.shape[2]:
            end_index = img_arr.shape[2]
    else:
        mid = img_arr.shape[1] // 2
        mid_phyc = list(intracranial_img.TransformIndexToPhysicalPoint([0, 0, mid]))[1]
        left_num = image_count // 2
        right_num = image_count - left_num
        st = intracranial_img.TransformPhysicalPointToIndex(
            [origin[0], mid_phyc - left_num * (thick_interval + thickness), origin[2]]
        )[1]
        slab_interval = int((thickness) // spacing[1])
        pixel_interval = int((thick_interval) // spacing[1])
        if st < 0:
            st = 0
            end_index = intracranial_img.TransformPhysicalPointToIndex(
                [
                    origin[0],
                    origin[1] + image_count * (thick_interval + thickness),
                    origin[2],
                ]
            )[1]
        else:
            end_index = intracranial_img.TransformPhysicalPointToIndex(
                [
                    origin[0],
                    mid_phyc + right_num * (thick_interval + thickness),
                    origin[2],
                ]
            )[1]
        if end_index > img_arr.shape[1]:
            end_index = img_arr.shape[1]
    logger.warning(f"mid_phyc: {mid_phyc}, origin: {origin}, spacing: {spacing}")
    logger.warning(
        f"start_index: {st}, end_index: {end_index}, slab_Interval: {slab_interval}, pixel_interval: {pixel_interval}"
    )
    return st, end_index, slab_interval, pixel_interval


def gen_mip(
    sitk_image, seg_lps_array, thickness, thick_interval, image_count, azimuth_type
):
    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    hu_volume_lps = sitk.GetArrayFromImage(sitk_image).astype(np.int16)
    new_seg_lps_array = seg_lps_array.copy()

    mask, hu_volume_lps = gen_Intracranial_data(
        seg_lps_array,
        hu_volume_lps,
        new_seg_lps_array,  # 膨胀后的mask
        IntracranialVR,
    )
    [zmin, zmax, ymin, ymax, xmin, xmax] = boundbox_3d(mask)
    cut_im = hu_volume_lps[zmin:zmax, ymin:ymax, xmin:xmax].copy()
    # 　获取颅内数据　img_arr
    img_arr = cut_im
    # sitk.WriteImage(sitk.GetImageFromArray(cut_im), "./cut_im.nii.gz")
    logger.info(f"***img_arr: {img_arr.shape}")
    center = get_dicom_poetry(img_arr, thickness, azimuth_type, spacing, origin)
    # TODO: 根据层厚和间隔计算张数, 需要优化下呢, 层间距用错了，当成层厚了 @txueduo
    start_index, end_index, slab_Interval, pixel_interval = get_st_ed(
        img_arr,
        spacing,
        origin,
        sitk_image,
        image_count,
        thickness,
        thick_interval,
        azimuth_type,
    )
    count = 1
    t0 = time.time()
    for i in range(start_index, end_index, pixel_interval + slab_Interval):
        if i + slab_Interval > end_index:
            break
        logger.warning(f"count: {count}")
        if azimuth_type == "axial":
            data = img_arr[i : i + slab_Interval, :, :]
        elif azimuth_type == "sagittal":
            data = img_arr[:, :, i : i + slab_Interval]
            data = np.ascontiguousarray(data)
        elif azimuth_type == "coronal":
            data = img_arr[:, i : i + slab_Interval, :]
            data = np.ascontiguousarray(data)
        file_name = os.path.join(
            output_dir, f"{MIP_POSITION_MAP[azimuth_type]}_MIP_{thickness}_{i}.png"
        )
        vtk_image = vtk_image_from_array(data, spacing, origin)
        convert_vtk_to_dcm(
            vtk_image,
            thickness,
            MATRIX_MAP[azimuth_type],
            center,
            file_name,
            spacing,
            azimuth_type,
        )
        # if count > image_count:
        #     break
        count += 1
    logger.warning(
        f"总共生成{azimuth_type} 位 MIP {count-1} 张, cost time : {time.time() - t0}"
    )


@pytest.fixture()
def data():
    root_dir = "/media/tx-deepocean/Data/DICOMS/demos/mra"
    bone_nii = os.path.join(
        root_dir, "1.2.392.200036.9116.2.6.1.37.2420991567.1550193082.682249.nii.gz"
    )
    bone_img = sitk.ReadImage(bone_nii)
    seg_path = os.path.join(root_dir, "cerebral-seg.nii.gz")
    seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    return dict(im=bone_img, seg_arr=seg_arr)


def test_mip(data):
    t_mip = time.time()
    sitk_image, seg_arr = data["im"], data["seg_arr"]
    gen_mip(
        sitk_image,
        seg_arr,
        thickness=20,
        thick_interval=10,
        image_count=5,
        azimuth_type="coronal",
    )
    logger.info(f"gen_mip cost: {time.time() - t_mip}")
