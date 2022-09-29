import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian


def np_array_to_dcm(ds, color_array, save_path, is_rgb=False, instance_number=20):
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
        ds.WindowWidth = 800
        ds.WindowCenter = 300
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
    ds.PixelData = color_array.tobytes()
    ds.save_as(save_path)


def process_data():
    # 读取nii
    mask_file = "/media/tx-deepocean/Data/DICOMS/demos/output_new_data_2d/1.3.46.670589.33.1.63790032426654424200002.5278662375047719733/TTP.nii.gz"
    mask_img = sitk.ReadImage(mask_file, sitk.sitkUInt8)

    # 读取dicom
    dcm_file = "/media/tx-deepocean/Data/DICOMS/demos/TMAX/TMAX023.dcm"
    dcm_img_hu = sitk.ReadImage(dcm_file)

    # 将hu值图像rescale到0-255的范围
    rescaler = sitk.RescaleIntensityImageFilter()
    rescaler.SetOutputMaximum(255)
    rescaler.SetOutputMinimum(0)
    dcm_img_grey = rescaler.Execute(mask_img)

    # 遍历生成伪彩图
    mask_array = sitk.GetArrayFromImage(mask_img)
    for dcm_slice in range(mask_array.shape[0]):
        if dcm_slice == 0:
            dcm_2d_array = mask_array[dcm_slice, :, :]
            # TODO: color_array = get_color_map(dcm_img_2d)

    # 缩减维度，三维图像变成二维，也可用sitk.sequeeze()
    # dcm_img_3d = sitk.GetArrayFromImage(dcm_img_hu)
    # dcm_img_2d = dcm_img_3d[0,:,:]
    # print(dcm_img_2d.dtype)

    origin_dcm_file = "/media/tx-deepocean/Data/DICOMS/demos/data/input/1.3.46.670589.33.1.63786793899195991600001.5714046135148655087/CN010023-2204291002-304-301_0001.dcm"
    ds = pydicom.read_file(dcm_file, force=True)
    return ds, dcm_2d_array


def get_color_map(dcm_img_2d):
    # 灰度图片变成伪彩色
    color_img = cv2.applyColorMap(dcm_img_2d, cv2.COLORMAP_JET)  # 自己测试turbo
    # for j in range(256):
    # print(color_img[0, j, 2], color_img[0, j, 1], color_img[0, j, 0])

    background_rgb = [color_img[0, 0, 0], color_img[0, 0, 1], color_img[0, 0, 2]]
    print(background_rgb)
    background_pixel = np.where(
        (color_img[:, :, 0] == background_rgb[0])
        & (color_img[:, :, 1] == background_rgb[1])
        & (color_img[:, :, 2] == background_rgb[2])
    )
    color_img[background_pixel] = [0, 0, 0]
    cv2.imshow("test", color_img)
    cv2.imwrite("./cvmap.png", color_img)
    # print(color_img.shape)
    return color_img


def main():
    # 获取数据
    ds, dcm_img_2d = process_data()

    # 伪彩图
    color_array = get_color_map(dcm_img_2d)
    # opencv 需要将BGR转RGB
    color_array = color_array[:, :, ::-1]
    # 存为dicom
    np_array_to_dcm(ds, color_array, "./colormap.dcm", is_rgb=True)


if __name__ == "__main__":
    main()
