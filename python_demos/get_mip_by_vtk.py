import json
import math
import time

import cv2
import numpy as np
import pydicom
import requests
import SimpleITK as sitk
import vtkmodules.all as vtk
from vtkmodules.util.vtkImageExportToArray import vtkImageExportToArray
from vtkmodules.util.vtkImageImportFromArray import vtkImageImportFromArray


def vtk_image_from_array(data, spacing, origin):
    vtk_image = vtkImageImportFromArray()
    vtk_image.SetArray(data)
    vtk_image.SetDataSpacing(spacing)
    vtk_image.SetDataOrigin(origin)
    # vtk_image.SetDataExtent(extent)
    vtk_image.Update()
    return vtk_image


def get_matrix(angle=90, axis="X") -> list:
    """欧拉角angel, 依次绕定轴xyz"""
    if axis == "X":
        rotation = [
            1,
            0,
            0,
            0,
            0,
            math.cos(angle),
            -math.sin(angle),
            0,
            0,
            math.sin(angle),
            math.cos(angle),
            0,
            0,
            0,
            0,
            1,
        ]
    elif axis == "Y":
        rotation = [
            math.cos(angle),
            0,
            math.sin(angle),
            0,
            0,
            1,
            0,
            0,
            -math.sin(angle),
            0,
            math.cos(angle),
            0,
            0,
            0,
            0,
            1,
        ]
    elif axis == "Z":
        rotation = [
            math.cos(angle),
            -math.sin(angle),
            0,
            0,
            math.sin(angle),
            math.cos(angle),
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
        ]
    center = []
    rotation = np.dot(rotation_x, np.dot(rotation_y, rotation_z))
    return rotation


def get_total_by_matrix(
    rebuild_spacing: float, slab_thickness: float, azimuth_type: int, matrix: list
):
    """根据matrix 方位， 返回重建后层数
    Args:
        slab_thickness: 20mm, 重建层厚
        azimuth_type: 1, 2, 3(轴冠矢)
    """
    total = 0
    st_origin = [111, 222, 222]
    # TODO: 获取旋转后三个方位右下角的原点
    origin = [111, 2222, 4444]
    if azimuth_type == 1:
        # TODO: 确认重建后的 spacing
        # 轴 Z
        total = math.floor((origin[2] - slab_thickness - st_origin[2]) / slab_thickness)
        # 获取旋转后 origin 范围
    elif azimuth_type == 2:
        # 冠 Y
        total = math.floor((origin[1] - slab_thickness - st_origin[1]) / slab_thickness)
    elif azimuth_type == 3:
        # 失 X
        total = math.floor((origin[0] - slab_thickness - st_origin[0]) / slab_thickness)
    else:
        raise "azimuth_type input error"

    return total


def get_mpr(
    slab_thickness: float,
    rebuild_spacing: float,
    matrix: list,
    rebuild_type: str,
    azimuth_type: str,
) -> dict:
    """任意mpr的mip
    Args:
        matrix: 4*4 [], lps 暂定
        azimuth_type: 1,2,3(轴冠失)
    Return:
        {"dcm_path": "", "boxes": [], counter: []}
    """
    res = {}
    nii_path = "/media/tx-deepocean/Data/DICOMS/demos/1.2.840.113619.2.416.77348009424380358976506205963520437809_nobone.nii.gz"
    dcm_path = "/media/tx-deepocean/Data/DICOMS/ct_cerebral/CN010002-13696724/1.2.840.113619.2.416.10634142502611409964348085056782520111/1.2.840.113619.2.416.77348009424380358976506205963520437809/1.2.840.113619.2.416.2106685279137426449336212617073446717.1"
    ds = pydicom.read_file(dcm_path, stop_before_pixels=True, force=True)
    # https://blog.csdn.net/qq_41023026/article/details/118891837
    img = sitk.ReadImage(nii_path)
    img_arr = sitk.GetArrayFromImage(img)  # (z,y,x)
    img_shape = img_arr.shape

    img_arr = img_arr[:, :, :]
    img_arr = np.ascontiguousarray(img_arr)

    # 根据层厚 SliceThickness 层间距 计算 mip 各方位总层数, SpacingBetweenSlices
    # total = get_total_by_matrix(matrix, extent)
    t0 = time.time()
    spacing = img.GetSpacing()
    print(f"***spacing: {spacing}")
    origin = img.GetOrigin()
    print(f"*****nii 原点: {origin}")
    st_origin = ds.ImagePositionPatient
    print(f"*****ImageOrientationPatient: {ds.ImageOrientationPatient}")

    reader = vtk_image_from_array(img_arr, spacing, origin)

    # 获取物理信息
    extent = reader.GetOutput().GetExtent()  # 维度
    print(f"extent: {extent}")  # (x, y, z)
    spacing = reader.GetOutput().GetSpacing()  # 间隔
    origin = reader.GetOutput().GetOrigin()  # 原点
    print(f"*****vtk 原点: {origin}")

    # TODO: 根据角度分别计算 X, Y, Z 旋转矩阵
    # matrix = get_matrix(angle=90, axis="X")
    # ras to 像素坐标
    ijk_to_ras_direction = [-1, 0, 0, 0, -1, 0, 0, 0, 1]

    # TODO: 根据层厚 slab_thickness 层间距 原点，计算重建层数起止位置 done
    # 轴状位
    if azimuth_type == 1:
        matrix = [
            0.975808,
            0,
            -0.218629,
            37.5841,
            0,
            -1,
            0,
            -6.81277,
            0.218629,
            0,
            0.975808,
            -104.264,
            0,
            0,
            0,
            1,
        ]
        rotate_m = [[0.975808, 0, 0.218629], [0, -1, 0], [-0.218629, 0, 0.975808]]
        # 最后一张原点
        ed_origin = [
            st_origin[0],
            st_origin[1],
            st_origin[2] + spacing[2] * img_shape[2],
        ]

        # 变换后的 总长度
        rotate_st_origin = np.dot(st_origin, rotate_m)
        rotate_ed_origin = np.dot(ed_origin, rotate_m)
        # 总长度
        # TODO： 计算 spacing， 搞清楚正负问题
        rotate_spacing = [0.5450804732219816, 0.5585940000000003, 0.6098799768055843]
        z_idx = math.floor((matrix[11] - rotate_st_origin[2]) / rotate_spacing[2])
        print([z_idx * i for i in rotate_spacing])
        k_idx = (rotate_st_origin + [z_idx * i for i in rotate_spacing]).tolist()
        rotate_st_origin = rotate_st_origin.tolist()
        rotate_ed_origin = rotate_ed_origin.tolist()

        print(rotate_st_origin, rotate_ed_origin)
        # 根据 slab_thickness 和 k_idx 获取 mip 区间
        # TODO: 封装成函数
        if (k_idx[2] - 0.5 * slab_thickness) < rotate_st_origin[2]:
            # TODO： 这样会导致，部分 mip 最后无变化
            min_idx = ijk_to_xyz(
                rotate_st_origin[2], rotate_st_origin[2], rotate_spacing[2]
            )
            max_idx = ijk_to_xyz(
                rotate_st_origin[2] + slab_thickness,
                rotate_st_origin[2],
                rotate_spacing[2],
            )
        elif (k_idx[2] + 0.5 * slab_thickness) > rotate_ed_origin[2]:
            print(1111111111111)
            max_idx = ijk_to_xyz(
                rotate_ed_origin[2], rotate_st_origin[2], rotate_spacing[2]
            )
            min_idx = ijk_to_xyz(
                rotate_ed_origin[2] - slab_thickness,
                rotate_st_origin[2],
                rotate_spacing[2],
            )
        else:
            min_idx = ijk_to_xyz(
                k_idx[2] - 0.5 * slab_thickness, rotate_st_origin[2], rotate_spacing[2]
            )
            max_idx = ijk_to_xyz(
                k_idx[2] + 0.5 * slab_thickness, rotate_st_origin[2], rotate_spacing[2]
            )
        slice_nums = max_idx - min_idx
        # 计算体数据中心点
        center = []
        center.append(origin[0] + spacing[0] * 0.5 * (extent[0] + extent[1]))  # 失
        center.append(origin[1] + spacing[1] * 0.5 * (extent[2] + extent[3]))  # 冠
        center.append(origin[2] + spacing[2] * 0.5 * (min_idx + max_idx))  # 轴
        print(f"*****center:  {center}")
        # reslice.SetOutputSpacing(spacing[0], spacing[1], rebuild_spacing) # 区分轴冠失

    elif azimuth_type == 2:
        # 冠状位
        matrix = [
            0.965339,
            -0.0903941,
            -0.244845,
            0.269552,
            0.244845,
            0.638547,
            0.729595,
            -6.46509,
            0.0903941,
            -0.764256,
            0.638547,
            60.4262,
            0,
            0,
            0,
            1,
        ]
        # 计算体数据中心点
        center = []
        center.append(origin[0] + spacing[0] * 0.5 * (extent[0] + extent[1]))  # 失
        center.append(origin[1] + spacing[1] * 0.5 * (min_idx + max_idx))  # 冠
        center.append(origin[2] + spacing[2] * 0.5 * (extent[4] + extent[5]))  # 轴
        print(f"*****center:  {center}")
    elif azimuth_type == 3:
        pass
        # 失状位
        matrix = [-1, 0, 0, 0.279233, 0, 1, 0, -6.81277, 0, 0, 1, 60.401, 0, 0, 0, 1]
        # 计算体数据中心点
        center = []
        center.append(origin[0] + spacing[0] * 0.5 * (min_idx + max_idx))  # 失
        center.append(origin[1] + spacing[1] * 0.5 * (extent[2] + extent[3]))  # 冠
        center.append(origin[2] + spacing[2] * 0.5 * (extent[4] + extent[5]))  # 轴
        print(f"*****center:  {center}")
    else:
        raise "azimuth_type input error"
    print(f"slice_nums: {slice_nums}， min_idx: {min_idx}, max_idx: {max_idx}")

    resliceAxes = vtk.vtkMatrix4x4()
    resliceAxes.DeepCopy(matrix)
    # TODO: 这个center影响最后出图，
    resliceAxes.SetElement(0, 3, center[0])
    resliceAxes.SetElement(1, 3, center[1])
    resliceAxes.SetElement(2, 3, center[2])
    # 通过 extent 截取对应的体数据
    extractVOI = vtk.vtkExtractVOI()
    extractVOI.SetInputConnection(reader.GetOutputPort())
    extractVOI.SetVOI(extent)
    extractVOI.Update()

    # mip SetResliceAxesOrigin()方法还可以用于提供切片将通过的(x，y，z)点。
    reslice = vtk.vtkImageSlabReslice()
    reslice.SetInputConnection(extractVOI.GetOutputPort())  # 截取的体数据
    reslice.SetOutputDimensionality(2)  # 设置输出为2维图片
    reslice.SetInterpolationModeToLinear()  # 差值

    rotate_spacing = reslice.GetOutputSpacing()
    print(f"rotate_spacing: {rotate_spacing}")

    if rebuild_type == "MIP":
        reslice.SetBlendModeToMax()
    elif rebuild_type == "MinIP":
        reslice.SetBlendModeToMin()
    elif rebuild_type == "AIP":
        reslice.SetBlendModeToMean()
    reslice.SetSlabThickness(slice_nums)  # 设置厚度
    reslice.SetResliceAxes(resliceAxes)  # 设置矩阵
    reslice.Update()

    vtk_img_export = vtkImageExportToArray()
    vtk_img_export.SetInputConnection(reslice.GetOutputPort())
    np_array = vtk_img_export.GetArray()
    np_array = np_array.astype(np.int16)
    print(np.min(np_array))
    np_array[np_array == 0] = np.min(np_array)
    result = sitk.GetImageFromArray(np_array)
    result.SetMetaData("0028|1050", "40")
    result.SetMetaData("0028|1051", "400")
    # TODO: 重新计算 ImagePositionPatient, orientation, slicelocation, PixelSpacing,
    sitk.WriteImage(result, "./test.dcm")

    # contour = get_model_contour()
    contour = []
    contour_dic = {}
    contour_dic["contours"] = [
        [[1, 2], [3, 4]],
        [[4, 2], [5, 4]],
    ]
    contour_dic["id"] = 1
    contour.append(contour_dic)

    boxes = []
    box_dic = {}
    box_dic["boxes"] = [300, 200, 320, 210, 289.5, 245]
    box_dic["id"] = 1
    boxes.append(box_dic)

    res["dcm_path"] = f"/tmp/{rebuild_type}.dcm"
    # res["sagittal_dcm_path"] = sagittal_dcm_path
    # res["coronal_dcm_path"] = coronal_dcm_path
    res["boxes"] = boxes
    res["contour"] = contour
    print(f"a mip cost: {time.time() - t0}")
    return res


def ijk_to_xyz(ijk, origin, spacing):
    """物理坐标转像素坐标"""
    if isinstance(ijk, float):
        xyz = math.floor((ijk - origin) / spacing)
    else:
        sub = np.sub(np.array(ijk), np.array(origin))
        xyz = np.divide(sub, np.array(spacing))
        xyz = np.int0(xyz).tolist()
    return xyz


def np_array_to_dcm(ds, save_path):
    ds.WindowCenter = 40
    ds.WindowWidth = 400
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    # TODO: 重新计算 ImagePositionPatient, orientation, slicelocation, PixelSpacing,

    ds.PixelData = np_array.tobytes()
    ds.is_implicit_VR = True
    ds.save_as(save_path)


def get_physical(series_path):

    image_shape = (original_ds.Columns, original_ds.Rows, len(dcm_list))
    return


def get_model_contour():
    """调用模型contour"""
    pass
    return []


def main():
    series_path = "/media/tx-deepocean/Data/DICOMS/ct_cerebral/CN010023-1510071117/1.3.46.670589.33.1.63780596181185791300001.5349461926886765269/1.3.46.670589.33.1.63780596345132168500001.4829782583723493361"
    slab_thickness = 50
    scale = 1
    # TODO: 层厚小于 1mm 情况前端控制？
    matrix = [
        -0.977967,
        -0.0432743,
        -0.204225,
        0.757874,
        -0.204864,
        0.0108364,
        0.978731,
        -142.976,
        -0.0401408,
        0.999004,
        -0.019463,
        11.8231,
        0,
        0,
        0,
        1,
    ]
    res = get_mpr(
        slab_thickness,
        slab_thickness * scale,
        matrix,
        rebuild_type="MIP",
        azimuth_type=1,
    )
    print(json.dumps(res))


if __name__ == "__main__":
    main()
