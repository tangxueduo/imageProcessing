import json
import time

import cv2
import numpy as np
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


def get_mpr(thickness: float, spacing: float, matrix: list, rebuild_type: str) -> dict:
    """任意mpr的mip
    Args:
        matrix: 4*4 [], 暂定
    Return:
        {"dcm_path": "", "boxes": [], counter: []}
    """
    res = {}
    nii_path = "/media/tx-deepocean/Data/DICOMS/demos/1.2.840.113619.2.416.77348009424380358976506205963520437809_nobone.nii.gz"
    # https://blog.csdn.net/qq_41023026/article/details/118891837
    # TODO: sitk 读取 niigz
    img = sitk.ReadImage(nii_path)
    img_arr = sitk.GetArrayFromImage(img)
    t0 = time.time()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    reader = vtk_image_from_array(img_arr, spacing, origin)
    # vtk 读较慢
    # reader = vtk.vtkNIFTIImageReader()
    # reader.SetFileName(nii_path)
    # reader.Update()

    # 高斯平滑 有需要再加
    # gaussianSmoothFilter = vtk.vtkImageGaussianSmooth()
    # gaussianSmoothFilter.SetInputData(reader.GetOutput())
    # gaussianSmoothFilter.SetDimensionality(2) # 二维高斯滤波
    # gaussianSmoothFilter.SetRadiusFactor(3) # 设置高斯模板的大小，当超出该模板的范围时，系数取0
    # gaussianSmoothFilter.SetStandardDeviation(0.6) # 标准差
    # gaussianSmoothFilter.Update()

    # 获取物理信息
    extent = reader.GetOutput().GetExtent()  # 维度
    print(f"***extent: {extent}")
    spacing = reader.GetOutput().GetSpacing()  # 间隔
    print(f"***spacing: {spacing}")
    origin = reader.GetOutput().GetOrigin()  # 原点
    print(f"*****origin: {origin}")
    # 计算体数据中心点
    center = []
    center.append(origin[0] + spacing[0] * 0.5 * (extent[0] + extent[1]))
    center.append(origin[1] + spacing[1] * 0.5 * (extent[2] + extent[3]))
    center.append(origin[2] + spacing[2] * 0.5 * (extent[4] + extent[5]))
    print(f"*****center:  {center}")

    # 任意角度矩阵 4*4
    # matrix = [
    #     1, 0, 0, 0,
    #     0, 1, 0, 0,
    #     0, 0, 1, 0,
    #     0, 0, 0, 1]

    # matrix = [
    # -0.977967, -0.0432743, -0.204225, 0.757874,
    # -0.204864, 0.0108364, 0.978731, -142.976,
    # -0.0401408, 0.999004, -0.019463, 11.8231,
    # 0, 0, 0, 1]
    matrix = [-1, 0, 0, 0.279233, 0, 1, 0, -6.81277, 0, 0, 1, 60.401, 0, 0, 0, 1]

    resliceAxes = vtk.vtkMatrix4x4()
    resliceAxes.DeepCopy(matrix)
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
    reslice.SetInterpolationModeToNearestNeighbor()  # 最近邻差值
    if rebuild_type == "MIP":
        reslice.SetBlendModeToMax()
    elif rebuild_type == "MinIP":
        reslice.SetBlendModeToMin()
    elif rebuild_type == "AIP":
        reslice.SetBlendModeToMean()
    reslice.SetSlabThickness(thickness)  # 设置厚度
    reslice.SetResliceAxes(resliceAxes)  # 设置矩阵
    reslice.Update()

    vtk_img_export = vtkImageExportToArray()
    vtk_img_export.SetInputConnection(reslice.GetOutputPort())
    sitk_array = vtk_img_export.GetArray()
    sitk_array = sitk_array.astype(np.int16)
    result = sitk.GetImageFromArray(sitk_array)
    result.SetMetaData("0028|1050", "300")
    result.SetMetaData("0028|1051", "800")
    result.SetMetaData("key", "")
    sitk.WriteImage(result, "./test.dcm")

    boxes = [
        [300, 200, 320, 210, 289.5, 245],
        [300, 200, 320, 210, 289.5, 245],
    ]
    contour = []
    res["dcm_path"] = f"/tmp/{rebuild_type}.dcm"
    # res[position]["sagittal_dcm_path"] = sagittal_dcm_path
    # res[position]["coronal_dcm_path"] = coronal_dcm_path
    # res["boxes"] = boxes
    # res["contour"] = contour
    print(f"a mip cost: {time.time() - t0}")
    return res


def get_physical(series_path):

    image_shape = (original_ds.Columns, original_ds.Rows, len(dcm_list))
    return


def main():
    series_path = "/media/tx-deepocean/Data/DICOMS/ct_cerebral/CN010023-1510071117/1.3.46.670589.33.1.63780596181185791300001.5349461926886765269/1.3.46.670589.33.1.63780596345132168500001.4829782583723493361"
    thickness = 10
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
    res = get_mpr(thickness, thickness * scale, matrix, rebuild_type="MinIP")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
