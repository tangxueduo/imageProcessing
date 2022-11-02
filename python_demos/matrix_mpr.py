# https://blog.csdn.net/qq_39942341/article/details/122442949
# https://blog.csdn.net/liushao1031177/article/details/124847587
import numpy as np
import SimpleITK as sitk
import vtkmodules.all as vtk
from vtkmodules.util.vtkImageExportToArray import vtkImageExportToArray

""" 补充正交mpr 矩阵切片 """


def get_matrix(position: str, angle):
    """失状位"""
    if position == "sagittal":
        elements = [0, 0, -1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1]
    elif position == "axial":
        # """ 轴状位 """
        elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    elif position == "coronal":
        """冠状位"""
        elements = [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1]
    else:
        print("position can not found")
    """ 其他角度 """
    elements = [
        -0.797459,
        0,
        -0.603374,
        0.222664,
        0,
        1,
        0,
        -6.81277,
        -0.603374,
        0,
        0.797459,
        60.5695,
        0,
        0,
        0,
        1,
    ]

    # elements = [
    #     1, 0.2, 0, 0,
    #     0, 1.2, 0, 0,
    #     0, 0, 1,0,
    #     0,0,0,1
    # ]
    return elements


def get_mpr_total():
    """根据方位来获取总张数，mpr应该是extent最大值，故不需要该接口"""
    pass


def get_mpr():
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(
        "/media/tx-deepocean/Data/DICOMS/RESULT/volume/1.2.840.113619.2.416.77348009424380358976506205963520437809.nii.gz"
    )
    reader.Update()
    vtkImage = reader.GetOutput()
    extent = vtkImage.GetExtent()
    spacing = vtkImage.GetSpacing()
    origin = vtkImage.GetOrigin()
    elements = get_matrix(position="axial", angle=0)

    #  512 ,512, 589
    dcm_idx = 0
    center = [
        origin[0] + spacing[0] * (extent[0] + extent[1]),  # 矢状面（ sagittal plane）S
        origin[1] + spacing[1] * (extent[2] + extent[3]),  # 冠状面（ coronal plane）C
        origin[2] + spacing[2] * (extent[4] + extent[5]),  # 横断面（transverse plane）A
    ]
    resliceAxes = vtk.vtkMatrix4x4()
    resliceAxes.DeepCopy(elements)
    for i in range(3):
        resliceAxes.SetElement(i, 3, center[i])
    reslice = vtk.vtkImageReslice()
    reslice.SetInputConnection(reader.GetOutputPort())
    reslice.SetOutputDimensionality(2)
    reslice.SetResliceAxes(resliceAxes)
    reslice.SetInterpolationModeToLinear()
    reslice.Update()

    # writer = vtk.vtkNIFTIImageWriter()
    # writer.SetFileName("test.nii.gz")
    # writer.SetInputData(reslice.GetOutput())
    # writer.Write()
    vtk_img_export = vtkImageExportToArray()
    vtk_img_export.SetInputConnection(reslice.GetOutputPort())
    sitk_array = vtk_img_export.GetArray()
    sitk_array = sitk_array.astype(np.int16)
    result = sitk.GetImageFromArray(sitk_array)
    result.SetMetaData("0028|1050", "300")
    result.SetMetaData("0028|1051", "800")
    sitk.WriteImage(result, "./test.dcm")


get_mpr()
