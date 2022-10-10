# https://blog.csdn.net/qq_39942341/article/details/122442949
# https://blog.csdn.net/liushao1031177/article/details/124847587
import numpy as np
import SimpleITK as sitk
import vtkmodules.all as vtk
from vtkmodules.util.vtkImageExportToArray import vtkImageExportToArray

reader = vtk.vtkNIFTIImageReader()
reader.SetFileName(
    "/media/tx-deepocean/Data/DICOMS/RESULT/volume/1.2.840.113619.2.416.77348009424380358976506205963520437809.nii.gz"
)
reader.Update()
vtkImage = reader.GetOutput()
extent = vtkImage.GetExtent()
spacing = vtkImage.GetSpacing()
origin = vtkImage.GetOrigin()
""" 失状位 """
# elements = [
#     0, 0, -1, 0,
#     1, 0, 0, 0,
#     0, -1, 0, 0,
#     0, 0, 0, 1
# ]
# """ 轴状位 """
# elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

""" 冠状位 """
# elements = [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1]

""" 其他角度 """
# elements = [
#     1, 0, 0, 0,
#     0, 8.66025, -0.5, 0,
#     0, 0.5, 0.866025, 0,
#     0, 0, 0, 1
# ]

# elements = [
#     1, 0.2, 0, 0,
#     0, 1.2, 0, 0,
#     0, 0, 1,0,
#     0,0,0,1
# ]

elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
#  512 ,512, 589
center = [
    origin[0] + spacing[0] * 512,  # 矢状面（sagittal plane）S
    origin[1] + spacing[1] * 300,  # 冠状面（coronal plane）C
    origin[2] + spacing[2] * 400,  # 横断面（transverse plane）A
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
result.SetMetaData("key", "")
sitk.WriteImage(result, "./test.dcm")
