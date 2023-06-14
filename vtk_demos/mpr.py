import numpy as np
import pydicom
import SimpleITK as sitk
import vtk
from loguru import logger
from vtk.util.numpy_support import numpy_to_vtk
from vtkmodules.util.vtkImageExportToArray import vtkImageExportToArray
from vtkmodules.util.vtkImageImportFromArray import vtkImageImportFromArray


def get_dicom_poetry(img_arr, slab_thickness, azimuth_type, spacing, origin):
    extent = list(img_arr.shape)
    # �| ��~M�azimuth_type�~[��~T��~H~G�~]��~N~Z度
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
    print(f"center: {center}")
    return center


def gen_mip_dcm(
    vtk_image,
    slab_thickness,
    matrix,
    center,
    spacing,
    _type: str = "MIP",
):
    vtk_matrix = vtk.vtkMatrix4x4()
    vtk_matrix.DeepCopy(matrix)

    if center:
        vtk_matrix.SetElement(0, 3, center[0])
        vtk_matrix.SetElement(1, 3, center[1])
        vtk_matrix.SetElement(2, 3, center[2])
    # �~H~G�~]��~O~P�~O~V
    reslice = vtk.vtkImageSlabReslice()
    reslice.SetInputConnection(vtk_image.GetOutputPort())
    reslice.SetOutputDimensionality(2)  # 设置��~S�~G�为2维�~[��~I~G
    # 设置�~O~X�~M��~_��~X�
    reslice.SetResliceAxes(vtk_matrix)
    # 线�~@��~O~R�~@�
    reslice.SetInterpolationModeToLinear()
    # �~H��~V��~X��~P��~X��~\~@大��~F度�~J~U影
    if _type == "MIP":
        reslice.SetBlendModeToMax()
    elif _type == "MinIP":
        reslice.SetBlendModeToMin()
    elif _type == "AIP":
        reslice.SetBlendModeToMean()

    reslice.SetSlabThickness(slab_thickness)
    reslice.Update()

    # imageResample = vtk.vtkImageResample()
    # imageResample.SetInputData(reslice.GetOutput())
    # imageResample.SetOutputDimensionality(2)
    # # # 如果是
    # imageResample.SetAxisMagnificationFactor(0, 1)
    # imageResample.SetAxisMagnificationFactor(1, round(230 / 160, 3))
    # imageResample.SetInterpolationModeToLinear()  # 重采样的方式
    # imageResample.SetOutputExtent([0, 511, 0, 221, 0, 0])
    # imageResample.ReleaseDataFlagOff()
    # imageResample.Update()
    resampler = vtk.vtkResampleToImage()
    resampler.SetInputConnection(reslice.GetOutputPort())
    width, height = reslice.GetOutput().GetDimensions()[:2]
    print(reslice.GetOutput())
    new_width, new_height = width, int((height - 1) / spacing[0] * spacing[2])
    print(new_height)
    # width, height = 512, 222
    resampler.SetSamplingDimensions(new_width, new_height, 1)
    resampler.Update()
    # vtk img to sitkImage
    vtk_img_export = vtkImageExportToArray()
    vtk_img_export.SetInputConnection(resampler.GetOutputPort())

    sitk_array = vtk_img_export.GetArray()
    sitk_array = sitk_array.astype(np.int16)
    result = sitk.GetImageFromArray(sitk_array)
    return result, reslice.GetOutput().GetSpacing()


def vtk_image_from_array(slice_data, spacing, origin):
    vtk_image = vtkImageImportFromArray()
    vtk_image.SetArray(slice_data)
    vtk_image.SetDataSpacing(spacing)
    vtk_image.SetDataOrigin(origin)
    # vtk_image.SetDataExtent(extent)
    vtk_image.Update()
    return vtk_image


def mpr(slice_index=100, azimuth_type="coronal"):
    dcm_path = "./demo.dcm"
    sitk_image = sitk.ReadImage(
        "/media/tx-deepocean/Data/DICOMS/demos/mra/1.3.46.670589.11.33383.5.0.9500.2022120114584315001.nii.gz"
    )
    hu_volume_lps = sitk.GetArrayFromImage(sitk_image)
    slab_thickness = 10
    azimuth_matrix = {
        "axial": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "sagittal": [0, 0, 1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1],
        "coronal": [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1],
    }
    image_shape = hu_volume_lps.shape
    azimuth_count_map = {
        "axial": image_shape[2],
        "coronal": image_shape[1],
        "sagittal": image_shape[0],
    }

    axesElements = azimuth_matrix.get(azimuth_type)
    total = azimuth_count_map.get(azimuth_type)
    slab_end = slice_index + slab_thickness
    if slab_end > total:
        raise IndexError
    if azimuth_type == "coronal":
        end_index = slab_end
        start_index = slice_index
    else:
        end_index = total - slice_index
        start_index = total - (slice_index + slab_thickness)

    # �~N��~O~V�~H~G�~I~G对��~T�~Z~Ddicom信�~A���~L��~F�~[~V�~\�MIP
    slice_index = int((slice_index * azimuth_count_map.get("axial")) / total)

    img_arr = hu_volume_lps
    spacing = sitk_image.GetSpacing()
    print(spacing)
    origin = sitk_image.GetOrigin()
    if azimuth_type == "axial":
        data = img_arr[start_index:end_index, :, :]
    elif azimuth_type == "sagittal":
        data = img_arr[:, :, start_index:end_index]
        data = np.ascontiguousarray(data)
    elif azimuth_type == "coronal":
        data = img_arr[:, start_index:end_index, :]
        data = np.ascontiguousarray(data)

    vtk_image = vtk_image_from_array(data, spacing, origin)
    center = get_dicom_poetry(img_arr, slab_thickness, azimuth_type, spacing, origin)
    print(center)
    width, height = sitk_image.GetSize()[:2]
    new_height = int((height - 1) / spacing[0] * spacing[2])
    output_origin = [(-width / 2.0 * spacing[0]), -new_height / 2.0 * spacing[0], 0.0]
    sitk_img, spcing = gen_mip_dcm(
        vtk_image, slab_thickness, axesElements, center, spacing
    )
    # 采样
    # sitk.WriteImage(sitk_img, dcm_path.replace("demo", "demo1"))
    # if azimuth_type != "axial":
    #     width, height = sitk_img.GetSize()[:2]
    #     print(spacing)
    #     new_spc = [0.429688, 0.599999, 1]

    #     new_height = int((height - 1) / spacing[0] * spacing[2])
    #     original_size = sitk_image.GetSize()
    #     new_spc = [(original_size[0] - 1) * spacing[0] / (new_height - 1)] * 3
    #     print(new_spc)
    #     print(f'height: {height}')
    #     sitk_img = sitk.Resample(
    #         sitk_img,
    #         [width, new_height, 1],
    #         sitk.Transform(),
    #         sitk.sitkLinear,
    #         [0,0,0],
    #         new_spc,
    #         sitk_img.GetDirection(),
    #         0,
    #         sitk_img.GetPixelID(),
    #     )
    sitk.WriteImage(sitk_img, dcm_path)


mpr()
