import json
import math
import time

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk
import vtkmodules.all as vtk
from vtkmodules.util.vtkImageExportToArray import vtkImageExportToArray
from vtkmodules.util.vtkImageImportFromArray import vtkImageImportFromArray

direction = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

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
    img,rebuild_spacing: float, slab_thickness: float, azimuth_type: int, matrix: list, img_shape: list, spacing: list, st_origin: list
):
    total = 0
    M = np.array([[matrix[0], matrix[1], matrix[2]], [matrix[4], matrix[5], matrix[6]], [matrix[8], matrix[9], matrix[10]]])
    direction = np.array([[-1,0,0],[0,-1,0], [0,0,1]])
    M = np.dot(M, direction)
    old_ijk = [matrix[3],matrix[7], matrix[11]]
    ed_origin = np.array([st_origin[0]+img_shape[2]*spacing[0], st_origin[1]+img_shape[1]*spacing[1], st_origin[2]+ img_shape[0]*spacing[2]])
    st_origin = np.array(st_origin)
    # 计算旋转前的 k_id
    if azimuth_type == 1:
        print(M)
        new_ed_origin = np.dot(ed_origin, M).tolist()
        new_st_origin = np.dot(st_origin, M).tolist()
        print(f'******st_origin: {st_origin}, ed_origin: {ed_origin}')
        print(f'******new_st_origin: {new_st_origin}, new_ed_origin: {new_ed_origin}')
        new_spacing = [rebuild_spacing, rebuild_spacing, rebuild_spacing]
        patient_origin = [new_st_origin[0], new_st_origin[1], new_ed_origin[2]]
        total = math.floor((new_ed_origin[2] - slab_thickness - new_st_origin[2]) / rebuild_spacing)
        k_idx = total - (ijk_to_xyz(old_ijk, patient_origin, new_spacing, M)[2])
    elif azimuth_type == 2:
        # 冠 Y
        print(M)
        new_ed_origin = np.dot(ed_origin, M).tolist()
        new_st_origin = np.dot(st_origin, M).tolist()
        print(f'******st_origin: {st_origin}, ed_origin: {ed_origin}')
        print(f'******new_st_origin: {new_st_origin}, new_ed_origin: {new_ed_origin}')
        new_spacing = [rebuild_spacing, rebuild_spacing, rebuild_spacing]
        patient_origin = [st_origin[0], st_origin[1], new_ed_origin[1]]
        # k_idx = img.TransformPhysicalPointToIndex(old_ijk)[1]
        k_idx = (ijk_to_xyz(old_ijk, patient_origin, new_spacing, M)[2])
        total = math.floor((new_ed_origin[2] - slab_thickness - new_st_origin[2]) / rebuild_spacing)

    elif azimuth_type == 3:
        # 失 X
        direction = np.array([[1,0,0], [0,1,0], [0,0,-1]])
        M = np.dot(M, direction)
        new_ed_origin = np.dot(ed_origin, M).tolist()
        new_st_origin = np.dot(st_origin, M).tolist()
        print(f'******st_origin: {st_origin}, ed_origin: {ed_origin}')
        print(f'*************new_st_origin: {new_st_origin}, new_ed_origin: {new_ed_origin}')
        new_spacing = [rebuild_spacing, rebuild_spacing, rebuild_spacing]
        # k_idx = img.TransformPhysicalPointToIndex(old_ijk)[0]
        patient_origin = [new_ed_origin[2], new_st_origin[0], new_ed_origin[1]]
        k_idx = ijk_to_xyz(old_ijk, patient_origin, new_spacing, M)[2]
        # 这个为什么是1？？？        
        total = math.floor((new_ed_origin[2] - slab_thickness - new_st_origin[2]) / rebuild_spacing)
    else:
        raise "azimuth_type input error"
    print(f'*************total: {total},k_idx:{k_idx}, patient_origin: {patient_origin}')
    return total, k_idx, patient_origin


def get_mpr(
    slab_thickness: float,
    rebuild_spacing: float,
    matrix: list,
    rebuild_type: str,
    azimuth_type: str,
) -> dict:
    res = {}
    nii_path = "/media/tx-deepocean/Data/DICOMS/demos/1.3.12.2.1107.5.1.4.105566.30000022070623520555500057495.nii.gz"
    dcm_path = "/media/tx-deepocean/Data/DICOMS/ct_cerebral/CN010002-13696724/1.2.840.113619.2.416.10634142502611409964348085056782520111/1.2.840.113619.2.416.77348009424380358976506205963520437809/1.2.840.113619.2.416.2106685279137426449336212617073446717.1"
    ds = pydicom.read_file(dcm_path, stop_before_pixels=True, force=True)
    img = sitk.ReadImage(nii_path)
    img_arr = sitk.GetArrayFromImage(img)  # (z,y,x)
    img_shape = img_arr.shape
    print(f'*****shape: {img_shape}')

    t0 = time.time()
    spacing = img.GetSpacing()
    st_origin = list(img.GetOrigin())
    print(f'******st_origin： {st_origin},spacing: {spacing}')
    reader = vtk_image_from_array(img_arr, spacing, st_origin)

    # 获取物理信息
    extent = reader.GetOutput().GetExtent()  # 维度
    print(f"extent: {extent}")  # (x, y, z)

    # TODO: 根据层厚 slab_thickness 层间距 原点，计算重建层数起止位置 done
    rotate_m = np.array([[matrix[0], matrix[1], matrix[2]], [matrix[4], matrix[5], matrix[6]], [matrix[8], matrix[9], matrix[10]]])
    direction = np.array([[-1,0,0],[0,-1,0], [0,0,1]])
    rotate_m = np.dot(rotate_m, direction)
    matrix_trans = rotate_m
    # ROI 体心 ijk
    ijk = [matrix[3], matrix[7], matrix[11]]
    # 技术问题, 未找到旋转后spacing计算方法, 写死
    rotate_spacing = [0.5, 0.5, 0.5]
    ed_origin = [            
        st_origin[0] + spacing[0] * img_shape[2],
        st_origin[1] + spacing[1] * img_shape[1],
        st_origin[2] + spacing[2] * img_shape[0],]
    # 根据层厚 SliceThickness 层间距 计算 mip 各方位总层数, SpacingBetweenSlices
    total,k_idx, _ = get_total_by_matrix(img, rebuild_spacing, slab_thickness, azimuth_type, matrix, img_shape, spacing, st_origin)

    if azimuth_type == 1:
        physical_idx = ijk[2]
        # 计算旋转前的 k_idx
        k_idx = img.TransformPhysicalPointToIndex(ijk)[2]
        print(f'**************k_idx: {k_idx}')
        # 变换后的 总长度
        rotate_st_origin = np.dot(st_origin, rotate_m).tolist()
        rotate_ed_origin = np.dot(ed_origin, rotate_m).tolist()
        print(f'******rotate_st_origin: {rotate_st_origin}, rotate_ed_origin: {rotate_ed_origin}')
        # 暂时注释，和前端联调看下效果
        patient_origin = [rotate_st_origin[0], rotate_st_origin[1], rotate_ed_origin[2]]
        # 根据 slab_thickness 和 physical_idx 获取 mip 区间
        #  左面不够 0.5 * 重建厚度
        left_bounding = [rotate_st_origin[0], rotate_st_origin[1], rotate_st_origin[2] + slab_thickness]
        right_bounding = [rotate_st_origin[0], rotate_st_origin[1], rotate_ed_origin[2] - slab_thickness]
        min_idx, max_idx = get_rebuild_range(patient_origin, rotate_st_origin, rotate_ed_origin, rotate_spacing, rotate_m, left_bounding, right_bounding, physical_idx, slab_thickness, azimuth_idx=2)
    elif azimuth_type == 2:
        # 计算旋转前的 k_idx
        # old_ijk = np.dot(np.linalg.inv(rotate_m), ijk)
        k_idx = img.TransformPhysicalPointToIndex(ijk)[1]
        print(f'k_idx: {k_idx}')
        # 变换后的 总长度
        rotate_st_origin = np.dot(st_origin, rotate_m).tolist()
        rotate_ed_origin = np.dot(ed_origin, rotate_m).tolist()
        print(f'********rotate_st_origin: {rotate_st_origin}, rotate_ed_origin:  {rotate_ed_origin}')
        patient_origin = [st_origin[0], st_origin[1], rotate_ed_origin[1]]
        physical_idx = ijk[1]
        print(f'*****physical_idx: {physical_idx}')
        left_bounding = [rotate_st_origin[0], rotate_st_origin[1] + slab_thickness, rotate_st_origin[2]]
        right_bounding = [rotate_ed_origin[0], rotate_ed_origin[1] - slab_thickness, rotate_ed_origin[2]]
        tmp1 = [rotate_st_origin[0], physical_idx - 0.5 * slab_thickness, rotate_st_origin[2]]
        tmp2 = [rotate_st_origin[0], physical_idx + 0.5 * slab_thickness, rotate_st_origin[2]]
        if (physical_idx - 0.5 * slab_thickness) < rotate_st_origin[2]:
            print(11111111111111)
            min_idx = ijk_to_xyz(
                rotate_st_origin, patient_origin, rotate_spacing, rotate_m
            )[2]
            max_idx = ijk_to_xyz(
                left_bounding,
                patient_origin,
                rotate_spacing,
                rotate_m
            )[2]
        # 右面不够 0.5 * 重建厚度
        elif (physical_idx + 0.5 * slab_thickness) > rotate_ed_origin[2]:
            print(2222222222222)
            max_idx = ijk_to_xyz(
                rotate_ed_origin, patient_origin, rotate_spacing, rotate_m
            )[2]
            min_idx = ijk_to_xyz(
                right_bounding,
                patient_origin,
                rotate_spacing,
                rotate_m
            )[2]
        else:
            print(33333333333333)
            print(f'tmp1, tmp2: {tmp1}, {tmp2}')
            min_idx = ijk_to_xyz(
                tmp1, patient_origin, rotate_spacing, rotate_m
            )[2]
            max_idx = ijk_to_xyz(
                tmp2, patient_origin, rotate_spacing, rotate_m
            )[2]
        # min_idx, max_idx = get_rebuild_range(patient_origin, rotate_st_origin, rotate_ed_origin, rotate_spacing, M, left_bounding, right_bounding, physical_idx, slab_thickness, azimuth_idx=1)
    elif azimuth_type == 3:
        # 失 X
        # old_ijk = np.dot(ijk, np.linalg.inv(rotate_m))
        k_idx = img.TransformPhysicalPointToIndex(ijk)[0]
        print(f'****k_idx: {k_idx}')
        direction = np.array([[1,0,0], [0,1,0], [0,0,-1]])
        rotate_m = np.dot(rotate_m, direction)
        print(f'*****rotate_m: {rotate_m}')
        # 变换后的 总长度
        rotate_st_origin = np.dot(st_origin, rotate_m).tolist()
        rotate_ed_origin = np.dot(ed_origin, rotate_m).tolist()
        print(f'*rotate_st_origin: {rotate_st_origin}, rotate_ed_origin: {rotate_ed_origin}')
        patient_origin = [rotate_ed_origin[2], rotate_st_origin[0], rotate_ed_origin[1]]
        physical_idx = ijk[0]
        left_bounding = [rotate_st_origin[2] + slab_thickness, rotate_st_origin[0], rotate_st_origin[1]]
        right_bounding = [rotate_ed_origin[2] - slab_thickness, rotate_ed_origin[0], rotate_ed_origin[1]]
        tmp1 = [physical_idx - 0.5 * slab_thickness, patient_origin[1], patient_origin[2]]
        tmp2 = [physical_idx + 0.5 * slab_thickness, patient_origin[1], patient_origin[2]]
        if (physical_idx - 0.5 * slab_thickness) < rotate_st_origin[2]:
            print(11111111111111)
            min_idx = ijk_to_xyz(
                rotate_st_origin, patient_origin, rotate_spacing, rotate_m
            )[0]
            max_idx = ijk_to_xyz(
                left_bounding,
                patient_origin,
                rotate_spacing,
                rotate_m
            )[0]
        # 右面不够 0.5 * 重建厚度
        elif (physical_idx + 0.5 * slab_thickness) > rotate_ed_origin[2]:
            print(2222222222222)
            max_idx = ijk_to_xyz(
                rotate_ed_origin, patient_origin, rotate_spacing, rotate_m
            )[0]
            min_idx = ijk_to_xyz(
                right_bounding,
                patient_origin,
                rotate_spacing,
                rotate_m
            )[0]
        else:
            print(33333333333333)
            print(f'tmp1, tmp2: {tmp1}, {tmp2}')
            min_idx = ijk_to_xyz(
                tmp1, patient_origin, rotate_spacing, rotate_m
            )[0]
            max_idx = ijk_to_xyz(
                tmp2, patient_origin, rotate_spacing, rotate_m
            )[0]
            # min_idx, max_idx = get_rebuild_range(patient_origin,  rotate_st_origin, rotate_ed_origin, rotate_spacing, M, left_bounding, right_bounding, physical_idx, slab_thickness, azimuth_idx=0)
    else:
        raise "azimuth_type input error"
    slice_nums = max_idx - min_idx
    print(f"slice_nums: {slice_nums}， min_idx: {min_idx}, max_idx: {max_idx}")
    
    resliceAxes = vtk.vtkMatrix4x4()
    resliceAxes.DeepCopy(matrix)
    for i in range(3):
        for j in range(3):
            resliceAxes.SetElement(i, j, matrix_trans[i][j])
            resliceAxes.SetElement(i, j, matrix_trans[i][j])
            resliceAxes.SetElement(i, j, matrix_trans[i][j])
    reslice = vtk.vtkImageSlabReslice()
    # reslice.SetOutputSpacing([spacing[0], spacing[1], rebuild_spacing])
    reslice.SetInputConnection(reader.GetOutputPort())  # 截取的体数据
    reslice.SetOutputDimensionality(2)  # 设置输出为2维图片
    reslice.SetInterpolationModeToLinear()  # 差值

    if rebuild_type == "MIP":
        reslice.SetBlendModeToMax()
    elif rebuild_type == "MinIP":
        reslice.SetBlendModeToMin()
    elif rebuild_type == "AIP":
        reslice.SetBlendModeToMean()
    reslice.SetSlabThickness(slice_nums)  # 设置厚度
    reslice.SetResliceAxes(resliceAxes)  # 设置矩阵
    # 由SetBackgroundColor或SetBackgroundLevel控制的位于Volume外部的像素的默认值。
    # SetResliceAxesOrigin()方法还可以用于提供切片将通过的(x，y，z)点。
    # reslice.SetResliceAxesOrigin(ijk)
    reslice.SetBackgroundLevel(-99999)
    reslice.Update()
    # origin_tmp = vtk.vtkDataObject().ORIGIN()
    # spacing_tmp = vtk.vtkDataObject().SPACING()
    # tmp =  [0,0,0]
    # reslice.GetOutputInformation(0).Get(origin_tmp, tmp)
    # print(f'tmp: {tmp}')

    # 落盘
    vtk_img_export = vtkImageExportToArray()
    vtk_img_export.SetInputConnection(reslice.GetOutputPort())
    np_array = vtk_img_export.GetArray()

    np_array = np_array.astype(np.int16)
    save_path = "./mip.dcm"
    result = sitk.GetImageFromArray(np_array)
    result.SetMetaData("0028|1050", "300")
    result.SetMetaData("0028|1051", "800")
    sitk.WriteImage(result, save_path)
    # np_array_to_dcm(ds, save_path, np_array, azimuth_type, matrix, st_origin)
    res["dcm_path"] = f"/tmp/{rebuild_type}.dcm"

    # res["boxes"] = boxes
    # res["contour"] = contour_res
    # res["idx"] = idx
    print(f"a mip cost: {time.time() - t0}")
    return res

def get_rebuild_range(patient_origin, rotate_st_origin,rotate_ed_origin, rotate_spacing, M, left_bounding,right_bounding, physical_idx, slab_thickness, azimuth_idx=2):
    """"""
    if azimuth_idx == 2:
        tmp1 = [rotate_st_origin[0], rotate_st_origin[1], physical_idx - 0.5 * slab_thickness]
        tmp2 = [rotate_st_origin[0], rotate_st_origin[1], physical_idx + 0.5 * slab_thickness]
    elif azimuth_idx == 1:
        tmp1 = [rotate_st_origin[0], physical_idx - 0.5 * slab_thickness, rotate_st_origin[2]]
        tmp2 = [rotate_st_origin[0], physical_idx + 0.5 * slab_thickness, rotate_st_origin[2]]
    else:
        tmp1 = [physical_idx - 0.5 * slab_thickness, rotate_st_origin[2], rotate_st_origin[0]]
        tmp2 = [physical_idx + 0.5 * slab_thickness, rotate_st_origin[2], rotate_st_origin[0]]

    if (physical_idx - 0.5 * slab_thickness) < rotate_st_origin[2]:
        print(11111111111111)
        min_idx = ijk_to_xyz(
            rotate_st_origin, patient_origin, rotate_spacing, M
        )[2]
        max_idx = ijk_to_xyz(
            left_bounding,
            patient_origin,
            rotate_spacing,
            M
        )[2]
    # 右面不够 0.5 * 重建厚度
    elif (physical_idx + 0.5 * slab_thickness) > rotate_ed_origin[2]:
        print(2222222222222)
        max_idx = ijk_to_xyz(
            rotate_ed_origin, patient_origin, rotate_spacing, M
        )[2]
        min_idx = ijk_to_xyz(
            right_bounding,
            patient_origin,
            rotate_spacing,
            M
        )[2]
    else:
        print(33333333333333)
        print(f'tmp1, tmp2: {tmp1}, {tmp2}')
        min_idx = ijk_to_xyz(
            tmp1, patient_origin, rotate_spacing, M
        )[2]
        max_idx = ijk_to_xyz(
            tmp2, patient_origin, rotate_spacing, M
        )[2]
    return min_idx, max_idx

def ijk_to_xyz(point_ijk, origin, spacing, direction):
    """物理坐标转像素坐标"""
    # index = inv(direction * spacing) * (point_ijk - origin)
    # index = (P-O)/D*S
    spacing = np.diag(spacing)
    tmp = np.abs(np.dot(np.linalg.inv(np.dot(direction, spacing)), np.subtract(point_ijk, origin)))
    xyz = np.int0(tmp)
    print(xyz)
    return xyz


def np_array_to_dcm(ds: pydicom.FileDataset, save_path:str, np_array:np.ndarray, azimuth_type: str, matrix: list, st_origin: list):
    """根据重建信息重新计算各个方位的tag
    Args:
        st_origin: 原体数据的第一张左上点 [-xx, -xxx, -xxx]
    """
    ds.WindowCenter = 300
    ds.WindowWidth = 800
    # np_array 需要更改这个 shape
    ds.Rows = np_array.shape[1]
    ds.Columns = np_array.shape[2]
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    # TODO: 重新计算 ImagePositionPatient, orientation, slicelocation, PixelSpacing
    if azimuth_type == 1:
        ds.SliceLocation = matrix[11]
        ds.ImagePositionPatient = [st_origin[0], st_origin[1], matrix[11]]
        ds.ImageOrientationPatient = [matrix[0], matrix[4], matrix[8], matrix[1], matrix[5], matrix[9]]
    if azimuth_type == 2:
        ds.SliceLocation = matrix[7]
        ds.ImagePositionPatient = [st_origin[0], matrix[7], st_origin[2]]
        ds.ImageOrientationPatient = [matrix[0], matrix[4], matrix[8], matrix[2], matrix[6], matrix[10]]
    if azimuth_type == 3:
        # x 为失状轴
        ds.SliceLocation = matrix[3]
        ds.ImagePositionPatient = [matrix[3], st_origin[1], st_origin[2]]
        ds.ImageOrientationPatient = [matrix[1], matrix[5], matrix[9], matrix[2], matrix[6], matrix[10]]

    ds.PixelSpacing = [0.5, 0.5, 0.5]
    ds.PixelData = np_array.tobytes()
    ds.is_implicit_VR = True
    ds.save_as(save_path)


def find_contours(mask: np.ndarray, label: int, position: int, idx: int) -> list:
    """获取某层的 mpr 的contour
    Args:
        mask: 3d array
        label: 需要找出轮廓的标签
        position: 1，2, 3(轴冠矢)
    """
    if position == 1:
        slice_arr = mask[idx, :, :]
    elif position == 2:
        slice_arr = mask[:, idx, :]
    elif position == 3:
        slice_arr = mask[:, :, idx]
    else:
        raise "position not found"
    contour, bboxes = find_2d_contours(slice_arr, label)
    return contour, bboxes

def find_2d_contours(slice_arr, label):
    """后面可根据具体需要可过滤部分 contour"""
    binary_arr = np.zeros(slice_arr.shape)
    binary_arr[slice_arr == 100] = 1
    # 获取对应方位2d层
    binary_arr = binary_arr.astype(np.uint8)
    # gray = cv.cvtColor(slice_arr, cv.COLOR_GRAY2BGR)
    # binary_arr = cv.threshold(binary_arr, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    # 确认 函数返回几个值
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, bboxes = [], []
    contours = []
    contours, hier = cv2.findContours(
        binary_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    print(f"contour 个数: {len(contours)}")
    if len(contours) == 0:
        return cnts, bboxes
    for contour in contours:
        # 返回最小外接矩形
        rect = cv2.minAreaRect(contour)
        # 获取四个顶点
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        x, y, w, h = cv2.boundingRect(contour)
        # img = cv.rectangle(slice_arr, (x, y), (x+w, y+h), (255), 1)
        # 画 bbox
        # img = cv.drawContours(binary_arr, [box], 0, (255), 1)
        # 画 contour
        # img = cv.drawContours(binary_arr, [contour], 0, (255), 1)
        cnts.append(contour.tolist())
        bboxes.append(box.tolist())
    return cnts, bboxes

def main():
    slab_thickness = 0.5
    scale = 1
    # TODO: 层厚小于 1mm 情况前端控制？
    # 确保matrix 坐标系正确
    # 轴
    # matrix = [
    #     0, 0, -1, 0.249023,
    #     -1, 0, 0, -176.5, 
    #     0, 1, 0, -623.05,
    #     0, 0, 0, 1 
    # ]
    # 冠任意
    matrix = [
    -0.458185, 0, -0.888857, 9.9871, 
    -0.888857, 0, 0.458185, -182.049, 
    0, 1, 0, -623.05,
    0, 0, 0, 1 
    ]
    # 矢任意 ijk要乘(手动)[-1, -1, 1], M 也要乘[-1,-1,1]
    # matrix = [
    # 0.888857, 0, -0.458185, -35.5807, 
    # -0.458185, 0, -0.888857, -245.713 ,
    # 0, 1, 0, -623.05, 
    # 0, 0, 0, 1,
    # ]  
    # 轴任意

    res = get_mpr(
        slab_thickness,
        slab_thickness * scale,
        matrix,
        rebuild_type="MIP",
        azimuth_type=2,
    )
    # print(json.dumps(res))


if __name__ == "__main__":
    main()
