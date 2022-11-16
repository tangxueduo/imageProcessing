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
    rebuild_spacing: float, slab_thickness: float, azimuth_type: str, matrix: list, img_shape: list, spacing: list, old_origin: list
):
    """根据matrix 方位， 返回重建后层数
    Args:
        slab_thickness: 20mm, 重建层厚
        azimuth_type: 1, 2, 3(轴冠矢)
    """
    total = 0
    M = np.array([[matrix[0], matrix[4], matrix[8]], [matrix[1], matrix[5], matrix[9]], [matrix[2], matrix[6], matrix[10]]])
    # TODO: 获取旋转后最后一张的左上角的原点
    ed_origin = [old_origin[0]+img_shape[0]+spacing[0], old_origin[1]+img_shape[1]*spacing[1], old_origin[2]+ img_shape[2]*spacing[2]]
    new_ed_origin = np.dot(ed_origin, M).tolist()
    new_st_origin = np.dot(old_origin, M).tolist()
    if azimuth_type == "axial":
        # TODO: 确认重建后的 spacing [0.5, 0.5, 0.5]
        total = math.floor((new_ed_origin[2] - slab_thickness - new_st_origin[2]) / rebuild_spacing)
        # 获取旋转后 origin 范围
    elif azimuth_type == "coronal":
        # 冠 Y
        total = math.floor((new_ed_origin[1] - slab_thickness - new_st_origin[1]) / rebuild_spacing)
    elif azimuth_type == "sagittal":
        # 失 X
        total = math.floor((new_ed_origin[0] - slab_thickness - new_st_origin[0]) / rebuild_spacing)
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
    nii_path = "/media/tx-deepocean/Data/DICOMS/RESULT/volume/1.2.840.113619.2.416.77348009424380358976506205963520437809.nii.gz"
    dcm_path = "/media/tx-deepocean/Data/DICOMS/ct_cerebral/CN010002-13696724/1.2.840.113619.2.416.10634142502611409964348085056782520111/1.2.840.113619.2.416.77348009424380358976506205963520437809/1.2.840.113619.2.416.2106685279137426449336212617073446717.1"
    ds = pydicom.read_file(dcm_path, stop_before_pixels=True, force=True)
    # https://blog.csdn.net/qq_41023026/article/details/118891837
    img = sitk.ReadImage(nii_path)
    img_arr = sitk.GetArrayFromImage(img)  # (z,y,x)
    img_shape = img_arr.shape
    print(img_shape)

    # 根据层厚 SliceThickness 层间距 计算 mip 各方位总层数, SpacingBetweenSlices
    # total = get_total_by_matrix(rebuild_spacing, slab_thickness, azimuth_type, matrix, img_shape, spacing, st_orgin, ed_origin)
    t0 = time.time()
    spacing = img.GetSpacing()
    st_origin = img.GetOrigin()
    print(f'******st_origin： {st_origin}')
    
    reader = vtk_image_from_array(img_arr, spacing, st_origin)

    # 获取物理信息
    extent = reader.GetOutput().GetExtent()  # 维度
    print(f"extent: {extent}")  # (x, y, z)

    # TODO: 根据层厚 slab_thickness 层间距 原点，计算重建层数起止位置 done
    rotate_m = np.array([[matrix[0], matrix[4], matrix[8]], [matrix[1], matrix[5], matrix[9]], [matrix[2], matrix[6], matrix[10]]])
    print(rotate_m)
    # ROI 体心 ijk
    ijk = [matrix[3], matrix[7], matrix[11]]

    # 技术问题, 未找到旋转后spacing计算方法, 写死
    rotate_spacing = [0.5, 0.5, 0.5]
    ed_origin = [            
        st_origin[0] + spacing[0] * img_shape[2],
        st_origin[1] + spacing[1] * img_shape[1],
        st_origin[2] + spacing[2] * img_shape[0],]
    if azimuth_type == 1:
        # 最后一张左上点
        # ed_origin = [
        #     st_origin[0],
        #     st_origin[1],
        #     st_origin[2] + spacing[2] * img_shape[2],
        # ]
        print(f'******ed_origin: {ed_origin}')
        physical_idx = ijk[2]
        print(f'*****physical_idx: {physical_idx}')

        # RAS_to_LPS 联调时注释下面代码
        direction = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        rotate_m = np.dot(direction, rotate_m)
        # 计算旋转前的 k_id
        old_ijk = np.dot(ijk, np.linalg.inv(rotate_m))
        k_idx = img.TransformPhysicalPointToIndex(old_ijk)[2]
        print(f'**************k_idx: {k_idx}')
        t1 = time.time()
        contours, boxes = find_contours(img_arr, 100, azimuth_type, k_idx)
        print(f'*****正交contour耗时: {time.time() - t1}')
        # 变换后的 总长度
        rotate_st_origin = np.dot(st_origin, rotate_m).tolist()
        rotate_ed_origin = np.dot(ed_origin, rotate_m).tolist()
        # 暂时注释，和前端联调看下效果
        # rotate_st_origin[0], rotate_st_origin[1] = st_origin[:2]
        # rotate_ed_origin[0], rotate_ed_origin[1] = st_origin[:2]
        print(f'********旋转后: {rotate_st_origin}, {rotate_ed_origin}')

        # 根据 slab_thickness 和 physical_idx 获取 mip 区间
        #  左面不够 0.5 * 重建厚度
        left_bounding = [rotate_st_origin[0], rotate_st_origin[1], rotate_st_origin[2] + slab_thickness]
        right_bounding = [rotate_st_origin[0], rotate_st_origin[1], rotate_ed_origin[2] - slab_thickness]
        min_idx, max_idx = get_rebuild_range(rotate_st_origin, rotate_ed_origin, rotate_spacing, rotate_m, left_bounding, right_bounding, physical_idx, slab_thickness, azimuth_idx=2)


    elif azimuth_type == 2:
        # 最后一张左上点
        # ed_origin = [
        #     st_origin[0],
        #     st_origin[1] + spacing[1] * img_shape[1],
        #     st_origin[2],
        # ]
        direction = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        rotate_m = np.dot(direction, rotate_m)
        # 计算旋转前的 k_id
        old_ijk = np.dot(ijk, np.linalg.inv(rotate_m))
        k_idx = img.TransformPhysicalPointToIndex(old_ijk)[1]
        t1 = time.time()
        contours, boxes = find_contours(img_arr, 100, azimuth_type, k_idx)
        # 变换后的 总长度
        rotate_st_origin = np.dot(st_origin, rotate_m).tolist()
        rotate_ed_origin = np.dot(ed_origin, rotate_m).tolist()
        # rotate_st_origin[0], rotate_st_origin[2] = st_origin[0], st_origin[2] 
        # rotate_ed_origin[0], rotate_ed_origin[2] = st_origin[0], st_origin[2]
        print(f'********旋转后: {rotate_st_origin}, {rotate_ed_origin}')

        physical_idx = ijk[1]
        print(f'*****physical_idx: {physical_idx}')

        left_bounding = [rotate_st_origin[0], rotate_st_origin[1] + slab_thickness, rotate_st_origin[2]]
        right_bounding = [rotate_ed_origin[0], rotate_ed_origin[1] - slab_thickness, rotate_ed_origin[2]]
        min_idx, max_idx = get_rebuild_range(rotate_st_origin, rotate_ed_origin, rotate_spacing, rotate_m, left_bounding, right_bounding, physical_idx, slab_thickness, azimuth_idx=1)
    elif azimuth_type == 3:
        # 最后一张左上点
        # ed_origin = [
        #     st_origin[0] + spacing[0] * img_shape[0],
        #     st_origin[1],
        #     st_origin[2],
        # ]
        direction = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        rotate_m = np.dot(direction, rotate_m)
        # 计算旋转前的 k_id
        old_ijk = np.dot(ijk, np.linalg.inv(rotate_m))
        k_idx = img.TransformPhysicalPointToIndex(old_ijk)[0]
        t1 = time.time()
        contours, boxes = find_contours(img_arr, 100, azimuth_type, k_idx)
        # 变换后的 总长度
        rotate_st_origin = np.dot(st_origin, rotate_m).tolist()
        rotate_ed_origin = np.dot(ed_origin, rotate_m).tolist()
        # rotate_st_origin[1], rotate_st_origin[2] = st_origin[1:]
        # rotate_ed_origin[1], rotate_ed_origin[2] = st_origin[1:]
        print(f'********旋转后: {rotate_st_origin}, {rotate_ed_origin}')

        physical_idx = ijk[0]
        print(f'*****physical_idx: {physical_idx}')
        left_bounding = [rotate_st_origin[0] + slab_thickness, rotate_st_origin[1], rotate_st_origin[2]]
        right_bounding = [rotate_ed_origin[0] - slab_thickness, rotate_ed_origin[1], rotate_ed_origin[2]]
        min_idx, max_idx = get_rebuild_range(rotate_st_origin, rotate_ed_origin, rotate_spacing, rotate_m, left_bounding, right_bounding, physical_idx, slab_thickness, azimuth_idx=0)
    else:
        raise "azimuth_type input error"
    slice_nums = max_idx - min_idx
    print(f"slice_nums: {slice_nums}， min_idx: {min_idx}, max_idx: {max_idx}")
    
    resliceAxes = vtk.vtkMatrix4x4()
    resliceAxes.DeepCopy(matrix)
    reslice = vtk.vtkImageSlabReslice()
    # setspacing 对最后出图无影响..？
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
    reslice.SetBackgroundLevel(-9999)
    reslice.Update()
    
    # 落盘
    vtk_img_export = vtkImageExportToArray()
    vtk_img_export.SetInputConnection(reslice.GetOutputPort())
    print(reader.GetOutput().GetExtent())
    np_array = vtk_img_export.GetArray()
    # print(np.min(np_array), np_array.shape)

    np_array = np_array.astype(np.int16)
    save_path = "./mip.dcm"
    np_array_to_dcm(ds, save_path, np_array, azimuth_type, matrix, st_origin)
    res["dcm_path"] = f"/tmp/{rebuild_type}.dcm"

    # res["boxes"] = boxes
    # res["contour"] = contour_res
    # res["idx"] = idx
    print(f"a mip cost: {time.time() - t0}")
    return res

def get_rebuild_range(rotate_st_origin, rotate_ed_origin, rotate_spacing, rotate_m, left_bounding,right_bounding, physical_idx, slab_thickness, azimuth_idx=2):
    """"""
    if azimuth_idx == 2:
        tmp1 = [rotate_st_origin[0], rotate_st_origin[1], physical_idx - 0.5 * slab_thickness]
        tmp2 = [rotate_st_origin[0], rotate_st_origin[1], physical_idx + 0.5 * slab_thickness]
    elif azimuth_idx == 1:
        tmp1 = [rotate_st_origin[0], physical_idx - 0.5 * slab_thickness, rotate_st_origin[2]]
        tmp2 = [rotate_st_origin[0], physical_idx + 0.5 * slab_thickness, rotate_st_origin[2]]
    else:
        tmp1 = [physical_idx - 0.5 * slab_thickness, rotate_st_origin[1], rotate_st_origin[2]]
        tmp2 = [physical_idx + 0.5 * slab_thickness, rotate_st_origin[1], rotate_st_origin[2]]

    if (physical_idx - 0.5 * slab_thickness) < rotate_st_origin[azimuth_idx]:
        print(11111111111111)
        min_idx = ijk_to_xyz(
            rotate_st_origin, rotate_st_origin, rotate_spacing, rotate_m
        )[2]
        max_idx = ijk_to_xyz(
            left_bounding,
            rotate_st_origin,
            rotate_spacing,
            rotate_m
        )[2]
    # 右面不够 0.5 * 重建厚度
    elif (physical_idx + 0.5 * slab_thickness) > rotate_ed_origin[azimuth_idx]:
        print(2222222222222)
        max_idx = ijk_to_xyz(
            rotate_ed_origin, rotate_st_origin, rotate_spacing, rotate_m
        )[2]
        min_idx = ijk_to_xyz(
            right_bounding,
            rotate_st_origin,
            rotate_spacing,
            rotate_m
        )[2]
    else:
        print(33333333333333)
        print(f'tmp1, tmp2: {tmp1}, {tmp2}')
        min_idx = ijk_to_xyz(
            tmp1, rotate_st_origin, rotate_spacing, rotate_m
        )[2]
        max_idx = ijk_to_xyz(
            tmp2, rotate_st_origin, rotate_spacing, rotate_m
        )[2]
    return min_idx, max_idx

def ijk_to_xyz(point_ijk, origin, spacing, direction):
    """物理坐标转像素坐标"""
    # index = inv(direction * spacing) * (point_ijk - origin)
    # index = (P-O)/D*S
    spacing = np.diag(spacing)
    tmp = np.abs(np.dot(np.linalg.inv(np.dot(direction, spacing)), np.abs(np.subtract(point_ijk, origin))))
    xyz = np.int0(tmp)
    print(xyz)
    return xyz


def np_array_to_dcm(ds: pydicom.FileDataset, save_path:str, np_array:np.ndarray, azimuth_type: str, matrix: list, old_origin: list):
    """根据重建信息重新计算各个方位的tag
    Args:
        old_origin: 原体数据的第一张左上点 [-xx, -xxx, -xxx]
    """
    ds.WindowCenter = 40
    ds.WindowWidth = 400
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
        ds.ImagePositionPatient = [old_origin[0], old_origin[1], matrix[11]]
        ds.ImageOrientationPatient = [matrix[0], matrix[4], matrix[8], matrix[1], matrix[5], matrix[9]]
    if azimuth_type == 2:
        ds.SliceLocation = matrix[7]
        ds.ImagePositionPatient = [old_origin[0], matrix[7], old_origin[2]]
        ds.ImageOrientationPatient = [matrix[0], matrix[4], matrix[8], matrix[2], matrix[6], matrix[10]]
    if azimuth_type == 3:
        # x 为失状轴
        ds.SliceLocation = matrix[3]
        ds.ImagePositionPatient = [matrix[3], old_origin[1], old_origin[2]]
        ds.ImageOrientationPatient = [matrix[1], matrix[5], matrix[9], matrix[2], matrix[6], matrix[10]]

    ds.PixelSpacing = [0.5, 0.5, 0.5]
    ds.PixelData = np_array.tobytes()
    ds.is_implicit_VR = True
    ds.save_as(save_path)



def get_physical(series_path):

    image_shape = (original_ds.Columns, original_ds.Rows, len(dcm_list))
    return


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
    series_path = "/media/tx-deepocean/Data/DICOMS/ct_cerebral/CN010023-1510071117/1.3.46.670589.33.1.63780596181185791300001.5349461926886765269/1.3.46.670589.33.1.63780596345132168500001.4829782583723493361"
    slab_thickness = 0.5
    scale = 1
    # TODO: 层厚小于 1mm 情况前端控制？
    # 确保matrix 坐标系正确
    matrix =[
        -0.88373, 0, -0.467998, -70.5794,
        0, 1, 0, -6.81277,
        -0.467998, 0, 0.88373, 194.274,
        0, 0, 0, 1 
        ]
    # matrix = [
    #     -1, 0, 0, 0.279233,
    #     0, 0, 1, -6.53347,
    #     0, 1, 0, 60.401,
    #     0, 0, 0, 1 
    # ]
    # matrix = [
    #     0, -0.467998, -0.88373, -6.4e-05,
    #     -1, 0, 0, -6.81277,
    #     0, 0.88373, -0.467998, 60.401,
    #     0, 0, 0, 1]
    res = get_mpr(
        slab_thickness,
        slab_thickness * scale,
        matrix,
        rebuild_type="MIP",
        azimuth_type=1,
    )
    # print(json.dumps(res))


if __name__ == "__main__":
    main()
