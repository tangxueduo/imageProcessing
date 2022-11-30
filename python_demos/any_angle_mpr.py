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

class MPR():
    def __init__(self):
        self.x = [1, 0, 0]
        self.y = [0 ,1, 0]
        self.z = [0, 0, -1]
        self.center = [0, 0, 0]

    def get_total_by_matrix(
        self, rebuild_spacing: float, 
        slab_thickness: float, 
        azimuth_type: int, 
        matrix: list, 
        img_shape: list, 
        spacing: list, 
        st_origin: list
    ):
        """根据matrix 方位， 返回重建后层数
        Args:
            slab_thickness: 20mm, 重建层厚
            azimuth_type: 1, 2, 3(轴冠矢)
        """
        total = 0
        M = np.array([[matrix[0], matrix[1], matrix[2]], [matrix[4], matrix[5], matrix[6]], [matrix[8], matrix[9], matrix[10]]])
        old_ijk = np.array([matrix[3],matrix[7], matrix[11]])
        ed_origin = np.array([st_origin[0]+img_shape[2]*spacing[0], st_origin[1]+img_shape[1]*spacing[1], st_origin[2]+ img_shape[0]*spacing[2]])
        # 获取patient_origin
        if azimuth_type == 1:
            patient_origin = [st_origin[0], st_origin[1], ed_origin[2]]
        elif azimuth_type == 2:
            patient_origin = [st_origin[0], st_origin[1], ed_origin[2]]
        elif azimuth_type == 3:
            patient_origin = [ed_origin[0], st_origin[1], ed_origin[2]]
        center_origin = [st_origin[0], st_origin[1], ed_origin[2]]
        step_MM =self._getObliqueThicknessMM(matrix, spacing, center_origin)
        obliqueLocal = self._getObliqueLocal(matrix, center_origin)
        print(f'*************step_MM: {step_MM}, obliqueLocal: {obliqueLocal}')

        flat_normal = obliqueLocal[2]
        # print(f'*****flat_normal: {flat_normal}')
        box_min_MM = 0.5 * np.array(spacing)
        box_size_MM = np.multiply(spacing, img_shape[::-1])
        box_size_MM = np.subtract(box_size_MM, box_min_MM)
        # print(f'*****box_size_MM: {box_size_MM}, box_min_MM: {box_min_MM}')

        b_min = [
            box_size_MM[0] if flat_normal[0] < 0 else box_min_MM[0],
            box_size_MM[1] if flat_normal[1] < 0 else box_min_MM[1],
            box_size_MM[2] if flat_normal[2] < 0 else box_min_MM[2],
        ]
        b_max = [
            box_size_MM[0] if flat_normal[0] > 0 else box_min_MM[0],
            box_size_MM[1] if flat_normal[1] > 0 else box_min_MM[1],
            box_size_MM[2] if flat_normal[2] > 0 else box_min_MM[2],
        ]
        # print(f'*******b_min: {b_min}')
        # print(f'*******b_max: {b_max}')
        
        step_MM = rebuild_spacing
        # 投影计算
        min_id = np.dot(b_min, flat_normal) / step_MM
        new_spacing = [rebuild_spacing, rebuild_spacing, rebuild_spacing]
        k_idx = np.floor((np.dot(obliqueLocal[3], flat_normal) / step_MM) - min_id + 0.5)
        total = np.floor(np.dot(b_max, flat_normal) / step_MM - min_id)
        # 计算旋转后 origin

        if azimuth_type == 1:
            direction = np.array([[1,0,0], [0,1,0], [0,0,-1]])
            M = np.dot(M, direction)
        elif azimuth_type == 2:
            direction = np.array([[1,0,0],[0,0,-1], [0,1,0]])
            M = np.dot(M, direction)
        elif azimuth_type == 3:
            direction = [ [0,1,0],[0,0,-1], [1,0,0]]
            M = np.dot(direction, M)
            # 旋转
            # direction = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
            # M = np.dot(M, direction)
        tmp = np.dot([512,512,362], M).tolist()
        print(f'tmp: {tmp}')
        patient_origin = np.dot(patient_origin, M)
        new_st_origin = np.dot(st_origin, M)
        new_ed_origin = np.dot(ed_origin, M)
        origin_lps = [st_origin[0], st_origin[1], ed_origin[2]]
        origin_lps = np.dot(origin_lps, M).tolist()
        print(f'st_origin: {st_origin}, ed_origin: {ed_origin}')
        print(f"new_st_origin: {new_st_origin}, new_ed_origin: {new_ed_origin}, patient_origin: {patient_origin}")
        length = np.dot(b_max, flat_normal) - np.dot(b_min, flat_normal) 
        print(f'length: {length}, min_id: {min_id}, total: {total},k_idx:{k_idx}, patient_origin: {patient_origin}')
        return total, k_idx, patient_origin, new_st_origin, new_ed_origin, obliqueLocal[3], origin_lps


    def get_mpr(
        self,
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
        st_origin = img.GetOrigin()
        print(f'******st_origin： {st_origin},spacing: {spacing}')
        reader = vtk_image_from_array(img_arr, spacing, st_origin)

        # 获取物理信息
        extent = reader.GetOutput().GetExtent()  # 维度
        print(f"extent: {extent}")  # (x, y, z)

        rotate_m = np.array([[matrix[0], matrix[1], matrix[2]], [matrix[4], matrix[5], matrix[6]], [matrix[8], matrix[9], matrix[10]]])
        ijk = [matrix[3], matrix[7], matrix[11]]
        new_spacing = [rebuild_spacing, rebuild_spacing, rebuild_spacing]
        ed_origin = [            
            st_origin[0] + spacing[0] * img_shape[2],
            st_origin[1] + spacing[1] * img_shape[1],
            st_origin[2] + spacing[2] * img_shape[0],]
        # 计算旋转前的 k_idx
        old_ijk = np.dot(ijk, np.linalg.inv(rotate_m))
        k_idx = img.TransformPhysicalPointToIndex(old_ijk)[2]
        print(f'**************k_idx: {k_idx}')
        # 根据层厚 SliceThickness 层间距 计算 mip 各方位总层数, SpacingBetweenSlices
        total,current_idx, patient_origin, new_st_origin, new_ed_origin, center, center_origin = self.get_total_by_matrix(rebuild_spacing, slab_thickness, azimuth_type, matrix, img_shape, spacing, st_origin)
        if azimuth_type == 1:
            physical_idx = center[2]
            # 根据 slab_thickness 和 physical_idx 获取 mip 区间
            tmp1 = [patient_origin[0], patient_origin[1], physical_idx - 0.5 * slab_thickness]
            tmp2 = [patient_origin[0], patient_origin[1], physical_idx + 0.5 * slab_thickness]
            #  左面不够 0.5 * 重建厚度
            left_bounding = [patient_origin[0], patient_origin[1], new_st_origin[2] + slab_thickness]
            right_bounding = [patient_origin[0], patient_origin[1], new_ed_origin[2] - slab_thickness]
            min_idx, max_idx = self.get_rebuild_range(img, tmp1, tmp2, patient_origin, new_st_origin, new_ed_origin, new_spacing, rotate_m, left_bounding, right_bounding, physical_idx, slab_thickness, azimuth_idx=2)
        elif azimuth_type == 2:
            physical_idx = center[1]
            print(f'*****physical_idx: {physical_idx}')
            left_bounding = [patient_origin[0], new_st_origin[1] + slab_thickness, patient_origin[2]]
            right_bounding = [patient_origin[0], new_ed_origin[1] - slab_thickness, patient_origin[2]]
            tmp1 = [patient_origin[0], physical_idx - 0.5 * slab_thickness, patient_origin[2]]
            tmp2 = [patient_origin[0], physical_idx + 0.5 * slab_thickness, patient_origin[2]]
            min_idx, max_idx = self.get_rebuild_range(img, tmp1, tmp2, patient_origin, new_st_origin, new_ed_origin, new_spacing, rotate_m, left_bounding, right_bounding, physical_idx, slab_thickness, azimuth_idx=1)
        elif azimuth_type == 3:
            # 失 X
            physical_idx = center[0]
            left_bounding = [new_st_origin[0] + slab_thickness, patient_origin[2], patient_origin[1]]
            right_bounding = [new_ed_origin[0] - slab_thickness, patient_origin[2], patient_origin[1]]
            tmp1 = [physical_idx - 0.5 * slab_thickness, patient_origin[2], patient_origin[1]]
            tmp2 = [physical_idx + 0.5 * slab_thickness, patient_origin[2], patient_origin[1]]
            print(f'********tmp1: {tmp1}, tmp2: {tmp2}, right_bounding: {right_bounding}')
            min_idx, max_idx = self.get_rebuild_range(img, tmp1,tmp2, patient_origin,  new_st_origin, new_ed_origin, spacing, rotate_m, left_bounding, right_bounding, physical_idx, slab_thickness, azimuth_idx=0)
        else:
            raise "azimuth_type input error"
        slice_nums = max_idx - min_idx
        print(f"slice_nums: {slice_nums}， min_idx: {min_idx}, max_idx: {max_idx}")
        
        resliceAxes = vtk.vtkMatrix4x4()
        resliceAxes.DeepCopy(matrix)
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
        reslice.SetResliceAxesOrigin(ijk)
        reslice.SetBackgroundLevel(-99999)
        reslice.Update()

        # 落盘
        vtk_img_export = vtkImageExportToArray()
        vtk_img_export.SetInputConnection(reslice.GetOutputPort())
        np_array = vtk_img_export.GetArray()

        np_array = np_array.astype(np.int16)
        save_path = "./mip.dcm"
        print(f'***rows, cols: {np_array.shape}')
        result = sitk.GetImageFromArray(np_array)
        result.SetMetaData("0028|1050", "300")
        result.SetMetaData("0028|1051", "800")
        sitk.WriteImage(result, save_path)
        # np_array_to_dcm(ds, save_path, np_array, azimuth_type, matrix, spacing, rebuild_spacing, center_origin, current_idx)
        res["dcm_path"] = f"/tmp/{rebuild_type}.dcm"

        # res["boxes"] = boxes
        # res["contour"] = contour_res
        # res["idx"] = idx
        print(f"a mip cost: {time.time() - t0}")
        return res
    
    def getCoord(self, point, center_position):
        p = np.subtract(point, center_position)
        return [np.dot(p, self.x), np.dot(p, self.y), np.dot(p, self.z)]

    def getCoordRot(self, point):
        return [np.dot(point, self.x), np.dot(point, self.y), np.dot(point, self.z)]
    def _getObliqueLocal(self, oblique, center_position):
        res = [0, 0, 0, 0]
        res[0] = self.getCoordRot([oblique[0], oblique[4], oblique[8]])
        res[1] = self.getCoordRot([oblique[1], oblique[5], oblique[9]])
        res[2] = self.getCoordRot([oblique[2], oblique[6], oblique[10]])
        res[3] = self.getCoord([oblique[3], oblique[7], oblique[11]], center_position)
        return res
    
    def _getObliqueThicknessMM(self, oblique, spacing, center_position):
        normalLocal = self._getObliqueLocal(oblique, center_position)[2]
        res = math.sqrt(
            (normalLocal[0] * spacing[0]) ** 2 +
            (normalLocal[1] * spacing[1]) ** 2 +
            (normalLocal[2] * spacing[2]) ** 2
        )
        return res

    def get_rebuild_range(self,img, tmp1, tmp2, patient_origin, new_st_origin, new_ed_origin, new_spacing, rotate_m, left_bounding,right_bounding, physical_idx, slab_thickness, azimuth_idx=2):
        """"""
        if (physical_idx - 0.5 * slab_thickness) < new_st_origin[azimuth_idx]:
            print(11111111111111)
            # min_idx = ijk_to_xyz(
            #     new_st_origin, patient_origin, new_spacing, rotate_m
            # )[2]
            # max_idx = ijk_to_xyz(
            #     left_bounding,
            #     patient_origin,
            #     new_spacing,
            #     rotate_m
            # )[2]
            max_idx = img.TransformPhysicalPointToIndex(left_bounding)[azimuth_idx]
            min_idx = img.TransformPhysicalPointToIndex(new_st_origin)[azimuth_idx]
        # 右面不够 0.5 * 重建厚度
        elif (physical_idx + 0.5 * slab_thickness) > new_ed_origin[azimuth_idx]:
            print(2222222222222)
            # max_idx = ijk_to_xyz(
            #     new_ed_origin, patient_origin, new_spacing, rotate_m
            # )[0]
            # min_idx = ijk_to_xyz(
            #     right_bounding,
            #     patient_origin,
            #     new_spacing,
            #     rotate_m
            # )[0]
            max_idx = img.TransformPhysicalPointToIndex(new_ed_origin)[azimuth_idx]
            min_idx = img.TransformPhysicalPointToIndex(right_bounding)[azimuth_idx]
        else:
            max_idx = img.TransformPhysicalPointToIndex(tmp2)[azimuth_idx]
            min_idx = img.TransformPhysicalPointToIndex(tmp1)[azimuth_idx]
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


def np_array_to_dcm(ds: pydicom.FileDataset, save_path:str, np_array:np.ndarray, azimuth_type: str, matrix: list, pixel_spacing: list, rebuild_spacing, center_origin, current_idx):
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
    if azimuth_type == 1: # 轴
        ds.PixelSpacing = pixel_spacing[:-1]
        ds.SliceLocation = center_origin[2] - (current_idx * rebuild_spacing) 
        ds.ImagePositionPatient = [center_origin[0], center_origin[1], ds.SliceLocation]
        ds.ImageOrientationPatient = [matrix[0], matrix[4], matrix[8], matrix[1], matrix[5], matrix[9]]
    if azimuth_type == 2: # 冠
        ds.PixelSpacing = [pixel_spacing[1], pixel_spacing[2]]
        ds.SliceLocation = center_origin[1] + (current_idx * rebuild_spacing) 
        ds.ImagePositionPatient = [center_origin[0], ds.SliceLocation, center_origin[2]]
        ds.ImageOrientationPatient = [matrix[0], matrix[4], matrix[8], matrix[2], matrix[6], matrix[10]]
    if azimuth_type == 3: # 矢
        # x 为失状轴
        ds.PixelSpacing = [pixel_spacing[0], pixel_spacing[2]]
        ds.SliceLocation = center_origin[0] + (current_idx * rebuild_spacing)
        ds.ImagePositionPatient = [ds.SliceLocation, center_origin[1], center_origin[2]]
        ds.ImageOrientationPatient = [matrix[1], matrix[5], matrix[9], matrix[2], matrix[6], matrix[10]]
        
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
    slab_thickness = 0.75
    scale = 1
    # TODO: 层厚小于 1mm 情况前端控制？
    # 确保matrix 坐标系正确
    # 轴正交
    # matrix = [
    # 1, 0, 0, 0, 0, 1, 0, -303.7509765625, 0, 0, -1, -622.2999877929688, 0, 0, 0, 1
    # ]
    # 冠正交
    # matrix = [1, 0, 0, 12.919495109927036, 0, 0, 1, -201.09511408418098, 0, -1, 0, -644.2071982831546, 0, 0, 0, 1]   
    # 矢正交
    # matrix = [
    #     0, 0, 1, 12.919495109927036, 
    #     1, 0, 0, -201.09511408418098,
    #     0, -1, 0, -644.2071982831546,
    #     0, 0, 0, 1]
    # 轴任意
    matrix = [
        -0.855904, -0.111132, -0.505053, -4.97851,
        0.111132, 0.914292, -0.389513, -180.34, 
        -0.505053, 0.389513, 0.770196, -640.749,
        0, 0, 0, 1 
        ]
    # 冠任意
    matrix = [
    -0.74568, 0, 0.666304, 0.0633316, 
    0.666304, 0, 0.74568, -175.587, 
    0, 1, 0, -623.05 ,
    0, 0, 0, 1 
    ]

    # 矢任意
    matrix = [
    0.666304, 0.739982, -0.0920129, -1.91371,
    0.74568, -0.661212, 0.0822183, -174.822, 
    3.50161e-16, -0.123395, -0.992358, -621.078, 
    0, 0, 0, 1 
    ]
    mpr = MPR()
    res = mpr.get_mpr(
        slab_thickness,
        slab_thickness / scale,
        matrix,
        rebuild_type="MIP",
        azimuth_type=3,
    )
    # print(json.dumps(res))


if __name__ == "__main__":
    main()
