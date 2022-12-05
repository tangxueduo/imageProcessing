import numpy as np
import pydicom
from pydicom import FileDataset

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
        center_origin: 原体数据的第一张左上点 [-xx, -xxx, -xxx]
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