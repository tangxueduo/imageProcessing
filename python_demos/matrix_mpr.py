# https://blog.csdn.net/qq_39942341/article/details/122442949
# https://blog.csdn.net/liushao1031177/article/details/124847587
import numpy as np
import SimpleITK as sitk
import pydicom
# import vtkmodules.all as vtk
import json
# from vtkmodules.util.vtkImageExportToArray import vtkImageExportToArray

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


def get_mpr_total(series_iuid: str, position: int) -> int:
    """根据方位来获取总张数，mpr应该是extent最大值，故不需要该接口
    Args:
        position: 1,2,3(轴冠失)
    """
    # 根据series_iuid 读取nii
    nii_path = "/media/tx-deepocean/Data/DICOMS/demos/1.3.12.2.1107.5.1.4.73124.30000018012823445126400002112.nii.gz"
    img = sitk.ReadImage(nii_path)
    img_arr = img.GetArrayFromImage(img)
    width, hight, depth = img_arr.shape()
    total = 0
    if position == 1:
        total = depth
    elif position == 2:
        total = hight
    elif position == 3:
        total = width
    else:
        raise "position input error"
    return total

def get_arr_by_mask(nii_path: str):
    """读取mask"""
    try:
        img = sitk.ReadImage(nii_path)
        img_arr = sitk.GetArrayFromImage(img)
    except Exception as e:
        print(e)
    return img_arr

def find_2d_contours(slice_arr, label):
    """后面可根据具体需要可过滤部分 contour"""
    binary_arr = np.zeros(slice_arr.shape)
    binary_arr[slice_arr == label] = 1
    # 获取对应方位2d层
    slice_arr = slice_arr.astype(np.uint8)
    # 确认 函数返回几个值
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, bboxes = [], []
    contours, hier = cv.findContours(
        binary_arr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    print(f"contour 个数: {len(contours)}")
    for contour in contours:
        rect = cv.minAreaRect(contour)
        # 获取四个顶点
        box = cv.boxPoints(rect)
        box = np.int0(box)
        print(f"*****bbox: {box}")
        # x, y, w, h = cv.boundingRect(cnts[0])
        # cv.rectangle(slice_arr, (x, y), (x+w, y+h), (0, 0, 0), 2)
        # 画 bbox
        img = cv.drawContours(slice_arr, [box], 0, (255, 0, 0), 1)
        # 画 contour
        img = cv.drawContours(slice_arr, contour, 0, (255, 255, 0), 1)
        cnts.append(contour)
        bboxes.append(box)
    return cnts, bboxes

def find_contours(matirx_res: dict, slice_idx: int, position: int) -> list:
    """后面可根据具体需要可过滤部分 contour"""
    # 遍历mask label 获取contour,bbox
    label = 1
    res,contours,single_mask = {}, [],{}
    for mask_info in matirx_res["children"]:
        mask_label = mask_info["maskLabel"]
        mask_name = mask_info["maskName"]
        mask_path = matirx_res["mask"][mask_name]["path"]
        # mask_arr = get_arr_by_mask(mask_path)
        single_mask[f"{mask_name}_{mask_label}"] = {}
        # if position == 1:
        #     mask_slice_arr = mask_arr[slice_idx-1,:,:]
        #     contours, bboxes = find_2d_contours(mask_slice_arr, mask_label)
        # elif position == 2:
        #     mask_slice_arr = mask_arr[:,slice_idx-1,:]
        #     contours, bboxes = find_2d_contours(mask_slice_arr, mask_label)
        # elif position == 3:
        #     contours, bboxes = find_2d_contours(mask_slice_arr, mask_label)
        #     mask_slice_arr = mask_arr[:,:,slice_idx-1]
        # else:
        #     raise "positon input error"
        cnts = [[[1,2], [3, 4]], [[4,6], [10, 8]]]
        bboxes = [[3, 4, 5, 6], [2, 2, 4, 4]]
        single_mask[f"{mask_name}_{mask_label}"]["contour"] = cnts
        single_mask[f"{mask_name}_{mask_label}"]["bboxes"] = bboxes
        contours.append(single_mask)
    # res["contours"] = contours
    # 直接返回 list
    return  contours


def get_mpr(series_iuid: str, positon: int, slice_idx: int) -> dict:
    """ 根据方位和层面返回正交 mpr 和 层面的contour, bbox """
    res = {}
    # 读取 nii
    nii_path = "/media/tx-deepocean/Data/DICOMS/demos/1.3.12.2.1107.5.1.4.73124.30000018012823445126400002112.nii.gz"
    dicom_file = "/media/tx-deepocean/Data/DICOMS/demos/1.2.840.113704.1.111.15016.1562149464.30/1.2.840.113704.1.111.4056.1562150522.187334"
    ds = pydicom.read_file(dicom_file, force=True)
    img = sitk.ReadImage(nii_path)
    spacing = img.GetSpacing()
    print(f'****spacing:{spacing}')
    img_arr = sitk.GetArrayFromImage(img)
    print(img_arr.shape)
    width, hight, depth = img_arr.shape[2], img_arr.shape[1], img_arr.shape[0]

    # 获取预测结果 (有feedback)
    matrix_res = {}
    matrix_res["children"] = []
    tmp = {}
    tmp["name"] = ""
    tmp["maskLabel"] = 1
    tmp["maskName"] = "liverMask"
    matrix_res["children"].append(tmp)
    matrix_res["mask"] = {}
    matrix_res["mask"]["liverMask"] = {"path": "result/11/liver.nii.gz"}
    print(matrix_res)
    # 出图, contour, bbox
    if positon == 1:
        slice_arr = img_arr[slice_idx-1,:,:]
        pixel_spacing = [spacing[0], spacing[1]] # 轴

    elif positon == 2:
        slice_arr = img_arr[:,slice_idx-1,:]
        print(slice_arr.shape)
        # slice_arr = np.flipud(img_arr[:,slice_idx-1,:]) #上下反转
        pixel_spacing = [spacing[2], spacing[1]]  # 冠
    elif position == 3:
        slice_arr = np.flipud(img_arr[:,:,slice_idx-1]) # 上下反转
        pixel_spacing = [spacing[2], spacing[0]] # 矢
    else:
        raise "positon not found"
    # TODO  pixel_array 落盘mpr
    print(slice_arr.shape)
    slice_arr = slice_arr.astype(np.int16)
    save_path = f"./mpr.dcm"

    np_array_to_dcm(ds, save_path, slice_arr, pixel_spacing, spacing)
    contour_res = find_contours(matrix_res, slice_idx, position)

    res["dcm_path"] = save_path
    res["contours"] = contour_res
    # save_path 返回
    return res

def np_array_to_dcm(ds, save_path, np_array, pixel_spacing,spacing):
    ds.WindowCenter = -600
    ds.WindowWidth = 1600
    ds.Rows = np_array.shape[0]
    ds.Columns = np_array.shape[1]
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PixelSpacing = pixel_spacing
    ds.SpacingBetweenSlices = spacing[0]
    ds.SliceThickness = spacing[0]
    ds.ImageOrientationPatient = [0,0,1,0,1,0]
    # TODO: 重新计算 ImagePositionPatient, orientation, slicelocation, PixelSpacing,
    ds.PixelData = np_array.tobytes()
    # ds.is_implicit_VR = True
    ds.save_as(save_path)

if __name__ == '__main__':
    series_iuid = "1.3.46.670589.33.1.63792961161330624600001.5231325296116986594"
    position = 2
    slice_idx = 271
    res = get_mpr(series_iuid=series_iuid, positon=position, slice_idx=slice_idx)
    res = get_mpr_total()
    print(json.dumps(res))

