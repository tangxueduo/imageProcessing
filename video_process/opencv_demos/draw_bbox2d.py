import json

import cv2 as cv
import numpy as np
import pydicom
import SimpleITK as sitk

"""需要考虑根据label 的 position， 确认 contour 属于 同label的 label1 还是 label2"""

dcm_path = "/media/tx-deepocean/Data/DICOMS/ct_cerebral/CN010002-13696724/1.2.840.113619.2.416.10634142502611409964348085056782520111/1.2.840.113619.2.416.77348009424380358976506205963520437809/1.2.840.113619.2.416.2106685279137426449336212617073446717.1"
ds = pydicom.read_file(dcm_path, force=True)


def find_contours(mask: np.ndarray, label: int, position: int, idx: int) -> list:
    """获取正交 mpr 的contour
    Args:
        mask: 3d array
        label: 需要找出轮廓的标签
        position: 1，2, 3(轴冠矢)
    """
    res, contours = {}, []
    print(mask.shape)
    if position == 1:
        slice_arr = mask[idx, :, :]
    elif position == 2:
        # TODO: 在这里做一下同样的翻转
        # slice_arr = mask[:, idx, :]
        slice_arr = np.flipud(mask[:, idx - 1, :])
    elif position == 3:
        # slice_arr = mask[:, :, idx]
        slice_arr = np.flipud(mask[:, :, idx - 1])
    else:
        raise "position not found"
    contour, bboxes = find_2d_contours(slice_arr, label)
    res["contour"] = contour
    res["bboxes"] = bboxes
    res["id"] = idx
    contours.append(res)
    return contours


def gray2rgb_array(gray_array):
    temp_array = gray_array
    window_width = 1700
    window_level = -600
    true_max_pt = window_level + (window_width / 2)
    true_min_pt = window_level - (window_width / 2)
    scale = 255 / (true_max_pt - true_min_pt)
    temp_array = np.clip(temp_array, true_min_pt, true_max_pt)
    min_pt_array = np.ones((temp_array.shape[0], temp_array.shape[1])) * true_min_pt
    temp_array = (temp_array - min_pt_array) * scale
    rgb_array = np.zeros((temp_array.shape[0], temp_array.shape[1], 3))
    rgb_array[:, :, 0] = temp_array
    rgb_array[:, :, 1] = temp_array
    rgb_array[:, :, 2] = temp_array
    return rgb_array


def np_array_to_dcm(ds, np_array):
    ww = 255
    wl = 127
    ds.WindowWidth = ww
    ds.WindowCenter = wl
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "RGB"
    ds.SamplesPerPixel = 3
    ds.PlanarConfiguration = 0
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    ds.PixelData = np_array.tobytes()
    ds.save_as("./res.dcm")


def find_2d_contours(slice_arr, label):
    """后面可根据具体需要可过滤部分 contour"""
    rgb_arr = gray2rgb_array(slice_arr)
    cv.imwrite("./result.png", rgb_arr)
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
    contours, hier = cv.findContours(
        binary_arr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    print(f"contour 个数: {len(contours)}")
    for contour in contours:
        rect = cv.minAreaRect(contour)
        # 获取四个顶点
        box = cv.boxPoints(rect)
        box = np.int0(box)
        x, y, w, h = cv.boundingRect(contour)
        # img = cv.rectangle(slice_arr, (x, y), (x+w, y+h), (255), 1)
        # 画 bbox
        box = ((145, 143), (165, 163))
        img = cv.drawContours(rgb_arr, [box], 0, (0, 255, 0), 1)
        # 画 contour
        # binary_arr 改成你的 rgb array, 255 改成你的 rgb 值
        img = cv.drawContours(rgb_arr, [contour], 0, (255), 1)
        cv.imwrite("./test.png", img)
        cnts.append(contour)
        bboxes.append(box)
    # # 画 bbox
    # cv.rectangle(rgb_arr, (145, 143), (165, 163), (255, 0, 0), 1)
    np_array_to_dcm(ds, rgb_arr[:, :, ::-1].astype(np.uint8))
    cv.imshow("bbox", rgb_arr)
    # cv.imwrite("./res.png", rgb_arr)
    while 1:
        if cv.waitKey(100) == 27:  # 100ms or esc(ascii 等于27, 跳出循环)
            break
    cv.destroyAllWindows()
    return cnts, bboxes


if __name__ == "__main__":
    nii_path = "/media/tx-deepocean/Data/DICOMS/demos/2.16.840.1.114492.164187109100211094.22166093759.14420.391.nii.gz"
    img = sitk.ReadImage(nii_path)
    img_arr = sitk.GetArrayFromImage(img)
    # aneurysm_mask_arr = np.zeros_like(img_arr)
    print(img_arr.shape)
    res = find_contours(img_arr, 100, 1, 336)
    print(json.dumps(res))
