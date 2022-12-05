import json

import cv2 as cv
import numpy as np
import SimpleITK as sitk

"""需要考虑根据label 的 position， 确认 contour 属于 同label的 label1 还是 label2"""


def find_contours(mask: np.ndarray, label: int, position: int, idx: int) -> list:
    """获取某层的 mpr 的contour
    Args:
        mask: 3d array
        label: 需要找出轮廓的标签
        position: 1，2, 3(轴冠矢)
    """
    res, contours = {}, []
    print(mask.shape)
    if position == 1:
        slice_arr = mask[idx, :, :]
        contour, bboxes = find_2d_contours(slice_arr, label)
        res["contour"] = contour
        res["bboxes"] = bboxes
        res["id"] = idx
    elif position == 2:
        slice_arr = mask[:, idx, :]
        contour, bboxes = find_2d_contours(slice_arr, label)
        res["contour"] = contour
        res["bboxes"] = bboxes
        res["id"] = idx
    elif position == 3:
        slice_arr = mask[:, :, idx]
        contour, bboxes = find_2d_contours(slice_arr, label)
        res["contour"] = contour
        res["bboxes"] = bboxes
        res["id"] = idx
    else:
        raise "position not found"
    contours.append(res)
    return contours


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
        x, y, w, h = cv.boundingRect(contour)
        # img = cv.rectangle(slice_arr, (x, y), (x+w, y+h), (255), 1)
        # 画 bbox
        # img = cv.drawContours(binary_arr, [box], 0, (255), 1)
        # 画 contour
        img = cv.drawContours(binary_arr, [contour], 0, (255), 1)
        cnts.append(contour)
        bboxes.append(box)

    cv.imshow("bbox", img)
    while 1:
        if cv.waitKey(100) == 27:  # 100ms or esc(ascii 等于27, 跳出循环)
            break
    cv.destroyAllWindows()
    return cnts, bboxes


if __name__ == "__main__":
    nii_path = "/media/tx-deepocean/Data/DICOMS/demos/cerebral_1/cerebral-seg.nii.gz"
    img = sitk.ReadImage(nii_path)
    img_arr = sitk.GetArrayFromImage(img)
    # aneurysm_mask_arr = np.zeros_like(img_arr)
    print(img_arr.shape)
    res = find_contours(img_arr, 100, 1, 479)
    print(json.dumps(res))
