import os
import time
import cv2
import numpy as np
# import pydicom
import pytest
import SimpleITK as sitk
from loguru import logger


def contours_in(plaque_mask, slice_idx, contours, plaque_label):
    tmp = 255
    cv2.drawContours(plaque_mask[slice_idx,:,:], contours, -1, tmp, -1)
    a = np.where(plaque_mask[slice_idx,:,:]==tmp)[0].reshape(-1,1)
    b = np.where(plaque_mask[slice_idx,:,:]==tmp)[1].reshape(-1,1)
    coordinate = np.concatenate([a,b], axis=1).tolist()
    inside = [tuple(x) for x in coordinate]
    return inside, plaque_mask


def get_contour(mask_np, slice_idx, plaque_label):
    # 绘制椭圆
    rgb_np = np.repeat(mask_np[slice_idx, :, :][..., None], 3, axis=-1)
    cv2.ellipse(rgb_np, (200, 200), (100, 50), 0, 0, 360, (0, 0, 255), -1, 8)

    """
        参数2 center：必选参数。用于设置待绘制椭圆的中心坐标，确定椭圆的位置
        参数3 axes：必选参数。用于设置待绘制椭圆的轴长度，为椭圆轴大小的一半。由于椭圆有两个轴，因此axes为一个包含两个值的元组
        参数4 angle：必选参数。用于设置待绘制椭圆的偏转角度（以度为单位）--顺时针为正
        参数5 startAngle：必选参数。用于设置待绘制椭圆的弧的起始角度（以度为单位）-x轴方向为0度
        参数6 endAngle：必选参数。用于设置待绘制椭圆的弧的终止角度（以度为单位）。
        参数7 color：必选参数。用于设置待绘制椭圆的颜色。
        参数8 thickness：可选参数。当该参数为正数时，表示待绘制椭圆轮廓的粗细；当该参数为负值时，表示待绘制椭圆是实心的。
        参数9 lineType：可选参数。用于设置线段的类型，可选8（8邻接连接线-默认）、4（4邻接连接线）和cv2.LINE_AA 为抗锯齿
    """
    # slice_arr = mask_np[slice_idx,:,:]
    # rgb_np = np.repeat(slice_arr[..., None], 3, axis=-1)

    # binary_arr = np.zeros(slice_arr.shape).astype("uint8")
    # binary_arr[slice_arr == plaque_label] = 1
    
    gray = cv2.cvtColor(rgb_np,cv2.COLOR_BGR2GRAY)  
    ret, binary_arr = cv2.threshold(gray,0,255,cv2.THRESH_BINARY) 

    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=5)
    contours, hierarchy = cv2.findContours(
        binary_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    logger.warning(f"****contours: {len(contours)}")
    if contours:
        cv2.drawContours(rgb_np, contours[0], -1, (0, 255, 0), 1)
        # cv2.imshow("contour", rgb_np)
        # key = cv2.waitKey(0)
        # if key == 27 or key == ord("q"):
        #     cv2.destroyAllWindows()
        # logger.warning(f"contours: {np.array(contours[0][:,0])}")
        logger.warning(f"contours: {len(contours[0])}")
        cv2.imwrite("./rgb.png", rgb_np)
    # 返回值
    return contours if contours else []


@pytest.fixture()
def prepare_data():
    root_dir = "/media/tx-deepocean/Data/DICOMS/demos/ct_heart"
    sitk_img = sitk.ReadImage(
        os.path.join(
            root_dir, "1.3.12.2.1107.5.1.4.76055.30000022100703040877800159851.mhd"
        )
    )
    img_np = sitk.GetArrayFromImage(sitk_img)
    plaque_mask_np = np.zeros(img_np.shape, dtype="uint8")
    plaque_mask_np[2, 200:210, 200:210] = 10
    logger.warning(f"img_shape: {img_np.shape}")
    operation = "expand"
    edit_result = {"plaque_id": 10, "contours": {2: get_contour(plaque_mask_np, 2, 10)}}
    return dict(img_np=img_np, plaque_np=plaque_mask_np, operation=operation, edit_result=edit_result)


def test_plaque_mask_update(prepare_data):
    plaque_mask_np = prepare_data["plaque_np"]
    plaque_id = int(prepare_data["edit_result"]["plaque_id"])  
    if prepare_data["operation"] == "expand":
        contours = prepare_data["edit_result"]["contours"]          
        for slice_idx, contour_coords in contours.items():
            # contour_coords是包含轮廓坐标的NumPy数组
            contour_coords = np.array(contour_coords)
            contours_in(plaque_mask_np, slice_idx, contour_coords, plaque_id)

    if prepare_data["operation"] == "reduce":
        pass
    if prepare_data["operation"] == "delete":
        plaque_mask_np[plaque_mask_np == plaque_id] = 0
    # 过滤hu， 遍历所有层找contour, 存mask
    # TODO 有bug， 存contour后的还是过滤hu后的
    plaque_mask_np[prepare_data["img_np"]<-30] = 0
    # TODO　有更好的存储方式
    np.save("./plaque_mask", plaque_mask_np)
    # 更新某个斑块的所有层面的contour
    res = []
    for slice_idx in range(plaque_mask_np.shape[0]):
        contour = get_contour(plaque_mask_np, slice_idx=slice_idx, plaque_label=plaque_id)
        if not contour:
            continue
        slice_res = {
            "slice_id": slice_idx,
            "contour": contour
            }
        res.append(slice_res)
    logger.warning(f"res: {len(res)}")
    return res
