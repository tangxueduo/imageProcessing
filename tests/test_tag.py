"""
TAG计算: 管腔内对比度衰减梯度，血管腔内密度衰减和动脉开口至末端长度之间的线性回归系数，单位 为HU/10mm, 反映了对比剂通过管腔的下降率
"""
import json
import os
import time
from typing import Dict

import cprlib
import cv2

# import matplotlib.pyplot as plt
import numpy as np
import pytest
import requests
import SimpleITK as sitk
from cprlib.centerline import Centerline, DiffMethod
from loguru import logger
from sklearn import linear_model

from tests.constants import (
    CPR_WIDTH,
    LUMEN_WIDTH,
    LUMEN_WIDTH_HEAD,
    TAG_VESSELS,
    TAGMETHODS,
    WIDER_VESSELS,
    CPRLibProps,
    TAGMethod,
)
from utils.contour import get_diameter

treeline_url = "http://172.16.8.59:3333/series/1.3.46.670589.33.1.63757623791122361500001.5451388697822215795/predict/ct_heart_treeline"
treeline_res = requests.get(treeline_url).json().get("treeLines").get("data")
smooth_cutoff = 0.5
lSpacing = 0.5


def read_lines_dict(in_lines) -> Dict[str, dict]:
    """
    将中线数据整理成固定格式
    """
    out: dict = {"lines": {}, "types": {}, "tangents": {}}
    if not in_lines:
        return out

    use_external_tangent: bool = "tangents" in in_lines
    logger.info(f"use external tangents is {use_external_tangent}")

    for vessel, vessel_info in in_lines["typeShow"].items():
        lines, types, tangents = [], [], []
        for idx in vessel_info["data"]:
            idx_i = str(idx)
            lines.extend(in_lines["lines"][idx_i])
            types.append({in_lines["types"][idx_i]: len(in_lines["lines"][idx_i])})
            if use_external_tangent:
                tangents.extend(in_lines["tangents"][idx_i])
        out["lines"][vessel] = lines
        out["types"][vessel] = types
        out["tangents"][vessel] = tangents
    return out


@pytest.fixture()
def prepare_data():
    root_dir = "/media/tx-deepocean/Data/DICOMS/demos/ct_heart"
    nii_path = os.path.join(
        root_dir, "1.3.46.670589.33.1.63757623791122361500001.5451388697822215795.mhd"
    )
    seg_path = os.path.join(root_dir, "output_lps_seg.nii.gz")
    fine_plaque_path = os.path.join(root_dir, "vessel_analysis.nii.gz")
    sitk_img = sitk.ReadImage(nii_path)
    seg_img = sitk.ReadImage(seg_path)
    fine_plaque = sitk.ReadImage(fine_plaque_path)
    line_dict = read_lines_dict(treeline_res)
    return dict(img=sitk_img, seg=seg_img, plg=fine_plaque, lines=line_dict)


def gen_img_generator(volume, vessel, centerline, tangents=None):
    # lumen_width = LUMEN_WIDTH if vessel in WIDER_VESSELS else LUMEN_WIDTH_HEAD
    lumen_width = 40
    if isinstance(volume, list):
        input_data = volume
    else:
        input_data = [volume]
    cnt = Centerline(
        centerline={"y": centerline},
        diff_method=DiffMethod(params={"kind": "kalman", "alpha": 1.5}),
        arclen_reparam=False,
    )
    img_gen = cprlib.CPRImageGenerator(
        input_data,
        centerline=cnt,
        straighten_resampling_spacing=0.5,
        straighten_resampling_shape=[lumen_width],
        straighten_probe_output_spacing=None,
        straighten_lumen_output_spacing=0.5,
        stretched_resampling_spacing=0.5,
        stretched_resampling_padding=CPR_WIDTH,  # 决定 CPR 宽度
        stretched_output_spacing=None,
        resample_output_image_to_isotropic=True,
    )
    return img_gen


def calculate_regression(
    res, single_vessel_tag_res, vessel_label, method=TAGMethod.LUMEN
):
    """"""
    TAG_INTERVAL = 5
    model = linear_model.LinearRegression()
    model.fit(
        np.array(single_vessel_tag_res["h_axis"]).reshape(-1, 1),
        np.array([info["hu"] for info in single_vessel_tag_res["v_axis"]]),
    )
    intercept = model.intercept_
    slope = model.coef_
    logger.warning(
        f"vessel: {vessel_label}, method: {method}, 截距:intercept: {intercept}\t 系数:slope: {slope}\n"
    )
    res[vessel_label][method] = {
        "slope": slope[0],
        "intercept": intercept,
        "interval": TAG_INTERVAL,
        "h_axis": single_vessel_tag_res["h_axis"],
        "v_axis": single_vessel_tag_res["v_axis"],
        "tag_region": single_vessel_tag_res["tag_region"],
    }
    return res


def get_tag_data(res, im, img_gen, vessel_label, centerline):
    """
    计算　TAG
    """
    lumen_lesion, noplaque_lesion, centerline_lesion = {}, {}, {}
    # 计算中线索引
    img_np = sitk.GetArrayFromImage(im)
    pix_indices = np.array(
        [im.TransformPhysicalPointToIndex(pt)[::-1] for pt in centerline]
    )
    pix_indices = np.minimum(
        np.maximum(np.round(pix_indices), 0), np.array(img_np.shape)[::-1] - 1
    ).astype("int")
    pixel_line_len = len(centerline)
    phycial_line_len = pixel_line_len * 0.5
    logger.warning(
        f"line_length: {pixel_line_len}, phycial_line_len: {phycial_line_len}"
    )
    for i in range(0, pixel_line_len):
        probe_res = img_gen.probe(
            slice_index=i,
            output_shape=(15, 15),
            output_spacing=[0.5, 0.5],
            post_processor_props=CPRLibProps,
        )
        probe_arr = sitk.GetArrayFromImage(probe_res["im"])
        contour_data = gen_diameter_area(probe_res["vessel_seg"], save_contour=False)
        if not contour_data:
            logger.warning("未找到 contour")
            continue
        # LUMEN
        lumen_lesion.update(
            {
                i
                * 0.5: {
                    "area": contour_data["area"],
                    "mean_diameter": contour_data["averageDiameter"],
                    "hu": probe_arr.mean().tolist(),
                }
            }
        )
        # 由于计算均按照ROI，故管腔和去斑块方式一起计算，减少计算量, REMOVE_PLAQUE
        tmp = np.nanmean(probe_arr[(probe_arr < 350)])
        # roi内无小于350的值
        if np.isnan(tmp):
            logger.warning(f"无满足要求的hu, 均值为{tmp}, 过滤该点")
            continue
        noplaque_lesion[i * 0.5] = {"hu": tmp.tolist()}
        # CENTERLINE
        centerline_lesion[i * 0.5] = {
            "hu": img_np[
                pix_indices[i][0], pix_indices[i][1], pix_indices[i][2]
            ].tolist()
        }
    lumen_res, noplaque_res, centerline_res = (
        {
            "h_axis": [key for key in lesion.keys()],
            "v_axis": list(lesion.values()),
            "tag_region": [0, phycial_line_len],
        }
        for lesion in [lumen_lesion, noplaque_lesion, centerline_lesion]
    )
    # 拟合回归方程
    for single_tag_res, method in zip(
        [lumen_res, noplaque_res, centerline_res],
        [TAGMethod.LUMEN, TAGMethod.REMOVE_PLAQUE, TAGMethod.CENTERLINE],
    ):
        res = calculate_regression(res, single_tag_res, vessel_label, method=method)
    return res


def gen_diameter_area(contour, save_contour=False):
    """
    计算长短径和面积
    """
    if contour.size == 0:
        return {}

    # contour 填充
    new_contour = contour.astype(np.float32)
    area, spacing = cv2.contourArea(new_contour), 0.5
    long, short = get_diameter(new_contour, (spacing, spacing))
    long_value, short_value = (
        round(float(long.length), 2),
        round(float(short.length), 2),
    )
    return {
        "longDiameter": {
            "value": long_value,
            "p1": [
                float(long.line_segment._point1._x),
                float(long.line_segment._point1._y),
            ],
            "p2": [
                float(long.line_segment._point2._x),
                float(long.line_segment._point2._y),
            ],
        },
        "shortDiameter": {
            "value": short_value,
            "p1": [
                float(short.line_segment._point1._x),
                float(short.line_segment._point1._y),
            ],
            "p2": [
                float(short.line_segment._point2._x),
                float(short.line_segment._point2._y),
            ],
        },
        "averageDiameter": round((long_value + short_value) / 2, 2),
        "area": round(float(area * spacing * spacing), 2),  # 像素面积转为物理面积
        "contour": contour.tolist() if save_contour else [],
    }


def add_alpha_channel(img):
    """为jpg图像添加alpha通道"""

    b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道

    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
    return img_new


def draw_mask(arr, canvas, color, opacity=1):
    alpha = (arr > 0) & (arr < 1000)
    colorMask = arr[:, :, None] * np.array(color)[None, None]
    canvas[alpha] = np.clip(
        canvas[alpha] * (1 - opacity) + colorMask[alpha] * opacity, 0, 255
    ).astype("uint8")
    return canvas


def apply_color_mapping(
    image, mask, re_image, height, width, hu_range: list = [0, 1000]
):
    start = 0  # 红色
    interval = 240  # 蓝色
    ctmin, ctmax = hu_range[0], hu_range[1]

    if ctmin >= ctmax:
        return "RET_STATUS_FATAL_ERROR"

    pixel_num = height * width
    for index in range(pixel_num):
        if mask[index] == 0:
            continue

        ct = image[index]
        if ct < ctmin or ct > ctmax:
            continue

        r, g, b, a = 0, 0, 0, 255
        h = start + interval * (ctmax - ct) / (ctmax - ctmin)
        s, v = 1.0, 1.0

        r, g, b = hsv_to_rgb(h, s, v)

        re_image[0 + index * 4] = b
        re_image[1 + index * 4] = g
        re_image[2 + index * 4] = r
        re_image[3 + index * 4] = a
    return re_image


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return int(v * 255), int(v * 255), int(v * 255)

    h /= 60.0
    i = int(h)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    if i == 0:
        return int(v * 255), int(t * 255), int(p * 255)
    elif i == 1:
        return int(q * 255), int(v * 255), int(p * 255)
    elif i == 2:
        return int(p * 255), int(v * 255), int(t * 255)
    elif i == 3:
        return int(p * 255), int(q * 255), int(v * 255)
    elif i == 4:
        return int(t * 255), int(p * 255), int(v * 255)
    else:
        return int(v * 255), int(p * 255), int(q * 255)


def get_tag_pseudo_colormap(
    img_gen,
    vessel,
):
    """获取tag伪彩图"""
    if vessel != "vessel9":
        return np.zeros(shape=(height, width, 4))
    angle, direction = 0, "landscape"
    cprlib_res = img_gen.lumen(angle=angle, post_processor_props=CPRLibProps)
    lumen_seg = sitk.GetArrayFromImage(cprlib_res["vessel_seg"])
    lumen_arr = sitk.GetArrayFromImage(cprlib_res["im"])
    canvas = np.repeat(lumen_arr[..., None], 3, axis=-1)
    if direction == "landscape":
        lumen_arr = np.rot90(lumen_arr, 1)
        lumen_seg = np.rot90(lumen_seg, 1)
    # 测试mask染色结果
    canvas = draw_mask(
        sitk.GetArrayFromImage(cprlib_res["vessel_seg"]), canvas, (0, 150, 150)
    )

    # 初始化一个rgba的
    (height, width) = lumen_arr.shape
    image = lumen_arr.astype("int16").flatten()
    mask = lumen_seg.astype("int16").flatten()
    re_image = np.zeros(shape=(width, height, 4), dtype=np.uint8).flatten()
    res_img = apply_color_mapping(
        image, mask, re_image, height, width, hu_range=[0, 1000]
    )
    res_img = np.reshape(res_img, (height, width, 4))
    print(res_img.shape)
    cv2.imwrite("./test.png", canvas)


def test_tag(prepare_data):
    # res = {}
    # t0 = time.time()
    im_volume = cprlib.CPRVolume(  # 一般的影像、分割都应该用这个
        vol_data=prepare_data["img"],  # SimpleITK.Image
        name="im",  # 这个名字之后会用来作为索引
        volume_type="image",  # 有image, mask, other三类，image同一个名字只能有一个，mask会默认用最临近插值
    )
    # 初始化管腔分割 cpr volume
    seg_volume = cprlib.CPRVolume(
        vol_data=prepare_data["plg"] == 4,  # 2 是斑块，4 是管腔
        name="vessel_seg",
        volume_type="mask",
    )
    # for vessel in TAG_VESSELS:
    #     res[vessel] = {}
    #     centerline = prepare_data["lines"]["lines"].get(vessel, None)
    #     # logger.warning(f"prepare_data['lines']: {prepare_data['lines']['lines'].keys()}")
    #     if centerline is None:
    #         for method in TAGMETHODS:
    #             res[vessel][method] = None
    #         continue
    #     img_gen = gen_img_generator(
    #         [im_volume, seg_volume], vessel, centerline
    #     )
    #     res[vessel] = {}
    #     t1 = time.time()
    #     res = get_tag_data(
    #         res,
    #         prepare_data["img"],
    #         img_gen,
    #         vessel,
    #         centerline,
    #     )
    #     logger.warning(f"{vessel} gen success and cost time: {time.time() - t1}")
    # import requests
    # url = "http://172.16.3.39:3333/series/1.3.12.2.1107.5.1.4.76346.30000021052006532895300034737/predict/ct_heart_tag"
    # requests.put(url, json=res, timeout=7)
    # logger.warning(f"最后结果，　{json.dumps(res)}, total cost: {time.time() - t0}")
    vessel = "vessel9"
    centerline = prepare_data["lines"]["lines"].get(vessel, None)
    img_gen = gen_img_generator([im_volume, seg_volume], vessel, centerline)
    get_tag_pseudo_colormap(img_gen, vessel)
    # 取糖尿病数据集
    # diabetes = datasets.load_diabetes()
    # # 选择一个特性
    # data_X = diabetes.data[:, np.newaxis,2]
    # # logger.warning(f"data_X: {data_X}")
    # # 训练接和测试集
    # x_train, x_test = data_X[:-20], data_X[-20:]
    # y_train, y_test = diabetes.target[:-20], diabetes.target[-20:]
    # model = linear_model.LinearRegression()
    # model.fit(x_train, y_train)
    # """
    # 决定系数　R平方: 决定系数反映了因变量y 的波动，
    # 有多少百分比能被自变量x（用机器学习的术语来说， x 就是特征）的波动所描述, 越高越好
    # """
    # r_sq = model.score(x_train, y_train)
    # intercept = model.intercept_
    # slope = model.coef_
    # logger.warning(f"r_sq: {r_sq}\t 截距:intercept: {intercept}\t 系数:slope: {slope}\n")
    # data_predicted = model.predict(x_train)
    # # logger.warning(f'预测结果:  {model.predict(100)}')
    # plt.title(f"Y = {round(slope[0], 3)}X + {round(intercept, 3)}", loc="right")
    # plt.scatter(x_train, y_train, color="black")
    # plt.plot(x_train, data_predicted, color="blue", linewidth=2)
    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    # # plt.text(0.05, 370, s=f"Y = {round(slope[0], 3)}X + {round(intercept, 3)}",fontdict={'family': 'serif', 'size': 16, 'color': 'black'})
    # plt.show()
