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
    WIDER_VESSELS,
    CPRLibProps,
    TAGMethod,
)
from utils.contour import get_diameter

treeline_url = "http://172.16.8.59:3333/series/1.2.392.200036.9116.2.2059767860.1617867078.10.1307500003.2/predict/ct_heart_treeline"
# treeline_url = "http://172.16.7.8:3333/series/1.2.156.112605.189250946070725.230325085230.3.8732.101235/predict/ct_aorta_treeline"
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
        root_dir, "1.2.392.200036.9116.2.2059767860.1617867078.10.1307500003.2.mhd"
    )
    seg_path = os.path.join(root_dir, "output_lps_seg.nii.gz")
    fine_plaque_path = os.path.join(root_dir, "plaque_analysis.nii.gz")
    sitk_img = sitk.ReadImage(nii_path)
    seg_img = sitk.ReadImage(seg_path)
    fine_plaque = sitk.ReadImage(fine_plaque_path)
    line_dict = read_lines_dict(treeline_res)
    return dict(img=sitk_img, seg=seg_img, plg=fine_plaque, lines=line_dict)


def gen_img_generator(volume, vessel, centerline, tangents=None):
    lumen_width = LUMEN_WIDTH if vessel in WIDER_VESSELS else LUMEN_WIDTH_HEAD
    if not tangents:
        cnt = Centerline(
            centerline={"y": centerline},
            diff_method=DiffMethod(params={"kind": "kalman", "alpha": 1.5}),
        )
    cnt = Centerline({"y": centerline, "T": tangents})
    img_gen = cprlib.CPRImageGenerator(
        [volume],
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


def get_tag_data(
    sitk_img, fine_seg_mask, lines_dict, vessel_label, method=TAGMethod.CENTERLINE
):
    """
    fine_seg: 标签1为管壁，标签2为斑块（包括钙化及非钙化），标签3为支架，标签4为管腔
    """
    lesion = {}
    im = cprlib.CPRVolume(  # 一般的影像、分割都应该用这个
        vol_data=sitk_img,  # SimpleITK.Image
        name="im",  # 这个名字之后会用来作为索引
        volume_type="image",  # 有image, mask, other三类，image同一个名字只能有一个，mask会默认用最临近插值
    )
    centerline = lines_dict["lines"].get(vessel_label, None)
    if not centerline:
        return None
    pixel_line_len = len(centerline)
    phycial_line_len = pixel_line_len * 0.5
    logger.warning(
        f"line_length: {pixel_line_len}, phycial_line_len: {phycial_line_len}"
    )
    # 初始化管腔分割 cpr volume
    seg = cprlib.CPRVolume(
        vol_data=fine_seg_mask == 4,  # 2 是斑块，4 是管腔
        name="vessel_seg",
        volume_type="mask",
    )
    # tag_interval = int(5/0.5)
    tangents = lines_dict.get("tangents", {}).get(vessel_label)
    img_seg_gen = gen_img_generator(seg, vessel_label, centerline, tangents)
    img_gen = gen_img_generator(im, vessel_label, centerline, tangents)
    for i in range(0, pixel_line_len - 3):
        # TODO: 计算截面　ROI　信息
        probe_seg = img_seg_gen.probe(
            slice_index=i,
            output_shape=(10, 10),
            output_spacing=[0.5, 0.5],
            post_processor_props=CPRLibProps,
        )
        # probe_seg_arr = sitk.GetArrayFromImage(probe_seg['vessel_seg'])
        # import pdb
        # pdb.set_trace()
        probe = img_gen.probe(
            slice_index=i,
            output_shape=(10, 10),
            output_spacing=[0.5, 0.5],
            post_processor_props=CPRLibProps,
        )
        probe_arr = sitk.GetArrayFromImage(probe["im"])
        if i == 100:
            cv2.imwrite("./test.png", probe_arr)
        contour_data = gen_diameter_area(probe_seg["vessel_seg"], save_contour=False)
        if not contour_data:
            logger.warning("未找到 contour")
            continue
        lesion.update(
            {
                i
                * 0.5: {
                    "area": contour_data["area"],
                    "mean_diameter": contour_data["averageDiameter"],
                    "hu": probe_arr.mean(),
                }
            }
        )
    res = {
        "h_axis": [key for key in lesion.keys()],
        "v_axis": list(lesion.values()),
        "tag_region": [0, phycial_line_len],
    }
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


def test_tag(prepare_data):
    res = {}
    t0 = time.time()
    for vessel in TAG_VESSELS:
        res[vessel] = {}
        t1 = time.time()
        # for method in TAGMETHODS:
        method = TAGMethod.CENTERLINE
        tag_dict = get_tag_data(
            prepare_data["img"],
            prepare_data["plg"],
            prepare_data["lines"],
            vessel,
            method=method,
        )
        if tag_dict is None:
            res[vessel][method] = None
            continue
        model = linear_model.LinearRegression()
        model.fit(
            np.array(tag_dict["h_axis"]).reshape(-1, 1),
            np.array([info["hu"] for info in tag_dict["v_axis"]]),
        )
        intercept = model.intercept_
        slope = model.coef_
        logger.warning(
            f"vessel: {vessel}, 截距:intercept: {intercept}\t 系数:slope: {slope}\n"
        )
        res[vessel][method] = {
            "slope": slope[0],
            "intercept": intercept,
            "interval": 5,
            "h_axis": tag_dict["h_axis"],
            "v_axis": tag_dict["v_axis"],
        }
        logger.warning(f"{vessel} gen success and cost time: {time.time() - t1}")
    logger.warning(f"最后结果，　{json.dumps(res)}, total cost: {time.time() - t0}")
    # TODO:　获取tag伪彩图

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
