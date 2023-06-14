import math
import os
import time
from typing import Dict

import cprlib
import cv2
import numpy as np
import requests
import SimpleITK as sitk
from cprlib.centerline import Centerline, DiffMethod
from loguru import logger

treeline_url = "http://172.16.3.30:3333/series/1.2.840.113619.2.416.242355036453511869474923342565595289454/predict/ct_cerebral_treeline"
# treeline_url = "http://172.16.7.8:3333/series/1.2.156.112605.189250946070725.230325085230.3.8732.101235/predict/ct_aorta_treeline"
treeline_res = requests.get(treeline_url).json().get("treeLines").get("data")
smooth_cutoff = 0.5
lSpacing = 0.5


def read_lines_dict(in_lines) -> Dict[str, dict]:
    """
    将中线数据整理成固定格式
    """
    out: dict = {"lines": {}, "types": {}}
    logger.info(str(in_lines.keys()))
    use_external_tangent: bool = "tangents" in in_lines
    if use_external_tangent:
        logger.info("发现tangents字段，读入json中提供的切线")
        out["tangents"] = {}
    if not in_lines:
        return out

    for label, vessel_info in in_lines["typeShow"].items():
        line, types = [], []
        if use_external_tangent:
            tangent = []
        for idx in vessel_info["data"]:
            idx_i = str(idx)
            line.extend(in_lines["lines"][idx_i])
            types.append({in_lines["types"][idx_i]: len(in_lines["lines"][idx_i])})
            if use_external_tangent:
                tangent.extend(in_lines["tangents"][idx_i])
        out["lines"][label] = line
        out["types"][label] = types
        if use_external_tangent:
            out["tangents"][label] = tangent
    logger.info(f'vessel: {out["lines"].keys()}')
    return out


def get_vessel_centerline(treeline, vessel_name="vessel1"):
    """["N", "B", "T", "y", "lSpacing"]"""
    complete_lines_map = read_lines_dict(treeline)

    centerlines = complete_lines_map["lines"].get(vessel_name)
    logger.info(len(centerlines))
    # np.save("/media/tx-deepocean/Data/DICOMS/demos/mra/vessel16", np.array(centerlines))
    model_tangents = complete_lines_map.get("tangents", {}).get(vessel_name, None)
    model_tangents = None
    if not model_tangents:
        # return Centerline(centerline={"y": centerlines}, diff_method=DiffMethod(params={'kind': 'kalman', 'alpha': 1.5}))
        return Centerline(centerline={"y": centerlines}, diff_method=None)
    return Centerline({"y": centerlines, "T": model_tangents})


def prepare_data():
    root_path = "/media/tx-deepocean/Data/DICOMS/demos/mra"
    nii_path = os.path.join(
        root_path, "1.2.840.113619.2.416.242355036453511869474923342565595289454.nii.gz"
    )
    seg_path = os.path.join(root_path, "cerebral-seg.nii.gz")
    plaque_path = os.path.join(root_path, "neck_plaque-seg.nii.gz")
    fine_plaque_path = os.path.join(root_path, "plaque_analysis.nii.gz")

    centerline = get_vessel_centerline(treeline_res)
    im = sitk.ReadImage(nii_path)
    seg = sitk.ReadImage(seg_path)
    plaque_seg = sitk.ReadImage(plaque_path)
    fine_plaque = sitk.ReadImage(fine_plaque_path)
    # seg_arr = sitk.GetArrayFromImage(seg)
    # seg_arr = np.where(seg_arr<9, seg_arr, 0)
    # logger.info(f"unique: {np.unique(seg_arr)}, shape: {seg_arr.shape}")

    logger.info(f"****shape: {im.GetSize()}, spacing: {im.GetSpacing()}")
    # data = dict(im=im, seg=sitk.ReadImage(seg_path), strIm=strIm, strSeg=strSeg, cnt=centerline, plqSeg=plqSeg)
    data = dict(im=im, seg=seg, cnt=centerline, plqSeg=fine_plaque, strSeg=seg)
    return data


def draw_contour(arr, canvas, color):
    ctr = np.clip(np.round(arr), 0, np.array(canvas.shape[:-1])[::-1] - 1).astype("int")
    # import pdb
    # pdb.set_trace()
    for i in range(ctr.shape[0] - 1):
        canvas = cv2.line(canvas, ctr[i], ctr[i + 1], color=color, thickness=2)
    return canvas


def draw_mask(arr, canvas, color, opacity=0.6):
    alpha = arr > 0
    colorMask = arr[:, :, None] * np.array(color)[None, None]
    canvas[alpha] = np.clip(
        canvas[alpha] * (1 - opacity) + colorMask[alpha] * opacity, 0, 255
    ).astype("uint8")
    return canvas


def draw(res, fname):
    canvas = sitk.GetArrayFromImage(sitk.IntensityWindowing(res["im"]))
    print(canvas.shape)
    canvas = np.repeat(canvas[..., None], 3, axis=-1)
    # if res["vessel"].shape[0] > 0:
    #     canvas = draw_contour(res["vessel"], canvas, (0, 200, 20))
    # logger.info(f"plaque: {res['plaque']}")
    if len(res["plaque"]) > 0:
        # 如果返回最大contour　需要[]
        for i in res["plaque"]:
            # for i in [res["plaque"]]:
            canvas = draw_contour(i, canvas, (255, 0, 0))
        # for i in [res["plaque_mask"]]:
        #     canvas = draw_contour(i, canvas, (0, 255, 0))
    canvas = draw_mask(sitk.GetArrayFromImage(res["plaque_mask"]), canvas, (0, 255, 0))
    print(fname)
    cv2.imwrite(fname, canvas)


def main():
    output_dir = "./data/output"
    data = prepare_data()
    postProps = {
        "_default": {
            "convert_mask_to_contour": True,
            "return_multiple_contours": True,
            "contour_draw_centerline_radius": 1,
            "contour_smooth_params": {"method": "fft", "cutoff": smooth_cutoff},
        },
        "plaque": {
            "convert_mask_to_contour": True,
            "return_multiple_contours": True,
            "contour_draw_centerline_radius": None,
            "contour_smooth_params": {"method": "fft", "cutoff": smooth_cutoff},
        },
        "plaque_mask": {
            "convert_mask_to_contour": True,
            "return_multiple_contours": False,
            "contour_draw_centerline_radius": None,
        },
    }

    im = cprlib.CPRVolume(  # 一般的影像、分割都应该用这个
        vol_data=sitk.Cast(data["im"], sitk.sitkFloat32),  # SimpleITK.Image
        name="im",  # 这个名字之后会用来作为索引
        volume_type="image",  # 有image, mask, other三类，image同一个名字只能有一个，mask会默认用最临近插值
    )
    stretched_inverse_mapping_resolution = None
    strSeg = cprlib.CPRStraightenVolume(
        vol_data=data["strSeg"] == 1,
        name="vessel",
        centerline=data["cnt"],
        volume_type="mask",
        inverse_mapping_resolution=stretched_inverse_mapping_resolution,
    )
    plqSeg = cprlib.CPRVolume(
        vol_data=data["plqSeg"] == 2, name="plaque", volume_type="mask"
    )
    plqSeg_mask = cprlib.CPRVolume(
        vol_data=data["plqSeg"], name="plaque_mask", volume_type="mask"
    )
    vesselSeg = cprlib.CPRVolume(
        vol_data=data["seg"], name="plaque", volume_type="mask"
    )

    straighten_resampling_spacing = [0.2, 0.2]
    straighten_resampling_shape = [200]
    straighten_lumen_output_spacing = [0.2, 0.2]
    stretched_resampling_spacing = 0.2
    stretched_output_spacing = None

    imgGen = cprlib.CPRImageGenerator(
        [im, plqSeg],
        centerline=data["cnt"],
        straighten_resampling_spacing=straighten_resampling_spacing,
        straighten_resampling_shape=straighten_resampling_shape,
        straighten_probe_output_spacing=None,
        straighten_lumen_output_spacing=straighten_lumen_output_spacing,
        stretched_resampling_spacing=stretched_resampling_spacing,
        stretched_output_spacing=stretched_output_spacing,
        resample_output_image_to_isotropic=True,
    )
    Items = [350]
    for i in Items:
        res = imgGen.lumen(angle=i, post_processor_props=postProps)
        draw(res, f"{output_dir}_lumen.jpg")
        res_arr = sitk.GetArrayFromImage(sitk.IntensityWindowing(res["im"]))
        logger.info(f"lumen shape: {res_arr.shape}, res sum: {res_arr.sum()}")
        # cv2.imwrite("./test.png", res_arr)

        t0 = time.time()
        # CPR
        # res_cpr = imgGen.stretched(theta=i, phi=20, post_processor_props=postProps)
        # draw(res_cpr, f"{output_dir}_cpr.jpg")
        # res_cpr_arr = sitk.GetArrayFromImage(sitk.IntensityWindowing(res_cpr["im"]))
        # cv2.imwrite("./test_cpr.png", res_cpr_arr)
        # print(f'CPR cost time is: {time.time() - t0}')

        # Probe
        # res_probe = imgGen.probe(slice_index=21, output_shape=[30, 30], output_spacing=[0.2, 0.2], post_processor_props=postProps)
        # draw(res_probe, f"{output_dir}_probe.jpg")
        # probe_arr = sitk.GetArrayFromImage(sitk.IntensityWindowing(res_probe["im"], -100, 800))

        # logger.info(f'probe shape: {probe_arr.shape}, {probe_arr.min(), probe_arr.max()}')
        # cv2.imshow("probe", probe_arr)
        # if cv2.waitKey(0) & 0xFF == 27:
        #     break
        # cv2.destroyAllWindows()
        # cv2.imwrite("./test_probe.png", probe_arr)


if __name__ == "__main__":
    main()
