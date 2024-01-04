import json
import os
import time

import cv2
import numpy as np
import pydicom
import requests
import SimpleITK as sitk

from infer_demos.cpr import CoronaryCprPredictor
from infer_demos.utils import get_lps_ijk

"""
已知中线 及 dicom 物理信息, 计算任意角度cpr
"""


def read_original_data(series_path):
    reader = sitk.ImageSeriesReader()
    name = list(reader.GetGDCMSeriesFileNames(series_path))
    reader.SetFileNames(name)
    return reader.Execute()


def get_physical(series_path):
    original_sitk_img = read_original_data(series_path)
    spacing = original_sitk_img.GetSpacing()
    thickness = (
        original_sitk_img.TransformIndexToPhysicalPoint(
            [0, 0, original_sitk_img.GetDepth() - 1]
        )[2]
        - original_sitk_img.TransformIndexToPhysicalPoint(
            [0, 0, original_sitk_img.GetDepth() - 2]
        )[2]
    )
    origin_lpi = original_sitk_img.TransformIndexToPhysicalPoint(
        [0, 0, original_sitk_img.GetDepth() - 1]
    )
    origin1 = list(origin_lpi).copy()
    origin1[2] -= thickness
    orientation = original_sitk_img.GetDirection()[:-3]
    original_lps_array = sitk.GetArrayFromImage(original_sitk_img)
    ds = pydicom.read_file(
        os.path.join(series_path, os.listdir(series_path)[0]), force=True
    )
    lps_ijk = get_lps_ijk(spacing[1], spacing[0], origin_lpi, origin1, orientation, 2.0)
    return original_lps_array, lps_ijk, ds


def read_rib_json(line_json):
    out = {}
    for label, line in line_json.items():
        out[label] = line["points"]
    return out


def main():
    t0 = time.time()
    series_path = "/media/tx-deepocean/Data/DICOMS/demos/1.3.12.2.1107.5.1.4.77154.30000020090723084966300249115"

    original_lps_array, lps_ijk, ds = get_physical(series_path)
    url = "http://172.16.6.6:3333/series/1.3.12.2.1107.5.1.4.77154.30000020090723084966300249115/predict/ct_chest_fracture"
    line_json = requests.get(url).json()["centerline"]
    rib_line = read_rib_json(line_json)

    cpr_render = CoronaryCprPredictor(original_lps_array[::-1], lps_ijk, rib_line)
    for label, points in rib_line.items():
        print(label)
        cpr_angle = 0
        cpr_render.predict_cpr(label, points, cpr_angle, ds)
    print(time.time() - t0)


if __name__ == "__main__":
    main()
