import math
import time

import numpy as np
import pydicom
import SimpleITK as sitk
from loguru import logger

from python_demos.tools import modify_dcm_tags

original_ds = pydicom.read_file(
    "/media/tx-deepocean/Data/DICOMS/demos/aorta/CE027001-118121600070-1809-402/CE027001-118121600070-1809-402_0272.dcm",
    force=True,
)


def gen_probe(
    original_lpi_array: np.ndarray,
    lps_ijk: np.ndarray,
    points_data: dict,
    cursor_index: int,
    spacing_x=0.5,
    spacing_y=0.5,
    width=50,
    height=50,
    ratio=2.0,
) -> np.ndarray:
    """
    Helper for gen a Probe.
    Args:
        points_data:
            {
                "points": [[x,y,z]...],
                "normals": [],
                "bi_normals": []
            }
        cursor_index: 当前探针索引
        spacing_x,spacing_y: 采样间距, 目前写死为 0.5
        p_w, p_h: 探针 size
    """
    im_depth, im_height, im_width = original_lpi_array.shape
    pixel_data = np.ones(shape=(height, width), dtype=np.int16) * (-1024)

    bi_normal = points_data.get("bi_normals")[cursor_index]
    normal = points_data.get("normals")[cursor_index]
    center = points_data.get("points")[cursor_index]

    w_off_set = np.dot(normal, spacing_x * width * 0.5)  # todo 对比前端确定spacing顺序
    h_off_set = np.dot(bi_normal, spacing_y * height * 0.5)

    origin = center.copy()
    origin = np.subtract(origin, w_off_set).tolist()
    origin = np.subtract(origin, h_off_set).tolist()

    # 矩阵数量积
    w_vector = np.dot(normal, spacing_x)
    h_vector = np.dot(bi_normal, spacing_y)
    # m = lps_ijk
    a = [origin[0], origin[1], origin[2], 1]
    m = np.reshape(lps_ijk, (4, 4))
    t = [0, 0, 0, 0]
    t0 = time.time()
    for i in range(height):
        for j in range(width):
            t[0], t[1], t[2] = np.dot(a, m)[:-1]
            if (
                0 <= t[2] < im_depth * ratio
                and 0 <= t[0] < im_width * ratio
                and 0 <= t[1] < im_height * ratio
            ):
                t[0] /= ratio
                t[1] /= ratio
                t[2] /= ratio
                t[2] = min(max(t[2], 0), im_depth - 1)

                x0, y0, z0 = math.floor(t[0]), math.floor(t[1]), math.floor(t[2])
                x1 = min(max(x0, 0), im_width - 1)
                y2 = min(max(y0, 0), im_height - 1)
                fz = min(max(z0, 0), im_depth - 1)
                x2 = min(max(x1 + 1, 0), im_width - 1)
                y1 = min(max(y2 + 1, 0), im_height - 1)
                cz = min(max(fz + 1, 0), im_depth - 1)

                if x1 == x2 or y1 == y2:
                    return -1024

                # hu1 = fn(original_lpi_array, t[0], t[1], x1, x2, y1, y2, fz)
                fQ11 = original_lpi_array[fz, y1, x1]
                fQ21 = original_lpi_array[fz, y1, x2]
                fQ12 = original_lpi_array[fz, y2, x1]
                fQ22 = original_lpi_array[fz, y2, x2]
                # TODO 重新确认这一中间变量的name
                temp = (x2 - x1) * (y2 - y1)
                hu1 = (
                    (fQ11 / temp) * ((x2 - t[0]) * (y2 - t[1]))
                    + (fQ21 / temp) * ((t[0] - x1) * (y2 - t[1]))
                    + (fQ12 / temp) * ((x2 - t[0]) * (t[1] - y1))
                    + (fQ22 / temp) * ((t[0] - x1) * (t[1] - y1))
                )
                # hu2 = fn(original_lpi_array, t[0], t[1], x1, x2, y1, y2, cz)
                fQ11 = original_lpi_array[fz, y1, x1]
                fQ21 = original_lpi_array[fz, y1, x2]
                fQ12 = original_lpi_array[fz, y2, x1]
                fQ22 = original_lpi_array[fz, y2, x2]
                # TODO 重新确认这一中间变量的name
                temp = (x2 - x1) * (y2 - y1)
                hu2 = (
                    (fQ11 / temp) * ((x2 - t[0]) * (y2 - t[1]))
                    + (fQ21 / temp) * ((t[0] - x1) * (y2 - t[1]))
                    + (fQ12 / temp) * ((x2 - t[0]) * (t[1] - y1))
                    + (fQ22 / temp) * ((t[0] - x1) * (t[1] - y1))
                )
                hu = (cz - t[2]) * hu1 + (t[2] - fz) * hu2
                pixel_data[i][j] = hu
                # pixel_data[i][j] = get_hu(original_lpi_array, t[0], t[1], t[2], ratio)
            a[:3] = np.add(a[:3], w_vector)
        a[:3] = np.add(a[:3], np.subtract(h_vector, np.dot(w_vector, width)))
    logger.debug(f"Probe cost time is : {time.time() - t0}")
    return pixel_data, origin


def get_hu(original_lpi_array, x, y, z, ratio=2.0):
    """"""
    d, h, w = original_lpi_array.shape
    x /= ratio
    y /= ratio
    z /= ratio
    z = min(max(z, 0), d - 1)

    x0, y0, z0 = math.floor(x), math.floor(y), math.floor(z)
    x1 = min(max(x0, 0), w - 1)
    y2 = min(max(y0, 0), h - 1)
    fz = min(max(z0, 0), d - 1)
    x2 = min(max(x1 + 1, 0), w - 1)
    y1 = min(max(y2 + 1, 0), h - 1)
    cz = min(max(fz + 1, 0), d - 1)

    if x1 == x2 or y1 == y2:
        return -1024

    hu1 = fn(original_lpi_array, x, y, x1, x2, y1, y2, fz)
    hu2 = fn(original_lpi_array, x, y, x1, x2, y1, y2, cz)
    hu = (cz - z) * hu1 + (z - fz) * hu2
    return hu


def fn(original_lpi_array, x, y, x1, x2, y1, y2, z):
    fQ11 = original_lpi_array[z, y1, x1]
    fQ21 = original_lpi_array[z, y1, x2]
    fQ12 = original_lpi_array[z, y2, x1]
    fQ22 = original_lpi_array[z, y2, x2]
    # TODO 重新确认这一中间变量的name
    temp = (x2 - x1) * (y2 - y1)
    hu = (
        (fQ11 / temp) * ((x2 - x) * (y2 - y))
        + (fQ21 / temp) * ((x - x1) * (y2 - y))
        + (fQ12 / temp) * ((x2 - x) * (y - y1))
        + (fQ22 / temp) * ((x - x1) * (y - y1))
    )
    return hu


# TODO: 迁移至公用处
def get_lps_ijk(spacing_y, spacing_x, origin0, origin1, orientation, ratio=2.0):
    # x 向量投影
    di = np.dot(orientation[:-3], spacing_y / ratio)
    # y　向量投影
    dj = np.dot(orientation[3:], spacing_x / ratio)
    dk = np.subtract(origin1, origin0)
    scale_k = [dk[0], dk[1], dk[2] / ratio]
    ijk_lps = np.array([*di, 0, *dj, 0, *scale_k, 0, *origin0, 1], dtype=float)
    # invert
    [
        a00,
        a01,
        a02,
        a03,
        a10,
        a11,
        a12,
        a13,
        a20,
        a21,
        a22,
        a23,
        a30,
        a31,
        a32,
        a33,
    ] = ijk_lps
    b00 = a00 * a11 - a01 * a10
    b01 = a00 * a12 - a02 * a10
    b02 = a00 * a13 - a03 * a10
    b03 = a01 * a12 - a02 * a11
    b04 = a01 * a13 - a03 * a11
    b05 = a02 * a13 - a03 * a12
    b06 = a20 * a31 - a21 * a30
    b07 = a20 * a32 - a22 * a30
    b08 = a20 * a33 - a23 * a30
    b09 = a21 * a32 - a22 * a31
    b10 = a21 * a33 - a23 * a31
    b11 = a22 * a33 - a23 * a32

    det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06
    if det == 0:
        return None
    det = 1.0 / det

    out = [
        (a11 * b11 - a12 * b10 + a13 * b09) * det,
        (a02 * b10 - a01 * b11 - a03 * b09) * det,
        (a31 * b05 - a32 * b04 + a33 * b03) * det,
        (a22 * b04 - a21 * b05 - a23 * b03) * det,
        (a12 * b08 - a10 * b11 - a13 * b07) * det,
        (a00 * b11 - a02 * b08 + a03 * b07) * det,
        (a32 * b02 - a30 * b05 - a33 * b01) * det,
        (a20 * b05 - a22 * b02 + a23 * b01) * det,
        (a10 * b10 - a11 * b08 + a13 * b06) * det,
        (a01 * b08 - a00 * b10 - a03 * b06) * det,
        (a30 * b04 - a31 * b02 + a33 * b00) * det,
        (a21 * b02 - a20 * b04 - a23 * b00) * det,
        (a11 * b07 - a10 * b09 - a12 * b06) * det,
        (a00 * b09 - a01 * b07 + a02 * b06) * det,
        (a31 * b01 - a30 * b03 - a32 * b00) * det,
        (a20 * b03 - a21 * b01 + a22 * b00) * det,
    ]
    return out


# TODO: 迁移至公用处
def get_probe_data(volume_path: str):
    """
    Helper to prepare need data to calculate probe.
    """
    sitk_img = sitk.ReadImage(volume_path)
    original_lpi_array = sitk.GetArrayFromImage(sitk_img)
    # 物理信息
    spacing_x, spacing_y = sitk_img.GetSpacing()[:2]
    origin_lpi = sitk_img.TransformIndexToPhysicalPoint([0, 0, sitk_img.GetDepth() - 1])
    thickness = (
        sitk_img.TransformIndexToPhysicalPoint([0, 0, sitk_img.GetDepth() - 1])[2]
        - sitk_img.TransformIndexToPhysicalPoint([0, 0, sitk_img.GetDepth() - 2])[2]
    )
    thickness = 1.5
    orientation = sitk_img.GetDirection()[:-3]
    origin1 = list(origin_lpi).copy()
    origin1[2] -= thickness

    lps_ijk = get_lps_ijk(
        spacing_y, spacing_x, origin_lpi, origin1, orientation, ratio=2.0
    )
    # 获取中线信息
    url = "http://172.16.3.35:3333/series/1.2.156.112605.189250946070725.20181216014141.3.5316.10/predict/ct_aorta_treeline"
    import requests

    treeline = requests.get(url).json()
    return original_lpi_array, lps_ijk, treeline, sitk_img.GetSpacing(), origin_lpi


def read_lines(in_lines):
    out = {"lines": {}, "types": {}}
    for label, idxs in in_lines["typeShow"].items():
        line = []
        types = []
        for idx in idxs["data"]:
            idx = str(idx)
            line.extend(in_lines["lines"][idx])
            types.append({in_lines["types"][idx]: len(in_lines["lines"][idx])})
        out["lines"][label] = line
        out["types"][label] = types
    return out


def calculate_frenet(curve_points):
    length = len(curve_points)
    new_points = curve_points[1:].copy()
    new_points.append(curve_points[-1])

    tangents = []
    for i in range(length):
        sub_p = np.subtract(new_points[i], curve_points[i])
        normalize_p = vec3_normalize(sub_p)
        tangents.append(normalize_p)

    normals: list = list()
    bi_normals: list = list()
    n = vec3_normalize(np.subtract(tangents[1], tangents[0]))
    b = vec3_normalize(np.cross(tangents[0], n))
    for i in range(length):
        t = tangents[i]
        n = vec3_normalize(np.cross(b, t))
        b = vec3_normalize(np.cross(t, n))
        normals.append(n)
        bi_normals.append(b)
    tangents[-1] = tangents[-2]
    normals[-1] = normals[-2]
    bi_normals[-1] = bi_normals[-2]
    return tangents, normals, bi_normals


def vec3_normalize(p):
    x, y, z = p
    len_p = x * x + y * y + z * z
    if len_p > 0:
        len_p = 1 / math.sqrt(len_p)
    out = [x * len_p, y * len_p, z * len_p]
    return out


def vec3_scale(a, b):
    out = [a[0] * b, a[1] * b, a[2] * b]
    return out


def _save_dcm(array, dcm_path, spacing, options):
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, dcm_path)
    modify_dcm_tags(dcm_path, original_ds, options)


if __name__ == "__main__":
    volume_path = "/media/tx-deepocean/Data/DICOMS/demos/aorta/1.2.156.112605.189250946070725.20181216014141.3.5316.10.nii.gz"
    original_lpi_array, lps_ijk, treeline, spacing, origin = get_probe_data(volume_path)
    all_line = read_lines(treeline["treeLines"]["data"])
    print(all_line["lines"].keys())
    _, normals, bi_normals = calculate_frenet(all_line["lines"]["vessel16"])
    points_data = {
        "points": all_line["lines"]["vessel16"],
        "normals": normals,
        "bi_normals": bi_normals,
    }
    cursor_index = 20
    probe_arr, _ = gen_probe(
        original_lpi_array[::-1], lps_ijk, points_data, cursor_index
    )
    import cv2

    cv2.imwrite("./demo.png", probe_arr)
    # _save_dcm(probe_arr, "./demo.dcm",spacing, {"origin": origin})
