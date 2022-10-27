import math

import numpy as np
import SimpleITK as sitk


def calculate_distances(points):
    out_distance = 0
    distance_list = []
    length = len(points)
    if length < 2:
        return 0, 0, []
    for i in range(1, length, 1):
        distance = vec3_distance(points[i], points[i - 1])
        out_distance += distance
        distance_list.append(distance)
    average_distance = out_distance / (length - 1)
    return out_distance, average_distance, distance_list


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


def get_lps_ijk(spacing_y, spacing_x, origin0, origin1, orientation, ratio=2.0):
    di = vec3_scale(orientation, spacing_y / ratio)
    dj = vec3_scale(orientation[3:], spacing_x / ratio)
    dk = np.subtract(origin1, origin0)
    scale_k = [dk[0], dk[1], dk[2] / ratio]
    ijk_lps = []
    ijk_lps.extend(di)
    ijk_lps.append(0)
    ijk_lps.extend(dj)
    ijk_lps.append(0)
    ijk_lps.extend(scale_k)
    ijk_lps.append(0)
    ijk_lps.extend(origin0)
    ijk_lps.append(1)
    for i in range(len(ijk_lps)):
        ijk_lps[i] = float(ijk_lps[i])
    try:
        out = np.linalg.inv(np.array(ijk_lps).reshape((4, 4)))
    except Exception as e:
        logger.error(f"Matrix is not a square or trans failed for {e}")
        out = None
    return out


def vec3_sub(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return [x1 - x2, y1 - y2, z1 - z2]


def vec3_normalize(p):
    x, y, z = p
    len_p = x * x + y * y + z * z
    if len_p > 0:
        len_p = 1 / math.sqrt(len_p)
    out = [x * len_p, y * len_p, z * len_p]
    return out


def vec3_cross(p1, p2):
    ax, ay, az = p1
    bx, by, bz = p2
    out = [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]
    return out


def vec3_scale(a, b):
    out = [a[0] * b, a[1] * b, a[2] * b]
    return out


def vec3_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec3_distance(a, b):
    x = b[0] - a[0]
    y = b[1] - a[1]
    z = b[2] - a[2]
    return math.sqrt((x * x + y * y + z * z))


def save_dcm(origin_ds, np_array, save_path):
    ds = origin_ds.copy()
    ds.Rows, ds.Columns = np_array.shape[0], np_array.shape[1]
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.WindowWidth = 2500
    ds.WindowCenter = 480
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelData = np_array.tobytes()
    ds.save_as(save_path)


def resize_hu(np_array: np.ndarray, final_width=512, final_height=512):
    """Resize np array to (final_width, final_height)"""
    lh, lw = np_array.shape
    if lw != 512 or lh != 512:
        new_width = _calculate_width(lw, lh)
        image = sitk.GetImageFromArray(np_array)
        original_spacing = image.GetSpacing()
        new_spacing = [(lw - 1) * original_spacing[0] / (new_width - 1)] * 2
        new_height = int((lh - 1) * original_spacing[1] / new_spacing[1])
        new_size = [new_width, new_height]
        image = sitk.Resample(
            image,
            new_size,
            sitk.Transform(),
            sitk.sitkLinear,
            image.GetOrigin(),
            new_spacing,
            image.GetDirection(),
            0,
            sitk.sitkInt16,
        )
        new_image = sitk.Image([final_width, final_height], sitk.sitkInt16)
        new_image = sitk.RescaleIntensity(new_image, -2000, -2000)
        x, y = (final_width - new_width) // 2, (final_height - new_height) // 2
        new_array = sitk.GetArrayFromImage(new_image)
        image_array = sitk.GetArrayFromImage(image)
        new_array[y : y + new_height, x : x + new_width] = image_array
        return new_array
    return np_array


def _calculate_width(width: int, height: int):
    aspect = width / height
    x, y = 512, 477

    def round_aspect(number, key):
        return max(min(math.floor(number), math.ceil(number), key=key), 1)

    if x / y >= aspect:
        x = round_aspect(y * aspect, key=lambda n: abs(aspect - n / y))
    return x
