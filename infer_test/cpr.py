import ctypes
import logging
import math
import multiprocessing
import os

import numpy as np
from numpy.ctypeslib import ndpointer
from PIL import Image

from infer_test.curve_interplolator import CurveInterpolator

# from common.comb_process_fucs import (
#     get_cpr_cursor,
#     gray2rgb_array,
#     insert_cpr_cursor,
#     insert_lumen_cursor,
#     insert_word,
#     resize_2d_hu,
#     resize_cpr,
#     resize_lumen,
#     resize_probe,
# )
from .utils import (
    calculate_distances,
    resize_hu,
    save_dcm,
    vec3_cross,
    vec3_distance,
    vec3_normalize,
    vec3_scale,
)


class CoronaryCprPredictor:
    """Render CPR, Lumen, Combination."""

    def __init__(self, original_lpi_array: np.ndarray, lps_ijk: np.ndarray, tree_line):
        """Inits class
        Args:
            original_lpi_array: lpi 坐标系下的 img_array
            lps_ijk:
        """
        self.original_lpi_array = original_lpi_array
        self.lps_ijk = lps_ijk
        self.tree_line = tree_line
        dirname, filename = os.path.split(os.path.abspath(__file__))
        lib = ctypes.cdll.LoadLibrary(os.path.join(dirname, "forCpr.so"))
        self.find_cpr_in_c = lib.findCpr
        self.find_cpr_in_c.restype = None
        self.find_cpr_in_c.argtypes = [
            ndpointer(ctypes.c_int16, flags="C_CONTIGUOUS"),  # im
            ndpointer(ctypes.c_int16, flags="C_CONTIGUOUS"),  # cprIm
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # line
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # m
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # u
            ctypes.c_int,  # height for cprIm
            ctypes.c_int,  # width
            ctypes.c_int,  # imD
            ctypes.c_int,  # imH
            ctypes.c_int,  # imW
        ]
        self.find_lumen_in_c = lib.findLumen
        self.find_lumen_in_c.restype = None
        self.find_lumen_in_c.argtypes = [
            ndpointer(ctypes.c_int16, flags="C_CONTIGUOUS"),  # im
            ndpointer(ctypes.c_int16, flags="C_CONTIGUOUS"),  # lumenIm
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # points
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # normals
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # tangents
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # m
            ctypes.c_float,  # averageDistance
            ctypes.c_float,  # radTheta
            ctypes.c_int,  # height for lumenIm
            ctypes.c_int,  # width
            ctypes.c_int,  # imD
            ctypes.c_int,  # imH
            ctypes.c_int,  # imW
        ]

    def predict_cpr(self, label: str, points: list, theta: int, ds):
        im_stretched, _ = self.predict_stretched(points, theta=theta, ratio=2.0)
        np_arr = resize_hu(im_stretched)
        # print(np_arr.shape)
        save_path = f"./{label}cpr.dcm"
        # save_dcm(ds, np_arr, save_path)

    def predict_stretched(
        self,
        points,
        phi=0,
        theta=0,
        ratio=2.0,
    ):
        """Render CPR
        Args:
            points: 中线
        Returns:
            Numpy
        """
        # cpr 这里 spacing 暂时写死 0.5
        spacing_y, spacing_x = 0.5, 0.5
        rad_phi = (phi / 180) * math.pi
        rad_theta = (theta / 180) * math.pi
        spec_vector = [
            math.cos(rad_phi) * math.cos(rad_theta),
            math.cos(rad_phi) * math.sin(rad_theta),
            math.sin(rad_phi),
        ]

        xs: list = list()
        length = len(points)
        for i in range(length):
            xs.append(np.dot(spec_vector, points[i]))
        max_x, min_x = max(xs) + 40, min(xs) - 40
        xs = [i - min_x for i in xs]

        distance_length = 0
        y = np.subtract(points[0], vec3_scale(spec_vector, xs[0]))
        ys: list = list()
        for i in range(length):
            yi = vec3_scale(spec_vector, xs[i])
            yi = np.subtract(points[i], yi)
            distance_length += np.linalg.norm(y - yi)
            ys.append(yi)
            y = ys[i]

        width = int(math.floor((max_x - min_x) / spacing_y + 1))
        height = int(math.floor(distance_length / spacing_x))

        curve = CurveInterpolator(ys, tension=0.5)
        ys2 = curve.get_points(step=0.5, samples=height)

        height = len(ys2)
        u = vec3_scale(spec_vector, spacing_x)
        m = self.lps_ijk

        pixel_data = self.find_hus(height, width, ys2, m, u, self.original_lpi_array)

        # plane_line = get_center_line_on_plane(
        #     ys, xs, spacing_y, spacing_x, width, height
        # )
        plane_line = []
        return pixel_data, plane_line

    def find_hus(self, height, width, ys, m, u, original_lpi_array):
        imD, imH, imW = original_lpi_array.shape
        cprIm = np.ones(shape=(height, width), dtype=np.int16) * (-1024)
        line_np = np.asarray(ys, dtype=np.float32)
        m_np = np.asarray(m, dtype=np.float32)
        u_np = np.asarray(u, dtype=np.float32)
        self.find_cpr_in_c(
            original_lpi_array.astype(np.int16),
            cprIm,
            line_np,
            m_np,
            u_np,
            height,
            width,
            imD,
            imH,
            imW,
        )
        return cprIm
