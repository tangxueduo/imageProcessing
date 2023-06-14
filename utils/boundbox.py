import ctypes

import numpy as np
from loguru import logger
from numpy.ctypeslib import ndpointer

# import os


def boundbox_3d(arrays: np.ndarray):
    """
    :param arrays: d,h,w = arrays.shape
    :return: [zmin,zmax, ymin,ymax, xmin,xmax]
    C: boundbox_uint8/boundbox_bool (unsigned char *im, int *bound, int depth, int height, int width)
    """
    lib = ctypes.cdll.LoadLibrary("utils/bound_box.so")
    arr_shape = arrays.shape
    if len(arr_shape) != 3:
        logger.error(
            "input arrays shape's length must be 3, but get shape is {}".format(
                arr_shape
            )
        )
        return None

    arr_type = arrays.dtype
    d, h, w = arr_shape
    bbox = np.zeros(shape=(6), dtype=np.int32)
    bbox[0], bbox[2], bbox[4] = d, h, w

    if arr_type == "uint8":
        func = lib.boundbox_uint8
        func.restype = None
        func.argtypes = [
            ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        func(arrays, bbox, d, h, w)
    elif arr_type == "bool":
        func = lib.boundbox_bool
        func.restype = None
        func.argtypes = [
            ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        func(arrays, bbox, d, h, w)
    else:
        logger.error(
            "input arrays' type must in [uint8, bool], but get {}".format(arr_type)
        )
        return None
    bbox[1] += 1
    bbox[3] += 1
    bbox[5] += 1
    if bbox[0] < bbox[1] and bbox[2] < bbox[3] and bbox[4] < bbox[5]:
        return bbox
    logger.warning("input arrays is zero array")
    return None
