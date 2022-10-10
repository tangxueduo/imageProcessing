import math

import cv2
import numpy as np
import SimpleITK as sitk
from mismatch_utils import (convert_ijk_to_xyz,
                            convert_noraml_to_standard_plane,
                            get_point_on_plane_by_yz, gray2rgb_array)

origin = [-123.0, 10.300000190734863, 193.8000030517578]
spacing = np.array([0.48828125, 0.48828125, 5])
centerline = {
    "point": [5.1300035134078, 125.09724815376586, 145.74301694617915],
    "vector": [-0.9854396172747589, -0.14509501507031347, 0.0886351922608738],
}

tmip_img = sitk.ReadImage(
    "/media/tx-deepocean/Data/DICOMS/demos/28/TMIP_NO_SKULL.nii.gz"
)
tmip_array = sitk.GetArrayFromImage(tmip_img)
spacing = np.array([0.48828125, 0.48828125, 5])
dcm_slice = 14
gray_array = tmip_array[dcm_slice, :, :]
rgb_array = gray2rgb_array(gray_array, 100, 50)


def get_line_coord(
    dcm_slice: int,
    coefficient_a: float,
    coefficient_b: float,
    coefficient_c: float,
    coefficient_d: float,
):
    """由物理坐标，获取层面中线信息"""
    [i, j, loc] = origin
    # slice_location 递减
    loc = loc - dcm_slice * spacing[2]
    p1, p2 = get_point_on_plane_by_yz(
        coefficient_a,
        coefficient_b,
        coefficient_c,
        coefficient_d,
        512,
        loc,
        spacing[1],
        origin[1],
    )
    # 物理坐标转像素坐标
    new_p1, new_p2 = [p1[0], p1[1], loc], [p2[0], p2[1], loc]
    xyzp1 = convert_ijk_to_xyz(new_p1, [i, j, loc], spacing)
    xyzp2 = convert_ijk_to_xyz(new_p2, [i, j, loc], spacing)
    (x1, y1), (x2, y2) = (xyzp1[0], xyzp1[1]), (
        xyzp2[0],
        xyzp2[1],
    )

    return (x1, y1), (x2, y2)


(
    coefficient_a,
    coefficient_b,
    coefficient_c,
    coefficient_d,
) = convert_noraml_to_standard_plane(centerline["point"], centerline["vector"])
(x1, y1), (x2, y2) = get_line_coord(
    dcm_slice,
    coefficient_a,
    coefficient_b,
    coefficient_c,
    coefficient_d,
)
cv2.line(rgb_array, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
cv2.imwrite("./result.jpg", rgb_array)
