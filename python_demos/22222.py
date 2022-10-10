import math

import cv2
import numpy as np
import SimpleITK as sitk
from mismatch_utils import convert_ijk_to_xyz, gray2rgb_array

tmip_img = sitk.ReadImage(
    "/media/tx-deepocean/Data/DICOMS/demos/28/TMIP_NO_SKULL.nii.gz"
)
# origin = tmip_img.GetOrigin()

origin = [-123.0, 10.300000190734863, 193.8000030517578]
print(f"11111111111111: {origin}")
tmip_array = sitk.GetArrayFromImage(tmip_img)
dcm_slice = 13


def plane_intersection(plane1_origin, plane1_normal, plane2_origin, plane2_normal):
    M = np.stack([plane1_normal.T, plane2_normal.T])
    B = np.stack([plane1_origin, plane2_origin])
    x0 = np.linalg.lstsq(M, np.diag(np.dot(M, B.T)), rcond=None)[0]
    _, _, D = np.linalg.svd(M)
    t = D[2]
    assert np.allclose(np.dot(x0 + t - plane1_origin, plane1_normal), 0)
    assert np.allclose(np.dot(x0 + t - plane2_origin, plane2_normal), 0)
    return x0, t


def line_intersection(x0, v0, x1, v1):
    [t0, t1] = np.linalg.lstsq(np.stack([v0, -v1]).T, x1 - x0, rcond=None)[0]
    # print(t0, t1)
    assert np.allclose(x0 + t0 * v0, x1 + t1 * v1)
    return x0 + t0 * v0


def get_abcd(p1, p2, p3):
    a = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
    b = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2])
    c = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    d = 0 - (a * p1[0] + b * p1[1] + c * p1[2])
    return a, b, c, d


gray_array = tmip_array[dcm_slice, :, :]
rgb_array = gray2rgb_array(gray_array, 100, 50)

spacing = np.array([0.48828125, 0.48828125, 5])
direction = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -0.99999980000004]])
# 中线点法式
line_normal = np.array([-0.9854396172747589, -0.14509501507031347, 0.0886351922608738])
line_point = np.array([5.1300035134078, 125.09724815376586, 145.74301694617915])
position_k = origin[2] + dcm_slice * spacing[2]
position = [origin[0], origin[1], position_k]
# 三点确定层面点法式
(x0, y0, z0) = [origin[0], origin[1], position_k]
(x1, y1, z1) = [origin[0], origin[1] + spacing[1], position_k]
(x2, y2, z2) = [origin[0] + spacing[0], origin[1], position_k]
a, b, c, d = get_abcd((x0, y0, z0), (x1, y1, z1), (x2, y2, z2))
a = a / math.sqrt(a ** 2 + b ** 2 + c ** 2)
b = b / math.sqrt(a ** 2 + b ** 2 + c ** 2)
c = c / math.sqrt(a ** 2 + b ** 2 + c ** 2)
print(a, b, c, d)

slice_normal = np.array([a, b, c])
slice_point = np.array([origin[0], origin[1], position_k])

# 获取xy坐标
# p0, n0 = plane_intersection(line_point, line_normal, curPlane[0], curPlaneN)
p0, n0 = plane_intersection(line_point, line_normal, slice_point, slice_normal)
print(f"模型预测结果的法线: {line_normal}")
print(f"****层面的法线: {slice_normal}")
print(f"*****p0, n0: {p0, n0}")
x1 = np.array(tmip_img.TransformIndexToPhysicalPoint([0, 30, dcm_slice]))
x2 = np.array(tmip_img.TransformIndexToPhysicalPoint([0, 482, dcm_slice]))
v1 = np.array(tmip_img.GetDirection()[:3])

# x1 = [origin[0], origin[1] + spacing[1] * 512*0.1, position_k]
# x2 = [origin[0], origin[1] + spacing[1] * 512*0.9, position_k]

p1 = line_intersection(p0, n0, x1, v1)
p2 = line_intersection(p0, n0, x2, v1)
print(f"直线输入: {p0, n0, x1, v1}")
# p_idx1 = tmip_img.TransformPhysicalPointToContinuousIndex(p1)[:2]
# p_idx2 = tmip_img.TransformPhysicalPointToContinuousIndex(p2)[:2]

# print(f'******模型的点: {p_idx1, p_idx2}')
# p0 = convert_ijk_to_xyz(p0, position, spacing)
# p = convert_ijk_to_xyz(p, position, spacing)
p_idx1 = convert_ijk_to_xyz(p1, position, spacing)
p_idx2 = convert_ijk_to_xyz(p2, position, spacing)
print(f"******我们的点: {p_idx1, p_idx2}")
cv2.circle(rgb_array, (int(p_idx1[0]), int(p_idx1[1])), 5, (255, 0, 0), 5)
cv2.line(
    rgb_array,
    (int(p_idx1[0]), int(p_idx1[1])),
    (int(p_idx2[0]), int(p_idx2[1])),
    (255, 0, 0),
    3,
)
cv2.imwrite("./result.jpg", rgb_array)
