import numpy as np

origin = np.array([-143, -135.908, -123.349])
orientation = np.array([1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000])
spacing = np.array([0.5586, 0.5586, 0.625])
print(f"******原始 origin spacing: {origin}, {spacing}")

"""
轴: (-123.349, 244.151)
冠： (-149.5335, 135.9080)
失: (-143, 142.4415)
"""
slice_index = 1
# 获取右上的原点
# 轴
ijk_k = origin[2] + spacing[2] * 588
# 失
ijk_j = origin[1] + spacing[1] * 511
# 冠
ijk_i = origin[0] + spacing[0] * 511
rb_origin = [ijk_i, ijk_j, ijk_k]

print(rb_origin)

# vtkMatirx
# matrix = [
#     -0.985206, 0, -0.171375, 5.83794,
#     0,         1,  0,        -6.81277,
#     -0.171375, 0, 0.985206, 28.445,
#     0, 0, 0, 1
# ]
# 轴位
matrix = np.array([[-0.877106, 0, -0.480296], [0, 1, 0], [-0.480296, 0, 0.877106]])
center = [-7.84112, -6.81277, 75.3017]

# spacing = [round(i, 5) for i in [0.5503300783328972, 0.5585940000000001, 0.6158521788023037]]

# 冠状为
# matrix = np.array([
#     [0, -0.17135, -0.985206],
#     [-1, 0, 0],
#     [0, 0.985206, -0.17135]
# ]
# )
matrix = matrix.T
direction = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
matrix = np.dot(direction, matrix)
print(matrix)

origin = np.dot(origin, matrix)
rb_origin = np.dot(rb_origin, matrix)

sp1 = np.dot(spacing, matrix)
sp1 = [0.49351133883290305, 0.5585940000000003, 0.5521802718442456]

print(f"变换后的origin {origin}, spacing {sp1}")
print(f"*******rb_origin: {rb_origin}")
z_idx = int((center[2] - origin[2]) / sp1[2])

k_min = origin[2] + sp1[2] * 0
k_max = rb_origin[2] + sp1[2] * 0
z_min = 0
z_max = int((k_max - origin[2]) / sp1[2])

print(z_idx, z_min, z_max, k_min, k_max)
