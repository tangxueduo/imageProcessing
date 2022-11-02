import numpy as np

# from skimage import measure
# from typing import Dict

# # cc, cc_num = measure.label(mask, return_num=True, connectivity=1)
# # arr = np.ones((16, 512, 512))
# # arr[2:250:250] = 1
# # arr[3:250:250] = 1
# # arr[4:250:250] = 1

# # arr[2:260:260] = 1
# # arr[3:260:260] = 1
# # arr[4:260:260] = 1

# arr = np.array([
#     [2, 3, 4],
#     [1, 2, 3],
#     [4, 5, 6]
# ])

# direction = np.array([
#     [-1,0,0],
#     [0,-1,0],
#     [0,0,1]
# ])

# rotate_m = np.array([
#     [0.965339, -0.0903941, -0.244845],
#     [0.244845, 0.638547, 0.729595],
#     [0.0903941, -0.764256, 0.638547]
#     ])
# rotate_m = rotate_m.T
# print(rotate_m)

# mask = np.ones((16, 512, 512))
# mask_slice = (mask[13,:,:])
# mask_slice_rotate = np.dot(rotate_m, mask_slice)
# print(mask_slice_rotate)


# # 左乘 变换矩阵-> M1 = T * M
# # matrix.setRotate(θ);
# # matrix.preTranslate(-10, -10); // 先乘 m` = m*t 右乘
# # matrix.postTranslate(10, 10); // 后乘 m` = t*m 左乘

# res = np.dot(direction, arr)
# print(res)


# sp1 = np.array([0.5076851935099742, 0.5585940000000003, 0.5680391231265176])
sp = np.array([0.558594, 0.558594, 0.625000])

m = np.array(
    [
        [-0.908863, 0, -0.417096],
        [0, 1, 0],
        [-0.417096, 0, 0.908863],
    ]
)
# 先转lps ,再转置
print(f"{m.T}, ******\n{m[0,:]}")
print("*******************")

print(sp)
print(np.dot(sp, m.T))
