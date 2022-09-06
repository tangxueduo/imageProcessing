import cv2
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

dcm_file = "/media/tx-deepocean/Data/DICOMS/demos/TMAX/TMAX023.dcm"
dcm_img_hu = sitk.ReadImage(dcm_file, sitk.sitkUInt8)


# 将hu值图像rescale到0-255的范围
rescaler = sitk.RescaleIntensityImageFilter()
rescaler.SetOutputMaximum(255)
rescaler.SetOutputMinimum(0)
dcm_img_grey = rescaler.Execute(dcm_img_hu)

# 缩减维度，三维图像变成二维，也可用sitk.sequeeze()
dcm_img_3d = sitk.GetArrayFromImage(dcm_img_grey)
dcm_img_2d = dcm_img_3d[0, :, :]

# 灰度图片变成伪彩色
color_img = cv2.applyColorMap(dcm_img_2d, cv2.COLORMAP_JET)  # 自己测试turbo

background_rgb = [color_img[0, 0, 0], color_img[0, 0, 1], color_img[0, 0, 2]]
background_pixel = np.where(
    (color_img[:, :, 0] == background_rgb[0])
    & (color_img[:, :, 1] == background_rgb[1])
    & (color_img[:, :, 2] == background_rgb[2])
)
color_img[background_pixel] = [0, 0, 0]

cv2.imshow("test", color_img)
cv2.imwrite("./cvmap.png", color_img)
# key = cv2.waitKey(0)
