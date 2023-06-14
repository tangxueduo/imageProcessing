# #coding=utf-8
# import SimpleITK as sitk
# import os
# import ctypes
# import numpy as np

# def dcm2nii(dcms_path, nii_path):
# 	# 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
#     reader = sitk.ImageSeriesReader()
#     dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
#     reader.SetFileNames(dicom_names)
#     image2 = reader.Execute()
# 	# 2.将整合后的数据转为array，并获取dicom文件基本信息
#     image_array = sitk.GetArrayFromImage(image2)  # z, y, x
#     origin = image2.GetOrigin()  # x, y, z
#     spacing = image2.GetSpacing()  # x, y, z
#     direction = image2.GetDirection()  # x, y, z
# 	# 3.将array转为img，并保存为.nii.gz
#     image3 = sitk.GetImageFromArray(image_array)
#     image3.SetSpacing(spacing)
#     image3.SetDirection(direction)
#     image3.SetOrigin(origin)
#     sitk.WriteImage(image3, nii_path)


# def get_heart_vr_image(
#     mylib, slice_height, slice_width, slice_count, dcm_arr, mask_arr
# ):
#     dcm_arr = dcm_arr.astype(np.int16)
#     image = dcm_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
#     mask = mask_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
#     mylib.GetHeartVRImage(
#         image,
#         mask,
#         ctypes.c_int(slice_width),
#         ctypes.c_int(slice_height),
#         ctypes.c_int(slice_count),
#     )
#     new_image = np.ctypeslib.as_array(
#         image, shape=(slice_count, slice_width, slice_height)
#     )
#     new_image = np.reshape(new_image, (slice_count, slice_width, slice_height))
#     new_mask = np.ctypeslib.as_array(
#         mask, shape=(slice_count, slice_width, slice_height)
#     )
#     new_mask = np.reshape(new_mask, (slice_count, slice_width, slice_height))
#     return new_image, new_mask


# def cut_img(seg_lps_array, origin_lps_array):
#     """
#     分割图像
#     :param seg_lps_array: mask数据
#     :param origin_lps_array: 原图数据
#     """
#     zs, ys, xs = np.where(seg_lps_array > 0)
#     zmin, zmax, ymin = np.min(zs), np.max(zs) + 2, np.min(ys)
#     ymax, xmin, xmax = np.max(ys) + 2, np.min(xs), np.max(xs) + 2
#     cut_box = {
#         "zmin": zmin,
#         "zmax": zmax,
#         "ymin": ymin,
#         "ymax": ymax,
#         "xmin": xmin,
#         "xmax": xmax,
#     }
#     cut_original_lps_array = origin_lps_array[zmin:zmax, ymin:ymax, xmin:xmax]
#     cut_seg_lps_array = seg_lps_array[zmin:zmax, ymin:ymax, xmin:xmax]
#     return cut_original_lps_array, cut_seg_lps_array, cut_box


# def save_mhd():
#     """生成并存储 mhd 文件, 用于体绘制渲染(vr/vr_tree/vmip)

#     冠脉中会生成三种 mhd 文件， 带心脏(vr)、不带心脏(vr_tree, vmip_tree)和带心脏(vmip)
#     """
#     sitk_img = sitk.ReadImage("/media/tx-deepocean/Data/DICOMS/demos/output_lr/dcm/data.nii.gz")
#     spacing = sitk_img.GetSpacing()
#     lps_np = sitk.GetArrayFromImage(sitk_img)
#     seg_sitk_img = sitk.ReadImage("/media/tx-deepocean/Data/DICOMS/demos/output_lr/output+debug/TX000001-1518026519-33402690-8/output_lps_seg.nii.gz")
#     seg_lps_np = sitk.GetArrayFromImage(seg_sitk_img)
#     myLib = ctypes.cdll.LoadLibrary("/usr/lib/libInteractiveGraphicsEngine.so")

#     (
#         dilate_lps_np,
#         dilate_seg_lps_np,
#     ) = get_heart_vr_image(
#         mylib=myLib,
#         slice_width=lps_np.shape[2],
#         slice_height=lps_np.shape[1],
#         slice_count=lps_np.shape[0],
#         dcm_arr=lps_np.flatten(),
#         mask_arr=seg_lps_np.flatten(),
#     )
#     output_path = "/media/tx-deepocean/Data/DICOMS/demos/output_lr/dcm"
#     mhds = [
#         {
#             "is_delete_heart": False,
#             "mhd_file": os.path.join(output_path, "vr.mhd"),
#         },
#         {
#             "is_delete_heart": True,
#             "mhd_file": os.path.join(output_path, "vr_tree.mhd"),
#         },
#     #     {
#     #         "is_delete_heart": False,
#     #         "is_vmip": True,
#     #         "mhd_file": os.path.join(output_path, "vmip.mhd"),
#     #     },
#     ]
#     (
#         cut_dilate_lps_np,
#         cut_dilate_seg_lps_np,
#         cut_box,
#     ) = cut_img(
#         dilate_seg_lps_np, dilate_lps_np
#     )
#     xmin, ymin, zmin = (
#         cut_box["xmin"],
#         cut_box["ymin"],
#         cut_box["zmin"],
#     )
#     origin = seg_sitk_img.GetOrigin()
#     for mhd in mhds:
#         im = cut_dilate_lps_np.copy()
#         im[cut_dilate_seg_lps_np == 0] = -120
#         if mhd["is_delete_heart"]:
#             im[cut_dilate_seg_lps_np == 2] = -120
#         im_sitk = sitk.GetImageFromArray(im)
#         im_sitk.SetSpacing(spacing)
#         im_sitk.SetOrigin(
#             (
#                 origin[0] + xmin * spacing[0],
#                 origin[1] + ymin * spacing[1],
#                 origin[2] + zmin * spacing[2],
#             )
#         )
#         im_sitk.SetDirection(seg_sitk_img.GetDirection())
#         sitk.WriteImage(im_sitk, mhd["mhd_file"])


# if __name__ == '__main__':
#     dcms_path = f'/media/tx-deepocean/Data/DICOMS/demos/output_lr/dcm/TX000001-1518026519-33402690-8'  # dicom序列文件所在路径
#     nii_path = f'/media/tx-deepocean/Data/DICOMS/demos/output_lr/dcm/data.nii.gz'  # 所需.nii.gz文件保存路径
#     # dcm2nii(dcms_path, nii_path)
#     save_mhd()
