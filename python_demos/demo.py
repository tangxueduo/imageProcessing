import cv2
import numpy as np
import SimpleITK as sitk

nii_path = "/media/tx-deepocean/Data/DICOMS/demos/cerebral_1/cerebral-seg.nii.gz"
img = sitk.ReadImage(nii_path)
img_arr = sitk.GetArrayFromImage(img)

binary_arr = np.zeros(img_arr.shape)
binary_arr[img_arr == 100] = 1
binary_img = sitk.GetImageFromArray(binary_arr)
sitk.WriteImage(binary_img, "./binary_cerebral.nii.gz")
# cv2.imshow("aneurysm", binary_arr)
# while True:
#     if cv2.waitKey(0) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()
