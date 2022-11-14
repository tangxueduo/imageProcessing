import numpy as np
import cv2
import SimpleITK as sitk


nii_path = "/media/tx-deepocean/Data/DICOMS/RESULT/volume/1.2.840.113619.2.416.77348009424380358976506205963520437809.nii.gz"
img_arr = sitk.GetArrayFromImage(sitk.ReadImage(nii_path))


roi = img_arr[400,:,:].astype(np.uint8)
roi[0,0] = 0
# roi[:,:]=(255,0,0)
print(roi.shape)

# 
pt1 = (74, 0)
pt2 = (0, 144)
pt3 = (144, 144)

triangle_cnt = np.array( [pt1, pt2, pt3] )
center = (512//2, 512//2)
M = cv2.getRotationMatrix2D(center, 40, 0.75)
# M = np.array([[0.821812, -0.168383, 100],[-0.0268499, -0.965716, 100]])
# M = np.array([
#     [    0.821812, -0.168383, 0.544309],
#     [-0.0268499, -0.965716, -0.258208],
#     [-0.569126, -0.197584, 0.798158]
#     ])
img = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]))
# img = np.dot(roi, M)
# cv2.drawContours(img, [triangle_cnt], 0, (0,255,0), -1)
# img = np.roll(img, 1)

cv2.imshow("origin", img)
cv2.imwrite("./test.png", img)

while 1:
    if cv2.waitKey(100) == 27:  # 100ms or esc(ascii 等于27, 跳出循环)
        break
cv2.destroyAllWindows()