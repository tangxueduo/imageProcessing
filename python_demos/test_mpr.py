#import nibabel as nib
# from PIL import Image
import time

import numpy as np
import SimpleITK as sitk

#https://blog.csdn.net/u014264373/article/details/119545255
#https://www.codenong.com/54742326/
#https://blog.csdn.net/lavinia_chen007/article/details/125389503

if __name__ == '__main__': 
  nii_path = "/media/tx-deepocean/Data/DICOMS/RESULT/volume/1.2.840.113619.2.416.77348009424380358976506205963520437809.nii.gz" #nii.gz路径
  dicom_path = "./mpr.dcm"
  sitk_img = sitk.ReadImage(nii_path)

  img = sitk.GetArrayFromImage(sitk_img)
  start = time.time()

  layernum = 450
  mpr_mode = "sagittal"
  if mpr_mode == "coronal":
    temp_img = np.flipud(img[:,layernum,:]) #上下反转
    select_img = sitk.GetImageFromArray(temp_img)
  elif mpr_mode == "axial":
    temp_img = img[layernum,:,:]
    select_img = sitk.GetImageFromArray(temp_img)
  elif mpr_mode == "sagittal":
    temp_img = np.flipud(img[:,:,layernum]) #上下反转
    select_img = sitk.GetImageFromArray(temp_img.astype(np.int16))

  select_img.SetMetaData('0028|1050', '300')
  select_img.SetMetaData('0028|1051', '800')
  sitk.WriteImage(select_img, dicom_path)
  
  end = time.time()
  print(end - start)


