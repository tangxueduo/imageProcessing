import time

import numpy as np
import pydicom
import SimpleITK as sitk

from python_demos.utils import array_to_dicom, draw_text, resize_hu

# https://blog.csdn.net/u014264373/article/details/119545255
# https://www.codenong.com/54742326/
# https://blog.csdn.net/lavinia_chen007/article/details/125389503

if __name__ == "__main__":
    nii_path = "/media/tx-deepocean/Data/DICOMS/demos/aorta/1.2.840.113619.2.404.3.1074448704.575.1620281889.753.4.nii.gz"  # nii.gz路径
    dicom_path = "./mpr.dcm"
    sitk_img = sitk.ReadImage(nii_path)

    img = sitk.GetArrayFromImage(sitk_img)
    start = time.time()

    layernum = 206
    mpr_mode = "sagittal"
    print(img.dtype)
    if mpr_mode == "coronal":
        temp_img = np.flipud(img[:, layernum, :])  # 上下反转
        select_img = sitk.GetImageFromArray(temp_img)
    elif mpr_mode == "axial":
        temp_img = img[layernum, :, :]
        select_img = sitk.GetImageFromArray(temp_img)
    elif mpr_mode == "sagittal":
        temp_img = np.flipud(img[:, :, layernum])  # 上下反转
        select_img = sitk.GetImageFromArray(temp_img.astype(np.int16))
    ds = pydicom.read_file(
        "/media/tx-deepocean/Data/DICOMS/demos/aorta/1.2.840.113619.2.404.3.1074448704.575.1620281889.892.436",
        force=True,
    )  # type:ignore

    temp_img = draw_text(
        resize_hu(temp_img),
        "MPR 1",
        ds,
    )
    print(temp_img.dtype, temp_img.shape)

    array_to_dicom(
        origin_ds=ds,
        np_array=temp_img.astype(np.int16),
        filename=dicom_path,
        is_vr=False,
        instance_number=1,
    )
    # select_img.SetMetaData('0028|1050', '300')
    # select_img.SetMetaData('0028|1051', '800')
    # sitk.WriteImage(select_img, dicom_path)

    end = time.time()
    print(end - start)
