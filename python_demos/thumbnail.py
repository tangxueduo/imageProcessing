import cv2
import numpy as np
import pydicom
import SimpleITK as sitk


def linear_transfor_intensity(image, window_center, window_width):
    return sitk.IntensityWindowing(
        image,
        window_center - window_width / 2,
        window_center + window_width / 2,
        0.0,
        255.0,
    )


def _get_thumbnail(image, window_center, window_width):
    if image.GetNumberOfComponentsPerPixel() == 1:
        # 灰度图像的亮度调整：0~255 后反转单色1
        if (
            image.HasMetaDataKey("0028|0004")
            and image.GetMetaData("0028|0004").strip() == "MONOCHROME1"
        ):
            # image = sitk.InvertIntensity(image, maximum=255)
            pass

    # 亮度映射(0-255)
    # MR and DSA default mapping
    if (
        (image.HasMetaDataKey("0008|0060") and image.GetMetaData("0008|0060") == "MR")
        or (
            image.HasMetaDataKey("0008|0060") and image.GetMetaData("0008|0060") == "XA"
        )
        or (
            image.HasMetaDataKey("0008|0060") and image.GetMetaData("0008|0060") == "DX"
        )
    ):
        itkimage = sitk.RescaleIntensity(image, 0, 255)
    else:
        # window linear mapping
        itkimage = linear_transfor_intensity(image, window_center, window_width)
    # sitk.Image to np.array
    # result = sitk.Cast(itkimage, sitk.sitkUInt8)
    # sitk.WriteImage(result, "./thumbnail.dcm")
    img_array = sitk.GetArrayFromImage(itkimage)
    img_array = img_array.astype()

    cv2.imwrite("result.jpg", img_array)
    print(1111)


def get_win_info(img):
    if not img.HasMetaDataKey("0028|1051") or not img.HasMetaDataKey("0028|1050"):
        img_array = sitk.GetArrayFromImage(img)
        window_width = img_array.max() - img_array.min()
        window_center = 0.5 * (img_array.max() + img_array.min())
    else:
        window_width = img.GetMetaData("0028|1051").strip()
        window_center = img.GetMetaData("0028|1050").strip()
    return float(window_center), float(window_width)


if __name__ == "__main__":
    dcm_path = "/media/tx-deepocean/Data/DICOMS/demos/fanseerror/CW023007-79492481.dcm"

    img = sitk.ReadImage(dcm_path)
    window_center, window_width = get_win_info(img)
    _get_thumbnail(img, window_center, window_width)
