import os

import numpy as np
import SimpleITK as sitk
from loguru import logger
from radiomics import featureextractor, imageoperations

root = "/media/tx-deepocean/Data/DICOMS/demos/ct_heart"
PARAMS = os.path.join("utils/Params.yaml")

origin_img_path = os.path.join(
    root, "1.2.392.200036.9116.2.6.1.44063.1796265406.1656894518.71296.nii.gz"
)

img_path = os.path.join(
    root,
    "1.2.392.200036.9116.2.6.1.44063.1796265406.1656894518.71297_5.12_PE-CTA_SureStar_20220704092609_4.nii",
)
image = sitk.ReadImage(img_path)
origin_image = sitk.ReadImage(origin_img_path)

print(image.GetOrigin(), origin_image.GetOrigin())


def deal_img_mask():
    origin_image_np = sitk.GetArrayFromImage(origin_image)
    mask_np = np.zeros_like(origin_image_np).astype("uint8")
    logger.warning(mask_np.shape)  # , z,y,x
    mask_np[5:100, 5:100, 10:100] = 1
    mask = sitk.GetImageFromArray(mask_np)
    mask.CopyInformation(origin_image)
    # 获取mask　roi 范围

    new_image_np = np.zeros_like(mask_np).astype("int16")
    new_image_np[5:10, :, :] = origin_image_np[5:10, :, :]
    new_image = sitk.GetImageFromArray(new_image_np)
    new_image.CopyInformation(origin_image)

    # sitk.WriteImage(origin_image, os.path.join(root, "image.nrrd"), True)
    # sitk.WriteImage(mask, os.path.join(root, "mask.nrrd"), True)
    return origin_image, mask


def cal_radiomics(image, mask, label=1):

    bb, correctedMask = imageoperations.checkMask(image, mask)
    logger.warning(bb)
    image, mask = imageoperations.cropToTumorMask(image, mask, bb, padDistance=2)
    extractor = featureextractor.RadiomicsFeatureExtractor(PARAMS)
    logger.info(f"Extraction parameters:\n\t, {extractor.settings}")
    # 原图，　平方，　梯度，指数，高斯拉普拉斯，小波，对数　每个内层dict里面还可以传递公式的其他因子
    # refer to: https://pyradiomics.readthedocs.io/en/v3.0.1/customization.html

    # Extract features
    result = extractor.execute(image, mask, label=label)
    logger.warning(len(result))
    # 过滤只返回array 结果数值
    result = {
        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()
    }
    return result


img, mask = deal_img_mask()
cal_radiomics(img, mask)
# cal_radiomics("/media/tx-deepocean/Data/DICOMS/demos/ct_heart/image.nrrd", "/media/tx-deepocean/Data/DICOMS/demos/ct_heart/mask.nrrd")
