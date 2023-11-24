import json
import os
import threading
import time
from collections import OrderedDict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import requests
import SimpleITK as sitk
from loguru import logger
from radiomics import (featureextractor, firstorder, glcm, gldm, glrlm, glszm,
                       ngtdm, shape, shape2D)

threading.current_thread().name = "Main"


url = "http://172.16.4.5:3333/series/1.2.392.200036.9116.2.6.1.44063.1796265406.1656894518.71296/predict/ct_lung"
ROOT_DIR = Path("/media/tx-deepocean/Data/DICOMS/demos/ct_heart").resolve()
NUM_OF_WORKERS = (
    cpu_count() - 1
)  # Number of processors to use, keep one processor free for other work
if (
    NUM_OF_WORKERS < 1
):  # in case only one processor is available, ensure that it is used
    NUM_OF_WORKERS = 1
# PARAMS = os.path.join("examples/Params.yaml")
CUSTOM_IMAGE_TYPES = {
    "Original": {},
    "Square": {},
    "Gradient": {},
    "Exponential": {},
    "LoG": {},
    "Wavelet": {},
    "Logarithm": {},
    "SquareRoot": {},
}


class GetRadiomics:
    def __init__(self) -> None:
        self.result = requests.get(url).json()
        # 准备基础数据 sitk_image, mask
        self.image_path = str(
            ROOT_DIR
            / "1.2.392.200036.9116.2.6.1.44063.1796265406.1656894518.71296.nii.gz"
        )
        # self.image_nrrd_path = convert_nifti_to_nrrd(str(image_path))

        self.image = sitk.ReadImage(self.image_path)
        self.image = sitk.Cast(self.image, sitk.sitkInt16)
        self.origin = self.image.GetOrigin()
        self.shape = self.image.GetSize()
        logger.warning(f"shape: {self.shape}")
        self.spacing = self.image.GetSpacing()
        logger.info("Init over")
        self.settings = {
            "resampledPixelSpacing": None,
            "binWidth": 10,
            "interpolator": sitk.sitkBSpline,
            "enableCExtensions": True,
            "additionalInfo": False,
        }

        # self.mask_lib = importlib.import_module("ClientGraphicsEngine")

    def cal_pyradiomics(self, req: Dict, base_features: bool = False):
        t0 = time.time()
        nodule_id = req["nodule_id"]
        # 结节轮廓转mask
        # mask_path = "./nodule.nii.gz"
        # convert_contour_to_mask(
        #     nodule_id, self.result, self.origin, self.shape, self.spacing, mask_path, self.mask_lib
        # )
        mask_np = np.zeros_like(sitk.GetArrayFromImage(self.image))
        mask_np[5:10, 5:10, 5:6] = 1
        self.mask = sitk.GetImageFromArray(mask_np)
        self.mask.CopyInformation(self.image)
        self.mask = sitk.Cast(self.mask, sitk.sitkInt16)
        # write to nrrd
        nrrd_path = str(ROOT_DIR / "image.nrrd")
        sitk.WriteImage(self.image, nrrd_path, True)
        sitk.WriteImage(self.mask, str(ROOT_DIR / "mask.nrrd"), True)

        # mask = sitk.ReadImage(mask_path)
        # inputCSV = "/media/tx-deepocean/Data/DICOMS/demos/ct_heart/input.csv"
        results = cal_custom_radiomics(
            self.settings, CUSTOM_IMAGE_TYPES, self.image, self.mask
        )

        if base_features:
            enabled_feature_clses = [
                firstorder.RadiomicsFirstOrder,
                shape.RadiomicsShape,
                shape2D.RadiomicsShape2D,
                glcm.RadiomicsGLCM,
                glrlm.RadiomicsGLRLM,
                glszm.RadiomicsGLSZM,
                gldm.RadiomicsGLDM,
                ngtdm.RadiomicsNGTDM,
            ]
            res = cal_base_features(settings, self.image, mask, enabled_feature_clses)

        logger.warning(f"cost time: {time.time() - t0}")
        # 调用　pyradiomics
        logger.warning(f"****result: {len(results)}")
        return results


def cal_custom_radiomics(settings, enable_image_types, image, mask, label=1):
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    logger.info(f"Extraction parameters:\n\t, {extractor.settings}")
    # 原图，　平方，　梯度，指数，高斯拉普拉斯，小波，对数　每个内层dict里面还可以传递公式的其他因子
    # refer to: https://pyradiomics.readthedocs.io/en/v3.0.1/customization.html

    extractor.enableImageTypes(**enable_image_types)
    # 只传一个，　　例如: 原始图像
    # extractor.enableImageTypeByName('Original')

    extractor.enableAllFeatures()
    # result = extractor.execute(image, mask, label=label)
    result = extractor.execute(
        "/media/tx-deepocean/Data/DICOMS/demos/ct_heart/image.nrrd",
        "/media/tx-deepocean/Data/DICOMS/demos/ct_heart/mask.nrrd",
        label=label,
    )
    logger.warning(len(result))
    # 过滤只返回array 结果数值
    result = {
        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()
    }
    return result


def convert_nifti_to_nrrd(nifti_path):
    # 读取 NIfTI 文件
    nifti_image = sitk.ReadImage(nifti_path)
    nrrd_path = nifti_path.replace(".nii.gz", ".nrrd")
    # 将图像保存为 NRRD 格式
    sitk.WriteImage(nifti_image, nrrd_path)
    return nrrd_path


def cal_base_features(
    settings: dict, image: sitk.Image, mask: sitk.Image, enabled_feature_clses: list
):
    res = {}
    for features_cls in enabled_feature_clses:
        features = features_cls(image, mask, **settings)
        features.enableAllFeatures()
        result = features.execute()
        res.update({k: round(v.tolist(), 4) for k, v in result.items()})
    return res


def convert_contour_to_mask(
    nodule_id, result, origin, shape, spacing, mask_path, mask_lib
):
    nodule_json = list(filter(lambda x: x["id"] == nodule_id, result))[0]
    graphics_nodule_json = {
        "boxes": nodule_json["boxes"],
        "contour3D": nodule_json["contour3D"],
        "volume_origin": origin,
        "volume_shape": shape,
        "volume_spacing": spacing,
        "output_path": mask_path,
    }
    # 　调用 GenerateLungVTP
    re_code = mask_lib.GenerateLungVTP(json.dumps(graphics_nodule_json))
    logger.debug(re_code)


def main():
    radiomics_obj = GetRadiomics()
    req = {
        "series_iuid": "1.3.46.670589.33.1.63781653264325099000003.5309891207461919284",
        "nodule_id": 1,  # 结节id
        "min3DDiameter": 2.22,  # 3D 最小径 来自模型
        "maxSliceArea": 2.22,  # 最大面面积　来自模型
        "quality": 2.22,  # 结节质量　来自模型
        "real_percent": 2.33,  # 实性占比，来自前端
    }
    pvat_res = radiomics_obj.cal_pyradiomics(req)


if __name__ == "__main__":
    main()
