import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.append("/usr/lib")
import ClientGraphicsEngine
import numpy as np
import pandas as pd
import SimpleITK as sitk
from app.subapp.ct_chest.schema.radiomics import RadiomicsReq
from app.subapp.ct_chest.utils.extra import ExtraReliability
from app.subapp.ct_chest.utils.repacs import repacs_cli
from loguru import logger
from radiomics import featureextractor, imageoperations

ROOT = Path(__file__).resolve()
PARAMS = str(ROOT.parent.parent / "utils" / "radiomics_params.yaml")


class CALRadiomics(object):
    def __init__(self, req: RadiomicsReq) -> None:
        self.nodule_id, self.series_iuid = req.nodule_id, req.series_iuid
        self.result, _ = repacs_cli.get_all_predict_result(self.series_iuid, "ct_lung")
        self.extra_info = ExtraReliability(self.series_iuid)
        self.image = sitk.ReadImage(self.extra_info.volume_path)
        self.image = sitk.Cast(self.image, sitk.sitkInt16)
        self.origin = self.image.GetOrigin()
        self.shape = self.image.GetSize()
        self.spacing = self.image.GetSpacing()
        mask_dir = os.path.join(
            self.extra_info.data_path, f"RESULT/ssrFeedback/ct_chest/{self.series_iuid}"
        )
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, "nodule_seg.nii.gz")
        convert_contour_to_mask(
            self.nodule_id,
            self.result,
            self.origin,
            self.shape,
            self.spacing,
            mask_path,
        )
        # 填充　mask
        self.mask = padding_mask(self.image, sitk.ReadImage(mask_path))

    def write_radiomics_to_df(self):
        df = pd.DataFrame()
        suffix = ""
        try:
            # cal radiomics
            res = cal_custom_radiomics(self.image, self.mask, label=1)
            # save to df
            df = pd.DataFrame.from_dict(res, orient="index")
            nodule_res = list(filter(lambda x: x["id"] == self.nodule_id, self.result))[
                0
            ]
            suffix = f'{nodule_res["keySliceId"]}_{nodule_res["type"]}.csv'
        except Exception:
            logger.warning(f"cal radiomics error for {traceback.format_exc()}")
        return df, suffix


def extract_roi(image: sitk.Image, st_positon: list, ed_position: list) -> sitk.Image:
    """extrac roi hu by roi mask position
    Args:
        image: hu values, 3d
        st_position: roi start posion
        ed_position: roi end position
    Return:
        extracted roi image.
    """
    image_np = sitk.GetArrayFromImage(image)
    empty_image = np.zeros_like(image_np)
    empty_image[
        st_positon[2] : ed_position[2],
        st_positon[1] : ed_position[1],
        st_positon[0] : ed_position[0],
    ] = image_np[
        st_positon[2] : ed_position[2],
        st_positon[1] : ed_position[1],
        st_positon[0] : ed_position[0],
    ]
    result_image = sitk.GetImageFromArray(empty_image)
    result_image.CopyInformation(image)
    return result_image


def convert_contour_to_mask(nodule_id, result, origin, shape, spacing, mask_path):
    nodule_json = list(filter(lambda x: x["id"] == nodule_id, result))[0]
    graphics_nodule_json = {
        "boxes": nodule_json["boxes"],
        "contour3D": nodule_json["contour3D"],
        "volume_origin": origin,
        "volume_shape": shape,
        "volume_spacing": spacing,
        "output_path": "",
        "mask_output_path": mask_path,
    }
    # 　调用 GenerateLungVTP
    ClientGraphicsEngine.GenerateLungVTP(json.dumps(graphics_nodule_json))


def cal_custom_radiomics(
    image: sitk.Image,
    mask: sitk.Image,
    label: int = 1,
):
    t0 = time.time()
    bb, correctedMask = imageoperations.checkMask(image, mask)
    image, mask = imageoperations.cropToTumorMask(image, mask, bb, padDistance=3)
    extractor = featureextractor.RadiomicsFeatureExtractor(PARAMS)
    logger.info(f"Extraction parameters:\n\t, {extractor.settings}")
    result = extractor.execute(image, mask, label=label)

    logger.warning(len(result))
    # 过滤只返回array 结果数值
    result = {
        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()
    }

    logger.warning(f"Cal radiamics cost: {time.time() - t0}")
    return result


def padding_mask(image, mask):
    """merge imageA into a empty imageB"""
    # empty_image = sitk.Image(image.GetSize(), mask.GetPixelID())
    empty_image_np = np.zeros_like(sitk.GetArrayFromImage(image)).astype("uint8")

    mask_origin = mask.GetOrigin()
    start_position = image.TransformPhysicalPointToIndex(mask_origin)
    logger.info(f"start_position: {start_position}")

    end_x = mask_origin[0] + mask.GetSpacing()[0] * mask.GetSize()[0]
    end_y = mask_origin[1] + mask.GetSpacing()[1] * mask.GetSize()[1]
    end_z = mask_origin[2] + mask.GetSpacing()[2] * mask.GetSize()[2]
    end_position = image.TransformPhysicalPointToIndex((end_x, end_y, end_z))
    logger.info(f"end_position: {end_position}")

    mask_np = sitk.GetArrayFromImage(mask)
    empty_image_np[
        start_position[2] : end_position[2],
        start_position[1] : end_position[1],
        start_position[0] : end_position[0],
    ] = mask_np
    empty_image_np[empty_image_np == 255] = 1
    empty_image = sitk.GetImageFromArray(empty_image_np)
    empty_image.CopyInformation(image)

    logger.warning(f"unique: {np.unique(mask_np)}")
    logger.warning(f"mask {mask_np.shape}")
    return empty_image
