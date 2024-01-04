import numpy as np
import SimpleITK as sitk
import time
import logger

def get_pvat_hits_radiomics(
    vessel_mask: sitk.Image,
    sitk_image: sitk.Image,
    fai_pvat: float,
    roimask_path: str,
    roimask_3d_path: str,
    fat_thres: list,
    water_thres: list,
    cacs_thres: list,
    fiber_thres: list,
    label_values: list,
    roi_label=255,
):
    t0 = time.time()
    hu_volume_np = sitk.GetArrayFromImage(sitk_image)
    roi_1diameter_img = sitk.ReadImage(roimask_path)
    roi_1diameter_img.CopyInformation(vessel_mask)
    roi_3diameter_img = sitk.ReadImage(roimask_3d_path)
    # PVAT_Radiomics = PVATRadiomics(sitk_image, roi_1diameter_img, roi_label)
    # pvat_hits_radiomics = PVAT_Radiomics.get_features(RadiomicsFeatureName.FIRST_ORDER)
    # pvat_hits_radiomics = {k: round(v, 2) for k, v in pvat_hits_radiomics.items()}
    pvat_hits_radiomics = {}
    # 直方图右侧展示组学特征中的14个
    [
        pvat_hits_radiomics.pop(k)
        for k in [
            "10Percentile",
            "90Percentile",
            "TotalEnergy",
            "RobustMeanAbsoluteDeviation",
            "RootMeanSquared",
        ]
    ]
    roi_3diameter_np = sitk.GetArrayFromImage(roi_3diameter_img)
    roi_1diameter_np = sitk.GetArrayFromImage(roi_1diameter_img)

    # 获取非血管周围３ｄ脂肪均值
    diff_set_mask = roi_3diameter_np.astype("bool") * roi_1diameter_np.astype("bool")

    # 血管周围非脂肪均值
    fai_nonpvat = round(
        np.nanmean(
            hu_volume_np[
                (hu_volume_np > fat_thres[0])
                & (hu_volume_np < fat_thres[1])
                & (diff_set_mask)
            ]
        ),
        2,
    )
    vpci = round((100 * (fai_pvat - fai_nonpvat) / abs(fai_pvat)), 2)
    sub_vessel_mask = np.isin(sitk.GetArrayFromImage(vessel_mask), label_values)

    total_vessel_pixel = np.count_nonzero(hu_volume_np * sub_vessel_mask)
    # 血管壁内纤维斑块/血管总体积
    fiber_pixel = np.count_nonzero(
        hu_volume_np
        * (
            (sub_vessel_mask)
            & (hu_volume_np > fiber_thres[0])
            & (hu_volume_np < fiber_thres[1])
        )
    )
    fpi = (
        round(fiber_pixel / total_vessel_pixel, 2)
        if total_vessel_pixel != 0
        else np.nan
    )
    # 血管周围水衰减平均值
    pvwi = round(
        np.nanmean(
            hu_volume_np[
                (sub_vessel_mask)
                & (hu_volume_np > water_thres[0])
                & (hu_volume_np < water_thres[1])
            ]
        ),
        2,
    )

    # 血管壁内钙化/血管总体积
    cacs_pixel = np.count_nonzero(
        hu_volume_np
        * (
            (sub_vessel_mask)
            & (hu_volume_np > cacs_thres[0])
            & (hu_volume_np < cacs_thres[1])
        )
    )

    ci = (
        round(cacs_pixel / total_vessel_pixel, 2) if total_vessel_pixel != 0 else np.nan
    )
    pvat_hits_radiomics.update(
        {
            "VPCI": vpci.tolist() if not np.isnan(vpci) else None,
            "FAI-nonPVAT": fai_nonpvat.tolist() if not np.isnan(fai_nonpvat) else None,
            "FPI": fpi if not np.isnan(fpi) else None,
            "PVWI": pvwi if not np.isnan(pvwi) else None,
            "CI": ci if not np.isnan(ci) else None,
        }
    )
    logger.warning(f"Calculate radiomics cost: {time.time() - t0}")
    return pvat_hits_radiomics
