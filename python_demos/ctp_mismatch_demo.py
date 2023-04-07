"""
根据模型提供的数据, 计算体积
"""
import json
import os
import time
from collections import defaultdict

import numpy as np
import SimpleITK as sitk
from head_ctp_mismatch import HeadCtpMismatchPredictor
from loguru import logger
from skimage import measure

from python_demos.mismatch_utils import (BRAIN_AREA_MAP, LEFT_BRAIN_LABEL,
                                         RIGHT_BRAIN_LABEL, ddict2dict,
                                         gray2rgb_array, judge_average_all)

# from multiprocessing import Process, Pool
COLOR_MAP = {
    "#FFFF00": [0, 255, 255],
    "#FF5959": [89, 89, 255],
    "#26CF70": [112, 207, 38],
    "#3F87F5": [245, 135, 63],
}
# 染色先后顺序
Tmax_thresholds = [
    {"color": "#26CF70", "threshold": 10},
    {"color": "#FFFF00", "threshold": 8},
    {"color": "#FF5959", "threshold": 6},
]
rCBF_thresholds = [
    {"color": "#26CF70", "threshold": 40},
    {"color": "#FFFF00", "threshold": 30},
    {"color": "#FF5959", "threshold": 20},
]
rCBV_thresholds = [
    {"color": "#26CF70", "threshold": 45},
    {"color": "#FFFF00", "threshold": 40},
    {"color": "#FF5959", "threshold": 35},
]
Mismatch_thresholds = [
    {"color": "#FFFF00", "threshold": 30},
    {"color": "#FF5959", "threshold": 6},
]


class Mismatch:
    def __init__(
        self,
        tmip_no_skull_itk,
        brain_area_img,
        tmax_img,
        cbf_img,
        cbv_img,
        mtt_img,
        ttp_img,
        centerline,
        spacing,
        origin,
    ) -> None:

        self.ctp_mis_predictor = HeadCtpMismatchPredictor(
            tmip_no_skull_itk,  # 去颅骨的Image
            brain_area_img,  # 脑分区Image
            centerline,  # 中线字典类型
            spacing,  # spacing：[x,y,z]
            origin,  # LPI的origin
        )
        self.tmip_no_skull_itk = tmip_no_skull_itk
        self.tmax_img = tmax_img
        self.cbf_img = cbf_img
        self.cbv_img = cbv_img
        self.mtt_img = mtt_img
        self.ttp_img = ttp_img
        self.brain_area_img = brain_area_img
        self.tmip_no_skull_arr = sitk.GetArrayFromImage(self.tmip_no_skull_itk)
        self.brain_area_array = sitk.GetArrayFromImage(self.brain_area_img)
        self.cbv_array = sitk.GetArrayFromImage(self.cbv_img)
        self.cbf_array = sitk.GetArrayFromImage(self.cbf_img)
        self.tmax_array = sitk.GetArrayFromImage(self.tmax_img)
        self.mtt_array = sitk.GetArrayFromImage(self.mtt_img)
        self.ttp_array = sitk.GetArrayFromImage(self.ttp_img)
        self.shape = self.tmip_no_skull_arr.shape[0]
        self.img_result = {}  # type: ignore
        self.voxel = spacing[0] * spacing[1] * spacing[2] / 1000

    def gen_rcbf(self):
        print("cbf", "process id:", os.getpid())
        t1 = time.time()
        (
            self.low_cbf_array,
            self.mid_cbf_array,
            self.high_cbf_array,
        ) = self.ctp_mis_predictor.get_cbf_mask(cbf_img=self.cbf_img)
        print(f"rcbf model cost is: {time.time() - t1}")
        self.stain(
            rCBF_thresholds,
            "rCBF_Mismatch_",
            self.low_cbf_array,
            self.mid_cbf_array,
            self.high_cbf_array,
        )

    def gen_rcbv(self):
        print("cbv", "process id:", os.getpid())
        t2 = time.time()
        (
            self.low_cbv_array,
            self.mid_cbv_array,
            self.high_cbv_array,
        ) = self.ctp_mis_predictor.get_cbv_mask(cbv_img=self.cbv_img)
        print(f"rcbv model cost is: {time.time() - t2}")
        self.stain(
            rCBV_thresholds,
            "rCBV_Mismatch_",
            self.low_cbv_array,
            self.mid_cbv_array,
            self.high_cbv_array,
        )

    def gen_tmax(self):
        print("tmax", "process id:", os.getpid())
        t3 = time.time()
        (
            self.low_tmax_array,
            self.mid_tmax_array,
            self.high_tmax_array,
        ) = self.ctp_mis_predictor.get_tmax_mask(tmax_img=self.tmax_img)
        print(f"rcbf model cost is: {time.time() - t3}")
        self.stain(
            Tmax_thresholds,
            "",
            self.low_tmax_array,
            self.mid_tmax_array,
            self.high_tmax_array,
        )

    def gen_mismatch(self):
        """出　mismatch 图"""
        for dcm_idx in range(self.shape):
            cbf_2d_arr = self.low_cbf_array[dcm_idx, :, :]
            tmax_2d_arr = self.low_tmax_array[dcm_idx, :, :]
            # 染色
            tmip_3d_arr = gray2rgb_array(
                self.tmip_no_skull_arr[dcm_idx, :, :], ww=100, wl=50, is_colormap=False
            ).copy()
            tmip_3d_arr[cbf_2d_arr == 1] = COLOR_MAP[Mismatch_thresholds[0]["color"]]
            tmip_3d_arr[tmax_2d_arr == 1] = COLOR_MAP[Mismatch_thresholds[1]["color"]]
            self.img_result["Mismatch_" + str(dcm_idx)] = tmip_3d_arr

    def stain(
        self, thresholds, pre_name, algo_low_array, alo_mid_array, algo_high_aray
    ):
        for dcm_idx in range(self.shape):
            _2d_low_arr = algo_low_array[dcm_idx, :, :]
            _2d_mid_arr = alo_mid_array[dcm_idx, :, :]
            _2d_high_arr = algo_high_aray[dcm_idx, :, :]
            # 染色
            tmip_3d_arr = gray2rgb_array(
                self.tmip_no_skull_arr[dcm_idx, :, :], ww=100, wl=50, is_colormap=False
            ).copy()
            tmip_3d_arr[np.where(_2d_low_arr == 1)] = COLOR_MAP[thresholds[0]["color"]]
            tmip_3d_arr[np.where(_2d_mid_arr == 1)] = COLOR_MAP[thresholds[1]["color"]]
            tmip_3d_arr[np.where(_2d_high_arr == 1)] = COLOR_MAP[thresholds[2]["color"]]
            self.img_result[pre_name + str(dcm_idx)] = tmip_3d_arr

    def get_lesions(self, threshold=100):
        """
        获取病灶列表, 目前由tmax>6s 作为连通域交集
        """
        dd = lambda: defaultdict(dd)
        lesions = dd()
        ctp_lesion = dd()
        (
            rcbf_sum_high,
            rcbf_sum_mid,
            rcbf_sum_low,
            rcbv_sum_high,
            rcbv_sum_mid,
            rcbv_sum_low,
            tmax_sum_high,
            tmax_sum_mid,
            tmax_sum_low,
            mis_sum_core,
            mis_sum_low,
            mis_sum_rcbf,
            mis_sum_tmax,
        ) = [0] * 13
        # 暂时用　1　连通
        connectivity = 1
        labels_out, N = measure.label(
            self.low_tmax_array > 0, return_num=True, connectivity=connectivity
        )
        print(f"cc out: {labels_out.shape}, N: {N}")
        if N == 0:
            print(f"Not found Connected domains")
            return None
        slice_ids = {}  # type:ignore
        properties = measure.regionprops(labels_out)
        for prop in properties:
            if prop.area > threshold:
                points = np.argwhere(labels_out == prop.label)
                zmin = np.min(points[:, 0])
                zmax = np.max(points[:, 0])
                slice_ids[prop.label] = [int(zmin), int(zmax)]
        lesions_label = [lesion_k for lesion_k in slice_ids.keys()]
        rcbf_lesions, rcbv_lesions, mis_lesions, tmax_lesions, lesion_brains = (
            [],
            [],
            [],
            [],
            [],
        )
        for lesion in lesions_label:
            brain_areas = []
            (
                mis_single_lesion,
                rcbf_single_lesion,
                rcbv_single_lesion,
                tmax_single_lesion,
            ) = [{} for _ in range(4)]
            brain_list = list(np.unique(self.brain_area_array[labels_out == lesion]))
            try:
                brain_list.remove(0)
            except Exception as e:
                logger.warning(f"0 not in list: {e}")
            # 过滤病灶没有经过任何　label 的情况
            if not brain_list:
                continue
            # 病灶所经过的脑区
            lesion_brains += brain_list
            for area_label in brain_list:
                brain_areas.append(BRAIN_AREA_MAP[area_label]["origin"])
                mis_single_lesion["section"] = brain_areas
                mis_single_lesion["slice_id"] = slice_ids[lesion]
                mis_single_lesion["info"] = {}
            res1 = np.count_nonzero(self.low_tmax_array[labels_out == lesion])
            res2 = np.count_nonzero(self.low_cbf_array[labels_out == lesion])
            print(f"res1: {res1}, res2 : {res2}, voxel: {self.voxel}")

            # TODO: mismatch 视图按照阈值配置
            res_mis_rcbf = res2
            res_mis_tmax = res1
            # Mismatch leisons info
            mis_single_lesion["info"][
                f"Tmax > {Mismatch_thresholds[1]['threshold']}s"
            ] = round(res_mis_tmax * self.voxel, 1)
            mis_sum_tmax += res_mis_tmax * self.voxel
            mis_single_lesion["info"][
                f"rCBF < {Mismatch_thresholds[0]['threshold']}%"
            ] = round(res_mis_rcbf * self.voxel, 1)
            mis_sum_rcbf += res_mis_rcbf * self.voxel

            mis_sum_low += res1 * self.voxel
            mis_sum_core += res2 * self.voxel
            mis_single_lesion["info"]["MismatchVolume"] = round(
                (mis_sum_tmax - mis_sum_rcbf) * self.voxel, 1
            )
            mis_single_lesion["info"]["rate"] = (
                "-" if (round(res2, 1) == 0) else round(res1 / res2, 1)
            )

            # rCBF lesions info
            res_cbf_high = np.count_nonzero(self.high_cbf_array[labels_out == lesion])
            res_cbf_mid = np.count_nonzero(self.mid_cbf_array[labels_out == lesion])
            res_cbf_low = np.count_nonzero(self.low_cbf_array[labels_out == lesion])
            rcbf_single_lesion["section"] = brain_areas
            rcbf_single_lesion["slice_id"] = slice_ids[lesion]
            rcbf_single_lesion["info"] = {}
            rcbf_single_lesion["info"][
                f"rCBF < {rCBF_thresholds[0]['threshold']}%"
            ] = round(res_cbf_high * self.voxel, 1)
            rcbf_sum_high += res_cbf_high * self.voxel
            rcbf_single_lesion["info"][
                f"rCBF < {rCBF_thresholds[1]['threshold']}%"
            ] = round(res_cbf_mid * self.voxel, 1)
            rcbf_sum_mid += res_cbf_mid * self.voxel
            rcbf_single_lesion["info"][
                f"rCBF < {rCBF_thresholds[2]['threshold']}%"
            ] = round(res_cbf_low * self.voxel, 1)
            rcbf_sum_low += res_cbf_low * self.voxel

            # rCBV lesions info
            res_cbv_high = np.count_nonzero(self.high_cbv_array[labels_out == lesion])
            res_cbv_mid = np.count_nonzero(self.mid_cbv_array[labels_out == lesion])
            res_cbv_low = np.count_nonzero(self.low_cbv_array[labels_out == lesion])
            rcbv_single_lesion["section"] = brain_areas
            rcbv_single_lesion["slice_id"] = slice_ids[lesion]
            rcbv_single_lesion["info"] = {}
            rcbv_single_lesion["info"][
                f"rCBV < {rCBV_thresholds[0]['threshold']}%"
            ] = round(res_cbv_high * self.voxel, 1)
            rcbv_sum_high += res_cbv_high * self.voxel
            rcbv_single_lesion["info"][
                f"rCBV < {rCBV_thresholds[1]['threshold']}%"
            ] = round(res_cbv_mid * self.voxel, 1)
            rcbv_sum_mid += res_cbv_mid * self.voxel
            rcbv_single_lesion["info"][
                f"rCBV < {rCBV_thresholds[2]['threshold']}%"
            ] = round(res_cbv_low * self.voxel, 1)
            rcbv_sum_low += res_cbv_low * self.voxel

            res_tmax_high = np.count_nonzero(self.high_tmax_array[labels_out == lesion])
            res_tmax_mid = np.count_nonzero(self.mid_tmax_array[labels_out == lesion])
            res_tmax_low = np.count_nonzero(self.low_tmax_array[labels_out == lesion])
            # Tmax
            tmax_single_lesion["section"] = brain_areas
            tmax_single_lesion["slice_id"] = slice_ids[lesion]
            tmax_single_lesion["info"] = {}
            tmax_single_lesion["info"][
                f"Tmax > {Tmax_thresholds[2]['threshold']}s"
            ] = round(res_tmax_high * self.voxel, 1)
            tmax_sum_high += res_tmax_high * self.voxel

            tmax_single_lesion["info"][
                f"Tmax > {Tmax_thresholds[1]['threshold']}s"
            ] = round(res_tmax_mid * self.voxel, 1)
            tmax_sum_mid += res_tmax_mid * self.voxel

            tmax_single_lesion["info"][
                f"Tmax > {Tmax_thresholds[0]['threshold']}s"
            ] = round(res_tmax_low * self.voxel, 1)
            tmax_sum_low += res_tmax_low * self.voxel

            rcbf_lesions.append(rcbf_single_lesion)
            rcbv_lesions.append(rcbv_single_lesion)
            tmax_lesions.append(tmax_single_lesion)
            mis_lesions.append(mis_single_lesion)
        if lesion_brains:
            lesions["Mismatch"]["rCBF_view"]["lesions"] = rcbf_lesions
            # rcbf summary
            lesions["Mismatch"]["rCBF_view"]["Mismatch_abnormality"][
                f"rCBF < {rCBF_thresholds[0].get('threshold')}%"
            ] = round(rcbf_sum_high, 1)
            lesions["Mismatch"]["rCBF_view"]["Mismatch_abnormality"][
                f"rCBF < {rCBF_thresholds[1].get('threshold')}%"
            ] = round(rcbf_sum_mid, 1)
            lesions["Mismatch"]["rCBF_view"]["Mismatch_abnormality"][
                f"rCBF < {rCBF_thresholds[2].get('threshold')}%"
            ] = round(rcbf_sum_low, 1)

            lesions["Mismatch"]["rCBV_view"]["lesions"] = rcbv_lesions
            # rcbv summary
            lesions["Mismatch"]["rCBV_view"]["Mismatch_abnormality"][
                f"rCBV < {rCBV_thresholds[0].get('threshold')}%"
            ] = round(rcbv_sum_high, 1)
            lesions["Mismatch"]["rCBV_view"]["Mismatch_abnormality"][
                f"rCBV < {rCBV_thresholds[1].get('threshold')}%"
            ] = round(rcbv_sum_mid, 1)
            lesions["Mismatch"]["rCBV_view"]["Mismatch_abnormality"][
                f"rCBV < {rCBV_thresholds[2].get('threshold')}%"
            ] = round(rcbv_sum_low, 1)

            lesions["Mismatch"]["Tmax_view"]["lesions"] = tmax_lesions
            # tmax summary
            lesions["Mismatch"]["Tmax_view"]["Mismatch_abnormality"][
                f"Tmax > {Tmax_thresholds[2].get('threshold')}s"
            ] = round(tmax_sum_high, 1)
            lesions["Mismatch"]["Tmax_view"]["Mismatch_abnormality"][
                f"Tmax > {Tmax_thresholds[1].get('threshold')}s"
            ] = round(tmax_sum_mid, 1)
            lesions["Mismatch"]["Tmax_view"]["Mismatch_abnormality"][
                f"Tmax > {Tmax_thresholds[0].get('threshold')}s"
            ] = round(tmax_sum_low, 1)
            lesions["Mismatch"]["Mismatch_view"]["lesions"] = mis_lesions
            # mismatch summary
            lesions["Mismatch"]["Mismatch_view"]["Mismatch_abnormality"][
                f"rCBF < {Mismatch_thresholds[0].get('threshold')}%"
            ] = round(mis_sum_rcbf, 1)

            lesions["Mismatch"]["Mismatch_view"]["Mismatch_abnormality"][
                f"Tmax > {Mismatch_thresholds[1].get('threshold')}s"
            ] = round(mis_sum_tmax, 1)

            lesions["Mismatch"]["Mismatch_view"]["Mismatch_abnormality"][
                "MismatchVolume"
            ] = round(round(mis_sum_tmax, 1) - round(mis_sum_rcbf, 1), 1)

            # 应对分母为 0
            lesions["Mismatch"]["Mismatch_view"]["Mismatch_abnormality"]["rate"] = (
                "-"
                if round(mis_sum_rcbf, 1) == 0
                else round(mis_sum_tmax / mis_sum_rcbf, 1)
            )
            lesion_brains = set(lesion_brains)
        report = self._get_brain_area_paras_report(lesion_brains)
        ctp_lesion["lesions"].update(lesions)

        ctp_lesion["lesions"]["CT_perfusion"]["report"] = report
        ctp_lesion["lesions"]["CT_perfusion"]["Mismatch_abnormality"][
            "infractionVolume"
        ] = round(mis_sum_core, 1)

        ctp_lesion["lesions"]["CT_perfusion"]["Mismatch_abnormality"][
            "lowPerfusionVolume"
        ] = round(mis_sum_low, 1)

        ctp_lesion["lesions"]["CT_perfusion"]["Mismatch_abnormality"][
            "penumbraVolume"
        ] = round(mis_sum_low - mis_sum_core, 1)
        ctp_lesion["lesions"]["CT_perfusion"]["Mismatch_abnormality"]["rate"] = (
            "-"
            if (round(mis_sum_core, 1) == 0)
            else round(mis_sum_low / mis_sum_core, 1)
        )
        return ctp_lesion

    def _get_brain_area_paras_report(self, lesion_brains):
        """获取病灶所经过的脑区的各项 CTP 参数信息"""
        report = {}
        print(f"****lesion_brains: {lesion_brains}")
        for brain_label in lesion_brains:
            # 左右脑区都有灌注，计算左脑区，对侧取反
            if brain_label == 15 or brain_label == 16:
                label_origin = brain_label
                origin_area = BRAIN_AREA_MAP[label_origin]["origin"]
                relative_area = None
            else:
                # label_origin: 1, 12,...
                label_origin = brain_label
                label_relative = BRAIN_AREA_MAP[label_origin]["relative"]
                # origin_area: L-Parietal lobe...
                origin_area = BRAIN_AREA_MAP[label_origin]["origin"]  # 脑区
                relative_area = BRAIN_AREA_MAP[label_relative]["origin"]  # 对侧脑区

            if origin_area in report or relative_area in report or label_origin == 16:
                continue
            # TODO: 获取中线左右侧的脑干和胼胝体

            if label_origin == 15:
                continue
                # TODO: 暂时不做处理后面优化
                # report = self._get_alone_brain_tendency(
                #     report,
                #     label_origin,
                #     origin_area,
                # )
            else:
                report = self._get_double_sides_tendency(
                    report, label_origin, label_relative, origin_area, relative_area
                )
        return report

    # TODO: 优化以下代码
    def _get_double_sides_tendency(
        self,
        report,
        label_origin,
        label_relative,
        origin_area,
        relative_area,
    ):
        left_area_cbv, right_area_cbv = (
            self.cbv_array[self.brain_area_array == label_origin].mean(),
            self.cbv_array[self.brain_area_array == label_relative].mean(),
        )
        left_area_cbf, right_area_cbf = (
            self.cbf_array[self.brain_area_array == label_origin].mean(),
            self.cbf_array[self.brain_area_array == label_relative].mean(),
        )
        left_area_mtt, right_area_mtt = (
            self.mtt_array[self.brain_area_array == label_origin].mean(),
            self.mtt_array[self.brain_area_array == label_relative].mean(),
        )
        left_area_ttp, right_area_ttp = (
            self.ttp_array[self.brain_area_array == label_origin].mean(),
            self.ttp_array[self.brain_area_array == label_relative].mean(),
        )
        left_area_tmax, right_area_tmax = (
            self.tmax_array[self.brain_area_array == label_origin].mean(),
            self.tmax_array[self.brain_area_array == label_relative].mean(),
        )
        # 左右脑区均有病灶
        if label_origin in LEFT_BRAIN_LABEL and label_relative in RIGHT_BRAIN_LABEL:
            print(1111111111111111111111)

            # 左脑区低灌注
            origin_low_perfusion = round(
                np.count_nonzero(
                    self.low_cbf_array[self.brain_area_array == label_origin]
                )
                * self.voxel,
                1,
            )
            # 右脑区低灌注
            relative_low_perfusion = round(
                np.count_nonzero(
                    self.low_tmax_array[self.brain_area_array == label_relative]
                )
                * self.voxel,
                1,
            )
            # 左脑区梗死区
            origin_core_penumbra = round(
                np.count_nonzero(
                    self.low_cbf_array[self.brain_area_array == label_origin]
                )
                * self.voxel,
                1,
            )

            # 右脑区梗死区
            relative_core_penumbra = round(
                np.count_nonzero(
                    self.low_cbf_array[self.brain_area_array == label_relative]
                )
                * self.voxel,
                1,
            )
            # cbf 趋势
            origin_cbf, relative_cbf = self.get_brain_area_tendency(
                left_area_cbf, right_area_cbf, both_area=True, is_blood=True
            )
            # cbv 趋势
            origin_cbv, relative_cbv = self.get_brain_area_tendency(
                left_area_cbv, right_area_cbv, both_area=True, is_blood=True
            )
            # mtt 趋势
            origin_mtt, relative_mtt = self.get_brain_area_tendency(
                left_area_mtt, right_area_mtt, both_area=True, is_time=True
            )
            # ttp 趋势
            origin_ttp, relative_ttp = self.get_brain_area_tendency(
                left_area_ttp, right_area_ttp, both_area=True, is_time=True
            )
            # tmax 趋势
            origin_tmax, relative_tmax = self.get_brain_area_tendency(
                left_area_tmax, right_area_tmax, both_area=True, is_time=True
            )
            report = judge_average_all(
                report,
                origin_area,
                origin_cbf,
                origin_cbv,
                origin_mtt,
                origin_ttp,
                origin_tmax,
                origin_core_penumbra,
                origin_low_perfusion,
            )
            report = judge_average_all(
                report,
                relative_area,
                relative_cbf,
                origin_cbv,
                relative_mtt,
                relative_ttp,
                relative_tmax,
                relative_core_penumbra,
                relative_low_perfusion,
            )

        # 病灶只在左脑区
        elif label_origin in LEFT_BRAIN_LABEL:
            print(22222222222222)

            # 计算 低灌注和梗死区
            origin_core_penumbra = round(
                np.count_nonzero(
                    self.low_cbf_array[self.brain_area_array == label_origin]
                )
                * self.voxel,
                1,
            )

            origin_low_perfusion = round(
                np.count_nonzero(
                    self.low_tmax_array[self.brain_area_array == label_origin]
                )
                * self.voxel,
                1,
            )
            origin_cbf, _ = self.get_brain_area_tendency(
                left_area_cbf, right_area_cbf, is_blood=True
            )
            #  cbv
            origin_cbv, _ = self.get_brain_area_tendency(
                left_area_cbv, right_area_cbv, is_blood=True
            )
            # mtt
            origin_mtt, _ = self.get_brain_area_tendency(
                left_area_mtt, right_area_mtt, is_time=True
            )
            # ttp
            origin_ttp, _ = self.get_brain_area_tendency(
                left_area_ttp, right_area_ttp, is_time=True
            )
            # tmax
            origin_tmax, _ = self.get_brain_area_tendency(
                left_area_tmax, right_area_tmax, is_time=True
            )
            report = judge_average_all(
                report,
                origin_area,
                origin_cbf,
                origin_cbv,
                origin_mtt,
                origin_ttp,
                origin_tmax,
                origin_core_penumbra,
                origin_low_perfusion,
            )
        elif label_origin in RIGHT_BRAIN_LABEL:
            print(333333333333333)
            print(label_relative)
            # 计算 低灌注体积和梗死区体积
            relative_core_penumbra = round(
                np.count_nonzero(
                    self.low_cbf_array[self.brain_area_array == label_relative]
                )
                * self.voxel,
                1,
            )

            relative_low_perfusion = round(
                np.count_nonzero(
                    self.low_tmax_array[self.brain_area_array == label_relative]
                )
                * self.voxel,
                1,
            )
            print(
                f"relative_core_penumbra: {relative_core_penumbra}, relative_low_perfusion: {relative_low_perfusion}"
            )
            relative_cbf, _ = self.get_brain_area_tendency(
                right_area_cbf, left_area_cbf, is_blood=True
            )

            #  cbv
            relative_cbv, _ = self.get_brain_area_tendency(
                right_area_cbv, left_area_cbv, is_blood=True
            )
            # mtt
            relative_mtt, _ = self.get_brain_area_tendency(
                right_area_mtt, left_area_mtt, is_time=True
            )
            # ttp
            relative_ttp, _ = self.get_brain_area_tendency(
                right_area_ttp, left_area_ttp, is_time=True
            )
            # tmax
            relative_tmax, _ = self.get_brain_area_tendency(
                right_area_tmax, left_area_tmax, is_time=True
            )
            report = judge_average_all(
                report,
                relative_area,
                relative_cbf,
                relative_cbv,
                relative_mtt,
                relative_ttp,
                relative_tmax,
                relative_core_penumbra,
                relative_low_perfusion,
            )
        return report

    def get_brain_area_tendency(
        self,
        label1,
        label2,
        both_area=False,
        is_time=False,
        is_blood=False,
        threshold=0.2,
    ):
        """获取脑区异常趋势
        Args:
            both_area: 是否涉及双侧脑区
            is_time: 是否为时间维度参数
            threshold: 左右持平的阈值
        Return: tendency
        """
        tmp = min(label1, label2) if is_blood else max(label1, label2)
        gap = abs(label1 - label2) / tmp
        if not both_area:
            if gap < threshold:
                tendency = "average"
            elif label1 > label2:
                tendency = "average" if is_blood else "above"
            else:
                tendency = "average" if is_time else "below"
            return tendency, ""
        else:
            if gap < threshold:
                tendency1, tendency2 = "average", "average"
            elif label1 > label2:
                if is_time:
                    tendency1, tendency2 = "above", "average"
                if is_blood:
                    tendency1, tendency2 = "average", "below"
            else:
                if is_time:
                    tendency1, tendency2 = (
                        "average",
                        "above",
                    )
                if is_blood:
                    tendency1, tendency2 = (
                        "below",
                        "average",
                    )
            return tendency1, tendency2

    def gen_mismatch_all(self):
        t0 = time.time()
        # TODO: 多进程
        self.gen_rcbf()
        self.gen_rcbv()
        self.gen_tmax()
        self.gen_mismatch()
        print(f"total cost: {time.time() - t0}")
        # 计算病灶，tmax>6s || rmtt>145%
        ctp_lesion = ddict2dict(self.get_lesions())
        return self.img_result, ctp_lesion


def read_nii(nii_path):
    return sitk.ReadImage(nii_path)


if __name__ == "__main__":
    ctp_dir = "/media/tx-deepocean/Data/DICOMS/demos/ctp_2/CN010023-2206081005-9448-301"
    tmip_no_skull_path = os.path.join(ctp_dir, "TMIP_NO_SKULL.nii.gz")
    tmip_no_skull_itk = sitk.ReadImage(tmip_no_skull_path)
    spacing, origin = tmip_no_skull_itk.GetSpacing(), tmip_no_skull_itk.GetOrigin()
    tmax_img = read_nii(os.path.join(ctp_dir, "TMAX.nii.gz"))
    cbv_img = read_nii(os.path.join(ctp_dir, "CBV.nii.gz"))
    cbf_img = read_nii(os.path.join(ctp_dir, "CBF.nii.gz"))
    mtt_img = read_nii(os.path.join(ctp_dir, "MTT.nii.gz"))
    ttp_img = read_nii(os.path.join(ctp_dir, "TTP.nii.gz"))
    brain_area_path = os.path.join(ctp_dir, "brain_area_mask.nii.gz")

    # perfusion_res = requests.get("http://10.0.40.59:3333/series/1.3.46.670589.33.1.63790194166890436500002.4699278885504849661-0/predict/ct_cerebral_perfusion").json()
    with open(os.path.join(ctp_dir, "result.json")) as fp:
        perfusion_res = json.load(fp)

    mis = Mismatch(
        tmip_no_skull_itk,
        read_nii(brain_area_path),
        tmax_img,
        cbf_img,
        cbv_img,
        mtt_img,
        ttp_img,
        centerline=perfusion_res["centerline"],
        spacing=spacing,
        origin=origin,
    )
    print("main process id:", os.getpid())
    mismach_result, lesions_result = mis.gen_mismatch_all()
    res = {"dicoms": mismach_result, "lesions": lesions_result}
    print(json.dumps(lesions_result))
