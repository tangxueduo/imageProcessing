import numpy as np

class FaiData:
    def __init__(
        self,
        plaqueThreshold: list,
        vesselSegmentName: str,
        plaqueId: int,
        series_iuid: str,
    ):
        self.plaqueThreshold = plaqueThreshold
        self.vesselSegmentName = vesselSegmentName
        self.plaqueId = plaqueId

    def _init_dada(self):
        # TODO: sitk 读取原图
        self.dcm_sitk_img = read_img()
        self.ct_heart_json = get_predict_result(self.series_iuid)

    def _get_plaque_histogram_data(
        self, plaqueThreshold: list, vesselSegmentName: str, plaqueId: int
    ) -> np.ndarray:
        """获取斑块分析直方图计算结果
        @plaqueThreshold:
        """
        res = {}
        # TODO:  根据 taskID 获取分割 mask
        plqvessel_seg = None
        cta_plaque_mask = None

        # 获取斑块分割 mask
        plauqes_split_result = plauqe_split(
            dicom=self.dcm_sitk_img,
            cta_plaque_mask=cta_plaque_mask,
            plqvessel_seg=plqvessel_seg,
            cta_json=self.ct_heart_json,
        )
        # 获取详细信息计算结果
        histogram_info = histogram_info_calc(
            dicom=self.dcm_sitk_img,
            cta_json=self.ct_heart_json,
            plauqes_split_result=plauqes_split_result,
            hu_thresholds=self.plaqueThreshold,
        )
        res["plaque_analysis_info"] = histogram_info
        # 获取直方图坐标轴数据
        histogram_axis_data = get_histogram_axis_data(
            plauqes_split_result=plauqes_split_result,
            vesselSegmentName=self.vesselSegmentName,
            plaqueId=self.plaqueId,
        )
        res["plaque_histogram_data"] = histogram_axis_data
        return res


def get_histogram_axis_data(
    plauqes_split_result: dict, vesselSegmentName: str, plaqueId: int
):
    """计算斑块内CT值"""
    res = {}
    h_axis = []
    v_axis = []
    res["h_axis"] = h_axis
    res["v_axis"] = v_axis
    return res
