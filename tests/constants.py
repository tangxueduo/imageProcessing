from typing import Dict


class TAGMethod:
    CENTERLINE = "centerline"  # 基于中心线
    LUMEN = "lumen"  # 基于官腔
    REMOVE_PLAQUE = "remove_plaques"  # 去斑块


TAGMETHODS = [TAGMethod.CENTERLINE, TAGMethod.LUMEN, TAGMethod.REMOVE_PLAQUE]
RADIUS_MAP: Dict[str, int] = {
    TAGMethod.CENTERLINE: 1,
    TAGMethod.LUMEN: 0,
    TAGMethod.REMOVE_PLAQUE: 350,
}
TAG_VESSELS = ["vessel2", "vessel5", "vessel9"]
CPR_WIDTH = 80
LUMEN_WIDTH = 150
LUMEN_WIDTH_HEAD = 50
# 生成 lumen 时需要加宽的血管
WIDER_VESSELS: Dict[str, str] = {
    "vessel1": "L-ICA",
    "vessel2": "R-ICA",
    "vessel3": "L-CCA",
    "vessel4": "R-CCA",
    "vessel7": "L-SA",
    "vessel8": "R-SA",
    "vessel9": "BCT",
    "vessel21": "BA",
    "vessel22": "L-VA",
    "vessel23": "R-VA",
}

# 不同的后处理参数
CPRLibProps = {
    "_default": {
        "convert_mask_to_contour": True,
        "return_multiple_contours": False,
        "contour_draw_centerline_radius": None,
        "contour_smooth_params": {"method": "fft", "cutoff": 0.5},
    },
    "vessel_seg": {
        "convert_mask_to_contour": True,
        "return_multiple_contours": False,
        "contour_draw_centerline_radius": None,
        "contour_smooth_params": {"method": "fft", "cutoff": 0.5},
    },
    "plq_seg": {
        "convert_mask_to_contour": False,
        "return_multiple_contours": True,
        "contour_draw_centerline_radius": None,
    },
}

IntracranialVR = "IntracranialVR"  # 颅内轴位VR

BONE_INTRACRANIAL_LABELS = [
    3,
    4,
    5,
    6,
    7,
    10,
    11,
    12,
    13,
    14,
    30,
    31,
    32,
    35,
    36,
    37,
    40,
    41,
    42,
    45,
    46,
    47,
    50,
    51,
    52,
    54,
    55,
    56,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    80,
]
RenderTypeShowLabels: Dict[str, Dict] = {
    # 颅内轴位 VR
    IntracranialVR: {
        "labels": BONE_INTRACRANIAL_LABELS,
        "label_seg": [],
    },
}
