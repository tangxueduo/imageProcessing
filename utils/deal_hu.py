import numpy as np


def gray2rgb_array_by_window(
    gray_array: np.ndarray, ww: float, wl: float, is_colormap: bool = False
):
    """根据窗宽窗位把hu 归一到0到255，灰度转rgb,实现较为简单，先归一[0,1],再映射到[0,255]

    Args:
        gray_array: 2d
        ww: WindowWidth, 窗宽
        wl: WindowCenter/WindowLevel, 窗位
        is_pseudo_colormap: 伪彩图时为True， 注：LUT table 为单通道，故这里需返回clip后的cv_8UC1

    Returns:
        is_pseudo_colormap 为true时， 返回单通道
        is_pseudo_colormap 为false时， 返三通道

    Warning:
        转dicom时，与OpenCV中的 r和b通道相反，开发者视需要进行更改
    """
    temp_array = gray_array
    window_width = ww
    window_level = wl
    true_max_pt = window_level + (window_width / 2)
    true_min_pt = window_level - (window_width / 2)

    scale = 255 / (true_max_pt - true_min_pt)
    temp_array = np.clip(temp_array, true_min_pt, true_max_pt)
    min_pt_array = np.ones((temp_array.shape[0], temp_array.shape[1])) * true_min_pt
    temp_array = (temp_array - min_pt_array) * scale

    if not is_colormap:
        rgb_array = np.repeat(temp_array[..., None], 3, axis=-1)
    else:
        rgb_array = temp_array
    return rgb_array
