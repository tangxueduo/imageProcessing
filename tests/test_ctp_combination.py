import base64
import math
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from utils.constants import (
    BGR_COLOR_MAP,
    CUSTOM_MAP,
    ROI_FORM_COLOR_MAP,
    ROI_ROWS_TITLE,
    SCREENSHOTS_COMBINATION_TYPES,
)
from utils.img_process import bgr_to_rgb, color_draw_text, gray2rgb_array

FONT = ImageFont.truetype("statics/SourceHanSansCN-Normal.ttf", 18)
roi_data = {
    "slice_idx": 15,
    "form_data": {
        "1-L": {
            "CBF": 17.406082887700535,
            "CBV": 1.9338235294117647,
            "MTT": 7.955882352941177,
            "TTP": 12.730949197860962,
            "Tmax": 2.3726604278074865,
            "color": "#E02020",
        },
        "1-R": {
            "CBF": 16.64376187460418,
            "CBV": 2.246675110829639,
            "MTT": 9.202026599113363,
            "TTP": 13.929702343255224,
            "Tmax": 2.0943635212159593,
            "color": "#E02020",
        },
        "2-L": {
            "CBF": 25.13910208517286,
            "color": "#F7B500",
            "CBV": 2.702370440806901,
            "MTT": 7.27460436052435,
            "TTP": 13.22726346532636,
            "Tmax": 2.5370508727840795,
        },
        "2-R": {
            "CBF": 27.879743816856305,
            "color": "#F7B500",
            "CBV": 2.9568712952238196,
            "MTT": 7.106493152551612,
            "TTP": 12.904476391633168,
            "Tmax": 2.3640389725420725,
        },
        "3-L": {
            "CBF": 36.950718685831625,
            "color": "#6DD400",
            "CBV": 5.379876796714579,
            "MTT": 10.362422997946611,
            "TTP": 17.382956878850102,
            "Tmax": 3.9815195071868583,
        },
        "3-R": {
            "CBF": 39.513853904282115,
            "color": "#6DD400",
            "CBV": 4.647355163727959,
            "MTT": 7.717044500419815,
            "TTP": 13.092359361880773,
            "Tmax": 2.2670025188916876,
        },
    },
}


def gen_color_img(gray_img, image_type: str, best_window: dict):
    height = gray_img.shape[1]
    ww = best_window[image_type][0]
    wl = best_window[image_type][1]
    max_hu, min_hu = (wl + (ww / 2)), (wl - ww / 2)
    bgr_arr = gray2rgb_array(gray_img, ww, wl, is_colormap=True).astype("uint8")
    # lut 自定义映射
    custom_map_arr = np.array(CUSTOM_MAP)

    channels = [cv2.LUT(bgr_arr, custom_map_arr[:, i]) for i in range(3)]
    bgr_arr = np.dstack(channels)
    # 背景着色
    bg_hu = -1
    hu_bgr_arr = np.repeat(gray_img[..., None], 3, axis=-1)
    background_pixel = np.where(
        (hu_bgr_arr[:, :, 0] == bg_hu)
        & (hu_bgr_arr[:, :, 1] == bg_hu)
        & (hu_bgr_arr[:, :, 2] == bg_hu)
    )
    bgr_arr[background_pixel] = [0, 0, 0]
    # draw colorbar , colorbar_h 等于 0.5*图_h
    colorbar_width = 16
    left_padding = 13
    colorbar_img = np.zeros([colorbar_width, 256], np.uint8)
    for i in range(colorbar_width):
        for j in range(256):
            colorbar_img[i, j] = j
    channels = np.array([cv2.LUT(colorbar_img, custom_map_arr[:, i]) for i in range(3)])
    color_img = np.rot90(np.dstack(channels)).astype("uint8")
    res = Image.fromarray(bgr_arr.astype("uint8"))
    res.paste(Image.fromarray(color_img), box=(left_padding, (height // 4)))
    draw = ImageDraw.Draw(res)
    scale_len = 4  # 刻度间隔的数量
    interval = (max_hu - min_hu) / scale_len
    for i in range(scale_len + 1):
        # cbf cbv 从大到小
        hu = str(round(min_hu + interval * (scale_len - i), 1))
        # ttp tmax mtt
        draw.text(
            (left_padding + colorbar_width + 1, (height // 4) + (height // 8) * i),
            hu,
            fill="#ffffff",
            font=FONT,
        )
    rgb_arr = bgr_to_rgb(np.array(res))
    return rgb_arr


def get_combination_images(
    sorted_img_types: list,
    mask_array_map: dict,
    best_window: dict,
    img_depth: int,
    width: int,
    height: int,
    layout: tuple = (2, 3),
):
    """生成组合图
    Args:
        sorted_img_types: 组合图中图片的order
        mask_array_map: 构成组合图的成分
        img_depth: 总的层面数
        width: 组合图宽度
        height: 组合图高度
        layout: 组合图布局
    Returns:
        {comb_index: np.ndarray}
    """
    if layout[0] * layout[1] != len(sorted_img_types):
        raise ValueError("image length not equals layout pool.")

    comb_res = {}
    new_size = (width // layout[1], height // layout[0])
    left_top_coordinates = []
    for row in range(layout[0]):
        for col in range(layout[1]):
            left_top_coordinates.append((new_size[0] * col, new_size[1] * row))
    logger.warning(f"left_top_coordinates: {left_top_coordinates}")

    best_window_for_comb = {}
    best_window_for_comb.update(best_window)
    best_window_for_comb["tMIP"] = [100, 50]

    for index in range(img_depth):
        i = 0
        text = f"组合图 {index+1}/{img_depth}"
        _, h = FONT.getsize(text)
        res = Image.new("RGB", (width, height + h), 0)
        for k in sorted_img_types:
            v = mask_array_map[k]
            # 灰度图
            if k == "tMIP":
                origin_img_np = gray2rgb_array(
                    v[index, :, :].astype("int16"),
                    best_window_for_comb["tMIP"][0],
                    best_window_for_comb["tMIP"][1],
                )
            # 出伪彩图
            else:
                origin_img_np = gen_color_img(v[index, :, :], k, best_window_for_comb)

            img = cv2.resize(
                origin_img_np.astype("uint8"),
                new_size,
                cv2.INTER_LINEAR,
            )
            img = color_draw_text(img, k)
            res.paste(Image.fromarray(img), box=left_top_coordinates[i])
            i += 1
        res = color_draw_text(np.array(res), text)
        comb_res[f"Combination_{index}"] = res
    return comb_res


def canvas2rgb_array(canvas):
    """Adapted from: https://stackoverflow.com/a/21940031/959926"""
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    ncols, nrows = canvas.get_width_height()
    scale = round(math.sqrt(buf.size / 3 / nrows / ncols))
    # return buf.reshape(scale * nrows, scale * ncols, 3)
    return buf.reshape(scale * nrows, scale * ncols, 3)


def canvas_to_pil_img(figure):
    plt_array = canvas2rgb_array(figure)
    plt_array = plt_array.astype(dtype="uint8")
    return Image.fromarray(plt_array)


def _get_width(tag: str) -> int:
    draw = ImageDraw.Draw(Image.new(mode="L", size=(10, 10)))
    text_width, _ = draw.textsize(tag, font=FONT)
    return text_width


@pytest.fixture()
def prepare_data():
    """获取生成CTP 组合图所需的必须数据"""
    root_dir = "/media/tx-deepocean/Data/DICOMS/demos/ctp_data"
    with open("/home/tx-deepocean/Downloads/img_base64.txt", "rb") as f:
        img_base64 = f.read()
    im_arr = np.frombuffer(
        base64.b64decode(img_base64), dtype=np.uint8
    )  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    # #B8BDD8 TODO 底下加文字
    text = "tMIP"
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text(
        ((img.shape[1] - _get_width(text)) // 2, img.shape[0] - 18),
        text,
        fill="#B8BDD8",
        font=FONT,
    )
    img = np.array(pil_img)
    return dict(tMIP=img, CBV=img, CBF=img, MTT=img, TTP=img, Tmax=img)


def gen_table(roi_data, width=1440, height=512):
    slice_idx = roi_data["slice_idx"]
    rows, cols = len(roi_data["form_data"].keys()), 7
    data = np.zeros((rows, cols)).tolist()
    # rcParams['figure.dpi'] = 100
    px = 1 / plt.rcParams["figure.dpi"]
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.facecolor"] = "black"
    # 36不一定是最优值，是一个合适值， 实现自适应
    tabel_heigt = int(36 * (rows + 1))
    fig = plt.figure(
        figsize=(width * px, tabel_heigt * px),
        frameon=True,
        facecolor="black",
        edgecolor="black",
    )
    # fig.set_figheight()
    ax1 = fig.add_subplot(111)
    # hide axes
    fig.patch.set_visible(True)
    fig.patch.set_facecolor("black")
    ax1.axis("off")
    ax1.axis("tight")
    cellColours = np.full(
        shape=(rows, cols), fill_value=[ROI_FORM_COLOR_MAP["tabel_background"]]
    ).tolist()
    colLabels = [f"层面: {slice_idx}"]
    colLabels.extend(ROI_ROWS_TITLE)
    print(chr(0xF081))
    # 获取data
    i = 0
    for position_k, values in roi_data["form_data"].items():
        data[i][0] = "○"
        data[i][1] = position_k
        data[i][2] = round(values["CBF"], 1)
        data[i][3] = round(values["CBV"], 1)
        data[i][4] = round(values["MTT"], 1)
        data[i][5] = round(values["TTP"], 1)
        data[i][6] = round(values["Tmax"], 1)
        i += 1

    tab1 = ax1.table(
        cellText=data,
        colLabels=colLabels,
        colColours=[ROI_FORM_COLOR_MAP["col_title"]] * cols,
        loc="center",
        cellColours=cellColours,
        colWidths=[0.19] * cols,
    )
    tab1.scale(1, 2)  # change row height
    tab1[0, 0].get_text().set_color(ROI_FORM_COLOR_MAP["slice_idx_color"])
    for row_idx in range(1, rows + 1):
        for col_idx in range(0, cols):
            tab1[row_idx, col_idx].get_text().set_color(ROI_FORM_COLOR_MAP["data"])
            tab1[row_idx, col_idx].get_text().set_horizontalalignment("center")
    row_idx = 1
    for roi_values in roi_data["form_data"].values():
        tab1[row_idx, 0].get_text().set_color(roi_values["color"])
        tab1[row_idx + 1, 0].get_text().set_color(roi_values["color"])
        row_idx += 1
        if row_idx >= rows:
            break

    # bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fig_width, fig_height = bbox.width*fig.dpi, bbox.height*fig.dpi
    # print(f"*****row_height: {row_height}, fig_width: {fig_width}, fig_height: {fig_height}")
    plt.savefig("table.png", bbox_inches="tight", dpi=300)
    table_np = np.array(canvas_to_pil_img(fig.canvas), dtype="uint8")[:, :, ::-1].copy()
    return table_np


def test_ctp_combination(prepare_data):
    """根据base64 和　表格数据生成大小为　128*3*3 宽的组合图"""
    t0 = time.time()
    best_window = {
        "tMIP": [100, 50],
        "CBF": [10, 20],
        "CBV": [10, 20],
        "MTT": [10, 20],
        "TTP": [10, 20],
        "Tmax": [10, 20],
    }
    comb_np = get_combination_images(
        SCREENSHOTS_COMBINATION_TYPES, prepare_data, best_window, width=1080, height=720
    )
    tabel_np = gen_table(roi_data, width=1080, height=720)
    final_width, final_height = comb_np.shape[1], (comb_np.shape[0] + tabel_np.shape[0])
    res = Image.new(mode="RGB", size=(final_width, final_height), color="black")
    res.paste(Image.fromarray(comb_np), (0, 0))
    res.paste(Image.fromarray(tabel_np), (0, comb_np.shape[0]))
    cv2.imwrite("./test.png", np.array(res))
    print(f"Total cost: {time.time() - t0}")
