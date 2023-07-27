import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = ImageFont.truetype("statics/SourceHanSansCN-Normal.ttf", 18)


def gray2rgb_array(gray_array, ww, wl, is_colormap=False):
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
        rgb_array = np.zeros((temp_array.shape[0], temp_array.shape[1], 3))
        rgb_array = np.repeat(temp_array[..., None], 3, axis=-1)
        # rgb_array[:, :, 0] = temp_array
        # rgb_array[:, :, 1] = temp_array
        # rgb_array[:, :, 2] = temp_array
    else:
        rgb_array = np.zeros((temp_array.shape[0], temp_array.shape[1]))
    return rgb_array


def bgr_to_rgb(rgb_array):
    rgb_array = rgb_array.astype(np.uint8)
    rgb_array = rgb_array[:, :, ::-1]
    return rgb_array


def _get_width(text: str) -> int:
    draw = ImageDraw.Draw(Image.new(mode="L", size=(10, 10)))
    text_width, _ = draw.textsize(text, font=FONT)
    return text_width


def color_draw_text(np_array: np.ndarray, text: str):
    pil_image = Image.fromarray(np_array)
    draw = ImageDraw.Draw(pil_image)
    draw.text(
        ((np_array.shape[1] - _get_width(text)) // 2, np_array.shape[0] - 18),
        text,
        fill="#ffffff",
        font=FONT,
    )
    return np.array(pil_image)
