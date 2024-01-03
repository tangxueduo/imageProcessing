import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = ImageFont.truetype("statics/SourceHanSansCN-Normal.ttf", 18)


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
