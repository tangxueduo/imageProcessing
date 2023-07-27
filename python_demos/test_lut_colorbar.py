import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.constants import CUSTOM_MAP

FONT = ImageFont.truetype("statics/SourceHanSansCN-Normal.ttf", 18)
width, height = 512, 512
res = Image.new("RGB", (512, 512), 0)

grey_img = np.zeros([16, 256], np.uint8)
custom_map_arr = np.array(CUSTOM_MAP)
for i in range(16):
    for j in range(256):
        grey_img[i, j] = j
channels = np.array(
    [cv2.LUT(grey_img, custom_map_arr[:, i]) for i in range(3)]
)  # 自己测试turbo
color_img = np.dstack(channels)
# color_img = cv2.applyColorMap(grey_img,cv2.COLORMAP_JET)
# for j in range(256):
#     print(f"{[color_img[0, j, 2], color_img[0, j, 1], color_img[0, j, 0]]}")
color_img = np.rot90(color_img).astype("uint8")

res.paste(Image.fromarray(color_img), box=(13, (height // 4)))
draw = ImageDraw.Draw(res)
draw.text((13 + 16 + 1, (height // 4)), "0.95", fill="#ffffff", font=FONT)
draw.text(
    (13 + 16 + 1, (height // 4) + (height / 8) * 1), "0.75", fill="#ffffff", font=FONT
)
draw.text(
    (13 + 16 + 1, (height // 4) + (height / 8) * 2), "0.50", fill="#ffffff", font=FONT
)
draw.text(
    (13 + 16 + 1, (height // 4) + (height / 8) * 3), "0.25", fill="#ffffff", font=FONT
)
draw.text(
    (13 + 16 + 1, (height // 4) + (height / 8) * 4), "0.00", fill="#ffffff", font=FONT
)
cv2.imwrite("./colormap.png", np.array(res))

# key = cv2.waitKey(0)
