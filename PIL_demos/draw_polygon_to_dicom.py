import logging
import math

import numpy as np
from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S %a",
)


def main():
    """
    Convert physical coordinates to image coordinates.
    """
    # 外参矩阵
    Out = np.mat(
        [
            [-0.117, -0.992, 0.028, -0.125],
            [-0.0033, -0.0278, -0.9996, 0.2525],
            [0.993, -0.1174, 0.00000315, 0.0716],
            [0, 0, 0, 1],
        ]
    )

    # 内参矩阵 (需要改)
    K = np.mat([[512/64, 0, 368.114], [0, 512/64, 223.969], [0, 0, 1]])
    """坐标转换"""

    world_coordinate = np.mat(
        [[92.4066588991363], [-216.3043427709651], [-226.3831620231209], [1]]
    )
    # 世界坐标系转换为相加坐标系 （Xw,Yw,Zw）--> (Xc,Yc,Zc)
    camera_coordinate = Out * world_coordinate
    print(f"相机坐标为:\n{camera_coordinate}")
    Zc = float(camera_coordinate[2])
    print(f"Zc={Zc}")

    # 相机坐标系转图像坐标系 (Xc,Yc,Zc) --> (x, y)  下边的f改为焦距
    focal = 1
    focal_length = np.mat([[focal, 0, 0, 0], [0, focal, 0, 0], [0, 0, 1, 0]])
    image_coordinate = (focal_length * camera_coordinate) / Zc
    print(f"图像坐标为:\n{image_coordinate}")

    # 图像坐标系转换为像素坐标系
    pixel_coordinate = K * image_coordinate
    print(f"像素坐标为:\n{pixel_coordinate}")
    logging.info(f"*****pixel_coordinate: {pixel_coordinate}")

    tri_x, tri_y, _ = (
        pixel_coordinate[0],
        pixel_coordinate[1],
        pixel_coordinate[2],
    )
    tmp = 15
    pl1 = (tri_x - 20, tri_y)
    pl2 = (tri_x - 20 - tmp, tri_y + tmp * math.sin(((7.5 * math.pi) / 180)))
    pl3 = (tri_x - 20 - tmp, tri_y - tmp * math.sin(((7.5 * math.pi) / 180)))
    pr1 = (tri_x + 20, tri_y)
    pr2 = (tri_x + 20 + tmp, tri_y + tmp * math.tan(((7.5 * math.pi) / 180)))
    pr3 = (tri_x + 20 + tmp, tri_y - tmp * math.tan(((7.5 * math.pi) / 180)))
    tri_cnt = [pl1, pl2, pl3]
    tri_cnt1 = [pr1, pr2, pr3]
    logging.info(f"tri_cnt: {tri_cnt}, tri_cnt1: {tri_cnt1}")

    img = Image.new("RGB", (512, 512), "#000000")
    img1 = ImageDraw.Draw(img)
    img1.polygon(tri_cnt, fill="#FF8C00", outline="blue")
    img1.polygon(tri_cnt1, fill="#FF8C00", outline="blue")
    img.show()
    img.save("/media/tx-deepocean/Data/DICOMS/RESULT/tmp/tmp.png")


if __name__ == "__main__":
    main()
