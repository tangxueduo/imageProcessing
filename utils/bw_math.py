from decimal import ROUND_HALF_UP, Decimal

import numpy as np


def round_half_up(number) -> int:
    # https://note.nkmk.me/python-round-decimal-quantize/
    return int(Decimal(str(number)).quantize(Decimal("0"), rounding=ROUND_HALF_UP))


def get_gaussian_kernel_1d(kernel_size, center_offset=0, sigma=1):
    if kernel_size % 2 == 1:
        lw = (kernel_size - 1) / 2
    else:
        lw = kernel_size / 2
    hg = kernel_size - lw
    x = np.arange(-lw, hg)
    phi_x = np.exp(((x - center_offset) ** 2) / -(2 * sigma ** 2))
    return phi_x


def gen_2d_gaussian(kernel_size=(5, 5, 5), center_offset=(0, 0, 0), sigma=(1, 1, 1)):
    phi_x = get_gaussian_kernel_1d(kernel_size[0], center_offset[0], sigma[0]).reshape(
        -1, 1
    )
    phi_y = get_gaussian_kernel_1d(kernel_size[1], center_offset[1], sigma[1]).reshape(
        1, -1
    )
    phi = phi_x * phi_y
    return phi


def gen_3d_gaussian(
    kernel_size=(224, 224, 20), center_offset=(0, 0, 0), sigma=(65, 65, 5)
):
    phi_x = get_gaussian_kernel_1d(kernel_size[0], center_offset[0], sigma[0]).reshape(
        -1, 1, 1
    )
    phi_y = get_gaussian_kernel_1d(kernel_size[1], center_offset[1], sigma[1]).reshape(
        1, -1, 1
    )
    phi_z = get_gaussian_kernel_1d(kernel_size[2], center_offset[2], sigma[2]).reshape(
        1, 1, -1
    )
    phi = phi_x * phi_y * phi_z
    return phi
