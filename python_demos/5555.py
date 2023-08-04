import numpy as np

pixel_array = []
data_min = pixel_array.min()
data_max = pixel_array.max()
data_range = data_max - data_min
wl, ww = 50, 100
min_hu = wl - ww / 2
max_hu = wl + ww / 2

pixel_array = np.piecewise(
    pixel_array,
    [pixel_array <= min_hu, pixel_array > max_hu],
    [
        data_min,
        data_max,
        lambda data: ((data - min_hu) / ww * data_range) + data_min,
    ],
)
