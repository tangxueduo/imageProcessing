import numpy as np
from skimage import measure

cc, cc_num = measure.label(mask, return_num=True, connectivity=1)
arr = np.ones((16, 512, 512))
arr[2:250:250] = 1
arr[3:250:250] = 1
arr[4:250:250] = 1

arr[2:260:260] = 1
arr[3:260:260] = 1
arr[4:260:260] = 1
