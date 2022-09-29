import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from PIL import Image


def hu_normalization(hu_array):
    temp_array = np.clip(hu_array, 0, 255)
    return temp_array


# dcm_file = "/media/tx-deepocean/Data/DICOMS/demos/output_new_data_2d/1.3.46.670589.33.1.63790032426654424200002.5278662375047719733/TMAX.nii.gz"
# mask_img = sitk.ReadImage(mask_file)

dcm_file = "/media/tx-deepocean/Data/DICOMS/demos/TMAX/TMAX023.dcm"
dcm_img = sitk.ReadImage(dcm_file)
mask_np_array = sitk.GetArrayFromImage(dcm_img)
print(f"mask_np_array: {mask_np_array.min()}, {mask_np_array.max()}")
print(
    f"mask_np_array[0,:,:]: {mask_np_array[0,:,:].min()}, {mask_np_array[0,:,:].max()}"
)
print(mask_np_array.shape)
img_s = hu_normalization(mask_np_array[0, :, :])
# img_s = mask_np_array[10,:,:]
print(img_s.shape, img_s.min(), img_s.max(), img_s.dtype)
pil_img = Image.fromarray(img_s)

plt.figure(facecolor="black")

st = plt.imshow(img_s, cmap=plt.cm.jet)  # 设置cmap为RGB图
plt.colorbar()  # 显示色度条
plt.axis("off")
plt.savefig("./tttt.png")
