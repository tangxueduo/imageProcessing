import time

import numpy as np
import SimpleITK as sitk

# t = time.time()
# lesion_nii = sitk.ReadImage("/media/tx-deepocean/Data/DICOMS/demos/aorta/lesion-seg.nii.gz")
# vessel_nii = sitk.ReadImage("/media/tx-deepocean/Data/DICOMS/demos/aorta/output_lps-seg.nii.gz")

# lesion_mask = sitk.GetArrayFromImage(lesion_nii)
# vessel_mask = sitk.GetArrayFromImage(vessel_nii)

# out = np.where(lesion_mask!=0, lesion_mask, vessel_mask)
# print(np.unique(out))

url = "http://172.16.3.35:3333/series/1.2.840.113619.2.404.3.1074448704.575.1620281836.47.4/feedback/ct_aorta_film_images"
import requests

res = requests.get(url).json()

res["filmImages"].pop("Custom")
# res["seriesImages"].pop("Custom")
print(res)
requests.put(url=url, json=res)
