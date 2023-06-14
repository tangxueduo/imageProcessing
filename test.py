# import requests
# import json
# url = "http://172.16.3.30:3333/series/1.2.840.113619.2.416.242355036453511869474923342565595289454/predict/ct_cerebral"
# ct_neck = "/home/tx-deepocean/Downloads/ct_neck_result.json"
# with open(ct_neck, "r") as f:
#     res = json.load(f)
#     print(res.keys())
# treeline_url = "http://172.16.3.30:3333/series/1.2.840.113619.2.416.242355036453511869474923342565595289454/predict/ct_neck_treeline"

# rq = requests.put(url, json=res)

# rq = requests.put(treeline_url, json={"treeLines": res["neck"]["arteryTree"]["curveTreeLines"]})

import os
import sys

import pydicom
import SimpleITK as sitk


def InitDicomFile():
    infometa = pydicom.dataset.Dataset()
    infometa.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    infometa.ImplementationVersionName = "Python " + sys.version
    infometa.MediaStorageSOPClassUID = "CT Image Storage"
    infometa.FileMetaInformationVersion = b"\x00\x01"
    return infometa


def read_dcm_by_instancenum():
    series_dir = "/media/tx-deepocean/Data/DICOMS/demos/mra/CT437287/1.2.392.200036.9125.2.138612190166.20230531000172/1.2.826.0.1.3680043.10.498.2628487157865322590071382151070583474"
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(series_dir)
    print(names)
    dic = {}
    for name in names:
        ds = pydicom.read_file(name, stop_before_pixels=False, force=True)
        instancenum = int(ds.InstanceNumber)
        dic[instancenum] = ds
    images = []
    numbers = list(dic.keys())
    numbers.sort(reverse=True)
    for num in numbers:
        images.append(dic[num])
    return images


images = read_dcm_by_instancenum()
count = 0

for image in images:
    image.ImagePositionPatient[2] += 0.75
    # info = pydicom.dataset.FileDataset({},image)
    # infometa = InitDicomFile()
    # ds = pydicom.dataset.FileDataset({},info,is_implicit_VR =True, file_meta=infometa)
    # image.sava_as(f"/media/tx-deepocean/Data/DICOMS/demos/mra/CT437288/{count}.dcm")
    pydicom.dcmwrite(
        f"/media/tx-deepocean/Data/DICOMS/demos/mra/CT437288/{count}.dcm", image
    )
    count += 1
