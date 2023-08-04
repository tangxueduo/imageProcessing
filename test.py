# import json
# import os
# with open("/home/tx-deepocean/Downloads/ct_neck_result.json", "r") as fp:
#     neck_res = json.load(fp)
# treeline = neck_res["neck"]["arteryTree"]["curveTreeLines"]["data"]


# def read_lines_dict(in_lines):
#     """
#     将中线数据整理成固定格式
#     """
#     out: dict = {"lines": {}, "types": {}, "tangents": {}}
#     if not in_lines:
#         return out

#     use_external_tangent: bool = "tangents" in in_lines
#     # logger.info(f"use external tangents is {use_external_tangent}")

#     for vessel, vessel_info in in_lines["typeShow"].items():
#         lines, types, tangents = [], [], []
#         for idx in vessel_info["data"]:
#             idx_i = str(idx)
#             lines.extend(in_lines["lines"][idx_i])
#             types.append({in_lines["types"][idx_i]: len(in_lines["lines"][idx_i])})
#             if use_external_tangent:
#                 tangents.extend(in_lines["tangents"][idx_i])
#         out["lines"][vessel] = lines
#         out["types"][vessel] = types
#         out["tangents"][vessel] = tangents
#     return out

# line_dict = read_lines_dict(treeline)
# print(len(line_dict["lines"]["vessel1"]))
# print(len(line_dict["tangents"]["vessel1"]))
# from typing import List

# import base64
# import numpy as np
# import cv2
# base64_str = ""
# nparr = np.frombuffer(base64.b64decode(base64_str), np.uint8)
# img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# cv2.imwrite("./test.png", img_np)
# print(img_np.shape)
# img_np = img_np.astype("uint8")
# import SimpleITK as sitk
# import pydicom
# ds = pydicom.read_file("/media/tx-deepocean/Data/DICOMS/demos/ct_heart/1.3.46.670589.33.1.63757724437160306400001.4846811560930707252/1.3.46.670589.33.1.63757724538407097400001.4907301501052470129")
# ds.Rows = img_np.shape[0]
# ds.Columns = img_np.shape[1]
# ds.PixelData = img_np.tobytes()
# ds.RescaleIntercept = 0
# ds.RescaleSlope = 1
# ds.BitsStored = 8
# ds.BitsAllocated = 8
# ds.HighBit = 7
# ds.PixelRepresentation = 0
# ds.WindowWidth = 255
# ds.WindowCenter = 127
# ds.PhotometricInterpretation = "RGB"
# ds.SamplesPerPixel = 3
# ds.PlanarConfiguration = 0
# # sitk_img = sitk.GetImageFromArray(img_np)
# ds.save_as("./test.dcm")
# # sitk.WriteImage(sitk_img, "./test.dcm")


# import json
# import requests

# url ="http://172.16.2.9:3333/series/1.3.12.2.1107.5.1.4.74241.30000023071301050808200051612/predict/ct_heart"
# treeline_url = "http://172.16.2.9:3333/series/1.3.12.2.1107.5.1.4.74241.30000023071301050808200051612/predict/ct_heart_treeline"

# with open("/home/tx-deepocean/Downloads/ct_heart_result.json", "r") as f:
#     res = json.load(f)
# requests.put(url=url, json=res)
# treeline = {
#     "treeLines": res["heart"]["arteryCoronary"][
#         "curveTreeLines"
#     ]
# }
# requests.put(url=treeline_url, json=treeline)
# url = requests.get("").json()
