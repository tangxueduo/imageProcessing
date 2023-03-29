import requests 
# import os

# url = "http://172.16.3.35:3333/series/1.2.156.112605.189250946070725.20181216014141.3.5316.10/predict/ct_aorta"
# with open("/media/tx-deepocean/Data/DICOMS/demos/aorta/new_mock_data/ct_aorta_result.json", "r") as fp:
#     aorta_res = fp.read()
# res = requests.put(url, json=aorta_res)
# treeline_url = "http://172.16.3.35:3333/series/1.2.156.112605.189250946070725.20181216014141.3.5316.10/predict/ct_aorta_treeline"
# ress = requests.get(url).json()
# res1 = requests.put(treeline_url, json={"treeLines": ress["aorta"]["aorta"]["curveTreeLines"]})

url = "http://172.16.3.35:3333/series/1.2.156.112605.189250946070725.20181216014141.3.5316.10/feedback/ct_aorta_series_images"

res = requests.get(url).json()
requests.put(url, json={"seriesImages": res["filmImages"]})