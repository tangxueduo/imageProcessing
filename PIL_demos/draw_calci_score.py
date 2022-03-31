import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import logging
import math
import SimpleITK as sitk

logger = logging.basicConfig()
def main():
    url = "http://172.16.6.6:3333/series/1.2.392.200036.9116.2.1796265542.1617755474.4.1362300001.1/predict/ct_heart_cac_score"
    res = requests.get(url).json()
    # print(res)
    # 画折线图 保存为png
    vessel_map = {"vessel1": "LM", "vessel2": "LAD", "vessel5": "LCX", "vessel9": "RCA"}
    cparameter = 0.777
    sliceThickness = 1.0
    score = res["summary"]["score"]
    # 画钙化积分表
    if (sliceThickness < 1.5):
        alpha = sliceThickness / 1.5
    elif (sliceThickness >= 3.5):
        alpha = sliceThickness / 3.5
    else:
        alpha = 1
    
    #列名
    col=["冠状动脉", "容积(mm3)", "等效质量(mg CaHA)", "钙化积分"]
    #行名
    row=["LM", "LAD", "LCX", "RCA", "总计"]
    x, y, z = res["summary"]["spacing"]
    tabel_list = []
    ctaVolumePixel, equivalentMassHU, calcificationScore = 0, 0, 0
    for k, v in res["lesions"].items():
        for index, val in enumerate(v["contour"]["data"]):
            ctaVolumePixel += val["pixelArea"]
            if (val["avgHU"] != -9999):
                equivalentMassHU += val["pixelArea"] * val["avgHU"]
            for i in range(4):
                calcificationScore += val["agatstonPixelArea"][i] * score[i]
        # 计算容积
        ctaVolume = round(ctaVolumePixel * x * y * z, 2)
        # 计算等效质量
        equivalentMass = round((equivalentMassHU * cparameter * x * y * z) / 1000, 2)
        # 计算钙化积分
        calcificationScore = round(calcificationScore * alpha, 2)
        tabel_list.append([k, ctaVolume, equivalentMass, calcificationScore])
        
    
    total = [0, 0, 0, 0]
    for i in range(len(tabel_list)):
        total[1] += tabel_list[i][1]
        total[2] += tabel_list[i][2]
        total[3] += tabel_list[i][3]
    total[0] = "总计"
    total[1] = round(total[1], 2)
    total[2] = round(total[2], 2)
    total[3] = round(total[3], 2)
    tabel_list.append(total)
    fig = plt.figure(figsize=(15,8))
    plt.rcParams["font.sans-serif"]=["SimHei"] #用来正常显示中文标签
    plt.rcParams["axes.unicode_minus"]=False #用来正常显示负号

    tab = plt.table(cellText=tabel_list, 
                colLabels=col, 
                loc='center', 
                cellLoc='center',
                rowLoc='center')
    tab.scale(1,2) 
    plt.axis('off')
    # plt.savefig('/media/tx-deepocean/Data/DICOMS/demos/Calcification_Score.png', bbox_inches='tight')
    # Extract the plot as an array
    plt_array = canvas2rgb_array(fig.canvas)
    dcm = sitk.GetImageFromArray(plt_array)
    original_size = plt_array.shape
    new_width = 512
    original_spacing = dcm.GetSpacing()
    new_spacing = [(original_size[0] - 1) * original_spacing[0]
                    / (new_width - 1)] * 2
    new_size = [new_width, int((original_size[1] - 1)
                                * original_spacing[1] / new_spacing[1])]
    print(new_size)
    image = sitk.Resample(image1=dcm, size=new_size,
                            transform=sitk.Transform(),
                            interpolator=sitk.sitkLinear,
                            outputOrigin=dcm.GetOrigin(),
                            outputSpacing=new_spacing,
                            outputDirection=dcm.GetDirection(),
                            defaultPixelValue=0,
                            outputPixelType=dcm.GetPixelID())
    sitk.WriteImage(image, "/media/tx-deepocean/Data/DICOMS/demos/Calcification_Score.dcm")
    print(plt_array.shape)

def canvas2rgb_array(canvas):
    """Adapted from: https://stackoverflow.com/a/21940031/959926"""
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    ncols, nrows = canvas.get_width_height()
    print(f'****ncols: {ncols}, nrows: {nrows}')
    scale = round(math.sqrt(buf.size / 3 / nrows / ncols))
    # return buf.reshape(scale * nrows, scale * ncols, 3)
    return buf.reshape(3, scale * nrows, scale * ncols)
if __name__ == '__main__':
    main()
    