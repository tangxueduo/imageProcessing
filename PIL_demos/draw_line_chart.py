import json
import logging
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import SimpleITK as sitk
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import BoxStyle
from matplotlib.path import Path
from PIL import Image


def main(sex, age, score):
    marker_point = (age, score)
    sex_data_dict = {
    "F": {
        "25%": [0, 0, 0, 0, 0, 0, 0],               
        "50%": [0, 0, 0, 0, 0, 4, 24],
        "75%": [0, 0, 0, 10, 33, 87, 123],
        "90%": [4, 9, 23, 66, 140, 310, 362],
      },
    "M": {
        "25%": [0, 0, 0, 0, 3, 14, 28], 
        "50%": [0, 0, 3, 16, 41, 118, 151],
        "75%": [2, 11, 44, 101, 187, 434, 569],
        "90%": [21, 64, 176, 320, 502, 804, 1178],
      },
    }
    max_dict = {"F": 362, "M": 1178}
    fig = plt.figure(dpi=300, facecolor="#171D2E")
    fig.set_figwidth(2)
    fig.set_figheight(2.4)

    ax= fig.add_axes([0.1,0.1,0.8,0.8])
    ax.patch.set_facecolor("#171D2E")
    x_range = [math.floor((35+39)/2), math.floor((40+44)/2), math.floor((45+49)/2), math.floor((50+54)/2), math.floor((55+59)/2), math.floor((60+64)/2), math.floor((65+70)/2)]
    ax.xaxis.set_ticks(np.arange(37, 70, 10))
    ax.set_ylim(bottom=0, top=2000, auto=True)
    # 将左边框和下边框的颜色设为灰色
    ax.spines['left'].set_color('#8A9EC3') 
    ax.spines['bottom'].set_color('#8A9EC3')
    # 去掉边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #移位置 设为原点相交
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    # ax.spines['left'].set_position(('data',0))
    # 修改轴线刻度值的颜色和大小
    plt.tick_params(axis='both', which='major', labelsize=5, colors="#8A9EC3", bottom=False, left=False, width=ax.get_window_extent().width)
    for k, v in sex_data_dict.items():
        if k == sex:
            for percent, yaxis in v.items():
                if percent == "25%":
                    line_color = "#0164FE"
                    y1_data = yaxis
                    ax.plot(x_range, y1_data, line_color, linewidth=1, label="25%")
                elif percent == "50%":
                    line_color = "#2AC7F6"
                    y2_data = yaxis
                    ax.plot(x_range, y2_data, line_color, linewidth=1, label="50%")
                elif percent == "75%":
                    line_color = "#F9A727"
                    y3_data = yaxis
                    ax.plot(x_range, y3_data, line_color, linewidth=1, label="75%")
                else:
                    y4_data = yaxis
                    line_color = "#FF5959"
                    ax.plot(x_range, y4_data, line_color, linewidth=1, label="90%")

    leg = plt.legend(bbox_to_anchor=(1.05, 0.9), loc='upper left', borderaxespad=0, prop={'size': 4}, labelcolor="#FFFFFF", facecolor="#171D2E", edgecolor="#171D2E")
    # title = plt.title('冠脉风险评估', fontsize=5, fontweight='bold', pad=10, color="#FFFFFF", ha="left", va="center", bbox={'facecolor':'#8A9EC3', 
    #         'alpha':0.3, 'pad':4})
    title = ax.set_title('冠脉风险评估', position=(.5, 1.02),
             backgroundcolor='#262E48', color='#FFFFFF',
             verticalalignment="bottom", horizontalalignment="right", fontsize=8)
    title._bbox_patch._mutation_aspect = 0.04
    title.get_bbox_patch().set_boxstyle("square", pad=12)
    
    # bb.set_boxstyle("ext", pad=0.4, width=ax.get_window_extent().width)
    ax.plot(age, score, color="#FFFFFF", marker="+", label="该患者所在位置", markersize=4)
    xais_txt = "年龄"
    yais_txt = "分数"
    plt.figtext(1.02, 0.08,
                xais_txt,
                horizontalalignment='left',
                size=5, weight='light',
                color="#8A9EC3",
            )
    plt.figtext(0, 0.88,
                yais_txt,
                horizontalalignment='left',
                size=5, weight='light',
                color="#8A9EC3",
            )
    plt.figtext(0.9, 0.85,
                "+ 该患者所在位置",
                horizontalalignment='left',
                size=5, weight='light',
                color="#FFFFFF",
            )
    plt.savefig('/media/tx-deepocean/Data/DICOMS/demos/000CACS1.png', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main('M', 50, 2000)
    