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
    fig.set_figheight(2)

    rect1 = [0, 0.83, 1, 0.08] # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect2 = [0.15, 0.2, 0.7, 0.6]

    #在fig中添加子图ax，并赋值位置rect
    ax1 = plt.axes(rect1)
    ax1.patch.set_facecolor("#262E48")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.text(0.02, 0.5, "冠脉风险评估", fontsize=5, va="center", ha="left", color="#FFFFFF")
    ax1.tick_params(axis='both', pad=0, which='major', top=False, right=False, bottom=False, left=False, width=ax1.get_window_extent().width)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = plt.axes(rect2)
    ax2.patch.set_facecolor("#171D2E")
    x_range = [math.floor((35+39)/2), math.floor((40+44)/2), math.floor((45+49)/2), math.floor((50+54)/2), math.floor((55+59)/2), math.floor((60+64)/2), math.floor((65+70)/2)]
    ax2.xaxis.set_ticks(np.arange(37, 70, 10))
    # ax2.set_ylim(bottom=0, top=1800, auto=True)
    ax2.set_xlim(37, 70)
    ax2.set_ylim(0, (1500 if score <= 1500 else score)  + 80)
    # 将左边框和下边框的颜色设为黑色
    ax2.spines['left'].set_color('#000000') 
    ax2.spines['bottom'].set_color('#000000')
    # 去掉边框
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.vlines([47, 57, 67], 0, score, colors='#000000', alpha = 0.05, linewidth=0.5)
    ax2.spines["bottom"].set_linewidth(0.5)
    ax2.spines["left"].set_linewidth(0.5)
    # ax2.spines['left'].set_position(('outward', 10))
    # ax2.spines['left'].set_position(('axes', 0.3))


    # 修改轴线刻度值的颜色和大小
    plt.tick_params(axis='both', pad=-2, which='major', labelsize=5, colors="#8A9EC3", bottom=False, left=False, width=ax2.get_window_extent().width)
    # 调整子图间距
    # plt.subplots_adjust(left=0.1,bottom=0.1,right=0.15,top=0.15,wspace=0.15,hspace=0.15)
    # plt.margins(2, 2)
    for k, v in sex_data_dict.items():
        if k == sex:
            for percent, yaxis in v.items():
                if percent == "25%":
                    line_color = "#0164FE"
                    y1_data = yaxis
                    ax2.plot(x_range, y1_data, line_color, linewidth=1, label="25%")
                elif percent == "50%":
                    line_color = "#2AC7F6"
                    y2_data = yaxis
                    ax2.plot(x_range, y2_data, line_color, linewidth=1, label="50%")
                elif percent == "75%":
                    line_color = "#F9A727"
                    y3_data = yaxis
                    ax2.plot(x_range, y3_data, line_color, linewidth=1, label="75%")
                else:
                    y4_data = yaxis
                    line_color = "#FF5959"
                    ax2.plot(x_range, y4_data, line_color, linewidth=1, label="90%")

    leg = plt.legend(bbox_to_anchor=(0.05, 0.9), loc='upper left', borderaxespad=0, prop={'size': 4}, labelcolor="#FFFFFF", facecolor="#171D2E", edgecolor="#171D2E")
    
    ax2.plot(age, score, color="#FFFFFF", marker="+", label="该患者所在位置", markersize=2)
    ax2.set_xlabel("年龄", labelpad=1, loc="right", fontdict=dict(fontsize=5, color='#8A9EC3',
                       weight='bold'))
    ax2.set_ylabel("分数", labelpad=1, loc="top", fontdict=dict(fontsize=5, color='#8A9EC3',
                       weight='bold'))

    plt.figtext(0.22, 0.76,
                "+",
                horizontalalignment='center',
                va="center",
                size=4, weight='light',
                color="#FFFFFF",
            )
    plt.figtext(0.27, 0.76,
                "该患者所在位置",
                horizontalalignment='left',
                va="center",
                size=4, weight='light',
                color="#FFFFFF",
            )
    plt.savefig('/media/tx-deepocean/Data/DICOMS/demos/000CACS1.png', pad_inches=0)


if __name__ == '__main__':
    main('M', 38, 50)
    