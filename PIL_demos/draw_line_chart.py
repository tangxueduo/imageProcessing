import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import logging
import math
import SimpleITK as sitk
import pandas as pd
from PIL import Image
import os
import time
import pydicom
from pydicom.dataset import FileDataset

def main(sex, age, score):
    marker_point = (age, score)
    sex_data_dict = {
    "F": {
      '25%': [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      ],
      '50%': [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 4, 4, 4, 4, 4, 24, 24, 24, 24, 24,
      ],
      '75%': [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 33, 33,
        33, 33, 33, 87, 87, 87, 87, 87, 123, 123, 123, 123, 123,
      ],
      '90%': [
        4, 4, 4, 4, 4, 9, 9, 9, 9, 9, 23, 23, 23, 23, 23, 66, 66, 66, 66, 66,
        140, 140, 140, 140, 140, 310, 310, 310, 310, 310, 362, 362, 362, 362,
        362,
      ],
    },
    "M": {
      '25%': [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3,
        3, 14, 14, 14, 14, 14, 28, 28, 28, 28, 28,
      ],
      '50%': [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 16, 16, 16, 16, 16, 41, 41,
        41, 41, 41, 118, 118, 118, 118, 118, 151, 151, 151, 151, 151,
      ],
      '75%': [
        2, 2, 2, 2, 2, 11, 11, 11, 11, 11, 44, 44, 44, 44, 44, 101, 101, 101,
        101, 101, 187, 187, 187, 187, 187, 434, 434, 434, 434, 434, 569, 569,
        569, 569, 569,
      ],
      '90%': [
        21, 21, 21, 21, 21, 64, 64, 64, 64, 64, 176, 176, 176, 176, 176, 320,
        320, 320, 320, 320, 502, 502, 502, 502, 502, 804, 804, 804, 804, 804,
        1178, 1178, 1178, 1178, 1178,
      ],
    },
    }
    max_dict = {"F": 362, "M": 1178}
    fig = plt.figure(dpi=300, facecolor="#171D2E")
    fig.set_figwidth(2)
    fig.set_figheight(2)

    ax= fig.add_axes([0.1,0.1,0.8,0.8])
    ax.patch.set_facecolor("#171D2E")
    x_range = np.arange(35, 70)
    ax.xaxis.set_ticks(np.arange(35, 70, 10), fontsize=4)
    ax.yaxis.set_ticks(np.arange(0, score, 500), fontsize=4)
    # ax.spines['left'].set_color('#8A9EC3')   #将左边框的颜色设为绿色
    # 去掉边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 修改轴线刻度值的颜色和大小
    plt.tick_params(axis='both', which='major', labelsize=5, colors="#8A9EC3", bottom=False, left=False)
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

    leg = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, prop={'size': 4}, labelcolor="#FFFFFF", facecolor="#171D2E", edgecolor="#171D2E")
    ax.set_title("冠脉风险评估", x=0.05, y=0.95, fontsize=5, color="#FFFFFF", loc="left")
    ax.plot(age, score, color="#FFFFFF", marker="+", markersize=4)
    xais_txt = "年龄"
    yais_txt = "分数"
    plt.figtext(0.85, 0.06,
                xais_txt,
                horizontalalignment='left',
                size=5, weight='light',
                color="#8A9EC3",
            )
    plt.figtext(0, 0.85,
                yais_txt,
                horizontalalignment='left',
                size=5, weight='light',
                color="#8A9EC3",
            )
    plt.savefig('/media/tx-deepocean/Data/DICOMS/demos/000CACS1.png', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main('M', 50, 2000)
    