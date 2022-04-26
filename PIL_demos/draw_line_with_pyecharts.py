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
from pyecharts.charts import Line
import pyecharts.options as opts
from pyecharts.faker import Faker
# 导入输出图片工具
from pyecharts.render import make_snapshot
# 使用snapshot-selenium 渲染图片
from snapshot_selenium import snapshot


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
    for k, v in sex_data_dict.items():
        if k == sex:
            for percent, yaxis in v.items():
                if percent == "25%":
                    line_color = "#0164FE"
                    y1_data = yaxis
                    # ax.plot(x_range, y1_data, line_color, linewidth=1, label="25%")
                elif percent == "50%":
                    line_color = "#2AC7F6"
                    y2_data = yaxis
                elif percent == "75%":
                    line_color = "#F9A727"
                    y3_data = yaxis
                else:
                    y4_data = yaxis
                    line_color = "#FF5959"
    line1=(
        Line()
        .add_xaxis(xaxis_data=['一','二','三','四','五','六','七','八','九'])
        .add_yaxis(
            '25%',
            y_axis=y1_data,
            linestyle_opts=opts.LineStyleOpts(width=2) # 线宽度设置为2
        )
        .add_yaxis(
            '50%',
            y_axis=y2_data,
            linestyle_opts=opts.LineStyleOpts(width=2),
        )
        .add_yaxis(
            '75%',
            y_axis=y3_data,
            linestyle_opts=opts.LineStyleOpts(width=2),
        )
        .add_yaxis(
            '90%',
            y_axis=y4_data,
            linestyle_opts=opts.LineStyleOpts(width=2),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title='line 对数轴示例'),
            xaxis_opts=opts.AxisOpts(name='x'),# 设置x轴名字属性
            yaxis_opts=opts.AxisOpts(
                type_='log', # 设置类型
                name='y', # 设置y轴名字属性
                splitline_opts=opts.SplitLineOpts(is_show=True),# 这里将分割线显示
                is_scale=True # 这里范围设置true是显示线的顶端，设置为False的话就只会显示前2个数值
            )
        )
    )

    make_snapshot(snapshot, line1.render(), "/media/tx-deepocean/Data/DICOMS/demos/000CACS1.png")
    # plt.savefig('/media/tx-deepocean/Data/DICOMS/demos/000CACS1.png', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main('M', 50, 2000)
    