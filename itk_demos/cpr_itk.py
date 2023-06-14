import os

import loguru
import numpy as np
import requests
import SimpleITK as sitk

treeline_url = "http://172.16.3.6:3333/series/1.3.12.2.1107.5.2.18.52322.2022120115530550585177087.0.0.0/predict/mr_head_treeline"
treeline_res = requests.get(treeline_url).json().get("treeLines").get("data")


def main():
    pass


if __name__ == "__main__":
    pass
