BRAIN_AREA_MAP = {
    1: {"origin": "L-Parietal lobe", "relative": 2},  # 左顶叶
    2: {"origin": "R-Parietal lobe", "relative": 1},  # 右顶叶
    3: {"origin": "L-Frontal lobe", "relative": 4},  # 左额叶
    4: {"origin": "R-Frontal lobe", "relative": 3},  # 右额叶
    5: {"origin": "L-Temporal lobe", "relative": 6},  # 左颞叶
    6: {"origin": "R-Temporal lobe", "relative": 5},  # 右颞叶
    7: {"origin": "L-Occipital lobe", "relative": 8},  # 左枕叶
    8: {"origin": "R-Occipital lobe", "relative": 7},  # 右枕叶
    9: {"origin": "L-Insula lobe", "relative": 10},  # 左岛叶
    10: {"origin": "R-Insula lobe", "relative": 9},  # 右岛叶
    11: {"origin": "L-Basal nucleus", "relative": 12},  # 左基底节
    12: {"origin": "R-Basal nucleus", "relative": 11},  # 右基底节
    13: {"origin": "L-Cerebellum", "relative": 14},  # 左小脑
    14: {"origin": "R-Cerebellum", "relative": 13},  # 右小脑
    15: "Brain stem",  # 脑干
    16: "Corpus callosum",  # 胼胝体
    17: {"origin": "L-Ventricle", "relative": 18},  # 左脑室
    18: {"origin": "R-Ventricle", "relative": 17},  # 右脑室
}

left_brain_label = [1, 3, 5, 7, 9, 11, 13, 17]
right_brain_label = [2, 4, 6, 8, 10, 14, 18]
