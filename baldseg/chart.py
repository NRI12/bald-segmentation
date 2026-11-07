import json

import numpy as np

TO_LEVEL = {"1": 0, "3V": 1, "4": 2, "5": 3, "6": 4, "7": 5}


def load(path):
    return json.load(open(path))


def stage(ratio, cfg):
    return cfg["stages"][int(np.searchsorted(cfg["thresholds"], ratio, side="right"))]


def level(ratio, cfg):
    s = stage(ratio, cfg)
    return int(ratio >= cfg["stage_ratios"][s]) if s == "2-3" else TO_LEVEL[s]
