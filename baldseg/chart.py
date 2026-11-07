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


PALETTE = np.array([(255, 255, 255), (225, 225, 225), (60, 170, 60), (230, 60, 60), (110, 110, 110)], np.uint8)


def reference(stage, cfg, root="weights/chart", scale=3):
    import cv2
    tiles = []
    for s in stage.split("-"):
        tile = cv2.resize(PALETTE[cv2.imread(f"{root}/{s}.png", cv2.IMREAD_GRAYSCALE)], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.putText(tile, f"Stage {s.replace('V', ' Vertex')}  R={cfg['chart_ratios'][s]:.0%}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        tiles.append(tile)
    return np.hstack(tiles)
