import argparse
import json
from pathlib import Path

import cv2
import numpy as np

BOXES = {"1": (107, 193), "2": (411, 192), "3": (720, 197), "3V": (1034, 194), "4": (107, 540), "5": (408, 529), "6": (712, 536), "7": (1028, 535)}
ORDER = ["1", "2-3", "3V", "4", "5", "6", "7"]


def chart_ratio(gray, x, y, cut):
    crop = gray[y - 4:y + 124, x - 4:x + 132]
    contours, _ = cv2.findContours((crop < 200).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    head = np.zeros(crop.shape, np.uint8)
    cv2.drawContours(head, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    ys, _ = np.nonzero(head)
    top, height = ys.min(), np.ptp(ys)
    row = np.nonzero(head[top + int(0.25 * height)])[0]
    yy, xx = np.mgrid[: crop.shape[0], : crop.shape[1]]
    skull = cv2.erode(head, np.ones((5, 5), np.uint8)).astype(bool) & (yy < top + cut * height) & (xx > row.min() + 3) & (xx < row.max() - 3)
    hair, bald = (skull & (crop < 110)).sum(), (skull & (crop > 200)).sum()
    return float(bald / (bald + hair))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("chart", type=Path)
    p.add_argument("--cut", type=float, default=0.65)
    p.add_argument("--out", type=Path, default=Path("weights/norwood_thresholds.json"))
    args = p.parse_args()

    gray = cv2.cvtColor(cv2.imread(str(args.chart)), cv2.COLOR_BGR2GRAY)
    ratios = {s: chart_ratio(gray, x, y, args.cut) for s, (x, y) in BOXES.items()}
    levels = {**{s: ratios[s] for s in ratios if s not in ("2", "3")}, "2-3": (ratios["2"] + ratios["3"]) / 2}
    values = [levels[s] for s in ORDER]
    thresholds = [(a + b) / 2 for a, b in zip(values, values[1:])]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"cut": args.cut, "chart_ratios": ratios, "stages": ORDER, "stage_ratios": dict(zip(ORDER, values)), "thresholds": thresholds}, open(args.out, "w"), indent=2)
    print(json.dumps({"chart_ratios": {k: round(v, 3) for k, v in ratios.items()}, "thresholds": [round(t, 3) for t in thresholds]}, indent=1))


if __name__ == "__main__":
    main()
