import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from baldseg import chart
from baldseg.core import LEVELS, bald_ratio, label_classmap
from baldseg.metrics import SegMeter, grade_report
from baldseg.model import load, predict, seg_items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--topdown", type=Path, default=Path("data_topdown"))
    p.add_argument("--weights", default="weights/segformer.pt")
    p.add_argument("--thresholds", default="weights/norwood_thresholds.json")
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    model, size = load(args.weights, "cuda" if torch.cuda.is_available() else "cpu")
    cfg = chart.load(args.thresholds)
    report = {}

    for split in ("valid", "test"):
        meter, errors = SegMeter(), []
        for image, label, names in seg_items(args.data, split):
            img = cv2.imread(str(image))
            gt, pred = label_classmap(label, names, img.shape[:2]), predict(model, img, size)
            meter.update(gt, pred)
            errors.append(abs(bald_ratio(gt) - bald_ratio(pred)))
        report[f"segmentation_{split}"] = {**meter.report(), "ratio_mae": float(np.mean(errors))}

    rows = [r | {"ratio": bald_ratio(predict(model, cv2.imread(str(args.topdown / r["file"])), size))} for r in csv.DictReader(open(args.topdown / "labels.csv"))]
    by_level = defaultdict(list)
    for r in rows:
        by_level[LEVELS[int(r["level"])]].append(r["ratio"])
    report["ratio_by_label"] = {k: {"median": float(np.median(v)), "mean": float(np.mean(v)), "n": len(v)} for k, v in sorted(by_level.items())}
    for split in ("valid", "test", "all"):
        sel = [r for r in rows if split == "all" or r["split"] == split]
        report[f"grading_{split}"] = grade_report([int(r["level"]) for r in sel], [chart.level(r["ratio"], cfg) for r in sel], len(LEVELS))

    args.out.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out / "metrics.json", "w"), indent=2, default=float)
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
