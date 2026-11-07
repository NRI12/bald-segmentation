import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from baldseg import chart, model as segformer, yolo
from baldseg.core import LEVELS, bald_ratio, label_classmap
from baldseg.metrics import SegMeter, grade_report


def evaluate(predict, args, cfg):
    report = {}
    for split in ("valid", "test"):
        meter, errors = SegMeter(), []
        for image, label, names in segformer.seg_items(args.data, split):
            img = cv2.imread(str(image))
            gt, pred = label_classmap(label, names, img.shape[:2]), predict(img)
            meter.update(gt, pred)
            errors.append(abs(bald_ratio(gt) - bald_ratio(pred)))
        report[f"segmentation_{split}"] = {**meter.report(), "ratio_mae": float(np.mean(errors))}

    rows = [r | {"ratio": bald_ratio(predict(cv2.imread(str(args.topdown / r["file"]))))} for r in csv.DictReader(open(args.topdown / "labels.csv"))]
    by_level = defaultdict(list)
    for r in rows:
        by_level[LEVELS[int(r["level"])]].append(r["ratio"])
    report["ratio_by_label"] = {k: {"median": float(np.median(v)), "mean": float(np.mean(v)), "n": len(v)} for k, v in sorted(by_level.items())}
    for split in ("valid", "test", "all"):
        sel = [r for r in rows if split == "all" or r["split"] == split]
        report[f"grading_{split}"] = grade_report([int(r["level"]) for r in sel], [chart.level(r["ratio"], cfg) for r in sel], len(LEVELS))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--topdown", type=Path, default=Path("data_topdown"))
    p.add_argument("--segformer", default="weights/segformer.pt")
    p.add_argument("--yolo", default="weights/yolo.pt")
    p.add_argument("--thresholds", default="weights/norwood_thresholds.json")
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    cfg = chart.load(args.thresholds)
    seg_model, size = segformer.load(args.segformer, "cuda" if torch.cuda.is_available() else "cpu")
    yolo_model = yolo.load(args.yolo)
    report = {
        "segformer": evaluate(lambda img: segformer.predict(seg_model, img, size), args, cfg),
        "yolo": evaluate(lambda img: yolo.predict(yolo_model, img), args, cfg),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out / "metrics.json", "w"), indent=2, default=float)
    keys = {"segmentation_test": ("dice", "iou"), "grading_test": ("accuracy", "within_1", "kappa_quadratic"), "grading_all": ("accuracy", "within_1", "kappa_quadratic")}
    for name, r in report.items():
        print(name, {k: {m: round(float(r[k]["bald"][m] if k.startswith("seg") else r[k][m]), 3) for m in ms} for k, ms in keys.items()}, "ratio_mae", round(r["segmentation_test"]["ratio_mae"], 3), flush=True)


if __name__ == "__main__":
    main()
