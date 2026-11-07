import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from baldseg import chart
from baldseg.core import GRADES, bald_ratio, result_classmap
from baldseg.metrics import grade_report
from baldseg.multitask import grade_items, load, predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grades", default="data_grade")
    p.add_argument("--ckpt-b", default="weights_b/norwood.pt")
    p.add_argument("--weights-a", type=Path, default=Path("weights"))
    p.add_argument("--thresholds", default="weights/norwood_thresholds.json")
    p.add_argument("--mc", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("results/top"))
    args = p.parse_args()

    cfg = chart.load(args.thresholds)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_b, size = load(args.ckpt_b, device)
    model_a = YOLO(args.weights_a / "best.pt")
    rows = []
    for split in ("train", "valid", "test"):
        for it in grade_items(args.grades, split):
            if it["view"] == "Top-Down":
                img = cv2.imread(str(it["image"]))
                ratio_b = bald_ratio(predict(model_b, img, size, args.mc)["cmap"])
                ratio_a = bald_ratio(result_classmap(model_a(img, verbose=False)[0]))
                rows.append({"split": split, "grade": it["grade"], "a": ratio_a, "b": ratio_b})

    report = {"thresholds": cfg["thresholds"], "stages": cfg["stages"]}
    for name in ("a", "b"):
        by_grade = defaultdict(list)
        for r in rows:
            by_grade[GRADES[r["grade"]]].append(r[name])
        report[f"{name}_ratio_by_label"] = {g: {"mean": float(np.mean(v)), "median": float(np.median(v)), "n": len(v)} for g, v in sorted(by_grade.items())}
        for split in ("test", "valid", "all"):
            sel = [r for r in rows if split == "all" or r["split"] == split]
            report[f"{name}_{split}"] = grade_report([r["grade"] for r in sel], [chart.level(r[name], cfg) for r in sel], len(GRADES))
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out / "metrics_top.json", "w"), indent=2, default=float)
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
