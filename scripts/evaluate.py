import argparse
import json
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from baldseg.core import GRADES, bald_ratio, grade, label_classmap, load_thresholds, result_classmap
from baldseg.metrics import SegMeter, grade_report, ratio_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--split", default="test")
    p.add_argument("--weights", type=Path, default=Path("weights"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.data))
    root = Path(cfg["path"])
    model = YOLO(args.weights / "best.pt")
    thresholds = load_thresholds(args.weights / "grades.json")
    meter, gt_r, pred_r, retest_r = SegMeter(), [], [], []

    for img_path in sorted((root / args.split / "images").iterdir()):
        img = cv2.imread(str(img_path))
        gt = label_classmap(root / args.split / "labels" / f"{img_path.stem}.txt", cfg["names"], img.shape[:2])
        pred = result_classmap(model(img, conf=args.conf, verbose=False)[0])
        retest = cv2.convertScaleAbs(cv2.flip(img, 1), alpha=1.1, beta=10)
        meter.update(gt, pred)
        gt_r.append(bald_ratio(gt))
        pred_r.append(bald_ratio(pred))
        retest_r.append(bald_ratio(result_classmap(model(retest, conf=args.conf, verbose=False)[0])))

    gt_g = [grade(r, thresholds) for r in gt_r]
    pred_g = [grade(r, thresholds) for r in pred_r]
    report = {
        "split": args.split,
        "images": len(gt_r),
        "segmentation": meter.report(),
        "ratio": ratio_report(gt_r, pred_r, gt_g, retest_r),
        "grading": grade_report(gt_g, pred_g, len(GRADES)),
        "thresholds": thresholds,
    }
    args.out.mkdir(exist_ok=True)
    json.dump(report, open(args.out / f"metrics_{args.split}.json", "w"), indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
