import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import spearmanr

from baldseg.core import GRADES, bald_ratio, label_classmap
from baldseg.metrics import SegMeter, grade_report, icc
from baldseg.multitask import grade_items, load, predict, seg_items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights_b/stage2.pt")
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--grades", default="data_grade")
    p.add_argument("--split", default="test")
    p.add_argument("--mc", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    model, size = load(args.weights, "cuda" if torch.cuda.is_available() else "cpu")
    meter, gt_r, mask_r, head_r, retest_r, unc, err = SegMeter(), [], [], [], [], [], []
    for it in seg_items(args.data, args.split):
        img = cv2.imread(str(it["image"]))
        gt = label_classmap(it["label"], it["names"], img.shape[:2])
        out = predict(model, img, size, args.mc)
        retest = predict(model, cv2.convertScaleAbs(cv2.flip(img, 1), alpha=1.1, beta=10), size, args.mc)
        meter.update(gt, out["cmap"])
        gt_r.append(bald_ratio(gt))
        mask_r.append(bald_ratio(out["cmap"]))
        head_r.append(out["ratio_head"])
        retest_r.append(bald_ratio(retest["cmap"]))
        unc.append(float(out["uncertainty"].mean()))
        err.append(float((gt != out["cmap"]).mean()))

    items = grade_items(args.grades, args.split)
    gt_g, pred_g = [], []
    for it in items:
        gt_g.append(it["grade"])
        pred_g.append(int(predict(model, cv2.imread(str(it["image"])), size, args.mc)["grade_probs"].argmax()))
    views = np.array([it["view"] for it in items])
    hit = np.array(gt_g) == np.array(pred_g)

    gt_r = np.array(gt_r)
    report = {
        "split": args.split,
        "segmentation": meter.report(),
        "ratio": {
            "mae_mask": float(np.abs(gt_r - mask_r).mean()),
            "mae_head": float(np.abs(gt_r - head_r).mean()),
            "icc_test_retest": icc(np.array(mask_r), np.array(retest_r)),
        },
        "uncertainty_error_spearman": float(spearmanr(unc, err).statistic),
        "grading": grade_report(gt_g, pred_g, len(GRADES)),
        "grading_by_view": {v: float(hit[views == v].mean()) for v in sorted(set(views))},
    }
    args.out.mkdir(exist_ok=True)
    json.dump(report, open(args.out / f"metrics_b_{args.split}.json", "w"), indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
