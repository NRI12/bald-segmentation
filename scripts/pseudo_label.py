import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from baldseg.multitask import IGNORE, grade_items, load, predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights_b/stage1.pt")
    p.add_argument("--grades", default="data_grade")
    p.add_argument("--out", type=Path, default=Path("data_grade/pseudo"))
    p.add_argument("--pixel-conf", type=float, default=0.9)
    p.add_argument("--min-coverage", type=float, default=0.6)
    p.add_argument("--mc", type=int, default=4)
    args = p.parse_args()

    model, size = load(args.weights, "cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    items = grade_items(args.grades, "train")
    kept = 0
    for it in items:
        out = predict(model, cv2.imread(str(it["image"])), size, args.mc)
        conf = out["probs"].max(0)
        if (conf >= args.pixel_conf).mean() < args.min_coverage:
            continue
        cv2.imwrite(str(args.out / f"{it['image'].stem}.png"), np.where(conf >= args.pixel_conf, out["cmap"], IGNORE).astype(np.uint8))
        kept += 1
    print(f"pseudo labels {kept}/{len(items)}")


if __name__ == "__main__":
    main()
