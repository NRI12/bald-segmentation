import argparse
import json
import shutil
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from baldseg.core import bald_ratio, fit_thresholds, label_classmap


def train_ratios(data):
    cfg = yaml.safe_load(open(data))
    root = Path(cfg["path"])
    ratios = []
    for img_path in sorted((root / cfg["train"]).iterdir()):
        shape = cv2.imread(str(img_path)).shape[:2]
        ratios.append(bald_ratio(label_classmap(root / "train/labels" / f"{img_path.stem}.txt", cfg["names"], shape)))
    return ratios


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--model", default="yolo11m-seg.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--grade-method", choices=["quantile", "kmeans"], default="quantile")
    p.add_argument("--out", type=Path, default=Path("weights"))
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device,
        patience=40, seed=0, degrees=90, flipud=0.5, fliplr=0.5, project="runs", name="seg", exist_ok=True,
    )
    args.out.mkdir(exist_ok=True)
    shutil.copy(Path(model.trainer.save_dir) / "weights/best.pt", args.out / "best.pt")
    thresholds = fit_thresholds(train_ratios(args.data), args.grade_method)
    json.dump({"method": args.grade_method, "thresholds": thresholds}, open(args.out / "grades.json", "w"), indent=2)
    print("thresholds", thresholds)


if __name__ == "__main__":
    main()
