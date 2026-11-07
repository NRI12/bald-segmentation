import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--model", default="yolo11m-seg.pt")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", type=Path, default=Path("weights/yolo.pt"))
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, patience=40, seed=0,
                degrees=90, flipud=0.5, fliplr=0.5, project="runs", name="yolo", exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(model.trainer.save_dir) / "weights/best.pt", args.out)


if __name__ == "__main__":
    main()
