import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from baldseg.core import VIEWS


def sheet(rows, title, tile=220):
    tiles = []
    for v in VIEWS:
        r = next((r for r in rows if r["view"] == v), None)
        img = cv2.resize(cv2.imread(r["path"]), (tile, tile)) if r else np.full((tile, tile, 3), 235, np.uint8)
        cv2.putText(img, v, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    bar = np.full((36, tile * len(VIEWS), 3), 255, np.uint8)
    cv2.putText(bar, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    return np.vstack([bar, np.hstack(tiles)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, default=Path("results/cv/report.json"))
    p.add_argument("--grades", type=Path, default=Path("data_grade"))
    p.add_argument("--out", type=Path, default=Path("results/cv/suspects"))
    args = p.parse_args()

    by_subject = {}
    for r in csv.DictReader(open(args.grades / "labels.csv")):
        by_subject.setdefault(r["subject"], []).append({**r, "path": str(args.grades / r["file"])})
    args.out.mkdir(parents=True, exist_ok=True)
    suspects = json.load(open(args.report))["suspects"]
    with open(args.out / "suspects.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "label", "pred", "conf", "images", "sheet", "corrected_norwood"])
        for s in suspects:
            name = f"{s['subject']}.jpg"
            title = f"{s['subject']}: label {s['label']} | model {s['pred']} ({s['conf']:.0%})"
            cv2.imwrite(str(args.out / name), sheet(by_subject[s["subject"]], title))
            w.writerow([s["subject"], s["label"], s["pred"], f"{s['conf']:.2f}", s["images"], name, ""])
    print(f"{len(suspects)} suspects -> {args.out}")


if __name__ == "__main__":
    main()
