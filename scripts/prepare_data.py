import argparse
import csv
import os
import random
import shutil
from pathlib import Path

import cv2
import yaml
from roboflow import Roboflow

from baldseg.core import clahe

SPLITS = {"train": 0.8, "valid": 0.1, "test": 0.1}


def download(args):
    project = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"]).workspace(args.workspace).project(args.project)
    return Path(project.version(args.version).download("yolov8", location=str(args.raw), overwrite=True).location)


def dhash(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8))
    return (g[:, 1:] > g[:, :-1]).tobytes()


def collect(raw):
    groups, seen = {}, set()
    for img_path in sorted(raw.glob("*/images/*")):
        img = cv2.imread(str(img_path))
        h = dhash(img)
        if h in seen:
            continue
        seen.add(h)
        label = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        groups.setdefault(img_path.name.split(".rf.")[0], []).append((img_path, label, img))
    return groups


def assign(groups, seed):
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    total, done, out = sum(map(len, groups.values())), 0, {}
    bounds = [(s, sum(list(SPLITS.values())[: i + 1])) for i, s in enumerate(SPLITS)]
    for k in keys:
        out[k] = next(s for s, b in bounds if done / total < b)
        done += len(groups[k])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default="fifa-eg5yw")
    p.add_argument("--project", default="hairloss-segmentation-lf62l")
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--raw", type=Path, default=Path("raw"))
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    raw = args.raw if (args.raw / "data.yaml").exists() else download(args)
    names = yaml.safe_load(open(raw / "data.yaml"))["names"]
    groups = collect(raw)
    split_of = assign(groups, args.seed)

    shutil.rmtree(args.out, ignore_errors=True)
    rows = []
    for source_id, items in groups.items():
        split = split_of[source_id]
        for img_path, label, img in items:
            for sub in ("images", "labels"):
                (args.out / split / sub).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.out / split / "images" / img_path.name), clahe(img))
            shutil.copy(label, args.out / split / "labels" / label.name)
            rows.append((img_path.name, source_id, split))

    with open(args.out / "splits.csv", "w", newline="") as f:
        csv.writer(f).writerows([("file", "source_id", "split"), *rows])
    root = args.out.resolve()
    yaml.safe_dump({"path": str(root), "train": "train/images", "val": "valid/images", "test": "test/images", "names": dict(enumerate(names))}, open(args.out / "data.yaml", "w"))
    for s in SPLITS:
        print(s, sum(r[2] == s for r in rows))


if __name__ == "__main__":
    main()
