import argparse
import csv
import shutil
from pathlib import Path

import cv2
import yaml

from baldseg.core import clahe
from baldseg.data import SPLITS, assign, dedupe_group, download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default="fifa-eg5yw")
    p.add_argument("--project", default="hairloss-segmentation-lf62l")
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--raw", type=Path, default=Path("raw"))
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not (args.raw / "data.yaml").exists():
        download(args.workspace, args.project, args.version, "yolov8", args.raw)
    names = yaml.safe_load(open(args.raw / "data.yaml"))["names"]
    items = [(img, img.parent.parent / "labels" / f"{img.stem}.txt") for img in sorted(args.raw.glob("*/images/*"))]
    groups = dedupe_group(items, lambda it: it[0].name.split(".rf.")[0])
    split_of = assign(groups, args.seed)

    shutil.rmtree(args.out, ignore_errors=True)
    rows = []
    for source_id, members in groups.items():
        split = split_of[source_id]
        for sub in ("images", "labels"):
            (args.out / split / sub).mkdir(parents=True, exist_ok=True)
        for img_path, label, img in members:
            cv2.imwrite(str(args.out / split / "images" / img_path.name), clahe(img))
            shutil.copy(label, args.out / split / "labels" / label.name)
            rows.append((img_path.name, source_id, split))

    with open(args.out / "splits.csv", "w", newline="") as f:
        csv.writer(f).writerows([("file", "source_id", "split"), *rows])
    yaml.safe_dump({"path": str(args.out.resolve()), "train": "train/images", "val": "valid/images", "test": "test/images", "names": dict(enumerate(names))}, open(args.out / "data.yaml", "w"))
    for s in SPLITS:
        print(s, sum(r[2] == s for r in rows))


if __name__ == "__main__":
    main()
