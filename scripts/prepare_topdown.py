import argparse
import csv
import re
import shutil
from pathlib import Path

import cv2

from baldseg.core import NORWOOD_TO_LEVEL, clahe
from baldseg.data import assign, dedupe_group, download

SPLITS = {"train": 0.7, "valid": 0.15, "test": 0.15}


def parse(path):
    level = int(path.parent.name.split("_")[1])
    subject, view = re.match(r"(\d+)-?(.+?)_(?:jpe?g|png)\.rf\.", path.name, re.I).groups()
    return level, subject, view


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default="uze")
    p.add_argument("--project", default="hair-loss-nq8hh")
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--raw", type=Path, default=Path("raw_norwood"))
    p.add_argument("--out", type=Path, default=Path("data_topdown"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not (args.raw / "train").exists():
        download(args.workspace, args.project, args.version, "folder", args.raw)
    items = [(img, *parse(img)) for img in sorted(args.raw.glob("*/LEVEL_*/*.jpg"))]
    groups = dedupe_group(items, lambda it: f"{it[1]}-{it[2]}")
    split_of = assign(groups, args.seed, SPLITS)

    shutil.rmtree(args.out, ignore_errors=True)
    rows = []
    for subject, members in groups.items():
        split = split_of[subject]
        for img_path, level, _, view, img in members:
            if view != "Top-Down":
                continue
            (args.out / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.out / split / img_path.name), clahe(img))
            rows.append((f"{split}/{img_path.name}", split, subject, level, NORWOOD_TO_LEVEL[level]))

    with open(args.out / "labels.csv", "w", newline="") as f:
        csv.writer(f).writerows([("file", "split", "subject", "norwood", "level"), *rows])
    for s in SPLITS:
        print(s, sum(r[1] == s for r in rows))


if __name__ == "__main__":
    main()
