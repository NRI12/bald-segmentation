import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

from baldseg.core import GRADES
from baldseg.metrics import grade_report
from baldseg.multitask import MultiTask, grade_items, seg_items, to_tensor
from scripts.train_b import fit


def folds(items, k, seed):
    by_level = defaultdict(set)
    for it in items:
        by_level[it["grade"]].add(it["subject"])
    fold_of, rng = {}, random.Random(seed)
    for subjects in by_level.values():
        subjects = sorted(subjects)
        rng.shuffle(subjects)
        fold_of.update({s: i % k for i, s in enumerate(subjects)})
    return fold_of


@torch.no_grad()
def grade_probs(model, img, size, device):
    model.eval()
    x = to_tensor(img, size)[None].to(device)
    return torch.cat([model(t)[2].softmax(1) for t in (x, x.flip(-1))]).mean(0).cpu().numpy()


def subject_report(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["subject"]].append(r)
    subjects = []
    for s, rs in groups.items():
        probs = np.mean([r["probs"] for r in rs], axis=0)
        subjects.append({"subject": s, "label": rs[0]["grade"], "pred": int(probs.argmax()), "conf": float(probs.max()), "images": len(rs)})
    return subjects


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--grades", default="data_grade")
    p.add_argument("--backbone", default="nvidia/mit-b2")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results/cv"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = [it for s in ("train", "valid", "test") for it in grade_items(args.grades, s)]
    seg = seg_items(args.data, "train") + seg_items(args.data, "valid")
    fold_of = folds(items, args.folds, args.seed)

    rows = []
    for k in range(args.folds):
        train = [it for it in items if fold_of[it["subject"]] != k]
        held = [it for it in items if fold_of[it["subject"]] == k]
        model = MultiTask(args.backbone).to(device)
        fit(model, seg + train, args.size, args.epochs, args.batch, args.lr, args.workers, device,
            lambda e, loss: print(f"fold {k + 1}/{args.folds} epoch {e + 1}/{args.epochs} loss {loss:.4f}", flush=True))
        for it in held:
            probs = grade_probs(model, cv2.imread(str(it["image"])), args.size, device)
            rows.append({"file": it["image"].name, "subject": it["subject"], "view": it["view"], "grade": it["grade"], "probs": probs.tolist()})
        fold_rows = [r for r in rows if fold_of[r["subject"]] == k]
        print(f"fold {k + 1} image accuracy {np.mean([np.argmax(r['probs']) == r['grade'] for r in fold_rows]):.3f}", flush=True)

    y = [r["grade"] for r in rows]
    pred = [int(np.argmax(r["probs"])) for r in rows]
    subjects = subject_report(rows)
    sy, sp = [s["label"] for s in subjects], [s["pred"] for s in subjects]
    suspects = sorted((s for s in subjects if abs(s["pred"] - s["label"]) >= 2 or (s["pred"] != s["label"] and s["conf"] >= 0.6)),
                      key=lambda s: (-abs(s["pred"] - s["label"]), -s["conf"]))
    report = {
        "folds": args.folds, "images": len(rows), "subjects": len(subjects),
        "image": grade_report(y, pred, len(GRADES)),
        "image_by_view": {v: float(np.mean([a == b for a, b, r in zip(y, pred, rows) if r["view"] == v])) for v in sorted({r["view"] for r in rows})},
        "subject": grade_report(sy, sp, len(GRADES)),
        "subject_kappa": cohen_kappa_score(sy, sp, weights="quadratic"),
        "suspects": [{**s, "label": GRADES[s["label"]], "pred": GRADES[s["pred"]]} for s in suspects],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out / "oof.json", "w"))
    json.dump(report, open(args.out / "report.json", "w"), indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
