import argparse
import csv
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
from ultralytics import YOLO

from baldseg.core import GRADES, NORWOOD_TO_GRADE, grader_features, result_classmap
from baldseg.face import FaceMasker, remove_face
from baldseg.metrics import grade_report


def classmap(model, img, conf, masker):
    cmap = result_classmap(model(img, conf=conf, verbose=False)[0])
    return remove_face(cmap, masker(img)) if masker else cmap


def features(model, root, rows, conf, masker):
    return np.array([grader_features(classmap(model, cv2.imread(str(root / r["file"])), conf, masker), r["view"]) for r in rows])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grades", type=Path, default=Path("data_grade"))
    p.add_argument("--weights", type=Path, default=Path("weights"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--face", action="store_true")
    args = p.parse_args()

    masker = FaceMasker(device="cuda" if torch.cuda.is_available() else "cpu") if args.face else None
    model = YOLO(args.weights / "best.pt")
    rows = list(csv.DictReader(open(args.grades / "labels.csv")))
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "valid", "test")}
    x = {s: features(model, args.grades, r, args.conf, masker) for s, r in split.items()}
    y = {s: np.array([NORWOOD_TO_GRADE[int(r["norwood"])] for r in rs]) for s, rs in split.items()}

    def fit(depth):
        return DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=0).fit(x["train"], y["train"])

    scores = {d: cohen_kappa_score(y["valid"], fit(d).predict(x["valid"]), weights="quadratic") for d in range(1, 7)}
    depth = max(scores, key=scores.get)
    tree = fit(depth)
    joblib.dump(tree, args.weights / "grader.joblib")

    test_views = np.array([r["view"] for r in split["test"]])
    pred = tree.predict(x["test"])
    report = {
        "depth": depth,
        "valid_kappa_by_depth": scores,
        "tree": grade_report(y["test"], pred, len(GRADES)),
        "tree_by_view": {v: float((pred[test_views == v] == y["test"][test_views == v]).mean()) for v in sorted(set(test_views))},
    }
    args.out.mkdir(exist_ok=True)
    json.dump(report, open(args.out / ("grading_a_face.json" if args.face else "grading_a.json"), "w"), indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
