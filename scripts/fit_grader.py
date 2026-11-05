import argparse
import csv
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
from ultralytics import YOLO

from baldseg.core import GRADES, NORWOOD_TO_GRADE, grader_features, result_classmap
from baldseg.metrics import grade_report


def features(model, root, rows, conf):
    return np.array([grader_features(result_classmap(model(cv2.imread(str(root / r["file"])), conf=conf, verbose=False)[0]), r["view"]) for r in rows])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grades", type=Path, default=Path("data_grade"))
    p.add_argument("--weights", type=Path, default=Path("weights"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    model = YOLO(args.weights / "best.pt")
    rows = list(csv.DictReader(open(args.grades / "labels.csv")))
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "valid", "test")}
    x = {s: features(model, args.grades, r, args.conf) for s, r in split.items()}
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
    json.dump(report, open(args.out / "grading_a.json", "w"), indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
