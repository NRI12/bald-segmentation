import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
from ultralytics import YOLO

from baldseg.core import GRADES, bald_ratio, result_classmap
from baldseg.metrics import grade_report
from baldseg.multitask import grade_items, load, predict
from baldseg.norwood_art import STAGES
from baldseg.topview import level_probs, match


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grades", default="data_grade")
    p.add_argument("--ckpt-b", default="weights_b/norwood.pt")
    p.add_argument("--weights-a", type=Path, default=Path("weights"))
    p.add_argument("--mc", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("results/top"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_b, size = load(args.ckpt_b, device)
    model_a = YOLO(args.weights_a / "best.pt")
    rows = {s: [] for s in ("train", "valid", "test")}
    for split in rows:
        for it in grade_items(args.grades, split):
            if it["view"] != "Top-Down":
                continue
            img = cv2.imread(str(it["image"]))
            out = predict(model_b, img, size, args.mc)
            row = {"grade": it["grade"], "subject": it["subject"], "head": out["grade_probs"]}
            for name, cmap in (("b", out["cmap"]), ("a", result_classmap(model_a(img, verbose=False)[0]))):
                m = match(cmap)
                probs = level_probs(m["probs"]) if m else np.eye(len(GRADES))[0]
                scores = [m["scores"][s] for s in STAGES] if m else [0.0] * len(STAGES)
                row[name] = {"level": int(probs.argmax()), "features": scores + [bald_ratio(cmap)]}
            rows[split].append(row)
        print(split, len(rows[split]), flush=True)

    report = {}
    for name in ("a", "b"):
        x = {s: np.array([r[name]["features"] for r in rs]) for s, rs in rows.items()}
        y = {s: np.array([r["grade"] for r in rs]) for s, rs in rows.items()}
        fit = lambda d: DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=0).fit(x["train"], y["train"])
        depth = max(range(1, 6), key=lambda d: cohen_kappa_score(y["valid"], fit(d).predict(x["valid"]), weights="quadratic"))
        tree = fit(depth)
        args.out.mkdir(parents=True, exist_ok=True)
        joblib.dump(tree, args.out / f"top_tree_{name}.joblib")
        for split in ("valid", "test"):
            report[f"{name}_template_{split}"] = grade_report(y[split], [r[name]["level"] for r in rows[split]], len(GRADES))
            report[f"{name}_template_tree_{split}"] = grade_report(y[split], tree.predict(x[split]), len(GRADES))
    for split in ("valid", "test"):
        report[f"b_head_{split}"] = grade_report([r["grade"] for r in rows[split]], [int(np.argmax(r["head"])) for r in rows[split]], len(GRADES))
    json.dump(report, open(args.out / "metrics_top.json", "w"), indent=2, default=float)
    print(json.dumps({k: {m: round(v, 3) for m, v in r.items() if m in ("accuracy", "within_1", "mae_stage", "kappa_quadratic")} for k, r in report.items()}, indent=1))


if __name__ == "__main__":
    main()
