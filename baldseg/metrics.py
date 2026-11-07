import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

from baldseg.core import CLASS_IDS


class SegMeter:
    def __init__(self):
        self.counts = {name: np.zeros(3) for name in CLASS_IDS}

    def update(self, gt, pred):
        for name, c in CLASS_IDS.items():
            g, p = gt == c, pred == c
            self.counts[name] += [(g & p).sum(), (~g & p).sum(), (g & ~p).sum()]

    def report(self):
        out = {}
        for name, (tp, fp, fn) in self.counts.items():
            out[name] = {
                "dice": 2 * tp / max(2 * tp + fp + fn, 1),
                "iou": tp / max(tp + fp + fn, 1),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
            }
        out["mean"] = {k: float(np.mean([out[n][k] for n in CLASS_IDS])) for k in out["hair"]}
        return out


def grade_report(gt, pred, levels):
    p, r, f1, _ = precision_recall_fscore_support(gt, pred, labels=range(levels), average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(gt, pred),
        "within_1": float(np.mean(np.abs(np.asarray(gt) - np.asarray(pred)) <= 1)),
        "mae_stage": float(np.mean(np.abs(np.asarray(gt) - np.asarray(pred)))),
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
        "kappa_quadratic": cohen_kappa_score(gt, pred, labels=range(levels), weights="quadratic"),
        "confusion_matrix": confusion_matrix(gt, pred, labels=range(levels)).tolist(),
    }
