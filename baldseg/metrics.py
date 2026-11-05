import numpy as np
from scipy.stats import spearmanr
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
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
        "kappa_quadratic": cohen_kappa_score(gt, pred, labels=range(levels), weights="quadratic"),
        "confusion_matrix": confusion_matrix(gt, pred, labels=range(levels)).tolist(),
    }


def ratio_report(gt_r, pred_r, gt_grades, retest_r):
    gt_r, pred_r = np.asarray(gt_r), np.asarray(pred_r)
    return {
        "mae": float(np.abs(gt_r - pred_r).mean()),
        "spearman_vs_grade": float(spearmanr(pred_r, gt_grades).statistic),
        "icc_test_retest": icc(pred_r, np.asarray(retest_r)),
    }


def icc(a, b):
    x = np.stack([a, b], 1)
    n, k = x.shape
    m = x.mean()
    msr = k * ((x.mean(1) - m) ** 2).sum() / (n - 1)
    mse = ((x - x.mean(1, keepdims=True) - x.mean(0) + m) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse))
