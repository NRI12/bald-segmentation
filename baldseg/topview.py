import cv2
import numpy as np

from baldseg.core import BALD
from baldseg.norwood_art import AXES, CENTER, STAGES, TO_LEVEL, stage_masks

SIZE = 256
TEMPLATES = {s: stage_masks(s, SIZE) for s in STAGES}


def head_region(cmap):
    head = (cmap > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(head)
    if n < 2:
        return None
    head = (lab == 1 + stats[1:, cv2.CC_STAT_AREA].argmax()).astype(np.uint8)
    contours, _ = cv2.findContours(head, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    return hull if len(hull) >= 5 else None


def aligned(cmap, hull, quarter):
    (cx, cy), (w, h), angle = cv2.fitEllipse(hull)
    if quarter % 2:
        w, h = h, w
    rot = cv2.getRotationMatrix2D((cx, cy), angle + 90 * quarter, 1.0)
    scale = np.diag([2 * AXES[0] * SIZE / w, 2 * AXES[1] * SIZE / h])
    m = scale @ rot[:, :2]
    t = np.array([CENTER[0] * SIZE, CENTER[1] * SIZE]) - m @ np.array([cx, cy])
    return cv2.warpAffine(cmap, np.hstack([m, t[:, None]]), (SIZE, SIZE), flags=cv2.INTER_NEAREST)


def agreement(norm, head, bald):
    pred, ref = (norm == BALD)[head], bald[head]
    iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
    return float((iou(pred, ref) + iou(~pred, ~ref)) / 2)


def match(cmap, temperature=25.0, min_bald=0.04):
    hull = head_region(cmap)
    if hull is None:
        return None
    best = None
    for quarter in range(4):
        norm = aligned(cmap, hull, quarter)
        scores = {}
        for s, (head, bald) in TEMPLATES.items():
            scores[s] = agreement(norm, head, bald)
        top = max(scores.values())
        if best is None or top > best[0]:
            best = (top, scores, norm)
    _, scores, norm = best
    if (norm == BALD).sum() < min_bald * (norm > 0).sum():
        scores = {s: (1.0 if s == "1" else v) for s, v in scores.items()}
    values = np.array([scores[s] for s in STAGES])
    probs = np.exp(temperature * (values - values.max()))
    return {"stage": max(scores, key=scores.get), "scores": scores, "probs": dict(zip(STAGES, probs / probs.sum())), "aligned": norm}


def render(norm, stage):
    head, bald = TEMPLATES[stage]
    img = np.full((SIZE, SIZE, 3), 255, np.uint8)
    img[norm == 1] = (60, 170, 60)
    img[norm == BALD] = (230, 60, 60)
    edge = cv2.morphologyEx(bald.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    outline = cv2.morphologyEx(head.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    img[outline] = (90, 90, 90)
    img[edge] = (0, 0, 0)
    return img


def level_probs(probs, levels=6):
    out = np.zeros(levels)
    for stage, p in probs.items():
        out[TO_LEVEL[stage]] += p
    return out
