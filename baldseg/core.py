import json

import cv2
import numpy as np

HAIR, BALD = 1, 2
CLASS_IDS = {"hair": HAIR, "bald": BALD}
GRADES = ["Stage 1-2", "Stage 3", "Stage 4", "Stage 5", "Stage 6", "Stage 7"]
NORWOOD_TO_GRADE = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
VIEWS = ["Front", "Top-Down", "Back", "Left", "Right"]


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def fill(cmap, poly, cls):
    if len(poly) >= 3:
        cv2.fillPoly(cmap, [np.round(poly).astype(np.int32)], cls)


def label_classmap(label_path, names, shape):
    h, w = shape
    cmap = np.zeros((h, w), np.uint8)
    rows = [line.split() for line in open(label_path).read().splitlines()]
    instances = [(CLASS_IDS[names[int(r[0])]], np.array(r[1:], float).reshape(-1, 2) * [w, h]) for r in rows if len(r) >= 7]
    for cls, poly in sorted(instances, key=lambda x: x[0]):
        fill(cmap, poly, cls)
    return cmap


def result_classmap(result):
    cmap = np.zeros(result.orig_shape, np.uint8)
    if result.masks is None:
        return cmap
    cls = result.boxes.cls.cpu().numpy().astype(int)
    for i in result.boxes.conf.cpu().numpy().argsort():
        fill(cmap, result.masks.xy[i], CLASS_IDS[result.names[cls[i]]])
    return cmap


def predict(model, img, conf=0.25):
    return result_classmap(model(clahe(img), conf=conf, verbose=False)[0])


def bald_ratio(cmap):
    bald, hair = (cmap == BALD).sum(), (cmap == HAIR).sum()
    return float(bald / (bald + hair)) if bald + hair else 0.0


def grader_features(cmap, view):
    n = cmap.size
    return [bald_ratio(cmap), (cmap == BALD).sum() / n, (cmap == HAIR).sum() / n, *(float(v == view) for v in VIEWS)]


def has_vertex(cmap, min_frac=0.01, ring=15, hair_frac=0.8):
    h, w = cmap.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats((cmap == BALD).astype(np.uint8))
    kernel = np.ones((ring, ring), np.uint8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_frac * cmap.size or x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        comp = (lab == i).astype(np.uint8)
        border = cv2.dilate(comp, kernel).astype(bool) & ~comp.astype(bool)
        if (cmap[border] == HAIR).mean() >= hair_frac:
            return True
    return False


def stage_label(level, cmap, view):
    vertex = level == 1 and view == "Top-Down" and has_vertex(cmap)
    return ("Stage 3 Vertex" if vertex else GRADES[level]), vertex


def fit_thresholds(ratios, method="quantile", levels=len(GRADES)):
    r = np.asarray(ratios, float)
    if method == "quantile":
        return np.quantile(r, np.linspace(0, 1, levels + 1)[1:-1]).tolist()
    from sklearn.cluster import KMeans
    centers = np.sort(KMeans(levels, n_init=10, random_state=0).fit(r[:, None]).cluster_centers_.ravel())
    return ((centers[1:] + centers[:-1]) / 2).tolist()


def grade(ratio, thresholds):
    return int(np.searchsorted(thresholds, ratio, side="right"))


def load_thresholds(path):
    return json.load(open(path))["thresholds"]
