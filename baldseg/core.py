import cv2
import numpy as np

HAIR, BALD = 1, 2
CLASS_IDS = {"hair": HAIR, "bald": BALD}
LEVELS = ["Stage 1-2", "Stage 3", "Stage 4", "Stage 5", "Stage 6", "Stage 7"]
NORWOOD_TO_LEVEL = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def label_classmap(label_path, names, shape):
    h, w = shape
    cmap = np.zeros((h, w), np.uint8)
    rows = [line.split() for line in open(label_path).read().splitlines()]
    instances = [(CLASS_IDS[names[int(r[0])]], np.array(r[1:], float).reshape(-1, 2) * [w, h]) for r in rows if len(r) >= 7]
    for cls, poly in sorted(instances, key=lambda x: x[0]):
        if len(poly) >= 3:
            cv2.fillPoly(cmap, [np.round(poly).astype(np.int32)], cls)
    return cmap


def bald_ratio(cmap):
    bald, hair = (cmap == BALD).sum(), (cmap == HAIR).sum()
    return float(bald / (bald + hair)) if bald + hair else 0.0
