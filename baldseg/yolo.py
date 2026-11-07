import cv2
import numpy as np

from baldseg.core import CLASS_IDS


def load(path):
    from ultralytics import YOLO
    return YOLO(path)


def predict(model, img, conf=0.25):
    result = model(img, conf=conf, verbose=False)[0]
    cmap = np.zeros(result.orig_shape, np.uint8)
    if result.masks is None:
        return cmap
    cls = result.boxes.cls.cpu().numpy().astype(int)
    for i in result.boxes.conf.cpu().numpy().argsort():
        poly = result.masks.xy[i]
        if len(poly) >= 3:
            cv2.fillPoly(cmap, [np.round(poly).astype(np.int32)], CLASS_IDS[result.names[cls[i]]])
    return cmap
