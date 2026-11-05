import os
import random

import cv2
from roboflow import Roboflow

SPLITS = {"train": 0.8, "valid": 0.1, "test": 0.1}


def download(workspace, project, version, fmt, location):
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    return rf.workspace(workspace).project(project).version(version).download(fmt, location=str(location), overwrite=True).location


def dhash(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8))
    return (g[:, 1:] > g[:, :-1]).tobytes()


def dedupe_group(items, key):
    groups, seen = {}, set()
    for item in items:
        img = cv2.imread(str(item[0]))
        h = dhash(img)
        if h not in seen:
            seen.add(h)
            groups.setdefault(key(item), []).append((*item, img))
    return groups


def assign(groups, seed, splits=SPLITS):
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    total, done, out = sum(map(len, groups.values())), 0, {}
    bounds, acc = [], 0.0
    for s, frac in splits.items():
        acc += frac
        bounds.append((s, acc))
    for k in keys:
        out[k] = next(s for s, b in bounds if done / total < b)
        done += len(groups[k])
    return out
