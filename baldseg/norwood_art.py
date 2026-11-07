import cv2
import numpy as np

STAGES = ["1", "2", "3", "3V", "4", "5", "6", "7"]
TO_LEVEL = {"1": 0, "2": 0, "3": 1, "3V": 1, "4": 2, "5": 3, "6": 4, "7": 5}
FRONT = {"1": (0.01, 0.03), "2": (0.03, 0.07), "3": (0.06, 0.15), "3V": (0.06, 0.15), "4": (0.10, 0.19), "5": (0.18, 0.24)}
VERTEX = {"3V": 0.07, "4": 0.10, "5": 0.17}
BAND = {"6": (0.55, 0.30), "7": (0.78, 0.45)}
CENTER, AXES = (0.5, 0.55), (0.34, 0.40)
HAIR, SKIN, LINE = (55, 55, 55), (255, 255, 255), (120, 120, 120)


def stage_masks(stage, size=256):
    yy, xx = np.mgrid[:size, :size] / size
    head_eq = ((xx - CENTER[0]) / AXES[0]) ** 2 + ((yy - CENTER[1]) / AXES[1]) ** 2
    head = head_eq <= 1
    if stage in BAND:
        inner, top = BAND[stage]
        return head, head & ~((head_eq > inner) & (yy > top))
    center, temple = FRONT[stage]
    forelock = 0.6 * center * np.exp(-(((xx - 0.5) / 0.07) ** 2))
    bald = head & (yy < 0.15 + center + temple * ((xx - 0.5) / AXES[0]) ** 2 - forelock)
    if stage in VERTEX:
        bald |= head & ((xx - 0.5) ** 2 + (yy - 0.62) ** 2 <= VERTEX[stage] ** 2)
    return head, bald


def illustration(stage, size=320):
    head, bald = stage_masks(stage, size)
    img = np.full((size, size, 3), 255, np.uint8)
    img[head] = HAIR
    img[bald] = SKIN
    s = size
    cv2.ellipse(img, (s // 2, int(CENTER[1] * s)), (int(AXES[0] * s), int(AXES[1] * s)), 0, 0, 360, LINE, 2, cv2.LINE_AA)
    cv2.fillConvexPoly(img, np.array([[s // 2 - 10, int(0.155 * s)], [s // 2 + 10, int(0.155 * s)], [s // 2, int(0.11 * s)]]), LINE)
    label = f"Stage {stage[0]}" + (" Vertex" if stage.endswith("V") else "")
    cv2.putText(img, label, (10, s - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 200), 2, cv2.LINE_AA)
    cv2.putText(img, "front", (s // 2 - 22, int(0.09 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, LINE, 1, cv2.LINE_AA)
    return img
