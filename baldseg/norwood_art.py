import cv2
import numpy as np

FRONT = {2: (0.03, 0.07), 3: (0.06, 0.15), 4: (0.10, 0.19), 5: (0.18, 0.24)}
VERTEX = {3: 0.07, 4: 0.10, 5: 0.17}
BAND = {6: (0.55, 0.30), 7: (0.78, 0.45)}
HAIR, SKIN, LINE = (55, 55, 55), (255, 255, 255), (120, 120, 120)


def illustration(stage, vertex=False, size=320):
    yy, xx = np.mgrid[:size, :size] / size
    head_eq = ((xx - 0.5) / 0.34) ** 2 + ((yy - 0.55) / 0.40) ** 2
    head = head_eq <= 1
    if stage in BAND:
        inner, top = BAND[stage]
        bald = head & ~((head_eq > inner) & (yy > top))
    else:
        center, temple = FRONT[stage]
        forelock = 0.6 * center * np.exp(-(((xx - 0.5) / 0.07) ** 2))
        bald = head & (yy < 0.15 + center + temple * ((xx - 0.5) / 0.34) ** 2 - forelock)
        radius = VERTEX.get(stage) if stage != 3 or vertex else None
        if radius:
            bald |= head & ((xx - 0.5) ** 2 + (yy - 0.62) ** 2 <= radius ** 2)
    img = np.full((size, size, 3), 255, np.uint8)
    img[head] = HAIR
    img[bald] = SKIN
    s = size
    cv2.ellipse(img, (s // 2, int(0.55 * s)), (int(0.34 * s), int(0.40 * s)), 0, 0, 360, LINE, 2, cv2.LINE_AA)
    cv2.fillConvexPoly(img, np.array([[s // 2 - 10, int(0.155 * s)], [s // 2 + 10, int(0.155 * s)], [s // 2, int(0.11 * s)]]), LINE)
    label = f"Stage {stage}" + (" Vertex" if vertex else "")
    cv2.putText(img, label, (10, s - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 200), 2, cv2.LINE_AA)
    cv2.putText(img, "front", (s // 2 - 22, int(0.09 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, LINE, 1, cv2.LINE_AA)
    return img
