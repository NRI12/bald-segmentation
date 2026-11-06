import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from baldseg.multitask import to_tensor

TOP = ("l_brow", "r_brow", "l_eye", "r_eye", "eye_g")
DOWN = ("nose", "mouth", "u_lip", "l_lip", "neck", "neck_l")
EARS = ("l_ear", "r_ear", "ear_r")
ALWAYS = TOP + DOWN + EARS + ("cloth", "hat")


class FaceMasker:
    def __init__(self, name="jonathandinu/face-parsing", device="cpu", size=512, min_pixels=30, margin=0.03):
        self.model = SegformerForSemanticSegmentation.from_pretrained(name).to(device).eval()
        self.ids = {v: int(k) for k, v in self.model.config.id2label.items()}
        self.device, self.size, self.min_pixels, self.margin = device, size, min_pixels, margin

    def _ids(self, names):
        return [self.ids[n] for n in names]

    @torch.no_grad()
    def parse(self, img):
        logits = self.model(pixel_values=to_tensor(img, self.size)[None].to(self.device)).logits
        return F.interpolate(logits, size=img.shape[:2], mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()

    def __call__(self, img):
        lab = self.parse(img)
        mask = np.isin(lab, self._ids(ALWAYS))
        top, down = np.isin(lab, self._ids(TOP)), np.isin(lab, self._ids(DOWN))
        if top.sum() < self.min_pixels or down.sum() < self.min_pixels:
            return mask
        ty, tx = np.nonzero(top)
        dy, dx = np.nonzero(down)
        origin = np.array([tx.mean(), ty.mean()])
        direction = np.array([dx.mean(), dy.mean()]) - origin
        parts = np.isin(lab, self._ids(TOP + DOWN + EARS)).astype(np.uint8)
        hull = np.zeros(lab.shape, np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(parts)), 1)
        k = max(3, int(self.margin * min(lab.shape)))
        hull = cv2.dilate(hull, np.ones((k, k), np.uint8)).astype(bool)
        yy, xx = np.mgrid[: lab.shape[0], : lab.shape[1]]
        below = (xx - origin[0]) * direction[0] + (yy - origin[1]) * direction[1] >= 0
        return mask | (hull & below)


def remove_face(cmap, face_mask):
    out = cmap.copy()
    out[face_mask] = 0
    return out
