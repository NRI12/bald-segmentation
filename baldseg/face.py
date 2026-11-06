import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from baldseg.multitask import to_tensor

TOP = ("l_brow", "r_brow", "l_eye", "r_eye")
DOWN = ("nose", "mouth", "u_lip", "l_lip")
ALWAYS = ("l_ear", "r_ear", "ear_r", "eye_g")
LOWER = ("neck", "neck_l", "cloth")


class FaceMasker:
    def __init__(self, name="jonathandinu/face-parsing", device="cpu", size=512, min_pixels=150, grow=0.35):
        self.model = SegformerForSemanticSegmentation.from_pretrained(name).to(device).eval()
        self.ids = {v: int(k) for k, v in self.model.config.id2label.items()}
        self.device, self.size, self.min_pixels, self.grow = device, size, min_pixels, grow

    def _mask(self, lab, names):
        return np.isin(lab, [self.ids[n] for n in names])

    @torch.no_grad()
    def parse(self, img):
        logits = self.model(pixel_values=to_tensor(img, self.size)[None].to(self.device)).logits
        return F.interpolate(logits, size=img.shape[:2], mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()

    def __call__(self, img):
        lab = self.parse(img)
        mask = self._mask(lab, ALWAYS)
        top, down = self._mask(lab, TOP), self._mask(lab, DOWN)
        if top.sum() < self.min_pixels or down.sum() < self.min_pixels:
            return mask
        ys, xs = np.nonzero(top | down)
        radius = int(self.grow * max(np.ptp(xs), np.ptp(ys)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        blob = cv2.dilate((top | down).astype(np.uint8), kernel).astype(bool)
        eyes = self._mask(lab, ("l_eye", "r_eye"))
        ty, tx = np.nonzero(top)
        dy, dx = np.nonzero(down)
        brow, mouth = np.array([tx.mean(), ty.mean()]), np.array([dx.mean(), dy.mean()])
        direction = mouth - brow
        level = brow + (0.35 * direction if eyes.sum() < self.min_pixels else (np.array(np.nonzero(eyes)[::-1]).mean(1) - brow))
        yy, xx = np.mgrid[: lab.shape[0], : lab.shape[1]]
        proj = (xx - level[0]) * direction[0] + (yy - level[1]) * direction[1]
        chin = proj >= (mouth - level) @ direction
        skin = self._mask(lab, ("skin",)) & (proj >= 0)
        beard = self._mask(lab, ("hair",)) & chin
        return mask | top | down | (self._mask(lab, LOWER) & chin) | (blob & (skin | beard))


def remove_face(cmap, face_mask):
    out = cmap.copy()
    out[face_mask] = 0
    return out
