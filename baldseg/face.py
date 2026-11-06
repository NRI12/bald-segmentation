import cv2
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import SegformerForSemanticSegmentation

from baldseg.multitask import to_tensor

TOP = ("l_brow", "r_brow", "l_eye", "r_eye")
DOWN = ("nose", "mouth", "u_lip", "l_lip")
ALWAYS = ("l_ear", "r_ear", "ear_r", "eye_g")
LOWER = ("neck", "neck_l", "cloth")
SIDE_VIEWS = ("Left", "Right")


class FaceMasker:
    def __init__(self, name="jonathandinu/face-parsing", device="cpu", size=512, min_pixels=150, grow=0.35, min_score=0.8):
        self.model = SegformerForSemanticSegmentation.from_pretrained(name).to(device).eval()
        self.ids = {v: int(k) for k, v in self.model.config.id2label.items()}
        self.detector = cv2.FaceDetectorYN.create(hf_hub_download("opencv/face_detection_yunet", "face_detection_yunet_2023mar.onnx"), "", (size, size), min_score)
        self.device, self.size, self.min_pixels, self.grow = device, size, min_pixels, grow

    def _mask(self, lab, names):
        return np.isin(lab, [self.ids[n] for n in names])

    @torch.no_grad()
    def parse(self, img):
        logits = self.model(pixel_values=to_tensor(img, self.size)[None].to(self.device)).logits
        return F.interpolate(logits, size=img.shape[:2], mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()

    def _cut(self, lab, level, mouth, region):
        direction = mouth - level
        yy, xx = np.mgrid[: lab.shape[0], : lab.shape[1]]
        proj = (xx - level[0]) * direction[0] + (yy - level[1]) * direction[1]
        chin = proj >= direction @ direction
        skin = self._mask(lab, ("skin",)) & (proj >= 0)
        beard = self._mask(lab, ("hair",)) & chin
        return (self._mask(lab, LOWER) & chin) | (region & (skin | beard))

    def _from_parser(self, lab):
        top, down = self._mask(lab, TOP), self._mask(lab, DOWN)
        if top.sum() < self.min_pixels or down.sum() < self.min_pixels:
            return None
        ys, xs = np.nonzero(top | down)
        radius = int(self.grow * max(np.ptp(xs), np.ptp(ys)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        region = cv2.dilate((top | down).astype(np.uint8), kernel).astype(bool)
        brow = np.array(np.nonzero(top)[::-1]).mean(1)
        mouth = np.array(np.nonzero(down)[::-1]).mean(1)
        eyes = self._mask(lab, ("l_eye", "r_eye"))
        level = np.array(np.nonzero(eyes)[::-1]).mean(1) if eyes.sum() >= self.min_pixels else brow + 0.35 * (mouth - brow)
        return top | down | self._cut(lab, level, mouth, region)

    def _from_detector(self, img, lab):
        best = None
        for k in range(4):
            im = np.ascontiguousarray(np.rot90(img, k))
            self.detector.setInputSize((im.shape[1], im.shape[0]))
            _, faces = self.detector.detect(im)
            if faces is not None and (best is None or faces[:, -1].max() > best[1][-1]):
                best = (k, faces[faces[:, -1].argmax()])
        if best is None:
            return None
        k, face = best
        rlab = np.rot90(lab, k)
        x, y, w, h = face[:4]
        points = face[4:14].reshape(5, 2)
        region = np.zeros(rlab.shape, bool)
        region[int(max(0, y - 0.15 * h)): int(y + 1.3 * h), int(max(0, x - 0.15 * w)): int(x + 1.15 * w)] = True
        return np.rot90(self._cut(rlab, points[:2].mean(0), points[3:].mean(0), region), -k)

    def __call__(self, img, view=None):
        lab = self.parse(img)
        face = self._from_parser(lab)
        if face is None and view in SIDE_VIEWS:
            face = self._from_detector(img, lab)
        return self._mask(lab, ALWAYS) | (face if face is not None else False)


def remove_face(cmap, face_mask):
    out = cmap.copy()
    out[face_mask] = 0
    return out
