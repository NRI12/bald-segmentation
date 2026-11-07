from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from baldseg.core import HAIR, BALD, label_classmap

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build(backbone, pretrained=True):
    if pretrained:
        return SegformerForSemanticSegmentation.from_pretrained(backbone, num_labels=3)
    return SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(backbone, num_labels=3))


def save(model, path, backbone, size):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": backbone, "size": size, "state": model.state_dict()}, path)


def load(path, device):
    ckpt = torch.load(path, map_location=device)
    model = build(ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval(), ckpt["size"]


def to_tensor(img, size):
    x = cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))


def seg_items(data_yaml, split):
    cfg = yaml.safe_load(open(data_yaml))
    root = Path(cfg["path"]) / split
    return [(p, root / "labels" / f"{p.stem}.txt", cfg["names"]) for p in sorted((root / "images").iterdir())]


class Samples(Dataset):
    def __init__(self, items, size, train):
        self.items, self.size, self.train = items, size, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        image, label, names = self.items[i]
        img = cv2.imread(str(image))
        mask = label_classmap(label, names, img.shape[:2])
        if self.train:
            k = np.random.randint(4)
            img, mask = np.rot90(img, k), np.rot90(mask, k)
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            img = cv2.convertScaleAbs(np.ascontiguousarray(img), alpha=np.random.uniform(0.8, 1.2), beta=np.random.uniform(-20, 20))
        mask = cv2.resize(np.ascontiguousarray(mask), (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return to_tensor(np.ascontiguousarray(img), self.size), torch.from_numpy(mask).long()


def seg_loss(logits, mask, boundary_weight=2.0, k=5):
    logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    m = mask[:, None].float()
    edge = (F.max_pool2d(m, k, 1, k // 2) != -F.max_pool2d(-m, k, 1, k // 2)).squeeze(1).float()
    ce = (F.cross_entropy(logits, mask, reduction="none") * (1 + boundary_weight * edge)).mean()
    prob = logits.softmax(1)
    onehot = F.one_hot(mask, 3).permute(0, 3, 1, 2)
    inter, denom = (prob * onehot).sum((0, 2, 3)), (prob + onehot).sum((0, 2, 3))
    return ce + 1 - ((2 * inter + 1) / (denom + 1))[1:].mean()


def postprocess(cmap, min_frac=0.002, k=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = np.zeros_like(cmap)
    for cls in (HAIR, BALD):
        m = cv2.morphologyEx(cv2.morphologyEx((cmap == cls).astype(np.uint8), cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        _, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= min_frac * cmap.size)
        out[np.isin(lab, keep[keep > 0])] = cls
    return out


@torch.no_grad()
def predict(model, img, size):
    x = to_tensor(img, size)[None].to(next(model.parameters()).device)
    logits = model(pixel_values=torch.cat([x, x.flip(-1)])).logits
    prob = (logits[0].softmax(0) + logits[1].flip(-1).softmax(0)) / 2
    prob = F.interpolate(prob[None], size=img.shape[:2], mode="bilinear", align_corners=False)[0]
    return postprocess(prob.argmax(0).cpu().numpy().astype(np.uint8))
