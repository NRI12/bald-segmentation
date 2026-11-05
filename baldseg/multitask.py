import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from baldseg.core import BALD, GRADES, HAIR, NORWOOD_TO_GRADE, bald_ratio, label_classmap

IGNORE = 255
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class MultiTask(nn.Module):
    def __init__(self, backbone, pretrained=True, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.seg = (SegformerForSemanticSegmentation.from_pretrained(backbone, num_labels=3) if pretrained
                    else SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(backbone, num_labels=3)))
        dim = self.seg.config.hidden_sizes[-1]
        self.drop = nn.Dropout(dropout)
        self.ratio = nn.Linear(dim, 1)
        self.grade = nn.Linear(dim, len(GRADES))

    def forward(self, x):
        out = self.seg(pixel_values=x, output_hidden_states=True)
        feat = self.drop(out.hidden_states[-1].mean((2, 3)))
        logits = F.interpolate(out.logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits, torch.sigmoid(self.ratio(feat)).squeeze(1), self.grade(feat)


def save(model, path, size):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": model.backbone, "size": size, "state": model.state_dict()}, path)


def load(path, device):
    ckpt = torch.load(path, map_location=device)
    model = MultiTask(ckpt["backbone"], pretrained=False)
    state = model.state_dict()
    model.load_state_dict({k: v for k, v in ckpt["state"].items() if k in state and v.shape == state[k].shape}, strict=False)
    return model.to(device).eval(), ckpt["size"]


def to_tensor(img, size):
    x = cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))


def seg_items(data_yaml, split):
    cfg = yaml.safe_load(open(data_yaml))
    root = Path(cfg["path"]) / split
    return [{"image": p, "label": root / "labels" / f"{p.stem}.txt", "names": cfg["names"]} for p in sorted((root / "images").iterdir())]


def grade_items(grade_dir, split, pseudo_dir=None):
    items = []
    for r in csv.DictReader(open(Path(grade_dir) / "labels.csv")):
        if r["split"] == split:
            png = pseudo_dir and Path(pseudo_dir) / f"{Path(r['file']).stem}.png"
            items.append({"image": Path(grade_dir) / r["file"], "mask": png if png and png.exists() else None, "grade": NORWOOD_TO_GRADE[int(r["norwood"])], "view": r["view"]})
    return items


class Samples(Dataset):
    def __init__(self, items, size, train):
        self.items, self.size, self.train = items, size, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.imread(str(it["image"]))
        if "label" in it:
            mask = label_classmap(it["label"], it["names"], img.shape[:2])
        elif it.get("mask"):
            mask = cv2.resize(cv2.imread(str(it["mask"]), cv2.IMREAD_GRAYSCALE), img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.full(img.shape[:2], IGNORE, np.uint8)
        known = mask != IGNORE
        ratio = bald_ratio(np.where(known, mask, 0)) if known.any() else float("nan")
        if self.train:
            img, mask = augment(img, mask)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return to_tensor(img, self.size), torch.from_numpy(mask).long(), torch.tensor(ratio), torch.tensor(it.get("grade", -1))


def augment(img, mask):
    k = np.random.randint(4)
    img, mask = np.rot90(img, k), np.rot90(mask, k)
    if np.random.rand() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    alpha, beta = np.random.uniform(0.8, 1.2), np.random.uniform(-20, 20)
    return cv2.convertScaleAbs(np.ascontiguousarray(img), alpha=alpha, beta=beta), np.ascontiguousarray(mask)


def boundary(mask, k=5):
    m = mask[:, None].float()
    return (F.max_pool2d(m, k, 1, k // 2) != -F.max_pool2d(-m, k, 1, k // 2)).squeeze(1).float()


def seg_loss(logits, mask, boundary_weight=2.0):
    valid = mask != IGNORE
    if not valid.any():
        return logits.sum() * 0
    ce = F.cross_entropy(logits, mask, ignore_index=IGNORE, reduction="none")
    ce = (ce * (1 + boundary_weight * boundary(mask)))[valid].mean()
    v = valid[:, None].float()
    prob = logits.softmax(1) * v
    onehot = F.one_hot(mask.clamp(max=2), 3).permute(0, 3, 1, 2) * v
    inter, denom = (prob * onehot).sum((0, 2, 3)), (prob + onehot).sum((0, 2, 3))
    return ce + 1 - ((2 * inter + 1) / (denom + 1))[1:].mean()


def soft_ratio(logits):
    p = logits.softmax(1)
    bald, hair = p[:, BALD].sum((1, 2)), p[:, HAIR].sum((1, 2))
    return bald / (bald + hair + 1e-6)


def total_loss(outputs, batch, grade_weight, lambdas=(1.0, 0.5, 0.2)):
    logits, ratio, grade = outputs
    _, mask, ratio_gt, grade_gt = batch
    loss = seg_loss(logits, mask)
    has_r, has_g = ~torch.isnan(ratio_gt), grade_gt >= 0
    if has_r.any():
        loss = loss + lambdas[0] * F.smooth_l1_loss(ratio[has_r], ratio_gt[has_r], beta=0.05)
    if has_g.any():
        g, y = grade[has_g].float(), grade_gt[has_g]
        expected = (g.softmax(1) * torch.arange(g.shape[1], device=g.device)).sum(1)
        loss = loss + lambdas[1] * (F.cross_entropy(g, y, weight=grade_weight) + 0.5 * F.smooth_l1_loss(expected, y.float()))
    return loss + lambdas[2] * F.smooth_l1_loss(ratio, soft_ratio(logits).detach(), beta=0.05)


def postprocess(cmap, min_frac=0.002, k=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = np.zeros_like(cmap)
    for cls in (HAIR, BALD):
        m = cv2.morphologyEx(cv2.morphologyEx((cmap == cls).astype(np.uint8), cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= min_frac * cmap.size)
        out[np.isin(lab, keep[keep > 0])] = cls
    return out


@torch.no_grad()
def predict(model, img, size, mc=8):
    device = next(model.parameters()).device
    model.eval()
    for m in model.modules():
        if "Drop" in type(m).__name__:
            m.train()
    x = to_tensor(img, size)[None].to(device)
    probs, ratios, grades = [], [], []
    for _ in range(mc):
        for flip in (False, True):
            logits, r, g = model(x.flip(-1) if flip else x)
            p = logits.softmax(1)
            probs.append(p.flip(-1) if flip else p)
            ratios.append(r)
            grades.append(g.softmax(1))
    model.eval()
    p = torch.cat(probs).mean(0)
    p = F.interpolate(p[None], size=img.shape[:2], mode="bilinear", align_corners=False)[0]
    entropy = -(p * p.clamp_min(1e-8).log()).sum(0) / np.log(3)
    ratios = torch.cat(ratios)
    return {
        "probs": p.cpu().numpy(),
        "cmap": postprocess(p.argmax(0).cpu().numpy().astype(np.uint8)),
        "uncertainty": entropy.cpu().numpy(),
        "ratio_head": float(ratios.mean()),
        "ratio_std": float(ratios.std()),
        "grade_probs": torch.cat(grades).mean(0).cpu().numpy(),
    }
