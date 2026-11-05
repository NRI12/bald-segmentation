import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

from baldseg.core import GRADES
from baldseg.metrics import SegMeter
from baldseg.multitask import MultiTask, Samples, grade_items, load, save, seg_items, total_loss


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    meter, gt_g, pred_g = SegMeter(), [], []
    for x, mask, _, grade in loader:
        logits, _, g = model(x.to(device))
        seg = grade < 0
        for gt, pred in zip(mask[seg].numpy(), logits[seg.to(device)].argmax(1).cpu().numpy()):
            meter.update(gt, pred)
        gt_g += grade[~seg].tolist()
        pred_g += g[(~seg).to(device)].argmax(1).cpu().tolist()
    dice = meter.report()["mean"]["dice"]
    kappa = cohen_kappa_score(gt_g, pred_g, weights="quadratic")
    return dice, kappa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--grades", default="data_grade")
    p.add_argument("--pseudo", default=None)
    p.add_argument("--init", default=None)
    p.add_argument("--backbone", default="nvidia/mit-b2")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("weights_b/stage1.pt"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_items = seg_items(args.data, "train") + grade_items(args.grades, "train", args.pseudo)
    valid_items = seg_items(args.data, "valid") + grade_items(args.grades, "valid")
    train = DataLoader(Samples(train_items, args.size, True), args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    valid = DataLoader(Samples(valid_items, args.size, False), args.batch, num_workers=args.workers)

    counts = np.bincount([it["grade"] for it in train_items if "grade" in it], minlength=len(GRADES))
    grade_weight = torch.tensor(counts.sum() / (len(GRADES) * np.maximum(counts, 1)), dtype=torch.float32, device=device)

    model = load(args.init, device)[0] if args.init else MultiTask(args.backbone).to(device)
    heads = [p for n, p in model.named_parameters() if not n.startswith("seg.segformer")]
    encoder = [p for n, p in model.named_parameters() if n.startswith("seg.segformer")]
    opt = torch.optim.AdamW([{"params": encoder, "lr": args.lr}, {"params": heads, "lr": args.lr * 10}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 10], total_steps=args.epochs * len(train), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in train:
            batch = [t.to(device) for t in batch]
            with torch.autocast(device, enabled=device == "cuda"):
                loss = total_loss(model(batch[0]), batch, grade_weight)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        dice, kappa = validate(model, valid, device)
        score = dice + kappa
        if score > best:
            best = score
            save(model, args.out, args.size)
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / len(train):.4f} dice {dice:.4f} kappa {kappa:.4f} best {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
