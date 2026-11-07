import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baldseg.metrics import SegMeter
from baldseg.model import Samples, build, save, seg_items, seg_loss


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    meter = SegMeter()
    for x, mask in loader:
        logits = torch.nn.functional.interpolate(model(pixel_values=x.to(device)).logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        for gt, pred in zip(mask.numpy(), logits.argmax(1).cpu().numpy()):
            meter.update(gt, pred)
    return meter.report()["mean"]["dice"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--backbone", default="nvidia/mit-b2")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("weights/segformer.pt"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = DataLoader(Samples(seg_items(args.data, "train"), args.size, True), args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    valid = DataLoader(Samples(seg_items(args.data, "valid"), args.size, False), args.batch, num_workers=args.workers)
    model = build(args.backbone).to(device)
    decoder = [p for n, p in model.named_parameters() if not n.startswith("segformer")]
    encoder = [p for n, p in model.named_parameters() if n.startswith("segformer")]
    opt = torch.optim.AdamW([{"params": encoder, "lr": args.lr}, {"params": decoder, "lr": args.lr * 10}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 10], total_steps=args.epochs * len(train), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, mask in train:
            x, mask = x.to(device), mask.to(device)
            with torch.autocast(device, enabled=device == "cuda"):
                loss = seg_loss(model(pixel_values=x).logits, mask)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        dice = validate(model, valid, device)
        if dice > best:
            best = dice
            save(model, args.out, args.backbone, args.size)
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / len(train):.4f} dice {dice:.4f} best {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
