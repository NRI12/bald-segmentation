import argparse
from pathlib import Path

import cv2
import gradio as gr
import joblib
import numpy as np

from baldseg.core import BALD, GRADES, HAIR, VIEWS, bald_ratio, clahe, grader_features, predict, stage_label
from baldseg.norwood_art import illustration

COLORS = {HAIR: (0, 255, 0), BALD: (255, 33, 33)}


def overlay(rgb, cmap):
    color = np.zeros_like(rgb)
    for cls, c in COLORS.items():
        color[cmap == cls] = c
    return np.where(cmap[..., None] > 0, (0.55 * rgb + 0.45 * color).astype(np.uint8), rgb)


def stage_outputs(level, cmap, view, probs):
    label, vertex = stage_label(level, cmap, view)
    detail = " | ".join(f"{g}: {p:.0%}" for g, p in zip(GRADES, probs))
    return f"Norwood {label}\n{detail}", illustration(level + 2, vertex)


def plan_a(weights, conf):
    from ultralytics import YOLO
    model = YOLO(weights / "best.pt")
    tree = joblib.load(weights / "grader.joblib")

    def run(rgb, view):
        cmap = predict(model, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), conf)
        probs = np.zeros(len(GRADES))
        probs[tree.classes_] = tree.predict_proba([grader_features(cmap, view)])[0]
        return overlay(rgb, cmap), None, f"{bald_ratio(cmap):.1%}", *stage_outputs(int(probs.argmax()), cmap, view, probs)

    return run


def plan_b(ckpt, mc):
    import torch
    from baldseg.multitask import load, predict as predict_b
    model, size = load(ckpt, "cuda" if torch.cuda.is_available() else "cpu")

    def run(rgb, view):
        out = predict_b(model, clahe(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), size, mc)
        heat = cv2.cvtColor(cv2.applyColorMap((out["uncertainty"] * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        r = f"{bald_ratio(out['cmap']):.1%} (head {out['ratio_head']:.1%} ± {out['ratio_std']:.1%})"
        return overlay(rgb, out["cmap"]), heat, r, *stage_outputs(int(out["grade_probs"].argmax()), out["cmap"], view, out["grade_probs"])

    return run


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights-a", type=Path, default=Path("weights"))
    p.add_argument("--ckpt-b", type=Path, default=Path("weights_b/norwood.pt"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--mc", type=int, default=8)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--examples", type=Path, default=Path("examples"))
    args = p.parse_args()
    plans = {}
    if args.ckpt_b.exists():
        plans["B - SegFormer đa nhiệm"] = plan_b(args.ckpt_b, args.mc)
    if (args.weights_a / "grader.joblib").exists():
        plans["A - YOLO11-seg"] = plan_a(args.weights_a, args.conf)
    gr.Interface(
        lambda rgb, view, plan: plans[plan](rgb, view),
        [gr.Image(type="numpy", label="Ảnh đầu"), gr.Dropdown(VIEWS, value="Top-Down", label="Góc chụp"), gr.Radio(list(plans), value=next(iter(plans)), label="Mô hình")],
        [gr.Image(label="Phân đoạn (xanh: tóc, đỏ: da đầu lộ)"), gr.Image(label="Bản đồ bất định (Phương án B)"), gr.Textbox(label="Tỷ lệ rụng tóc R"), gr.Textbox(label="Mức độ (Hamilton-Norwood)", lines=2), gr.Image(label="Minh họa stage (nhìn từ đỉnh đầu)")],
        examples=[[str(f), f.stem.split("_")[-1], next(iter(plans))] for f in sorted(args.examples.glob("*.jpg"))],
        title="Đánh giá mức độ rụng tóc",
    ).launch(server_name="0.0.0.0", server_port=args.port)
