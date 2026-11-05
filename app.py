import argparse
from pathlib import Path

import cv2
import gradio as gr
import joblib
import numpy as np

from baldseg.core import BALD, GRADES, HAIR, VIEWS, bald_ratio, clahe, grade, grader_features, load_thresholds, predict

COLORS = {HAIR: (0, 255, 0), BALD: (255, 33, 33)}


def overlay(rgb, cmap):
    color = np.zeros_like(rgb)
    for cls, c in COLORS.items():
        color[cmap == cls] = c
    return np.where(cmap[..., None] > 0, (0.55 * rgb + 0.45 * color).astype(np.uint8), rgb)


def plan_a(weights, conf):
    from ultralytics import YOLO
    model = YOLO(weights / "best.pt")
    tree_path = weights / "grader.joblib"
    tree = joblib.load(tree_path) if tree_path.exists() else None
    thresholds = load_thresholds(weights / "grades.json")

    def run(rgb, view):
        cmap = predict(model, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), conf)
        r = bald_ratio(cmap)
        level = int(tree.predict([grader_features(cmap, view)])[0]) if tree else grade(r, thresholds)
        return overlay(rgb, cmap), None, f"{r:.1%}", GRADES[level]

    return run


def plan_b(weights, mc):
    import torch
    from baldseg.multitask import load, predict as predict_b
    model, size = load(weights / "stage2.pt", "cuda" if torch.cuda.is_available() else "cpu")

    def run(rgb, view):
        out = predict_b(model, clahe(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), size, mc)
        heat = cv2.cvtColor(cv2.applyColorMap((out["uncertainty"] * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        probs = ", ".join(f"{p:.0%}" for p in out["grade_probs"])
        r = f"{bald_ratio(out['cmap']):.1%} (head {out['ratio_head']:.1%} ± {out['ratio_std']:.1%})"
        return overlay(rgb, out["cmap"]), heat, r, f"{GRADES[int(out['grade_probs'].argmax())]} [{probs}]"

    return run


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights-a", type=Path, default=Path("weights"))
    p.add_argument("--weights-b", type=Path, default=Path("weights_b"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--mc", type=int, default=8)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--examples", type=Path, default=Path("examples"))
    args = p.parse_args()
    plans = {}
    if (args.weights_b / "stage2.pt").exists():
        plans["B - SegFormer đa nhiệm"] = plan_b(args.weights_b, args.mc)
    if (args.weights_a / "best.pt").exists():
        plans["A - YOLO11-seg"] = plan_a(args.weights_a, args.conf)
    gr.Interface(
        lambda rgb, view, plan: plans[plan](rgb, view),
        [gr.Image(type="numpy", label="Ảnh đầu"), gr.Dropdown(VIEWS, value="Top-Down", label="Góc chụp"), gr.Radio(list(plans), value=next(iter(plans)), label="Mô hình")],
        [gr.Image(label="Phân đoạn (xanh: tóc, đỏ: da đầu lộ)"), gr.Image(label="Bản đồ bất định (Phương án B)"), gr.Textbox(label="Tỷ lệ rụng tóc R"), gr.Textbox(label="Mức độ")],
        examples=[[str(f), f.stem.split("_")[-1], next(iter(plans))] for f in sorted(args.examples.glob("*.jpg"))],
        title="Đánh giá mức độ rụng tóc",
    ).launch(server_name="0.0.0.0", server_port=args.port)
