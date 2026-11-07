import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from baldseg import chart
from baldseg.core import BALD, HAIR, bald_ratio, clahe, predict
from baldseg.norwood_art import illustration

COLORS = {HAIR: (0, 255, 0), BALD: (255, 33, 33)}


def name(stage):
    return "Stage " + stage.replace("V", " Vertex")


def overlay(rgb, cmap):
    color = np.zeros_like(rgb)
    for cls, c in COLORS.items():
        color[cmap == cls] = c
    return np.where(cmap[..., None] > 0, (0.55 * rgb + 0.45 * color).astype(np.uint8), rgb)


def table(cfg, current):
    bounds = [0.0] + cfg["thresholds"] + [1.0]
    return "\n".join(f"{'>> ' if s == current else '   '}{name(s)}: R từ {lo:.0%} đến {hi:.0%}" for s, lo, hi in zip(cfg["stages"], bounds, bounds[1:]))


def plan_a(weights, conf):
    from ultralytics import YOLO
    model = YOLO(weights / "best.pt")
    return lambda bgr: predict(model, bgr, conf)


def plan_b(ckpt, mc):
    import torch
    from baldseg.multitask import load, predict as predict_b
    model, size = load(ckpt, "cuda" if torch.cuda.is_available() else "cpu")
    return lambda bgr: predict_b(model, clahe(bgr), size, mc)["cmap"]


def assess(plans, cfg, plan, rgb):
    if rgb is None:
        raise gr.Error("Hãy tải lên một ảnh chụp từ trên xuống đỉnh đầu.")
    cmap = plans[plan](cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not (cmap > 0).any():
        raise gr.Error("Không tìm thấy vùng tóc/da đầu trong ảnh.")
    ratio = bald_ratio(cmap)
    stage = chart.stage(ratio, cfg)
    return overlay(rgb, cmap), f"{ratio:.1%}", f"Norwood {name(stage)}", table(cfg, stage), illustration(stage)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights-a", type=Path, default=Path("weights"))
    p.add_argument("--ckpt-b", type=Path, default=Path("weights_b/norwood.pt"))
    p.add_argument("--thresholds", default="weights/norwood_thresholds.json")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--mc", type=int, default=4)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--examples", type=Path, default=Path("examples"))
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    cfg = chart.load(args.thresholds)
    plans = {}
    if args.ckpt_b.exists():
        plans["B - SegFormer"] = plan_b(args.ckpt_b, args.mc)
    if (args.weights_a / "best.pt").exists():
        plans["A - YOLO11-seg"] = plan_a(args.weights_a, args.conf)
    samples = sorted(str(f) for f in args.examples.glob("*/Top-Down.jpg"))

    with gr.Blocks(title="Đánh giá rụng tóc Hamilton-Norwood") as demo:
        gr.Markdown("# Đánh giá mức độ rụng tóc theo thang Hamilton-Norwood (ảnh đỉnh đầu)\nChỉ dùng **một ảnh chụp từ trên xuống đỉnh đầu**. Tỷ lệ da đầu lộ R = vùng hói / (vùng hói + vùng tóc) được so với các ngưỡng R đo trên hình nhìn từ trên xuống của từng mức trong thang Hamilton-Norwood.")
        with gr.Row():
            image = gr.Image(type="numpy", label="Ảnh đỉnh đầu (Top-Down)", height=320)
            seg = gr.Image(label="Phân đoạn (xanh: tóc, đỏ: da đầu lộ)", height=320)
            art = gr.Image(label="Hình mẫu của mức", height=320)
        with gr.Row():
            with gr.Column():
                plan = gr.Radio(list(plans), value=next(iter(plans)), label="Mô hình phân đoạn")
                button = gr.Button("Đánh giá", variant="primary")
            with gr.Column():
                ratio = gr.Textbox(label="Tỷ lệ da đầu lộ R")
                stage = gr.Textbox(label="Mức độ (Hamilton-Norwood)")
            levels = gr.Textbox(label="Ngưỡng R theo thang Hamilton-Norwood", lines=7, value=table(cfg, None))
        gr.Examples([[s, next(iter(plans))] for s in samples], [image, plan], label="Ảnh mẫu từ tập test")
        button.click(lambda pl, img: assess(plans, cfg, pl, img), [plan, image], [seg, ratio, stage, levels, art])
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
