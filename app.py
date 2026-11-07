import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from baldseg.core import BALD, HAIR, bald_ratio, clahe, predict
from baldseg.norwood_art import STAGES, illustration
from baldseg.topview import match, render

COLORS = {HAIR: (0, 255, 0), BALD: (255, 33, 33)}
NAMES = {s: f"Stage {s[0]}" + (" Vertex" if s.endswith("V") else "") for s in STAGES}


def overlay(rgb, cmap):
    color = np.zeros_like(rgb)
    for cls, c in COLORS.items():
        color[cmap == cls] = c
    return np.where(cmap[..., None] > 0, (0.55 * rgb + 0.45 * color).astype(np.uint8), rgb)


def plan_a(weights, conf):
    from ultralytics import YOLO
    model = YOLO(weights / "best.pt")
    return lambda bgr: predict(model, bgr, conf)


def plan_b(ckpt, mc):
    import torch
    from baldseg.multitask import load, predict as predict_b
    model, size = load(ckpt, "cuda" if torch.cuda.is_available() else "cpu")
    return lambda bgr: predict_b(model, clahe(bgr), size, mc)["cmap"]


def assess(plans, plan, rgb):
    if rgb is None:
        raise gr.Error("Hãy tải lên một ảnh chụp từ trên xuống đỉnh đầu.")
    cmap = plans[plan](cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    m = match(cmap)
    if m is None:
        raise gr.Error("Không tìm thấy vùng đầu trong ảnh.")
    stage = m["stage"]
    ranking = sorted(STAGES, key=lambda s: -m["scores"][s])
    detail = "\n".join(f"{NAMES[s]}: độ khớp {m['scores'][s]:.2f} ({m['probs'][s]:.0%})" for s in ranking)
    return overlay(rgb, cmap), render(m["aligned"], stage), f"Norwood {NAMES[stage]}", detail, f"{bald_ratio(cmap):.1%}", illustration(stage)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights-a", type=Path, default=Path("weights"))
    p.add_argument("--ckpt-b", type=Path, default=Path("weights_b/norwood.pt"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--mc", type=int, default=4)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--examples", type=Path, default=Path("examples"))
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    plans = {}
    if args.ckpt_b.exists():
        plans["B - SegFormer"] = plan_b(args.ckpt_b, args.mc)
    if (args.weights_a / "best.pt").exists():
        plans["A - YOLO11-seg"] = plan_a(args.weights_a, args.conf)
    samples = sorted(str(f) for f in args.examples.glob("*/Top-Down.jpg"))

    with gr.Blocks(title="Đánh giá rụng tóc Hamilton-Norwood") as demo:
        gr.Markdown("# Đánh giá mức độ rụng tóc theo thang Hamilton-Norwood (ảnh đỉnh đầu)\nChỉ dùng **một ảnh chụp từ trên xuống đỉnh đầu**. Mask tóc/hói được căn theo hình elip của đầu rồi so với hình nhìn từ trên xuống của từng mức trong thang Hamilton-Norwood (Stage 1, 2, 3, 3 Vertex, 4, 5, 6, 7); mức có độ khớp cao nhất là kết quả.")
        with gr.Row():
            image = gr.Image(type="numpy", label="Ảnh đỉnh đầu (Top-Down)", height=320)
            with gr.Column():
                plan = gr.Radio(list(plans), value=next(iter(plans)), label="Mô hình phân đoạn")
                button = gr.Button("Đánh giá", variant="primary")
                stage = gr.Textbox(label="Mức độ (Hamilton-Norwood)")
                ratio = gr.Textbox(label="Tỷ lệ da đầu lộ R")
        with gr.Row():
            seg = gr.Image(label="Phân đoạn (xanh: tóc, đỏ: da đầu lộ)", height=300)
            aligned = gr.Image(label="Đầu đã căn chỉnh và viền mẫu của mức khớp nhất", height=300)
            art = gr.Image(label="Hình mẫu mức khớp nhất", height=300)
        scores = gr.Textbox(label="Độ khớp với từng mức", lines=8)
        gr.Examples([[s, next(iter(plans))] for s in samples], [image, plan], label="Ảnh mẫu từ tập test")
        button.click(lambda pl, img: assess(plans, pl, img), [plan, image], [seg, aligned, stage, scores, ratio, art])
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
