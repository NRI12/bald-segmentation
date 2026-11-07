import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

from baldseg import chart
from baldseg.core import BALD, HAIR, bald_ratio, clahe
from baldseg.model import load, predict
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


def assess(model, size, cfg, rgb):
    if rgb is None:
        raise gr.Error("Hãy tải lên một ảnh chụp từ trên xuống đỉnh đầu.")
    cmap = predict(model, clahe(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), size)
    if not (cmap > 0).any():
        raise gr.Error("Không tìm thấy vùng tóc/da đầu trong ảnh.")
    ratio = bald_ratio(cmap)
    stage = chart.stage(ratio, cfg)
    return overlay(rgb, cmap), f"{ratio:.1%}", f"Norwood {name(stage)}", table(cfg, stage), illustration(stage)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/segformer.pt")
    p.add_argument("--thresholds", default="weights/norwood_thresholds.json")
    p.add_argument("--examples", type=Path, default=Path("examples"))
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    model, size = load(args.weights, "cuda" if torch.cuda.is_available() else "cpu")
    cfg = chart.load(args.thresholds)

    with gr.Blocks(title="Đánh giá rụng tóc Hamilton-Norwood") as demo:
        gr.Markdown("# Đánh giá mức độ rụng tóc theo thang Hamilton-Norwood (ảnh đỉnh đầu)\nTải lên **một ảnh chụp từ trên xuống đỉnh đầu**. Tỷ lệ da đầu lộ R = vùng hói / (vùng hói + vùng tóc) được so với các ngưỡng R đo trên hình nhìn từ trên xuống của từng mức trong thang Hamilton-Norwood.")
        with gr.Row():
            image = gr.Image(type="numpy", label="Ảnh đỉnh đầu", height=320)
            seg = gr.Image(label="Phân đoạn (xanh: tóc, đỏ: da đầu lộ)", height=320)
            art = gr.Image(label="Hình mẫu của mức", height=320)
        with gr.Row():
            with gr.Column():
                button = gr.Button("Đánh giá", variant="primary")
                ratio = gr.Textbox(label="Tỷ lệ da đầu lộ R")
                stage = gr.Textbox(label="Mức độ (Hamilton-Norwood)")
            levels = gr.Textbox(label="Ngưỡng R theo thang Hamilton-Norwood", lines=7, value=table(cfg, None))
        gr.Examples(sorted(str(f) for f in args.examples.glob("*.jpg")), [image], label="Ảnh mẫu từ tập test")
        button.click(lambda img: assess(model, size, cfg, img), [image], [seg, ratio, stage, levels, art])
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
