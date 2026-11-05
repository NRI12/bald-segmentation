import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

from baldseg.core import BALD, GRADES, HAIR, bald_ratio, grade, load_thresholds, predict

COLORS = {HAIR: (0, 255, 0), BALD: (255, 33, 33)}


def build(weights, conf):
    model = YOLO(weights / "best.pt")
    thresholds = load_thresholds(weights / "grades.json")

    def run(rgb):
        cmap = predict(model, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), conf)
        color = np.zeros_like(rgb)
        for cls, c in COLORS.items():
            color[cmap == cls] = c
        overlay = np.where(cmap[..., None] > 0, (0.55 * rgb + 0.45 * color).astype(np.uint8), rgb)
        r = bald_ratio(cmap)
        return overlay, f"{r:.1%}", GRADES[grade(r, thresholds)]

    return gr.Interface(
        run,
        gr.Image(type="numpy", label="Ảnh đầu"),
        [gr.Image(label="Phân đoạn (xanh: tóc, đỏ: da đầu lộ)"), gr.Textbox(label="Tỷ lệ rụng tóc R"), gr.Textbox(label="Mức độ")],
        title="Đánh giá mức độ rụng tóc",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=Path("weights"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    build(args.weights, args.conf).launch(server_name="0.0.0.0", server_port=args.port)
